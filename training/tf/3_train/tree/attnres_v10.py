# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/attnres_v10.py, sha256 37465386400f032a), GPL-3.
# AttnRes: Kimi Team (G. Chen, Y. Zhang, J. Su, W. Xu et al.), "Attention Residuals", arXiv 2603.15031 (2026).
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""v10 fused AttnRes depth-mix (replace mode: no output norm, key RMSNorm weight = ones, scale 1), drop-in for
v7 arch.depth_mix / fla.ops.attnres.fused_attnres(q, sources, ones).

    o = sum_l softmax_l(logit) v_l,   logit_l = (v_l . q) * rsqrt(mean(v_l^2) + eps)     per token

vs fla 0.5.1 fused_attnres (one program per token, bwd re-reads all sources to recompute the mix, 100 MB fp32 dqw
round trip):
  fwd: BT tokens per program, every source read ONCE (online softmax); stores o (activation dtype), the fp32 mix
       (for an exact delta in bwd) and lse.  No per-source rstd/logit stores.
  bwd: ONE pass over the sources: rstd/logit recomputed from v, p = exp(logit - lse), dv_l written, dq reduced in
       registers per program (tiny [programs, D] partials).
Exposed as a torch.library custom op (opaque to dynamo: no graph break per read, inductor keeps fusing around it).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-6
L2MAX = 16


def _ptrs(ts):
    L2 = 8 if len(ts) <= 8 else 16
    assert len(ts) <= L2
    for t in ts:
        assert t.is_contiguous() and t.data_ptr() % 16 == 0
    return tuple(ts) + (ts[0],) * (L2 - len(ts)), L2


@triton.jit
def _src(res, l: tl.constexpr):
    return res[l]


@triton.jit
def _ar_fwd(q, res, o, o32, lse, N, L, D: tl.constexpr, L2: tl.constexpr, eps: tl.constexpr,
            BT: tl.constexpr, BD: tl.constexpr, SAVE32: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    rm = rows < N
    cm = cols < D
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    b_q = tl.load(q + cols, mask=cm, other=0.).to(tl.float32)
    m = tl.full([BT], float("-inf"), tl.float32)
    acc = tl.zeros([BT], tl.float32)
    out = tl.zeros([BT, BD], tl.float32)
    for l in tl.static_range(L2):
        if l < L:
            v = tl.load(res[l] + offs, mask=msk, other=0.).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)
            s = tl.sum(v * b_q[None, :], axis=1) * rstd
            mn = tl.maximum(m, s)
            r = tl.exp(m - mn)
            p = tl.exp(s - mn)
            acc = acc * r + p
            out = out * r[:, None] + p[:, None] * v
            m = mn
    out = out / acc[:, None]
    tl.store(o + offs, out.to(o.dtype.element_ty), mask=msk)
    if SAVE32:
        tl.store(o32 + offs, out, mask=msk)
    tl.store(lse + rows, m + tl.log(acc), mask=rm)


@triton.jit
def _ar_bwd(q, res, dres, do, o32, lse, dq_part, N, L, D: tl.constexpr, L2: tl.constexpr, eps: tl.constexpr,
            BT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    rm = rows < N
    cm = cols < D
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    b_q = tl.load(q + cols, mask=cm, other=0.).to(tl.float32)
    b_do = tl.load(do + offs, mask=msk, other=0.).to(tl.float32)
    b_o = tl.load(o32 + offs, mask=msk, other=0.).to(tl.float32)
    b_lse = tl.load(lse + rows, mask=rm, other=0.)
    delta = tl.sum(b_do * b_o, axis=1)
    dqa = tl.zeros([BD], tl.float32)
    for l in tl.static_range(L2):
        if l < L:
            v = tl.load(res[l] + offs, mask=msk, other=0.).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)
            s = tl.sum(v * b_q[None, :], axis=1) * rstd
            p = tl.where(rm, tl.exp(s - b_lse), 0.)
            dp = tl.sum(v * b_do, axis=1)
            ds = p * (dp - delta)
            k = v * rstd[:, None]
            dv = p[:, None] * b_do + (ds * rstd)[:, None] * (b_q[None, :] - k * (s / D)[:, None])
            tl.store(dres[l] + offs, dv.to(dres[l].dtype.element_ty), mask=msk)
            dqa += tl.sum(ds[:, None] * k, axis=0)
    tl.store(dq_part + pid.to(tl.int64) * D + cols, dqa, mask=cm)


_CFG = {"fwd": (32, 4), "bwd": (32, 8)}   # (BT, num_warps), tuned by tune_ar.py


import os
SAVE32 = os.environ.get("AR10_SAVE32", "1") == "1"


def ar_fwd(q, srcs, save32=None):
    save32 = SAVE32 if save32 is None else save32
    v0 = srcs[0]
    D = v0.shape[-1]
    N = v0.numel() // D
    res, L2 = _ptrs(srcs)
    o = torch.empty_like(v0)
    o32 = torch.empty(v0.shape if save32 else (0,), dtype=torch.float32, device=v0.device)
    lse = torch.empty(N, dtype=torch.float32, device=v0.device)
    BT, nw = _CFG["fwd"]
    _ar_fwd[(triton.cdiv(N, BT),)](q, res, o, o32 if save32 else o, lse, N, len(srcs), D=D, L2=L2, eps=EPS, BT=BT,
                                   BD=triton.next_power_of_2(D), SAVE32=save32, num_warps=nw)
    return o, o32, lse


def ar_bwd(do, q, srcs, o32, lse):
    v0 = srcs[0]
    D = v0.shape[-1]
    N = v0.numel() // D
    res, L2 = _ptrs(srcs)
    dvs = [torch.empty_like(s) for s in srcs]
    dres, _ = _ptrs(dvs)
    BT, nw = _CFG["bwd"]
    grid = triton.cdiv(N, BT)
    part = torch.empty((grid, D), dtype=torch.float32, device=v0.device)
    _ar_bwd[(grid,)](q, res, dres, do.contiguous(), o32, lse, part, N, len(srcs), D=D, L2=L2, eps=EPS, BT=BT,
                     BD=triton.next_power_of_2(D), num_warps=nw)
    return dvs, part.sum(0)


# ------------------------------------------------------------------ custom op (opaque to dynamo)
@torch.library.custom_op("v10::attnres", mutates_args=())
def _op(q: torch.Tensor, srcs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ar_fwd(q, srcs)


@_op.register_fake
def _(q, srcs):
    v0 = srcs[0]
    return (torch.empty_like(v0), v0.new_empty(v0.shape if SAVE32 else (0,), dtype=torch.float32),
            v0.new_empty((v0.numel() // v0.shape[-1],), dtype=torch.float32))


@torch.library.custom_op("v10::attnres_bwd", mutates_args=())
def _op_bwd(do: torch.Tensor, q: torch.Tensor, srcs: list[torch.Tensor], o32: torch.Tensor,
            lse: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    dvs, dq = ar_bwd(do, q, srcs, o32, lse)
    return dvs, dq.to(q.dtype)


@_op_bwd.register_fake
def _(do, q, srcs, o32, lse):
    return [torch.empty_like(s) for s in srcs], torch.empty_like(q)


def _setup(ctx, inputs, output):
    q, srcs = inputs
    o, o32, lse = output
    ctx.save_for_backward(q, o32 if o32.numel() else o, lse, *srcs)   # without the fp32 mix, delta uses o


def _backward(ctx, do, d_o32, d_lse):
    q, o32, lse, *srcs = ctx.saved_tensors
    dvs, dq = _op_bwd(do, q, list(srcs), o32, lse)
    return dq, dvs


_op.register_autograd(_backward, setup_context=_setup)


def depth_mix(sources, w):
    """sources: list of [..., D] tensors (same shape/dtype), w: [D] fp32 query -> [..., D]"""
    D = sources[0].shape[-1]
    flat = [s.reshape(-1, D).contiguous() for s in sources]
    o, _, _ = _op(w, flat)
    return o.view(sources[0].shape)


def install(verbose=True):
    """route v7 arch.depth_mix (replace and gated modes both call it) through the v10 op"""
    import arch
    arch.depth_mix = lambda sources, q: depth_mix(sources, q.w)
    if verbose:
        print("attnres_v10: arch.depth_mix -> v10 fused custom op", flush=True)


# ====================================================================================================================
# opt-in: depth-mix + (gated combine) + RMSNorm in one op:  y = rms_norm(h),  h = mix (replace) | x + alpha (mix - x)
# rms_norm as F.rms_norm(h) with eps=None on a bf16 h: eps = finfo(bfloat16).eps = 2**-7; computed here on the fp32 h.
# ====================================================================================================================
EPS_RMS = float(torch.finfo(torch.bfloat16).eps)


@triton.jit
def _arn_fwd(q, res, x, alpha, y, o32, lse, N, L, D: tl.constexpr, L2: tl.constexpr, eps: tl.constexpr,
             eps_r: tl.constexpr, GATED: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    rm = rows < N
    cm = cols < D
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    b_q = tl.load(q + cols, mask=cm, other=0.).to(tl.float32)
    m = tl.full([BT], float("-inf"), tl.float32)
    acc = tl.zeros([BT], tl.float32)
    out = tl.zeros([BT, BD], tl.float32)
    for l in tl.static_range(L2):
        if l < L:
            v = tl.load(res[l] + offs, mask=msk, other=0.).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)
            s = tl.sum(v * b_q[None, :], axis=1) * rstd
            mn = tl.maximum(m, s)
            r = tl.exp(m - mn)
            p = tl.exp(s - mn)
            acc = acc * r + p
            out = out * r[:, None] + p[:, None] * v
            m = mn
    out = out / acc[:, None]
    tl.store(o32 + offs, out, mask=msk)
    tl.store(lse + rows, m + tl.log(acc), mask=rm)
    if GATED:
        xv = tl.load(x + offs, mask=msk, other=0.).to(tl.float32)
        a = tl.load(alpha).to(tl.float32)
        hp = xv + a * (out - xv)
    else:
        hp = out
    ro = tl.rsqrt(tl.sum(hp * hp, axis=1) / D + eps_r)
    tl.store(y + offs, (hp * ro[:, None]).to(y.dtype.element_ty), mask=msk)


@triton.jit
def _arn_bwd(q, res, dres, x, alpha, dx, da_part, dy, o32, lse, dq_part, N, L, D: tl.constexpr, L2: tl.constexpr,
             eps: tl.constexpr, eps_r: tl.constexpr, GATED: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    rm = rows < N
    cm = cols < D
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    b_q = tl.load(q + cols, mask=cm, other=0.).to(tl.float32)
    b_mix = tl.load(o32 + offs, mask=msk, other=0.)
    b_dy = tl.load(dy + offs, mask=msk, other=0.).to(tl.float32)
    b_lse = tl.load(lse + rows, mask=rm, other=0.)
    if GATED:
        xv = tl.load(x + offs, mask=msk, other=0.).to(tl.float32)
        a = tl.load(alpha).to(tl.float32)
        hp = xv + a * (b_mix - xv)
    else:
        hp = b_mix
    ro = tl.rsqrt(tl.sum(hp * hp, axis=1) / D + eps_r)
    xh = hp * ro[:, None]
    d_hp = ro[:, None] * (b_dy - xh * (tl.sum(b_dy * xh, axis=1) / D)[:, None])
    if GATED:
        b_do = a * d_hp
        tl.store(dx + offs, ((1. - a) * d_hp).to(dx.dtype.element_ty), mask=msk)
        tl.store(da_part + pid, tl.sum(tl.sum(tl.where(msk, d_hp * (b_mix - xv), 0.), axis=1), axis=0))
    else:
        b_do = d_hp
    delta = tl.sum(b_do * b_mix, axis=1)
    dqa = tl.zeros([BD], tl.float32)
    for l in tl.static_range(L2):
        if l < L:
            v = tl.load(res[l] + offs, mask=msk, other=0.).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)
            s = tl.sum(v * b_q[None, :], axis=1) * rstd
            p = tl.where(rm, tl.exp(s - b_lse), 0.)
            dp = tl.sum(v * b_do, axis=1)
            ds = p * (dp - delta)
            k = v * rstd[:, None]
            dv = p[:, None] * b_do + (ds * rstd)[:, None] * (b_q[None, :] - k * (s / D)[:, None])
            tl.store(dres[l] + offs, dv.to(dres[l].dtype.element_ty), mask=msk)
            dqa += tl.sum(ds[:, None] * k, axis=0)
    tl.store(dq_part + pid.to(tl.int64) * D + cols, dqa, mask=cm)


@torch.library.custom_op("v10::attnres_norm", mutates_args=())
def _opn(q: torch.Tensor, srcs: list[torch.Tensor], x: torch.Tensor, alpha: torch.Tensor,
         gated: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v0 = srcs[0]
    D = v0.shape[-1]
    N = v0.numel() // D
    res, L2 = _ptrs(srcs)
    y = torch.empty_like(v0)
    o32 = torch.empty(v0.shape, dtype=torch.float32, device=v0.device)
    lse = torch.empty(N, dtype=torch.float32, device=v0.device)
    BT, nw = _CFG["fwd"]
    _arn_fwd[(triton.cdiv(N, BT),)](q, res, x, alpha, y, o32, lse, N, len(srcs), D=D, L2=L2, eps=EPS, eps_r=EPS_RMS,
                                    GATED=gated, BT=BT, BD=triton.next_power_of_2(D), num_warps=nw)
    return y, o32, lse


@_opn.register_fake
def _(q, srcs, x, alpha, gated):
    v0 = srcs[0]
    return (torch.empty_like(v0), v0.new_empty(v0.shape, dtype=torch.float32),
            v0.new_empty((v0.numel() // v0.shape[-1],), dtype=torch.float32))


@torch.library.custom_op("v10::attnres_norm_bwd", mutates_args=())
def _opn_bwd(dy: torch.Tensor, q: torch.Tensor, srcs: list[torch.Tensor], x: torch.Tensor, alpha: torch.Tensor,
             o32: torch.Tensor, lse: torch.Tensor, gated: bool) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    v0 = srcs[0]
    D = v0.shape[-1]
    N = v0.numel() // D
    res, L2 = _ptrs(srcs)
    dvs = [torch.empty_like(s) for s in srcs]
    dres, _ = _ptrs(dvs)
    BT, nw = _CFG["bwd"]
    grid = triton.cdiv(N, BT)
    part = torch.empty((grid, D), dtype=torch.float32, device=v0.device)
    dx = torch.empty_like(x) if gated else torch.empty(0, device=v0.device, dtype=v0.dtype)
    dap = torch.empty(grid if gated else 1, dtype=torch.float32, device=v0.device)
    _arn_bwd[(grid,)](q, res, dres, x, alpha, dx if gated else x, dap, dy.contiguous(), o32, lse, part, N, len(srcs),
                      D=D, L2=L2, eps=EPS, eps_r=EPS_RMS, GATED=gated, BT=BT, BD=triton.next_power_of_2(D), num_warps=nw)
    da = dap.sum().view(alpha.shape).to(alpha.dtype) if gated else torch.zeros_like(alpha)
    return dvs, part.sum(0).to(q.dtype), (dx if gated else torch.zeros_like(x)), da


@_opn_bwd.register_fake
def _(dy, q, srcs, x, alpha, o32, lse, gated):
    return [torch.empty_like(s) for s in srcs], torch.empty_like(q), torch.empty_like(x), torch.empty_like(alpha)


def _setup_n(ctx, inputs, output):
    q, srcs, x, alpha, gated = inputs
    _, o32, lse = output
    ctx.gated = gated
    ctx.save_for_backward(q, x, alpha, o32, lse, *srcs)


def _backward_n(ctx, dy, d_o32, d_lse):
    q, x, alpha, o32, lse, *srcs = ctx.saved_tensors
    dvs, dq, dx, da = _opn_bwd(dy, q, list(srcs), x, alpha, o32, lse, ctx.gated)
    return dq, dvs, (dx if ctx.gated else None), (da if ctx.gated else None), None


_opn.register_autograd(_backward_n, setup_context=_setup_n)


def depth_mix_norm(sources, w, x=None, alpha=None):
    """rms_norm(mix) (replace) or rms_norm(x + alpha (mix - x)) (gated; x must be the LAST source, as in arch)"""
    D = sources[0].shape[-1]
    flat = [s.reshape(-1, D).contiguous() for s in sources]
    gated = x is not None
    xf = x.reshape(-1, D).contiguous() if gated else flat[0]
    a = alpha.reshape(1).float() if gated else torch.zeros(1, device=flat[0].device)
    y, _, _ = _opn(w, flat, xf, a, gated)
    return y.view(sources[0].shape)
