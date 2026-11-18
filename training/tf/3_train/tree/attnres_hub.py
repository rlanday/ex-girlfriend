# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/attnres_hub.py, sha256 a13677f7b70b664e), GPL-3.
# AttnRes: Kimi Team (G. Chen, Y. Zhang, J. Su, W. Xu et al.), "Attention Residuals", arXiv 2603.15031 (2026).
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""v10 AttnRes with a per-forward "hub": every source gets ONE gradient pass instead of autograd's per-read adds.

Why (v10 msg 1158): in replace-B2 every block summary feeds many later reads. With a per-read op (fla or attnres_v10)
each read's backward writes a full [T, D] gradient per source and autograd sums them with bf16 aten::add_ outside any
compiled graph: ~180 (read, source) pairs per step, ~17.6 ms of the +35..46 ms AttnRes GPU cost on RTX PRO 6000.

Here a read's backward computes only per-row scalars per source (dp = v.do, logit s; lse, delta = sum p dp) and dq in
one pass over its sources (no stored fp32 mix), and returns NO source gradient.
Each source passes once through an identity "tap"; autograd runs the tap's backward only after every read consuming
that source has run its backward, and the tap then builds the whole source gradient in one pass, accumulating in fp32:

    dv_j = g_in + sum_r [ p_rj do_r + rstd_j ds_rj q_r - rstd_j^2 v_j ds_rj s_rj / D ]

(the per-read formula of attnres_v10 / fla, summed over the reads r that consume source j).

    hub = attnres_hub.Hub()          # one per forward (compute_last_activation)
    x = hub.source(x)                # every tensor that will be a read source, once, right after it is created
    h = hub.read(q.w, [x, b1, ...])  # o = sum_l softmax_l(q . rmsnorm(v_l)) v_l   (same op as attnres_v10.depth_mix)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import attnres_v10 as A

EPS = A.EPS


@triton.jit
def _bwd_scalars(q, res, do, LSE, DP, S, DEL, dq_part, N, L, D: tl.constexpr, L2: tl.constexpr,
                 eps: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """one pass over the sources, no stored mix: per source l and row t store dp = v.do and the logit s (fp32 [L, N]),
    per row lse and delta = sum_l p_l dp_l (= do . o exactly); dq = sum_t (sum_l p_l dp_l k_l - delta sum_l p_l k_l).
    Nothing written here is read back here (no intra-program global RAW: Triton inserts no barrier for it)."""
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    rm = rows < N
    cm = cols < D
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    b_q = tl.load(q + cols, mask=cm, other=0.).to(tl.float32)
    b_do = tl.load(do + offs, mask=msk, other=0.).to(tl.float32)
    # lse is recomputed here too (online softmax over the same logits as the forward)
    m = tl.full([BT], float("-inf"), tl.float32)
    acc = tl.zeros([BT], tl.float32)
    delta = tl.zeros([BT], tl.float32)
    A2 = tl.zeros([BT, BD], tl.float32)
    Kb = tl.zeros([BT, BD], tl.float32)
    for l in tl.static_range(L2):
        if l < L:
            v = tl.load(res[l] + offs, mask=msk, other=0.).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)
            s = tl.sum(v * b_q[None, :], axis=1) * rstd
            dp = tl.sum(v * b_do, axis=1)
            mn = tl.maximum(m, s)
            r = tl.exp(m - mn)
            e = tl.exp(s - mn)
            acc = acc * r + e
            delta = delta * r + e * dp
            k = v * rstd[:, None]
            A2 = A2 * r[:, None] + (e * dp)[:, None] * k
            Kb = Kb * r[:, None] + e[:, None] * k
            m = mn
            so = l * N + rows
            tl.store(DP + so, dp, mask=rm)
            tl.store(S + so, s, mask=rm)
    lse = m + tl.log(acc)
    delta = delta / acc
    A2 = A2 / acc[:, None]
    Kb = Kb / acc[:, None]
    tl.store(LSE + rows, lse, mask=rm)
    tl.store(DEL + rows, delta, mask=rm)
    dqa = tl.sum(tl.where(rm[:, None], A2 - delta[:, None] * Kb, 0.), axis=0)
    tl.store(dq_part + pid.to(tl.int64) * D + cols, dqa, mask=cm)


@triton.jit
def _tap_bwd(v, g, out, dptr, LSE, DP, S, DEL, Q, R, N, D: tl.constexpr, eps: tl.constexpr, HAS_G: tl.constexpr,
             BT: tl.constexpr, BD: tl.constexpr):
    """dv = g + sum_r p_r do_r + rstd ds_r q_r - rstd^2 v ds_r s_r / D   over the R reads that consumed this source.
    dptr: int64 [R] data pointers of the bf16 do_r; LSE/DP/S/DEL: fp32 [R, N], p_r = exp(s_r - lse_r),
    ds_r = p_r (dp_r - delta_r); Q: fp32 [R, D]"""
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    rm = rows < N
    cm = cols < D
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    b_v = tl.load(v + offs, mask=msk, other=0.).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(b_v * b_v, axis=1) / D + eps)
    if HAS_G:
        acc = tl.load(g + offs, mask=msk, other=0.).to(tl.float32)
    else:
        acc = tl.zeros([BT, BD], tl.float32)
    cs = tl.zeros([BT], tl.float32)
    for r in range(R):
        dp_r = tl.load(dptr + r).to(tl.pointer_type(tl.bfloat16))
        b_do = tl.load(dp_r + offs, mask=msk, other=0.).to(tl.float32)
        so = r.to(tl.int64) * N + rows
        sv = tl.load(S + so, mask=rm, other=0.)
        p = tl.where(rm, tl.exp(sv - tl.load(LSE + so, mask=rm, other=0.)), 0.)
        ds = p * (tl.load(DP + so, mask=rm, other=0.) - tl.load(DEL + so, mask=rm, other=0.))
        cs += ds * sv
        b_q = tl.load(Q + r * D + cols, mask=cm, other=0.)
        acc += p[:, None] * b_do + (ds * rstd)[:, None] * b_q[None, :]
    acc -= (rstd * rstd * cs / D)[:, None] * b_v
    tl.store(out + offs, acc.to(out.dtype.element_ty), mask=msk)


_CFG = {"scal": (16, 2), "tap": (16, 8)}   # (BT, num_warps), tune_hub.py on RTX PRO 6000: ~1.4 TB/s both


class Hub:
    """per-forward registry: source id -> list of (do, lse, dp row, s row, delta, q) left by the reads' backward"""

    def __init__(self):
        self.pending = {}
        self.n = 0

    @torch._dynamo.disable
    def source(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() or not t.requires_grad:
            return t
        return _tap(t, self)

    @torch._dynamo.disable
    def read(self, w: torch.Tensor, sources: list[torch.Tensor]) -> torch.Tensor:
        D = sources[0].shape[-1]
        if not torch.is_grad_enabled():
            return A.depth_mix(sources, w)
        ids = []
        for s in sources:
            j = getattr(s, "_v10_hub_id", None)
            assert j is not None and j[0] is self, "AttnRes hub: every read source must come from hub.source()"
            ids.append(j[1])
        flat = [s.reshape(-1, D) for s in sources]
        o = _Read.apply(w, self, tuple(ids), *flat)
        return o.view(sources[0].shape)


@torch._dynamo.disable
def _tap(t, hub):
    j = hub.n
    hub.n += 1
    hub.pending[j] = []
    o = _Tap.apply(t.contiguous(), hub, j)
    o._v10_hub_id = (hub, j)
    return o


class _Tap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, hub, j):
        ctx.hub, ctx.j = hub, j
        ctx.save_for_backward(v)
        ctx.set_materialize_grads(False)
        return v.view_as(v)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        ent = ctx.hub.pending.pop(ctx.j, [])
        if not ent:
            return g, None, None
        D = v.shape[-1]
        N = v.numel() // D
        R = len(ent)
        dos = [e[0] for e in ent]
        # pinned + non_blocking: no host sync per source (a blocking copy here costs ~6 ms/step of launch-ahead)
        dptr = torch.tensor([d.data_ptr() for d in dos], dtype=torch.int64, pin_memory=True).to(v.device, non_blocking=True)
        LSE = torch.stack([e[1] for e in ent])
        DP = torch.stack([e[2] for e in ent])
        S = torch.stack([e[3] for e in ent])
        DEL = torch.stack([e[4] for e in ent])
        Q = torch.stack([e[5] for e in ent])
        out = torch.empty_like(v)
        if g is not None:
            g = g.contiguous()
        BT, nw = _CFG["tap"]
        _tap_bwd[(triton.cdiv(N, BT),)](v, g if g is not None else v, out, dptr, LSE, DP, S, DEL, Q, R, N, D=D, eps=EPS,
                                       HAS_G=g is not None, BT=BT, BD=triton.next_power_of_2(D), num_warps=nw)
        del dos   # the do_r stay alive until here through `ent`
        return out, None, None


class _Read(torch.autograd.Function):
    @staticmethod
    @torch._dynamo.disable
    def forward(ctx, w, hub, ids, *srcs):
        srcs = [s.contiguous() for s in srcs]
        o, _, _ = A.ar_fwd(w, srcs, save32=False)    # no fp32 mix / lse needed: bwd recomputes from the sources
        ctx.hub, ctx.ids = hub, ids
        ctx.save_for_backward(w, *srcs)
        return o

    @staticmethod
    @torch._dynamo.disable
    def backward(ctx, do):
        w, *srcs = ctx.saved_tensors
        L = len(srcs)
        D = srcs[0].shape[-1]
        N = srcs[0].numel() // D
        do = do.contiguous()
        assert do.dtype == torch.bfloat16
        res, L2 = A._ptrs(srcs)
        DP = torch.empty((L, N), dtype=torch.float32, device=do.device)
        S = torch.empty_like(DP)
        LSE = torch.empty((N,), dtype=torch.float32, device=do.device)
        DEL = torch.empty_like(LSE)
        BT, nw = _CFG["scal"]
        grid = triton.cdiv(N, BT)
        part = torch.empty((grid, D), dtype=torch.float32, device=do.device)
        _bwd_scalars[(grid,)](w, res, do, LSE, DP, S, DEL, part, N, L, D=D, L2=L2, eps=EPS, BT=BT,
                              BD=triton.next_power_of_2(D), num_warps=nw)
        wq = w.float()
        for l, j in enumerate(ctx.ids):
            ctx.hub.pending[j].append((do, LSE, DP[l], S[l], DEL, wq))
        dq = part.sum(0).to(w.dtype)
        return (dq, None, None) + (None,) * L
