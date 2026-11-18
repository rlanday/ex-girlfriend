# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/fakequant.py, sha256 36dd253cb3d10879), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""v10 fused fake-quant for QAT: drop-in replacement for QuantizeFunction (activations, int8) and rx3.PerRowQuantize
(per-row mixed-precision weights).

Forward is bit-identical to the eager path (same fp32 ops: x / s with IEEE division, round-half-even, clamp, * s,
round-to-nearest cast). Backward: grad_input is bit-identical (STE mask); the scale gradient is the same formula
reduced in a different order (fp32), then rounded to the grad dtype exactly as the reference does.

Instead of saving the fp32 `scaled` tensor (2x the activation in bytes), the backward recomputes it from the input.

    import fakequant; fakequant.install(raw_model)   # after rx3.install_mixed, before torch.compile / DDP
"""
from __future__ import annotations

import math
import types

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from pysrc.quantization import Quantize, QuantizeFunction


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _fq_fwd(x_ptr, s_ptr, q_ptr, y_ptr, M, C, BS: tl.constexpr, PER_ROW: tl.constexpr,
            QMIN: tl.constexpr, QMAX: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    """x [M, C]; block of element (m, c): PER_ROW -> m (weights), else c // BS (activations, scale per column block)"""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BC)
    rm = rows < M
    cm = cols < C
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    if PER_ROW:
        s = tl.load(s_ptr + rows, mask=rm, other=1.0)
        s = s.to(tl.bfloat16).to(tl.float32)[:, None]
        q = tl.load(q_ptr + rows, mask=rm, other=1.0)[:, None]
        r = libdevice.rint(libdevice.div_rn(x, s))
        r = tl.maximum(tl.minimum(r, q), -q)
    else:
        s = tl.load(s_ptr + cols // BS, mask=cm, other=1.0)
        s = s.to(tl.bfloat16).to(tl.float32)[None, :]
        r = libdevice.rint(libdevice.div_rn(x, s))
        r = tl.minimum(tl.maximum(r, QMIN), QMAX)
    y = s * r
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=msk)


@triton.jit
def _fq_bwd(x_ptr, s_ptr, q_ptr, g_ptr, gx_ptr, part_ptr, M, C, BS: tl.constexpr, PER_ROW: tl.constexpr,
            QMIN: tl.constexpr, QMAX: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    """gx = g * inside (STE); scale-grad term = g * (qmin | qmax | round(sc)-sc).
    PER_ROW: part[m] = row sum (complete). else: part[pid, c] = column partial sums (host reduces pids + blocks)."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BC)
    rm = rows < M
    cm = cols < C
    msk = rm[:, None] & cm[None, :]
    offs = rows[:, None].to(tl.int64) * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    if PER_ROW:
        s = tl.load(s_ptr + rows, mask=rm, other=1.0)
        s = s.to(tl.bfloat16).to(tl.float32)[:, None]
        q = tl.load(q_ptr + rows, mask=rm, other=1.0)[:, None]
        lo = -q
        hi = q
    else:
        s = tl.load(s_ptr + cols // BS, mask=cm, other=1.0)
        s = s.to(tl.bfloat16).to(tl.float32)[None, :]
        lo = QMIN
        hi = QMAX
    sc = libdevice.div_rn(x, s)
    small = sc < lo
    big = sc > hi
    inside = (~small) & (~big)
    gx = tl.where(inside, g, 0.0)
    tl.store(gx_ptr + offs, gx.to(gx_ptr.dtype.element_ty), mask=msk)
    term = tl.where(small, lo, tl.where(big, hi, libdevice.rint(sc) - sc))
    t = tl.where(msk, g * term, 0.0)
    if PER_ROW:
        tl.store(part_ptr + rows, tl.sum(t, axis=1), mask=rm)
    else:
        tl.store(part_ptr + pid.to(tl.int64) * C + cols, tl.sum(t, axis=0), mask=cm)


@triton.jit
def _fq_bwd_act(x_ptr, s_ptr, g_ptr, gx_ptr, part_ptr, M, C, BS: tl.constexpr,
                QMIN: tl.constexpr, QMAX: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
    """activation backward, persistent over row tiles: part[pid, c] = column sums of g * term over this program's tiles"""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BC)
    cm = cols < C
    s = tl.load(s_ptr + cols // BS, mask=cm, other=1.0)
    s = s.to(tl.bfloat16).to(tl.float32)[None, :]
    acc = tl.zeros((BC,), dtype=tl.float32)
    for t0 in range(pid * BM, M, nprog * BM):
        rows = t0 + tl.arange(0, BM)
        msk = (rows < M)[:, None] & cm[None, :]
        offs = rows[:, None].to(tl.int64) * C + cols[None, :]
        x = tl.load(x_ptr + offs, mask=msk, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=msk, other=0.0).to(tl.float32)
        sc = libdevice.div_rn(x, s)
        small = sc < QMIN
        big = sc > QMAX
        inside = (~small) & (~big)
        tl.store(gx_ptr + offs, tl.where(inside, g, 0.0).to(gx_ptr.dtype.element_ty), mask=msk)
        term = tl.where(small, QMIN, tl.where(big, QMAX, libdevice.rint(sc) - sc))
        acc += tl.sum(tl.where(msk, g * term, 0.0), axis=0)
    tl.store(part_ptr + pid.to(tl.int64) * C + cols, acc, mask=cm)


_NPROG = None


def _act_cfg(C):
    """(BM, num_warps) for fwd and bwd, swept on RTX PRO 6000 at M=131072 (tune_fq.py)"""
    BC = triton.next_power_of_2(C)
    if BC <= 256:
        return BC, (16, 8), (16, 4)
    return BC, (16, 16), (16, 8)


def _cfg(M, C):
    BC = triton.next_power_of_2(C)
    BM = max(1, min(256, 8192 // BC))
    return BM, BC


# ------------------------------------------------------------------------------------------- autograd
class FusedActQuant(torch.autograd.Function):
    """== QuantizeFunction for buckets >= 5, scale_dtype bf16, epsilon None; blocks along the last dim"""

    @staticmethod
    def forward(ctx, x, flat_scale, bs, qmin, qmax, C):
        xc = x.contiguous()
        M = xc.numel() // C
        y = torch.empty_like(xc)
        BC, (BM, nw), _ = _act_cfg(C)
        _fq_fwd[(triton.cdiv(M, BM),)](xc, flat_scale, flat_scale, y, M, C, BS=bs, PER_ROW=False,
                                       QMIN=qmin, QMAX=qmax, BM=BM, BC=BC, num_warps=nw)
        ctx.save_for_backward(xc, flat_scale)
        ctx.meta = (bs, qmin, qmax, C)
        return y

    @staticmethod
    def backward(ctx, g):
        xc, flat_scale = ctx.saved_tensors
        bs, qmin, qmax, C = ctx.meta
        g = g.contiguous()
        M = xc.numel() // C
        global _NPROG
        if _NPROG is None:
            _NPROG = 4 * torch.cuda.get_device_properties(g.device).multi_processor_count
        BC, _, (BM, nw) = _act_cfg(C)
        grid = min(triton.cdiv(M, BM), _NPROG)
        gx = torch.empty_like(g)
        part = torch.empty((grid, C), dtype=torch.float32, device=g.device)
        _fq_bwd_act[(grid,)](xc, flat_scale, g, gx, part, M, C, BS=bs,
                             QMIN=qmin, QMAX=qmax, BM=BM, BC=BC, num_warps=nw)
        gs = part.sum(0).view(-1, bs).sum(-1)
        gs = gs / math.sqrt(xc.numel() * qmax)
        return gx, gs.to(g.dtype), None, None, None, None


class FusedRowQuant(torch.autograd.Function):
    """== rx3.PerRowQuantize for 2-D weights (one block = one row), per-row range [-q, q]"""

    @staticmethod
    def forward(ctx, w, flat_scale, qrow):
        wc = w.contiguous()
        M, C = wc.shape
        y = torch.empty_like(wc)
        BM, BC = _cfg(M, C)
        _fq_fwd[(triton.cdiv(M, BM),)](wc, flat_scale, qrow, y, M, C, BS=C, PER_ROW=True,
                                       QMIN=0, QMAX=0, BM=BM, BC=BC, num_warps=4)
        ctx.save_for_backward(wc, flat_scale, qrow)
        return y

    @staticmethod
    def backward(ctx, g):
        wc, flat_scale, qrow = ctx.saved_tensors
        g = g.contiguous()
        M, C = wc.shape
        BM, BC = _cfg(M, C)
        gx = torch.empty_like(g)
        part = torch.empty((M,), dtype=torch.float32, device=g.device)
        _fq_bwd[(triton.cdiv(M, BM),)](wc, flat_scale, qrow, g, gx, part, M, C, BS=C, PER_ROW=True,
                                       QMIN=0, QMAX=0, BM=BM, BC=BC, num_warps=4)
        gs = part / torch.sqrt(wc.numel() * qrow)
        return gx, gs.to(g.dtype), None


# --------------------------------------------------------------------------------------------- install
def _supported(q) -> bool:
    return (q.buckets >= 5 and q.scale_dtype == torch.bfloat16 and q.arithmetic_dtype == torch.float32
            and q.epsilon is None)


def install(raw_model, verbose=False):
    na = nw = 0
    for name, mod in raw_model.named_modules():
        if not isinstance(mod, Quantize) or not _supported(mod.quant):
            continue
        orig = mod.forward
        if mod.batched:
            def fwd(self, input, initialize_scales_steps, return_ints, _orig=orig):
                if initialize_scales_steps is not None or return_ints:
                    return _orig(input, initialize_scales_steps, return_ints)
                assert input.shape[-len(self.size):] == self.size
                return FusedActQuant.apply(input, self.scale.view(-1), self.quant.block_size,
                                           self.quant.qmin, self.quant.qmax, self.n_blocks * self.quant.block_size)
            na += 1
        elif hasattr(mod, "qrow") and len(mod.size) == 2 and mod.quant.block_size == mod.size[-1]:
            def fwd(self, input, initialize_scales_steps, return_ints, _orig=orig):
                if initialize_scales_steps is not None or return_ints:
                    return _orig(input, initialize_scales_steps, return_ints)
                assert input.shape == self.size
                return FusedRowQuant.apply(input, self.scale.view(-1), self.qrow)
            nw += 1
        else:
            continue
        mod.forward = types.MethodType(fwd, mod)
    if verbose:
        print("fakequant: fused %d activation quantizers, %d per-row weight quantizers" % (na, nw), flush=True)
    return na, nw
