# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/arch.py, sha256 0ad1daaaed6a2be2), GPL-3.
# AttnRes: Kimi Team (G. Chen, Y. Zhang, J. Su, W. Xu et al.), "Attention Residuals", arXiv 2603.15031 (2026).
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""v7 architecture arms for the cmix-neif transformer (Attention Residuals), installed on a built rx3 model.

    arch = {"attnres": {"mode": "replace" | "gated", "block": <layers per block, 0 = Full>}}

Attention Residuals (Kimi Team, arXiv 2603.15031; official pseudocode in the Opus addendum): every sub-layer
(mixer, MLP) reads a softmax mix over earlier outputs instead of the plain residual sum.  For source v_i the logit
is w . rmsnorm(v_i), computed as (v_i @ w) * rsqrt(mean(v_i^2) + eps) so no normalized copy is stored; the mix is
sum_i softmax_i * v_i.  w (one d_model vector per sub-layer, plus one for the final output) is zero-initialised,
i.e. uniform weights at the start.

mode "replace": the paper's form.  Sources = [x0 = initial stream (token+paket+prior embedding)] + one summary per
  completed block (sum of its sub-layer outputs) + the running partial sum of the current block (block > 0), or
  every single sub-layer output (block == 0, Full).  The config must turn skip_connections and
  token_embedding_connections OFF (they are subsumed: x0 is source 0).
mode "gated": on top of the unchanged baseline stream x (U-Net skips, token-embedding re-injection kept).  Each
  sub-layer reads  h = x + alpha * (mix(S + [x]) - x)  with a scalar alpha per sub-layer initialised to 0, so the
  initial function is EXACTLY the baseline (warm-start safe); the stream x itself is updated as in the baseline.

New parameters are 1-D and live inside model.blocks (or the Transformer for the final query), so
pysrc.optimizer.make_optimizers puts them into AdamW (blocks: ndim != 2) -- see install() for the final query.
Only the bf16 path is supported (no gradient checkpointing of the sub-layers).
"""
from __future__ import annotations

import os
import types

import torch
import torch.utils.checkpoint as _tckpt
from torch import nn

from pysrc.model import KimiLinearAttention, rms_norm, make_rope_args, make_flex_attention_block_mask, InitializeScales


EPS = 1e-6


class DepthQuery(nn.Module):
    """one AttnRes read: a zero-initialised query vector (+ the gate alpha in gated mode)"""

    def __init__(self, d: int, gated: bool):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32)) if gated else None


def _inv_rms(v: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(torch.linalg.vector_norm(v, dim=-1, dtype=torch.float32).square() / v.shape[-1] + EPS)


def _logit(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # w . rmsnorm(v) without materialising an fp32 copy of v: bf16 matvec, fp32 norm
    return (v @ w.to(v.dtype)).float() * _inv_rms(v)                    # [B, T]


class BlockStack:
    """completed block summaries as one [k, B, T, D] tensor (+ cached 1/rms [k, B, T]); grows at block boundaries"""

    def __init__(self, first: torch.Tensor):
        self.V = first.unsqueeze(0)
        self.R = _inv_rms(first).unsqueeze(0)

    def append(self, v: torch.Tensor) -> None:
        self.V = torch.cat([self.V, v.unsqueeze(0)], 0)
        self.R = torch.cat([self.R, _inv_rms(v).unsqueeze(0)], 0)

    def mix(self, q: "DepthQuery", extra: list[torch.Tensor]) -> torch.Tensor:
        V = self.V
        lb = (V @ q.w.to(V.dtype)).float() * self.R                     # [k, B, T]: one batched matvec
        le = [_logit(v, q.w).unsqueeze(0) for v in extra]
        p = torch.cat([lb] + le, 0).softmax(0).to(V.dtype)               # [k+e, B, T]
        k = V.shape[0]
        out = (p[:k].unsqueeze(-1) * V).sum(0)                          # one fused weighted reduction
        for j, v in enumerate(extra):
            out = out + p[k + j].unsqueeze(-1) * v
        return out


_ONES = {}


def _fused_mix_impl(sources: list[torch.Tensor], q: DepthQuery) -> torch.Tensor:
    """fla 0.5.1 fused AttnRes kernel (fla.ops.attnres): sources via a pointer table, no stacking; the key RMSNorm
    weight is fixed to ones (a learned one would be redundant with the query: w.(g*n) = (w*g).n)"""
    from fla.ops.attnres import fused_attnres
    v0 = sources[0]
    key = (v0.device, v0.shape[-1])
    if key not in _ONES:
        # not an inference tensor even if the first call runs under inference_mode (QAT scale init, eval-first entry
        # points); found by v6 msg 870
        with torch.inference_mode(False):
            _ONES[key] = torch.ones(v0.shape[-1], device=v0.device, dtype=torch.float32)
    return fused_attnres(q.w, [s.contiguous() for s in sources], _ONES[key], rms_eps=EPS)


fused_mix_eager = torch._dynamo.disable(_fused_mix_impl)


# ---- the same fla kernels as a torch.library custom op: opaque to dynamo, so no graph break per read
@torch.library.custom_op("v7::attnres_fwd", mutates_args=())
def _op_fwd(q: torch.Tensor, residuals: list[torch.Tensor], eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from fla.ops.attnres.fused import fused_attnres_fwd, _build_ptr_table
    res = _build_ptr_table(residuals)
    w = torch.ones(residuals[0].shape[-1], device=q.device, dtype=torch.float32)
    o, _, rstd, logit, lse = fused_attnres_fwd(q=q, residuals=residuals, res=res, w=w, ow=None, eps=eps, scale=1.0,
                                               checkpoint_level=1)
    return o, rstd, logit, lse


@_op_fwd.register_fake
def _(q, residuals, eps):
    r0 = residuals[0]
    L = len(residuals)
    return (torch.empty_like(r0), r0.new_empty((L, *r0.shape[:-1]), dtype=torch.float32),
            r0.new_empty((L, *r0.shape[:-1]), dtype=torch.float32), r0.new_empty(r0.shape[:-1], dtype=torch.float32))


@torch.library.custom_op("v7::attnres_bwd", mutates_args=())
def _op_bwd(do: torch.Tensor, q: torch.Tensor, residuals: list[torch.Tensor], rstd: torch.Tensor, logit: torch.Tensor,
            lse: torch.Tensor, eps: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    from fla.ops.attnres.fused import fused_attnres_bwd, _build_ptr_table
    res = _build_ptr_table(residuals)
    w = torch.ones(residuals[0].shape[-1], device=q.device, dtype=torch.float32)
    dvs, dq, _, _ = fused_attnres_bwd(do=do.contiguous(), q=q, residuals=residuals, res=res, w=w, ow=None, o_pre=None,
                                      rstd=rstd, logit=logit, lse=lse, eps=eps, scale=1.0, checkpoint_level=1)
    return list(dvs), dq


@_op_bwd.register_fake
def _(do, q, residuals, rstd, logit, lse, eps):
    return [torch.empty_like(r) for r in residuals], torch.empty_like(q)


def _setup(ctx, inputs, output):
    q, residuals, eps = inputs
    _, rstd, logit, lse = output
    ctx.save_for_backward(q, rstd, logit, lse, *residuals)
    ctx.eps = eps


def _backward(ctx, do, d_rstd, d_logit, d_lse):
    q, rstd, logit, lse, *residuals = ctx.saved_tensors
    dvs, dq = _op_bwd(do, q, list(residuals), rstd, logit, lse, ctx.eps)
    return dq, dvs, None


_op_fwd.register_autograd(_backward, setup_context=_setup)


def fused_mix_op(sources: list[torch.Tensor], q: "DepthQuery") -> torch.Tensor:
    D = sources[0].shape[-1]
    flat = [s.reshape(-1, D).contiguous() for s in sources]
    o, _, _, _ = _op_fwd(q.w, flat, EPS)
    return o.view(sources[0].shape)
TRACE_FUSED = False   # True: let dynamo trace fla's triton autograd.Function (fewer graph breaks)


def fused_mix(sources, q):
    return fused_mix_op(sources, q) if TRACE_FUSED else fused_mix_eager(sources, q)


IMPL = "fused"   # "fused" (fla kernel) | "stack" (BlockStack, block mode) | "naive" (python loop)
HUB = False      # v10 attnres_hub: block mode, every read via one per-forward hub (arch {"attnres": {"hub": true}})


def _hub_mod():
    import attnres_hub
    return attnres_hub


def depth_mix(sources: list[torch.Tensor], q: DepthQuery) -> torch.Tensor:
    if IMPL == "fused":
        return fused_mix(sources, q)
    logits = torch.stack([_logit(v, q.w) for v in sources], 0)          # [N, B, T] (small)
    p = logits.softmax(0).to(sources[0].dtype)
    out = p[0].unsqueeze(-1) * sources[0]
    for i in range(1, len(sources)):
        out = out + p[i].unsqueeze(-1) * sources[i]
    return out


def _attention_call(block, h, cu, rope_args, value_embedding, window_size, fbm, init):
    if isinstance(block.attention, KimiLinearAttention):
        return block.attention(rms_norm(h), cumulative_sequence_lengths=cu, value_embedding=value_embedding,
                               initialize_scales=init)
    return block.attention(rms_norm(h), cumulative_sequence_lengths=cu, rope_args=rope_args,
                           value_embedding=value_embedding, window_size=window_size,
                           flex_attention_block_mask=fbm, initialize_scales=init, layer=block.layer)


def install(raw, cfg, arch: dict | None, main: bool = True) -> dict:
    """patch raw (a pysrc.model.Transformer after install_paket) in place; returns a description"""
    if not arch or not arch.get("attnres"):
        return {}
    if getattr(raw, "_v7_arch", None) is not None:      # already installed (e.g. by a patched rx3.build_model)
        return raw._v7_arch
    global IMPL
    a = arch["attnres"]
    IMPL = a.get("impl", IMPL)
    global TRACE_FUSED
    TRACE_FUSED = bool(a.get("trace", False))
    global HUB
    HUB = bool(a.get("hub", False))
    assert not HUB or int(a.get("block", 2)) > 0, "attnres hub: block mode only"
    mode = a.get("mode", "replace")
    block_size = int(a.get("block", 2))
    assert mode in ("replace", "gated"), mode
    if mode == "replace":
        assert not cfg.skip_connections and not cfg.token_embedding_connections, \
            "attnres replace: set overrides skip_connections=false, token_embedding_connections=false"
    gated = mode == "gated"
    d = cfg.d_model
    dev = next(raw.parameters()).device
    for b in raw.blocks:
        b.q_mix = DepthQuery(d, gated).to(dev)
        b.q_mlp = DepthQuery(d, gated).to(dev)
    # the final read lives in the last block so make_optimizers (blocks -> AdamW for 1-D) covers it
    raw.blocks[-1].q_final = DepthQuery(d, gated).to(dev)
    n_new = sum(p.numel() for b in raw.blocks for n, p in b.named_parameters() if n.startswith("q_"))

    def compute_last_activation(self, input_tokens, priors, cumulative_sequence_lengths, override_window_size=None,
                                initialize_scales=InitializeScales(None, None)):
        cfg_ = self.cfg
        batch_size, sequence_length = input_tokens.shape
        window_size = override_window_size if override_window_size is not None else cfg_.window_size
        fbm = (make_flex_attention_block_mask(cumulative_sequence_lengths=cumulative_sequence_lengths,
                                              sequence_length=sequence_length, window_size=window_size,
                                              device=input_tokens.device)
               if cfg_.attention_implementation == "flex_attention" else None)
        if priors is not None:
            priors = priors.to(cfg_.activation_dtype)
        tok = self.embedding(input_tokens, initialize_scales=initialize_scales)
        if cfg_.embedding_norm:
            tok = rms_norm(tok)
        x = tok
        prior_embeddings = None
        if cfg_.prior_embedding:
            prior_embeddings = self.prior_embedding(self.prior_embedding_input(priors), initialize_scales=initialize_scales)
            if cfg_.prior_embedding_norm:
                prior_embeddings = rms_norm(prior_embeddings)
            x = x + prior_embeddings
        if cfg_.value_embedding_pattern is not None:
            hs = (batch_size, sequence_length, cfg_.n_key_value_heads, cfg_.d_head)
            ves = [e(input_tokens, initialize_scales=initialize_scales).view(hs) for e in self.value_embeddings]
            ve_by_layer = [ves[cfg_.value_embedding_pattern[l]] if cfg_.value_embedding_pattern[l] is not None else None
                           for l in range(cfg_.n_layers)]
        else:
            ve_by_layer = [None] * cfg_.n_layers
        rope_args = make_rope_args(sequence_length=sequence_length, cfg=cfg_, device=x.device)
        cu = cumulative_sequence_lengths

        skip_src = list(range(cfg_.n_layers // 2))
        skip_dst = list(range((cfg_.n_layers + 1) // 2, cfg_.n_layers))
        skips = []
        skw = self.skip_connection_weights() if cfg_.skip_connections else None

        # 2.0: activation checkpointing of the sub-layers in the qat/fp32 phases (cfg.gradient_checkpointing is
        # True there, False in bf16), so the QAT fits 32-GB cards at micro batch 131072 (the arch path of cmix-neif-beehive
        # ignored the flag and ran out of memory on 32-GB cards).  Recomputation in eager mode, values unchanged;
        # TF2_AR_CKPT=0 switches it off.  Never under no_grad/inference (evals, scale init).
        ckpt = (bool(cfg_.gradient_checkpointing) and self.training and torch.is_grad_enabled()
                and os.environ.get("TF2_AR_CKPT", "1") == "1")
        use_stack = block_size and IMPL == "stack"
        stack = BlockStack(x) if use_stack else None   # block mode, stacked implementation
        hub = _hub_mod().Hub() if HUB else None   # v10: per-forward AttnRes hub (one grad pass per source)
        if hub is not None:
            x = hub.source(x)
        done_blocks = [x]          # block mode: source 0 = initial stream, then one summary per completed block
        singles = [x]              # Full mode: every source separately
        partial = None             # running sum of the current block's sub-layer outputs (block mode)

        def mixed(q, extra):
            if block_size == 0:
                return depth_mix(singles + extra, q)
            cur = ([partial] if partial is not None else []) + extra
            if hub is not None:
                return hub.read(q.w, done_blocks + cur)
            return stack.mix(q, cur) if use_stack else depth_mix(done_blocks + cur, q)

        def read(q, stream):
            if not gated:
                return mixed(q, [])
            m = mixed(q, [hub.source(stream) if hub is not None else stream])
            return stream + q.alpha.to(stream.dtype) * (m - stream)

        def emit(out):
            nonlocal partial
            if block_size == 0:
                singles.append(out)
            else:
                partial = out if partial is None else partial + out
                if hub is not None:
                    partial = hub.source(partial)

        for layer, block in enumerate(self.blocks):
            if block_size and layer % block_size == 0 and partial is not None:
                if use_stack:
                    stack.append(partial)
                else:
                    done_blocks.append(partial)
                partial = None
            # baseline stream bookkeeping (gated mode keeps it exactly as Block.forward / Transformer do)
            if gated:
                if cfg_.skip_connections and layer in skip_dst:
                    x = x + skw[skip_dst.index(layer)] * skips.pop()
                if cfg_.token_embedding_connections or cfg_.prior_embedding_connections:
                    x = block.residual_stream_coefficient() * x
                if cfg_.token_embedding_connections:
                    x = x + block.token_embedding_coefficient() * tok
            h = read(block.q_mix, x)
            if ckpt:
                out = _tckpt.checkpoint(_attention_call, block, h, cu, rope_args, ve_by_layer[layer], window_size, fbm,
                                        initialize_scales, use_reentrant=False)
            else:
                out = _attention_call(block, h, cu, rope_args, ve_by_layer[layer], window_size, fbm, initialize_scales)
            emit(out)
            if gated:
                x = x + out
            h = read(block.q_mlp, x)
            if ckpt:
                out = _tckpt.checkpoint(lambda hh, b=block: b.mlp(rms_norm(hh), initialize_scales=initialize_scales), h,
                                        use_reentrant=False)
            else:
                out = block.mlp(rms_norm(h), initialize_scales=initialize_scales)
            emit(out)
            if gated:
                x = x + out
                if cfg_.skip_connections and layer in skip_src:
                    skips.append(x)
        h = read(self.blocks[-1].q_final, x)
        return rms_norm(h)

    raw.compute_last_activation = types.MethodType(compute_last_activation, raw)
    desc = {"attnres": {"mode": mode, "block": block_size, "new_params": n_new}}
    raw._v7_arch = desc
    if main:
        print("arch:", desc, flush=True)
    return desc
