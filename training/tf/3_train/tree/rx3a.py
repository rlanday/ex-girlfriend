# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/rx3a.py, sha256 333c3dfd677ecc05), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""rx3: one experiment = (optional) training phase + (optional) full-package eval.

    torchrun --nproc_per_node N rx3.py <config.json>

config keys:
  name, out (dir), phase: bf16 | qat | fp32 (numerics + default schedule)
  overrides: dict of TransformerConfig fields (e.g. n_key_value_heads, value_embedding_pattern, d_mlp)
  epochs (0 = eval only), lr, warmup_mult, final_mult, warmup_frac, scale_steps
  init: checkpoint path (optional), init_strict (bool)
  seed, mixed: path to .npz {module_name: int array of qmax per row} (qat/fp32 only)
  eval: true/false, eval_every (epochs, 0 = only at end), compile (bf16 only, default true)
  ve_sentinel: "zero" (default) | "clamp"   # value-embedding lookup for sentinel tokens >= 208
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
import types

TFT = os.environ.get("TFT_TREE", "/home/user/work/tft")
sys.path.insert(0, TFT)

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import pysrc.train as pysrc_train
from pysrc.model import Transformer, InitializeScales
from pysrc.optimizer import AdamWConfig, MuonConfig, OptimizerConfig
from pysrc.quantization import Quantize, QuantizeFunction
from pysrc.train import TrainingConfig, train
from pysrc.logger import LoggerConfig

from tft import config as tftcfg
from tft import runner as tftrun

import safe_prefetch  # 2.0: a loader failure must not end an epoch silently
assert safe_prefetch.install()


# ---------------------------------------------------------------- VE wrapper
class VEWrap(nn.Module):
    """value embedding indexed by the token field of the packed input (bits 0-7)"""

    def __init__(self, inner, vocab: int, mode: str):
        super().__init__()
        self.inner = inner
        self.vocab = vocab
        self.mode = mode

    def forward(self, input_tokens, initialize_scales):
        tok = input_tokens & 0xFF
        ok = tok < self.vocab
        if self.mode == "class":   # sentinel class c (token 208+c) -> row c
            idx = torch.where(ok, tok, tok - self.vocab)
            return self.inner(idx, initialize_scales=initialize_scales)
        if self.mode == "clamp":
            idx = torch.clamp(tok, max=self.vocab - 1)
            return self.inner(idx, initialize_scales=initialize_scales)
        idx = torch.where(ok, tok, torch.zeros_like(tok))
        out = self.inner(idx, initialize_scales=initialize_scales)
        return out * ok.unsqueeze(-1).to(out.dtype)


# ------------------------------------------------ per-row mixed precision
class PerRowQuantize(torch.autograd.Function):
    """QuantizeFunction (LSQ-style) with a per-block integer range [-q, q]"""

    @staticmethod
    def forward(ctx, blocks, flat_scale, quant, qrow):
        input_dtype = blocks.dtype
        blocks = blocks.to(quant.arithmetic_dtype)
        scale = flat_scale.unsqueeze(0).unsqueeze(-1).to(quant.scale_dtype).type_as(blocks)
        if quant.epsilon is not None:
            eps = torch.tensor(quant.epsilon, dtype=scale.dtype, device=scale.device)
            scale = (2 * (scale >= 0).type_as(scale) - 1) * scale.abs().maximum(eps)
        scaled = blocks / scale
        q = qrow.view(1, -1, 1).type_as(scaled)
        quantized = torch.maximum(torch.minimum(scaled.round(), q), -q)
        ctx.save_for_backward(scaled, q)
        ctx.quant = quant
        return (scale * quantized).to(input_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        scaled, q = ctx.saved_tensors
        quant = ctx.quant
        gdt = grad_out.dtype
        g = grad_out.to(quant.arithmetic_dtype)
        small = scaled < -q
        big = scaled > q
        inside = ~(small | big)
        grad_in = g * inside.type_as(g)
        term = small.type_as(g) * (-q) + big.type_as(g) * q + inside.type_as(g) * (scaled.round() - scaled)
        gs = (g * term).sum(0).sum(-1)
        # same normalization as QuantizeFunction: sqrt(numel of all blocks * qmax), here with the row's own qmax
        gs = gs / torch.sqrt(scaled.numel() * q.view(-1))
        return grad_in.to(gdt), gs.to(gdt), None, None


def install_mixed(raw_model, spec_path, main):
    spec = np.load(spec_path)
    n = 0
    for name, mod in raw_model.named_modules():
        if not isinstance(mod, Quantize) or mod.batched:
            continue
        key = name
        if key not in spec.files:
            continue
        qrow = torch.tensor(spec[key].astype(np.float32), device=mod.scale.device)
        assert qrow.numel() == mod.n_blocks, (name, qrow.numel(), mod.n_blocks)
        mod.register_buffer("qrow", qrow)

        def fwd(self, input, initialize_scales_steps, return_ints):
            blocks = self.reshape_input(input)
            if initialize_scales_steps is not None:
                # scaled_max init with the row's own range
                with torch.no_grad():
                    s = blocks.type_as(self.scale).abs().amax((0, -1)) / self.qrow
                    s = s.reshape(self.scale.shape).type_as(self.scale)
                    torch.maximum(self.scale, s, out=self.scale)
                return input
            if return_ints:
                with torch.no_grad():
                    sc = self.scale.view(1, -1, 1).to(self.quant.scale_dtype).type_as(blocks)
                    q = self.qrow.view(1, -1, 1).type_as(blocks)
                    ints = torch.maximum(torch.minimum((blocks.to(self.quant.arithmetic_dtype) / sc).round(), q), -q)
                return ints.to(torch.int64).view(input.shape)
            out = PerRowQuantize.apply(blocks, self.scale.view(-1), self.quant, self.qrow)
            return out.view(input.shape)

        mod.forward = types.MethodType(fwd, mod)
        n += 1
    if main:
        print("mixed precision: %d weight quantizers got per-row ranges from %s" % (n, spec_path), flush=True)
    return n


# ------------------------------------------------------------------- model
def build_model(cfgj, phase, device, main):
    g = tftcfg.geometry(os.environ.get("TFT_GEOMETRY", "poc8m_w2048"))
    cfg = tftcfg.make_config(phase, geometry_name=g.name, variable_sequence_length=phase != "fp32")
    ov = dict(cfgj.get("overrides", {}))
    if ov:
        cfg = dataclasses.replace(cfg, **ov)
    raw = Transformer(cfg).to(device)
    from zctx import zpaket
    from zctx.zpmodel import install_paket
    install_paket(raw, cfg, x=int(os.environ.get("PK_LAST_X", str(zpaket.DEFAULT_X))),
                  d_slot=int(os.environ.get("PK_SLOT_DIM", "16")), use_last=True, use_match=True)
    if getattr(raw, "value_embeddings", None) is not None:
        for i in range(len(raw.value_embeddings)):
            raw.value_embeddings[i] = VEWrap(raw.value_embeddings[i], cfg.vocabulary_size,
                                             cfgj.get("ve_sentinel", "zero"))
    raw = raw.to(device)
    if cfgj.get("arch"):   # v7: architecture arms (AttnRes); must run BEFORE load_init so the new keys exist
        import arch
        arch.install(raw, cfg, cfgj["arch"], main)
    return raw, cfg


def load_init(raw, path, strict, device, main):
    sd = torch.load(path, map_location=device, weights_only=True)
    # per-row ranges of mixed precision are re-installed from the npz by install_mixed (after this load)
    sd = {k: v for k, v in sd.items() if not k.endswith(".qrow")}
    res = raw.load_state_dict(sd, strict=strict)
    if main:
        print("loaded %s (strict=%s) missing=%d unexpected=%d" % (path, strict, len(res.missing_keys), len(res.unexpected_keys)), flush=True)
        if res.unexpected_keys:
            print("  unexpected:", res.unexpected_keys[:10], flush=True)
        if res.missing_keys:
            print("  missing (first):", res.missing_keys[:10], flush=True)


# -------------------------------------------------------------------- eval
@torch.no_grad()
def full_eval(raw, data, rank, world, device):
    raw.eval()
    n = data.n_articles
    groups = list(data._iter_micro_batch_articles(range(n), drop_last=False))
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    tok_sum = torch.zeros((), dtype=torch.float64, device=device)
    none = InitializeScales(weight_steps=None, activation_steps=None)
    for gi in range(rank, len(groups), world):
        mb = data._make_micro_batch(groups[gi])
        loss = raw(input_tokens=mb.input_tokens, priors=mb.priors, loss_mask=mb.loss_mask,
                   output_tokens=mb.output_tokens, cumulative_sequence_lengths=mb.cumulative_sequence_lengths,
                   override_window_size=None)
        loss_sum += loss.double()
        tok_sum += mb.loss_mask.sum().double()
    if world > 1:
        dist.all_reduce(loss_sum)
        dist.all_reduce(tok_sum)
    raw.train()
    nats = (loss_sum / tok_sum).item()
    return {"nats_per_tok": nats, "bits_per_tok": nats / math.log(2), "n_scored": int(tok_sum.item())}


# -------------------------------------------------------------------- main
def main():
    cfgj = json.load(open(sys.argv[1]))
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    main_p = rank == 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    out = cfgj["out"]
    os.makedirs(out, exist_ok=True)
    seed = int(cfgj.get("seed", 1))
    torch.manual_seed(seed * 1000 + rank)

    phase = cfgj["phase"]
    raw, cfg = build_model(cfgj, phase, device, main_p)
    if cfgj.get("init"):
        load_init(raw, cfgj["init"], bool(cfgj.get("init_strict", False)), device, main_p)
    if cfgj.get("mixed") and phase in ("qat", "fp32"):
        install_mixed(raw, cfgj["mixed"], main_p)
    if cfgj.get("l1rho") and phase in ("qat", "fp32"):
        # entropy pressure on the quantized codes: add rho*|g| * u/|u| to each quantized weight's gradient,
        # u = sign(w)/row_scale = d/dw sum|w/s|  (pulls codes toward 0 -> peakier int distribution, smaller container)
        rho = float(cfgj["l1rho"]); nh = 0
        mods = dict(raw.named_modules())
        for name, mod in mods.items():
            if not name.endswith(".quantize_weight") or getattr(mod, "batched", False):
                continue
            owner = mods.get(name[: -len(".quantize_weight")])
            if owner is None or not hasattr(owner, "weight") or owner.weight.ndim != 2:
                continue
            w = owner.weight

            def hook(g, w=w, q=mod):
                with torch.no_grad():
                    sc = q.scale.detach().reshape(-1, 1).abs().clamp_min(1e-12).to(g.dtype)
                    if sc.shape[0] != g.shape[0]:
                        return g
                    u = torch.sign(w.detach()).to(g.dtype) / sc
                    un = u.norm()
                    if not torch.isfinite(un) or un == 0:
                        return g
                    return g + rho * g.norm() * u / un
            w.register_hook(hook); nh += 1
        if main_p:
            print("l1rho %.3f: entropy-pressure hooks on %d quantized weights" % (rho, nh), flush=True)
    nparams = sum(p.numel() for p in raw.parameters())
    if main_p:
        print(json.dumps({"name": cfgj["name"], "phase": phase, "params": nparams, "world": world}), flush=True)

    epochs = int(cfgj.get("epochs", 0))
    base = tftcfg.IVANOV_SCHEDULES[phase]
    wfrac = float(cfgj.get("warmup_frac", tftcfg.WARMUP_FRACTION[phase]))
    sched = tftcfg.PhaseSchedule(
        epochs=max(epochs, 1),
        warmup_epochs=min(max(epochs, 1), round(wfrac * max(epochs, 1))),
        learning_rate=float(cfgj.get("lr", base.learning_rate)),
        warmup_learning_rate_multiplier=float(cfgj.get("warmup_mult", base.warmup_learning_rate_multiplier)),
        final_learning_rate_multiplier=float(cfgj.get("final_mult", base.final_learning_rate_multiplier)),
        initialize_scales_steps=(None if phase != "qat" or cfgj.get("no_scale_init") else int(cfgj.get("scale_steps", base.initialize_scales_steps))),
    )
    data = tftrun.make_data_loader(max(epochs, 1), device, rank, world)
    result = {"name": cfgj["name"], "config": cfgj, "params": nparams, "world": world, "evals": []}

    if epochs > 0:
        model = DDP(raw)
        if phase == "bf16" and cfgj.get("compile", True):
            model = torch.compile(model)
        ckpt = os.path.join(out, "model.tch")
        tftrun._CHECKPOINT.update(path=ckpt, every_epochs=int(cfgj.get("ckpt_every", 1)), rank=rank, raw_model=raw)
        eval_every = int(cfgj.get("eval_every", 0))

        class EvalLogger(tftrun.CheckpointingLogger):
            def train_step(self, epoch, loss, n_tokens, learning_rates, last_step):
                prev = int(self._last_epoch)
                super().train_step(epoch=epoch, loss=loss, n_tokens=n_tokens, learning_rates=learning_rates, last_step=last_step)
                if eval_every and int(epoch) != prev and int(epoch) % eval_every == 0 and not last_step:
                    r = full_eval(raw, data, rank, world, device)
                    r.update(epoch=int(epoch), time=time.time())
                    result["evals"].append(r)
                    if main_p:
                        print("EVAL", json.dumps(r), flush=True)
                        json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)

        pysrc_train.Logger = EvalLogger
        tcfg = TrainingConfig(
            initialize_scales_steps=sched.initialize_scales_steps,
            initialize_weight_scales_first=True,
            optimizer_config=OptimizerConfig(
                learning_rate_multiplier_schedule=tftrun.cosine_multiplier_schedule(
                    epochs=sched.epochs, warmup_epochs=sched.warmup_epochs,
                    warmup_learning_rate_multiplier=sched.warmup_learning_rate_multiplier,
                    final_learning_rate_multiplier=sched.final_learning_rate_multiplier),
                second_order_config=MuonConfig(base_learning_rate=sched.learning_rate, weight_decay=0.1,
                                               momentum=0.95, adjust_lr_fn="match_rms_adamw"),
                adamw_config=AdamWConfig(base_learning_rate=sched.learning_rate, weight_decay=0.01, betas=(0.95, 0.99)),
            ),
            window_size_schedule=None,
            logger_config=LoggerConfig(log_to_stdout=False, file=os.path.join(out, "log.jsonl") if main_p else None,
                                       wandb_project=None, wandb_run_name=cfgj["name"]),
        )
        t0 = time.time()
        train(raw_model=raw, model=model, data=data, cfg=tcfg)
        result["train_s"] = time.time() - t0
        if main_p:
            torch.save(raw.state_dict(), ckpt)
    if cfgj.get("eval", True):
        t0 = time.time()
        r = full_eval(raw, data, rank, world, device)
        r.update(epoch=epochs, tag="final", eval_s=time.time() - t0, time=time.time())
        result["evals"].append(r)
        if main_p:
            print("EVAL", json.dumps(r), flush=True)
    if main_p:
        result["done"] = time.time()
        json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
