# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/rx7r.py, sha256 9696be341241538c), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""rx7r (v6, admin 2021 long-QAT directive): rx7q + four opt-in keys, everything else unchanged.
  hold_tail {"hold_epochs": H, "hold_mult": m, "tail_plan_epochs": T, "tail_from_epoch": e0}: LR multiplier m for the
      first H stage epochs (exact epoch boundary in batches), then the T-epoch plan from e0 to its end mapped onto the
      remaining batches (= rx7q plan_tail over the rest of the run; v3's production tail is {T 2000, e0 1344}).
  rate_term.start_epoch S: C1 contributes nothing before stage epoch S; its setup() runs at the first step of epoch S
      on the weights of that moment (as v3's production C1 is set up at E1344); rate.json exists only from then on.
  keep_epochs [..]: immutable milestone copies of the state at these stage epochs (in addition to keep_every).
  clamp_to_grid true: after every optimizer step clamp each quantized weight row to +-qrow*s*(1-1e-6) (s = bf16 scale),
      the edge of the STE mask: forward unchanged (those weights quantize to +-qrow anyway), clipped weights get gradient again.
  cosine_exact {"from": m0, "to": m1}: (admin 2084) for N updates and update index i = 0..N-1, multiplier
      m1 + (m0 - m1) * (1 + cos(pi * i / (N - 1))) / 2 (first update m0, last m1); replaces the plan shape.
  rate_term.apply_from_epoch E: C1 is SET UP at the start as usual (same pools in every arm), but its gradient is applied
      only from the first update after the cumulative-batch boundary of stage epoch E (exact, from the loader).
  init_only true: after scale init + fakequant + C1 setup, save the state at step 0 and exit (shared post-init state).
      A resume from a step-0 state does NOT redo the scale init.
  probe_full_eval true (with probe_only): also run the full eval (QAT mode) after the reload and print EVAL.
  probe {"batches": k}: fixed-input check = mean loss of the first k eval micro-batches (eval mode), printed as PROBE
      before every STOP/stop_at state save; probe_only true: resume, print PROBE, re-save model/opt to out/reload, exit.

rx7q (v7): rx7 + a resumable QAT-phase continuation (A1 / YOLO-1.5: QAT inside the cooldown).

rx7q keys on top of rx7: phase "qat" (default "bf16"), mixed (Fisher npz, installed right after init), scale_steps
(activation scale-init steps, default tft qat 256), compile (also for qat, default true), init_opt_phase ("bf16":
the init_opt moments come from a bf16-phase model with the same arch). Typical A1 probe: init = YOLO-1 e1800
model.tch, init_opt = its opt.pt, lr 5e-4, anneal {from_mult 0.226, to_mult 0.025}, epochs 64, mixed = Fisher npz.

rx7 (v7): rx6 + architecture arms (arch.py) + warm-start continuation.

rx7 keys on top of rx6: arch (see arch.py), rewarm {"start","peak","frac","final"} (cosine start->peak over the first
frac of the plan, then cosine peak->final), anneal {"from_mult": m0, "to_mult": m1} = cosine LR multiplier m0 -> m1
over the plan (replaces the YOLO plan shape), init_opt (opt.pt of the init checkpoint; moments carried over by
parameter NAME, new parameters start fresh).

rx6: resumable long bf16 run for the yolo (v6). Same model/numerics/loop as rx3 + pysrc.train.train.

    torchrun --nproc_per_node 8 rx6.py <config.json>

config keys (rx3 keys plus):
  epochs        total epochs T of the ONE plan (loader permutations and LR fraction are over T)
  state_every   save full state (model + Muon/AdamW moments + step) every N epochs (default 5)
  keep_every    also keep a milestone copy every N epochs (default 100)
  resume        "auto" (default): continue from <out>/state if it exists; "none": cold start
  stop_at       optional epoch: save state and exit cleanly at that epoch boundary
A file <out>/STOP makes the run save its state and exit at the next epoch boundary.

A resume rebuilds the loader with the same T (identical seed-0 permutations), skips the batches
already trained without reading their data, restores both optimizers and continues the same
cosine plan (fraction_done = batches_done / total_batches), so it is the run that would have
happened without the interruption, up to nondeterminism of the GPU kernels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rx3  # noqa: E402  (sets up sys.path for the trainer tree)
import arch  # noqa: E402
import speedpatch  # noqa: E402  (v3 throughput patch: KDA disable_recompute + fla outside dynamo; bit-identical, parity-tested msg 677)
speedpatch.install()
speedpatch.install_pkd()  # v3 msg 680: host cu_seqlens from the pkd loader too (parity-tested)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from pysrc.data import Batch  # noqa: E402
from safe_prefetch import prefetch  # noqa: E402  2.0: re-raises loader failures
from pysrc.optimizer import AdamWConfig, MuonConfig, OptimizerConfig, make_optimizers  # noqa: E402
from tft import config as tftcfg  # noqa: E402
from tft import runner as tftrun  # noqa: E402


# 2.0: rank check.  At the update counts of RANK_CHECK_STEPS (default 20,1000,
# counted from the start of this launch) every rank all-gathers a checksum of its parameters; any difference stops
# the run ('replicas differ', the chain greps for it).
def rank_check(raw, n_launch, step, rank, world, device):
    steps = [int(x) for x in os.environ.get("RANK_CHECK_STEPS", "20,1000").split(",") if x.strip()]
    if n_launch not in steps or world < 2:
        return
    with torch.no_grad():
        v = torch.zeros(2, dtype=torch.float64, device=device)
        for p in raw.parameters():
            d = p.detach().double()
            v[0] += d.sum()
            v[1] += d.abs().sum()
        allv = [torch.zeros_like(v) for _ in range(world)]
        dist.all_gather(allv, v)
    vals = [tuple(float(x) for x in t.tolist()) for t in allv]
    ok = all(x == vals[0] for x in vals)
    if rank == 0:
        print("rank check step %d (global %d): %s -> %s" % (n_launch, step, " | ".join("%.17g/%.17g" % x for x in vals),
              "EQUAL" if ok else "DIFFER"), flush=True)
    if not ok:
        raise RuntimeError("replicas differ at step %d" % step)


def batches(data, skip: int):
    """data._iter_no_prefetch with the first `skip` batches passed over without building them"""
    done = 0
    for epoch in range(data.epochs):
        per = data._batches_per_epoch[epoch]
        if done + per <= skip:          # whole epoch already trained
            done += per
            continue
        for k, groups in enumerate(data._iter_batch_article_groups(data._permutations[epoch]), start=1):
            done += 1
            if done <= skip:
                continue
            yield Batch(
                micro_batches=[data._make_micro_batch(a) for i, a in enumerate(groups)
                               if i % data.world_size == data.rank],
                epoch=epoch + k / per,
                fraction_done=done / data._total_batches,
                is_last=done == data._total_batches,
            )


def plan_signature(cfgj, data, world):
    keys = ("phase", "overrides", "epochs", "lr", "opt_batch_tokens", "warmup_mult", "final_mult", "warmup_frac", "seed", "ve_sentinel",
            "arch", "anneal", "rewarm", "init", "plan_tail", "rate_term", "mixed", "hold_tail", "clamp_to_grid", "cosine_exact")
    if cfgj.get("stretch"):
        keys = keys + ("stretch",)
    sig = {k: cfgj.get(k) for k in keys}
    sig.update(world=world, total_batches=data._total_batches,
               micro=tftcfg.MICRO_BATCH_TOKENS, accum=data.gradient_accumulation_steps)
    return sig


RATE_MOD = None   # set in main() when 'rate_term' is enabled


def make_hold_tail(plan, ht, batches_per_epoch, total_batches):
    """rx7r: hold multiplier for H epochs, then plan(T) from e0 to 1 over the rest (fractions in batches)"""
    H, m = int(ht["hold_epochs"]), float(ht.get("hold_mult", 1.0))
    T_old, e0 = int(ht["tail_plan_epochs"]), float(ht["tail_from_epoch"])
    old_plan = plan(T_old)
    f0 = sum(batches_per_epoch[:H]) / total_batches
    span = (T_old - e0) / T_old

    def schedule(f):
        f = min(max(f, 0.0), 1.0)
        if f < f0:
            return m
        return old_plan(min(e0 / T_old + span * (f - f0) / (1.0 - f0), 1.0))
    return schedule, f0


def probe_loss(raw, data, k, device):
    """rx7r: mean loss/token over the first k eval micro-batches (fixed input), eval mode, this rank"""
    raw.eval()
    groups = list(data._iter_micro_batch_articles(range(data.n_articles), drop_last=False))[:k]
    ls = torch.zeros((), dtype=torch.float64, device=device)
    ts = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for g in groups:
            mb = data._make_micro_batch(g)
            ls += raw(input_tokens=mb.input_tokens, priors=mb.priors, loss_mask=mb.loss_mask, output_tokens=mb.output_tokens,
                      cumulative_sequence_lengths=mb.cumulative_sequence_lengths, override_window_size=None).double()
            ts += mb.loss_mask.sum().double()
    raw.train()
    return (ls / ts).item()


def save_state(out, raw, opts, step, epoch, sig, keep_every, keep_epochs=()):
    d = os.path.join(out, "state")
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save(raw.state_dict(), os.path.join(tmp, "model.tch"))
    torch.save({"second_order": opts.second_order.state_dict(), "adamw": opts.adamw.state_dict()},
               os.path.join(tmp, "opt.pt"))
    if RATE_MOD is not None:   # C1 frozen pool/row labels etc. (v5 msg 971): restored on resume instead of setup()
        json.dump(RATE_MOD.state_dict(raw), open(os.path.join(tmp, "rate.json"), "w"))
    meta = {"step": step, "epoch": epoch, "time": time.time(), "plan": sig}
    for f in ("model.tch", "opt.pt", "rate.json"):
        if not os.path.exists(os.path.join(tmp, f)):
            continue
        h = hashlib.sha256(open(os.path.join(tmp, f), "rb").read()).hexdigest()
        meta["sha256_" + f] = h
    json.dump(meta, open(os.path.join(tmp, "state.json"), "w"), indent=1)
    old = d + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(d):
        os.rename(d, old)
    os.rename(tmp, d)
    shutil.rmtree(old, ignore_errors=True)
    if (keep_every and epoch % keep_every == 0) or epoch in keep_epochs:
        m = os.path.join(out, "milestones", "e%04d" % epoch)
        shutil.rmtree(m, ignore_errors=True)
        shutil.copytree(d, m)
    print("STATE saved epoch %d step %d" % (epoch, step), flush=True)


def carry_optimizer_state(cfgj, raw, opts, device, main_p):
    """moments of init_opt (saved by rx6 for the PLAIN model) -> this model's optimizers, matched by parameter name"""
    ref, ref_cfg = rx3.build_model(dict(cfgj, arch=cfgj.get("init_opt_arch")), cfgj.get("init_opt_phase", "bf16"),
                                   torch.device("cpu"), False)
    # the unpatched rx3.build_model does not install arch: do it here so the reference optimizer has the same
    # parameter groups as the model that saved init_opt (AttnRes queries are in model.blocks -> AdamW group)
    # arch.install sets module globals (IMPL, TRACE_FUSED, HUB) read at forward time: restore the main model's values,
    # otherwise e.g. init_opt_arch without "hub" silently switches the hub off for the trained model (v7 15:00 bug)
    saved_globals = (arch.IMPL, arch.TRACE_FUSED, arch.HUB)
    arch.install(ref, ref_cfg, cfgj.get("init_opt_arch"), False)
    arch.IMPL, arch.TRACE_FUSED, arch.HUB = saved_globals
    ref_opts = make_optimizers(ref, OptimizerConfig(
        learning_rate_multiplier_schedule=lambda f: 1.0,
        second_order_config=MuonConfig(base_learning_rate=1.0, weight_decay=0.1, momentum=0.95, adjust_lr_fn="match_rms_adamw"),
        adamw_config=AdamWConfig(base_learning_rate=1.0, weight_decay=0.01, betas=(0.95, 0.99))))
    o = torch.load(cfgj["init_opt"], map_location="cpu", weights_only=False)
    ref_opts.second_order.load_state_dict(o["second_order"])
    ref_opts.adamw.load_state_dict(o["adamw"])
    by_name = {}
    ref_names = {id(p): n for n, p in ref.named_parameters()}
    for opt in (ref_opts.second_order, ref_opts.adamw):
        for p, st in opt.state.items():
            by_name[ref_names[id(p)]] = st
    n_ok = n_new = 0
    names = {id(p): n for n, p in raw.named_parameters()}
    for opt in (opts.second_order, opts.adamw):
        for g in opt.param_groups:
            for p in g["params"]:
                st = by_name.get(names[id(p)])
                if st is None:
                    n_new += 1
                    continue
                opt.state[p] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in st.items()}
                n_ok += 1
    if main_p:
        print("init_opt: carried moments for %d parameters, %d new parameters start fresh" % (n_ok, n_new), flush=True)


def main():
    cfgj = json.load(open(sys.argv[1]))
    phase = cfgj.get("phase", "bf16")
    assert phase in ("bf16", "qat"), "rx7q runs bf16 or qat phases"
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

    raw, cfg = rx3.build_model(cfgj, phase, device, main_p)
    arch_desc = arch.install(raw, cfg, cfgj.get("arch"), main_p)
    if cfgj.get("arch"):
        # AttnRes reads have a different source count per sub-layer position, so dynamo specialises the arch frame
        # per position (33 variants); the default recompile_limit (8) makes it fall back to EAGER for the rest,
        # which hid the v10 kernel's no-graph-break gain in the v7 acceptance pair (0.997x). Numerics unchanged.
        torch._dynamo.config.recompile_limit = max(int(torch._dynamo.config.recompile_limit), 64)
    if cfgj.get("fmlp") and phase == "bf16":
        import fmlp        # v10: relu^2 MLP as one custom op, bit-identical (msg 1005); skips quantized MLPs
        fmlp.install(raw)
    if cfgj.get("kseg10"):
        import kseg10      # v10: segment-parallel KDA recurrence, bit-identical on real layouts; needs install_pkd
        kseg10.install()
    if cfgj.get("kda_bwdf"):
        import kda_bwd_fused   # v10: fused raw-CUDA KDA backward (wy_dqkg + intra), not bitwise, more accurate (msg 1008)
        kda_bwd_fused.install()
    if cfgj.get("attnres_v10"):
        import attnres_v10   # v10's fused AttnRes custom op (msg 940): swaps arch.depth_mix for both modes
        attnres_v10.install(verbose=main_p)
    if cfgj.get("init"):
        rx3.load_init(raw, cfgj["init"], bool(cfgj.get("init_strict", False)), device, main_p)
    if phase == "qat" and cfgj.get("init") and cfgj.get("lift_subnormal", True):
        # tf/3_train/fix_subnormal_rows.py rule on the loaded weights: every row of a 2-D float tensor with
        # 0 < |max| < 1e-30 is scaled to |max| 1e-30 (direction kept); else QAT scales overflow (NaN)
        with torch.no_grad():
            n_rows = 0
            for k, v in raw.state_dict().items():
                if not torch.is_tensor(v) or v.dim() != 2 or not v.is_floating_point():
                    continue
                m = v.float().abs().amax(1)
                idx = ((m < 1e-30) & (m > 0)).nonzero().flatten()
                if len(idx):
                    f = v.float()
                    f[idx] = f[idx] / m[idx, None] * 1e-30
                    v.copy_(f.to(v.dtype))
                    n_rows += len(idx)
        if main_p:
            print("lift_subnormal: %d rows lifted to |max| 1e-30" % n_rows, flush=True)
    if phase == "qat" and cfgj.get("mixed"):
        rx3.install_mixed(raw, cfgj["mixed"], main_p)   # before any resume load: registers the .qrow buffers
    nparams = sum(p.numel() for p in raw.parameters())

    epochs = int(cfgj["epochs"])
    base = tftcfg.IVANOV_SCHEDULES["bf16"]
    wfrac = float(cfgj.get("warmup_frac", tftcfg.WARMUP_FRACTION["bf16"]))
    lr = float(cfgj.get("lr", base.learning_rate))
    t0 = time.time()
    data = tftrun.make_data_loader(epochs, device, rank, world)
    if main_p:
        print("loader for %d epochs: %d batches, built in %.0f s" % (epochs, data._total_batches, time.time() - t0), flush=True)
    sig = plan_signature(cfgj, data, world)

    def plan(n_epochs):
        return tftrun.cosine_multiplier_schedule(
            epochs=n_epochs, warmup_epochs=min(n_epochs, round(wfrac * n_epochs)),
            warmup_learning_rate_multiplier=float(cfgj.get("warmup_mult", base.warmup_learning_rate_multiplier)),
            final_learning_rate_multiplier=float(cfgj.get("final_mult", base.final_learning_rate_multiplier)))

    schedule = plan(epochs)
    if cfgj.get("rewarm"):
        # re-warm continuation: cosine start -> peak over the first `frac` of the plan, then cosine peak -> final
        rw = cfgj["rewarm"]
        r0, rp, rf, rfin = float(rw["start"]), float(rw["peak"]), float(rw["frac"]), float(rw["final"])

        def schedule(f, r0=r0, rp=rp, rf=rf, rfin=rfin):
            f = min(max(f, 0.0), 1.0)
            if rf > 0 and f < rf:
                return r0 + (rp - r0) * 0.5 * (1.0 - math.cos(math.pi * f / rf))
            g = (f - rf) / (1.0 - rf) if rf < 1 else 1.0
            return rfin + (rp - rfin) * 0.5 * (1.0 + math.cos(math.pi * g))
    if cfgj.get("plan_tail"):
        # continue ANOTHER plan's LR exactly: YOLO-1's 2000-epoch plan (8x->1x over 2/3, cosine -> 0.025x) from
        # epoch from_epoch on, mapped onto this run's (total - from) epochs; YOLO-1.5 then differs from YOLO-1 only in QAT
        pt = cfgj["plan_tail"]
        T_old, e0 = int(pt["epochs"]), float(pt["from_epoch"])
        old_plan = plan(T_old)
        span = (T_old - e0) / T_old

        def schedule(f, old_plan=old_plan, e0=e0, T_old=T_old, span=span):
            return old_plan(min(e0 / T_old + span * min(max(f, 0.0), 1.0), 1.0))
    if cfgj.get("hold_tail"):
        assert not cfgj.get("plan_tail") and not cfgj.get("rewarm"), "hold_tail replaces plan_tail/rewarm"
        schedule, _f0 = make_hold_tail(plan, cfgj["hold_tail"], data._batches_per_epoch, data._total_batches)
        if main_p:
            print("hold_tail: %s, switch at fraction %.6f (stage epoch %d); LR at f=0 %.5f, f0- %.5f, f0 %.5f, f=1 %.5f"
                  % (json.dumps(cfgj["hold_tail"]), _f0, int(cfgj["hold_tail"]["hold_epochs"]), schedule(0.0),
                     schedule(_f0 - 1e-9), schedule(_f0), schedule(1.0)), flush=True)
    if cfgj.get("cosine_exact"):
        _ce = cfgj["cosine_exact"]; _N = data._total_batches
        _m0, _m1 = float(_ce.get("from", 1.0)), float(_ce.get("to", 0.025))

        def schedule(f, _N=_N, _m0=_m0, _m1=_m1):
            i = min(max(int(round(f * _N)) - 1, 0), _N - 1)   # trainer passes f = (i + 1) / N for update index i
            return _m1 + (_m0 - _m1) * 0.5 * (1.0 + math.cos(math.pi * i / max(_N - 1, 1)))
        if main_p:
            print("cosine_exact: N %d updates, LR(i=0) %.6f, LR(mid) %.6f, LR(i=N-1) %.6f" % (_N, schedule(1 / _N), schedule((_N // 2 + 1) / _N), schedule(1.0)), flush=True)
    if cfgj.get("anneal"):
        m0, m1 = float(cfgj["anneal"]["from_mult"]), float(cfgj["anneal"]["to_mult"])

        def schedule(f, m0=m0, m1=m1):
            return m1 + (m0 - m1) * 0.5 * (1.0 + math.cos(math.pi * min(max(f, 0.0), 1.0)))
    old_sig = None
    st = cfgj.get("stretch")
    if st:
        # TIME-STRETCH: the run was planned for st["from_epochs"]; from step st["at_step"] on, the rest of that
        # plan is spread over the longer horizon `epochs` (LR continuous at at_step, same cosine shape, same
        # seed-0 permutations for the first from_epochs epochs).
        e0, s0 = int(st["from_epochs"]), int(st["at_step"])
        b_old = sum(data._batches_per_epoch[:e0])
        b_new = data._total_batches
        assert e0 < epochs and 0 < s0 < b_old < b_new, (e0, s0, b_old, b_new)
        old_plan = plan(e0)

        def schedule(f, old_plan=old_plan, s0=s0, b_old=b_old, b_new=b_new):
            done = f * b_new
            f_old = done / b_old if done <= s0 else (s0 + (done - s0) * (b_old - s0) / (b_new - s0)) / b_old
            return old_plan(min(f_old, 1.0))

        old_sig = dict(sig, epochs=e0, total_batches=b_old)
        old_sig.pop("stretch", None)
        if main_p:
            print("stretch: plan of %d epochs (%d batches) stretched to %d epochs (%d batches) from step %d"
                  % (e0, b_old, epochs, b_new, s0), flush=True)

    if cfgj.get("fastmuon"):
        import fastmuon   # v10: batched-by-shape Muon, same algorithm, opt.pt-compatible both ways (msg 921)
        fastmuon.install(verbose=main_p)
        make_optimizers.__globals__["Muon"] = fastmuon.FastMuon
    opts = make_optimizers(raw, OptimizerConfig(
        learning_rate_multiplier_schedule=schedule,
        second_order_config=MuonConfig(base_learning_rate=lr, weight_decay=0.1, momentum=0.95,
                                       adjust_lr_fn="match_rms_adamw"),
        adamw_config=AdamWConfig(base_learning_rate=lr, weight_decay=0.01, betas=(0.95, 0.99)),
    ))

    if cfgj.get("init_opt") and not os.path.isdir(os.path.join(out, "state")):
        carry_optimizer_state(cfgj, raw, opts, device, main_p)

    step = 0
    sdir = None
    saved_plan = None
    if cfgj.get("resume", "auto") == "auto":
        # the newest hash-valid state among state / state.old / state.tmp (save_state swaps them by renames)
        best = None
        for cand in ("state", "state.old", "state.tmp"):
            d = os.path.join(out, cand)
            try:
                meta = json.load(open(os.path.join(d, "state.json")))
                _rt = (meta.get("plan") or {}).get("rate_term")
                need = ["model.tch", "opt.pt"] + (["rate.json"] if _rt and int(meta["epoch"]) > int(_rt.get("start_epoch", 0)) else [])
                ok = all(hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() == meta["sha256_" + f]
                         for f in need)   # KeyError (missing hash) -> except -> candidate rejected (v5 msg 978)
            except Exception:  # noqa: BLE001
                continue
            if ok and (best is None or meta["step"] > best[1]["step"]):
                best = (d, meta)
        remnants = [c for c in ("log.jsonl", "milestones", "state", "state.old", "state.tmp") if os.path.exists(os.path.join(out, c))]
        if best is None and remnants:
            raise SystemExit("refusing a cold start: %s has %s but no hash-valid state (restore one, or set resume none)" % (out, remnants))
        if best is not None:
            sdir, meta = best
            saved_plan = dict(meta.get("plan") or {})
            cur = json.loads(json.dumps(sig))
            changed = sorted(k for k in set(meta["plan"]) | set(cur) if meta["plan"].get(k) != cur.get(k))
            allowed = set(cfgj.get("plan_transition", []))
            if changed and set(changed) <= allowed:
                # explicit, recorded plan transition (e.g. enabling C1 at a resume, v5 msg 972)
                if main_p:
                    print("PLAN TRANSITION at step %d: %s" % (int(meta["step"]), changed), flush=True)
                meta["plan"] = cur
                changed = []
            # key-wise, missing == None, so states written before a key existed still resume when nothing changed
            if changed:
                ok_stretch = old_sig is not None and meta["plan"] == json.loads(json.dumps(old_sig)) \
                    and int(meta["step"]) == int(cfgj["stretch"]["at_step"])
                assert ok_stretch, ("plan changed", meta["plan"], sig)
            raw.load_state_dict(torch.load(os.path.join(sdir, "model.tch"), map_location=device, weights_only=True), strict=True)
            o = torch.load(os.path.join(sdir, "opt.pt"), map_location=device, weights_only=False)
            opts.second_order.load_state_dict(o["second_order"])
            opts.adamw.load_state_dict(o["adamw"])
            step = int(meta["step"])
            print("EK2-OPTRANG rank %d: optimizer state loaded (%d + %d states, step %d)" % (rank, len(opts.second_order.state),
                  len(opts.adamw.state), step), flush=True)   # 2.0: every rank loads it
            if main_p:
                print("RESUMED from %s: epoch %d step %d" % (sdir, meta["epoch"], step), flush=True)

    result_path = os.path.join(out, "result.json")
    result = {"name": cfgj["name"], "config": cfgj, "params": nparams, "world": world, "evals": [], "arch": arch_desc}
    if step and os.path.isfile(result_path):
        result["evals"] = json.load(open(result_path)).get("evals", [])
    if main_p:
        print(json.dumps({"name": cfgj["name"], "params": nparams, "world": world, "start_step": step,
                          "total_batches": data._total_batches}), flush=True)

    model = DDP(raw)
    if phase == "qat" and step == 0 and sdir is None:
        # cold QAT start: the upstream scale initialisation (1 weight-scale step, then scale_steps activation-scale
        # batches, under inference_mode; all-reduced across ranks), exactly as pysrc.train.train does it
        import types as _types
        import pysrc.train as _ptrain
        ss = int(cfgj.get("scale_steps", tftcfg.IVANOV_SCHEDULES["qat"].initialize_scales_steps))
        t0s = time.time()
        _ptrain.initialize_scales(raw_model=raw, model=model, data=data, cfg=_types.SimpleNamespace(
            initialize_scales_steps=ss, initialize_weight_scales_first=True, window_size_schedule=None))
        if main_p:
            print("QAT scale init: %d activation-scale batches in %.0f s" % (ss, time.time() - t0s), flush=True)
            torch.save(raw.state_dict(), os.path.join(out, "init_q0_model.tch"))   # rx7r: immutable post-init copy (not a resumable state)
            print("rx7r: post-init model saved to init_q0_model.tch", flush=True)
    if phase == "qat" and cfgj.get("fakequant"):
        # v10's fused Triton fake-quant (bit-identical to QuantizeFunction / rx3.PerRowQuantize), installed after
        # install_mixed and after scale init, as v10 specifies (msg 914)
        import fakequant
        fakequant.install(raw, verbose=main_p)
        if cfgj.get("qlinear"):
            import qlinear   # v10: fake-quant fused into Triton GEMMs; NOT bitwise (dW at bf16 rounding), needs a paired check
            qlinear.install(raw)
    # QAT compile is OPT-IN: plain inductor drops the bf16 rounding of the scales (+0.18% loss, v10 msg 914);
    rate_mod, rate_info = None, {}
    if cfgj.get("rate_term"):
        # C1 rate term (v5's module, interface agreed in msg 956): setup once; add_grad ACCUMULATES into p.grad on
        # every optimizer step after the micro-batch backward and before opts.step(); identical on every rank
        import importlib
        rt = cfgj["rate_term"]
        rate_mod = importlib.import_module(rt["module"])
        global RATE_MOD
        RATE_MOD = rate_mod
        if rt.get("c1fast"):
            # v10 fused C1 value+grad (msg 1103/1110), NOT bit-identical to the helper. rate_term.c1fast pins the
            # c1fast.py sha256 prefix; rate_term is in the plan signature, so switching backend at a resume is a
            # recorded plan_transition (v5 msg 1107), and a different file than the pinned one is refused.
            import c1fast
            c1fast.install(rate_mod)   # explicit: the sys.modules scan also matches torch.ops/torch.classes
            backend = str(getattr(rate_mod.PooledSoftRate, "C1_BACKEND", ""))
            assert backend.startswith("c1fast:") and backend[7:].startswith(str(rt["c1fast"])), \
                "c1fast backend %r does not match pinned %r" % (backend, rt["c1fast"])
            if main_p:
                print("rate_term: C1 backend %s" % backend, flush=True)
        rj =os.path.join(sdir, "rate.json") if sdir else None
        if rj and os.path.exists(rj):
            rate_mod.load_state_dict(raw, json.load(open(rj)), cfg=cfgj)
            if main_p:
                print("rate_term: restored C1 state from %s" % rj, flush=True)
        else:
            c1_start = int(cfgj["rate_term"].get("start_epoch", 0))
            ep_now = int(json.load(open(os.path.join(sdir, "state.json")))["epoch"]) if sdir else 0
            if c1_start > 0 and ep_now <= c1_start:   # v3 2024 note 2: a q==start state has no rate.json yet
                RATE_MOD = None            # rx7r: deferred; set up at the first step of stage epoch c1_start
                if main_p:
                    print("rate_term: C1 deferred to stage epoch %d (now epoch %d): zero contribution until then" % (c1_start, ep_now), flush=True)
            else:
                first_enable = step > 0 and saved_plan is not None and not saved_plan.get("rate_term") \
                    and "rate_term" in cfgj.get("plan_transition", [])
                assert step == 0 or first_enable, \
                    "resuming a C1 run without rate.json would change the objective (first enable needs plan_transition)"
                rate_mod.setup(raw, cfgj)
    # WARNING (v10 msg 917): compiling the REFERENCE fake-quant changes the forward loss by ~1% even with
    # emulate_precision_casts + division_rounding -> do NOT use compile_qat without 'fakequant' and a 200-step parity
    # test. Recommended A1 path: fakequant true + overrides gradient_checkpointing false + eager (bit-identical, ~4x).
    if (phase == "bf16" and cfgj.get("compile", True)) or (phase == "qat" and cfgj.get("compile_qat", False)):
        if phase == "qat":
            # v10 msg 918: inductor removes the bf16 rounding of the scale in the REFERENCE quantizer (70.6% of weight
            # elements differ; no inductor flag fixes it) -> compiled QAT only with v10's opaque fused kernels
            assert cfgj.get("fakequant"), "compile_qat requires fakequant: compiling the reference quantizer trains with fp32 scales"
            import torch._inductor.config as _ic
            _ic.emulate_precision_casts = True
            try:   # v5 msg 916: fp32 div must be div_rn like eager (x/s in fake-quant), a separate inductor switch
                _ic.eager_numerics.division_rounding = True
            except AttributeError:
                pass
        model = torch.compile(model)
    model.train()
    log_path = os.path.join(out, "log.jsonl" if step == 0 else "log.s%d.jsonl" % step)
    log_every = int(cfgj.get("log_every", 16))
    win_loss = torch.zeros((), dtype=torch.float64, device=device)   # summed over the window, this rank
    win_tok = torch.zeros((), dtype=torch.float64, device=device)
    win_t = time.time()
    win_steps = 0

    def flush_log(epoch_f, lr_mult):
        # one collective + one sync per window instead of two syncs + a collective per step (logging only)
        nonlocal win_loss, win_tok, win_t, win_steps
        v = torch.stack([win_loss, win_tok])
        dist.all_reduce(v)
        if not bool(torch.isfinite(v).all()):
            # fail loudly (nonzero exit; run_arm.sh retries from the last good state, then gives up) instead of
            # training on through NaN/inf -- QAT on lifted-but-tiny rows or a bad scale init can do that
            raise RuntimeError("non-finite training loss in the window ending at step %d" % step)
        if main_p:
            dt = time.time() - win_t
            with open(log_path, "a") as f:
                f.write(json.dumps({"epoch": epoch_f, "step": step, "train/loss": (v[0] / v[1]).item(),
                                    "steps": win_steps, "steps_per_s": win_steps / max(dt, 1e-9),
                                    "optim/lr_multiplier": lr_mult, **{"rate/" + k: v for k, v in rate_info.items()}}) + "\n")
        win_loss.zero_(); win_tok.zero_(); win_t = time.time(); win_steps = 0

    done_epochs, acc = 0, 0
    for per in data._batches_per_epoch:
        if acc + per > step:
            break
        acc += per
        done_epochs += 1
    assert acc == step, "state not at an epoch boundary (step %d, boundary %d)" % (step, acc)
    eval_every = int(cfgj.get("eval_every", 100))
    state_every = int(cfgj.get("state_every", 5))
    keep_every = int(cfgj.get("keep_every", 100))
    stop_at = cfgj.get("stop_at")
    clamp_list = []
    if cfgj.get("clamp_to_grid"):
        params, mods = dict(raw.named_parameters()), dict(raw.named_modules())
        for qn, qm in mods.items():
            if not qn.endswith(".quantize_weight") or getattr(qm, "qrow", None) is None:
                continue
            w = params.get(qn.removesuffix(".quantize_weight") + ".weight")
            if w is None or w.dim() != 2 or qm.scale.numel() != w.shape[0]:
                continue
            clamp_list.append((w, qm.scale, qm.qrow))
        if main_p:
            print("clamp_to_grid: %d quantized weight matrices clamped to the STE edge after every step" % len(clamp_list), flush=True)
    keep_epochs = tuple(int(e) for e in cfgj.get("keep_epochs", []))
    c1_start = int(cfgj["rate_term"].get("start_epoch", 0)) if cfgj.get("rate_term") else 0
    c1_step = sum(data._batches_per_epoch[:c1_start])   # v3 2024 note 1: set C1 up on the exact epoch-boundary weights
    c1_apply_epoch = int(cfgj["rate_term"].get("apply_from_epoch", 0)) if cfgj.get("rate_term") else 0
    c1_apply_step = sum(data._batches_per_epoch[:c1_apply_epoch])
    if main_p and cfgj.get("rate_term"):
        print("rate_term: C1 gradient applied from update index %d (stage epoch %d boundary; setup at start)" % (c1_apply_step, c1_apply_epoch), flush=True)
    probe_k = int((cfgj.get("probe") or {}).get("batches", 0))
    if cfgj.get("init_only"):
        if main_p:
            pl = probe_loss(raw, data, probe_k, device) if probe_k else float("nan")
            save_state(out, raw, opts, 0, 0, sig, 0)
            print("INIT_ONLY: shared post-init state saved at step 0; PROBE %.10f k %d" % (pl, probe_k), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    if cfgj.get("probe_only"):
        pl = probe_loss(raw, data, probe_k, device)
        if main_p:
            d = os.path.join(out, "reload"); os.makedirs(d, exist_ok=True)
            torch.save(raw.state_dict(), os.path.join(d, "model.tch"))
            torch.save({"second_order": opts.second_order.state_dict(), "adamw": opts.adamw.state_dict()}, os.path.join(d, "opt.pt"))
            print("PROBE %.10f k %d step %d (probe_only, after reload)" % (pl, probe_k, step), flush=True)
        if cfgj.get("probe_full_eval"):
            r = rx3.full_eval(raw, data, rank, world, device)
            if main_p:
                print("EVAL", json.dumps(dict(r, epoch=int(step and done_epochs), step=step, tag="probe_full_eval")), flush=True)
        dist.destroy_process_group()
        return
    A = data.gradient_accumulation_steps
    prev_epoch = done_epochs
    tstart = time.time()
    n_launch = 0   # 2.0: updates in this launch (rank check)

    # v10 opt-in debug switch 'nan_probe' (default off, not in the plan signature): keeps a pre-step copy of the
    # model state, checks every step's micro-batch losses + (DDP-reduced) grads for non-finite values with ONE sync,
    # and on the first hit saves state + this rank's micro-batches to out/nan_dump.r<rank>.pt and raises. Replay
    # with nan_replay.py (toggles each kernel switch on the dumped step). Grads are all-reduced, so a non-finite
    # grad shows on every rank in the same step (no rank waits on a collective).
    nan_probe = cfgj.get("nan_probe")
    nan_snap = None
    for batch in prefetch(batches(data, step)):
        lrs = opts.update_learning_rate_schedule(fraction_done=batch.fraction_done)
        if nan_probe:
            sd_now = raw.state_dict()
            if nan_snap is None:
                nan_snap = {k: v.detach().clone() for k, v in sd_now.items()}
            else:
                torch._foreach_copy_([nan_snap[k] for k in sd_now], list(sd_now.values()))
            mb_losses = []
        for mb in batch.micro_batches:
            loss = model(input_tokens=mb.input_tokens, priors=mb.priors, loss_mask=mb.loss_mask,
                         output_tokens=mb.output_tokens, cumulative_sequence_lengths=mb.cumulative_sequence_lengths,
                         override_window_size=None)
            win_loss += loss.detach().double()
            win_tok += mb.loss_mask.sum().double()
            (loss / A).backward()
            if nan_probe:
                mb_losses.append(loss.detach().float())
        if nan_probe:
            gl = [p.grad for p in raw.parameters() if p.grad is not None]
            ok_t = torch.stack([torch.isfinite(torch.stack(mb_losses)).all()] + [torch.isfinite(g).all() for g in gl]).all()
            if not bool(ok_t):
                bad_g = [n for n, p in raw.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                info = ([float(x) for x in mb_losses], bad_g[:40], len(bad_g))
                torch.save({"step": step, "rank": rank, "state": nan_snap, "mb_losses": info,
                            "micro_batches": [{k: getattr(mb, k) for k in ("input_tokens", "priors", "loss_mask",
                                                                           "output_tokens", "cumulative_sequence_lengths")}
                                              for mb in batch.micro_batches]},
                           os.path.join(out, "nan_dump.r%d.pt" % rank))
                print("NAN_PROBE rank %d step %d:" % (rank, step), info, flush=True)
                raise RuntimeError("nan_probe hit at step %d" % step)
        if rate_mod is not None and RATE_MOD is None and step >= c1_step:
            rate_mod.setup(raw, cfgj)      # rx7r: deferred C1 setup on the weights at the start of stage epoch c1_start
            RATE_MOD = rate_mod
            if main_p:
                print("rate_term: C1 ENABLED at step %d (stage epoch %d), lambda %g tau %s" % (step, int(batch.epoch),
                      float(cfgj["rate_term"]["lambda"]), cfgj["rate_term"].get("tau")), flush=True)
                print("rate_term: C1 setup on the weights after %d updates = stage epoch %d boundary" % (step, c1_start), flush=True)
        if rate_mod is None or RATE_MOD is None:
            rate_info = {"c1_on": 0}
        elif step < c1_apply_step:
            rate_info = {"c1_on": 0, "c1_set_up": 1}
        elif step % int(cfgj["rate_term"].get("every", 1)) == 0:
            if step == c1_apply_step and c1_apply_step > 0 and main_p:
                print("rate_term: C1 gradient ACTIVATED at update index %d" % step, flush=True)
            # task grads here are SUM-CE / A, then DDP-mean over ranks, i.e. N_scored_total / (A * world) times the
            # gradient of mean nats/token. A mean-nats lambda must be scaled by that common factor (v5 msg 959):
            # one scalar all-reduce of this step's scored-token count, identical on every rank.
            n_sc = torch.stack([mb.loss_mask.sum() for mb in batch.micro_batches]).sum().double()
            dist.all_reduce(n_sc)
            grad_normalizer = float(n_sc.item()) / (A * world)
            rate_info = rate_mod.add_grad(raw, float(cfgj["rate_term"]["lambda"]), step,
                                          grad_normalizer=grad_normalizer) or {}
        opts.step()
        if clamp_list:
            with torch.no_grad():
                for w, sc, qr in clamp_list:
                    lim = (qr.reshape(-1).float() * sc.detach().reshape(-1).to(torch.bfloat16).float().abs() * (1 - 1e-6))[:, None].to(w.dtype)
                    w.copy_(torch.maximum(torch.minimum(w, lim), -lim))
        opts.zero_grad()
        step += 1
        win_steps += 1
        n_launch += 1
        rank_check(raw, n_launch, step, rank, world, device)   # 2.0
        if cfgj.get("stop_step") and step >= int(cfgj["stop_step"]):
            # parity / smoke runs: stop after exactly N optimizer steps, no state save, no eval
            flush_log(batch.epoch, lrs.multiplier)
            if main_p:
                print("STOP_STEP reached at step %d" % step, flush=True)
                result.update(stop_step=step, done=time.time())
                json.dump(result, open(result_path, "w"), indent=1)
            dist.destroy_process_group()
            return
        rolled = int(batch.epoch) != prev_epoch or batch.is_last
        if win_steps >= log_every or rolled:
            flush_log(batch.epoch, lrs.multiplier)

        ep = int(batch.epoch)   # the last batch of epoch e carries epoch == e + 1 exactly
        if ep == prev_epoch and not batch.is_last:
            continue
        prev_epoch = ep
        if main_p:
            el = time.time() - tstart
            print("EPOCH %d step %d lr_mult %.5f  %.1f s since start" % (ep, step, lrs.multiplier, el), flush=True)
        if ((eval_every and ep % eval_every == 0) or ep in cfgj.get("eval_epochs", [])) and not batch.is_last:
            r = rx3.full_eval(raw, data, rank, world, device)
            r.update(epoch=ep, step=step, time=time.time())
            result["evals"].append(r)
            if main_p:
                print("EVAL", json.dumps(r), flush=True)
                json.dump(result, open(result_path, "w"), indent=1)
        stop = os.path.exists(os.path.join(out, "STOP")) or (stop_at is not None and ep >= int(stop_at))
        if stop and probe_k:
            pl = probe_loss(raw, data, probe_k, device)
            if main_p:
                print("PROBE %.10f k %d step %d (before save)" % (pl, probe_k, step), flush=True)
        if main_p and ((state_every and ep % state_every == 0) or stop or batch.is_last):
            save_state(out, raw, opts, step, ep, sig, keep_every, keep_epochs)
        dist.barrier()
        if stop and not batch.is_last:
            if main_p:
                print("STOP at epoch %d (state saved)" % ep, flush=True)
            dist.destroy_process_group()
            return

    if step != data._total_batches:   # 2.0: an epoch ended short (count against the plan)
        raise RuntimeError("loop ended at step %d of %d batches" % (step, data._total_batches))
    if main_p:
        torch.save(raw.state_dict(), os.path.join(out, "model.tch"))
        result["train_s_last_section"] = time.time() - tstart
    r = rx3.full_eval(raw, data, rank, world, device)
    r.update(epoch=epochs, tag="final", time=time.time())
    result["evals"].append(r)
    if main_p:
        print("EVAL", json.dumps(r), flush=True)
        result["done"] = time.time()
        json.dump(result, open(result_path, "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)   # 2.0: no hang or abort in the interpreter teardown
