# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/fisher_alloc_a.py, sha256 dc805e965080c667), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""Per-row Fisher sensitivity + rate-distortion level allocation for mixed precision.

    torchrun --nproc_per_node 1 fisher_alloc.py <config.json>

config: name, out, init (bf16 teacher), overrides, batches (number of optimizer batches for Fisher),
        levels (list of qmax, e.g. [3, 7, 15]), budgets (list of relative rate budgets vs uniform qmax=7, e.g. [1.0, 1.05, 1.1])
Writes: out/fisher.npz (per weight-matrix row Fisher-weighted distortions and rates per option),
        out/alloc_<budget>.npz ({quantizer module name: qmax per row}) and out/alloc_summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.distributed as dist

import rx3a as rx3
from pysrc.model import InitializeScales


def entropy_bits(sym: np.ndarray, axis=-1) -> np.ndarray:
    """empirical order-0 entropy (bits per symbol) along axis, symbols are small ints"""
    lo = sym.min()
    s = (sym - lo).astype(np.int64)
    k = int(s.max()) + 1
    out = np.zeros(sym.shape[0])
    for r in range(sym.shape[0]):
        c = np.bincount(s[r], minlength=k).astype(np.float64)
        p = c[c > 0] / c.sum()
        out[r] = -(p * np.log2(p)).sum()
    return out


def main():
    cfgj = json.load(open(sys.argv[1]))
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    out = cfgj["out"]
    os.makedirs(out, exist_ok=True)
    raw, cfg = rx3.build_model(cfgj, "bf16", device, rank == 0)
    if cfgj.get("dequant_from_qat"):
        # v3 (Oct 14, allocation refresh): init is a QAT checkpoint. Build the qat model, load it with its per-row
        # ranges (cfg "mixed"), replace every quantized weight by its fake-quantized (deployed, clipped) value, and
        # load those into the bf16 model, so sensitivities are measured at the deployed weights, not the latent ones.
        qm, _ = rx3.build_model(cfgj, "qat", device, False)
        rx3.load_init(qm, cfgj["init"], False, device, rank == 0)
        if cfgj.get("mixed"):
            rx3.install_mixed(qm, cfgj["mixed"], rank == 0)
        sd = {k: v for k, v in qm.state_dict().items()}
        nq = 0
        with torch.no_grad():
            for n, m in qm.named_modules():
                if isinstance(m, rx3.Quantize) and not m.batched and n.endswith("quantize_weight"):
                    wk = n[: -len("quantize_weight")] + "weight"
                    if wk in sd:
                        owner = dict(qm.named_modules())[n[: -len(".quantize_weight")]]
                        sd[wk] = m(owner.weight, initialize_scales_steps=None, return_ints=False).detach().clone()
                        nq += 1
        bsd = raw.state_dict()
        res = raw.load_state_dict({k: v.to(bsd[k].dtype) for k, v in sd.items() if k in bsd}, strict=False)
        if rank == 0:
            print("dequant_from_qat: %d weights replaced by their fake-quantized values; missing %d" % (nq, len(res.missing_keys)), flush=True)
        del qm
    else:
        rx3.load_init(raw, cfgj["init"], True, device, rank == 0)
    raw.train()

    # weight matrices that the QAT config quantizes per row (the names of their quantize_weight modules)
    qraw, _ = rx3.build_model(cfgj, "qat", torch.device("cpu"), False)
    qnames = [n for n, m in qraw.named_modules() if isinstance(m, rx3.Quantize) and not m.batched]
    del qraw
    params = dict(raw.named_parameters())
    targets = {}
    for qn in qnames:
        pn = qn[: -len("quantize_weight")] + "weight"
        if pn in params and params[pn].ndim == 2:
            targets[qn] = params[pn]
    if rank == 0:
        print("fisher targets: %d matrices, %d params" % (len(targets), sum(p.numel() for p in targets.values())), flush=True)

    fisher = {qn: torch.zeros_like(p, dtype=torch.float32) for qn, p in targets.items()}
    data = rx3.tftrun.make_data_loader(1, device, rank, world)
    nb = int(cfgj.get("batches", 64))
    none = InitializeScales(weight_steps=None, activation_steps=None)
    it = iter(data)
    ntok = 0.0
    for b in range(nb):
        batch = next(it)
        for mb in batch.micro_batches:
            raw.zero_grad(set_to_none=True)
            loss = raw(input_tokens=mb.input_tokens, priors=mb.priors, loss_mask=mb.loss_mask,
                       output_tokens=mb.output_tokens, cumulative_sequence_lengths=mb.cumulative_sequence_lengths,
                       override_window_size=None)
            loss.backward()
            ntok += float(mb.loss_mask.sum())
            with torch.no_grad():
                for qn, p in targets.items():
                    if p.grad is not None:
                        fisher[qn] += p.grad.float() ** 2
        if rank == 0 and b % 8 == 0:
            print("fisher batch %d/%d" % (b, nb), flush=True)
    if world > 1:
        for f in fisher.values():
            dist.all_reduce(f)

    if rank == 0:
        levels = [int(q) for q in cfgj.get("levels", [3, 7, 15])]
        rows_D, rows_R, rows_n, rows_key = [], [], [], []
        store = {}
        # class-pooled rate (the container codes every row of a class with one shared adaptive tree):
        # pooled symbol histogram per level over all rows, then each row's cost under it
        lv = [int(q) for q in cfgj.get("levels", [3, 7, 15])]
        pooled = {q: np.ones(2 * q + 1) for q in lv}
        for qn, p in targets.items():
            w = p.detach().float().cpu().numpy()
            amax = np.abs(w).max(1, keepdims=True); amax[amax == 0] = 1e-30
            for q in lv:
                sym = np.clip(np.round(w / (amax / q)), -q, q).astype(np.int64) + q
                pooled[q] += np.bincount(sym.ravel(), minlength=2 * q + 1)
        logp = {q: -np.log2(pooled[q] / pooled[q].sum()) for q in lv}
        for qn, p in targets.items():
            w = p.detach().float().cpu().numpy()
            F = fisher[qn].cpu().numpy()
            amax = np.abs(w).max(1, keepdims=True)
            amax[amax == 0] = 1e-30
            D = np.zeros((w.shape[0], len(levels)))
            R = np.zeros((w.shape[0], len(levels)))
            Rc = np.zeros((w.shape[0], len(levels)))
            for k, q in enumerate(levels):
                s = amax / q
                sym = np.clip(np.round(w / s), -q, q)
                err = (sym * s - w) ** 2
                D[:, k] = 0.5 * (F * err).sum(1)
                R[:, k] = entropy_bits(sym.astype(np.int64)) * w.shape[1]
                Rc[:, k] = logp[q][sym.astype(np.int64) + q].sum(1)
            store[qn + "|D"] = D
            store[qn + "|R"] = R
            store[qn + "|Rc"] = Rc
            if cfgj.get("rate") == "class":
                R = Rc
            rows_D.append(D); rows_R.append(R); rows_n.append(np.full(w.shape[0], w.shape[1])); rows_key += [qn] * w.shape[0]
        np.savez(os.path.join(out, "fisher.npz"), **store)
        D = np.concatenate(rows_D); R = np.concatenate(rows_R)
        base_k = levels.index(7)
        R0 = R[:, base_k].sum(); D0 = D[:, base_k].sum()
        summary = {"levels": levels, "rows": int(D.shape[0]), "R_uniform7_bits": float(R0), "D_uniform7": float(D0),
                   "ntok_fisher": ntok, "allocs": []}
        keys = np.array(rows_key)
        for budget in cfgj.get("budgets", [1.0, 1.05, 1.1]):
            target = R0 * float(budget)

            def pick(lam):
                return np.argmin(D + lam * R, axis=1)

            def rate(lam):
                return R[np.arange(len(R)), pick(lam)].sum()

            hi = 1.0
            while rate(hi) > target:
                hi *= 10
            lo = hi
            while rate(lo) <= target and lo > 1e-30:
                lo /= 10
            for _ in range(200):          # geometric bisection: rate(lo) > target >= rate(hi)
                mid = math.sqrt(lo * hi)
                if rate(mid) > target:
                    lo = mid
                else:
                    hi = mid
            k = pick(hi)
            alloc = {}
            for qn in targets:
                sel = keys == qn
                alloc[qn] = np.array([levels[i] for i in k[sel]], dtype=np.int16)
            np.savez(os.path.join(out, "alloc_%.3f.npz" % budget), **alloc)
            frac = [float((k == i).mean()) for i in range(len(levels))]
            Rs = float(R[np.arange(len(R)), k].sum()); Ds = float(D[np.arange(len(D)), k].sum())
            summary["allocs"].append({"budget": budget, "lambda": hi, "R_bits": Rs, "R_rel": Rs / R0, "D": Ds,
                                      "D_rel": Ds / D0, "frac_per_level": frac})
            print("budget %.3f: R %.4f x, D %.4f x, fractions %s" % (budget, Rs / R0, Ds / D0, frac), flush=True)
        json.dump(summary, open(os.path.join(out, "alloc_summary.json"), "w"), indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
