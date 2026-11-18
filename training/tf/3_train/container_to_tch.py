#!/usr/bin/env python3
"""container_to_tch.py: a finished beehive container (v4 + ENC_INT5M, or v3/v4/v5) back into the trainer.

    TFT_TREE=<trainer tree> python3 container_to_tch.py --container <c4> --cfg <cfg.json> \
        --strict <out_fp32.tch> --warm <out_bf16.tch> [--wiederbelebe] [--report <json>]

cfg.json carries "overrides", "arch", "ve_sentinel" as the beehive configs (training/beehive/cfg/*.json).

(a) --strict: state dict of the fp32-phase model (the numerics of the strict deployed eval): every quantized weight
    = q * bf16(scale) row by row, its .quantize_weight.scale = bf16(scale), the activation scales and the raw f32
    tensors as stored.  Dead MLP neurons removed by prune_dead_neurons.py are re-inserted as zero rows of mlp.up
    (scale = median of the live rows) and zero columns of mlp.down; the function is unchanged (relu(0)^2 = 0).
    Check: the model's own quantizer gives back exactly the container ints of every tensor (padded rows = 0).
(b) --warm: state dict of the bf16-phase model (no quantizer scales) with the same dequantized weights, for a bf16
    warm start (rx7q 'init').  --wiederbelebe: the re-inserted dead neurons get a fresh mlp.up row (normal, the std of
    the live rows of that layer) and keep a zero mlp.down column, so the function at the start is unchanged and the
    neuron can learn again (wiederbelebe = revive).
Does not need a GPU.  The container decoder is ../export/export4/weights_compress_v3.py (formats v3 to v5 plus ENC_INT5M).
The inverse of training/beehive/export/export_v3_ar.py and export4 of cmix-neif-beehive
(Vladimir Malinovskii, 2026, 92c07c0b, GPL-3).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tree"))
sys.path.insert(0, os.path.join(HERE, "export", "export4"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rx3a  # noqa: E402  (puts TFT_TREE on sys.path)
import arch  # noqa: E402
import weights_compress_v3 as wc  # noqa: E402
from pysrc.export_weights import _ACTIVATION_SCALE_SUFFIXES  # noqa: E402

MLP_UP = re.compile(r"^blocks\.(\d+)\.mlp\.up\.weight$")
MLP_DOWN = re.compile(r"^blocks\.(\d+)\.mlp\.down\.weight$")


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def build(cfgj, phase):
    raw, cfg = rx3a.build_model(cfgj, phase, torch.device("cpu"), False)   # installs arch when cfgj["arch"]
    return raw, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--strict", required=True)
    ap.add_argument("--warm", default="")
    ap.add_argument("--wiederbelebe", action="store_true")
    ap.add_argument("--sentinel-null", action="store_true",
                    help="warm state for the new text form : sentinel rows 208..215 of "
                         "embedding.weight and of both slots of embedding.last_table (208..215, 425..432) set to 0")
    ap.add_argument("--report", default="")
    ap.add_argument("--qrow", default="", help="npz {quantize_weight module: qmax per row} from the container rows (1/3/7 by max|q|)")
    ap.add_argument("--seed", type=int, default=20261023)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cfgj = json.load(open(a.cfg))
    cont = wc.read_tensor_file_v3(a.container)
    rep = {"container": a.container, "n_container_tensors": len(cont)}

    raw, cfg = build(cfgj, "fp32")
    sd = raw.state_dict()
    used = set()
    new = {}
    pruned = {}
    zero_scale_rows = {}
    qrows = {}
    for key, ref in sd.items():
        if key.endswith(".qrow"):
            continue
        if key.endswith(".weight") and key[:-7] + ".quantize_weight.scale" in sd:
            q = cont[key + ".q"][1].astype(np.int32)
            s = bf16_to_f32(cont[key + ".scale"][1])
            used.update((key + ".q", key + ".scale"))
            shape = tuple(ref.shape)
            m_up, m_dn = MLP_UP.match(key), MLP_DOWN.match(key)
            if q.shape != shape:
                if m_up and q.shape[1:] == shape[1:] and q.shape[0] < shape[0]:
                    n = shape[0] - q.shape[0]
                    pruned[int(m_up.group(1))] = n
                    q = np.concatenate([q, np.zeros((n,) + shape[1:], np.int32)], 0)
                    s = np.concatenate([s, np.full(n, np.float32(np.median(s)), np.float32)])
                elif m_dn and q.shape[0] == shape[0] and q.shape[1] < shape[1]:
                    n = shape[1] - q.shape[1]
                    q = np.concatenate([q, np.zeros((shape[0], n), np.int32)], 1)
                else:
                    raise SystemExit("shape %s: container %s, model %s" % (key, q.shape, shape))
            # rows the exporter zeroed (scale and ints 0, rule 'rows below 2^-20 zeroed'): the engine reads q = 0, the
            # trainer's quantizer would compute 0/0 -> a tiny positive scale (1e-30, the lifting value of
            # fix_subnormal_rows.py) keeps w = 0 -> q = 0 -> output 0, exactly the deployed row
            z = s == 0
            if z.any():
                assert not q[z].any(), key
                s = s.copy()
                s[z] = np.float32(1e-30)
                zero_scale_rows[key] = int(z.sum())
            w = q.astype(np.float32) * s.reshape((-1,) + (1,) * (q.ndim - 1))
            mq = np.abs(q.reshape(q.shape[0], -1)).max(1)
            qrows[key[:-7] + ".quantize_weight"] = np.where(mq <= 1, 1, np.where(mq <= 3, 3, 7)).astype(np.float32) if not MLP_DOWN.match(key) else np.full(q.shape[0], 7, np.float32)
            new[key] = torch.from_numpy(np.ascontiguousarray(w)).to(ref.dtype)
            sk = key[:-7] + ".quantize_weight.scale"
            new[sk] = torch.from_numpy(s.copy()).reshape(sd[sk].shape).to(sd[sk].dtype)
        elif key.endswith(".quantize_weight.scale"):
            continue   # set together with its weight
        elif key.endswith(_ACTIVATION_SCALE_SUFFIXES):
            v = bf16_to_f32(cont[key][1])
            used.add(key)
            new[key] = torch.from_numpy(v.copy()).reshape(ref.shape).to(ref.dtype)
        else:
            if key not in cont:
                raise SystemExit("model tensor %s not in the container" % key)
            v = cont[key][1]
            used.add(key)
            assert tuple(v.shape) == tuple(ref.shape), (key, v.shape, ref.shape)
            new[key] = torch.from_numpy(np.ascontiguousarray(v)).to(ref.dtype)
    unused = sorted(k for k in cont if k not in used)
    rep["unused_container_tensors"] = unused
    assert all(k.startswith(("rope.", "config.")) for k in unused), unused
    rep["pruned_neurons_reinserted"] = pruned
    rep["zero_scale_rows"] = {"tensors": len(zero_scale_rows), "rows": sum(zero_scale_rows.values())}
    missing, unexpected = raw.load_state_dict(new, strict=False)
    missing = [k for k in missing if not k.endswith(".qrow")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])

    # check: the model's quantizers give back the container ints
    mods = dict(raw.named_modules())
    n_chk = n_bad = 0
    with torch.no_grad():
        for key in new:
            if not (key.endswith(".weight") and key[:-7] + ".quantize_weight.scale" in new):
                continue
            qm = mods[key[:-7]].quantize_weight
            ints = qm(mods[key[:-7]].weight, initialize_scales_steps=None, return_ints=True).cpu().numpy()
            q = cont[key + ".q"][1].astype(np.int64)
            if MLP_UP.match(key):
                ok = np.array_equal(ints[: q.shape[0]], q) and not ints[q.shape[0]:].any()
            elif MLP_DOWN.match(key):
                ok = np.array_equal(ints[:, : q.shape[1]], q) and not ints[:, q.shape[1]:].any()
            else:
                ok = np.array_equal(ints, q)
            n_chk += 1
            n_bad += 0 if ok else 1
            if not ok:
                qq = q if not (MLP_UP.match(key) or MLP_DOWN.match(key)) else None
                det = ""
                if qq is not None:
                    bad = np.flatnonzero((ints.reshape(ints.shape[0], -1) != qq.reshape(qq.shape[0], -1)).any(1))
                    sc = new[key[:-7] + ".quantize_weight.scale"].reshape(-1)
                    det = " rows %s, scale %s, q %s, ints %s" % (bad[:4].tolist(), [float(sc[i]) for i in bad[:2]],
                          [qq[i].reshape(-1)[:4].tolist() for i in bad[:1]], [ints[i].reshape(-1)[:4].tolist() for i in bad[:1]])
                print("INT MISMATCH", key, det, flush=True)
    rep["int_check"] = {"tensors": n_chk, "mismatch": n_bad}
    assert n_bad == 0
    sdo = raw.state_dict()
    if a.qrow:   # rows of the container as .qrow buffers too (export_v3_ar installs "mixed" before its strict load)
        for m, v in qrows.items():
            sdo[m + ".qrow"] = torch.from_numpy(v.copy())
    torch.save(sdo, a.strict)
    if a.qrow:
        np.savez(a.qrow, **qrows)
        rep["qrow"] = {"file": a.qrow, "modules": len(qrows), "rows_1_3_7": [int(sum((v == c).sum() for v in qrows.values())) for c in (1, 3, 7)]}
    print("strict (fp32 phase): %d tensors, %d quantized weights int-equal to the container, pruned re-inserted %s -> %s"
          % (len(new), n_chk, pruned, a.strict), flush=True)

    if a.warm:
        wraw, _ = build(cfgj, "bf16")
        wsd = wraw.state_dict()
        take = {k: new[k].to(wsd[k].dtype) for k in wsd if k in new}
        miss = [k for k in wsd if k not in take]
        assert not miss, miss[:10]
        if a.wiederbelebe:
            g = torch.Generator().manual_seed(a.seed)
            for layer, n in pruned.items():
                up = take["blocks.%d.mlp.up.weight" % layer]
                live = up[: up.shape[0] - n].float()
                std = float(live.std())
                up[up.shape[0] - n:] = (torch.randn((n, up.shape[1]), generator=g) * std).to(up.dtype)
            rep["revived"] = pruned
        if a.sentinel_null:
            V = int(cfg.vocabulary_size)
            e = take["embedding.weight"]; lt = take["embedding.last_table.weight"]
            slot = lt.shape[0] // 2
            assert e.shape[0] == V + 8 and slot == V + 9, (e.shape, lt.shape)
            e[V:V + 8] = 0
            lt[V:V + 8] = 0
            lt[slot + V:slot + V + 8] = 0
            rep["sentinel_null"] = {"embedding": [V, V + 8], "last_table": [[V, V + 8], [slot + V, slot + V + 8]]}
        wraw.load_state_dict(take, strict=True)
        torch.save(wraw.state_dict(), a.warm)
        rep["warm_tensors"] = len(take)
        print("warm (bf16 phase): %d tensors%s -> %s" % (len(take), ", dead neurons revived" if a.wiederbelebe else "", a.warm), flush=True)
    rep["n_params_fp32_model"] = int(sum(p.numel() for p in raw.parameters()))
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "unused_container_tensors"}), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
