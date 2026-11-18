# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/export/export_v3_ar.py, sha256 32ce0166ccb52925), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""export_v3_ar.py <config.json> <checkpoint.tch> <out_dir>   (v7, HUT-EXP-AR)

v3's export_v3.py (work/v3, sha in v3_export_py.tar 08089bc2) extended for the recommended architecture:
  - AttnRes replace-B2 (cfg "arch": {"attnres": {"mode": "replace", "block": 2, ...}}): arch.install() is run on the
    model BEFORE the strict load, so the 33 depth queries blocks.{l}.q_mix.w / q_mlp.w (l = 0..15) and
    blocks.15.q_final.w are real parameters of the exported model.
  - per-token PPMd prior head (cfg overrides.prior_logit_mixing = true): logit_mixing_head.weight [1, 192], no bias.
Both are unquantized f32 tensors; they are allowlisted as raw f32 (export step 0) and rounded to bf16 values by
export step 1 (wpack2_bf16 classes "attnres" and "head"). FORM is signalled by tensor presence (the coder's loader
convention): q_*.w present and the skip / residual / token-embedding coefficient tensors absent => replace-mode
AttnRes; logit_mixing_head.weight present => prior head. Without "arch" and prior_logit_mixing the export is
byte-identical to export_v3.py (same code path; the extra suffixes are never matched).
Run with the env of exp.env (package paths) and env.sh, on one GPU.
"""
import dataclasses
import json
import os
import sys

# training/beehive/tree: rx3.py (021e3965, sets up the trainer tree from TFT_TREE) and arch.py (AttnRes install).
# (Development hosts loaded the same rx3 and an earlier arch.py that differs only in the unused "hub" mode; the
# packaged path was re-verified to reproduce the shipped container byte for byte, see training/README.md section 12.5.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tree"))
import numpy as np
import torch

import rx3
from tft import config as tftcfg
from tft import export_weights_poc as ewp

AR_SUFFIXES = (".q_mix.w", ".q_mlp.w", ".q_final.w")
HEAD_SUFFIX = "logit_mixing_head.weight"


def main():
    cfgj = json.load(open(sys.argv[1]))
    ckpt, out_dir = sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(os.environ.get("EXPORT_DEVICE", "cuda"))
    raw, cfg = rx3.build_model(cfgj, "fp32", dev, True)
    if cfgj.get("arch"):
        import arch
        a = dict(cfgj["arch"].get("attnres", {}))
        assert a.get("mode", "replace") == "replace", "exporter supports replace-mode AttnRes only (gated: not this model's form)"
        assert int(a.get("block", 2)) == 2, "exporter/engine contract is Block-2"
        arch.install(raw, cfg, cfgj["arch"], True)
    if cfgj.get("mixed"):
        rx3.install_mixed(raw, cfgj["mixed"], True)
    raw.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True), strict=True)
    raw.eval()
    sd_keys = list(raw.state_dict())
    n_ar = sum(1 for k in sd_keys if k.startswith("blocks.") and k.endswith(AR_SUFFIXES))
    has_head = HEAD_SUFFIX in sd_keys
    if cfgj.get("arch"):
        assert n_ar == 2 * cfg.n_layers + 1, n_ar
        for k in sd_keys:   # replace mode: the static skips / re-injection must be gone
            assert not k.endswith(("token_embedding_coefficient.value", "residual_stream_coefficient.value")), k
            assert k != "skip_connection_weights.value", k
    else:
        assert n_ar == 0
    assert has_head == bool(cfg.prior_logit_mixing), (has_head, cfg.prior_logit_mixing)
    # the VE wrapper adds an ".inner" level; the exporter only needs state_dict + modules
    g0 = tftcfg.geometry(os.environ.get("TFT_GEOMETRY", "poc8m_w2048"))
    # v3 Oct 16: carry the trained attention window too (window-knob arms, admin 1733 runtime gate); unchanged for w2048
    g = dataclasses.replace(g0, name=g0.name + "_v3x", d_mlp=cfg.d_mlp, n_key_value_heads=cfg.n_key_value_heads,
                            window_size=cfg.window_size)
    tftcfg.GEOMETRIES[g.name] = g
    ewp._p11_geom = lambda g=g: g   # sidecar 'fenster' = the model's window, not the env geometry's
    real_expected = ewp.expected_counts

    def counts(_g, _paket=None):
        sd = raw.state_dict()
        q = {k for k in sd if k.endswith(".weight") and k[:-7] + ".quantize_weight.scale" in sd}
        act = sum(1 for k in sd if k.endswith(ewp._ACTIVATION_SCALE_SUFFIXES))
        other = sum(1 for k in sd if k not in q and not k.endswith(".quantize_weight.scale")
                    and not k.endswith(ewp._ACTIVATION_SCALE_SUFFIXES))
        return {"n_quantized": len(q), "n_activation_scales": act, "n_raw_fp32": other}
    ewp.expected_counts = counts
    extra = ["value_embedding_coefficients.value"] + list(AR_SUFFIXES) + [HEAD_SUFFIX]
    ewp._UNQUANTIZED_FP32_SUFFIXES = tuple(ewp._UNQUANTIZED_FP32_SUFFIXES) + tuple(
        s for s in extra if s not in ewp._UNQUANTIZED_FP32_SUFFIXES)
    if cfgj.get("mixed"):
        # v3 Oct 16 (fix): per-row mixed ranges, ported verbatim from v3 export_v3.py 82e5156d (the exporter of every
        # earlier finish arm). The previous pass-through ignored each row's qmax (3/7/15), so a mixed model failed
        # export_weights' int check, and the qrow buffers stayed in the state dict as raw fp32 tensors.
        qrow_by_ptr = {}
        mods_by_name = dict(raw.named_modules())
        for name, mod in mods_by_name.items():
            if "qrow" in getattr(mod, "_buffers", {}):
                q = mod._buffers.pop("qrow")   # keep it out of the state dict the exporter walks
                mod.qrow = q
            if hasattr(mod, "qrow") and os.environ.get("EXPORT_CLAMP_SPARSE") == "1" and (
                    ".mlp.down." in name or name.startswith("prior_embedding.")):
                mod.qrow = torch.clamp(mod.qrow, max=7.0)   # sparse int4 column arenas hold |q| <= 7 only
            if hasattr(mod, "qrow") and name.endswith(".quantize_weight"):
                owner = mods_by_name[name[: -len(".quantize_weight")]]
                qrow_by_ptr[owner.weight.data_ptr()] = mod.qrow.detach().float()

        def qrows(value, scale_fp32):
            scale = scale_fp32.to(torch.bfloat16).to(torch.float32)
            scaled = value.float() / scale.unsqueeze(-1)
            q = qrow_by_ptr.get(value.data_ptr())
            if q is None:
                return scaled.round().clamp(-7, 7).to(torch.int8), scale
            q = q.to(scaled.device).unsqueeze(-1)
            return torch.maximum(torch.minimum(scaled.round(), q), -q).to(torch.int8), scale
        ewp.quantize_weight_rows = qrows
        import inspect
        src = inspect.getsource(ewp.export_weights).replace(
            "assert int(ints.abs().max().item()) <= 7", "assert int(ints.abs().max().item()) <= 31")
        exec(compile(src, ewp.__file__, "exec"), ewp.__dict__)
        n_wide = sum(int((q > 7).sum()) for q in qrow_by_ptr.values())
        print("mixed export: %d quantizers with row ranges, %d rows of qmax 15" % (len(qrow_by_ptr), n_wide), flush=True)
    out = os.path.join(out_dir, "weights-" + g0.name + ".bin")
    paket = ewp.paket_from_env()
    result = ewp.export_weights(raw, out, g.name, paket)
    ewp.write_sidecar(out, result)
    print("exported", out, json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))})[:400])
    print("v7 AR/head: %d AttnRes query tensors, prior head %s" % (n_ar, has_head))
    ewp.expected_counts = real_expected


if __name__ == "__main__":
    main()
