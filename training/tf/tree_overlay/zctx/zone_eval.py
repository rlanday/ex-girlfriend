#!/usr/bin/env python3
"""loss of a transformer checkpoint per ZONE, on the training package, one forward pass

Answers the question the training curve cannot answer: WHERE does an arm
gain or lose?  The zone map (`zonemap.py`) can do this only for the EXPORTED
transformer on a 10 MB excerpt, because it needs a dump file with all
probability rows.  Here a checkpoint is enough: the model runs over a fixed
excerpt of the package, the loss is kept per position and split by zone.

Two uses:
  * TITLE: does a transformer with a zone/structure signal (arm z4/z1) learn
    the starts of titles, where the plain transformer loses?
    -> run the same excerpt for ref and for the arm and compare the TITEL
       row.
  * gating arms: does the context gap spill over into the NEIGHBOURING zones?

The comparison is valid only between checkpoints of THE SAME epoch.

    python -m zctx.zone_eval <pkgdir> <checkpoint.tch> [n_positions] [arm]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    pkgdir, ckpt = sys.argv[1], sys.argv[2]
    n_want = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    arm = sys.argv[4] if len(sys.argv) > 4 else "none"

    from tft import config as tftcfg
    from pysrc.model import InitializeScales, Transformer

    from zctx import zgate, zonemap, zonemap2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = tftcfg.make_config("bf16", geometry_name=os.environ.get("TFT_GEOMETRY", "poc4m"),
                             variable_sequence_length=True)
    model = Transformer(cfg).to(device)
    if arm in ("z4", "z4c"):
        from zctx.zmodel import install_z4
        install_z4(model, cfg, x=int(os.environ.get("ZCTX_LAST_X", "4")),
                   d_slot=int(os.environ.get("ZCTX_LAST_DIM", "16")))
    elif arm == "g1":
        from zctx.zgate import N_LENGTH_CLASSES
        from zctx.zmodel import install_gate
        install_gate(model, cfg, N_LENGTH_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()

    tokens = np.fromfile(os.path.join(pkgdir, "ppmd-bytes.uint8"), dtype=np.uint8)
    bnd = np.fromfile(os.path.join(pkgdir, "article-boundaries.int32"),
                      dtype=np.int32).astype(np.int64)
    bitmap = zgate.load_bitmap(pkgdir)
    v = len(bitmap)
    priors = np.memmap(os.path.join(pkgdir, "ppmd-probs.float16"), dtype=np.float16,
                       mode="r", shape=(len(tokens), v))

    # one contiguous excerpt of whole articles, from article 0
    end_art = int(np.searchsorted(bnd, n_want))
    end_art = max(2, min(end_art, len(bnd) - 1))
    n = int(bnd[end_art])
    print("excerpt: articles 0..%d, %d tokens" % (end_art, n))

    tk = tokens[:n].astype(np.int64)
    marker, _, _ = zonemap.derive_text_marker(tk, bnd[: end_art + 1])
    cmark = zonemap2.derive_comment_marker(tk, bnd[: end_art + 1], bitmap)
    zone = zonemap2.classify(tk, bnd[: end_art + 1], bitmap, marker, cmark)
    ZN = zonemap2.NAMES

    chunk = tftcfg.MICRO_BATCH_TOKENS
    losses = np.full(n - 1, np.nan, dtype=np.float32)
    iz = InitializeScales(None, None)
    with torch.no_grad():
        for a in range(0, end_art):
            s, e = int(bnd[a]), int(bnd[a + 1])
            body = e - s - 1
            if body <= 0 or body > chunk:
                continue
            inp = torch.from_numpy(tokens[s : e - 1].astype(np.int64)).view(1, -1).to(device)
            out = torch.from_numpy(tokens[s + 1 : e].astype(np.int64)).view(1, -1).to(device)
            pr = torch.from_numpy(np.array(priors[s : e - 1])).view(1, body, v).to(device)
            cs = torch.tensor([[0, body]], dtype=torch.int32, device=device)
            act = model.compute_last_activation(
                input_tokens=inp, priors=pr.to(cfg.activation_dtype),
                cumulative_sequence_lengths=cs, override_window_size=None,
                initialize_scales=iz)
            logits = model.last_activation_to_logits(
                act, priors=pr.to(cfg.activation_dtype), initialize_scales=iz)
            l = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), out.view(-1), reduction="none")
            losses[s : s + body] = l.detach().cpu().numpy()

    # PPMd at the same positions, so that the map has the same column as the
    # dump version without needing a cmix run
    ppmd = np.full(n - 1, np.nan, dtype=np.float32)
    TINY = np.float32(6.0e-8)
    for a in range(0, n - 1, 500_000):
        b2 = min(a + 500_000, n - 1)
        rows = np.asarray(priors[a:b2]).astype(np.float32)
        tgt = tokens[a + 1 : b2 + 1].astype(np.int64)
        ppmd[a:b2] = -np.log(np.maximum(rows[np.arange(b2 - a), tgt], TINY))

    ok = ~np.isnan(losses)
    z = zone[: n - 1]
    print()
    hd = "%-12s %10s %8s %10s %10s %11s %12s" % (
        "zone", "positions", "share", "PPMD", "TF", "d/token", "d bytes")
    print(hd); print("-" * len(hd))
    rows_out = []
    for zi, name in enumerate(ZN):
        sel = ok & (z == zi)
        k = int(sel.sum())
        if not k:
            continue
        d = float((losses[sel] - ppmd[sel]).sum())
        rows_out.append((name, k, 100.0 * k / int(ok.sum()), float(ppmd[sel].mean()),
                         float(losses[sel].mean()), float((losses[sel] - ppmd[sel]).mean()),
                         d / np.log(2) / 8))
    for r in sorted(rows_out, key=lambda r: r[6]):
        print("%-12s %10d %7.2f%% %10.4f %10.4f %+11.5f %+12.1f" % r)
    print()
    print("total        %10d          %10.4f %10.4f %+11.5f"
          % (int(ok.sum()), float(ppmd[ok].mean()), float(losses[ok].mean()),
             float((losses[ok] - ppmd[ok]).mean())))
    out = os.environ.get("ZONE_EVAL_TSV")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("zone\tpositionen\tanteil_pct\tppmd\ttf\tdelta_pro_token"
                    "\tdelta_bytes\n")
            for r in rows_out:
                f.write("%s\t%d\t%.4f\t%.6f\t%.6f\t%+.6f\t%+.1f\n" % r)
        print("table: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
