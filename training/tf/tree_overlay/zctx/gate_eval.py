#!/usr/bin/env python3
"""Evaluation of the gating arms in TOTAL nats over the same original material

The training loss of a gating arm is not directly comparable with that of the
reference: the arm lacks the gated positions, and those are the expensive
ones, so a lower loss per token proves nothing.  What is compared is what the
target system pays in total:

    total(ref)  = N_all  x  loss_ref
    total(arm)  = N_kept x loss_arm  +  nats_PPMD(gated)

At the gated positions the transformer does not run; there the slot pays the
distribution it delivers without it, the PPMd prior.  That is not estimated
but counted exactly on the SAME package (`zgate pkgstat`, the same set of
positions and the same pairing as `ppmd_loss.py`, calibrated against the PPMd
anchor 1.7519378).

This gives the threshold the arm has to beat:

    loss_arm <  (N_all x loss_ref - nats_PPMD(gated)) / N_kept

The comparison is made at the highest common epoch index (the arms stop
earlier than the 55-epoch reference but follow its LR schedule), and sigma is
taken at exactly this index from the twin pair ref/ref2, not from the 55-epoch
value.

    python -m zctx.gate_eval <ref.jsonl> <ref2.jsonl> <arm.jsonl> <pkgstat.json> [M]
"""

from __future__ import annotations

import json
import math
import os
import sys

NAT_TO_BYTE = 1.0 / math.log(2) / 8.0


def epoch_losses(path: str) -> list:
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "epoch_loss" not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "train/epoch_loss" in d:
                out.append((float(d["epoch"]), float(d["train/epoch_loss"])))
    return out


def tail_mean(values: list, upto: int, frac: float = 0.10) -> float:
    """Mean of the last `frac` of the epochs up to and including `upto`

    With this method the z1 delta of +0.000632 is reproduced exactly.
    """
    v = [x for x in values if x[0] <= upto + 1e-9]
    assert v, "no epochs up to %d" % upto
    k = max(1, int(round(len(v) * frac)))
    return sum(x[1] for x in v[-k:]) / k


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    ref_p, ref2_p, arm_p, pkg_p = sys.argv[1:5]
    m = sys.argv[5] if len(sys.argv) > 5 else "1"

    ref, ref2, arm = (epoch_losses(p) for p in (ref_p, ref2_p, arm_p))
    stat = json.load(open(pkg_p, encoding="utf-8"))[m]
    n_gated = int(stat["n_gated"])
    n_kept = int(stat["n_kept"])
    nats_gated = float(stat["nats_gated"])
    n_all = n_gated + n_kept

    e_common = int(min(ref[-1][0], ref2[-1][0], arm[-1][0]))
    l_ref = tail_mean(ref, e_common)
    l_ref2 = tail_mean(ref2, e_common)
    l_arm = tail_mean(arm, e_common)
    # Convention: sigma = |ref - ref2| of the twin pair.  It is deliberately
    # conservative; the usual estimate of the spread of a single measurement
    # from a pair is |diff|/sqrt(2), which is smaller.  Both are printed; the
    # verdict uses the convention, so that the numbers are on the same scale
    # as the z1/z3 results.
    sigma_simple = abs(l_ref - l_ref2)
    sigma = sigma_simple / math.sqrt(2.0)

    print("comparison at the highest common epoch index: %d" % e_common)
    print("  (arm curve ends at %.2f, ref at %.2f, ref2 at %.2f)"
          % (arm[-1][0], ref[-1][0], ref2[-1][0]))
    print()
    print("positions (package, M=%s):" % m)
    print("  total      %12d" % n_all)
    print("  kept       %12d  (%.4f %%)" % (n_kept, 100.0 * n_kept / n_all))
    print("  gated      %12d  (%.4f %%)" % (n_gated, 100.0 * n_gated / n_all))
    print("  PPMD there %12.6f nats/position  (sum %.1f)"
          % (nats_gated / n_gated, nats_gated))
    print()
    print("loss (mean of the last 10 %% up to epoch %d):" % e_common)
    print("  ref        %.6f" % l_ref)
    print("  ref2       %.6f" % l_ref2)
    print("  arm        %.6f   (on the KEPT positions)" % l_arm)
    print("  sigma      %.6f  (convention |ref-ref2|; statistical |d|/sqrt2 = %.6f)"
          % (sigma_simple, sigma))
    print()

    for name, lr in (("ref", l_ref), ("ref2", l_ref2)):
        schwelle = (n_all * lr - nats_gated) / n_kept
        ges_ref = n_all * lr
        ges_arm = n_kept * l_arm + nats_gated
        d_nats = ges_arm - ges_ref
        print("against %s (loss %.6f):" % (name, lr))
        print("  threshold for the arm      %.6f nats/token" % schwelle)
        print("  arm lies                   %+.6f below/above"
              % (l_arm - schwelle))
        print("  total ref                  %15.1f nats" % ges_ref)
        print("  total arm                  %15.1f nats" % ges_arm)
        print("  DELTA                      %+15.1f nats = %+.1f bytes"
              % (d_nats, d_nats * NAT_TO_BYTE))
        if sigma_simple > 0:
            print("  in sigma (loss level)      %+.1f sigma"
                  % ((l_arm - schwelle) / sigma_simple))
        print()

    print("TIME SIDE (model rows the transformer computes at all):")
    rows_env = os.environ.get("ZGATE_ROWS")
    if rows_env:
        rows = int(rows_env)
        print("  rows of the arm            %12d  (%+.3f %% vs. ref)"
              % (rows, 100.0 * (rows - n_all) / n_all))
    else:
        print("  arm g0 (blind)             %12d  (%+.3f %%)"
              % (n_kept, 100.0 * (n_kept - n_all) / n_all))
        print("  arm g1: rows = kept + sentinel rows "
              "(set ZGATE_ROWS; the loader prints them in its start log)")
    print()
    print("READING: DELTA negative = better for us (project convention).  The "
          "number applies to the 200 M training package; on enwik9 it scales "
          "with the stream length, NOT with the result on the excerpt (the excerpt at "
          "offset 499M, with 11.5 %% digits, holds about three times as many digits "
          "as this package with 3.2 %%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
