#!/usr/bin/env python3
"""read the JSONL logs of the twins and turn the loss difference into bytes

Torch free: it reads only the `train/epoch_loss` lines that `tft/runner.py`
writes, so it runs on the PC while the pod is still busy.

Sign convention: anything GOOD for us is NEGATIVE.  A
twin that lowers the loss therefore prints a negative dS.

The three exchange rates are estimates of the development project, not
derived here:

    33 / 41 / 51 KB of archive9 per 0.001 nats/byte

The 33 comes from a single pairing in Ivanov's notes; the physical ceiling at
100 % pass-through would be 105,762 B.  A result that is a gain only at the
51 KB rate is not counted as a gain.

The weight side is charged at his measured container density, 0.478 B per
parameter, and DOUBLE, because the compressor binary counts twice in S:

    dS = d_archive9 + 2 * 0.478 * d_parameters

    python -m zctx.eval_zctx compare <reference.jsonl> <arm.jsonl> [--params N]
    python -m zctx.eval_zctx spread <ref1.jsonl> <ref2.jsonl> [...]
"""

from __future__ import annotations

import json
import sys

RATES_BYTES_PER_MILLINAT = (33_000, 41_000, 51_000)
BYTES_PER_PARAMETER = 0.478
S_FACTOR = 2                       # the compressor counts twice in S
WINDOW_TOKENS = 199_997_476        # pkg200m
FULL_CORPUS_TOKENS = 601_500_000   # his figure for the whole coder stream


def read_epoch_losses(path: str) -> list:
    """[(epoch, epoch_loss)] in file order"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train/epoch_loss" in record:
                out.append((float(record["epoch"]), float(record["train/epoch_loss"])))
    return out


def read_step_losses(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train/loss" in record:
                out.append((float(record["epoch"]), float(record["train/loss"])))
    return out


def delta_to_bytes(delta_nats: float, delta_parameters: int = 0,
                   extra_bytes: float = 0.0) -> dict:
    """delta nats/byte -> delta archive9 and delta S at the three rates

    `extra_bytes` is a fixed payload the arm adds to the container BEFORE the
    x2 of S: arm z3 ships a 416 B order 0 prior table, arm z1 a <= 1 KB phase
    table.  Parameters and payload are charged the same way, only the density
    differs.
    """
    weight = S_FACTOR * (BYTES_PER_PARAMETER * delta_parameters + extra_bytes)
    out = {"delta_nats": delta_nats, "delta_parameters": delta_parameters,
           "extra_bytes": extra_bytes, "weight_cost_in_S": round(weight, 1)}
    for rate in RATES_BYTES_PER_MILLINAT:
        archive = delta_nats / 0.001 * rate
        out["archive9_@%dKB" % (rate // 1000)] = round(archive, 1)
        out["S_@%dKB" % (rate // 1000)] = round(archive + weight, 1)
    return out


def compare(reference: str, arm: str, delta_parameters: int = 0,
            sigma: float | None = None, extra_bytes: float = 0.0) -> dict:
    a = read_epoch_losses(reference)
    b = read_epoch_losses(arm)
    assert a and b, "no train/epoch_loss records; wrong file?"
    n = min(len(a), len(b))
    if len(a) != len(b):
        print("WARNING: %d reference epochs vs %d arm epochs, comparing the "
              "first %d (a twin must run the SAME schedule)" % (len(a), len(b), n))
    delta_last = b[n - 1][1] - a[n - 1][1]
    tail = max(1, n // 10)
    mean_a = sum(x[1] for x in a[n - tail : n]) / tail
    mean_b = sum(x[1] for x in b[n - tail : n]) / tail
    delta_tail = mean_b - mean_a

    result = {
        "epochs_compared": n,
        "reference_final_epoch_loss": round(a[n - 1][1], 6),
        "arm_final_epoch_loss": round(b[n - 1][1], 6),
        "delta_final": round(delta_last, 6),
        "delta_last_%d_epochs" % tail: round(delta_tail, 6),
    }
    result.update(delta_to_bytes(delta_tail, delta_parameters, extra_bytes))
    if sigma is not None:
        result["seed_sigma"] = round(sigma, 6)
        result["signal_over_sigma"] = round(delta_tail / sigma, 2) if sigma else None
        result["verdict"] = (
            "carries (|delta| > 3 sigma and negative)"
            if delta_tail < -3 * sigma
            else "not distinguishable from seed noise"
            if abs(delta_tail) <= 3 * sigma
            else "HURTS by more than 3 sigma"
        )
    return result


def spread(paths: list) -> dict:
    """seed to seed standard deviation of the final epoch loss"""
    finals = []
    for p in paths:
        e = read_epoch_losses(p)
        assert e, "no train/epoch_loss records in " + p
        finals.append(e[-1][1])
    mean = sum(finals) / len(finals)
    if len(finals) < 2:
        return {"n_runs": 1, "final": finals[0],
                "note": "one run only, no spread; rerun poc4m or take it from "
                        "the runs of the model-size sweep"}
    variance = sum((x - mean) ** 2 for x in finals) / (len(finals) - 1)
    sigma = variance ** 0.5
    return {
        "n_runs": len(finals),
        "finals": [round(x, 6) for x in finals],
        "mean": round(mean, 6),
        "sigma": round(sigma, 6),
        "3sigma_in_bytes_@41KB": round(3 * sigma / 0.001 * 41_000, 1),
    }


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "compare":
        params = int(args[args.index("--params") + 1]) if "--params" in args else 0
        sig = float(args[args.index("--sigma") + 1]) if "--sigma" in args else None
        extra = float(args[args.index("--extra-bytes") + 1]) if "--extra-bytes" in args else 0.0
        flags = {"--params", "--sigma", "--extra-bytes"}
        files = [a for i, a in enumerate(args[1:], start=1)
                 if a not in flags and args[i - 1] not in flags]
        assert len(files) >= 2, "give a reference log and an arm log"
        result = compare(files[0], files[1], params, sig, extra)
    elif args[0] == "spread":
        result = spread(args[1:])
    else:
        print(__doc__)
        return 2
    for key, value in result.items():
        print(str(key).ljust(30) + str(value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
