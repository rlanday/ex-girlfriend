#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Our own tool, but it reads and writes the reference weight format of
# fx2-cmix-transformer by Vladimir Ivanov (GPL-3.0) and imports his
# `pysrc.export_weights`.  See ../../ATTRIBUTION.md.
#
# Copyright (C) 2026 Marcus Woelk
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""bf16 rounding probe for the RAW fp32 tensors of a weight container.

An earlier lossless study showed that the 62 raw f32/i32 tensors (ENC_PLANE4) of
the 4 M model are LOSSLESSLY incompressible: the container already sits 110 B
under the exact entropy bound, the mantissas are noise.  The only lever left is
a LOSSY one: rounding them to bfloat16 precision, which is worth -37,700 B per
container of the 4 M model (the submission ships the container twice).  That is
a MODEL change (Ivanov leaves exactly these tensors unquantized), so it needs a
rate measurement, not a table.

This tool builds the rounded container.  It reads the reference file
(FX2TFW01), rounds the selected classes of f32 tensors to bf16 (round to
nearest, ties to even, the same rule torch.bfloat16 uses) and writes a new
reference file, from which weights_compress can write a v2 or v3 container.

    python3 wpack2_bf16.py <in.bin> <out.bin> [class ...]

Classes (default: all of conv, beta, dt_bias, norm, decay, scale):
    conv      *_convolution.weight                 18 tensors, 13,824 values
    beta      *.beta_projection.weight              6 tensors,  3,456 values
    dt_bias   *.attention.dt_bias                   6 tensors,  1,152 values
    norm      *.output_fused_norm_gate.weight       6 tensors,    384 values
    decay     *.log_baseline_decay_rate             6 tensors,     18 values
    scale     skip/residual/token coefficients     17 tensors,     20 values

NOT touched, on purpose:
    rope.inv_freq: the decoder recomputes rope.sin/rope.cos from it, so
      rounding 32 values would perturb 8,388,608 table entries for 64 B.
    config.ints / config.kimi: i32 configuration, not weights.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from pysrc.export_weights import (  # noqa: E402
    DTYPE_FLOAT32,
    read_tensor_file,
    write_tensor_file,
)

KLASSEN = ("conv", "beta", "dt_bias", "norm", "decay", "scale")


def klasse_of(name: str) -> str:
    if name.endswith("_convolution.weight"):
        return "conv"
    if name.endswith(".beta_projection.weight"):
        return "beta"
    if name.endswith(".attention.dt_bias"):
        return "dt_bias"
    if name.endswith(".output_fused_norm_gate.weight"):
        return "norm"
    if name.endswith(".log_baseline_decay_rate"):
        return "decay"
    if name.endswith("_coefficient.value") or name == "skip_connection_weights.value":
        return "scale"
    return "-"


def to_bf16(a: np.ndarray) -> np.ndarray:
    """round float32 to bfloat16 precision, ties to even, value kept as f32"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bias = np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))
    r = ((u + bias) & np.uint64(0xFFFF0000)).astype(np.uint32)
    out = r.view(np.float32).reshape(a.shape)
    assert np.all(np.isfinite(out)), "bf16 rounding overflowed to inf"
    return out


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    want = tuple(sys.argv[3:]) or KLASSEN
    for k in want:
        if k not in KLASSEN:
            sys.stderr.write("unknown class %r (known: %s)\n"
                             % (k, " ".join(KLASSEN)))
            return 2

    tensors = read_tensor_file(src)
    entries = []
    zahl = {k: [0, 0, 0.0] for k in KLASSEN}  # tensors, values, max rel. error
    for name, (dtype_code, array) in tensors.items():
        k = klasse_of(name)
        if dtype_code == DTYPE_FLOAT32 and k in want:
            neu = to_bf16(array)
            alt = np.ascontiguousarray(array, dtype=np.float32)
            nz = alt != 0
            rel = np.max(np.abs(neu[nz] - alt[nz]) / np.abs(alt[nz])) if nz.any() else 0.0
            zahl[k][0] += 1
            zahl[k][1] += int(array.size)
            zahl[k][2] = max(zahl[k][2], float(rel))
            array = neu
        entries.append((name, dtype_code, np.ascontiguousarray(array)))
    write_tensor_file(dst, entries)

    print("rounded to bf16 (round half to even):")
    ges_t = ges_v = 0
    for k in KLASSEN:
        t, v, rel = zahl[k]
        if t:
            print("  %-8s %3d tensors %6d values  max. rel. error %.3e"
                  % (k, t, v, rel))
            ges_t += t
            ges_v += v
    print("  %-8s %3d tensors %6d values" % ("TOTAL", ges_t, ges_v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
