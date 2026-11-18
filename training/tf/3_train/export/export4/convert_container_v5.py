#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  The container formats this script reads and writes (magic
# FX2TFWC4 and FX2TFWC5), the range coder and the bit trees are
# `weights_compress_v3.py` next to this file, which is itself a fork of
# `pysrc/weights_compress.py` from fx2-cmix-transformer by Vladimir Ivanov
# (GPL-3.0); formats v1 and v2, the range coder, the bit trees and the RoPE
# recomputation are his.  Our addition is this script.  See
# ../../ATTRIBUTION.md.
#
# Copyright (C) 2026 Vladimir Ivanov (original)
# Copyright (C) 2026 Marcus Woelk (this script)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""convert a finished v4 container (FX2TFWC4) into format v5 (FX2TFWC5)

Format v5 is described in weights_compress_v3.py.  It codes the same tensors
as v4, bit for bit; only the coder models change (no tensor is touched):

  ENC_INT4W  the int4 values through one of 24 counting trees, chosen by a
             causal estimate of row width x column width (integer arithmetic)
  ENC_F32HI  the bf16-rounded f32 tensors as bf16 with trees of their own
  ENC_BF16C  every row scale with trees of its own per canonical name

Steps, each one checked:

  1. identity probe: the v4 writer of this tree reproduces the input container
     BIT FOR BIT (else the byte counts below would be meaningless).
  2. write v5.
  3. read v5 back: every tensor bit-identical to the input, same order.

Deterministic: no randomness, no network, no environment.  Same input, same
output, byte for byte.  The C++ loader of this entry reads v5; a loader
without v5 support refuses it loudly.

Usage, from the tree root (a torch free stand in for `pysrc.export_weights`
is used automatically when torch is not installed):

  python3 tf/4_export/convert_container_v5.py <in.c4> <out.c5>
"""

import hashlib
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# installs the torch free stand in for pysrc.export_weights when needed
import prune_dead_neurons  # noqa: E402,F401
import weights_compress_v3 as wc3  # noqa: E402


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def same_tensor(a, b):
    return (a[0] == b[0] and a[1].dtype == b[1].dtype and a[1].shape == b[1].shape
            and a[1].tobytes() == b[1].tobytes())


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = argv[1:]
    if os.path.exists(dst):
        print("refusing to overwrite %s" % dst, file=sys.stderr)
        return 3
    with open(src, "rb") as f:
        magic = f.read(8)
    if magic != wc3.MAGIC_V4:
        print("STOP: %s is not a v4 container (magic %r)" % (src, magic), file=sys.stderr)
        return 4
    print("input  %s: %d B, sha256 %s" % (src, os.path.getsize(src), sha256(src)))
    tensors = wc3.read_tensor_file_v3(src)
    order = list(tensors)
    entries = [(n, tensors[n][0], tensors[n][1]) for n in order]

    fd, tmp = tempfile.mkstemp(prefix="v5_identity_", suffix=".c4",
                               dir=os.path.dirname(os.path.abspath(dst)))
    os.close(fd)
    try:
        wc3.write_tensor_file_v4(tmp, entries)
        with open(src, "rb") as a, open(tmp, "rb") as b:
            ident = a.read() == b.read()
    finally:
        os.unlink(tmp)
    if not ident:
        print("STOP: this writer does NOT reproduce %s bit for bit.  Nothing was "
              "written." % src, file=sys.stderr)
        return 5
    print("identity probe (v4 writer): BIT IDENTICAL")

    wc3.write_tensor_file_v5(dst, entries)
    back = wc3.read_tensor_file_v3(dst)
    ok = list(back) == order and all(same_tensor(back[n], tensors[n]) for n in order)
    print("read back v5 against the input: %s (%d tensors)"
          % ("BIT IDENTICAL" if ok else "DIFFERENT", len(order)))
    if not ok:
        os.unlink(dst)
        print("STOP: output removed", file=sys.stderr)
        return 6
    so, sn = os.path.getsize(src), os.path.getsize(dst)
    print("output %s: %d B, sha256 %s" % (dst, sn, sha256(dst)))
    print("container %+d B against the input (%+d B in S, the container ships "
          "twice)" % (sn - so, 2 * (sn - so)))
    print("the compressor must support container format v5 to read this "
          "container.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
