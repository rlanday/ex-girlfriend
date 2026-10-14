#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  The container formats this script reads and writes (magic
# FX2TFWC3 and FX2TFWC4), the range coder and the bit trees are
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

"""convert a finished v3 container (FX2TFWC3) into format v4 (FX2TFWC4)

Format v4 is described in weights_compress_v3.py.  It codes the same tensors
as v3 with two changes: counting trees on the two low byte planes of the f32
tensors, and one flag bit per int4 row that lets an all-zero row go uncoded.

Before writing, the rows of collapsed attention channels are set to zero
(zero_collapsed_rows in weights_compress_v3.py): an int4 `<x>.q` row outside
the MLP whose bf16 row scale satisfies log2|scale| < LIMIT gets int4 row 0 AND
scale 0.  LIMIT is an explicit parameter, default -20.  On the 8M head this
zeroes 86 rows of 29 channels; with them zero, the compressed archive stays
byte for byte the same (measured on two beds), so the rows were redundancy.
Use --no-zero-rows to convert without touching any tensor.

Steps, each one checked:

  1. identity probe: the v3 writer of this tree reproduces the input container
     BIT FOR BIT.  If not, the container was written by a different coder and
     every byte count below would be meaningless, so the script stops.
  2. zero the collapsed rows and list them per tensor (stdout, optional TSV).
  3. write v4.
  4. read v4 back: every tensor bit-identical to the zeroed set, same order;
     against the input only the listed rows and their scales differ.

Deterministic: no randomness, no network, no environment.  Same input, same
output, byte for byte.  The C++ loader of this entry reads v4; a loader
without v4 support refuses it loudly.

Usage, from the tree root (a torch free stand in for `pysrc.export_weights`
is used automatically when torch is not installed):

  python3 tf/4_export/convert_container_v4.py <in.c3> <out.c4>
          [--zero-rows-log2 LIMIT | --no-zero-rows] [--report <tsv>]
"""

import hashlib
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# installs the torch free stand in for pysrc.export_weights when needed
import prune_dead_neurons  # noqa: E402,F401
import weights_compress_v3 as wc3  # noqa: E402


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def identity_probe(path, entries):
    """does the v3 writer of this tree reproduce this container bit for bit?"""
    fd, tmp = tempfile.mkstemp(prefix="v4_identity_", suffix=".c3",
                               dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        wc3.write_tensor_file_v3(tmp, entries)
        with open(path, "rb") as a, open(tmp, "rb") as b:
            return a.read() == b.read()
    finally:
        os.unlink(tmp)


def same_tensor(a, b):
    return (a[0] == b[0] and a[1].dtype == b[1].dtype and a[1].shape == b[1].shape
            and a[1].tobytes() == b[1].tobytes())


def main(argv):
    args = list(argv[1:])
    report_path = None
    if "--report" in args:
        i = args.index("--report")
        report_path = args[i + 1]
        del args[i:i + 2]
    limit = wc3.ZERO_ROWS_LOG2
    if "--zero-rows-log2" in args:
        i = args.index("--zero-rows-log2")
        limit = float(args[i + 1])
        del args[i:i + 2]
    if "--no-zero-rows" in args:
        args.remove("--no-zero-rows")
        limit = None
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = args
    if os.path.exists(dst):
        print("refusing to overwrite %s" % dst, file=sys.stderr)
        return 3
    with open(src, "rb") as f:
        magic = f.read(8)
    if magic != wc3.MAGIC_V3:
        print("STOP: %s is not a v3 container (magic %r)" % (src, magic),
              file=sys.stderr)
        return 4

    print("input  %s: %d B, sha256 %s" % (src, os.path.getsize(src), sha256(src)))
    tensors = wc3.read_tensor_file_v3(src)
    order = list(tensors)
    entries = [(n, tensors[n][0], tensors[n][1]) for n in order]

    print("identity probe: writing %s back through the v3 coder ..." % src)
    if not identity_probe(src, entries):
        print("STOP: this writer does NOT reproduce %s bit for bit.  The "
              "container in hand was written by a different coder.  Nothing "
              "was written." % src, file=sys.stderr)
        return 5
    print("identity probe: BIT IDENTICAL")

    if limit is None:
        report = []
        print("zero rows: off (no tensor is changed)")
    else:
        entries, report = wc3.zero_collapsed_rows(entries, limit)
        per_tensor = {}
        for qname, row, _lg, _cols, _nz in report:
            per_tensor.setdefault(qname, []).append(row)
        print("zero rows: log2|scale| < %g outside the MLP: %d rows in %d tensors"
              % (limit, len(report), len(per_tensor)))
        for qname in sorted(per_tensor):
            print("  %s: rows %s" % (qname, " ".join(str(r) for r in per_tensor[qname])))
    if report_path:
        with open(report_path, "w") as f:
            f.write("tensor\trow\tlog2_abs_scale\tvalues\tnonzero_before\n")
            for qname, row, lg, cols, nz in report:
                f.write("%s\t%d\t%.2f\t%d\t%d\n" % (qname, row, lg, cols, nz))
        print("report written: %s" % report_path)

    wc3.write_tensor_file_v4(dst, entries)

    back = wc3.read_tensor_file_v3(dst)  # reads v3 and v4, the magic decides
    want = {n: (c, a) for n, c, a in entries}
    ok = list(back) == order and all(same_tensor(back[n], want[n]) for n in order)
    print("read back v4 against the zeroed tensors: %s (%d tensors)"
          % ("BIT IDENTICAL" if ok else "DIFFERENT", len(order)))
    zeroed_q = {r[0] for r in report}
    zeroed_s = {q[:-2] + ".scale" for q in zeroed_q}
    rows_of = {}
    for qname, row, _lg, _cols, _nz in report:
        rows_of.setdefault(qname, set()).add(row)
    changed_ok = True
    for n in order:
        if same_tensor(back[n], tensors[n]):
            continue
        if n in zeroed_q:
            a0 = tensors[n][1]
            a1 = back[n][1]
            rows = a0.shape[0] if a0.ndim > 1 else 1
            d = (a0.reshape(rows, -1) != a1.reshape(rows, -1)).any(axis=1)
            changed_ok &= set(np.flatnonzero(d).tolist()) <= rows_of[n]
        elif n in zeroed_s:
            d = tensors[n][1].ravel() != back[n][1].ravel()
            changed_ok &= set(np.flatnonzero(d).tolist()) <= rows_of[n[:-6] + ".q"]
        else:
            changed_ok = False
            print("  unexpected change: %s" % n)
    print("against the input: %s" % ("only the listed rows and scales differ"
                                      if changed_ok else "UNEXPECTED DIFFERENCES"))
    if not (ok and changed_ok):
        os.unlink(dst)
        print("STOP: output removed", file=sys.stderr)
        return 6

    so, sn = os.path.getsize(src), os.path.getsize(dst)
    print("output %s: %d B, sha256 %s" % (dst, sn, sha256(dst)))
    print("container %+d B against the input (%+d B in S, the container ships "
          "twice)" % (sn - so, 2 * (sn - so)))
    print("the compressor must support container format v4 to read this "
          "container.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
