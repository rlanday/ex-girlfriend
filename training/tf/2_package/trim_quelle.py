#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Trim mask of the raw prior raster from its origin raster ppmd-quelle.u8.

The prior run writes one raster row per coder step: per coded stream byte and
per byte injected through all models (the context in front of the article
lines of the author article, the slot heads, the census labels and page
context, the computed Form I texts, ...).  ppmd-quelle.u8 is index-parallel:
0 = the row of a coded stream byte, 1 = an injected byte.  The package keeps
the stream rows behind the cut (the end of the metadata block with the census
metablock: the last PPMd restart stands there), so everything in front of the
cut's first stream row goes, and behind it every injected row.  This replaces
the reconstruction of the injections from the dump flags (prior_mask.py),
which cannot see an injection into a gated zone, and the assumption of one
restart.

usage: trim_quelle.py <ppmd-quelle.u8> <cut> <stream bytes> <raw ppmd-bytes.uint8>
                      <coder stream> <out trim mask> <out ppmd-bytes of the stream rows>
(needs VOCAB_BITMAP in the environment, as the trim step of build_package.sh)
"""
import os
import sys

import numpy as np

qp, schnitt, nstrom, rawp, koder, out, stp = sys.argv[1:8]
schnitt, nstrom = int(schnitt), int(nstrom)
q = np.fromfile(qp, dtype=np.uint8)
raw = np.fromfile(rawp, dtype=np.uint8)
assert len(q) == len(raw), "origin raster and raw raster differ in length"
assert set(int(v) for v in np.unique(q)) <= {0, 1}
strom = np.flatnonzero(q == 0)
assert len(strom) == nstrom, f"stream rows {len(strom):,} != stream bytes {nstrom:,}"
grenze = int(strom[schnitt])
bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
menge = np.array([v for v in range(256) if (bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
lut = np.full(256, 255, np.uint8)
lut[menge] = np.arange(len(menge), dtype=np.uint8)
bett = np.fromfile(koder, dtype=np.uint8)
n = len(bett)
kopf = np.array([1] + [0x80 | ((n >> s) & 0x7F) for s in (28, 21, 14, 7, 0)], dtype=np.uint8)
soll = np.concatenate([lut[kopf], lut[bett]])
assert len(soll) == nstrom and (soll != 255).all()
st_h = raw[grenze:][q[grenze:] == 0]
ok = bool(len(st_h) == nstrom - schnitt and (st_h == soll[schnitt:]).all())
print(f"raw rows {len(raw):,}, stream rows {nstrom:,}, injected rows {int(q.sum()):,} "
      f"(behind the cut {int(q[grenze:].sum()):,}); first package row {grenze:,}")
print(f"stream rows behind the cut equal the stream: {ok} ({len(st_h):,})")
assert ok, "raw raster behind the cut does not match the coder stream"
np.concatenate([soll[:schnitt], st_h]).tofile(stp)
t = np.ones(len(raw), np.uint8)
t[grenze:] = q[grenze:]
t.tofile(out)
print(f"trim mask: {len(t):,} rows, remove {int(t.sum()):,}, keep {int((t == 0).sum()):,}")
