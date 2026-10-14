#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Mask over the rows of the raw prior table: is this row an injected byte?

    prior_mask.py <run dir with paket-zeilen.bin> <stream bytes> <out.u8> [--bit5]

(`--shift N` is accepted and not used.)

Why.  `ppmd-probs.float16` and `ppmd-bytes.uint8` of the prior run are written
in the raster of the coder's model steps: one row per stream byte (also per
gated one) plus one row per byte that the coder injects through all models.
The training package needs the stream raster.  The source of the mask is the
coder's row dump itself: a flag bit marks every injected row, and the order of
the dump rows is the order of the coder's steps.

Which bit.  Alias rows (bit5) go only to the transformer and get no PPMd row;
author rows (bit6) run through all models.  The PPMd raster is therefore
stream + bit6 rows (default), not stream + bit5 rows (`--bit5`).

Construction.  For every stream byte p a 0, directly followed by k(p) ones,
where k(p) is the number of injected rows that stand in the dump between the
stream row p and the next stream row.  Stream positions that are not fed (the
metadata block zone, the closing tokens of documents) have no dump row and
therefore k = 0; nothing is injected there either.

Acceptance (done by the caller): mask length == number of raw prior rows, and
the trimmed ppmd-bytes == segment header + coder stream through the 208-value
set.
"""
import sys, numpy as np
BIT = 64 if "--bit5" not in sys.argv else 32

D      = sys.argv[1].rstrip("/")
NSTROM = int(sys.argv[2])          # stream bytes including the 6-byte header
OUT    = sys.argv[3]
SHIFT  = 6 if "--shift" not in sys.argv else int(sys.argv[sys.argv.index("--shift")+1])

# memmap and narrow columns: the full-stream dump is 7 GB; np.fromfile would
# copy it, and an int64 column would add another 4.7 GB.
SATZ = np.dtype([("src","<i4"),("tok","u1"),("p1","u1"),("p2","u1"),
                 ("mb","u1"),("mc","u1"),("fl","u1"),("res","<u2")])
R0 = np.memmap(f"{D}/paket-zeilen.bin", dtype=SATZ, mode="r")
src = np.asarray(R0["src"])          # int32, 2.4 GB
fl  = np.asarray(R0["fl"])           # uint8, 0.6 GB
# drop the header record (src == -2); the excerpt offset is NOT subtracted:
# the mask runs in coder coordinates, and the 6 header bytes are stream bytes.
if src[0] == -2:
    src, fl = src[1:], fl[1:]
okb = (fl & BIT) != 0
print(f"dump rows {len(src):,}  stream rows {int((src>=0).sum()):,}  injected rows {int(okb.sum()):,}")

# the last stream position seen before each row, block by block so that no
# second 2.4 GB copy is made
pos_okb = np.empty(int(okb.sum()), dtype=np.int64)
S = 1 << 24; g = 0; lauf = np.int32(-1)
for a0 in range(0, len(src), S):
    b0 = min(a0 + S, len(src))
    blk = src[a0:b0].copy()
    np.maximum(blk, lauf, out=blk, where=blk < 0)
    np.maximum.accumulate(blk, out=blk)
    lauf = blk[-1]
    sel = blk[okb[a0:b0]]
    pos_okb[g:g+len(sel)] = sel; g += len(sel)
assert g == len(pos_okb)
assert (pos_okb >= 0).all(), "injected row before the first stream row"
k = np.bincount(pos_okb, minlength=NSTROM).astype(np.int64)
assert len(k) == NSTROM, f"injection behind stream byte {len(k)-1} >= NSTROM {NSTROM}"
gesamt = NSTROM + int(k.sum())
print(f"stream bytes {NSTROM:,} + injected {int(k.sum()):,} = coder rows {gesamt:,}")

# write the mask piece by piece (no Python loop over 558 million rows)
with open(OUT, "wb") as g:
    S = 1 << 22
    for a in range(0, NSTROM, S):
        b = min(a+S, NSTROM)
        kk = k[a:b]
        laeng = kk + 1
        m = np.ones(int(laeng.sum()), dtype=np.uint8)
        ends = np.cumsum(laeng)
        starts = ends - laeng
        m[starts] = 0                     # the stream byte itself
        g.write(m.tobytes())
import os
print(f"{OUT}: {os.path.getsize(OUT):,} B (expected {gesamt:,})  "
      f"{'OK' if os.path.getsize(OUT)==gesamt else 'MISMATCH'}")
