#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Token stream and prior0 table of the training package

    VOCAB_BITMAP=<vocab-bitmap.u8> python3 ppmd_bytes.py <paket-zeilen.bin> \
        <token_start> <n_tokens> <package-dir> <coder stream>

Writes `ppmd-bytes.uint8`: the coder stream with its 6-byte segment header,
mapped through the 208-value byte set, from position `token_start` (the first
byte behind the metadata block) on.  It checks that every stream row of the
dump behind the cut carries exactly that token.

Writes `z3-prior0.float16`: the order-0 distribution of these tokens,
(count + 1) / (N + 208) per byte value, as float16.  It is the replacement prior
of every row without a stream target, and the coder carries the same table.
"""
import sys, os, numpy as np, hashlib
roh, s, nt, Z, bett = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
SATZ = np.dtype([("src","<i4"),("tok","u1"),("p1","u1"),("p2","u1"),("mb","u1"),("mc","u1"),("fl","u1"),("res","<u2")])
bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
menge = np.array([v for v in range(256) if (bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
lut = np.full(256, 255, np.uint8); lut[menge] = np.arange(len(menge), dtype=np.uint8)
b = np.fromfile(bett, dtype=np.uint8); n = len(b)
# segment header: 0x01 and the stream length in five 7-bit groups
kopf = np.array([1] + [0x80 | ((n >> k) & 0x7F) for k in (28, 21, 14, 7, 0)], dtype=np.int64)
raster = np.concatenate([lut[kopf], lut[b]])
assert (raster != 255).all(), "byte outside the 208-value set"
pk = raster[s:]; assert len(pk) == nt, (len(pk), nt)
pk.tofile(Z + "/ppmd-bytes.uint8")
r = np.memmap(roh, dtype=SATZ, mode="r"); src = np.asarray(r["src"]); tok = np.asarray(r["tok"])
sel = src >= s
abw = int((tok[sel] != raster[src[sel]]).sum())
print(f"dump stream rows behind the cut {int(sel.sum()):,} of {nt:,} positions (the rest are positions that are not fed), token mismatches against the stream {abw}")
assert abw == 0
c = np.bincount(pk, minlength=256)[:208].astype(np.float64)
p = ((c + 1.0) / (c.sum() + 208.0)).astype(np.float16); p.tofile(Z + "/z3-prior0.float16")
print("ppmd-bytes sha", hashlib.sha256(pk.tobytes()).hexdigest())
print("prior0 sha    ", hashlib.sha256(p.tobytes()).hexdigest())
