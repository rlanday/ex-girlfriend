#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Removes the masked rows from a table and writes the rest to a new file.

    trim.py <mask.u8> <in> <out> <row width in bytes>

Mask byte 1 = remove the row, 0 = keep it.  Used for `ppmd-bytes.uint8`
(width 1); the 228 GB prior table is trimmed in place by `trim_inplace.py`.
"""
import sys, numpy as np, os
amk, qin, qout, breite = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
al = np.fromfile(amk, dtype=np.uint8).astype(bool)
n = len(al); soll_in = n*breite
if os.path.getsize(qin) != soll_in:
    sys.exit(f"ERROR: {qin} has {os.path.getsize(qin):,} B, expected {soll_in:,} ({n:,} x {breite})")
keep = ~al; nk = int(keep.sum())
print(f"{qin}: {n:,} rows -> {nk:,} (removed {n-nk:,} rows), width {breite} B")
S = max(1, (1 << 26) // breite)          # about 64 MiB per block
with open(qin, "rb") as f, open(qout, "wb") as g:
    for i in range(0, n, S):
        j = min(i+S, n)
        buf = np.frombuffer(f.read((j-i)*breite), dtype=np.uint8).reshape(j-i, breite)
        sel = buf[keep[i:j]]
        if len(sel): g.write(sel.tobytes())
print(f"{qout}: {os.path.getsize(qout):,} B  (expected {nk*breite:,})  {'OK' if os.path.getsize(qout)==nk*breite else 'MISMATCH'}")
