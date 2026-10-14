#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Removes rows from a LARGE table in the same file, without a copy.

    trim_inplace.py <mask.u8> <file> <width> [--journal <path>] [--trocken]

Mask byte 1 = remove the row, 0 = keep it.  `--trocken` is a dry run.

Why in place.  `ppmd-probs.float16` of the full stream is about 231 GB before
trimming (one row of 416 B per coder step).  A trimmed copy would need twice
that on one disk.

Why it is safe.  Rows are only removed, never inserted.  The write position
therefore always trails the read position (write_off <= read_off at every
moment), and a block is read completely before anything is written.  Only
data that has already been read is overwritten.  At the end the file is
truncated to the expected size.

Restartable.  After every block (read_off, write_off, block index) is written
to <journal> and fsync is called.  If the run stops, the same call continues
there; without the journal an interrupted run would destroy the table.
`TRIM_ROWS` (rows per block) exists only for the restart test.
"""
import os, sys, numpy as np

MASKE, DATEI, BREITE = sys.argv[1], sys.argv[2], int(sys.argv[3])
JOUR = DATEI + ".trimjournal"
if "--journal" in sys.argv: JOUR = sys.argv[sys.argv.index("--journal")+1]
TROCKEN = "--trocken" in sys.argv

m = np.fromfile(MASKE, dtype=np.uint8)
n = len(m)
keep = m == 0
nk = int(keep.sum())
groesse = os.path.getsize(DATEI)
soll_in = n * BREITE
soll_out = nk * BREITE
print(f"mask       : {MASKE}  {n:,} rows, {n-nk:,} of them to remove")
print(f"file       : {DATEI}  {groesse:,} B")
print(f"expected   : in  {soll_in:,} B   out  {soll_out:,} B   (width {BREITE})")
if groesse == soll_out:
    print("the file already has the expected size; apparently trimmed, nothing to do.")
    sys.exit(0)
if groesse != soll_in:
    sys.exit(f"ERROR: file is {groesse:,} B, expected {soll_in:,} B ({n:,} x {BREITE})")
if TROCKEN:
    print("--trocken: nothing written."); sys.exit(0)

R = int(os.environ.get("TRIM_ROWS", "0")) or max(1, (1 << 28) // BREITE)   # about 256 MiB per block
                                     # TRIM_ROWS only for the restart test
blk0, read_off, write_off = 0, 0, 0
if os.path.exists(JOUR):
    t = open(JOUR).read().split()
    blk0, read_off, write_off = int(t[0]), int(t[1]), int(t[2])
    print(f"JOURNAL found: continuing at block {blk0} (read {read_off:,} write {write_off:,})")

f = os.open(DATEI, os.O_RDWR)
try:
    blk = blk0
    while True:
        a = blk * R
        if a >= n: break
        b = min(a + R, n)
        os.lseek(f, read_off, os.SEEK_SET)
        rein = os.read(f, (b - a) * BREITE)
        assert len(rein) == (b - a) * BREITE, f"short read at block {blk}"
        buf = np.frombuffer(rein, dtype=np.uint8).reshape(b - a, BREITE)
        raus = buf[keep[a:b]]
        if len(raus):
            os.lseek(f, write_off, os.SEEK_SET)
            g = raus.tobytes()
            geschrieben = 0
            while geschrieben < len(g):
                geschrieben += os.write(f, g[geschrieben:])
            write_off += len(g)
        read_off += (b - a) * BREITE
        blk += 1
        os.fsync(f)
        with open(JOUR, "w") as j:
            j.write(f"{blk} {read_off} {write_off}\n"); j.flush(); os.fsync(j.fileno())
        if blk % 20 == 0:
            print(f"  block {blk}: read {read_off/1e9:.1f} GB, written {write_off/1e9:.1f} GB "
                  f"({100.0*read_off/soll_in:.1f} %)", flush=True)
    assert write_off == soll_out, f"written {write_off:,} != expected {soll_out:,}"
    os.ftruncate(f, soll_out)
    os.fsync(f)
finally:
    os.close(f)
os.remove(JOUR)
print(f"DONE: {DATEI} is {os.path.getsize(DATEI):,} B (expected {soll_out:,}) "
      f"{'OK' if os.path.getsize(DATEI)==soll_out else 'MISMATCH'}")
