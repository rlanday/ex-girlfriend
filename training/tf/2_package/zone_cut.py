#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Cut point of the training package from the zone mask of the dump run.

The gated zone 3 starts at the first stream bytes and covers the metadata block
(with the census metablock behind it).  The package starts behind its end: in
the text form there the slots of the text start follow (dictionary article,
author article, census pseudo article: transformer documents with their head
rows), then page 0.  A text-form stream has one further short zone-3 run, the
title line of page 0 behind the last slot (the gate's run-on, as behind the
metadata block of the earlier form); it stays in the package's zone mask and is
not scored there (mk_pkgdump keeps zone 0 only).  On a stream of the earlier
form there is one run and every value below equals the one the chain computed.

usage:
  zone_cut.py cut  <zonenmaske.u8>                          -> prints the cut
  zone_cut.py gate <zonenmaske.u8> <cut> <out gate_cut.u8>  -> zone mask behind the cut
  zone_cut.py skip <paket-zeilen.bin> <cut>                 -> prints SKIP for mk_pkgdump
"""
import sys

import numpy as np

REC = 12   # bytes per dump row (dumpv2.REC); int32 src first


def laeufe(m):
    nz = (m != 0).astype(np.int8)
    d = np.diff(np.concatenate([[0], nz, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def cut(path):
    m = np.fromfile(path, dtype=np.uint8)
    h = np.bincount(m, minlength=6)
    print("# histogram", {i: int(h[i]) for i in range(6) if h[i]})
    assert int(h[5]) == 0
    assert set(int(v) for v in np.unique(m)) <= {0, 3}, "zones other than 0 and 3"
    r = laeufe(m)
    assert r and r[0][0] <= 8, "the first zone-3 run does not start at the stream head"
    a, b = int(r[0][0]), int(r[0][1])
    print(f"# metadata block {a:,}..{b:,} ({b - a + 1:,} B), stream bytes {len(m):,}, "
          f"n_tokens {len(m) - (b + 1):,}")
    for x, y in r[1:]:
        print(f"# further zone-3 run {int(x):,}..{int(y):,} ({int(y - x + 1):,} B): stays in the "
              f"package's zone mask, not scored")
    print(b + 1)


def gate(path, schnitt, out):
    m = np.fromfile(path, dtype=np.uint8)
    g = m[schnitt:]
    g.tofile(out)
    h = {int(v): int(c) for v, c in zip(*np.unique(g, return_counts=True))}
    r = laeufe(g)
    print(f"cut {len(g):,} B, histogram {h}, zone-3 runs behind the cut {len(r)}"
          + "".join(f" [{int(x):,}..{int(y):,}]" for x, y in r[:4]))
    assert set(h) <= {0, 3}
    assert all(y - x + 1 <= 4096 for x, y in r), "a long zone-3 run behind the cut"


def skip(path, schnitt):
    roh = np.memmap(path, dtype=np.uint8, mode="r")
    n = len(roh) // REC
    src = np.array(roh[:n * REC].reshape(n, REC)[:, :4]).view(np.int32).ravel()
    # the last stream row in front of the cut; the rows behind it up to the
    # first package stream row are inserted rows (the head of slot 1), and the
    # package keeps them.  Row 0 is the header row, mk_pkgdump/dumpv2 skips it
    # by itself, so the row index equals SKIP.
    vor = np.flatnonzero((src >= 0) & (src < schnitt))
    j = int(vor[-1])
    erst = int(np.flatnonzero(src >= schnitt)[0])
    zw = src[j + 1:erst]
    assert (zw < 0).all(), "stream rows between the cut and the first package row"
    print(f"# last stream row in front of the cut {j:,} (src {int(src[j]):,}), first package "
          f"stream row {erst:,}, inserted rows between {erst - j - 1:,}", file=sys.stderr)
    print(j)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "cut" and len(a) == 2:
        cut(a[1])
    elif a and a[0] == "gate" and len(a) == 4:
        gate(a[1], int(a[2]), a[3])
    elif a and a[0] == "skip" and len(a) == 3:
        skip(a[1], int(a[2]))
    else:
        sys.exit(__doc__)
