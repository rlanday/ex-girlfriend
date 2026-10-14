#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""title-lengths.int32 of the training package

    title_lengths.py <article-boundaries.int32> <pages.tsv> <out.int32>
                     [--bett-off 0]

One int32 per document: the length of its title line in bytes, including the
closing line feed.  The source is the page table of `pages.py` (columns
titel_start/titel_len), not the zone mask: the title is predicted in this
training form, so the coder's zone mask has no title zone to read it from.

Rule per document: the title of the first page inside the document range.
With merged redirect runs a document holds many pages, and the first one is
taken.  `--bett-off` converts the stream positions of the page table into the
positions of the package (they start behind the metadata block).
"""
import sys, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("boundaries"); ap.add_argument("grenzen"); ap.add_argument("out")
ap.add_argument("--bett-off", type=int, default=0)
a = ap.parse_args()

b = np.fromfile(a.boundaries, dtype=np.int32).astype(np.int64)
G = np.loadtxt(a.grenzen, delimiter="\t", skiprows=1, dtype=np.int64)
ts = G[:, 3] - a.bett_off          # titel_start in package coordinates
tl = G[:, 4]
srt = np.argsort(ts); ts = ts[srt]; tl = tl[srt]

out = np.zeros(len(b) - 1, dtype=np.int32)
ohne = 0
for i in range(len(b) - 1):
    lo, hi = int(b[i]), int(b[i + 1])
    j = np.searchsorted(ts, lo)
    if j < len(ts) and ts[j] < hi:
        out[i] = int(tl[j])
    else:
        ohne += 1
out.tofile(a.out)
print(f"title-lengths: {len(out)} documents, sum {int(out.sum()):,} B, "
      f"min {int(out.min())} max {int(out.max())}, without a title in range: {ohne}")
print(f"  source: page table ({len(G):,} pages), not the zone mask "
      f"(this form has no title zone)")
