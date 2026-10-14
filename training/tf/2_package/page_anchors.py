#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Page boundaries on the coder stream, metadata block in front

    python3 page_anchors.py <coder stream> <out.tsv>

Every page ends with exactly one anchor `  </page>\\n` (after WRT the bytes
20 20 4c 2f df 98 4e 0a), 243,425 times and never inside text.  The title line
of the next page starts directly behind the anchor and ends at the first 0x0a.

Two rules follow from the metadata block standing in front of the pages:
 (1) Page 0 does not start at the stream header but behind the metadata block.
     The start is found with the rule the decoder uses itself: every line of
     the tab-separated block form carries at least one tab, no title line
     does, so the first line without a tab after the stream header is the
     title line of page 0.  (In the packed form of the block this rule does
     not hold; `pages.py` then sets the row of page 0 from the zone mask.)
 (2) The last anchor opens no page (behind it stands the tail), so the table
     has exactly 243,425 rows, pages 0 to 243,424.

Columns (German names kept, the table is read by other tools): seite = page,
anker_off = anchor offset, seiten_start = page start, titel_start/titel_len =
title start/length.  Page 0 has anker_off = -1.  Exit 1 if the anchor count or
the tail length differ from the expected values.
"""
import sys

ANK = bytes([0x20, 0x20, 0x4c, 0x2f, 0xdf, 0x98, 0x4e, 0x0a])  # '  </page>\n' post-WRT
KOPF_ENDE = 29          # end of the stream header in this form
TAIL = 6222             # .intro/.coda appendix at the end of the stream

P   = sys.argv[1]
OUT = sys.argv[2]

d = open(P, "rb").read()
N = len(d)
print(f"# stream: {P}  {N:,} B")

# ---- (1) end of the block / start of page 0 by the tab rule
pos0 = KOPF_ENDE
zeilen = 0
while True:
    nl = d.index(b"\n", pos0)
    if b"\t" not in d[pos0:nl]:
        break
    zeilen += 1
    pos0 = nl + 1
st0 = pos0
print(f"# metadata block: from {KOPF_ENDE:,}, {zeilen:,} lines with a tab, "
      f"length {st0 - KOPF_ENDE:,} B")
print(f"# page 0 starts at {st0:,}")

# ---- collect the anchors
pos = []
s = 0
while True:
    i = d.find(ANK, s)
    if i < 0:
        break
    pos.append(i)
    s = i + 1
print(f"# anchors: {len(pos):,}")

# ---- (2) checks that expose a wrong position at once
letztes_ankerende = pos[-1] + len(ANK)
print(f"# last anchor ends at {letztes_ankerende:,}; "
      f"rest up to the end of the stream {N - letztes_ankerende:,} B (expected tail {TAIL})")
ok = True
if N - letztes_ankerende != TAIL:
    print(f"! TAIL DIFFERS: {N - letztes_ankerende} != {TAIL}"); ok = False
if len(pos) != 243425:
    print(f"! ANCHOR COUNT DIFFERS: {len(pos):,} != 243,425"); ok = False
# every anchor but the last opens a page; page 0 comes from (1)
seiten = len(pos)
print(f"# pages: 1 (page 0) + {len(pos)-1:,} (from anchors) = {seiten:,}")

with open(OUT, "w") as f:
    f.write("seite\tanker_off\tseiten_start\ttitel_start\ttitel_len\n")
    nl0 = d.index(b"\n", st0)
    f.write(f"0\t-1\t{st0}\t{st0}\t{nl0-st0}\n")
    for k, a in enumerate(pos[:-1]):
        st = a + len(ANK)
        nl = d.index(b"\n", st)
        f.write(f"{k+1}\t{a}\t{st}\t{st}\t{nl-st}\n")
print(f"# written: {OUT}")
for k in (0, 1, 100000, 243424):
    if k == 0:
        st = st0; a = -1
    else:
        a = pos[k-1]; st = a + len(ANK)
    nl = d.index(b"\n", st)
    print(f"  page {k}: anchor {a:,}  title {nl-st} B")
sys.exit(0 if ok else 1)
