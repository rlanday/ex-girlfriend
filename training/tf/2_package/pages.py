#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Page table of the packed coder stream

    python3 pages.py <coder stream> <out.tsv> <token_start>
    python3 pages.py --selbstprobe <earlier coder stream>

Runs `page_anchors.py` (rows of pages 1 to 243,424 from the page anchors) and
replaces the row of page 0.  The rule of page_anchors.py for page 0 (first line
without a tab) does not hold in the packed form, where the metadata block is
coded and has lines without a tab.  Page 0 is therefore taken from the end of
the metadata block in the zone mask: token_start = end of the block + 1 in the
coordinates of the dump (coder stream + 6-byte segment header), and the title
line of page 0 is the line that ends at the last '\\n' before stream position
token_start - 6.

`--selbstprobe` (self-test) checks this rule on the coder stream of the
earlier package form, where the metadata block ends at 5,163,705: it must
give the row '0 -1 5163692 5163692 5'.
"""
import os, sys, subprocess

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_anchors.py")


def seite0(d, ts):
    e = ts - 6                       # first stream position behind the gate zone
    nl = d.rfind(b"\n", 0, e)        # end of the title line of page 0
    st = d.rfind(b"\n", 0, nl) + 1   # start of that line
    return st, nl - st


def main():
    bett, out, ts = sys.argv[1], sys.argv[2], int(sys.argv[3])
    roh = out + ".roh"
    subprocess.run(["python3", TOOL, bett, roh], check=True)
    zeilen = open(roh).read().splitlines()
    assert zeilen[0].startswith("seite"), zeilen[0]
    assert zeilen[1].split("\t")[0] == "0"
    d = open(bett, "rb").read()
    st, ln = seite0(d, ts)
    anker1 = int(zeilen[2].split("\t")[1])
    assert st < anker1, (st, anker1)
    zeilen[1] = f"0\t-1\t{st}\t{st}\t{ln}"
    n = len(zeilen) - 1
    assert n == 243425, n
    open(out, "w").write("\n".join(zeilen) + "\n")
    print(f"page 0: {st} title {ln} B ({d[st:st+ln]!r}); pages {n}; anchor of page 1 {anker1}")


if __name__ == "__main__":
    if sys.argv[1] == "--selbstprobe":
        d = open(sys.argv[2], "rb").read(5163800)
        st, ln = seite0(d, 5163705)
        print("self-test on the earlier stream:", st, ln, "->", "OK" if (st, ln) == (5163692, 5) else "WRONG")
        sys.exit(0 if (st, ln) == (5163692, 5) else 1)
    main()
