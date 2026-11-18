#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Page table of the packed coder stream

    python3 pages.py <coder stream> <out.tsv> <token_start> --anker <hex> --seiten N
                     [--zonen <zonenmaske.u8>] [--praefix]
    python3 pages.py --selbstprobe <earlier coder stream>

Runs `page_anchors.py` (rows of pages 1 to N-1 from the page anchors; the
anchor bytes and the page count come from the tree and the input, see there)
and sets the row of page 0.  Page 0 does not follow an anchor; it is taken
from the zone mask of the dump run (coordinates: coder stream + 6-byte segment
header):

 * text form (`--zonen`, the mask has a second zone-3 run): the text start
   (slots 1 to 3) stands between the metadata block and page 0, and the gate
   runs on over the title line of page 0 as a short zone-3 run of its own.
   Page 0 starts at that run; its title is the line that starts there.
 * earlier form (one zone-3 run): token_start = end of the block + 1, and the
   title line of page 0 is the line that ends at the last '\\n' before stream
   position token_start - 6.

`--praefix`: the coder stream is the prefix that a probe bed gave; the page
count and the tail are not checked, a page whose title line is cut off is
left out.

`--selbstprobe` (self-test, by hand only) checks the earlier rule on the coder
stream of the earlier package form, where the metadata block ends at
5,163,705: it must give the row '0 -1 5163692 5163692 5'.  Its numbers belong
to that stream and to nothing the recipe runs.
"""
import os, sys, subprocess

import numpy as np

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_anchors.py")


def seite0(d, ts):
    e = ts - 6                       # first stream position behind the gate zone
    nl = d.rfind(b"\n", 0, e)        # end of the title line of page 0
    st = d.rfind(b"\n", 0, nl) + 1   # start of that line
    return st, nl - st


def seite0_textform(d, zonen, ts):
    """start and title length of page 0 from the second zone-3 run, or None"""
    m = np.fromfile(zonen, dtype=np.uint8)
    nz = (m != 0).astype(np.int8)
    dd = np.diff(np.concatenate([[0], nz, [0]]))
    a, b = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1
    if len(a) < 2:
        return None
    assert int(b[0]) + 1 == ts, (int(b[0]) + 1, ts)
    st = int(a[1]) - 6
    nl = d.index(b"\n", st)
    assert nl - st < int(b[1] - a[1] + 1), "the second zone-3 run does not hold the title line"
    return st, nl - st


def main():
    args = sys.argv[1:]

    def opt(name):
        if name not in args:
            return None
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    praefix = "--praefix" in args
    if praefix:
        args.remove("--praefix")
    zonen, anker, seiten = opt("--zonen"), opt("--anker"), opt("--seiten")
    if not anker or (seiten is None and not praefix):
        sys.exit("pages.py: --anker <hex> and --seiten <N> are required (page_anchors.py --anker-aus-baum / --seiten-zaehlen)")
    bett, out, ts = args[0], args[1], int(args[2])
    roh = out + ".roh"
    cmd = ["python3", TOOL, bett, roh, "--anker", anker] + (["--seiten", seiten] if seiten else []) \
        + (["--praefix"] if praefix else [])
    subprocess.run(cmd, check=True)
    zeilen = open(roh).read().splitlines()
    assert zeilen[0].startswith("seite"), zeilen[0]
    assert zeilen[1].split("\t")[0] == "0"
    d = open(bett, "rb").read()
    r = seite0_textform(d, zonen, ts) if zonen else None
    form = "text form (second zone-3 run)" if r else "earlier form (line in front of the cut)"
    st, ln = r if r else seite0(d, ts)
    anker1 = int(zeilen[2].split("\t")[1])
    assert st < anker1, (st, anker1)
    zeilen[1] = f"0\t-1\t{st}\t{st}\t{ln}"
    n = len(zeilen) - 1
    if not praefix:
        assert n == int(seiten), (n, seiten)
    open(out, "w").write("\n".join(zeilen) + "\n")
    print(f"page 0 ({form}): {st} title {ln} B ({d[st:st+ln]!r}); pages {n}; anchor of page 1 {anker1}")


if __name__ == "__main__":
    if sys.argv[1] == "--selbstprobe":
        d = open(sys.argv[2], "rb").read(5163800)
        st, ln = seite0(d, 5163705)
        print("self-test on the earlier stream:", st, ln, "->", "OK" if (st, ln) == (5163692, 5) else "WRONG")
        sys.exit(0 if (st, ln) == (5163692, 5) else 1)
    main()
