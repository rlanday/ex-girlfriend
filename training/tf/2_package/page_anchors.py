#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Page boundaries on the coder stream, metadata block in front

    python3 page_anchors.py <coder stream> <out.tsv> --anker <hex> [--seiten N]
                            [--anker-wie <stream>] [--praefix]
    python3 page_anchors.py --anker-aus-baum <src/cmix-neifv9>
    python3 page_anchors.py --seiten-zaehlen <enwik9>

Every page ends with exactly one anchor `  </page>\\n` (after WRT), and never
inside text.  The title line of the next page starts directly behind the
anchor and ends at the first 0x0a.  No value of a stream form is written in
here; each comes from the tree or the input:

 * the anchor bytes (`--anker`, hex) are read from the tree with
   `--anker-aus-baum`: the definitions of the page anchor in
   src/neif_autor.h (kAnker), src/neif_textanfang_pack.h (kAnker) and
   src/neif_okb_alias.h (NEIF_OKB_ALIAS_OPEN_INIT) must agree;
 * the page count (`--seiten`) is counted in the input with
   `--seiten-zaehlen`: the number of `</page>` in enwik9 (an invariant of the
   input, 243,425 for enwik9);
 * `--anker-wie <stream>` (the canonical stream of the same bed run, the
   input of every stage that writes the coder stream) must carry as many
   anchors as the input has pages, too: the stages between the two streams
   keep every page anchor.  The tail behind the last anchor (the .intro/.coda
   appendix, WRT coded) must not be empty; whether it equals the tail of that
   stream is reported, not demanded: a stage behind the canonical stream may
   rewrite it (the rule stages work up to the stream end).

Page 0 starts behind the text start, not behind an anchor; its row is left
open here (-1) and `pages.py` sets it from the zone mask of the dump run.  The
last anchor opens no page (behind it stands the tail), so the table has as
many rows as there are anchors, pages 0 to N-1.

Columns (German names kept, the table is read by other tools): seite = page,
anker_off = anchor offset, seiten_start = page start, titel_start/titel_len =
title start/length.  Exit 1 if the anchor count of the stream or of
`--anker-wie` differs from `--seiten`, or the tail is empty.

`--praefix`: the stream is the prefix of a probe bed; the anchor count and the
tail are not checked, and every anchor whose following title line is complete
opens a page (the last page of a prefix has no closing anchor).
"""
import re
import sys


def anker_aus_baum(src):
    """the page anchor of the tree; the three definitions must agree"""
    def lies(rel, rx):
        s = open(f"{src}/{rel}", encoding="latin-1").read()
        m = re.findall(rx, s, re.S)
        if not m:
            sys.exit(f"page_anchors: no page anchor definition in {src}/{rel}")
        return [bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{1,2})", g)) for g in m]
    stellen = {
        "src/neif_autor.h kAnker": lies("src/neif_autor.h", r"kAnker\[8\]\s*=\s*\{([^}]*)\}"),
        "src/neif_textanfang_pack.h kAnker": lies("src/neif_textanfang_pack.h", r"kAnker\[8\]\s*=\s*\{([^}]*)\}"),
        "src/neif_okb_alias.h NEIF_OKB_ALIAS_OPEN_INIT":
            lies("src/neif_okb_alias.h", r"#define\s+NEIF_OKB_ALIAS_OPEN_INIT\s*\{([^}]*)\}"),
    }
    werte = {w for v in stellen.values() for w in v}
    if len(werte) != 1 or len(next(iter(werte))) < 2 or not next(iter(werte)).endswith(b"\n"):
        sys.exit("page_anchors: the page anchor definitions of the tree disagree: "
                 + "; ".join(f"{k} {[w.hex() for w in v]}" for k, v in stellen.items()))
    return next(iter(werte))


def seiten_zaehlen(pfad, nadel=b"</page>"):
    """number of `nadel` in a file (default b'</page>' in the input), read in blocks"""
    n, rest = 0, b""
    with open(pfad, "rb") as f:
        while True:
            blk = f.read(1 << 26)
            if not blk:
                break
            d = rest + blk
            n += d.count(nadel)
            # carry a tail one byte shorter than the needle: a match across
            # two blocks is found once, no match lies inside the tail alone
            rest = d[-(len(nadel) - 1):]
    return n


def letzter_anker(pfad, ank):
    """offset of the last anchor and the bytes behind it, read from the end"""
    with open(pfad, "rb") as f:
        f.seek(0, 2)
        n = f.tell()
        lang = 1 << 22
        while True:
            a = max(0, n - lang)
            f.seek(a)
            d = f.read(n - a)
            i = d.rfind(ank)
            if i >= 0:
                return a + i, d[i + len(ank):]
            if a == 0:
                return -1, b""
            lang <<= 2


def main():
    if sys.argv[1:2] == ["--anker-aus-baum"]:
        print(anker_aus_baum(sys.argv[2]).hex())
        return 0
    if sys.argv[1:2] == ["--seiten-zaehlen"]:
        print(seiten_zaehlen(sys.argv[2]))
        return 0
    args = sys.argv[1:]

    def opt(name, default=None):
        if name in args:
            i = args.index(name)
            v = args[i + 1]
            del args[i:i + 2]
            return v
        return default
    praefix = "--praefix" in args
    if praefix:
        args.remove("--praefix")
    hexa = opt("--anker")
    seiten_soll = opt("--seiten")
    anker_wie = opt("--anker-wie")
    if not hexa:
        sys.exit("page_anchors: --anker <hex> missing (page_anchors.py --anker-aus-baum <src>)")
    ank = bytes.fromhex(hexa)
    P, OUT = args[0], args[1]

    d = open(P, "rb").read()
    N = len(d)
    print(f"# stream: {P}  {N:,} B; anchor {ank.hex()}")
    pos = []
    s = 0
    while True:
        i = d.find(ank, s)
        if i < 0:
            break
        pos.append(i)
        s = i + 1
    print(f"# anchors: {len(pos):,}")
    if not pos:
        print("! NO ANCHOR in the stream")
        return 1
    letztes_ankerende = pos[-1] + len(ank)
    schwanz = d[letztes_ankerende:]
    print(f"# last anchor ends at {letztes_ankerende:,}; tail behind it {len(schwanz):,} B")
    ok = True
    if praefix:
        print("# prefix of a probe bed: anchor count and tail not checked")
    else:
        if seiten_soll is None:
            print("! --seiten missing: the page count of the input is not checked"); ok = False
        elif len(pos) != int(seiten_soll):
            print(f"! ANCHOR COUNT DIFFERS: {len(pos):,} != pages of the input {int(seiten_soll):,}"); ok = False
        if anker_wie:
            n2 = seiten_zaehlen(anker_wie, ank)
            a2, s2 = letzter_anker(anker_wie, ank)
            print(f"# anchors of {anker_wie}: {n2:,}; tail behind its last anchor ({a2:,}) {len(s2):,} B, "
                  + ("the same bytes" if s2 == schwanz else
                     "other bytes than the tail of this stream (a stage between the two streams rewrote it)"))
            if seiten_soll is not None and n2 != int(seiten_soll):
                print(f"! ANCHOR COUNT OF {anker_wie} DIFFERS: {n2:,} != pages of the input {int(seiten_soll):,}"); ok = False
        if not schwanz:
            print("! EMPTY TAIL: the stream ends on an anchor"); ok = False
    print(f"# pages: {len(pos):,} (page 0 is set by pages.py from the zone mask)")
    with open(OUT, "w") as f:
        f.write("seite\tanker_off\tseiten_start\ttitel_start\ttitel_len\n")
        f.write("0\t-1\t-1\t-1\t-1\n")
        for k, a in enumerate(pos if praefix else pos[:-1]):
            st = a + len(ank)
            nl = d.find(b"\n", st)
            if nl < 0:
                break           # prefix: the title line of the last page is cut off
            f.write(f"{k+1}\t{a}\t{st}\t{st}\t{nl-st}\n")
    print(f"# written: {OUT}")
    for k in (1, 100000, len(pos) - 1):
        if 1 <= k <= len(pos) - 1:
            a = pos[k - 1]
            st = a + len(ank)
            nl = d.find(b"\n", st)
            print(f"  page {k}: anchor {a:,}  title {nl-st if nl >= 0 else -1} B")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
