#!/usr/bin/env python3
# Appends new words to a dictionary whose word set is otherwise fixed (by
# default the first 365 candidates that are not in it yet).
#   combine.py <base.dic> <candidates.txt> <out.dic> [N] [--order alpha|...]
# <candidates.txt>: one "<word>\t<gain>" per line (output of make_w4.py)
import sys


def parse_dic(p):
    words, cur = [], bytearray()
    for c in open(p, "rb").read():
        if 97 <= c <= 122:
            cur.append(c)
        elif cur:
            words.append(cur.decode("ascii")); cur = bytearray()
    if cur:
        words.append(cur.decode("ascii"))
    return words


def main():
    base, cands, out = sys.argv[1:4]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 365
    order = sys.argv[sys.argv.index("--order") + 1] if "--order" in sys.argv else "alpha"
    dic = parse_dic(base)
    have = set(dic)
    new = []
    for ln in open(cands, encoding="utf-8"):
        w = ln.split("\t")[0].strip()
        if w and w not in have:
            new.append(w)
        if len(new) >= N:
            break
    dic += (sorted(new) if order == "alpha" else new)
    open(out, "wb").write(("\n".join(dic) + "\n").encode("ascii"))
    print("%s: %d lines (+%d)" % (out, len(dic), len(new)))


if __name__ == "__main__":
    main()
