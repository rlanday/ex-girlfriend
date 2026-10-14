#!/usr/bin/env python3
"""Value statistics of BITLSTM weight files (NBLB version 1 or 2).
  python3 blobstat.py <file> [<file> ...]
"""
import math
import struct
import sys


def load(p):
    with open(p, "rb") as f:
        mg = f.read(4); ver = struct.unpack("<I", f.read(4))[0]
        c, nin, inw, n1 = struct.unpack("<4i", f.read(16))
        rest = f.read()
    if ver == 2:
        n = len(rest) // 2
        vals = [struct.unpack("<f", struct.pack("<HH", 0, struct.unpack_from("<H", rest, 2 * i)[0]))[0] for i in range(n)]
    else:
        n = len(rest) // 4
        vals = list(struct.unpack("<%df" % n, rest[:4 * n]))
    return (c, nin, inw, n1, ver, vals)


for p in sys.argv[1:]:
    c, nin, inw, n1, ver, v = load(p)
    nz = sum(1 for x in v if x != 0.0)
    mx = max(abs(x) for x in v) if v else 0.0
    rms = math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0
    # win is the first block: C*inw values
    nw = c * inw
    win = v[:nw]
    wr = math.sqrt(sum(x * x for x in win) / len(win)) if win else 0.0
    wmx = max(abs(x) for x in win) if win else 0.0
    print("%-46s ver=%d C=%d nin=%d  values %d, of which %d != 0 (%.1f %%)" % (p.split("/")[-1], ver, c, nin, len(v), nz, 100.0 * nz / len(v)))
    print("    all:   max|w| %.6f  rms %.6f" % (mx, rms))
    print("    win:   max|w| %.6f  rms %.6f   (input weights, %d values)" % (wmx, wr, nw))
