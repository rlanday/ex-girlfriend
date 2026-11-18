#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Writes src/bit_head_weights.h of the package source from three BITLSTM weight files.

    python3 weights_header.py <metablock.nblb> <text1.nblb> <text2.nblb> > bit_head_weights.h

The three weight files (NBLB version 2, bfloat16) must share one geometry.
The header holds the compact slot table that Head::LoadBlob of
src/bit_head.cpp reads: per slot only the live values (cells below H, input
columns that carry a weight; every other value is +0.0), in the order of the
loader loop (input weights column by column, then input bias, recurrent
weights, gate bias, output weights, output bias), as one byte plane of
exponents followed by one of sign and mantissa.  The tool decodes the table
again with the loader loop and stops on any difference; the FNV-1a-64 of the
decoded float32 patterns of all slots goes to stderr and into a comment.

With the three files of bitl/record/ the output is byte for byte the file
src/bit_head_weights.h of this entry.
"""

import hashlib
import struct
import sys

SLOTS = ("metablock", "text 1", "text 2")

HEAD = (
    "// Pretrained weights of the bit-level correction head, compiled into the\n"
    "// binary: the compressor, its helper processes and the decompressor in\n"
    "// archive9 read the same bytes and open no file.  The layout is the one\n"
    "// Head::LoadBlob reads (compact slot table: exponent bytes, then sign and\n"
    "// mantissa bytes of the live values; else the weight file with magic,\n"
    "// version and geometry).\n"
    "#ifndef BIT_HEAD_WEIGHTS_H_\n#define BIT_HEAD_WEIGHTS_H_\n\n"
    "// Generated from the weight file; do not edit.\n")


def read_blob(path):
    data = open(path, "rb").read()
    if len(data) < 24 or data[:4] != b"NBLB":
        sys.stderr.write("%s is not a BITLSTM weight file\n" % path)
        return None
    ver, cells, nin, inw, n1 = struct.unpack("<I4i", data[4:24])
    if ver != 2:
        sys.stderr.write("%s has version %d, expected 2 (bfloat16)\n" % (path, ver))
        return None
    values = cells * inw + cells + 8 * cells * cells + 4 * cells + cells + 1
    if len(data) != 24 + 2 * values:
        sys.stderr.write("%s has %d B, its geometry needs %d B\n" % (path, len(data), 24 + 2 * values))
        return None
    return data, (cells, nin, inw, n1), hashlib.sha256(data).hexdigest()


def byte_array(name, data):
    out = ["static const unsigned char %s[%d] = {\n" % (name, len(data))]
    for i in range(0, len(data), 16):
        out.append("  " + ",".join("0x%02x" % b for b in data[i:i + 16]) + ",\n")
    out.append("};\n")
    return out


def compact(blobs, geo):
    """Compact slot table of the loader loop, or None if the round trip fails."""
    C, nin, inw, n1 = geo
    N = [C * inw, C, 4 * C * C, 4 * C * C, 4 * C, C, 1]
    base = [0]
    for n in N[:-1]:
        base.append(base[-1] + n)
    nbyte = (inw + 7) // 8

    def live(k, i, H, cm):
        # field k, index i: input weights (k = 0) are walked column by column
        q, m = divmod(i, C)
        if k == 0:
            return m * inw + q, m < H and ((cm[q >> 3] >> (q & 7)) & 1) == 1
        return i, (q % C) < H and m < H

    vals, Hs, masks, seqs = [], [], [], []
    for data in blobs:
        v = struct.unpack("<%dH" % sum(N), data[24:])
        win, wih, whh = v[0:N[0]], v[base[2]:base[3]], v[base[3]:base[4]]
        # a cell is live if any value of its row or column is not zero
        cell = [False] * C
        for r in range(C):
            if any(win[r * inw:(r + 1) * inw]) or v[base[1] + r] or v[base[5] + r]:
                cell[r] = True
        for a in range(4 * C):
            if v[base[4] + a]:
                cell[a % C] = True
            for c in range(C):
                if wih[a * C + c] or whh[a * C + c]:
                    cell[a % C] = True
                    cell[c] = True
        H = max([r + 1 for r in range(C) if cell[r]] or [0])
        cm = bytearray(nbyte)
        for j in range(inw):
            if any(win[r * inw + j] for r in range(H)):
                cm[j >> 3] |= 1 << (j & 7)
        seq = []
        for k in range(7):
            for i in range(N[k]):
                idx, lv = live(k, i, H, cm)
                if lv:
                    seq.append(v[base[k] + idx])
        vals.append(v); Hs.append(H); masks.append(bytes(cm)); seqs.append(seq)
    off = [0]
    for s in seqs:
        off.append(off[-1] + len(s))
    allv = [x for s in seqs for x in s]
    n = len(allv)
    W = bytes(((x >> 7) & 0xff) for x in allv) + bytes((((x >> 8) & 0x80) | (x & 0x7f)) for x in allv)
    # round trip with the loader loop: every float32 pattern of every slot
    fnv = 14695981039346656037
    for s in range(len(blobs)):
        e = off[s]
        dec = [None] * sum(N)
        for k in range(7):
            for i in range(N[k]):
                idx, lv = live(k, i, Hs[s], masks[s])
                u = 0
                if lv:
                    sm = W[n + e]
                    u = ((sm & 0x80) << 24) | (W[e] << 23) | ((sm & 0x7f) << 16)
                    e += 1
                dec[base[k] + idx] = u
        if e != off[s + 1]:
            return None
        for a, b in zip(dec, vals[s]):
            if a != (b << 16):
                return None
        for u in dec:
            for b in struct.pack("<I", u):
                fnv = ((fnv ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return W, n, off, Hs, masks, fnv


def main():
    if len(sys.argv) != 4:
        sys.stderr.write(__doc__)
        return 2
    blobs, geo = [], None
    out = [HEAD]
    for k, path in enumerate(sys.argv[1:4]):
        r = read_blob(path)
        if r is None:
            return 1
        data, g, sha = r
        if geo is not None and g != geo:
            sys.stderr.write("slot %d has geometry %s, an earlier slot %s\n" % (k, g, geo))
            return 1
        geo = g
        blobs.append(data)
        out.append("// slot %d (%s): %d B, NBLB version 2, sha256 %s\n" % (k, SLOTS[k], len(data), sha))
    kp = compact(blobs, geo)
    if kp is None:
        sys.stderr.write("the compact table does not reproduce the weight files\n")
        return 1
    W, n, off, Hs, masks, fnv = kp
    sys.stderr.write("compact slot table, %d live values (%d B instead of %d B), cells %s, round trip exact, "
                     "FNV-1a-64 of the weights %016x\n"
                     % (n, len(W), sum(len(b) for b in blobs), "/".join(str(h) for h in Hs), fnv))
    out.append("// geometry of every slot blob (cells/inputs/input width/stage-1 width)\n\n")
    out.append("namespace bit_head {\n")
    out.append("// live values of all slots: exponent bytes, then sign and mantissa bytes;\n"
               "// FNV-1a-64 of the decoded weights %016x\n" % fnv)
    out += byte_array("kBlobW", W)
    out.append("static constexpr unsigned int kBlobN = %d;\n" % n)
    out.append("static constexpr unsigned int kSlotOff[3] = {%s};\n" % ", ".join(str(o) for o in off[:3]))
    out.append("static constexpr unsigned char kSlotH[3] = {%s};\n" % ", ".join(str(h) for h in Hs))
    out.append("static constexpr unsigned char kSlotCols[3][%d] = {%s};\n"
               % (len(masks[0]), ", ".join("{" + ",".join("0x%02x" % b for b in cm) + "}" for cm in masks)))
    out.append("static constexpr unsigned int kBlobTafelN[3] = {%s};\n" % ", ".join(str(len(b)) for b in blobs))
    out.append("}\n\n#endif\n")
    sys.stdout.buffer.write("".join(out).encode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
