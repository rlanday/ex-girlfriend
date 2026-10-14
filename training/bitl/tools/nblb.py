#!/usr/bin/env python3
"""Weight file of the BITLSTM (NBLB).

Format (native little-endian, uncompressed):
  char[4] "NBLB" | u32 version | i32 cells, nin, inw, n1
  win[cells*inw] bin[cells] wih[4CC] whh[4CC] bg[4C] wout[C] bout
Version 1 stores every value as f32 (what the trainer writes), version 2 as
bf16, the upper 16 bits of the f32 pattern after round-half-even (what the
compressor ships).  The loader expands bf16 exactly by a 16-bit shift.

  python3 nblb.py info <file>            geometry, parameter count, file size
  python3 nblb.py v2 <in.nblb> <out>     f32 (version 1) -> bf16 (version 2)
"""
import struct
import sys

MAGIC = b"NBLB"


def read(path):
    d = open(path, "rb").read()
    assert d[:4] == MAGIC, "not an NBLB file"
    ver, C, nin, inw, n1 = struct.unpack_from("<Iiiii", d, 4)
    assert ver in (1, 2), ver
    off = 24

    def take(n):
        nonlocal off
        if ver == 1:
            v = list(struct.unpack_from("<%df" % n, d, off)); off += 4 * n
        else:
            hs = struct.unpack_from("<%dH" % n, d, off); off += 2 * n
            v = [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in hs]
        return v
    b = dict(cells=C, nin=nin, inw=inw, n1=n1)
    b["win"] = take(C * inw); b["bin"] = take(C)
    b["wih"] = take(4 * C * C); b["whh"] = take(4 * C * C)
    b["bg"] = take(4 * C); b["wout"] = take(C); b["bout"] = take(1)
    assert off == len(d), (off, len(d))
    return b


def write(path, b):
    C, inw = b["cells"], b["inw"]
    out = bytearray(MAGIC)
    out += struct.pack("<Iiiii", 1, C, b["nin"], inw, b["n1"])
    for k in ("win", "bin", "wih", "whh", "bg", "wout", "bout"):
        out += struct.pack("<%df" % len(b[k]), *b[k])
    open(path, "wb").write(bytes(out))
    return len(out)


def write_v2(path, b):
    """Version 2: the same layout, every value as bf16 (upper 16 bits,
    round-half-even)."""
    C, inw = b["cells"], b["inw"]
    out = bytearray(MAGIC)
    out += struct.pack("<Iiiii", 2, C, b["nin"], inw, b["n1"])
    for k in ("win", "bin", "wih", "whh", "bg", "wout", "bout"):
        for x in b[k]:
            out += struct.pack("<I", struct.unpack("<I", struct.pack("<f", bf16(x)))[0])[2:4]
    open(path, "wb").write(bytes(out))
    return len(out)


def bf16(x):
    """Round to bfloat16, round-half-even, as Bf16() in bit_head.cpp."""
    p = struct.unpack("<I", struct.pack("<f", x))[0]
    r = (p >> 16) & 1
    p = (p + 0x7FFF + r) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", p))[0]


def nparams(b):
    C = b["cells"]
    return C * (b["nin"] + 1) + C + 8 * C * C + 4 * C + C + 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "info":
        b = read(sys.argv[2])
        print("cells=%d nin=%d inw=%d n1=%d params=%d file=%d B" %
              (b["cells"], b["nin"], b["inw"], b["n1"], nparams(b),
               len(open(sys.argv[2], 'rb').read())))
    elif len(sys.argv) == 4 and sys.argv[1] == "v2":
        n = write_v2(sys.argv[3], read(sys.argv[2]))
        print("%s -> %s (NBLB v2 / bf16, %d B)" % (sys.argv[2], sys.argv[3], n))
    else:
        sys.exit(__doc__)
