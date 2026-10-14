#!/usr/bin/env python3
"""Reader of the coder's row dump, record format v2.

The contract is the record format written by the dump mode that
trainmodes.patch adds to a copy of the entry's source (cmix_dump, cmix_priors).
Summary:

  record 0 (header, recognised by src == -2):
      int32 -2 | 'P' 'K' 'D' | uint8 version | uint8 record size |
      uint8 number of used flag bits | uint16 0
  data records (12 B):
      int32 src_ein | uint8 tok_ein | uint8 prev1 | uint8 prev2 |
      uint8 match_b | uint8 match_cls | uint8 flags | uint16 reserved

  flags: bit0 inserted · bit1 MASKED (loader column maske = NOT bit1) ·
         bit2 document start · bit3 redirect page detected · bit4 prior0 row ·
         bit5 reinjected alias or author byte · bit6 author byte ·
         bit7 metadata prompt row

Usage:
    dumpleser.py <file> [--kopf] [--zeilen N] [--zaehlen] [--dokumente]

  --kopf       report the header record only
  --zeilen N   print the first N data records readably
  --zaehlen    counts over the whole file (rows, per bit, documents)
  --dokumente  length of every document (rows): the first 40 and the sum

As a module: `for r in saetze(path): ...` yields tuples
    (src, tok, prev1, prev2, match_b, match_cls, flags)
(German names: Satz = record, Kopf = header, Zeile = row, Dokument = document.)
"""
import struct
import sys

REC = 12
MAGIC = -2
BIT = {
    0: "inserted",
    1: "masked",
    2: "doc_start",
    3: "redirect",
    4: "prior0",
    5: "alias_author",
    6: "author",
    7: "meta",
}


def kopf(pfad):
    """(version, record size, bits), or None if the file has no header"""
    with open(pfad, "rb") as f:
        b = f.read(REC)
    if len(b) < REC:
        raise SystemExit("file shorter than one record: " + pfad)
    src, = struct.unpack_from("<i", b, 0)
    if src != MAGIC:
        return None
    if b[4:7] != b"PKD":
        raise SystemExit("record 0 has src=-2 but no PKD tag; "
                         "unknown format")
    return (b[7], b[8], b[9])


def saetze(pfad):
    """all data records (the header record is skipped)"""
    k = kopf(pfad)
    with open(pfad, "rb") as f:
        if k is not None:
            f.read(REC)
        while True:
            blk = f.read(REC * 8192)
            if not blk:
                return
            for i in range(0, len(blk) - REC + 1, REC):
                src, = struct.unpack_from("<i", blk, i)
                tok, p1, p2, mb, mc, fl = struct.unpack_from("<6B", blk, i + 4)
                yield (src, tok, p1, p2, mb, mc, fl)


def flagtext(fl):
    an = [BIT[b] for b in sorted(BIT) if fl & (1 << b)]
    return ",".join(an) if an else "-"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pfad = sys.argv[1]
    opt = sys.argv[2:]
    k = kopf(pfad)
    if k is None:
        print("HEADER RECORD: none; contract v1 (only bit0/bit1 used)")
    else:
        print("HEADER RECORD: version %d, record size %d B, used flag bits %d"
              % k)
    if "--kopf" in opt:
        return

    if "--zeilen" in opt:
        n = int(opt[opt.index("--zeilen") + 1])
        print("%10s %5s %5s %5s %5s %4s  %s"
              % ("src", "tok", "prev1", "prev2", "mb", "mcl", "flags"))
        for i, r in enumerate(saetze(pfad)):
            if i >= n:
                break
            print("%10d %5d %5d %5d %5d %4d  %s"
                  % (r[0], r[1], r[2], r[3], r[4], r[5], flagtext(r[6])))

    if "--zaehlen" in opt or "--dokumente" in opt:
        n = 0
        cnt = {b: 0 for b in BIT}
        strom = 0
        doklen = []
        akt = 0
        for r in saetze(pfad):
            n += 1
            if r[0] >= 0:
                strom += 1
            fl = r[6]
            for b in BIT:
                if fl & (1 << b):
                    cnt[b] += 1
            if fl & 4:
                if akt:
                    doklen.append(akt)
                akt = 1
            else:
                akt += 1
        if akt:
            doklen.append(akt)
        if "--zaehlen" in opt:
            print("records total      : %d" % n)
            print("of them stream rows: %d" % strom)
            for b in sorted(BIT):
                print("  bit%d %-12s: %d" % (b, BIT[b], cnt[b]))
            print("documents (bit2)   : %d" % cnt[2])
        if "--dokumente" in opt:
            print("documents: %d, rows total %d, median %d, longest %d"
                  % (len(doklen), sum(doklen),
                     sorted(doklen)[len(doklen) // 2] if doklen else 0,
                     max(doklen) if doklen else 0))
            print("first 40 lengths: %s"
                  % ",".join(str(x) for x in doklen[:40]))


if __name__ == "__main__":
    main()
