#!/usr/bin/env python3
"""Prints "record_bytes header_bytes columns n1" of a trace file (format in
../FORMAT.md).  The record length is read from the file, never assumed.
  python3 trace_geom.py <trace>
"""
import struct
import sys

with open(sys.argv[1], "rb") as f:
    magic, ver, n1, rec, twin = struct.unpack("<5I", f.read(20))
    f.read(16 * twin)
    tcols = struct.unpack("<Q", f.read(8))[0] if ver >= 3 else 0
    head = f.tell()
if magic != 0x4c544942: sys.exit("not a trace file (magic)")
ncol = bin(tcols).count("1") if tcols else n1
print("%d %d %d %d" % (rec, head, ncol, n1))
