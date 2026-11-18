#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Checks the prior0 table of the package source against a training package.

    python3 prior0_check.py <src/tf1/tf1_prior0_set_v14.h> [<z3-prior0.float16>]

The header carries the replacement prior of inserted rows as 208 float16
values (hexadecimal) and the sha256 of their little-endian bytes
(PRIOR0_SHA256).  The tool packs the 208 values, checks them against that
sha256 and, given the table of a training package (416 B), against the file.
Exit 0 = all equal, 1 = a difference, 2 = unreadable input.
"""
import hashlib
import re
import struct
import sys


def main():
    if len(sys.argv) not in (2, 3):
        sys.stderr.write(__doc__)
        return 2
    text = open(sys.argv[1], encoding='utf-8').read()
    m = re.search(r'#define\s+PRIOR0_SHA256\s+"([0-9a-f]{64})"', text)
    vals = [int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]{4})\b', text)]
    if not m or len(vals) != 208:
        sys.stderr.write('header: PRIOR0_SHA256 or 208 values not found (%d values)\n' % len(vals))
        return 2
    table = struct.pack('<208H', *vals)
    sha = hashlib.sha256(table).hexdigest()
    ok = sha == m.group(1)
    print('header table %s, stated %s: %s' % (sha, m.group(1), 'EQUAL' if ok else 'DIFFERENT'))
    if len(sys.argv) == 3:
        data = open(sys.argv[2], 'rb').read()
        same = data == table
        print('package table %d B, sha256 %s: %s' % (len(data), hashlib.sha256(data).hexdigest(),
                                                    'EQUAL' if same else 'DIFFERENT'))
        ok = ok and same
    return 0 if ok else 1


sys.exit(main())
