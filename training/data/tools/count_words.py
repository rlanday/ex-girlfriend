#!/usr/bin/env python3
"""Word count over enwik9 with the exact WRT tokenisation of dictionary.cpp.

Token rule (Dictionary::Encode): [A-Z]?[a-z]+ | [A-Z]+ (greedy, in this
order); words are case-folded.  Snapshots of the count after 20/50/100 MB and
at the end (1 GB).  Output: TSV word<TAB>count (count >= 2), sorted by
descending frequency, ties alphabetically.
"""
import re, sys, collections

SRC = '${DATA}/enwik9'
OUT = '${WORKDIR}/s9/counts'
SNAPS = [20_000_000, 50_000_000, 100_000_000, 1_000_000_000]
TOK = re.compile(rb'[A-Z]?[a-z]+|[A-Z]+')

import os
os.makedirs(OUT, exist_ok=True)
counts = collections.Counter()
carry = b''
pos = 0
snap_iter = iter(SNAPS)
next_snap = next(snap_iter)

def dump(n, counts):
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(f'{OUT}/counts_{n//1_000_000}m.tsv', 'w') as f:
        for w, c in items:
            if c >= 2:
                f.write(f'{w.decode()}\t{c}\n')
    print(f'snapshot {n}: {len(items)} distinct', flush=True)

with open(SRC, 'rb') as f:
    while True:
        want = min(1 << 24, next_snap - pos) if next_snap else (1 << 24)
        chunk = f.read(want)
        if not chunk:
            break
        buf = carry + chunk
        pos += len(chunk)
        at_snap = (pos == next_snap)
        if at_snap:
            # snapshot boundary: count exactly up to byte N, no carry across it
            toks = TOK.findall(buf)
            carry = b''
        else:
            # hold back the last (possibly incomplete) run of letters
            m = re.search(rb'[A-Za-z]+$', buf)
            if m and m.start() > 0:
                carry = buf[m.start():]
                buf = buf[:m.start()]
            elif m:
                carry = buf
                buf = b''
            else:
                carry = b''
            toks = TOK.findall(buf)
        for t in toks:
            counts[t.lower()] += 1
        if at_snap:
            dump(pos, counts)
            try:
                next_snap = next(snap_iter)
            except StopIteration:
                next_snap = None
if carry:
    for t in TOK.findall(carry):
        counts[t.lower()] += 1
if pos not in SNAPS:
    dump(pos, counts)
print('done', pos, flush=True)
