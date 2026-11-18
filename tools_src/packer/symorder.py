#!/usr/bin/env python3
"""symorder.py - function order for the second link.

  symorder.py <ximg> <linker map> > symbol_order.txt

<ximg> is the sstripped image of the FIRST link after the packer's address
transform (cmcut_tool xform), <linker map> the -Wl,-Map of that link (the map
of the shipped binary, not the one of cmix_aux).  The input sections of the
output section .text are cut out of the image.  The hot sections of the
profile (.text.hot.*) stay first in their order; the others are chained
greedily: start with the first one, then always the not yet placed section
that shares the most distinct 4-byte strings with the last placed one (ties:
the smaller index).  The packer's context-mixing model then sees similar code next to
each other.  Only .text is ordered; ordering the encoder-only section as
well was measured worse.  Output: one symbol per line in the new order, for
lld -Wl,--symbol-ordering-file (with --no-warn-symbol-ordering).
Deterministic (no hash order is used for decisions); standard library only."""
import re
import struct
import sys


def main():
    ximg, mapf = sys.argv[1:3]
    img = open(ximg, 'rb').read()
    phoff = struct.unpack_from('<Q', img, 32)[0]
    phn = struct.unpack_from('<H', img, 56)[0]
    loads = []
    for j in range(phn):
        t, fl, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', img, phoff + 56 * j)
        if t == 1 and fs:
            loads.append((va, va + fs, va - off))
    lines = open(mapf).read().split('\n')
    secs = []          # [vaddr, name, first symbol]
    tstart = tsize = None
    intext = False
    for i, line in enumerate(lines):
        f = line.split()
        if len(f) == 5 and f[4] == '.text':
            intext = True
            tstart, tsize = int(f[0], 16), int(f[2], 16)
            continue
        if intext and len(f) == 5 and f[4].startswith('.') and ':(' not in f[4]:
            break
        if not intext:
            continue
        m = re.match(r'.*:\((\.text[^)]*)\)', line)
        if m:
            sym = None
            if i + 1 < len(lines):
                g = lines[i + 1].split(None, 4)
                if len(g) == 5 and ':(' not in g[4]:
                    sym = g[4].strip()
            secs.append([int(f[0], 16), m.group(1), sym])
    if not secs:
        sys.exit('symorder: no .text input sections in the map')
    secs.sort(key=lambda s: s[0])
    d = None
    for lo, hi, dd in loads:
        if lo <= tstart < hi:
            d = dd
    if d is None:
        sys.exit('symorder: .text is not in a PT_LOAD of the image')
    blocks = []
    for i, (a, name, sym) in enumerate(secs):
        end = secs[i + 1][0] if i + 1 < len(secs) else tstart + tsize
        blocks.append(img[a - d:end - d])
    grams = [sorted(set(b[k:k + 4] for k in range(max(0, len(b) - 3)))) for b in blocks]
    index = {}
    for i, g in enumerate(grams):
        for x in g:
            index.setdefault(x, []).append(i)
    # The sections that the profile marks hot (.text.hot.*) keep their place
    # and their order: they come first, as the first link has them.  Only the
    # rest is chained.  Chaining all of them made the metablock part of the
    # coder 3.3 % slower (hot code spread over the whole .text).
    hot = [i for i in range(len(secs)) if secs[i][1].startswith('.text.hot.')]
    rest = [i for i in range(len(secs)) if not secs[i][1].startswith('.text.hot.')]
    left = set(rest[1:])
    order = hot + rest[:1]
    cur = rest[0]
    while left:
        cnt = {}
        for x in grams[cur]:
            for j in index[x]:
                if j in left:
                    cnt[j] = cnt.get(j, 0) + 1
        if cnt:
            best = min(cnt, key=lambda j: (-cnt[j], j))
        else:
            best = min(left)
        order.append(best)
        left.remove(best)
        cur = best
    # section name -> link symbol (mangled suffix of .text.<name>, else the first symbol)
    out, used = [], set()
    for i in order:
        name, sym = secs[i][1], secs[i][2]
        suf = re.sub(r'^\.text\.(hot\.|unlikely\.|startup\.|exit\.|split\.)?', '', name)
        if suf and suf not in ('startup', 'exit', 'hot', 'unlikely', '.text') and name != '.text':
            s = suf
        elif sym and re.match(r'^[A-Za-z_][A-Za-z0-9_.$]*$', sym):
            s = sym
        else:
            continue
        if s not in used:
            used.add(s)
            out.append(s)
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
