#!/usr/bin/env python3
"""Hint package (the older 6-block format) that derives cb_cbow_w4e.dic
byte-exactly from enwik9.

Changed against the earlier builder for english.dic (44,515 words):
  * candidate set M = 50,000 instead of M = N (a cheaper insertion list H2
    against a dearer deletion list H1; the optimum was measured with cmix)
  * H5 (G2 ranks) in suffix order (words sorted by their reversed string),
    2 bytes big endian
  * H6 (G3 run ids) in suffix order, 2 bytes big endian
Both sides can compute every order (the decoder knows the word set once H1/H2
are applied), so the orders cost nothing.

Stream layout (each block varint(len) + payload):
  H1 deletions      indices of the surplus candidates (delta varint)
  H2 insertions     position in the final sorted list (delta varint)
                    + LCP byte + suffix + 0x00
  H3 group flags    one byte per word of the final sorted list (0=G1,1=G2,2=G3)
  H4 G1 ranks       80 bytes (position inside G1)
  H5 G2 ranks       3840 x 2 bytes BE, words in suffix order
  H6 G3 run ids     40960 x 2 bytes BE, words in suffix order
"""
import argparse

COUNTS = '${WORKDIR}/b6hints/counts_100m.tsv'
G1N, G2N = 80, 3840
DEFAULT_M = 50000


def varint(x):
    out = bytearray()
    while True:
        b = x & 0x7F; x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b); return bytes(out)


def read_varints(data, pos, n):
    vals = []
    for _ in range(n):
        x = 0; sh = 0
        while True:
            b = data[pos]; pos += 1
            x |= (b & 0x7F) << sh
            if not (b & 0x80):
                break
            sh += 7
        vals.append(x)
    return vals, pos


def derive_candidates(M):
    scored = []
    for line in open(COUNTS):
        w, c = line.rstrip('\n').split('\t')
        scored.append((-((int(c) - 2) * (len(w) - 3)), w))
    scored.sort()
    return sorted(w for _, w in scored[:M])


def suffix_order(words):
    return sorted(words, key=lambda w: w[::-1])


def build(ref_path, M, out_path):
    ref_words = [w for w in open(ref_path).read().split('\n') if w]
    N = len(ref_words)
    assert len(set(ref_words)) == N
    ref_sorted = sorted(ref_words)
    cand = derive_candidates(M)
    cset, rset = set(cand), set(ref_sorted)

    dels = [i for i, w in enumerate(cand) if w not in rset]
    h1 = varint(len(dels)); prev = 0
    for i in dels:
        h1 += varint(i - prev); prev = i

    adds = [(i, w) for i, w in enumerate(ref_sorted) if w not in cset]
    h2 = varint(len(adds)); prev = 0
    for i, w in adds:
        h2 += varint(i - prev); prev = i
        p = ref_sorted[i - 1] if i > 0 else ''
        l = 0
        while l < len(p) and l < len(w) and p[l] == w[l]:
            l += 1
        h2 += bytes([l]) + w[l:].encode() + b'\x00'

    g1s, g2s = set(ref_words[:G1N]), set(ref_words[G1N:G1N + G2N])
    h3 = bytes(0 if w in g1s else 1 if w in g2s else 2 for w in ref_sorted)
    h4 = bytes({w: i for i, w in enumerate(ref_words[:G1N])}[w] for w in sorted(g1s))
    pos2 = {w: i for i, w in enumerate(ref_words[G1N:G1N + G2N])}
    h5 = b''.join(pos2[w].to_bytes(2, 'big') for w in suffix_order(g2s))

    g3 = ref_words[G1N + G2N:]
    runid = {g3[0]: 0}; rid = 0
    for a, b in zip(g3, g3[1:]):
        if b < a:
            rid += 1
        runid[b] = rid
    h6 = b''.join(runid[w].to_bytes(2, 'big') for w in suffix_order(g3))

    blob = b''.join(varint(len(x)) + x for x in (h1, h2, h3, h4, h5, h6))
    open(out_path, 'wb').write(blob)
    print('ref=%s N=%d M=%d runs=%d' % (ref_path.split('/')[-1], N, M, rid + 1))
    print('  H1 %d B (del=%d)  H2 %d B (add=%d)  H3 %d B  H4 %d B  H5 %d B  H6 %d B  total %d B'
          % (len(h1), len(dels), len(h2), len(adds), len(h3), len(h4), len(h5), len(h6), len(blob)))
    return blob


def reconstruct(blob, M):
    parts = []; pos = 0
    for _ in range(6):
        (ln,), pos = read_varints(blob, pos, 1)
        parts.append(blob[pos:pos + ln]); pos += ln
    h1, h2, h3, h4, h5, h6 = parts
    cand = derive_candidates(M)
    (nd,), p = read_varints(h1, 0, 1)
    dels, p = read_varints(h1, p, nd)
    acc = 0; dset = set()
    for d in dels:
        acc += d; dset.add(acc)
    words = [w for i, w in enumerate(cand) if i not in dset]
    (na,), p = read_varints(h2, 0, 1)
    acc = 0
    for _ in range(na):
        (d,), p = read_varints(h2, p, 1)
        acc += d
        lcp = h2[p]; p += 1
        e = h2.index(b'\x00', p)
        suf = h2[p:e].decode(); p = e + 1
        prevw = words[acc - 1] if acc > 0 else ''
        words.insert(acc, prevw[:lcp] + suf)
    ref_sorted = words
    sg1 = [w for w, g in zip(ref_sorted, h3) if g == 0]
    sg2 = [w for w, g in zip(ref_sorted, h3) if g == 1]
    sg3 = [w for w, g in zip(ref_sorted, h3) if g == 2]
    g1 = [None] * len(sg1)
    for w, i in zip(sg1, h4):
        g1[i] = w
    g2 = [None] * len(sg2)
    for j, w in enumerate(suffix_order(sg2)):
        g2[(h5[2 * j] << 8) | h5[2 * j + 1]] = w
    ord3 = suffix_order(sg3)
    ids = [(h6[2 * j] << 8) | h6[2 * j + 1] for j in range(len(sg3))]
    runs = [[] for _ in range(max(ids) + 1)]
    for w, r in zip(ord3, ids):
        runs[r].append(w)
    g3 = [w for r in runs for w in sorted(r)]
    return '\n'.join(g1 + g2 + g3) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='${WORKDIR}/b6hints/dicts/cb_cbow_w4e.dic')
    ap.add_argument('--cand', type=int, default=DEFAULT_M)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    blob = build(a.ref, a.cand, a.out)
    rec = reconstruct(blob, a.cand)
    ref = open(a.ref).read()
    if not ref.endswith('\n'):
        ref += '\n'
    print('  reconstruction byte-exact:', rec == ref)
