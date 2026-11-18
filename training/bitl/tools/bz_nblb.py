#!/usr/bin/env python3
# bz_nblb.py: NBLB weight blobs of the BITLSTM head (numpy).
#   info <blob>                          geometry, file size, packed sizes (xz -9e, lzma alone)
#   v2 <in> <out> [k]                    bf16 (NBLB v2), mantissa rounded to k bits (7 = plain bf16), RNE
#   aufweiten <in> <out> <ref_F> <keep_M> <keep_F> <nhid_M>   blob of a narrower form (stage-1 mask keep_M,
#                                        nhid_M TF columns) widened to the full form of <ref_F> (same cells): zero
#                                        weights on the columns the narrow form does not read; the same model
#   leer <ref> <out>                     all weights 0 (control: the head then adds nothing)
# Layout of the input row (src/bit_head.cpp Head): [stage-1 kept cols][m1raw t t2 sgn bitpos x8][sse][prev]
# [hid][tf][lhid]; win is cells x inw row-major.
import sys, struct, lzma, subprocess, os
import numpy as np

def read(p):
    d = open(p, 'rb').read()
    assert d[:4] == b'NBLB', p
    ver, C, nin, inw, n1 = struct.unpack_from('<Iiiii', d, 4)
    assert ver in (1, 2), ver
    sizes = [C * inw, C, 4 * C * C, 4 * C * C, 4 * C, C, 1]
    n = sum(sizes)
    if ver == 1:
        v = np.frombuffer(d, dtype='<f4', count=n, offset=24).astype(np.float32)
        assert 24 + 4 * n == len(d)
    else:
        h = np.frombuffer(d, dtype='<u2', count=n, offset=24).astype(np.uint32) << 16
        v = h.view(np.float32)
        assert 24 + 2 * n == len(d)
    return dict(C=C, nin=nin, inw=inw, n1=n1, v=v.copy(), sizes=sizes)

def split(b):
    out, o = [], 0
    for s in b['sizes']:
        out.append(b['v'][o:o + s]); o += s
    return out

def write(p, b, ver, k=7):
    hdr = b'NBLB' + struct.pack('<Iiiii', ver, b['C'], b['nin'], b['inw'], b['n1'])
    if ver == 1:
        body = b['v'].astype('<f4').tobytes()
    else:
        u = b['v'].astype(np.float32).view(np.uint32).astype(np.uint64)
        h = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16)            # bf16, round half even
        s = 7 - k
        if s > 0:
            h = ((h + (1 << (s - 1)) - 1 + ((h >> s) & 1)) >> s) << s
        body = h.astype('<u2').tobytes()
    open(p, 'wb').write(hdr + body)
    return len(hdr) + len(body)

def packed(p):
    d = open(p, 'rb').read()
    xz = subprocess.run(['xz', '-9e', '-c'], input=d, capture_output=True).stdout
    lz = lzma.compress(d, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    return len(d), len(xz), len(lz)

def layout(n1keep, nhid, nlhid, nsse):
    base = n1keep
    prev = n1keep + 12 + nsse
    hid = prev + 1
    lhid = hid + nhid
    return base, prev, hid, lhid

def main():
    c = sys.argv[1]
    if c == 'info':
        b = read(sys.argv[2])
        r, x, l = packed(sys.argv[2])
        print(f"{sys.argv[2]}: C={b['C']} nin={b['nin']} inw={b['inw']} n1={b['n1']} werte={len(b['v'])} "
              f"datei={r} xz9e={x} lzma={l} max|w|={np.abs(b['v']).max():.4g}")
    elif c == 'v2':
        b = read(sys.argv[2]); k = int(sys.argv[4]) if len(sys.argv) > 4 else 7
        n = write(sys.argv[3], b, 2, k)
        print(f'{sys.argv[3]}: v2 k={k} {n} B')
    elif c == 'aufweiten':
        # aufweiten <in> <out> <ref_F> <keep_M> <keep_F> <nhid_M> [nlhid (both forms, 32)] [nhid_F (32)]
        m = read(sys.argv[2]); f = read(sys.argv[4])
        km, kf, nhm = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        nl = int(sys.argv[8]) if len(sys.argv) > 8 else 32
        nhf = int(sys.argv[9]) if len(sys.argv) > 9 else 32
        assert m['C'] == f['C'] and m['n1'] == f['n1']
        C = m['C']
        cm = [j for j in range(m['n1']) if km >> j & 1]; cf = [j for j in range(f['n1']) if kf >> j & 1]
        assert set(cm) <= set(cf), 'metablock columns must be a subset'
        nsse_m = m['nin'] - (len(cm) + 12 + nhm + nl)
        nsse_f = f['nin'] - (len(cf) + 12 + nhf + nl)
        assert nsse_m == nsse_f and nsse_m in (0, 2), (nsse_m, nsse_f)
        nsse = nsse_m
        bm, pm, hm, lm = layout(len(cm), nhm, nl, nsse)
        bf, pf, hf, lf = layout(len(cf), nhf, nl, nsse)
        mp = {}
        for i, j in enumerate(cm): mp[i] = cf.index(j)           # stage-1 column j
        for j in range(12 + nsse): mp[bm + j] = bf + j          # m1raw, t, t2, sgn, bitpos, sse
        mp[pm] = pf                                             # prev
        for j in range(nhm): mp[hm + j] = hf + j                # hid (if the metablock form keeps it)
        for j in range(nl): mp[lm + j] = lf + j                 # lhid
        assert len(mp) == m['nin'] + 1, (len(mp), m['nin'])
        pm_ = split(m); pf_ = split(f)
        win = np.zeros((C, f['inw']), np.float32)
        wm = pm_[0].reshape(C, m['inw'])
        for a, z in mp.items(): win[:, z] = wm[:, a]
        out = dict(C=C, nin=f['nin'], inw=f['inw'], n1=f['n1'], sizes=f['sizes'])
        out['v'] = np.concatenate([win.ravel()] + pm_[1:]).astype(np.float32)
        n = write(sys.argv[3], out, 1)
        print(f"{sys.argv[3]}: aufgeweitet {m['nin']}/{m['inw']} -> {f['nin']}/{f['inw']}, {n} B, nsse={nsse}")
    elif c == 'zufall':
        # zufall <trained> <out> [seed]: control blob of the same geometry, every array drawn from a normal
        # distribution with that array's mean and sd; exact zeros (columns of an aufgeweitet blob) stay zero
        b = read(sys.argv[2]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 12345
        rng = np.random.default_rng(seed)
        parts = split(b); out = []
        for a in parts:
            nz = a != 0
            r = np.zeros_like(a)
            if nz.any():
                r[nz] = rng.normal(float(a[nz].mean()), float(a[nz].std()) if nz.sum() > 1 else 0.0, int(nz.sum()))
            out.append(r.astype(np.float32))
        b['v'] = np.concatenate(out)
        print(f"{sys.argv[3]}: {write(sys.argv[3], b, 1)} B zufall seed {seed}")
    elif c == 'leer':
        b = read(sys.argv[2]); b['v'][:] = 0
        print(f"{sys.argv[3]}: {write(sys.argv[3], b, 1)} B leer")
    else:
        print(__doc__ or 'info|v2|aufweiten|leer'); sys.exit(2)

if __name__ == '__main__':
    main()
