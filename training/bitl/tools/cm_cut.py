#!/usr/bin/env python3
# cm_cut.py <spur> <kanal> <r0> <r1> <out> <cols|all> [nohid]
# trace of the compressor (v2, 20-byte header, records 8 + 4*n1) + channel (BAZ1: n_tf | hid | lhid | zone) -> packed store
# "p16" (bg_kern.StoreP): rec.npy int16 [N, 3 + ncol] = base_p bits, flags, m1raw f16, the chosen
# stage-1 columns f16 (round half even); hid.npy float16 [Nb, nhid] (nohid: one zero column, the caller must zero
# the transformer block); meta.txt.  Reads only (memmap), one thread.
import sys, os, time, struct
import numpy as np

spur, kanal, r0, r1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
colarg = sys.argv[6]
nohid = len(sys.argv) > 7 and sys.argv[7] == 'nohid'
t0 = time.time()
sp = np.memmap(spur, np.uint8, 'r')
mg, ver, n1, rec, nw = struct.unpack('<5I', bytes(sp[:20]))
assert mg == 0x4c544942 and ver == 2 and nw == 0 and rec == 8 + 4 * n1, (mg, ver, n1, rec, nw)
cols = list(range(n1)) if colarg == 'all' else [int(c) for c in colarg.split(',')]
ns = (len(sp) - 20) // rec
r1 = min(r1, ns)
N = r1 - r0
os.makedirs(out, exist_ok=True)
R = np.lib.format.open_memmap(os.path.join(out, 'rec.npy'), 'w+', np.int16, (N, 3 + len(cols)))
ci = np.array(cols)
CH = 1 << 21
for a in range(0, N, CH):
    b = min(N, a + CH)
    rr = np.asarray(sp[20 + (r0 + a) * rec:20 + (r0 + b) * rec]).reshape(b - a, rec)
    o = np.empty((b - a, 3 + len(cols)), np.int16)
    o[:, 0] = rr[:, 0:2].copy().view('<i2')[:, 0]
    o[:, 1] = rr[:, 2]
    o[:, 2] = rr[:, 4:8].copy().view('<f4')[:, 0].astype(np.float16).view(np.int16)
    o[:, 3:] = rr[:, 8:].copy().view('<f4')[:, ci].astype(np.float16).view(np.int16)
    R[a:b] = o
R.flush()
del R
ka = np.memmap(kanal, np.uint8, 'r')
hh = struct.unpack('<7I', bytes(ka[:28]))
assert hh[0] == 0x315a4142, hh
ntf, nhid = hh[1], hh[2]
wk = hh[1] + hh[2] + hh[4] + (1 if hh[6] > 0 else 0)
e0, e1 = r0 >> 3, ((r1 - 1) >> 3) + 1
nk = (len(ka) - 28) // (4 * wk)
assert e1 <= nk, (e1, nk)
H = np.lib.format.open_memmap(os.path.join(out, 'hid.npy'), 'w+', np.float16, (e1 - e0, 1 if nohid else nhid))
zmin, zmax = 99.0, -1.0
for a in range(e0, e1, CH):
    b = min(e1, a + CH)
    kk = np.asarray(ka[28 + a * wk * 4:28 + b * wk * 4]).copy().view('<f4').reshape(b - a, wk)
    H[a - e0:b - e0] = 0 if nohid else kk[:, ntf:ntf + nhid].astype(np.float16)
    if hh[6] > 0:
        zmin = min(zmin, float(kk[:, wk - 1].min())); zmax = max(zmax, float(kk[:, wk - 1].max()))
H.flush()
with open(os.path.join(out, 'meta.txt'), 'w') as f:
    f.write('r0=%d r1=%d cols=%s nhid=%d format=p16 zone_min=%g zone_max=%g nohid=%d n1=%d\n' % (
        r0, r1, ','.join(map(str, cols)), 1 if nohid else nhid, zmin, zmax, int(nohid), n1))
print('fertig %.0fs' % (time.time() - t0), open(os.path.join(out, 'meta.txt')).read().strip())
