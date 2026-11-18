#!/usr/bin/env python3
# tb_cut.py: packed store "pk1" of a long text zone of the BITLSTM trace (v2: 20-byte header,
# records 164 B = u16 base_p, u8 flags, u8 0, f32 m1raw, f32 x 39 stage-1 columns; channel BAZ1: 28-byte header,
# rows of 37 f32 = tf 4 | sketch 32 | zone, one row per 8 records), built from a STREAM (stdin), so the raw trace
# never lands on a disk.  Lossless against the p16 store of cm_cut.py (same f16 rounding, round half even):
#   rec.raw int16 [N, 29] for absolute records [r0, r1):
#     0 base_p bits | 1 packed: bit0 override, bit1 coded bit, bits 2-4 bit position = clamp(trunc(f16(col 23)), 0, 7),
#     bits 5-14 columns 24..33 (each exactly 0 or 1) | 2 m1raw f16 | 3 columns 34 (bits 0-1) and 38 (bits 2-3) as
#     value + 1 (each exactly -1, 0 or 1) | 4..28 f16 of columns 1..22, 35, 36, 37
#   hid.raw float16 [Nh, 32]: sketch of channel rows r0>>3 .. (r1-1)>>3
#   meta.txt (format=pk1 r0 r1 n1 nhid fcols bitcols terncols)
# Every value that would not survive the packing is COUNTED (bad_*); the zone is only usable with all of them 0.
#   tb_cut.py init  <dir> <r0> <r1>                        allocate rec.raw / hid.raw (sparse), meta.txt
#   tb_cut.py spur  <dir> <A> <ra> <rb>                    stdin = spur.bin bytes from offset A (A <= 20 + 164 ra);
#                                                           writes records [ra, rb); reads stdin to EOF
#   tb_cut.py kanal <dir> <A> <ea> <eb>                    stdin = kanal.bin bytes from offset A (A <= 28 + 148 ea)
#   tb_cut.py file  <dir> <spur|kanal> <path> <A> <ra> <rb> same from a file holding the bytes from offset A (tests)
# Output: <dir>/seg/<kind>_<ra>.json = sha256 and byte count of ALL stdin bytes, rows written, census (min, max, sum,
# sum of squares per column, ones, overrides, bad counts; channel: zone runs).  One thread (numpy), memory ~1 GB.
import sys, os, json, hashlib, time
import numpy as np

REC = 164; KREC = 148; NCOL = 39; NHID = 32; WK = 37
FCOLS = list(range(1, 23)) + [35, 36, 37]
BITCOLS = list(range(24, 34))
TERNCOLS = [34, 38]
NPK = 4 + len(FCOLS)            # 29
CH = 1 << 20                    # records per chunk


def meta(d):
    m = dict(l.split('=', 1) for l in open(os.path.join(d, 'meta.txt')).read().split())
    return int(m['r0']), int(m['r1'])


def init(d, r0, r1):
    assert r0 % 8 == 0
    os.makedirs(os.path.join(d, 'seg'), exist_ok=True)
    N = r1 - r0; Nh = ((r1 - 1) >> 3) + 1 - (r0 >> 3)
    for name, size in (('rec.raw', N * NPK * 2), ('hid.raw', Nh * NHID * 2)):
        with open(os.path.join(d, name), 'ab') as f:
            f.truncate(size)
    with open(os.path.join(d, 'meta.txt'), 'w') as f:
        f.write('format=pk1 r0=%d r1=%d n1=%d nhid=%d npk=%d fcols=%s bitcols=%s terncols=%s\n' % (
            r0, r1, NCOL, NHID, NPK, ','.join(map(str, FCOLS)), ','.join(map(str, BITCOLS)), ','.join(map(str, TERNCOLS))))
    print('init %s N=%d Nh=%d' % (d, N, Nh))


class Reader:
    """stdin (or file) with sha256 of every byte read; skip() and exact read()."""

    def __init__(self, f):
        self.f = f; self.h = hashlib.sha256(); self.n = 0

    def read(self, k):
        out = []; got = 0
        while got < k:
            b = self.f.read(k - got)
            if not b:
                break
            self.h.update(b); self.n += len(b); out.append(b); got += len(b)
        return b''.join(out)

    def drain(self):
        while True:
            b = self.f.read(1 << 24)
            if not b:
                return
            self.h.update(b); self.n += len(b)


def spur(d, rd, A, ra, rb):
    r0, r1 = meta(d)
    assert r0 <= ra < rb <= r1
    start = 20 + REC * ra
    assert A <= start
    if A == 0:
        hdr = rd.read(20)
        mg, ver, n1, rec, nw = np.frombuffer(hdr, '<u4')
        assert mg == 0x4c544942 and ver == 2 and n1 == NCOL and rec == REC and nw == 0, (mg, ver, n1, rec, nw)
        skip = start - 20
    else:
        skip = start - A
    while skip > 0:
        b = rd.read(min(skip, 1 << 24)); assert b, 'stream ends before the first record'; skip -= len(b)
    R = np.memmap(os.path.join(d, 'rec.raw'), np.int16, 'r+', shape=(r1 - r0, NPK))
    st = dict(mn=np.full(NCOL, np.inf), mx=np.full(NCOL, -np.inf), s=np.zeros(NCOL), q=np.zeros(NCOL),
              m1mn=np.inf, m1mx=-np.inf, bpmn=65535, bpmx=0, ones=0, ovr=0, bad_flags=0, bad_bpos=0,
              bad_bit=np.zeros(len(BITCOLS), np.int64), bad_tern=np.zeros(len(TERNCOLS), np.int64), rows=0)
    t0 = time.time(); r = ra
    while r < rb:
        k = min(CH, rb - r)
        buf = rd.read(k * REC)
        assert len(buf) == k * REC, 'stream short at record %d (%d of %d B)' % (r, len(buf), k * REC)
        rr = np.frombuffer(buf, np.uint8).reshape(k, REC)
        bp = rr[:, 0:2].copy().view('<u2')[:, 0]
        fl = rr[:, 2].astype(np.int32)
        m1 = rr[:, 4:8].copy().view('<f4')[:, 0]
        c = rr[:, 8:].copy().view('<f4').reshape(k, NCOL)
        o = np.empty((k, NPK), np.int16)
        o[:, 0] = bp.view(np.int16)
        c16 = c.astype(np.float16)
        b23 = c16[:, 23].astype(np.float32)
        bpos = np.clip(np.trunc(b23), 0, 7).astype(np.int32)
        st['bad_bpos'] += int(((b23 != np.trunc(b23)) | (b23 < 0) | (b23 > 7)).sum())
        st['bad_flags'] += int(((fl & ~3) != 0).sum())
        pk = (fl & 3) | (bpos << 2)
        for i, j in enumerate(BITCOLS):
            v = c16[:, j].astype(np.float32)
            st['bad_bit'][i] += int(((v != 0) & (v != 1)).sum())
            pk |= (v == 1).astype(np.int32) << (5 + i)
        o[:, 1] = pk.astype(np.uint16).view(np.int16)
        o[:, 2] = m1.astype(np.float16).view(np.int16)
        tz = np.zeros(k, np.int32)
        for i, j in enumerate(TERNCOLS):
            v = c16[:, j].astype(np.float32)
            st['bad_tern'][i] += int(((v != 0) & (v != 1) & (v != -1)).sum())
            tz |= (np.clip(v, -1, 1).astype(np.int32) + 1) << (2 * i)
        o[:, 3] = tz.astype(np.int16)
        o[:, 4:] = c16[:, FCOLS].view(np.int16)
        R[r - r0:r - r0 + k] = o
        cc = c.astype(np.float64)
        st['mn'] = np.minimum(st['mn'], cc.min(0)); st['mx'] = np.maximum(st['mx'], cc.max(0))
        st['s'] += cc.sum(0); st['q'] += (cc * cc).sum(0)
        st['m1mn'] = min(st['m1mn'], float(m1.min())); st['m1mx'] = max(st['m1mx'], float(m1.max()))
        st['bpmn'] = min(st['bpmn'], int(bp.min())); st['bpmx'] = max(st['bpmx'], int(bp.max()))
        st['ones'] += int(((fl >> 1) & 1).sum()); st['ovr'] += int((fl & 1).sum()); st['rows'] += k
        r += k
        if (r - ra) % (CH * 64) < CH:
            print('spur %d: %d/%d %.0fs' % (ra, r - ra, rb - ra, time.time() - t0), flush=True)
    R.flush(); del R
    rd.drain()
    out = dict(kind='spur', A=A, ra=ra, rb=rb, sha256=rd.h.hexdigest(), bytes=rd.n, rows=st['rows'],
               ones=st['ones'], ovr=st['ovr'], bad_flags=st['bad_flags'], bad_bpos=st['bad_bpos'],
               bad_bit=st['bad_bit'].tolist(), bad_tern=st['bad_tern'].tolist(), m1=[st['m1mn'], st['m1mx']],
               bp=[st['bpmn'], st['bpmx']], mn=st['mn'].tolist(), mx=st['mx'].tolist(), sum=st['s'].tolist(),
               sumsq=st['q'].tolist(), sec=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(d, 'seg', 'spur_%d.json' % ra), 'w'))
    nbad = st['bad_flags'] + st['bad_bpos'] + int(st['bad_bit'].sum()) + int(st['bad_tern'].sum())
    print('SPUR_FERTIG %d %d rows=%d bytes=%d sha256=%s bad=%d badd=%d/%d/%d/%d %.0fs' % (
        ra, rb, st['rows'], rd.n, out['sha256'], nbad, st['bad_flags'], st['bad_bpos'], int(st['bad_bit'].sum()),
        int(st['bad_tern'].sum()), time.time() - t0), flush=True)


def kanal(d, rd, A, ea, eb):
    r0, r1 = meta(d)
    e0, e1 = r0 >> 3, ((r1 - 1) >> 3) + 1
    assert e0 <= ea < eb <= e1
    start = 28 + KREC * ea
    assert A <= start
    if A == 0:
        hh = np.frombuffer(rd.read(28), '<u4')
        assert hh[0] == 0x315a4142 and hh[1] == 4 and hh[2] == NHID and hh[4] == 0 and hh[6] == 1, hh
        skip = start - 28
    else:
        skip = start - A
    while skip > 0:
        b = rd.read(min(skip, 1 << 24)); assert b, 'stream ends before the first row'; skip -= len(b)
    H = np.memmap(os.path.join(d, 'hid.raw'), np.float16, 'r+', shape=(e1 - e0, NHID))
    runs = []; zlast = None; tfnz = 0; hmn = np.inf; hmx = -np.inf; t0 = time.time(); e = ea
    while e < eb:
        k = min(CH, eb - e)
        buf = rd.read(k * KREC)
        assert len(buf) == k * KREC, 'stream short at row %d' % e
        kk = np.frombuffer(buf, '<f4').reshape(k, WK)
        H[e - e0:e - e0 + k] = kk[:, 4:4 + NHID].astype(np.float16)
        z = np.rint(kk[:, WK - 1]).astype(np.int32)
        ch = np.flatnonzero(np.diff(z)) + 1
        if zlast is None or z[0] != zlast:
            runs.append([e, int(z[0])])
        for c in ch:
            runs.append([e + int(c), int(z[c])])
        zlast = int(z[-1])
        tfnz += int((kk[:, 0:4] != 0).any(1).sum())
        hmn = min(hmn, float(kk[:, 4:4 + NHID].min())); hmx = max(hmx, float(kk[:, 4:4 + NHID].max()))
        e += k
    H.flush(); del H
    rd.drain()
    out = dict(kind='kanal', A=A, ea=ea, eb=eb, sha256=rd.h.hexdigest(), bytes=rd.n, rows=eb - ea,
               zone_runs=runs[:2000], n_runs=len(runs), tf_nonzero_rows=tfnz, hid=[hmn, hmx], sec=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(d, 'seg', 'kanal_%d.json' % ea), 'w'))
    print('KANAL_FERTIG %d %d bytes=%d sha256=%s zone_runs=%d %.0fs' % (ea, eb, rd.n, out['sha256'], len(runs),
                                                                      time.time() - t0), flush=True)


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'init':
        init(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif m == 'spur':
        spur(sys.argv[2], Reader(sys.stdin.buffer), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif m == 'kanal':
        kanal(sys.argv[2], Reader(sys.stdin.buffer), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif m == 'file':
        with open(sys.argv[4], 'rb') as f:
            (spur if sys.argv[3] == 'spur' else kanal)(sys.argv[2], Reader(f), int(sys.argv[5]), int(sys.argv[6]),
                                                       int(sys.argv[7]))
    else:
        sys.exit(__doc__)
