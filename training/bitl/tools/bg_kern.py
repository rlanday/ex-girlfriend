#!/usr/bin/env python3
# bg_kern.py: PyTorch replica of the BITLSTM correction head (bit_head.cpp of the development tree)
# in the replay form of the C++ replay tool of the development tree (state reset every 1,024 head bits).
#
# Read off the C++ code (not assumed):
#  * feature row x (inw = (nin+1+7)&~7, zero padded):
#      [0, n1keep)            stage-1 columns of the KEEP mask, in column order, raw (XSCALE=0)
#      n1keep+0               m1raw
#      n1keep+1..3            t/T, (t/T)^2, sgn(t);  t = logf(p1/(1-p1)), p1 = base_p/65536, T = 12.203
#      n1keep+4..11           one-hot of the true bit position (int)stage1[BPOSCOL] clamped 0..7
#      kPrev = n1keep+12      previous fed bit (0.5 before the first one)
#      kHid  .. +nhid         TF sketch: first nhid floats of the 64-sketch in the channel (per 8 trace records)
#      kErr  .. +nerr         residuals bit - base_p/65536 of the last nerr fed bits, newest first (0 before)
#  * head: z = win.x + bin; a = z*sigmoid(z); LSTMCell(C) gates i,f,g,o (wih.a + whh.h + bg);
#    c = f*c + i*g; h = o*tanh(c); delta = wout.h + bout.
#  * the recurrent state (h, c) is RESET to 0 at the start of every window of SEQ = 1024 head bits, counted from
#    the first fed bit (replay: <frombits>); frozen heads keep the same rhythm.  Windows are independent sequences.
#  * override bits: forward runs (state advances), no correction (q = base_p), no loss.
#  * coding: d = clamp(delta, +-8); q = (unsigned)(1 + 65534*sigmoid(t + d)) in [1, 65535]; cost -log2 of q/65536.
#  * training: loss = BCE(sigmoid(t + delta)) (unclamped), summed over a group of 16 windows, divided by the number
#    of non-override bits; global L2 clip 1.0; Adam b1 0.9 b2 0.999 eps 1e-8 outside the bias correction
#    (lr_t = lr*sqrt(1-b2^t)/(1-b1^t)); one step per group; an incomplete last group/window is dropped.
#    Each epoch of the C++ trainer is a new process: fresh Adam moments, lr = cosine over epochs (integer 1e-6).
import math, struct, os
import numpy as np
import torch
import torch.nn.functional as Fn

W = 1024          # window of the program: state reset every W head bits
C16 = 16          # cells of the earlier 16-cell form
KT = np.float32(12.203)
INV_T = float(np.float32(1.0) / KT)
CLAMP = 8.0       # clamp of the correction in logit space
REC = 340         # trace record of 8 + 4 x 83 B (v2, n1 83)
NAUX = 69         # channel entry: n_tf 4 + hid 64 + zone 1

L_COLS = [j for j in range(35) if j != 23]   # column mask of the trace build (34351349759)


class Arm:
    def __init__(self, name, keep, nhid=32, nerr=0, bposcol=23, cells=C16):
        self.name, self.keep, self.nhid, self.nerr, self.bposcol, self.C = name, list(keep), nhid, nerr, bposcol, cells
        self.n1keep = len(self.keep)
        self.kPrev = self.n1keep + 12
        self.kHid = self.kPrev + 1
        self.kErr = self.kHid + nhid
        self.nin = self.n1keep + 12 + nhid + nerr
        self.inw = ((self.nin + 1) + 7) & ~7
        self.n1 = 83


def maske(keep, keep2=0):
    """BITL_KEEP / BITL_KEEP2 -> trace columns (neif_bitl.cpp: column j < 63 bit j of KEEP, else bit j-63 of KEEP2)."""
    return [j for j in range(63) if keep >> j & 1] + [63 + j for j in range(20) if keep2 >> j & 1]


# arms of earlier replay builds (their build lines, last token wins): KEEP, KEEP2, TFHID, ERRH
_ARMDEF = {
    'tL': (34351349759, 0, 32, 0), 't0': (34351349759, 0, 0, 0), 'tE4': (34351349759, 0, 32, 4),
    'tE8': (34351349759, 0, 32, 8), 't0E4': (34351349759, 0, 0, 4), 'tnoFW': (34347155455, 0, 32, 0),
    'tTF': (70265656573951, 0, 32, 0), 'tTB': (3332886233087, 0, 32, 0),
    'tMH': (9223301702461947903, 1048575, 32, 0), 'tTBMH': (9223305000996831231, 1048575, 32, 0),
    't0TBMH': (9223305000996831231, 1048575, 0, 0), 'tTFE4': (70265656573951, 0, 32, 4),
    't0TFE4': (70265656573951, 0, 0, 4), 'tTFMHE4': (9223371933767172095, 1048575, 32, 4),
    't0TFMHE4': (9223371933767172095, 1048575, 0, 4),
    'tTFMH': (9223371933767172095, 1048575, 32, 0),   # tTFMHE4 without ERRH (no replay build)
    # metablock zone heads (META_OFF=0, TFHID 0): Lmeta, Lmeta+M, Lmeta+M+H
    'mL': (3670015, 0, 0, 0), 'mM': (9223301668114268159, 15, 0, 0), 'mMH': (9223301668114268159, 1048575, 0, 0),
}
ARMS = {n: Arm(n, maske(k, k2), nhid=h, nerr=e) for n, (k, k2, h, e) in _ARMDEF.items()}
ARMS['tL32'] = Arm('tL32', L_COLS, cells=32)   # C32 text blob, geometry of tL
assert ARMS['tL'].keep == L_COLS


# ------------------------------------------------------------------ NBLB blobs (Head::LoadBlob / DumpBlob)
def blob_read(path):
    d = open(path, 'rb').read()
    assert d[:4] == b'NBLB', path
    ver, C, nin, inw, n1 = struct.unpack_from('<Iiiii', d, 4)
    n = C * inw + C + 8 * C * C + 4 * C + C + 1
    if ver == 1:
        v = np.frombuffer(d, '<f4', n, 24).astype(np.float32)
        assert len(d) == 24 + 4 * n
    elif ver == 2:
        v = (np.frombuffer(d, '<u2', n, 24).astype(np.uint32) << 16).view(np.float32)
        assert len(d) == 24 + 2 * n
    else:
        raise ValueError('version %d' % ver)
    o = 0
    out = dict(C=C, nin=nin, inw=inw, n1=n1)
    for k, s in (('win', C * inw), ('bin', C), ('wih', 4 * C * C), ('whh', 4 * C * C), ('bg', 4 * C), ('wout', C), ('bout', 1)):
        out[k] = v[o:o + s].copy(); o += s
    out['win'] = out['win'].reshape(C, inw)
    out['wih'] = out['wih'].reshape(4 * C, C)
    out['whh'] = out['whh'].reshape(4 * C, C)
    return out


def blob_write(path, b, ver=1):
    hdr = b'NBLB' + struct.pack('<Iiiii', ver, b['C'], b['nin'], b['inw'], b['n1'])
    v = np.concatenate([np.asarray(b[k], np.float32).ravel() for k in ('win', 'bin', 'wih', 'whh', 'bg', 'wout', 'bout')])
    if ver == 1:
        body = v.astype('<f4').tobytes()
    else:   # bf16, round half even (bz_nblb.py v2, k = 7)
        u = v.view(np.uint32).astype(np.uint64)
        body = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype('<u2').tobytes()
    with open(path, 'wb') as f:
        f.write(hdr + body)
    return len(hdr) + len(body)


# ------------------------------------------------------------------ deterministic init (Head::Build, Rng)
class Rng:
    M = (1 << 64) - 1

    def __init__(self, seed):
        self.s = seed & self.M

    def next(self):
        self.s = (self.s + 0x9e3779b97f4a7c15) & self.M
        z = self.s
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & self.M
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & self.M
        return z ^ (z >> 31)

    def uniform(self, b):
        r = np.float32(self.next() >> 40)
        u = r * np.float32(1.0 / 16777216.0)
        return np.float32((np.float32(2.0) * u - np.float32(1.0)) * np.float32(b))


def blob_init(arm, seed):
    C, inw, nin = arm.C, arm.inw, arm.nin
    rng = Rng(seed)
    bin_lim = np.float32(1.0) / np.sqrt(np.float32(inw))
    win = np.zeros((C, inw), np.float32); binv = np.zeros(C, np.float32)
    for r in range(C):
        for j in range(inw):
            win[r, j] = rng.uniform(bin_lim)
        win[r, nin + 1:] = 0.0
        binv[r] = rng.uniform(bin_lim)
    hid_lim = np.float32(1.0) / np.sqrt(np.float32(C))
    wih = np.array([rng.uniform(hid_lim) for _ in range(4 * C * C)], np.float32).reshape(4 * C, C)
    whh = np.array([rng.uniform(hid_lim) for _ in range(4 * C * C)], np.float32).reshape(4 * C, C)
    bg = np.array([rng.uniform(hid_lim) for _ in range(4 * C)], np.float32)
    bg[C:2 * C] += np.float32(1.0)
    return dict(C=C, nin=nin, inw=inw, n1=arm.n1, win=win, bin=binv, wih=wih, whh=whh, bg=bg,
                wout=np.zeros(C, np.float32), bout=np.zeros(1, np.float32))


# ------------------------------------------------------------------ data store
class Store:
    """Compact columns of the trace for absolute records [r0, r1): bp u16, fl u8, m1 f32, cols f32 [N, ncol]
    (trace columns `colmap`), hid f32 [Nb, 32] for channel entries r0>>3 .. (r1-1)>>3."""

    def __init__(self, d, device='cpu', mmap=False, cols=None):
        # cols: load only these trace columns (an arm's KEEP set + the bit-position column)
        meta = dict(l.split('=', 1) for l in open(os.path.join(d, 'meta.txt')).read().split())
        self.r0, self.r1 = int(meta['r0']), int(meta['r1'])
        self.colmap = [int(c) for c in meta['cols'].split(',')]
        self.nhid = int(meta['nhid'])
        mm = 'r' if mmap else None
        ld = lambda n: np.load(os.path.join(d, n), mmap_mode=mm)
        cs = ld('cols.npy')
        if cols is not None:
            want = sorted(set(cols))
            miss = [j for j in want if j not in self.colmap]
            assert not miss, 'store %s lacks trace columns %s' % (d, miss)
            idx = [self.colmap.index(j) for j in want]
            if idx != list(range(len(self.colmap))):
                src = np.load(os.path.join(d, 'cols.npy'), mmap_mode='r')
                cs = np.empty((src.shape[0], len(idx)), np.float32)
                for a in range(0, src.shape[0], 1 << 22):
                    cs[a:a + (1 << 22)] = src[a:a + (1 << 22)][:, idx]
                self.colmap = want
        self.np = dict(bp=ld('bp.npy'), fl=ld('fl.npy'), m1=ld('m1.npy'), cols=cs, hid=ld('hid.npy'))
        self.device = device
        self.t = None

    def to(self, device, pin=False, half=False):
        # half: cols, m1, hid kept as float16 (frozen-eval probe: <= 0.12 B on 14.75 KB), widened per batch
        t = {}
        for k, v in self.np.items():
            a = torch.from_numpy(np.ascontiguousarray(v))
            if k == 'bp':
                a = a.to(torch.int32)
            elif k == 'fl':
                a = a.to(torch.uint8)
            elif half:
                a = a.to(torch.float16)
            t[k] = a.pin_memory() if pin else a.to(device)
        self.t = t
        self.device = device
        return self

    def col(self, j):
        return self.colmap.index(j)


def extract(spur, kanal, r0, r1, out, cols, nhid=32, chunk=1 << 20, spur_r0=0, kanal_e0=0, rec=REC, naux=NAUX):
    """Raw trace (records of REC bytes behind a 20-byte header, or a raw slice starting at record spur_r0) ->
    Store directory."""
    os.makedirs(out, exist_ok=True)
    N = r1 - r0
    hdr = 20 if spur_r0 == 0 else 0
    sp = np.memmap(spur, np.uint8, 'r')
    ka = np.memmap(kanal, np.uint8, 'r')
    khdr = 28 if kanal_e0 == 0 else 0
    bp = np.lib.format.open_memmap(os.path.join(out, 'bp.npy'), 'w+', np.uint16, (N,))
    fl = np.lib.format.open_memmap(os.path.join(out, 'fl.npy'), 'w+', np.uint8, (N,))
    m1 = np.lib.format.open_memmap(os.path.join(out, 'm1.npy'), 'w+', np.float32, (N,))
    cs = np.lib.format.open_memmap(os.path.join(out, 'cols.npy'), 'w+', np.float32, (N, len(cols)))
    ci = np.array(cols)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        o = hdr + (r0 - spur_r0 + a) * rec
        rr = np.asarray(sp[o:o + (b - a) * rec]).reshape(b - a, rec)
        bp[a:b] = rr[:, 0:2].copy().view('<u2')[:, 0]
        fl[a:b] = rr[:, 2]
        m1[a:b] = rr[:, 4:8].copy().view('<f4')[:, 0]
        cs[a:b] = rr[:, 8:].copy().view('<f4')[:, ci]
    e0, e1 = r0 >> 3, ((r1 - 1) >> 3) + 1
    o = khdr + (e0 - kanal_e0) * naux * 4
    kk = np.asarray(ka[o:o + (e1 - e0) * naux * 4]).copy().view('<f4').reshape(e1 - e0, naux)
    np.save(os.path.join(out, 'hid.npy'), np.ascontiguousarray(kk[:, 4:4 + nhid]))
    zone = kk[:, naux - 1]
    for x in (bp, fl, m1, cs):
        x.flush()
    with open(os.path.join(out, 'meta.txt'), 'w') as f:
        f.write('r0=%d r1=%d cols=%s nhid=%d zone_max=%g zone_min=%g\n' % (
            r0, r1, ','.join(map(str, cols)), nhid, float(zone.max()), float(zone.min())))


# ------------------------------------------------------------------ model
class Kopf(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        C = arm.C
        self.arm = arm
        self.inp = torch.nn.Linear(arm.inw, C)
        self.lstm = torch.nn.LSTM(C, C, batch_first=True)
        self.out = torch.nn.Linear(C, 1)

    def load_blob(self, b):
        a = self.arm
        assert (b['C'], b['nin'], b['inw']) == (a.C, a.nin, a.inw), (b['C'], b['nin'], b['inw'], a.C, a.nin, a.inw)
        with torch.no_grad():
            self.inp.weight.copy_(torch.from_numpy(b['win']))
            self.inp.bias.copy_(torch.from_numpy(b['bin']))
            self.lstm.weight_ih_l0.copy_(torch.from_numpy(b['wih']))
            self.lstm.weight_hh_l0.copy_(torch.from_numpy(b['whh']))
            self.lstm.bias_ih_l0.copy_(torch.from_numpy(b['bg']))
            self.lstm.bias_hh_l0.zero_()
            self.out.weight.copy_(torch.from_numpy(b['wout']).view(1, -1))
            self.out.bias.copy_(torch.from_numpy(b['bout']).view(1))
        return self

    def blob(self):
        a = self.arm
        g = lambda t: t.detach().float().cpu().numpy().copy()
        win = g(self.inp.weight)
        win[:, a.nin + 1:] = 0.0
        return dict(C=a.C, nin=a.nin, inw=a.inw, n1=a.n1, win=win, bin=g(self.inp.bias),
                    wih=g(self.lstm.weight_ih_l0), whh=g(self.lstm.weight_hh_l0),
                    bg=g(self.lstm.bias_ih_l0) + g(self.lstm.bias_hh_l0),
                    wout=g(self.out.weight).ravel(), bout=g(self.out.bias).ravel())

    def params(self):
        # the parameters of the C++ head (bias_hh stays 0: the C++ cell has ONE gate bias)
        return [self.inp.weight, self.inp.bias, self.lstm.weight_ih_l0, self.lstm.weight_hh_l0,
                self.lstm.bias_ih_l0, self.out.weight, self.out.bias]

    def forward(self, x):
        z = self.inp(x)
        a = z * torch.sigmoid(z)
        h, _ = self.lstm(a)
        return self.out(h).squeeze(-1)

    def forward_hc(self, x, hc=None):
        # with a carried state (window 0: the state is never reset)
        z = self.inp(x)
        a = z * torch.sigmoid(z)
        h, hc2 = self.lstm(a, hc)
        return self.out(h).squeeze(-1), hc2

    def forward_loop(self, x):
        # explicit cell loop (reference for the cuDNN/oneDNN LSTM)
        C = self.arm.C
        z = self.inp(x)
        a = z * torch.sigmoid(z)
        P = a @ self.lstm.weight_ih_l0.T + self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0
        B, T = x.shape[0], x.shape[1]
        h = x.new_zeros(B, C); c = x.new_zeros(B, C)
        whhT = self.lstm.weight_hh_l0.T
        hs = []
        for t in range(T):
            pre = P[:, t] + h @ whhT
            i = torch.sigmoid(pre[:, 0:C]); f = torch.sigmoid(pre[:, C:2 * C])
            g = torch.tanh(pre[:, 2 * C:3 * C]); o = torch.sigmoid(pre[:, 3 * C:])
            c = f * c + i * g
            h = o * torch.tanh(c)
            hs.append(h)
        return self.out(torch.stack(hs, 1)).squeeze(-1)


# ------------------------------------------------------------------ features of a batch of windows
class Feld:
    """Fed sequence = absolute trace records [F, T) of a store; windows of W head bits from F."""

    def __init__(self, store, F, T, w=W):
        assert store.r0 <= F and T <= store.r1
        self.s, self.F, self.T, self.w = store, F, T, w      # w = window (state reset every w head bits)
        self.L = T - F
        ww = w if w > 0 else self.L                           # w = 0: one window = the whole fed sequence
        self.nwin = (self.L + ww - 1) // ww
        self.nfull = self.L // ww

    def batch(self, arm, wins, device):
        s, t = self.s, self.s.t
        sd = t['bp'].device                                              # where the store lives
        wins = torch.as_tensor(wins, dtype=torch.int64, device=sd)
        w = self.w
        p = (wins[:, None] * w + torch.arange(w, device=sd)[None, :])  # relative position in the fed sequence
        valid = p < self.L
        pc = torch.where(valid, p, torch.zeros_like(p))
        ia = self.F - s.r0 + pc                                          # store row
        B = wins.shape[0]

        def g(name, idx):
            v = t[name][idx.reshape(-1)].to(device, non_blocking=True).reshape(*idx.shape, *t[name].shape[1:])
            return v.float() if v.dtype == torch.float16 else v
        bp = g('bp', ia)
        fl = g('fl', ia).to(torch.int32)
        m1 = g('m1', ia)
        keep_i = torch.tensor([s.col(j) for j in arm.keep], dtype=torch.int64, device=device)
        cols_all = g('cols', ia)                                         # (B, W, ncol)
        x = torch.zeros(B, w, arm.inw, dtype=torch.float32, device=device)
        x[:, :, :arm.n1keep] = cols_all[:, :, keep_i]
        n1 = arm.n1keep
        x[:, :, n1] = m1
        p1 = bp.to(torch.float32) / 65536.0
        tt = torch.log(p1 / (1.0 - p1))
        tl = tt * INV_T
        x[:, :, n1 + 1] = tl
        x[:, :, n1 + 2] = tl * tl
        x[:, :, n1 + 3] = torch.sign(tt)
        bpos = cols_all[:, :, s.col(arm.bposcol)].to(torch.int32).clamp(0, 7).to(torch.int64)
        x.scatter_(2, (n1 + 4 + bpos).unsqueeze(-1), 1.0)
        bit = ((fl >> 1) & 1).to(torch.float32)
        ovr = (fl & 1).bool()
        # previous fed bit and residual history (cross the window edge, not the start of the fed sequence)
        for k in range(max(1, arm.nerr)):
            q = pc - 1 - k
            ok = (q >= 0) & valid
            iq = self.F - s.r0 + torch.where(ok, q, torch.zeros_like(q))
            okd = ok.to(device)
            bq = g('bp', iq).to(torch.float32)
            fq = ((g('fl', iq).to(torch.int32) >> 1) & 1).to(torch.float32)
            if k == 0:
                x[:, :, arm.kPrev] = torch.where(okd, fq, torch.full_like(fq, 0.5))
            if k < arm.nerr:
                x[:, :, arm.kErr + k] = torch.where(okd, fq - bq / 65536.0, torch.zeros_like(fq))
        if arm.nhid:
            hi = ((self.F + pc) >> 3) - (s.r0 >> 3)
            x[:, :, arm.kHid:arm.kHid + arm.nhid] = g('hid', hi)[:, :, :arm.nhid]
        return x, tt, bp, bit, ovr, valid.to(device)


# ------------------------------------------------------------------ costs (bitl_replay accounting)
def kosten(delta, tt, bp, bit, ovr, valid):
    d = delta.clamp(-CLAMP, CLAMP)
    pnew = torch.sigmoid(tt + d)
    q = torch.floor(65534.0 * pnew + 1.0).clamp(1, 65535)
    q = torch.where(ovr, bp.to(q.dtype), q)
    pb = bp.to(torch.float64) / 65536.0
    ph = q.to(torch.float64) / 65536.0
    b = bit.bool()
    cb = -torch.log2(torch.where(b, pb, 1.0 - pb))
    ch = -torch.log2(torch.where(b, ph, 1.0 - ph))
    v = valid
    return float(cb[v].sum()), float(ch[v].sum()), int(v.sum())


@torch.no_grad()
def bewerten_lang(model, store, F, T, device, chunk=4096):
    """window 0: one sequence from F to T, state never reset (sequential, batch 1)."""
    model.eval()
    fe = Feld(store, F, T, w=chunk)
    sb = sh = 0.0; n = 0; hc = None
    for k in range(fe.nwin):
        x, tt, bp, bit, ovr, valid = fe.batch(model.arm, [k], device)
        delta, hc = model.forward_hc(x, hc)
        hc = (hc[0].contiguous(), hc[1].contiguous())
        b_, h_, n_ = kosten(delta, tt, bp, bit, ovr, valid)
        sb += b_; sh += h_; n += n_
    return dict(bits=n, base_B=sb / 8, head_B=sh / 8, delta_B=(sh - sb) / 8)


@torch.no_grad()
def bewerten(model, feld, device, bwin=0, loop=False):
    if feld.w == 0:
        return bewerten_lang(model, feld.s, feld.F, feld.T, device)
    if not bwin:
        bwin = max(1, 512 * 1024 // feld.w)
    model.eval()
    sb = sh = 0.0; n = 0
    for a in range(0, feld.nwin, bwin):
        wins = list(range(a, min(feld.nwin, a + bwin)))
        x, tt, bp, bit, ovr, valid = feld.batch(model.arm, wins, device)
        delta = model.forward_loop(x) if loop else model(x)
        b_, h_, n_ = kosten(delta, tt, bp, bit, ovr, valid)
        sb += b_; sh += h_; n += n_
    return dict(bits=n, base_B=sb / 8, head_B=sh / 8, delta_B=(sh - sb) / 8)


# ------------------------------------------------------------------ the C++ optimiser
class AdamC:
    B1 = float(np.float32(0.9))
    B2 = float(np.float32(0.999))
    EPS = float(np.float32(1e-8))

    def __init__(self, params):
        self.p = params
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.b1t = 1.0; self.b2t = 1.0; self.n = 0

    @torch.no_grad()
    def step(self, lr, clip=1.0):
        gs = [p.grad for p in self.p]
        ss = sum(float((g.double() ** 2).sum()) for g in gs)
        if not math.isfinite(ss):
            return False
        nrm = math.sqrt(ss)
        sc = float(np.float32(clip / nrm)) if (clip > 0 and nrm > clip) else 1.0
        self.b1t *= self.B1; self.b2t *= self.B2
        lrt = float(np.float32(lr * math.sqrt(1.0 - self.b2t) / (1.0 - self.b1t)))
        for p, g, m, v in zip(self.p, gs, self.m, self.v):
            if sc != 1.0:
                g = g * sc
            m.mul_(self.B1).add_(g, alpha=1.0 - self.B1)
            v.mul_(self.B2).addcmul_(g, g, value=1.0 - self.B2)
            p.sub_(lrt * m / (v.sqrt() + self.EPS))
        self.n += 1
        return True


def verlust(model, x, tt, bit, ovr, valid):
    delta = model(x)
    m = (valid & ~ovr)
    nv = int(m.sum())
    l = Fn.binary_cross_entropy_with_logits(tt + delta, bit, reduction='none')
    return (l * m).sum() / max(nv, 1), nv


def lr_cos(e, E, a, b):
    # be_train.sh: int(round(b + (a-b)*0.5*(1+cos(pi*(e-1)/max(E-1,1))))) in 1e-6, then (float)(atof/1e6)
    v = int(round(b + (a - b) * 0.5 * (1 + math.cos(math.pi * (e - 1) / max(E - 1, 1)))))
    return float(np.float32(v / 1000000.0)), v


# ------------------------------------------------------------------ packed store "p16" (bg_cut.py): long zones
class StoreP:
    """rec.npy int16 [N, 3 + n1] (base_p bits, flags, f16 m1raw, f16 columns), hid.npy f16 [Nb, nhid].
    Kept on the host (memmap, shared page cache) or moved to the device when it fits."""

    def __init__(self, d, device='cpu'):
        meta = dict(l.split('=', 1) for l in open(os.path.join(d, 'meta.txt')).read().split())
        assert meta.get('format') == 'p16', d
        self.r0, self.r1 = int(meta['r0']), int(meta['r1'])
        self.colmap = [int(c) for c in meta['cols'].split(',')]
        self.nhid = int(meta['nhid'])
        rec = np.load(os.path.join(d, 'rec.npy'), mmap_mode='r')
        hid = np.load(os.path.join(d, 'hid.npy'), mmap_mode='r')
        if device == 'cpu':
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.rec = torch.from_numpy(rec)
                self.hid = torch.from_numpy(hid)
        else:
            self.rec = torch.from_numpy(np.ascontiguousarray(rec)).to(device)
            self.hid = torch.from_numpy(np.ascontiguousarray(hid)).to(device)
        self.N = self.rec.shape[0]

    def col(self, j):
        return self.colmap.index(j)


class FeldP:
    """Feld on a StoreP: one gather of [start-8, start+w) per window (the 8 records before the window carry the
    previous bit and the residuals), one gather of the w/8 channel entries."""

    def __init__(self, store, F, T, w=W):
        assert store.r0 <= F - 8 or store.r0 == F, 'store must hold 8 records before F'
        assert T <= store.r1 and F % 8 == 0 and w % 8 == 0 and w > 0
        self.s, self.F, self.T, self.w = store, F, T, w
        self.L = T - F
        self.nwin = (self.L + w - 1) // w
        self.nfull = self.L // w

    def fetch(self, wins):
        """the gathers (on the store's device); a host store returns pinned tensors (prefetch thread)"""
        s, w = self.s, self.w
        sd = s.rec.device
        wins = torch.as_tensor(wins, dtype=torch.int64, device=sd)
        B = wins.shape[0]
        st0 = self.F - s.r0 + wins * w
        idx = (st0[:, None] + torch.arange(-8, w, device=sd)[None, :]).clamp(0, s.N - 1)
        e0 = (self.F >> 3) - (s.r0 >> 3)
        hidx = (e0 + wins[:, None] * (w // 8) + torch.arange(w // 8, device=sd)[None, :]).clamp(0, s.hid.shape[0] - 1)
        if sd.type == 'cpu':
            R = torch.empty((B * (w + 8), s.rec.shape[1]), dtype=s.rec.dtype).pin_memory()
            Hh = torch.empty((B * (w // 8), s.hid.shape[1]), dtype=s.hid.dtype).pin_memory()
            torch.index_select(s.rec, 0, idx.reshape(-1), out=R)
            torch.index_select(s.hid, 0, hidx.reshape(-1), out=Hh)
        else:
            R = torch.index_select(s.rec, 0, idx.reshape(-1))
            Hh = torch.index_select(s.hid, 0, hidx.reshape(-1))
        return wins, R, Hh

    def batch(self, arm, wins, device, pre=None):
        s, w = self.s, self.w
        wins, R, Hh = pre if pre is not None else self.fetch(wins)
        B = wins.shape[0]
        R = R.to(device, non_blocking=True).view(B, w + 8, -1)
        Hh = Hh.to(device, non_blocking=True).view(B, w // 8, -1)
        p = (wins.to(device)[:, None] * w + torch.arange(w, device=device)[None, :])
        valid = p < self.L
        bpa = R[:, :, 0].int() & 0xFFFF
        fla = R[:, :, 1].int()
        bita = ((fla >> 1) & 1).float()
        bp = bpa[:, 8:]
        fl = fla[:, 8:]
        m1 = R[:, 8:, 2].view(torch.float16).float()
        cols = R[:, 8:, 3:].view(torch.float16).float()
        keep_i = torch.tensor([s.col(j) for j in arm.keep], dtype=torch.int64, device=device)
        x = torch.zeros(B, w, arm.inw, dtype=torch.float32, device=device)
        x[:, :, :arm.n1keep] = cols[:, :, keep_i]
        n1 = arm.n1keep
        x[:, :, n1] = m1
        p1 = bp.to(torch.float32) / 65536.0
        tt = torch.log(p1 / (1.0 - p1))
        tl = tt * INV_T
        x[:, :, n1 + 1] = tl
        x[:, :, n1 + 2] = tl * tl
        x[:, :, n1 + 3] = torch.sign(tt)
        bpos = cols[:, :, s.col(arm.bposcol)].to(torch.int32).clamp(0, 7).to(torch.int64)
        x.scatter_(2, (n1 + 4 + bpos).unsqueeze(-1), 1.0)
        bit = bita[:, 8:]
        ovr = (fl & 1).bool()
        resid = bita - bpa.float() / 65536.0
        for k in range(max(1, arm.nerr)):
            ok = (p - 1 - k) >= 0
            sl = slice(8 - 1 - k, 8 - 1 - k + w)
            if k == 0:
                x[:, :, arm.kPrev] = torch.where(ok, bita[:, sl], torch.full_like(bit, 0.5))
            if k < arm.nerr:
                x[:, :, arm.kErr + k] = torch.where(ok, resid[:, sl], torch.zeros_like(bit))
        if arm.nhid:
            x[:, :, arm.kHid:arm.kHid + arm.nhid] = Hh[:, :, :arm.nhid].float().repeat_interleave(8, dim=1)
        return x, tt, bp, bit, ovr, valid


def lade_feld(store_dir, F, T, w=W, device='cpu', half=False, arm=None, pin=False):
    """Feld for either store format (p16 -> FeldP, else Store/Feld)."""
    if 'format=p16' in open(os.path.join(store_dir, 'meta.txt')).read():
        return FeldP(StoreP(store_dir, device), F, T, w)
    st = Store(store_dir, cols=(arm.keep + [arm.bposcol]) if arm is not None else None).to(device, pin=pin, half=half)
    return Feld(st, F, T, w)
