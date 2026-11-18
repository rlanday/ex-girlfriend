#!/usr/bin/env python3
# tb_kern.py: the packed store "pk1" (tb_cut.py) for the head of cm_kern.py.
#   StorePK  host memmaps rec.raw int16 [N, 29] and hid.raw f16 [Nh, 32] (shared page cache / /dev/shm)
#   FeldPK   FeldP gathers (prefetch-able, pinned) on the packed rows; batch() first decodes them on the device into
#            the p16 row of cm_cut.py (base_p, flags, m1raw, f16 columns 1..38) bit for bit, then computes the head
#            input exactly as cm_kern.FeldK.batch, except the previous bit: vorbit_reset=True gives 0.5 at the first
#            record of the fed sequence F (the switch between the three weight sets resets h, c and the previous bit);
#            False reads record F-1 like FeldK.
#   lade_feld(store, F, T, vorbit_reset=True)  pk1 -> FeldPK, otherwise cm_kern.lade_feld
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_kern as BK
import cm_kern as CK
from bg_kern import INV_T

F16_ONE = 0x3C00
F16_MONE = -17408   # 0xBC00 as int16


class StorePK:
    def __init__(self, d, device='cpu'):
        meta = dict(l.split('=', 1) for l in open(os.path.join(d, 'meta.txt')).read().split())
        assert meta.get('format') == 'pk1', d
        assert device == 'cpu', 'pk1 store lives on the host'
        self.r0, self.r1 = int(meta['r0']), int(meta['r1'])
        self.npk, self.nhid = int(meta['npk']), int(meta['nhid'])
        self.fcols = [int(c) for c in meta['fcols'].split(',')]
        self.bitcols = [int(c) for c in meta['bitcols'].split(',')]
        self.terncols = [int(c) for c in meta['terncols'].split(',')]
        assert self.fcols == list(range(1, 23)) + [35, 36, 37] and self.bitcols == list(range(24, 34)) \
            and self.terncols == [34, 38], 'decoder is written for this layout'
        self.N = self.r1 - self.r0
        Nh = ((self.r1 - 1) >> 3) + 1 - (self.r0 >> 3)
        rec = np.memmap(os.path.join(d, 'rec.raw'), np.int16, 'r', shape=(self.N, self.npk))
        hid = np.memmap(os.path.join(d, 'hid.raw'), np.float16, 'r', shape=(Nh, self.nhid))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.rec = torch.from_numpy(rec)
            self.hid = torch.from_numpy(hid)
        self.colmap = list(range(1, 39))     # the decoded row carries columns 1..38 like a p16 store

    def col(self, j):
        return self.colmap.index(j)


def decode_pk1(R):
    """R int16 [M, 29] (device) -> p16 row int16 [M, 41]: base_p, flags, m1raw, f16 bits of columns 1..38."""
    M = R.shape[0]
    o = torch.empty((M, 41), dtype=torch.int16, device=R.device)
    pk = R[:, 1].to(torch.int32) & 0xFFFF
    o[:, 0] = R[:, 0]
    o[:, 1] = (pk & 3).to(torch.int16)
    o[:, 2] = R[:, 2]
    o[:, 3:25] = R[:, 4:26]                                            # columns 1..22
    o[:, 3 + 22] = ((pk >> 2) & 7).to(torch.float16).view(torch.int16)  # column 23 = bit position
    for i in range(10):                                                 # columns 24..33
        o[:, 3 + 23 + i] = (((pk >> (5 + i)) & 1) * F16_ONE).to(torch.int16)
    tz = R[:, 3].to(torch.int32)
    o[:, 3 + 33] = ((tz & 3) - 1).to(torch.float16).view(torch.int16)  # column 34
    o[:, 3 + 34:3 + 37] = R[:, 26:29]                                   # columns 35, 36, 37
    o[:, 3 + 37] = (((tz >> 2) & 3) - 1).to(torch.float16).view(torch.int16)   # column 38
    return o


class FeldPK(BK.FeldP):
    def __init__(self, store, F, T, w=BK.W, vorbit_reset=True):
        super().__init__(store, F, T, w)
        self.vorbit_reset = vorbit_reset

    def batch(self, arm, wins, device, pre=None):
        s, w = self.s, self.w
        wins, R, Hh = pre if pre is not None else self.fetch(wins)
        B = wins.shape[0]
        R = decode_pk1(R.to(device, non_blocking=True)).view(B, w + 8, -1)
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
        have = [i for i, j in enumerate(arm.keep) if j in s.colmap]
        assert set(range(arm.n1keep)) - set(have) <= set(arm.zero_x), 'store lacks kept columns that are not zeroed'
        keep_i = torch.tensor([s.col(arm.keep[i]) for i in have], dtype=torch.int64, device=device)
        x = torch.zeros(B, w, arm.inw, dtype=torch.float32, device=device)
        x[:, :, have] = cols[:, :, keep_i]
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
        ok = (p - 1) >= 0 if self.vorbit_reset else (self.F + p - 1) >= 0
        x[:, :, arm.kPrev] = torch.where(ok, bita[:, 7:7 + w], torch.full_like(bit, 0.5))
        if arm.nhid:
            x[:, :, arm.kHid:arm.kHid + arm.nhid] = Hh[:, :, :arm.nhid].float().repeat_interleave(8, dim=1)
        if arm.zero_x:
            x[:, :, arm.zero_x] = 0.0
        return x, tt, bp, bit, ovr, valid


def lade_feld(store_dir, F, T, w=BK.W, device='cpu', vorbit_reset=True):
    if 'format=pk1' in open(os.path.join(store_dir, 'meta.txt')).read():
        return FeldPK(StorePK(store_dir), F, T, w, vorbit_reset=vorbit_reset)
    return CK.lade_feld(store_dir, F, T, w, device=device)


def prefetch(fe, order, depth=3):
    """generator of pre-gathered batches (pinned) from a host store, gathered in a thread two batches ahead"""
    import queue, threading
    qq = queue.Queue(maxsize=depth)

    def _prod():
        try:
            for ww in order:
                qq.put(fe.fetch(ww))
        except BaseException as ex:   # hand the error to the consumer (no silent short epoch)
            qq.put(ex)
            return
        qq.put(None)
    threading.Thread(target=_prod, daemon=True).start()
    n = 0
    while True:
        it = qq.get()
        if it is None:
            break
        if isinstance(it, BaseException):
            raise it
        n += 1
        yield it
    assert n == len(order), 'prefetch delivered %d of %d batches' % (n, len(order))
