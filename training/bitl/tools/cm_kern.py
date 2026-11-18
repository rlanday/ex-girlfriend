#!/usr/bin/env python3
# cm_kern.py: the BITLSTM head of this entry in torch, on top of
# bg_kern.py.  Read off Head::Adjust() of the development tree (src/bit_head.cpp of this entry):
#   * stage-1 row n1 = 22 base columns + 13 extra columns + 4 SEMCACHE columns = 39; the head reads every column but column 0
#     and the bit-position column kBposCol = 22 + 1 = 23 -> 37 kept columns, compacted in column order
#   * bit-position one-hot = clamp((int)stage1[23], 0, 7) (true bit position); without the extra columns (earlier
#     trees) it was (s.i & 7) = bposcol None below
#   * TF sketch 32 = channel hid block (projected to 32 values by the trace build), held per 8 records
#   * zero_x: input positions forced to 0 (columns that are constant in the zone); their weights start at 0 and
#     get no gradient, so the blob does not depend on the constant value (container-independent).
# The fed sequence starts at trace record 0 (F = 0): windows of SEQ 1024 from 0, prev bit 0.5 at record 0.
import os
import numpy as np
import torch
import bg_kern as K
from bg_kern import Kopf, AdamC, blob_read, blob_write, kosten, lr_cos, INV_T  # noqa: F401

CANON_KEEP = [j for j in range(1, 39) if j != 23]


class ArmK(K.Arm):
    def __init__(self, name, keep=CANON_KEEP, nhid=32, cells=16, n1=39, zero_cols=(), zero_hid=False, bposcol=23):
        super().__init__(name, keep, nhid=nhid, nerr=0, bposcol=bposcol, cells=cells)
        self.n1 = n1
        zx = [self.keep.index(j) for j in zero_cols if j in self.keep]
        if zero_hid:
            zx += list(range(self.kHid, self.kHid + nhid))
        self.zero_x = sorted(set(zx))


def blob_init(arm, seed):
    b = K.blob_init(arm, seed)
    if arm.zero_x:
        b['win'][:, arm.zero_x] = 0.0
    return b


class FeldK(K.FeldP):
    """FeldP with the bit position of this head ((F + p) & 7) and the zeroed input positions."""

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
        # kept columns missing in the store (compact store: constant columns left out) stay 0; they must be zeroed
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
        if arm.bposcol is not None:
            bpos = cols[:, :, s.col(arm.bposcol)].to(torch.int32).clamp(0, 7).to(torch.int64)
        else:
            bpos = ((self.F + p) & 7).to(torch.int64)
        x.scatter_(2, (n1 + 4 + bpos).unsqueeze(-1), 1.0)
        bit = bita[:, 8:]
        ovr = (fl & 1).bool()
        ok = (self.F + p - 1) >= 0
        x[:, :, arm.kPrev] = torch.where(ok, bita[:, 7:7 + w], torch.full_like(bit, 0.5))
        if arm.nhid:
            x[:, :, arm.kHid:arm.kHid + arm.nhid] = Hh[:, :, :arm.nhid].float().repeat_interleave(8, dim=1)
        if arm.zero_x:
            x[:, :, arm.zero_x] = 0.0
        return x, tt, bp, bit, ovr, valid


def lade_feld(store_dir, F, T, w=K.W, device='cpu'):
    return FeldK(K.StoreP(store_dir, device), F, T, w)
