#!/usr/bin/env python3
# tb_check.py <p16_store> <pk1_store> <F> <T> -- acceptance of the packed store against cm_cut.py's p16
# store of the same records: (1) every decoded pk1 row equals the p16 row bit for bit (base_p, flags, m1raw,
# columns 1..38), (2) the channel sketch is equal, (3) the head input of FeldPK (previous bit read like FeldK) equals
# cm_kern.FeldK.batch for C16 and C32 on all windows, (4) with the previous-bit reset only (window 0, record 0) differs.
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_kern as BK
import cm_kern as CK
import tb_kern as TK

d16, dpk, F, T = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s16 = BK.StoreP(d16, 'cpu'); spk = TK.StorePK(dpk)
assert (s16.r0, s16.r1) == (spk.r0, spk.r1), ((s16.r0, s16.r1), (spk.r0, spk.r1))
assert s16.colmap == spk.colmap, (s16.colmap, spk.colmap)
dec = TK.decode_pk1(spk.rec.clone())
eq = torch.equal(dec, s16.rec.clone())
print('(1) rows %d: decoded pk1 == p16 %s' % (s16.N, eq))
if not eq:
    bad = (dec != s16.rec.clone()).any(0).nonzero().ravel().tolist()
    print('    differing row positions', bad); sys.exit(1)
eqh = torch.equal(spk.hid.clone(), s16.hid.clone())
print('(2) sketch rows %d equal %s' % (s16.hid.shape[0], eqh))
if not eqh:
    sys.exit(1)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ok = True
for C in (16, 32):
    arm = CK.ArmK('m%d' % C, cells=C)
    f16 = CK.FeldK(s16, F, T); fpk = TK.FeldPK(spk, F, T, vorbit_reset=False); fpr = TK.FeldPK(spk, F, T)
    for a in range(0, f16.nwin, 64):
        w = list(range(a, min(f16.nwin, a + 64)))
        r1 = f16.batch(arm, w, dev); r2 = fpk.batch(arm, w, dev); r3 = fpr.batch(arm, w, dev)
        same = all(torch.equal(x, y) for x, y in zip(r1, r2))
        dx = (r1[0] != r3[0]).nonzero().tolist()
        okr = all(torch.equal(x, y) for x, y in zip(r1[1:], r3[1:])) and (
            dx == [] or all(t[0] == 0 and t[1] == 0 and t[2] == arm.kPrev for t in dx)) and (a > 0 or w[0] != 0 or
                                                                                         float(r3[0][0, 0, arm.kPrev]) == 0.5)
        if not same or not okr:
            ok = False
            print('    C%d windows %d..: equal %s, reset-diff %s' % (C, a, same, dx[:5]))
    print('(3/4) C%d windows %d: FeldPK == FeldK %s (vorbit reset only at window 0 record 0)' % (C, f16.nwin, ok))
print('CHECK', 'OK' if ok else 'FEHLER')
sys.exit(0 if ok else 1)
