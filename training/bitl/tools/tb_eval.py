#!/usr/bin/env python3
# tb_eval.py <store> <F> <T> <cells> <blob> [<blob> ...] [--k 7,5,4,3,2] [--f32] -- in-sample frozen
# replay (torch; = C++ bitl_replay within 0.7 B on the metadata zone) of each blob over [F, T), with the
# weights as stored (--f32) and rounded like the shipped NBLB v2 file at each mantissa width k (bz_nblb.py v2).
# One line per (blob, k): delta_B against the base (= P - OFF of the zone) and the NBLB value count / bf16 file size.
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_kern as BK
import cm_kern as CK
import tb_kern as TK
from tb_qft import qk

ap = argparse.ArgumentParser()
ap.add_argument('store'); ap.add_argument('F', type=int); ap.add_argument('T', type=int); ap.add_argument('cells', type=int)
ap.add_argument('blobs', nargs='+')
ap.add_argument('--k', default='7,5,4,3,2')
ap.add_argument('--f32', action='store_true')
ap.add_argument('--zero-cols', default='')
ap.add_argument('--device', default='cuda')
ap.add_argument('--no-vorbit-reset', action='store_true', help='previous bit at F from record F-1 (segment inside a blob zone)')
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
zc = [int(c) for c in a.zero_cols.split(',') if c != '']
arm = CK.ArmK('e%d' % a.cells, cells=a.cells, zero_cols=zc)
fe = TK.lade_feld(a.store, a.F, a.T, vorbit_reset=not a.no_vorbit_reset)
m = BK.Kopf(arm).to(a.device)
C, inw = arm.C, arm.inw
werte = C * inw + C + 8 * C * C + 4 * C + C + 1
for b in a.blobs:
    bl = BK.blob_read(b)
    ks = ([None] if a.f32 else []) + [int(x) for x in a.k.split(',') if x]
    for k in ks:
        m.load_blob(bl)
        if k is not None:
            with torch.no_grad():
                for p in m.params():
                    p.copy_(qk(p.detach(), k))
        r = BK.bewerten(m, fe, a.device)
        print('EVAL %s %s [%d,%d)%s bits=%d base_B=%.1f head_B=%.1f delta_B=%+.1f werte=%d datei_bf16=%d' % (
            os.path.basename(b), 'f32' if k is None else 'k%d' % k, a.F, a.T, '' if not a.no_vorbit_reset else ' fortl', r['bits'], r['base_B'], r['head_B'], r['delta_B'],
            werte, 24 + 2 * werte), flush=True)
