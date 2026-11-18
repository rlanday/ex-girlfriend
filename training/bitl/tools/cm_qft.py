#!/usr/bin/env python3
# cm_qft.py <store> <F> <T> <blob_in> <out_prefix> [options] -- mantissa-aware fine-tuning of a trained
# blob: the forward pass sees the weights rounded like the shipped NBLB v2 file (bf16, round half even, mantissa
# rounded to --k bits, bz_nblb.py v2), the gradient goes to fp32 master weights (straight through), same loss,
# windows, Adam (C++ form) and clip as cm_train.py.  Eval (in-sample, frozen, rounded weights) at epoch 0 and every
# --eval-every epochs; the best rounded blob is written as <out>_best.nblb (version 1, values exactly on the k grid,
# so `bz_nblb.py v2 ... k` reproduces them bit for bit), the last as <out>_e<E>.nblb.
#   cm_qft.py qtest <blob> <k> <ref_v2_blob>    rounding check against a bz_nblb.py v2 file (bitwise)
import argparse, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_kern as BK
import cm_kern as K


def qk(w, k):
    """float32 tensor -> float32 tensor on the bf16 grid with k mantissa bits (RNE twice, as bz_nblb.py write)."""
    u = w.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    h = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    s = 7 - k
    if s > 0:
        h = ((h + (1 << (s - 1)) - 1 + ((h >> s) & 1)) >> s) << s
    h = h & 0xFFFF
    hs = torch.where(h >= 0x8000, h - 0x10000, h)
    return (hs << 16).to(torch.int32).view(torch.float32)


if len(sys.argv) > 1 and sys.argv[1] == 'qtest':
    b = BK.blob_read(sys.argv[2]); k = int(sys.argv[3]); r = BK.blob_read(sys.argv[4])
    bad = 0
    for n in ('win', 'bin', 'wih', 'whh', 'bg', 'wout', 'bout'):
        a = qk(torch.from_numpy(np.ascontiguousarray(b[n], np.float32)), k).numpy()
        bad += int((a.view(np.uint32) != np.ascontiguousarray(r[n], np.float32).view(np.uint32)).sum())
    print('qtest k=%d: %d Werte verschieden' % (k, bad))
    sys.exit(1 if bad else 0)

ap = argparse.ArgumentParser()
ap.add_argument('store'); ap.add_argument('F', type=int); ap.add_argument('T', type=int)
ap.add_argument('blob'); ap.add_argument('out')
ap.add_argument('--cells', type=int, default=16)
ap.add_argument('--k', type=int, default=4)
ap.add_argument('--zero-cols', default='')
ap.add_argument('--zero-hid', action='store_true')
ap.add_argument('--epochs', type=int, default=64)
ap.add_argument('--lr0', type=float, default=1000.0)
ap.add_argument('--lr1', type=float, default=10.0)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--seed', type=int, default=1112363057)
ap.add_argument('--clip', type=float, default=1.0)
ap.add_argument('--eval-every', type=int, default=8)
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--pidfile', default='')
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if a.pidfile:
    with open(a.pidfile, 'a') as f:
        f.write('%d %s\n' % (os.getpid(), os.path.basename(a.out)))
LOG = a.out + '.log'


def say(s):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + s
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


dev = a.device
zc = [int(c) for c in a.zero_cols.split(',') if c != '']
arm = K.ArmK('q%d' % a.cells, cells=a.cells, zero_cols=zc, zero_hid=a.zero_hid)
fe = K.lade_feld(a.store, a.F, a.T, device=dev)
m = BK.Kopf(arm).to(dev)
m.load_blob(BK.blob_read(a.blob))
ps = m.params()
with torch.no_grad():
    m.lstm.bias_hh_l0.zero_()


def runden():
    keep = [p.detach().clone() for p in ps]
    with torch.no_grad():
        for p in ps:
            p.copy_(qk(p.detach(), a.k))
    return keep


def zurueck(keep):
    with torch.no_grad():
        for p, q in zip(ps, keep):
            p.copy_(q)


def auswerten(e):
    keep = runden()
    r = BK.bewerten(m, fe, dev)
    b = m.blob()
    zurueck(keep)
    say('  eval e%d k%d in-sample frozen bits=%d base_B=%.1f head_B=%.1f delta_B=%+.1f' % (
        e, a.k, r['bits'], r['base_B'], r['head_B'], r['delta_B']))
    return r['delta_B'], b


spe = fe.nfull // a.batch
total = spe * a.epochs
say('=== %s qft C%d k%d from %s, window [%d,%d), epochs %d, lr %g -> %g, batch %d, seed %d' % (
    os.path.basename(a.out), a.cells, a.k, os.path.basename(a.blob), a.F, a.T, a.epochs, a.lr0, a.lr1, a.batch, a.seed))
best, bb = auswerten(0)
BK.blob_write('%s_best.nblb' % a.out, bb, 1)
say('  best e0 delta_B=%+.1f' % best)
gen = torch.Generator().manual_seed(a.seed)
opt = BK.AdamC(ps)
step = 0
T0 = time.time()
for e in range(1, a.epochs + 1):
    m.train()
    t0 = time.time()
    perm = torch.randperm(fe.nfull, generator=gen).tolist()
    order = [sorted(perm[i * a.batch:(i + 1) * a.batch]) for i in range(spe)]
    nskip = 0
    for wins in order:
        x, tt, bp, bit, ovr, valid = fe.batch(arm, wins, dev)
        keep = runden()
        for p in ps:
            p.grad = None
        m.lstm.bias_hh_l0.grad = None
        delta = m(x)
        msk = (valid & ~ovr)
        l = torch.nn.functional.binary_cross_entropy_with_logits(tt + delta, bit, reduction='none')
        loss = (l * msk).sum() / max(int(msk.sum()), 1)
        loss.backward()
        zurueck(keep)
        q = step / max(1, total)
        lr = (a.lr1 + (a.lr0 - a.lr1) * 0.5 * (1 + math.cos(math.pi * min(q, 1.0)))) * 1e-6
        if not opt.step(lr, a.clip):
            nskip += 1
        step += 1
    if dev.startswith('cuda'):
        torch.cuda.synchronize()
    say('epoche %d/%d lr=%.6g %.1fs steps=%d skip=%d' % (e, a.epochs, lr, time.time() - t0, spe, nskip))
    if e % a.eval_every == 0 or e == a.epochs:
        d, b = auswerten(e)
        if d < best:
            best = d
            BK.blob_write('%s_best.nblb' % a.out, b, 1)
            say('  best e%d delta_B=%+.1f' % (e, best))
        if e == a.epochs:
            BK.blob_write('%s_e%d.nblb' % (a.out, e), b, 1)
say('BLOB %s_best.nblb ENDE nach %.0f s, best %+.1f' % (a.out, time.time() - T0, best))
