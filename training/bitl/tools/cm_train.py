#!/usr/bin/env python3
# cm_train.py <store> <F> <T> <out_prefix> [options] -- offline trainer of the metablock blob for the
# head of cm_kern.py, recipe "gross" (large batch) of an earlier GPU trainer: minibatches of --batch windows (SEQ 1024),
# windows shuffled per epoch (seeded), one Adam (C++ form) for the run, cosine per step lr0 -> lr1 (1e-6 units),
# global L2 clip, mean BCE over the non-override bits.
# Deadline (--bis 'YYYY-mm-dd HH:MM'): after epoch 1 the epoch count is cut to what fits before the deadline
# (x 0.92) and the cosine is laid over that count; after every epoch the run stops if the next epoch would end
# after the deadline.  The last blob is always written; with --eval-every the best in-sample frozen blob is kept
# as <out>_best.nblb.  --pidfile: own PID appended (the b2 automaton may end it).
import argparse, time, sys, os, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_kern as BK
import cm_kern as K

ap = argparse.ArgumentParser()
ap.add_argument('store'); ap.add_argument('F', type=int); ap.add_argument('T', type=int); ap.add_argument('out')
ap.add_argument('--cells', type=int, default=16)
ap.add_argument('--zero-cols', default='', help='trace columns (kept set) forced to 0')
ap.add_argument('--zero-hid', action='store_true')
ap.add_argument('--epochs', type=int, default=512)
ap.add_argument('--lr0', type=float, default=10000.0)
ap.add_argument('--lr1', type=float, default=100.0)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--seed', type=int, default=1112363057)
ap.add_argument('--clip', type=float, default=1.0)
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--bis', default='')
ap.add_argument('--eval-every', type=int, default=32)
ap.add_argument('--pidfile', default='')
ap.add_argument('--ckpt', default='64,128,256,384,512,768,1024')
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if a.pidfile:
    with open(a.pidfile, 'a') as f:
        f.write('%d %s\n' % (os.getpid(), os.path.basename(a.out)))
dev = a.device
LOG = a.out + '.log'
deadline = (int(a.bis) if a.bis.isdigit() else time.mktime(time.strptime(a.bis, '%Y-%m-%d %H:%M'))) if a.bis else 0


def say(s):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + s
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


zc = [int(c) for c in a.zero_cols.split(',') if c != '']
arm = K.ArmK('m%d' % a.cells, cells=a.cells, zero_cols=zc, zero_hid=a.zero_hid)
fe = K.lade_feld(a.store, a.F, a.T, device=dev)
m = BK.Kopf(arm).to(dev)
m.load_blob(K.blob_init(arm, a.seed))
ps = m.params()
spe = fe.nfull // a.batch
E = a.epochs
total = spe * E
say('=== %s C%d window [%d,%d) = %d bits, %d windows (%d full), epochs %d, lr %g -> %g, batch %d, seed %d, '
    'clip %g, zero_x %s, nin %d inw %d, deadline %s' % (os.path.basename(a.out), a.cells, a.F, a.T, fe.L, fe.nwin,
                                                        fe.nfull, E, a.lr0, a.lr1, a.batch, a.seed, a.clip,
                                                        arm.zero_x, arm.nin, arm.inw, a.bis or '-'))
ckpts = set(int(x) for x in a.ckpt.split(',') if x)
gen = torch.Generator().manual_seed(a.seed)
opt = BK.AdamC(ps)
step = 0
best = None
T0 = time.time()
e = 0
while e < E:
    e += 1
    m.train()
    t0 = time.time()
    perm = torch.randperm(fe.nfull, generator=gen).tolist()
    order = [sorted(perm[i * a.batch:(i + 1) * a.batch]) for i in range(spe)]
    rest = sorted(perm[spe * a.batch:]) + list(range(fe.nfull, fe.nwin))
    sb = sh = 0.0; nb = 0; nskip = 0
    for wins in order:
        x, tt, bp, bit, ovr, valid = fe.batch(arm, wins, dev)
        for p in ps:
            p.grad = None
        m.lstm.bias_hh_l0.grad = None
        delta = m(x)
        with torch.no_grad():
            b_, h_, n_ = BK.kosten(delta.detach(), tt, bp, bit, ovr, valid)
        sb += b_; sh += h_; nb += n_
        msk = (valid & ~ovr)
        l = torch.nn.functional.binary_cross_entropy_with_logits(tt + delta, bit, reduction='none')
        loss = (l * msk).sum() / max(int(msk.sum()), 1)
        loss.backward()
        q = step / max(1, total)
        lr = (a.lr1 + (a.lr0 - a.lr1) * 0.5 * (1 + math.cos(math.pi * min(q, 1.0)))) * 1e-6
        if not opt.step(lr, a.clip):
            nskip += 1
        step += 1
    with torch.no_grad():
        for i in range(0, len(rest), 512):
            x, tt, bp, bit, ovr, valid = fe.batch(arm, rest[i:i + 512], dev)
            b_, h_, n_ = BK.kosten(m(x), tt, bp, bit, ovr, valid)
            sb += b_; sh += h_; nb += n_
    if dev.startswith('cuda'):
        torch.cuda.synchronize()
    dt = time.time() - t0
    say('epoche %d/%d lr=%.6g %.1fs steps=%d skip=%d bits=%d base_B=%.1f head_B=%.1f delta_B=%+.1f' % (
        e, E, lr, dt, spe, nskip, nb, sb / 8, sh / 8, (sh - sb) / 8))
    if e == 1 and deadline:
        fit = int((deadline - time.time()) / dt * 0.92) + 1
        if fit < E:
            E = max(2, fit); total = spe * E
            say('Frist %s: Epochen auf %d gekuerzt (%.1f s je Epoche)' % (a.bis, E, dt))
    if e in ckpts:
        BK.blob_write('%s_e%d.nblb' % (a.out, e), m.blob(), 1)
    last = (e == E) or (deadline and time.time() + dt * 1.05 > deadline)
    if (a.eval_every and e % a.eval_every == 0) or last:
        r = BK.bewerten(m, fe, dev)
        say('  eval e%d in-sample frozen bits=%d base_B=%.1f head_B=%.1f delta_B=%+.1f' % (
            e, r['bits'], r['base_B'], r['head_B'], r['delta_B']))
        if best is None or r['delta_B'] < best:
            best = r['delta_B']
            BK.blob_write('%s_best.nblb' % a.out, m.blob(), 1)
            say('  best e%d delta_B=%+.1f' % (e, best))
    if last:
        if e < E:
            say('FRIST: Halt nach Epoche %d von %d' % (e, E))
        break
fin = '%s_e%d.nblb' % (a.out, e)
BK.blob_write(fin, m.blob(), 1)
say('BLOB %s ENDE nach %.0f s' % (fin, time.time() - T0))
