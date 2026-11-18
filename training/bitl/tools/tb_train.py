#!/usr/bin/env python3
# tb_train.py <store> <F> <T> <out_prefix> [options] -- offline trainer of a TEXT blob for the head
# of cm_kern.py (n1 39, keep 1..38 without 23, TF sketch 32), recipe "gross" (large batch) of cm_train.py:
# minibatches of --batch windows (SEQ 1024 from F), windows shuffled per epoch (seeded), ONE Adam (C++ form) for the
# run, cosine per step lr0 -> lr1 (1e-6 units), global L2 clip, mean BCE over the non-override bits.  Additions:
#   * pk1 host store (tb_kern.py), gathers in a prefetch thread (an error in the thread ends the run, never a short
#     epoch), steps per epoch counted against the plan
#   * checkpoint every --ckpt-every epochs (<out>.ckpt.pt: weights, Adam moments, step, epoch, plan E, RNG state, best)
#     and --resume (continues the same cosine; the plan E is kept)
#   * --bis 'YYYY-mm-dd HH:MM' (fresh start only): after epoch 1 the plan E is cut to what fits (x 0.92)
#   * <out>.STOP present after an epoch: checkpoint, exit 5 (pause; --resume continues bit-identically)
#   * previous bit 0.5 at F (the switch between the weight sets resets it, --no-vorbit-reset reads record F-1)
# Blobs: <out>_e<E>.nblb at --blobs epochs and at the end, <out>_best.nblb = best in-sample frozen eval (every
# --eval-every epochs and at the end).  Log <out>.log; marker <out>.FERTIG at the end.
import argparse, time, sys, os, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_kern as BK
import cm_kern as CK
import tb_kern as TK

ap = argparse.ArgumentParser()
ap.add_argument('store'); ap.add_argument('F', type=int); ap.add_argument('T', type=int); ap.add_argument('out')
ap.add_argument('--cells', type=int, default=32)
ap.add_argument('--zero-cols', default='')
ap.add_argument('--zero-hid', action='store_true')
ap.add_argument('--epochs', type=int, default=256)
ap.add_argument('--lr0', type=float, default=3000.0)
ap.add_argument('--lr1', type=float, default=30.0)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--seed', type=int, default=1112363057)
ap.add_argument('--clip', type=float, default=1.0)
ap.add_argument('--device', default='cuda')
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--bis', default='')
ap.add_argument('--eval-every', type=int, default=32)
ap.add_argument('--ckpt-every', type=int, default=4)
ap.add_argument('--blobs', default='16,32,64,128,192,256,384,512,768,1024')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--no-vorbit-reset', action='store_true')
ap.add_argument('--max-steps', type=int, default=0, help='probe: stop after this many steps of epoch 1')
ap.add_argument('--stop-after', type=int, default=0, help='test: exit after this epoch (simulated crash)')
ap.add_argument('--pidfile', default='')
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if a.pidfile:
    with open(a.pidfile, 'a') as f:
        f.write('%d %s\n' % (os.getpid(), os.path.basename(a.out)))
dev = a.device
LOG = a.out + '.log'
CK_PATH = a.out + '.ckpt.pt'


def say(s):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + s
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


zc = [int(c) for c in a.zero_cols.split(',') if c != '']
arm = CK.ArmK('t%d' % a.cells, cells=a.cells, zero_cols=zc, zero_hid=a.zero_hid)
fe = TK.lade_feld(a.store, a.F, a.T, vorbit_reset=not a.no_vorbit_reset)
m = BK.Kopf(arm).to(dev)
m.load_blob(CK.blob_init(arm, a.seed))
ps = m.params()
spe = fe.nfull // a.batch
E = a.epochs
total = spe * E
gen = torch.Generator().manual_seed(a.seed)
opt = BK.AdamC(ps)
step = 0; e = 0; best = None; resumed = False
if a.resume and os.path.exists(CK_PATH):
    c = torch.load(CK_PATH, map_location='cpu', weights_only=False)
    m.load_state_dict(c['model'])
    for t, v in zip(opt.m, c['adam_m']):
        t.copy_(v.to(dev))
    for t, v in zip(opt.v, c['adam_v']):
        t.copy_(v.to(dev))
    opt.b1t, opt.b2t, opt.n = c['b1t'], c['b2t'], c['n']
    step, e, E, best = c['step'], c['epoch'], c['E'], c['best']
    total = spe * E
    gen.set_state(c['gen'])
    resumed = True
    say('RESUME %s at epoch %d/%d step %d best %s' % (os.path.basename(a.out), e, E, step, best))
say('=== %s C%d window [%d,%d) = %d bits, %d windows (%d full), epochs %d, lr %g -> %g, batch %d, seed %d, '
    'clip %g, zero_x %s, nin %d inw %d, vorbit_reset %d, deadline %s, steps/epoch %d' % (
        os.path.basename(a.out), a.cells, a.F, a.T, fe.L, fe.nwin, fe.nfull, E, a.lr0, a.lr1, a.batch, a.seed, a.clip,
        arm.zero_x, arm.nin, arm.inw, int(not a.no_vorbit_reset), a.bis or '-', spe))
deadline = (int(a.bis) if a.bis.isdigit() else time.mktime(time.strptime(a.bis, '%Y-%m-%d %H:%M'))) if a.bis else 0
blobs = set(int(x) for x in a.blobs.split(',') if x)
T0 = time.time()


def save_ckpt():
    c = dict(model=m.state_dict(), adam_m=[t.detach().cpu() for t in opt.m], adam_v=[t.detach().cpu() for t in opt.v],
             b1t=opt.b1t, b2t=opt.b2t, n=opt.n, step=step, epoch=e, E=E, best=best, gen=gen.get_state(),
             args=vars(a))
    torch.save(c, CK_PATH + '.tmp')
    os.replace(CK_PATH + '.tmp', CK_PATH)


while e < E:
    e += 1
    m.train()
    t0 = time.time()
    perm = torch.randperm(fe.nfull, generator=gen).tolist()
    order = [sorted(perm[i * a.batch:(i + 1) * a.batch]) for i in range(spe)]
    rest = sorted(perm[spe * a.batch:]) + list(range(fe.nfull, fe.nwin))
    sb = sh = 0.0; nb = 0; nskip = 0; nst = 0
    for pre in TK.prefetch(fe, order):
        x, tt, bp, bit, ovr, valid = fe.batch(arm, None, dev, pre=pre)
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
        step += 1; nst += 1
        if a.max_steps and nst >= a.max_steps:
            break
    if a.max_steps:
        torch.cuda.synchronize()
        dt = time.time() - t0
        say('PROBE %d steps %.1f s = %.3f s/step -> %.0f s per epoch (%d steps), delta_B so far %+.1f on %d bits' % (
            nst, dt, dt / nst, dt / nst * spe, spe, (sh - sb) / 8, nb))
        sys.exit(0)
    assert nst == spe, 'epoch %d: %d of %d steps' % (e, nst, spe)
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
    if e == 1 and deadline and not resumed:
        fit = int((deadline - time.time()) / dt * 0.92) + 1
        if fit < E:
            E = max(2, fit); total = spe * E
            say('Frist %s: Epochen auf %d gekuerzt (%.1f s je Epoche)' % (a.bis, E, dt))
    if e in blobs:
        BK.blob_write('%s_e%d.nblb' % (a.out, e), m.blob(), 1)
    last = (e == E)
    if (a.eval_every and e % a.eval_every == 0) or last:
        r = BK.bewerten(m, fe, dev)
        say('  eval e%d in-sample frozen bits=%d base_B=%.1f head_B=%.1f delta_B=%+.1f' % (
            e, r['bits'], r['base_B'], r['head_B'], r['delta_B']))
        if best is None or r['delta_B'] < best:
            best = r['delta_B']
            BK.blob_write('%s_best.nblb' % a.out, m.blob(), 1)
            say('  best e%d delta_B=%+.1f' % (e, best))
    if e % a.ckpt_every == 0 or last:
        save_ckpt()
    if os.path.exists(a.out + '.STOP') and not last:
        save_ckpt()
        say('STOP-Datei: Halt nach Epoche %d von %d, Checkpoint geschrieben (weiter mit --resume)' % (e, E))
        sys.exit(5)
    if a.stop_after and e == a.stop_after and not last:
        say('STOP-AFTER %d (Test)' % e)
        sys.exit(3)
fin = '%s_e%d.nblb' % (a.out, e)
BK.blob_write(fin, m.blob(), 1)
say('BLOB %s ENDE nach %.0f s, best %s' % (fin, time.time() - T0, best))
open(a.out + '.FERTIG', 'w').write(time.strftime('%F %T') + '\n')
