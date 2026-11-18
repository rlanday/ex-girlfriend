#!/usr/bin/env python3
# tb_zone1.py <spurmarken.txt> <article_offsets.tsv> [k=143227] -- limits of text zone 1 of the BITLSTM trace
# of the trace run (coder stream c35d3073, 501.062.281 B), as the switch between the three weight sets defines them:
#   F = first 1.024 grid point after the metablock (zone 3 ends at record 30.990.992) = 30.991.360
#   T = first 1.024 grid point at or after the first head bit of page k (calendar block, k 143.227 of the v4 order)
# The trace has no byte field (ADEPTH skips bits and whole bytes), so the record index of the page start is read off
# the mark writer of the trace run (every 30 s: progress line "<p> <archive>", sizes of spur.bin and kanal.bin):
#   * records at a mark: kanal.bin rows x 8 (one row per 8 head bits; stdio flushes in 4.096 B = 27,7 rows, so the
#     produced count is in [8 rows, 8 rows + 229]); spur.bin lags by up to 65.536 records (block buffer)
#   * coder byte at a mark: in the progress bin [p, p + 0.01) % of the stream
#   * bound: last mark whose bin ends at or before the page start (coder still before the page) gives LO, the first
#     mark whose bin starts at or after it (coder past the page start) gives HI: LO <= record of the page start < HI
#   * estimate: two record rates per second (before / in the calendar block, below)
import sys, bisect
L = 501062281
F = 30991360
K = int(sys.argv[3]) if len(sys.argv) > 3 else 143227
koff = {}
for l in open(sys.argv[2]):
    if l.startswith('#'):
        continue
    f = l.split()
    if len(f) >= 3 and f[0].isdigit():
        koff[int(f[0])] = int(f[1])
assert len(koff) == 243425, len(koff)
PB = koff[K]
M = []
for l in open(sys.argv[1]):
    if l.startswith('#'):
        continue
    f = l.split()
    if len(f) < 8:
        continue
    try:
        t, p, kb = int(f[2]), float(f[3]), int(f[6])
    except ValueError:
        continue
    rows = (kb - 28) // 148
    M.append((t, p, 8 * rows, f[0] + ' ' + f[1]))
M.sort()
lo_b = lambda p: int(round(p * 100)) * L // 10000          # first byte of bin p
hi_b = lambda p: (int(round(p * 100)) + 1) * L // 10000    # first byte after bin p
i1 = max(i for i, m in enumerate(M) if hi_b(m[1]) <= PB)
i2 = min(i for i, m in enumerate(M) if lo_b(m[1]) >= PB)
LO, HI = M[i1][2], M[i2][2] + 229
print('page k %d starts at stream byte %d = %.4f %%' % (K, PB, 100.0 * PB / L))
print('bracketing marks: %s p %.2f (bin [%d, %d)) records >= %d  |  %s p %.2f (bin [%d, %d)) records <= %d'
      % (M[i1][3], M[i1][1], lo_b(M[i1][1]), hi_b(M[i1][1]), M[i1][2], M[i2][3], M[i2][1], lo_b(M[i2][1]),
         hi_b(M[i2][1]), HI))
print('hard bound: %d <= R(page start) < %d  (%d records = %.0f windows)' % (LO, HI, HI - LO, (HI - LO) / 1024.0))
# estimate: the interval i1..i2 splits into tau seconds before the page start at the pre-calendar record rate and the
# rest at the calendar rate (rates in records per second from the 3 neighbouring intervals on each side, kanal sizes;
# the coder time per byte is TF-bound, the rate per second is stable to a few per cent, the rate per byte is not
# observable finer than a 0.01 % bin); spread = all combinations of the min/max neighbouring rates
pre = [(M[k][2] - M[k - 1][2]) / float(M[k][0] - M[k - 1][0]) for k in range(i1 - 2, i1 + 1)]
post = [(M[k + 1][2] - M[k][2]) / float(M[k + 1][0] - M[k][0]) for k in range(i2, i2 + 3)]
Q = M[i2][2] - M[i1][2]; D = float(M[i2][0] - M[i1][0])
def split(rp, rc):
    tau = (Q - D * rc) / (rp - rc)
    return tau, M[i1][2] + rp * tau
rpre = sum(pre) / 3; rpost = sum(post) / 3
tau, RE = split(rpre, rpost)
alle = [split(a_, c_)[1] for a_ in (min(pre), max(pre)) for c_ in (min(post), max(post))]
print('record rate before %.0f /s (%s), after %.0f /s (%s); interval %s -> %s: %d records in %.0f s'
      % (rpre, ', '.join('%.0f' % x for x in pre), rpost, ', '.join('%.0f' % x for x in post), M[i1][3], M[i2][3], Q, D))
v = (hi_b(M[i2][1]) + lo_b(M[i2][1]) - hi_b(M[i1][1]) - lo_b(M[i1][1])) / 2.0 / D
print('split: page start %.2f s after %s; implied coder byte at that mark %d (bin %s), calendar speed %.0f B/s'
      % (tau, M[i1][3], PB - int(v * tau), 'ok' if lo_b(M[i1][1]) <= PB - int(v * tau) < hi_b(M[i1][1]) else 'NICHT im Bin', v))
R = int(round(RE)); spread = int(max(abs(x - RE) for x in alle))
R = min(max(R, LO), HI)
T = -(-R // 1024) * 1024
print('R(page start) ~ %d, rate spread [%d, %d] (+- %d records)' % (R, int(min(alle)), int(max(alle)), spread))
print('F = %d (= 30.265 x 1.024)  T = %d (= %d x 1.024)  zone 1 = %d records = %d windows'
      % (F, T, T // 1024, T - F, (T - F) // 1024))
print('T band from the hard bound: [%d, %d] = %d windows; from the rate spread [%d, %d] = +- %d windows'
      % (-(-LO // 1024) * 1024, -(-HI // 1024) * 1024, (-(-HI // 1024) - (-(-LO // 1024))),
         -(-int(min(alle)) // 1024) * 1024, -(-int(max(alle)) // 1024) * 1024, spread // 1024 + 1))
