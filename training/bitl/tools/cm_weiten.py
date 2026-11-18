#!/usr/bin/env python3
# cm_weiten.py (weiten = widen) <in> <out> <cells_out> -- widen an NBLB blob of the BITLSTM head to more LSTM cells
# (numpy only, NBLB version kept: v1 f32 or v2 bf16 bit patterns copied unchanged).
# The new cells get all-zero weights and biases: z = win.x + bin = 0, a = z*sigmoid(z) = 0, gates i = f = o = 0.5,
# g = tanh(0) = 0, so c stays 0 and h = 0 for every new cell; the old cells read the new ones only through zero
# columns of wih/whh and the output layer gives them weight 0.  The widened head computes the same function; in the
# C++ head the extra terms are exact zeros, so the sums keep their value (checked in the replay, not assumed).
# Layout (src/bit_head.cpp Head, bg_kern.blob_read): win C x inw (row = cell), bin C, wih 4C x C and whh 4C x C (gate
# blocks i, f, g, o of C rows each, column = cell), bg 4C (same blocks), wout C, bout 1.
import struct, sys
import numpy as np

src, dst, C2 = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = open(src, 'rb').read()
assert d[:4] == b'NBLB', src
ver, C, nin, inw, n1 = struct.unpack_from('<Iiiii', d, 4)
assert ver in (1, 2) and C2 >= C and C2 % 16 == 0, (ver, C, C2)
dt = '<f4' if ver == 1 else '<u2'
sizes = [C * inw, C, 4 * C * C, 4 * C * C, 4 * C, C, 1]
v = np.frombuffer(d, dt, sum(sizes), 24)
assert len(d) == 24 + v.itemsize * sum(sizes)
o = 0
p = []
for s in sizes:
    p.append(v[o:o + s]); o += s
win, binv, wih, whh, bg, wout, bout = p
W = np.zeros((C2, inw), dt); W[:C] = win.reshape(C, inw)
B = np.zeros(C2, dt); B[:C] = binv
IH = np.zeros((4, C2, C2), dt); IH[:, :C, :C] = wih.reshape(4, C, C)
HH = np.zeros((4, C2, C2), dt); HH[:, :C, :C] = whh.reshape(4, C, C)
G = np.zeros((4, C2), dt); G[:, :C] = bg.reshape(4, C)
O = np.zeros(C2, dt); O[:C] = wout
body = np.concatenate([W.ravel(), B, IH.ravel(), HH.ravel(), G.ravel(), O, bout]).astype(dt).tobytes()
open(dst, 'wb').write(d[:4] + struct.pack('<Iiiii', ver, C2, nin, inw, n1) + body)
print('%s: C%d -> C%d, v%d, %d B' % (dst, C, C2, ver, 24 + len(body)))
