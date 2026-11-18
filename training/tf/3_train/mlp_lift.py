#!/usr/bin/env python3
"""mlp_lift.py <in.tch> <out.tch>: before the qat init of a tail: every ALL-ZERO row of a 2-D weight gets
the value 1e-30 in its first column (the lifting value of fix_subnormal_rows.py).  Otherwise the qat scale init gives
that row the scale 0, the fake-quant forward computes 0/0 = NaN, the activation scales become NaN, and the C1 setup refuses
('Unlifted zero MLP scale' / 'Nonfinite MLP activation scale', c1_rate_v2.snapshot_export_topology).  A lifted row stays
functionally zero (|w| 1e-30); the exporter zeroes it again (rows below 2^-20) or strikes it (dead MLP neuron).
Writes out == in when no row is zero.  Zero rows of the start state: rows the exporter of
cmix-neif-beehive had zeroed (85), dead neurons re-inserted as zero rows (102), sentinel rows zeroed for the
new stream form."""
import sys

import torch

sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
n, t = 0, 0
for k, v in sd.items():
    if k.endswith(".weight") and torch.is_tensor(v) and v.dim() == 2 and v.is_floating_point():
        z = (v.float().abs().amax(1) == 0).nonzero().flatten()
        if len(z):
            v[z, 0] = torch.tensor(1e-30, dtype=torch.float32).to(v.dtype)
            n += len(z)
            t += 1
torch.save(sd, sys.argv[2])
print("mlp_lift: %d all-zero rows in %d weights lifted to 1e-30" % (n, t), flush=True)
