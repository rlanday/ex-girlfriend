"""Repair rows whose largest magnitude has decayed below 1e-30.

    python fix_subnormal_rows.py <bf16 checkpoint in> <checkpoint out>

Rule: every row of every 2-D floating point tensor whose |max| is below 1e-30
is scaled so that its |max| becomes 1e-30 (its direction stays); rows that are
all zero stay unchanged but are counted.  Prints the number of rows found and
of matrices, then the first 40 (tensor, rows) pairs.

Why: after 200 more bf16 epochs (section 8 of the chain) some rows had maxima
down to 4.2e-42, below the smallest normal fp32 number (1.18e-38), and qat on
that state produced NaN from its second step on.  The epoch-123 state already
had such rows; the section 7 end (smallest row maximum 1.4e-35) and the
repaired state ran cleanly in a short qat probe (about 25 steps each).
probe_row_max.py lists the smallest row maxima of a checkpoint.

This is the code that ran on the section 8 bf16 end, with only the two paths
taken from the command line and the output in English: 77 rows in 35 matrices
(mlp.up, key/query projection and convolution, forget_gate_projection.down),
sha256 ce2ce0c933dd3569... -> 31,504,447 bytes, sha256
377e6a7feaf413d6d2dbc52dd3cf91f4415cd73608c8ada196d9f73e2819f1ab.
Run it with the trainer python (.venv/bin/python in the trainer tree).
"""
import sys
import torch
src=sys.argv[1]
d=torch.load(src,map_location="cpu",weights_only=False)
sd=d["model"] if isinstance(d,dict) and "model" in d else d
n=0; per={}
for k,v in sd.items():
    if not torch.is_tensor(v) or v.dim()!=2 or not v.is_floating_point(): continue
    m=v.float().abs().amax(1)
    idx=(m<1e-30).nonzero().flatten()
    if len(idx):
        per[k]=len(idx); n+=len(idx)
        f=v.float()
        for i in idx:
            mi=f[i].abs().max()
            f[i] = (f[i]/mi*1e-30) if mi>0 else f[i]
        sd[k].copy_(f.to(v.dtype))
print("repaired rows",n,"in",len(per),"matrices"); print(sorted(per.items())[:40])
torch.save(d,sys.argv[2])
