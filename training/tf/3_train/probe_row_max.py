"""List all-zero rows/columns and the smallest row maxima of checkpoints.

    python probe_row_max.py <checkpoint> [<checkpoint> ...]

For each checkpoint: the number of 2-D tensors with all-zero rows or columns
(with the first six), and the four smallest row maxima with their tensor
names.  A row maximum below 1.18e-38 (subnormal in fp32) makes qat run into
NaN; repair such a state with fix_subnormal_rows.py before qat.

Values seen in this chain: section 7 end 1.40e-35; epoch 123 of section 8 and
the section 8 bf16 end 4.21e-42 (blocks.2.attention.forget_gate_projection.down
and others); no all-zero rows or columns.
This is the code of the diagnosis run, with the checkpoint paths taken from
the command line and the output in English.  lade = load.
"""
import sys
import torch
def lade(p):
    d=torch.load(p,map_location="cpu",weights_only=False)
    if isinstance(d,dict) and "model" in d: d=d["model"]
    return {k:v.float() for k,v in d.items() if torch.is_tensor(v)}
for n,p in [(a,a) for a in sys.argv[1:]]:
    S=lade(p); z=0; zt=[]; tiny=[]
    for k,v in S.items():
        if v.dim()!=2: continue
        r=(v.abs().amax(1)==0).sum().item(); c=(v.abs().amax(0)==0).sum().item()
        if r or c: zt.append((k,r,c))
        m=v.abs().amax(1).min().item()
        tiny.append((m,k))
    tiny.sort()
    print(n,"matrices with zero rows/columns:",len(zt), zt[:6])
    print("   smallest row maxima:",[("%.2e"%m,k) for m,k in tiny[:4]])
