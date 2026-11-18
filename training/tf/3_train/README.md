# Stages (d) and (e): training and export of the transformer

The model, the start state, the chain and the export are described in sections
5 and 6 of `../../README.md`. This page lists the files and the commands.

| file | what it is |
|---|---|
| `tree/` | the training scripts of `cmix-neif-beehive` (commit `92c07c0b`) with the changes of this version, marked `# 2.0`; they run on top of the training tree (`TFT_TREE`) |
| `tree/rx7q.py` | resumable bf16 run (blocks b1 to b3) |
| `tree/rx7r.py` | `rx7q.py` plus the initialisation (`init_only`) and the run of the quantization-aware stage |
| `tree/rx3.py`, `tree/rx3a.py` | model and loader setup, the fp32 stage and the strict evaluation |
| `tree/arch.py`, `attnres_v10.py`, `attnres_hub.py` | attention residuals (fused operator, the hub mode is not used) |
| `tree/fisher_alloc_a.py` | allocation of 2, 3 or 4 bits per weight row |
| `tree/c1_rate_v2.py`, `tree/fakequant.py` | size term and fake quantization of the quantization-aware stage |
| `tree/fastmuon.py`, `tree/speedpatch.py` | Muon batched by shape, throughput patch of the KDA kernels (same values) |
| `tree/safe_prefetch.py` | data-loader prefetch that raises a failure in the training loop |
| `tree/exp.env` | the training-form environment of `cmix-neif-beehive` (micro-batch 262,144); `reproduce.sh` sets the environment of the shipped chain itself |
| `container_to_tch.py` | weight container to training state (start state, section 5.2) |
| `mlp_lift.py` | gives all-zero weight rows the value 1e-30 before the quantization-aware stage, so their scales stay finite |
| `chain/` | the executed configurations (`<W>` = work directory) and `results/` |
| `reproduce.sh` | runs the chain stage by stage |
| `export/export_v3_ar.py` | checkpoint to reference weight file (with the tensors of the attention residuals and the prior head) |
| `export/export4/` | bf16 rounding, formats v3, v4 and v5, dead neurons, collapsed rows |
| `export/np2/np2_container.py` | writer and reader of the container formats NP and NP2 in Python (NumPy, one core, a few minutes); `np <v5> <out> k fw np2 dead` writes the shipped container, `read <container> [<raw>]` decodes one |
| `export/np2/tf1_np_codec.cpp` | the same writer in C++ (built in the development tree, `../../README.md` section 6) |

```bash
# the start state needs the container of cmix-neif-beehive in <W>/start/beehive.c4
TFT_TREE=<training tree> PKG=<training package> W=<work directory> NGPU=8 bash reproduce.sh
TFT_TREE=<training tree> PKG=<training package> W=<work directory> bash reproduce.sh export
```

The chain ran in stages with automatic restarts at epoch ends; `rx7q.py` and
`rx7r.py` resume from their last saved state when started again with the same
configuration.
