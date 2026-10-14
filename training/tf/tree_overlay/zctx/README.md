# `zctx/`: the training form of the transformer

The modules here make the training rows match what the coder feeds the
transformer, with gate zones, prompts, sentinel rows, the predecessor and match
channels, the int64 packing and the loaders. `tft/runner.py` selects the
loader with `ZCTX_ARM`; the chain documented in `training/README.md` uses
`pkd`, the loader that reads the coder's own fed-row dump (`zpdump.py`). The
other arms are the earlier forms that led to it. The training form is
described in section 4 of `training/README.md`, the link-context rule in
`SPEC_Z3.md`.

`zloader.py`, `zploader.py`, `zpdump.py`, `zmodel.py`, `zpmodel.py`,
`zmatch.py` and `zabc.py` derive from Vladimir Ivanov's `pysrc/data.py` and
`pysrc/model.py` (fx2-cmix-transformer, GPL-3); they subclass his classes and
override single methods, and `pysrc/` is not changed. Each file names its
origin in its header.

| file | role |
|---|---|
| `zpdump.py` | loader of the shipped weights (arm `pkd`): reads the rows the coder fed; the same file as `tf/2_package/zpdump.py` |
| `dumpv2.py` | reader of the dump format v2; the same file as `tf/2_package/dumpv2.py` |
| `zploader.py`, `zpmodel.py`, `zpaket.py` | the combined context package as one arm (`pk`), with the rows rebuilt in Python |
| `zgrammar.py` | stream grammar (article separator, title end, link token, paragraph) and `verify`, which re-derives it from a package |
| `z3_extract.py` | the link-context extractor, its output format and checker (`SPEC_Z3.md`) |
| `zloader.py`, `zmodel.py`, `zfeat.py` | the first arms: article ordinal and stream position as a model input (`z1`), the link-context prefix (`z3`) |
| `zlast.py` | the last stream tokens as a direct model input (the predecessor channel) |
| `zmatch.py` | the compressor's match model as an input channel |
| `zgate.py`, `gate_dry.py`, `gate_eval.py` | the causal gating rule for digit zones, its dry run and its evaluation |
| `zonemap.py`, `zonemap2.py`, `zone_eval.py` | where the transformer gains or loses, per zone of the stream |
| `zabc.py` | two further arms on the `z1` packing |
| `eval_zctx.py` | turns the loss difference of two runs into bytes |
| `slice_lite.py` | cuts a small package without priors for dry runs |
| `dryrun_zctx.py` | checks that need no GPU |
| `run_zctx.sh` | start script per arm |
| `__init__.py` | package marker |

The zone, class and stage labels of `zonemap.py` and `zonemap2.py` and a few
TSV column names are German identifiers.
