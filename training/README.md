# `training/`: how the pre-trained parts of this entry were produced

This directory is **documentation and tooling, not part of the entry**.
`build.sh` never enters it; nothing here is compiled into `archive9` or into the
decompressor. It serves reproducibility and attribution.

1. **Reproducibility.** Two parts of the compressor are trained offline and
   shipped as frozen weights, the byte-level **transformer** (a weight container
   appended to both executables) and the **BITLSTM**, the small LSTM that
   corrects the final probability behind the mixer (7,386 B of bfloat16
   weights compiled into the program). This directory describes, step by step
   and with checksums, how both were produced from `enwik9` alone, and ships
   the tools that did it. Section 9 says what can be rerun bit-exactly and what
   cannot.
2. **Attribution.** The transformer and its trainer descend from Vladimir
   Ivanov's `fx2-cmix-transformer`; the BITLSTM, name and design, comes from
   David Freelan's `bitlstm32` in `cmix-obias`; the compressor descends from
   cmix / fx-cmix / cmix-lex. `ATTRIBUTION.md` names the borrowed parts, their
   authors and their licences.

Everything in this directory is GPL-3, like the rest of the entry, except the
third-party files listed in `../third_party/README.md` (copies in
`tf/tree_upstream/src/ds/`, altered `avx_mathfun.h` routines in
`tf/tree_upstream/cpp_infer/` and in `bitl/trainer/bit_head.cpp`) and the
Wikipedia excerpts in `tf/tree_upstream/prof_input/`.

---

## 0. The short version

```
enwik9
  |  (a) the compressor's own preprocessing, a "-e" run of the stream tool
  v
coder stream  552,750,198 B, 208 distinct byte values          tf/1_dump/README.md
  |
  +--(b) dump run (cmix -n, dump mode)    --> fed-row dump, zone mask, match channel
  +--(b) prior run (cmix -n, PPMd only)   --> PPMd prior per stream byte
  |
  v (c) package build                                         tf/2_package/
training package: 593,115,275 rows, 547,586,001 stream tokens, 227.8 GB of priors
  |
  v (d) GPU training: 1,000 epochs bf16 (sections over successive     tf/3_train/
  |     stream forms), subnormal rows lifted,
  |     32 epochs quantization-aware (int4 weights, int8 activations), 6 epochs fp32
  v
checkpoint (.tch) --> reference weight file (.bin) + manifest (.bin.json)
  |
  v (e) export: bf16 rounding, format v3, dead neurons removed,     tf/4_export/
  |     collapsed rows zeroed, format v4
weights-…_v4.c4   3,716,667 B, sha256 ba982cd5…   (appended to both executables)

enwik9 --(f) trace run ("-e", trace build) --> trace of the BITLSTM's inputs
          --> cut at 418,837,176 bits --> replay training, 32 passes      bitl/
          --> bfloat16 --> 7,386 B, sha256 c72957f4… (compiled into the program)
```

The same coder stream is the source of the dictionary, hint, order and PGO
artefacts (`data/README.md`).

**The coder is the specification.** Every prompt, gate zone, sentinel row and
side channel that the coder shows the transformer must be present in training,
byte for byte. The training rows are therefore not reconstructed in Python; they
are **dumped by the coder itself** and read back by the trainer. The contract is
the record format of the dump (`tf/1_dump/README.md`), as read by
`tf/2_package/dumpv2.py`, together with the training form of section 4.
`dumpv2.py` is the single reader that package build, acceptance tests and data
loader share.

---

## 1. Prerequisites

### Hardware

| step | needs | measured for the shipped parts |
|---|---|---|
| coder stream (a) | 1 CPU core | about 1 min (prepare-only run) |
| dump and prior runs (b) | 1 CPU core and 9 GB RAM each, about 250 GB free disk | tools of `tf/1_dump`, one i9-13900 core: 22 to 25 min (dump, 6.5 GB peak RSS); 1 h 2 min to 1 h 12 min (prior run, 8.6 GB peak RSS) plus 12 to 14 min trimming and probes |
| package build (c) | 1 CPU core, 32 GB RAM | about 1 min for the row package (28 GB peak RSS) |
| GPU training (d) | 4 GPUs with 32 GB each | on 4 cards of the RTX 5090 class about 6 min per bf16 epoch, 22 min per qat epoch, 32 min per fp32 epoch |
| export (e) | CPU only, numpy and torch | about 2 min |
| trace run (f) | 1 CPU core, about 40 GB disk for the first 20 % | about 14 h to the cut |
| replay training (f) | 1 CPU core | 3 h 10 min for 32 passes |

The priors file `ppmd-probs.float16` is **227,795,776,416 B** (208 float16 per
token). It is the only large artefact of the transformer pipeline and the reason
the training host needs either a large local disk or a fast HTTP fetch.

### Software

* a C++17 compiler with AVX2+FMA (clang 17 and 18 were used)
* Python 3.11 (the upstream lock file requires `>=3.11,<3.12`), NumPy
* PyTorch (CUDA build) for training; `fla` (flash-linear-attention) and `triton`
  for the KDA layers. These are upstream dependencies, unchanged; the shipped
  weights were trained with torch 2.12.1, fla 0.5.1, triton 3.7.1, numpy 2.4.6
  and CPython 3.11.15
* no network access is required for any computation; every input is `enwik9`
  itself or derived from it by the tools listed here (the three tools of
  `tf/1_dump/` are built from the source of this entry), the trace build of the
  development tree (section 10a) or one of the shipped artefacts of
  `data/README.md`

### The training tree: upstream base and the cmix-neif overlay

The trainer is not a program of its own that happens to use an upstream library.
It **is** the upstream tree with the cmix-neif files laid on top. To keep that
boundary visible file by file, this package ships the two halves separately:

```
tf/tree_upstream/    the fx2-cmix-transformer tree, verbatim, 189 files, about
                     4.3 MB; LICENSE and every copyright header untouched
tf/tree_overlay/     cmix-neif files only: tft/, zctx/, run/,
                     paket11_warmstart.py
```

To build the working tree:

```bash
cp -a tf/tree_upstream <work>
cp -a tf/tree_overlay/. <work>/           # adds tft/, zctx/, run/, the warm start
python3 tf/2_package/patch_paket11.py <work>   # idempotent, individually marked edits
```

`patch_paket11.py` writes no upstream file. Its edits go to the overlay files
(`zctx/`, `tft/`), and it copies three helpers (`dumpv2.py`, `zpdump.py`,
`paket11_warmstart.py`) into the working tree. The overlay files of this
package already carry its edits, so it reports "already set" for every edit
and only copies the helpers; it is kept because the helpers come from
`tf/2_package/`.

**The overlay is the code the shipped weights were trained with.** One upstream
file differed on the training hosts. There `pysrc/data.py` carried an optional
threaded loader that is off by default and was not used; the upstream file is
shipped.

The upstream trained checkpoints, `models/` (71 MB), are not shipped. They are
not needed to reproduce anything here, and this entry does not use them.

Which upstream snapshot: the overlay was written against the 2026-07-24 release,
and that is what `tf/tree_upstream/` contains. A later upstream snapshot
(2026-08-21) exists; its `pysrc/` is byte-identical to this one. The inference
engine of the compressor (`src/tf1/`) follows the 2026-08-21 `cpp_infer/`. See
`ATTRIBUTION.md`.

---

## 2. Stage (a): the coder stream

The transformer is trained on the **coder stream**, i.e. `enwik9` after the
compressor's own preprocessing (dictionary/WRT substitution, article
reordering, comment-index extraction, the metadata block at the head of the
stream, the redirect block, the page form). It is *not* trained on raw `enwik9`,
and it is **hard-coupled to this exact preprocessing**, so changing the
preprocessing invalidates the model and forces a retrain.

| property | value |
|---|---|
| length | 552,750,198 B (the coder sees 552,750,204 B with its 6-byte segment header) |
| sha256 | `8e0b264a777cb56a6be48c58002450c4711b1cebf3ffcb0129a32ef34273c3ba` |
| distinct byte values | exactly 208 (`vocab-bitmap.u8`, popcount 208) |
| article order | md5 of `.main_reordered` `28ca879dcfdafe7833b042f58cfee517` (`data/README.md` section 3) |
| gate zone | segment bytes 1 to 5,164,202: the metadata block (section 4.3 of the top-level `README.md`); withheld from the transformer |
| training tokens | 547,586,001, from segment byte 5,164,203 to the end |

A compression run of the submitted build codes exactly this stream.

**A stream produced by `cmix -c` / `-s` / `-n` from raw input is NOT this
stream.** Those modes skip the XML preprocessor. Stamp tables and dump rows only
from a stream that a `-e` run produced; `-n` is used below only on such a stream.

---

## 3. Stage (b): the coder-side runs

Three coder-side runs produce everything the trainer reads, namely the stream
run of stage (a), the **dump run** (dump-only mode: every producer that feeds
the transformer runs, its forward pass is skipped) and the **prior run**
(PPMd-only mode). `tf/1_dump/trainmodes.patch` adds the three modes to a copy
of the source of this entry, `tf/1_dump/build_dump_tools.sh` builds the tools;
run lines, outputs and checksums are in `tf/1_dump/README.md`. The dump is
7,117,383,324 B, 593,115,277 records of 12 B each including a header record.

**Target rule** (established on the material and checked by the package build):
*the target of row j is the next stream row of the same document*; inserted rows
(prompts, sentinels, the heading reminder) are skipped. The last row of a
document targets the document's closing token, which is coded but never fed and
comes from `doc-last-ziel.int32`. The simpler rule "row j+1" is wrong where
prompt rows of the next page follow the last stream row of a page.

---

## 4. Stage (c): the training package

`tf/2_package/build_package.sh` lists the steps with their commands:

1. cut at the end of the gate zone (zone mask: zone 3 on bytes 1 to 5,164,202,
   nothing withheld behind it);
2. `mk_pkgdump.py`: the rows of the package, the document table and the closing
   tokens, from the dump, with the arms of the coder and the attention window
   2,048;
3. the title lengths from the page table of the stream;
4. `ppmd-bytes.uint8` (token index per stream position) and the `prior0` table,
   the replacement prior of inserted rows: `float16((count + 1) / (N + 208))`
   over the training tokens, sha256 `591218150f09…`;
5. the priors: keep the rows of stream bytes behind the metadata block, drop
   the rows of bytes the coder injects (section 3 of `tf/1_dump/README.md`);
6. acceptance: the loader against the raw dump (`loader_vs_dump.py`), the
   structural check of the trainer (`tft.dataset check`), a full pass of the
   dump loader on the CPU (`loader_probe.py`), and the table of package
   properties (`package_checks.py`), which checks for no document longer than
   2^17 rows, every scored position scored once, no sentinel targets, the
   prompt rows masked as the coder masks them, zone mask behind the cut all
   zero, prior rows summing to 1 within float16 precision, and the prior of row
   *i* predicting token *i+1*.
   One row compares block counts with a reference package; it fails whenever
   the article order differs from that reference, as it does for this package.

### What the package of the shipped weights looks like

| | value |
|---|---:|
| rows | 593,115,275 |
| stream rows / inserted rows | 547,413,565 / 45,701,710 |
| sentinel rows | 7,153,276 |
| redirect page rows | 2,901,367 |
| rows with the replacement prior (`prior0`) | 39,319,969 |
| **scored positions** | **547,585,942**, each exactly once |
| documents | 172,437; longest 131,037 rows; none above 2^17 |
| `zeilen.dump` | 7,117,383,300 B, sha256 `16ed3882…` |
| `ppmd-bytes.uint8` | 547,586,001 B, sha256 `bc419ee2…` |
| `ppmd-probs.float16` | 227,795,776,416 B, sha256 `3ee22a26…` |
| `z3-prior0.float16` | 416 B, sha256 `591218150f09541909a980fca4c23fa044556bc1dcd2a5f592010103070deed6` |

The full list with checksums is in `MANIFEST.tsv` (rows starting with `REF:`).

### The training form, in one page

* **Document layout:** on pages of the redirect block the redirect row, then the
  page rows (page number, page id, contributor, date), then the title, then the
  prompt rows (synonyms, link context, aliases), then the body with its insertions (contributor row, heading reminders, digit-run
  classes). The content of each kind is described in section 4.2 of the
  top-level `README.md`. The title is fed and predicted; prompt and page rows
  are fed and never scored. Every inserted row carries `src = -1`. It receives
  the replacement prior vector `prior0` instead of a PPMd prior, except the last
  row of an inserted block, which receives the PPMd row of the next stream byte,
  as in the coder.
  **`prior0` must be the same on the coder side**, so the compressor carries the
  table of this package (`src/tf1/tf1_prior0_set_v9.h`).
* **Sentinel (`SENT`):** after every digit run, a row with token index
  `V + class` (V = 208, 8 classes, so 208 ... 215). The row *digit -> SENT*
  is the scored one; the row *SENT -> X* is muted, so X is scored exactly once,
  as the coder codes it once.
* **Predecessor channel:** the two preceding tokens of the **fed** sequence of
  the document, across gate gaps, as the decoder sees them. Document start is
  its own row (`V + 8 = 216`) in the predecessor table.
* **Match channel:** the match model's predicted byte and
  `min(raw bit length / 8, 15)` per row, looked up by the stream position of the
  input. Inserted rows get row 208 (no mark).
* **Packing:** one int64 per input row, bits 0 to 7 token (0 to V+7), 8 to 15
  predecessor t-1, 16 to 23 predecessor t-2, 24 to 32 match byte (9 bit, 208 = no
  mark), 33 to 36 match length class. The coder passes the same four values as
  separate fields (`src/tf1/tf1_paket.h`).
* **Piece cut:** documents are cut at 2^17 = 131,072 **fed rows**, the length of
  the RoPE table; the loader checks that no document exceeds it.
* **Redirect pages:** a run of consecutive redirect pages and the page that
  follows it form one document.
* **Vocabulary:** V = 208, frozen; the bitmap is 32 B with popcount 208.

The exported weights record the form they were trained on, with the list of
arms (prompt and channel options) that the dump contained, the `prior0` sha256,
the window and the row layout (`tf/4_export/`, export manifest). The arm list is a
record of the coder version that wrote the dump; the trainer does not interpret
it, because the rows come from the dump.

---

## 5. Stage (d): GPU training

### Geometry

`poc8m_w2048`: the upstream 8 M configuration on the 208-value vocabulary of
cmix-neif, attention window 2,048.

| V | d_model | layers | heads x d_head | d_mlp | KDA pattern | window |
|---:|---:|---:|---|---:|---|---:|
| 208 | 192 | 16 | 3 x 64 | 768 | `[T,T,T,F] x 4` (12 KDA layers, 4 attention layers) | 2,048 |

`tft/param_count.py` is the registry of geometries. Window variants differ
**only** in `window_size` and have the same tensor shapes, so a checkpoint can
be continued at another window. Engine constraints: `n_layers % 4 == 0`, `d_model % 16 == 0`.

### Batching and schedule

* Micro-batch 131,072 tokens per rank, optimizer batch 2^20 = 1,048,576 tokens,
  so gradient accumulation is 2 on 4 cards or 4 on 2 cards. Step count and schedule
  do not depend on the card count.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* Learning rates, from the upstream recipes: **bf16** 5e-4, cosine from 8x
  down to 1x over the first two thirds of the epochs, then cosine down to
  0.025x; **qat** (int4 weights, int8 activations) 2.5e-6, cosine from 25x down
  to 1x over all epochs, after 257 batches of scale initialisation; **fp32** 1e-7,
  cosine down to 1e-15.
* `ZCTX_SEED=20260901` seeds torch on each rank. The data loader draws the
  article permutation of each epoch from a fixed generator (seed 0) when a run
  starts, so every run, a resumed one included, begins with the same sequence of
  permutations.

### Training on package D

A reproduction trains the transformer on package D, the training package of
the stream of this build (section 4). The recipe uses the settings of row 08
of the chain below, with one plan over 1,000 bf16 epochs:

1. stages (a) to (c), package D with its priors;
2. bf16 from a cold start on package D at the window 2,048, with the
   environment of row 08 of the chain (`tf/3_train/chain/08_w2048.env`)
   without its warm start (`TFT_WARMSTART`, `TFT_WARMSTART_OPT`) and with
   `TFT_EPOCHS_BF16=1000`, as many bf16 epochs as the shipped weights saw in
   all;
3. the lifting of subnormal rows of the bf16 end (below);
4. 32 epochs quantization-aware and 6 at full precision (`qat_arm.sh` with
   `tf/3_train/chain/10_fp32.env`);
5. the export to the container and the binding of the new container to the
   compressor (section 6);
6. optional: new BITLSTM weights from a trace of a compression run with the
   new container (section 7). The trace writer belongs to the development tree
   and is not part of this package. Without this step the compressor runs with
   the shipped BITLSTM weights, which were trained for the shipped transformer.

`tf/3_train/README.md` gives the commands. The host setup starts step 2 on two
cards, as it started the cold start of row 01. On cards of the RTX 5090 class
steps 2 and 4 take about 215 h, computed from the epoch times of section 1 with
twice the four-card time per bf16 epoch on two cards. The recipe as a whole was
not run. The shipped weights are the end of the chain below, whose sections
each had their own plan.

### The chain of the shipped weights

The bf16 phase ran in sections. The first was trained cold; every later one
started **warm** from the bf16 state of the previous one, with a fresh
learning-rate plan, on the training package and at the window of its row.
Before the quantization-aware phase, subnormal weight rows were lifted (below).
The table is `tf/3_train/chain/CHAIN.tsv`; the environment of each section,
including the resumed parts, is in `tf/3_train/chain/`.

| # | section | starts from | package | window | epochs | end (sha256) | seen |
|---|---|---|---|---:|---|---|---:|
| 1 to 7 | earlier stream forms | cold, then each from the one before | A to C | 1024 to 2560 | 800 bf16 | `02ef775c…` | 800 |
| 8 | window 2048 | 7 | **D** | 2048 | 200 bf16 | see 8a | 1,000 |
| 8a | lifting of subnormal rows (below) | 8 | | | | `377e6a7f…` | |
| 9 | quantization-aware | 8a | **D** | 2048 | 32 qat | `2a698b86…` | |
| 10 | cooldown | 9 | **D** | 2048 | 6 fp32 | `15b1d56b…` | |

Packages: **D** is the package of section 4 (the stream of this entry). **A**,
**B** and **C** were built the same way from dumps of earlier versions of the
compressor, on earlier forms of the stream, the article order and the PPMd
priors; this package does not rebuild them, and `MANIFEST.tsv` lists their
checksums. The last 200 bf16 epochs and all qat and fp32 epochs ran on
package D.

**Lifting of subnormal rows (8a).** Weight rows whose largest magnitude is
below the smallest normal float32 make the row scales of the quantization-aware
phase overflow (NaN). `tf/3_train/fix_subnormal_rows.py` scales every row of a
2-D weight matrix whose largest magnitude is below 1e-30 up to a largest
magnitude of 1e-30 and changes nothing else; on the bf16 end of section 8 it
lifted 77 rows in 35 matrices.

**Warm start versus resume.** A warm start (`TFT_WARMSTART`,
`TFT_WARMSTART_OPT`) loads weights and optimizer state into a fresh run with a
fresh plan; `paket11_warmstart.py` does it form-aware (the predecessor table
is relocated block-wise when its row count changes). A resume continues a plan,
with `TFT_SCHED_TOTAL_EPOCHS` and `TFT_SCHED_DONE_EPOCHS` making the runner place
the remaining epochs on the original cosine. Never warm-start by placing a file in
`models/`.

### Running it

`tf/3_train/README.md` gives the commands. They cover the composed tree,
`kette_arm.sh` for a section, `wiederanlauf.sh` for a resume, `qat_arm.sh` for the
quantization-aware and fp32 phases on an existing bf16 state, the fp32 phase
with a chosen epoch count, and the host setup that fetches the package and the
priors and checks them by content (the sha256 of the whole priors file and the
row sums of random samples; a length check passes on a file with holes).

---

## 6. Stage (e): export and container

```
bash run/export_chain.sh                       # in the training tree, on the GPU host:
                                               #   models/poc8m_w2048-pkd-q4-fp32.tch
                                               #   -> weights-poc8m_w2048.bin (+ .bin.json, .vocab-bitmap.u8)
TREE=<work> bash tf/4_export/export.sh weights-poc8m_w2048.bin <out>   # CPU
```

`export.sh` runs four steps, each checked by the tool that performs it:

| step | tool | result for the shipped weights |
|---|---|---|
| input | reference weight file `FX2TFW01` from the checkpoint | 41,624,765 B, sha256 `614e2ea5…` |
| bf16 rounding of the raw fp32 tensors | `wpack2_bf16.py` | 117 tensors, 37,708 values |
| container format v3 | `weights_compress_v3.py compress3` | 3,737,929 B, sha256 `8cff9bb1…` |
| remove the structurally dead MLP neurons | `prune_dead_neurons.py` | 70 of 12,288; 3,726,968 B, sha256 `aa1b266f…` |
| zero the rows of collapsed attention channels, format v4 | `convert_container_v4.py` | 86 rows of 29 channels; 3,716,667 B, sha256 `ba982cd5da95c88c076d63b7767dad12e0c510ea3dd99adc0e213e707b45e719` |

These steps were rerun for this package on a different machine from the
shipped `.bin`; all three containers came out bit for bit identical.

### Container formats

| format | magic | what it is |
|---|---|---|
| v1 | `FX2TFWC1` | uniform 1/15 range coding of the int4 weights, everything else raw |
| v2 | `FX2TFWC2` | one range-coded stream with adaptive binary models (LZMA-style 11-bit probabilities, shift-5 update); `rope.sin` / `rope.cos` are not stored but recomputed from `rope.inv_freq` by a bit-exact host port of the CUDA `__nv_sinf`/`__nv_cosf` |
| v3 | `FX2TFWC3` | v2 with counting instead of decaying probabilities on the three stationary trees (int4 weights, names, metadata) |
| **v4** | **`FX2TFWC4`** | v3 plus counting trees on the two low byte planes of the f32 tensors (zero after the bf16 rounding), one flag bit per int4 row that lets an all-zero row go uncoded, and names and metadata predicted from a template |

The container shipped with this entry is **format v4**. v1 and v2 are Ivanov's;
v3 and v4 are this entry's (`tf/4_export/weights_compress_v3.py`, documented in
its header), and the loader decides on the magic.

### bf16 rounding of the raw tensors

The raw f32 tensors (convolutions, beta projections, dt biases, norms, decay
rates, scale coefficients) are rounded to bfloat16 precision (round to nearest,
ties to even, the rule of `torch.bfloat16`). This is a *model* change (upstream
leaves exactly these tensors unquantized); `rope.inv_freq` and the i32
configuration are not touched.

### Dead neurons and collapsed channels

A neuron of an MLP block is **structurally dead** when its quantized output is 0
for every possible input. `prune_dead_neurons.py` decides that from the
container alone, with the closed form of the inference arithmetic (header of
the tool), and removes row j of `mlp.up` and column j of `mlp.down`; the
compressor takes each layer's MLP width from the tensor shape. In the shipped
transformer 70 of 12,288 neurons are dead.

In 29 attention channels the row scale of the query, key and forget-gate
projections (once also of the value and output-gate projections) collapsed
during training to between 2^-20 and 2^-102. `convert_container_v4.py` sets
those 86 int4 rows and their scales to zero (rule: an int4 tensor outside the
MLP with `log2|scale| < -20`). Unlike the dead neurons there is no closed form,
so the decision rests on measurement. With the rows zeroed, the compressed
output is byte for byte the same on two test beds with the transformer active.

Both tools first prove that their writer reproduces the input container bit
for bit, list every change, and read the result back.

### Export manifest

The exporter writes a manifest beside the weights (`.bin.json`) with geometry,
row layout of the embedding (`embedding_rows` 216), the arm list of the training
package, the `prior0` sha256, the window and the vocabulary bitmap. The manifest of
the shipped weights is `tf/4_export/weights-poc8m_w2048.bin.json`.

### Binding the weights to the compressor

In the submitted package the container is appended to both executables, and
each program writes it out of its own image at start-up. The package source
carries the **relative** path of the container and an FNV-1a-64 fingerprint over
the whole container file (`src/transformer.h`, `WeightsPath()` and `WeightsFingerprint()`), and the `prior0` table
with its sha256 (`src/tf1/tf1_prior0_set_v9.h`). The binary contains no
`getenv`; it opens exactly this one container and refuses any other. The stamping tools
belong to the development tree. To build with a retrained container, copy it to
the shipped file name and write the fingerprint into `WeightsFingerprint()`;
the program prints the fingerprint it reads when it refuses a container.

---

## 7. The BITLSTM

The BITLSTM adds a learned correction to the final probability in logit space
(top-level `README.md` section 4.4). The design is David
Freelan's (`bitlstm32` in `cmix-obias`, `ATTRIBUTION.md` section 3). Its
shipped weights were trained offline, on a trace of a compression run of this
configuration; `bitl/README.md` describes the chain and ships the tools:

1. a **trace run**: a build variant of the compressor records, for every bit
   the BITLSTM sees, its inputs (the final probability, 13 columns of the
   stage-one mixer row, the raw output of the final mixer, the coded bit) and,
   once per eight such bits, the projections of the transformer and byte-LSTM
   states;
2. the **cut**: the training range is the first 418,837,176 bits of the trace,
   which end at about 20 % of the stream (`bitl/tools/cut_trace.sh`);
3. the **replay training**: the BITLSTM's own C++ code (the forward pass and
   the learning step of the compressor) replays the trace in stream order, 32
   passes, learning rate from 0.003 to 0.00003 along a cosine, 16 cells;
4. **bfloat16** conversion of the last pass: 7,386 B, sha256
   `c72957f48c2ec81afa3aed2baae7acd964c17f96bb0406b8195818fc708a8e7d`, which the
   build compiles into the program.

At run time the shipped weights stay frozen up to bit 418,837,176 of the bits
the BITLSTM sees, the end of their training range (about 20 % of the stream);
from the next window of 1,024 such bits on, the BITLSTM learns online, starting
from these weights with fresh optimizer moments.

**Why only the first 20 %.** Weights trained on the whole stream would have
needed a trace of a complete run and, for the larger amount of data, a longer
training, about three days together. The training range was therefore cut at
20 %, with 32 passes over it; behind it the BITLSTM learns online.

The rest of the predictor never sees the corrected probability. One trace
therefore holds the inputs of the BITLSTM for any of its weights, and the
replay reproduces what the coder would compute.

The inputs of the BITLSTM (the mixer outputs and the sketch of the transformer
state) change with a retrained transformer; BITLSTM weights for it need a new
trace. The compressor checks the BITLSTM weights only for their geometry
(`Head::LoadBlob` in `src/bit_head.cpp`), so with a retrained transformer of
the same geometry it also runs with the shipped weights. New weights replace
the byte array of `src/bit_head_weights.h`; the point where the BITLSTM starts
to learn online is the constant 418837176 in `src/bit_head.cpp` and follows the
cut of the trace.

## 8. Dictionary, hints, article order, PGO profile

See `data/README.md`.

---

## 9. What is and is not bit-reproducible

| artefact | reproducible? |
|---|---|
| coder stream from `enwik9` | **yes, deterministically**, a function of `enwik9` and the build |
| dump, zone mask, priors, training package | **yes, deterministically**, with the tools of `tf/1_dump/` and `tf/2_package/` |
| **the trained transformer weights** | **no**; a compression run with a retrained transformer shows whether it is as good as the shipped one (the shipped BITLSTM weights were trained for the shipped transformer, section 7). The bf16 end of section 8 before the lifting of subnormal rows is recorded by the first 16 hex digits of its sha256. |
| container from given weights | **yes, deterministically**, rerun for this package (section 6) |
| trace of the BITLSTM | deterministic given the development tree and the build; a rerun of the trace was not compared byte for byte |
| BITLSTM weights from a given trace | **yes, deterministically** on the CPU (fixed seed, fixed order, one thread). The trainer of this package gives the same weights as the original trainer on the first 2,000,000 bits of the trace (32 passes); the full 418,837,176 bits were not rerun |
| the archive from given weights | **yes, deterministically**; this is what the entry's own verification proves |

The weight container and the BITLSTM weights are therefore **inputs** to the
build, like the dictionary, and are reproducible in *method* but not in *bits*.
Everything downstream of them is bit-exact.

## 10. CHANGES: what is new in cmix-neif, relative to the upstream training code

Authorship and licences: `ATTRIBUTION.md`.

| new in cmix-neif | what it does |
|---|---|
| `tft/` | geometry registry and parameter/size accounting (`param_count.py`), dataset checks, runner with the continued plan of a resumed run, PoC train/quantize/cooldown entry points, `export_weights_poc.py` (adds the manifest) |
| `zctx/` | everything that makes the training rows match the coder, namely the dump reader with the target rule, the dump loader with the position check, the int64 packing, the embedding with sentinel rows and the predecessor and match channels (`zpmodel.py`); plus the earlier Python reconstructions of gate zones, prompts and channels |
| `paket11_warmstart.py` | form-aware warm start |
| `tf/1_dump/`, `tf/2_package/` | the patch with the coder-side modes and its build script, package builder (`mk_pkgdump.py`), the steps of the package build, acceptance tests |
| `tf/3_train/` | the environment of every section of the chain, phase scripts, host setup, the lifting of subnormal rows |
| `tf/4_export/` | bf16 rounding, container formats v3 and v4, removal of dead neurons, zeroing of collapsed rows |
| `bitl/` | trace cut, replay trainer, bfloat16 conversion of the BITLSTM weights |
| vocabulary 208 instead of 205 | the preprocessing of cmix-neif produces a different byte set |
| training source | the coder's own fed-row dump instead of a reconstruction of the coder's rules |

Unchanged from upstream: the file of the model (`pysrc/model.py`), the KDA and
attention kernels, the optimizer, the quantization scheme, container formats v1
and v2, and the C++ inference reference (`cpp_infer/`). The overlay extends the
model only by subclassing its embedding (`zctx/zpmodel.py`: sentinel rows,
predecessor and match channels).

## 10a. Names of the development tree used by these tools

The trace run uses a build variant of the development tree of the compressor,
and so did the original coder-side runs (`tf/1_dump/` reproduces them from the
source of this entry). Its switches are resolved to fixed values in the package
source; the tools here still name some of them, because they are the run lines
that were used:

| name | where | what it is |
|---|---|---|
| `tf/1_dump/features_*.txt` | stage (b) | record: the switch lists of the three original builds of the development tree, verbatim |
| `NEIF-PAKET11` (marker) | `tf/2_package/patch_paket11.py` | tags each edit of the patcher so that a second run can skip it |
| `NEIF_TF1_PRIOR0_*` | `tf/2_package/prior0_diff.py` | the tool writes the header of a new `prior0` table into the development tree, in that tree's names; the package source carries the result as `src/tf1/tf1_prior0_set_v9.h` |
| `NEIF_LIST` | `data/tools/build_variant.sh` | makefile variable that collects the switches of a build; the package `makefile` carries the resolved flags |
| `bitl/trace/FEATURES_trace.txt`, `CMIX_BITL_TRACE*` | stage (f) | switch list of the trace build and the environment that names the trace files and selects the 13 recorded columns |

Some shell scripts read a machine, a checkout or a data or work directory from
environment variables and stop if the variable is unset. The scripts of
`tf/3_train/` assume the directory layout of a rented GPU host under `/root/`.

## 11. Index

* `MANIFEST.tsv`: every file here with origin, size and sha256; large artefacts
  (stream, dump, package, priors, checkpoints of the chain, containers, the
  trace) as references with checksums, not embedded.
* `ATTRIBUTION.md`: authorship, licences, upstream URLs and commits.
* `tf/tree_upstream/`: the upstream training tree, verbatim.
  `tf/tree_overlay/`: cmix-neif files only. Section 1 explains how they combine.
* `tf/1_dump`, `tf/2_package`, `tf/3_train`, `tf/4_export`: the stages.
* `bitl/`: the BITLSTM.
* `data/`: dictionary, hints, article order, PGO profile.
