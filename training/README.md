# `training/`: how the pre-trained parts of this entry were produced

This directory is **documentation and tooling, not part of the entry**.
`build.sh` never enters it; nothing here is compiled into `archive9` or into the
decompressor. It serves reproducibility and attribution.

1. **Reproducibility.** Two parts of the compressor are trained offline and
   shipped as frozen weights, the byte-level **transformer** (a weight container
   appended to both executables) and **BITLSTM**, the small LSTM that corrects
   the final probability behind the mixer (three weight sets in a 32-cell
   layout, compiled into the program). This directory describes how both were produced
   from `enwik9`, ships the tools and the executed configurations, and gives
   the checksums at which a rerun can be compared. Section 9 says what can be
   rerun bit-exactly and what cannot.
2. **Attribution.** The architecture of the transformer and its training recipe
   are Vladimir Malinovskii's `cmix-neif-beehive`, whose weights were the start
   of the training; its trainer descends from Vladimir Ivanov's
   `fx2-cmix-transformer`; the design of BITLSTM comes from David Freelan's
   `cmix-obias`; the compressor descends from cmix, fx-cmix, cmix-lex and
   cmix-neif-1.0. `ATTRIBUTION.md` names the borrowed parts, their authors and
   their licences.

The pipeline is that of `cmix-neif-1.0` (its `training/` directory,
`https://github.com/marwoelk/cmix-neif-1.0`), with the stream form, the model,
the training recipe, the container formats and the training of BITLSTM of this
version. Section 10 lists the differences.

Everything in this directory is GPL-3, like the rest of the entry, except the
third-party files listed in `../third_party/README.md` (copies in
`tf/tree_upstream/src/ds/`, altered `avx_mathfun.h` routines in
`tf/tree_upstream/cpp_infer/`) and the Wikipedia excerpts in
`tf/tree_upstream/prof_input/`.

---

## 0. The short version

```
enwik9
  |  (a) the compressor's own preprocessing, a "-e" run of a stream build
  v
coder stream  501,062,281 B, sha256 c35d3073..., 208 byte values         tf/1_dump/
  |
  +--(b) dump run  (cmix -n, dump build)       --> fed-row dump, zone mask, match channel
  +--(b) prior run (cmix -n, PPMd-only build)  --> PPMd prior per fed byte
  |
  v (c) package build                                                    tf/2_package/
training package: 488,038,367 tokens, 203,023,960,672 B of priors
  |
  v (d) GPU training, start = the weights of cmix-neif-beehive          tf/3_train/
  |     100 + 100 + 400 epochs bf16, allocation of 2, 3 or 4 bits per weight row,
  |     96 epochs quantization-aware with a term for the container size, 16 epochs fp32
  v
checkpoint (.tch) --> reference weight file (.bin) + manifest (.bin.json)
  |
  v (e) export at window 2048: bf16 rounding, v3, dead neurons, collapsed   tf/3_train/export/
  |     rows zeroed, v4, v5, NP2
weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5  3,744,629 B, sha256 729694cb...
                                                 (appended to both executables)

enwik9 --(f) trace run ("-e", trace build) --> inputs of BITLSTM for every bit it sees
          --> three zones (metadata blocks, text 1, text 2)
          --> GPU training per zone, fine-tuning on rounded weights          bitl/
          --> bfloat16, 3 x 22,426 B --> src/bit_head_weights.h (compiled into the program)
```

The coder-side tools of (a), (b) and (f) are plain builds of the source of
this entry with the patch `tf/1_dump/trainmodes.patch` (section 3).

**The coder is the specification.** Every prompt, gate zone, fed row and side
channel that the coder shows the transformer must be present in training, byte
for byte. The training rows are therefore dumped by the coder itself and read
back by the trainer, as in `cmix-neif-1.0` (`tf/1_dump/README.md` section 4,
`tf/2_package/dumpv2.py`).

---

## 1. Prerequisites

### Hardware and time of the runs that produced the shipped parts

| step | hardware | time |
|---|---|---|
| (a) to (c), dump recipe | 1 CPU core per coder run, 50 GiB RAM (64 recommended), 245 GiB disk | EPYC 7502P: 6 h 18 min in all (dump run 2 h 20 min and prior run 4 h 45 min in parallel; 45.0 GiB peak in the package checks) |
| (d) start state | CPU | a few minutes |
| (d) b1, 100 epochs bf16 | 4 x RTX 5090 (32 GB) | 7.4 h |
| (d) b2, 100 epochs bf16 | 8 x RTX 5090 | 6.1 h |
| (d) b3, 400 epochs bf16 | 8 x RTX 5090 | about 18.7 h (17.3 h for epochs 30 to 400) |
| (d) allocation, init | 1 and 8 cards | a few minutes |
| (d) 96 epochs quantization-aware | 8 x RTX 5090 | 12.2 h |
| (d) 16 epochs fp32 and strict evaluation | 8 x RTX 5090 | 4.6 h |
| (e) export, v5, NP2 | CPU | a few minutes |
| (f) trace run | 1 machine, about 300 GB disk for the trace | about 18.5 h |
| (f) BITLSTM training | 1 GPU per zone | metadata zone RTX 3060, about 30 min; text 1 RTX 5060 Ti, about 4.4 h; text 2 RTX 5060 Ti, about 2.7 h; fine-tuning under 1 h per zone |

The priors file `ppmd-probs.float16` (416 B per token) is the only large
artefact of the transformer pipeline. The training hosts held the package and
the priors in memory (`/dev/shm`); the priors compress to about 118 GB with
`zstd -3` for the transfer.

### Software

* clang 18 and lld 18 (as in `install.sh` of the entry) and `patch` for the
  coder-side tools; AVX2 and FMA
* Python 3.11 with NumPy for the package tools and the container writers
* for training: CPython 3.11.15, torch 2.12.1 (CUDA 13.0 build), triton 3.7.1,
  `flash-linear-attention` (`fla`) 0.5.1 with the fused AttnRes kernel
  `fla.ops.attnres`, numpy 2.4.6, flash-attn 2.8.3; the upstream lock file
  pins the rest
* no network access is needed for any computation; every input is `enwik9`, a
  tool or artefact named here, or the weight container of `cmix-neif-beehive`
  (section 5.2)

### The training tree

As in `cmix-neif-1.0`, the trainer is the upstream tree with the files of this
project laid on top:

```
tf/tree_upstream/    the fx2-cmix-transformer tree (2026-07-24 release), verbatim,
                     189 files; LICENSE and every copyright header untouched
tf/tree_overlay/     the overlay of cmix-neif: tft/, zctx/, run/, paket11_warmstart.py
tf/3_train/tree/     the training scripts of cmix-neif-beehive with the changes of
                     this version; they run on top of the training tree (TFT_TREE)
```

```bash
cp -a tf/tree_upstream <work>
cp -a tf/tree_overlay/. <work>/
python3 tf/2_package/patch_paket11.py <work>     # idempotent, individually marked edits
```

The training hosts ran a tree of the same composition. It carried further
edits for other experiments of this project (a per-layer MLP width, a window
per attention layer, an all-rank load of the optimizer state in the warm start
of `tft/runner.py`, an optional threaded data loader in `pysrc/data.py`), and
it lacked a check of the exporter that the package directory is set
(`tft/export_weights_poc.py`). None of these edits is active in the
configuration of this chain, whose scripts load their state themselves
(`rx7q.py`). A comparison of the code of the overlay files with the trainer
tree from which the hosts were set up (syntax trees without comments and
docstrings) found no other difference. The form shipped here is the overlay
without these edits and the upstream tree as released.

---

## 2. Stage (a): the coder stream

The transformer is trained on the **coder stream**, `enwik9` after the
compressor's own preprocessing (README sections 3 and 4 of the entry). It is
hard-coupled to this exact preprocessing; a change of the preprocessing forces
a retrain.

| property | value |
|---|---|
| length | 501,062,281 B |
| sha256 | `c35d3073eeba70f2…` |
| distinct byte values | 208 (`tf/2_package/vocab-bitmap.u8`) |
| dictionary | 520,804 B, sha256 `0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535` (`dictionary/english.dic`) |
| article order | residual of 40,694 B, sha256 `0eec59a305774b5a46b17b83a31274aafd5081ba387a3536def3372a442ba59d` (`src/readalike_prepr/data/new_article_order`) |
| gate zone | the metadata block and the census metadata block, the first 13,023,920 positions of the coded bytes (with the 6-byte segment header), withheld from the transformer |
| training tokens | 488,038,367 (coder stream plus its 6-byte segment header, from the cut on) |

A compression run of the submitted build codes exactly this stream.

---

## 3. Stage (b): the coder-side runs

Builds of the compressor produce what the trainer reads
(`tf/1_dump/README.md`). They are the bed and stream run, the **dump run**, in
which every producer that feeds the transformer runs and its forward pass is
skipped, and the **prior run** with PPMd alone. The source of the entry has
the switches of these modes resolved to the values of the submitted build;
`tf/1_dump/trainmodes.patch` restores them as `TRAIN_MODE` 1 to 5 (stream,
dump, priors, bed and the trace of section 7), and
`tf/1_dump/build_dump_tools.sh <entry directory> <tools directory>` builds the
five tools. The shipped package was made with the same modes built from the
development tree of the compressor (build lines in `tf/1_dump/`); on the same
inputs the tools of the patch write the same files, byte for byte
(`tf/1_dump/README.md`, section 6). The dump of the shipped package has
543,832,786 rows (487,865,904 stream rows, 55,966,882 inserted rows).

---

## 4. Stage (c): the training package

`tf/2_package/build_package.sh` runs stages (a) to (c) in one call and stops on
any finding; it checks the dump and the prior raster against the coder stream,
and the finished package with the loader of the trainer (`loader_probe.py` over
every micro-batch) and `abnahme_paket.py`. With `ENTRY=<entry directory>` it
builds the tools from the source of the entry with the patch and takes the
values of the stream form from `tf/1_dump/stream_form.txt` and
`tf/1_dump/forced_title_prefixes.h`; with `TREE=<development tree>`, the form
that built the shipped package, it builds them from the development tree and
also checks the build lines against the coder line of the entry.

| | value |
|---|---:|
| `zeilen.dump` | 6,525,993,408 B, sha256 `f76f5e5df2a8c991256c75e0dbf63df7d4dc6a1cdcb71a9996179e088f583d9f` |
| `ppmd-bytes.uint8` | 488,038,367 B, sha256 `721ad4f3f8a0a7957598cbc62b8ef495e9c620652486123757975b73274568bd` |
| `z3-prior0.float16` (prior of inserted rows) | 416 B, sha256 `ceb297e328d9af5daa72f47b750267b1c1978e8d382d1d254692fe2dc31cbafc` |
| `ppmd-probs.float16` | 203,023,960,672 B, sha256 `dc71f3113288cb9aff95adb0cf4f10af353b11888618df7ff187b360f57cbb0a` |
| `pkg.tar` (package without the priors) | 7,026,718,720 B, sha256 `cdf0ea186edfa59d87f2fadbd2c61b038e9f88297bae5e686875c067d84bcb52` |
| scored positions | 487,298,491 |
| micro-batches of the loader probe | 2,126 |

The prior of inserted rows must be the same on the coder side. The source of
the entry carries it as `src/tf1/tf1_prior0_set_v14.h`;
`tf/2_package/prior0_check.py` compares the table of that header with the
package file (it is the same table).

The training form follows `cmix-neif-1.0` (section 4 of its `training/README.md`)
with the changes of the stream form of this version (top-level `README.md`,
section 4). It feeds no rows after digit runs and adds the rows of the text
start and of the census pages, the computed values and the link context of
this version.

---

## 5. Stage (d): GPU training

### 5.1 Model

The model has the architecture of `cmix-neif-beehive`. It keeps the 16 layers
at d_model 192 of `cmix-neif-1.0` (three KDA layers followed by one attention
layer, four times) on the 208-value vocabulary and changes the following.

| change | configuration key | code |
|---|---|---|
| one key and value head in the attention layers | `n_key_value_heads: 1` | upstream option |
| value embeddings on layers 3, 7, 11 and 15 | `value_embedding_pattern`, `ve_sentinel: "clamp"` | upstream option, `tree/rx3.py` |
| MLP width 1024 | `d_mlp: 1024` | upstream option |
| a head that mixes the logits of the PPMd prior into the output | `prior_logit_mixing: true` | upstream option |
| attention residuals, replace mode, block 2 | `arch.attnres`, `skip_connections: false`, `token_embedding_connections: false` | `tree/arch.py`, `tree/attnres_v10.py`, `tree/attnres_hub.py` |

9,285,596 parameters in training (9,325,119 with the quantizers), attention
window 2,048 in every attention layer (`window_size: 2048`; `cmix-neif-beehive`
exported at 1,536).

### 5.2 Start state

The training started from the weight container of `cmix-neif-beehive`
(`weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4`, 3,649,198 B, sha256
`141ee13c316811157bea167846b3f45472affff7adaa1b8b263e30621e490885`, in the
repository `github.com/galqiwi/cmix-neif-beehive`, commit `92c07c0b`). Those
weights were trained by Vladimir Malinovskii from a random initialisation on
the training package of `cmix-neif-1.0`, which was built from `enwik9` alone.
`tf/3_train/container_to_tch.py` turns the container back into a training
state:

* every quantized weight is dequantized row by row; the conversion checks
  that the quantizer of the model gives back the integers of the container
  (155 of 155 tensors);
* the 102 MLP neurons that the exporter of `cmix-neif-beehive` had removed get
  a fresh row in `mlp.up` and a zero column in `mlp.down` (the function at the
  start is unchanged, the neurons can learn again);
* the rows 208 to 215 of the embedding and of the predecessor table are set to
  zero, because the stream form of this version feeds no rows after digit runs;
* the optimizer state of `cmix-neif-beehive` is not published; b1 starts with
  fresh moments and a short ramp of the learning rate.

### 5.3 The chain of the shipped weights

`tf/3_train/chain/` holds the configuration of every stage as it was executed
(host paths replaced by `<W>`), `chain/results/` the result files of the runs.
In the bf16 and quantization-aware stages the learning rate is 5e-4 times the multiplier in the table.

| stage | configuration | starts from | epochs | learning rate | cards | end (sha256 of `model.tch`) |
|---|---|---|---:|---|---:|---|
| b1 | `01_b1_bf16.json` | start state | 100 bf16 | 0.1 to 1 over the first 2 %, then cosine to 0.025 | 4 | `42502d8a…` |
| b2 | `02_b2_bf16.json` | b1 with its optimizer state | 100 bf16 | cosine 1 to 0.025 | 8 | `ec4ca263…` |
| b3 | `03_b3_bf16_epochs_1_to_29.json`, `03_b3_bf16_epochs_30_to_400.json` | b2 with its optimizer state | 400 bf16 | epochs 1 to 29 on a cosine 1 to 0.025 over 400 epochs (0.987 at the end of epoch 29), then constant 1 | 8 | `21a2de8c…` |
| allocation | `04_fisher.json` | b3 | | | 1 | `alloc_1.000.npz` `eb17b7b6…` |
| init | `05_qat_init.json` | b3, all-zero rows lifted (`mlp_lift.py`), moments of b3 | | | 8 | `start.tch` `ee9408d9…` |
| qat | `06_qat.json` | init | 96 | cosine 1 to 0.025 over all updates, size term from the first update | 8 | `995b7aa5…` |
| fp32 | `07_fp32.json` | qat | 16 | 1e-7 as such (`lr`), cosine to the final multiplier 1e-8 (`final_mult`), that is to 1e-15 | 8 | `0f08724b…` |

* **Allocation** (`tree/fisher_alloc_a.py`): squared gradients per weight row
  over 48 batches, levels 2, 3 and 4 bits, a budget of 1.00 of the uniform
  4-bit size. Of the 39,355 rows, 0.25 % got 2 bits, none 3 bits and the rest
  4 bits (`results/fisher_alloc_summary.json`).
* **Quantization-aware training as the cool-down** (`tree/rx7r.py`): int4
  weight rows of the allocation, int8 activations, 256 batches of scale
  initialisation, the moments of b3 carried by parameter name, and the size
  term of `tree/c1_rate_v2.py` (a differentiable estimate of the coded size of
  every quantized row, lambda 2.7e-9 nats per weight bit, tau 0.25, with the
  rules of the exporter) on every update. This is the recipe of
  `cmix-neif-beehive` with 96 instead of 64 epochs.
* **fp32** (`tree/rx3a.py`): 16 epochs instead of the 1 of
  `cmix-neif-beehive`, then the strict evaluation of the quantized model as the
  engine runs it.

The evaluations of the trainer on the training package (cross-entropy in bits
per token over the 487,298,491 scored positions; in-sample, a record of the
runs and no measure of the compression) were 1.3475 after b1, 1.3450 after b2,
1.3774 after b3 (not cooled down), 1.4095 at the end of the quantization-aware
stage (quantized forward pass) and 1.4051 in the strict evaluation after fp32.

`tf/3_train/reproduce.sh` runs the chain stage by stage with these
configurations. The training scripts of this version differ from those of
`cmix-neif-beehive` in the four points below, each marked `# 2.0` in the file.

* every rank loads the optimizer state of a warm start, and at updates 20 and
  1,000 of every launch all ranks compare a checksum of their parameters; a
  difference stops the run;
* a failure of the data loader is raised in the training loop
  (`tree/safe_prefetch.py`); the upstream prefetch ends the epoch early without
  an error, and the loop also counts its steps against the plan;
* activation checkpointing of the sublayers in the quantization-aware and fp32
  stages, so they fit cards with 32 GB at micro-batches of 131,072 tokens;
* the process ends with `os._exit(0)` after the process group is destroyed.

The attention window of 2,048 in training and export is a key of the
configurations (`window_size`).

---

## 6. Stage (e): export and container

| step | tool | result |
|---|---|---|
| reference weight file `FX2TFW01` from the fp32 checkpoint, at window 2048 | `tf/3_train/export/export_v3_ar.py` | 43,087,894 B, sha256 `fa0ea606ff8fda91784b6b3ae1c875ed5e6ed5f4a39e87cbf91d45403361392c` |
| bf16 rounding of the raw fp32 tensors, container format v3 | `export4/wpack2_bf16.py`, `export4/weights_compress_v3.py` | 3,777,558 B, sha256 `2bdc20aa…` |
| removal of structurally dead MLP neurons | `export4/prune_dead_neurons.py` | none of 16,384 is dead; container unchanged |
| zeroing of collapsed attention rows, format v4 | `export4/convert_container_v4.py` | 3,769,904 B, sha256 `38b7c162…` |
| format v5 (entropy coding of the weights) | `export4/convert_container_v5.py` | 3,767,643 B, sha256 `11ef8fdf…` |
| format NP2 (order of interchangeable neuron groups coded by their permutation; dead key dimensions of the KDA layers zeroed) | `export/np2/np2_container.py np <v5> <out> k fw np2 dead` | 3,744,629 B, sha256 `729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563` |

`export4/export.sh` runs the second to fourth step; when no neuron is dead it
passes the container on unchanged. For this package the steps from the
reference weight file of the chain to the container were rerun on another
machine with the tools of this directory; every intermediate and the
container came out bit for bit identical.

`np2_container.py` needs NumPy only and takes 3.5 min on one core. It checks
that the v5 writer reproduces its input, writes the container and reads it
back (every tensor equal to the permuted one). Its reader gives the same
tensors as the reader of the compressor's development tree (the decoded raw
files of the shipped container are identical), and one weight changed in the
input gives another container. The shipped container was written by the C++
form of the writer, `export/np2/tf1_np_codec.cpp`, which includes the
container reader of the development tree (it holds all container formats; the
reader in the source of the entry holds only the shipped format) and was built
inside that tree:

```
g++ -O2 -std=c++17 -DNEIF_TF1_WC4=1 -DNEIF_TF1_WC5=1 -DNEIF_TF1_WC5K=1 \
    -DNEIF_TF1_QMIX=1 -DNEIF_TF1_NP2=1 tools_src/tf1_np_codec.cpp src/tf1/weights_io.cpp
```

Formats NP and NP2 order the neurons of some groups so that the digits of the
permutation in the factorial number system spell bytes of the coded stream,
which are then not stored (bits-back coding). The groups with the fast-weight
layers and the KDA dimensions change the order of floating-point sums in the
engine, so the transformer output differs from the v5 container at the
rounding level.

**Binding to the compressor.** The source of the entry names the container by
its relative path and an FNV-1a-64 fingerprint over the whole file
(`src/transformer.h`, `WeightsPath()` and `WeightsFingerprint()`) and carries
the prior table of section 4 (`src/tf1/tf1_prior0_set_v14.h`). To build with a
retrained container, copy it to the shipped file name and write its
fingerprint into `WeightsFingerprint()`.

---

## 7. BITLSTM

BITLSTM adds a learned correction to the final probability in logit space
(top-level `README.md`). Its design is David Freelan's (`ATTRIBUTION.md`
section 3). This version ships three frozen weight sets in a 32-cell layout
(the metadata set uses 16 cells, the two text sets 32) and no learning path in
the program; `bitl/README.md` describes the chain:

1. a **trace run**, `cmix_trace` of `tf/1_dump/build_dump_tools.sh` (the
   source of the entry with `tf/1_dump/trainmodes.patch`), records for every
   bit BITLSTM sees its inputs, 1,694,239,343 records in all; the trace of the
   shipped weights was written by a build of the development tree
   (`bitl/README.md`, section 1);
2. the trace is cut into three **zones** on the grid of 1,024-bit windows (the
   metadata block and the census metadata block, the text up to the calendar
   block at page 143,227 of the article order, and the rest of the text);
3. a **GPU trainer** in PyTorch (`bitl/tools/cm_train.py`, `tb_train.py`)
   trains one weight set per zone in-sample, 256 windows per Adam step, with a
   learning rate on a cosine from 1e-2 to 1e-4 for the metadata set and from
   3e-3 to 3e-5 for the two text sets;
4. the weights are rounded to bfloat16 with fewer mantissa bits and fine-tuned
   on the rounded values (`cm_qft.py`, `tb_qft.py`, `bz_nblb.py`);
5. `bitl/tools/weights_header.py` writes `src/bit_head_weights.h` of the entry
   from the three files of `bitl/record/`, byte for byte.

A retrained transformer changes the inputs of BITLSTM; new BITLSTM weights for
it need a new trace.

---

## 8. Dictionary, hint package, article order, PGO profile

The dictionary (`dictionary/english.dic`) is that of `cmix-neif-1.0`, byte for
byte, and so is the article order. This version codes the hint package
(`dictionary/english.hints`, 47,166 B) and the order residual (40,694 B) in a
new form with a context-mixing coder of its own; the compressor derives the
same dictionary and the same order from them. Their writers belong to the
development tree. The derivation of the word set, the hints and the order is
in section 8 and `data/README.md` of the `training/` directory of
`cmix-neif-1.0`. The PGO profile (`pgo_data/default.profdata`) was recorded
with an instrumented build of the configuration of this entry, from a
compression of `enwik9` and a decompression of the start of its archive, on
an earlier state of the source; the 18 functions changed since then are
compiled without profile data.

---

## 9. What is and is not bit-reproducible

| artefact | reproducible? |
|---|---|
| coder stream, dump, priors, training package | **yes, deterministically**, a function of `enwik9` and of the tools of `tf/1_dump/build_dump_tools.sh` (the source of the entry with `trainmodes.patch`); they write the same files as the tools of the development tree that made the shipped package (`tf/1_dump/README.md`, section 6) |
| prior table of the source | **yes**; `tf/2_package/prior0_check.py` compares it with the package |
| start state from the container of `cmix-neif-beehive` | the weights, yes (integer check); the bytes of `torch.save` depend on the torch version |
| **the trained transformer weights** | **no**; the GPU stages reproduce the recipe, and their bits depend on the order of reductions, the card count and the library versions. The end of every stage is recorded by its sha256 (section 5.3) |
| container from the reference weight file | **yes, deterministically**; rerun for this package up to format NP2, bit for bit, with the tools of this directory (section 6) |
| trace of BITLSTM | deterministic given `cmix_trace` (the source of the entry with `trainmodes.patch`) and the container; the trace itself is not part of this package |
| BITLSTM weights from the trace | **no**, GPU training; the result files are in `bitl/record/` |
| `src/bit_head_weights.h` from the three weight files | **yes**, byte for byte (`weights_header.py`) |
| the archive from given weights | **yes, deterministically**; this is what the entry's own verification proves |

The weight container and the BITLSTM weights are therefore **inputs** to the
build, like the dictionary, and are reproducible in method but not in bits.
Everything downstream of them is bit-exact.

Not in this package are the development tree of the compressor (with the
writers of the hint package, of the order residual and of the PGO profile),
the training package and the priors (about 210 GB), the checkpoints of the
chain and the trace (about 300 GB).

---

## 10. Changes relative to the training of cmix-neif-1.0

| | cmix-neif-1.0 | this version |
|---|---|---|
| stream form | coder stream 552,750,198 B | 501,062,281 B (top-level `README.md`, sections 3 and 4) |
| dump tools | `trainmodes.patch` on the entry's source (stream, dump, priors) | `trainmodes.patch` on the entry's source, generated from the development tree (stream, dump, priors, bed and the trace of BITLSTM) |
| prior table | set 9 | set 14 (`src/tf1/tf1_prior0_set_v14.h`) |
| model | 16 layers, d_mlp 768, three key and value heads, skips and token re-injection | the architecture of `cmix-neif-beehive` (section 5.1) |
| start | cold start, then 1,000 bf16 epochs over four stream forms | the weights of `cmix-neif-beehive`, 600 bf16 epochs on this stream form |
| trainer | `tft/` runner | the scripts of `cmix-neif-beehive` (`tf/3_train/tree/`) with four changes (section 5.3) |
| quantization | uniform int4, 32 qat epochs at lr 2.5e-6 | 2, 3 or 4 bits per row, 96 qat epochs as the cool-down with a size term |
| fp32 | 6 epochs | 16 epochs |
| container | format v4 | format NP2 (formats v5, NP and NP2 are new; writer in Python and in C++) |
| BITLSTM | one weight set of 16 cells, C++ replay trainer on the first 20 % of the stream, online learning behind it | three weight sets in a 32-cell layout (the metadata set uses 16 cells, the two text sets 32), GPU trainer, frozen, no learning path |

---

## 10a. Names of the development tree used by these tools

The source of the entry has the build switches of the development tree
resolved and carries no names with the prefix `NEIF_`. In this directory these
names remain where a tool works with the run
modes of the development tree. They name the switches, the environment
variables and the build lines of those modes as the development tree names
them, so that the patch, the recipe and the records match the build that made
the shipped parts.

| name | where | what it is |
|---|---|---|
| `NEIF_*` macros and identifiers | `tf/1_dump/trainmodes.patch` | the code of the run modes, generated from the development tree |
| `NEIF_TF1_*`, `NEIF_PPMD_ONLY_*` | `tf/1_dump/README.md`, `tf/2_package/*` | environment variables that name the output files of the dump and prior runs, and macros the recipe reads |
| `tf/1_dump/features_*.txt`, `bitl/trace/features_trace.txt` | stages (b) and (f) | the build lines of the development tree that made the shipped package and trace, as switch lists |
| `NEIF-PAKET11` (marker) | `tf/2_package/patch_paket11.py` | tags each edit of the patcher so that a second run can skip it |
| `NEIF_TF1_*` build flags | `tf/3_train/export/np2/tf1_np_codec.cpp` | the format paths of the development tree's container reader the C++ form of the NP2 writer is built with |
| `src/cmix-neifv9`, `scripts/zeilen_vgl.py` | the `--tree` form of `tf/1_dump/build_dump_tools.sh`, the `TREE=` form of `tf/2_package/build_package.sh`, comments of `tf/2_package/manifest_arms.sh`, `tf/2_package/page_anchors.py` and `tf/3_train/export/np2/tf1_np_codec.cpp` | the directory of the compressor source in the development tree, and its line check (`zeilen_vgl`, line comparison) |

Some tools and file names keep German words, for example `zeilen.dump` (rows),
`zonenmaske` (zone mask), `spur` (trace), `kanal` (channel), `vorbit`
(previous bit), `wiederbelebe` (revive), `weiten` (widen), `ende` (end),
`quelle` (source), `seiten` (pages), `leser` (reader), `abnahme`
(acceptance) and `pruef` (check).

## 11. Index

* `MANIFEST.tsv`: every file here with origin, size and sha256; large artefacts
  as references with checksums, not embedded.
* `ATTRIBUTION.md`: authorship, licences, upstream URLs and commits.
* `tf/tree_upstream/`: the upstream training tree, verbatim. `tf/tree_overlay/`:
  the overlay of cmix-neif. Section 1 explains how they combine.
* `tf/1_dump`, `tf/2_package`: stages (a) to (c).
* `tf/3_train`: stage (d) and, in `export/`, stage (e).
* `bitl/`: BITLSTM.
