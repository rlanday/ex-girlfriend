# The BITLSTM: trace, replay training, weights

The BITLSTM is the small LSTM that corrects the final probability behind the
mixer (section 4.4 of the top-level `README.md`); name and design are from
David Freelan's `bitlstm32` in cmix-obias.  This entry ships trained weights for
it, 7,386 bytes in bfloat16, compiled into the program as the byte array of
`src/bit_head_weights.h`.  The compressor runs the BITLSTM with these weights
frozen up to its model bit 418,837,176, the end of the training range (about
20 % of the coder stream), and lets it learn online from the next 1,024-bit
window on.  This directory describes how the weights were produced from
`enwik9` and holds the trainer and the scripts.

## The chain

```
enwik9
  |  (1) trace run: one compression run of a measurement build
  v      trace/run_trace.sh, trace/FEATURES_trace.txt
trace + channel     the BITLSTM's inputs, per model bit (FORMAT.md)
  |  (2) cut: the first 418,837,176 model bits, at 20 % of the coder stream
  v      tools/cut_trace.sh  ->  record/ (bit count, prefix sizes and sha256)
  |  (3) training: 32 replay passes, learning rate 0.003 -> 0.00003 (cosine)
  v      tools/train_blob.sh, trainer/  ->  weights of pass 32, f32
  |  (4) rounding to bfloat16: tools/nblb.py v2  ->  7,386 B, sha256 c72957f4...
  v
  (5) the bytes become the array in src/bit_head_weights.h
```

A "model bit" is a coded bit on which the BITLSTM runs (FORMAT.md).

**(1) Trace run.**  A build variant of the compressor compresses `enwik9` and
writes, for every model bit, what the BITLSTM receives from the coder, namely the
coder's probability, the stage-1 mixer row, the raw output of the final mixer,
the override flag and the coded bit (the trace, 60 bytes per bit), and every
eight such bits the inputs that change once per byte, among them fixed 32-value
projections of the transformer state and of the byte-LSTM state (the channel,
276 bytes).  The predictor never sees the corrected probability, so these
inputs do not depend on the BITLSTM's weights, so one run serves every training
pass.  The build variant comes from the development tree of this entry, which
is not part of this package.  Its build line (`trace/FEATURES_trace.txt`) is the
model configuration of the entry, written in the options of that tree, plus
options of a measurement build, among them:

* `BITL_MESS=1`, `BITL_TRACE=1`: the measurement form of the BITLSTM with the
  trace writer;
* `BITL_META_OFF=1`: no BITLSTM inside the metadata block, as in the entry;
* `REUSE_EXTRACTED=1`: the run takes the dictionary (`.dict`, `.dict.comp`) and
  the article order (`.new_article_order`) from the run directory instead of
  deriving them; they come from a compression run of the same configuration;
* `TF1_SELBSTAUSKUNFT=1`: the transformer engine reports on stderr that it
  runs, which `run_trace.sh` checks.

The run sets five environment variables of that tree.  They are `CMIX_BITL_TRACE` and
`CMIX_BITL_TRACE_AUX` (paths of trace and channel), `CMIX_BITL_TRACE_COLS`
(mask of the stage-1 columns to record: 0..6 and 21..26, the 13 the BITLSTM
reads), `CMIX_BITL_TRACE_AUX_PROJ=32` and `CMIX_BITL_TRACE_AUX_LPROJ=32` (write
the 32-value projections instead of the raw vectors).  The run used the
transformer weights of the entry in container format v3 (sha256 `aa1b266f...`;
the shipped format-v4 container differs from it in 86 rows of collapsed
attention channels set to zero, `../README.md` section 6) and input files with
sha256 `0ab9f2d2...` (`.dict`), `14ce984d...` (`.dict.comp`) and `c9f8b100...`
(`.new_article_order`).

**(2) Cut.**  The weights are trained on the first 20 % of the stream only.
Weights for the whole stream would need a complete trace run and, for the
larger amount of data, a longer training, together about three days.
`tools/cut_trace.sh` waits until the trace run has passed 20.00 % of the coder
stream, takes the number of model bits both files hold, keeps 200,000 bits
away from the growing end and rounds down to a multiple of 8; for the shipped
weights this gave B = 418,837,176.  The first model bit is the first bit after
the metadata block, at 0.93 % of the stream.  `record/` holds B, the prefix
sizes and the sha256 of both prefixes.

**(3) Training.**  `trainer/replay_trainer` replays the trace in stream order
through the BITLSTM's own code (`trainer/bit_head.cpp`) and lets it learn with
its own learning step, which is truncated backpropagation over windows of 1,024
model bits from a zero state, 16 windows per Adam step, mean loss over the bits
and global gradient-norm clip 1.0.  `tools/train_blob.sh` runs 32 passes over the
B bits.  Pass e uses the
learning rate 30 + 2970 x (1 + cos(pi (e-1)/31)) / 2, in units of 1e-6 and
rounded (3000, 2992, ..., 30); every pass after the first starts from the
weights of the previous one with fresh optimizer moments; the initial draw uses
the seed 1112363057.  A pass took 6 to 8 minutes on one core.
`record/training.txt` lists the in-sample value of each pass.  This value is the
cost of the replayed bits with the correction minus without it while the BITLSTM
learns; it is no archive size.

**(4) bfloat16.**  `tools/nblb.py v2` rounds the f32 weights of pass 32 to
bfloat16 (round-half-even).

**(5) Embedding.**  A stamping tool of the development tree writes the bytes of
that file unchanged as a byte array into `src/bit_head_weights.h`; the
compressor checks them like a file (magic, version, geometry 16/89/96/28).  The
point where the BITLSTM starts to learn online is the constant 418837176 in
`src/bit_head.cpp`; weights trained on another number of bits need it changed
there.

Weights for a retrained transformer need a new trace, because the inputs of the
BITLSTM (the mixer outputs and the sketch of the transformer state) change with
it. The compressor checks the weights only for their geometry
(`Head::LoadBlob`), so it also runs a retrained transformer of the same
geometry with the shipped weights.

## Files

```
FORMAT.md               trace, channel and weight file
trace/FEATURES_trace.txt  build line of the trace run, one token per line
trace/run_trace.sh      command sequence of the trace run
trainer/bit_head.h      the BITLSTM in the trainer's form (configuration constants)
trainer/bit_head.cpp    the BITLSTM: forward pass, learning step, weight file I/O
trainer/isa_floor.h     AVX2+FMA check
trainer/replay_trainer.cpp  replays a trace through the BITLSTM, one pass per run
trainer/build.sh        builds the trainer (clang 18, the flags of the training)
tools/cut_trace.sh      step (2)
tools/train_blob.sh     steps (3) and (4)
tools/nblb.py           weight file: f32 to bf16, info
tools/blobstat.py       value statistics of a weight file
tools/trace_geom.py     record length, header length and columns of a trace
record/                 the cut, the training passes, sha256 of the weights
```

## Reproducing the weights

```
bash trainer/build.sh
TRACE=<trace> CHANNEL=<channel> PROGRESS=<progress.log of the run> OUT=<dir> \
  bash tools/cut_trace.sh <name>
TRACE=<trace> CHANNEL=<channel> OUT=<dir> bash tools/train_blob.sh <name> 418837176
```

With `<name>` as in `record/README.md`, `sha256sum -c` of `record/SHA256SUMS`
in `<dir>` checks the weights after pass 1, after pass 32 and in bfloat16.

What this package does not contain:

* the development tree that builds the trace writer, so the trace run itself
  cannot be repeated from this package;
* the trace and its channel file (25.1 GB and 14.4 GB up to the cut).
  `record/mark.txt` identifies them by size and sha256.  Whether a new trace
  run gives the same bytes has not been checked.

The trainer is deterministic.  Built from these sources with clang 18.1.3 it
has the same machine code as the binary the shipped weights were trained with,
apart from the text of its messages and the names of its options; on the same
input it writes the same weights (checked on the first 2,000,000 model bits of
the trace, 32 passes).

## Known differences

* The trace build is a measurement build (clang 17, without profile-guided and
  link-time optimisation); the shipped compressor is built with clang 18, a
  profile and link-time optimisation.  The BITLSTM itself is compiled with
  value-safe floating point in both, but the code that computes its inputs is
  not.  The models in front of the BITLSTM learn online on their own
  floating-point results, so another build is another run, and the shipped
  compressor computes inputs that differ from those of this trace along the
  stream.
* In the trace run the BITLSTM itself had 32 cells and learned online; the
  shipped one has 16 cells.  The recorded inputs do not depend on this, because
  they are taken before the BITLSTM and the two projections have their own
  fixed seeds.
* `src/bit_head.cpp` of the entry has the same forward pass, features and
  learning step as the trainer's BITLSTM code.  It adds the compiled-in
  weights and the switch from frozen to online learning, and it skips an
  optimizer step whose gradient norm is not finite; in the training no such
  step occurred (the weights after pass 1 and pass 32 are finite).
* The trainer still carries diagnostics that were not used for the shipped
  weights (`REPLAY_TRUST`, `REPLAY_TRUST_COL`, `REPLAY_WINDOW_PROFILE`,
  `REPLAY_FREEZE_AT`, `REPLAY_CTRL_*`).
