# Lossless weights-file compression

Alternative save (Python) / load (C++) functions for `cpp_infer/data/weights.bin`.
The original `write_tensor_file` / `WeightsFile::load` are untouched;
`bin/test_weights_compressed <reference.bin> <compressed>` verifies that both
loaders yield bit-for-bit identical tensors.

- encoder / Python decoder: `pysrc/weights_compress.py`
  (`write_tensor_file_v1/v2`, `read_tensor_file_v1/v2`, CLI
  `compress` (v2) / `compress1` (v1) / `decompress` / `verify`)
- C++ decoder: `WeightsFile::load_compressed` in
  `src/weights_io_compressed.cpp` (dispatches on the magic; compiled `-Os`,
  it is load-time-only code)

## Results (reference file: 39,615,533 bytes)

| format | size | ratio |
|---|---|---|
| FX2TFW01 (original) | 39,615,533 | 1.00x |
| FX2TFWC1 (v1: log2(15) bits/int4 weight, rest raw) | 36,613,691 | 1.08x |
| FX2TFWC2 (v2: everything below) | 2,930,652 | 13.52x |

v1 stores each of the 5,868,864 quantized int4 weights (15 possible values,
[-7, 7]) at log2(15) = 3.907 bits via a range coder with a uniform 1/15 model;
bf16/f32/i32 payloads and metadata stay raw.

## v2: ideas tried, and gain vs. decompressor-code cost

Marginal decompressor code measured on `-Os` `.text` by ablating one feature
at a time (whole v1+v2 decoder: 4,889 bytes of code; at the default `-O3` it
inflates to 20.7 KB for zero benefit, hence the `-Os` rule in the Makefile).

| idea | bytes saved | marginal code | kept |
|---|---|---|---|
| recompute rope.sin/rope.cos instead of storing (33,554,432 B of f32) | 33,554,432 | 1,160 B | yes |
| int4: adaptive 15-symbol bit-tree instead of uniform 1/15 (3.786 vs 3.907 bits/weight) | 63,347 | 202 B | yes |
| bf16 scales: adaptive hi-byte model + lo-byte model contexted on hi (52,166 B raw -> 23,377) | 28,789 | 324 B | yes |
| non-rope f32 + i32: per-byte-plane adaptive models (113,216 B raw -> 97,312) | 15,904 | 196 B | yes |
| tensor names: order-2 adaptive char model (22,213 B raw -> 5,038) | 17,175 | ~0 B | yes |
| metadata (dtype/ndim/shape/lengths): order-1 byte model (6,672 B raw -> 1,622) | 5,050 | shared | yes |

Ideas measured and **not** used:
- order-1 context for int4 weights (previous weight): conditional entropy
  3.7858 vs 3.7860 bits — the quantized weights are essentially i.i.d.
- per-tensor int4 frequency tables: +3.7 KB over global adaptive at best,
  adaptation already captures it.
- rope as correctly-rounded sin + coded ulp deltas (the fallback if exact
  recomputation had failed): deltas are only {-1,0,+1} but cost 0.63
  bits/value even with rounding-boundary-distance context, ~660 KB total —
  obsoleted by the exact reproduction.

## Recomputing the rope tables bit-exactly

`rope.sin`/`rope.cos` are `torch.outer(arange(131072), inv_freq).sin()/.cos()`
computed in fp32 on CUDA — 85% of the file, but fully determined by the
128-byte `rope.inv_freq`. The decoder reproduces CUDA's `sinf`/`cosf`
bit-exactly with a host port of libdevice `__nv_sinf`/`__nv_cosf`, transcribed
from the LLVM IR of `/usr/local/cuda/nvvm/libdevice/libdevice.10.bc` (CUDA
13.0): Cody-Waite 3-step FMA reduction for |a| < 105615, Payne-Hanek integer
reduction above, shared polynomial kernel (cos = sin with quadrant + 1).
Verified bit-identical on all 8,388,608 table entries, including the 25,457
Payne-Hanek arguments (position*inv_freq[0] >= 105615).

The Python encoder verifies recomputability before dropping the tables (and
falls back to storing them if the check fails, e.g. for a checkpoint whose
tables came from different hardware). Its emulation reproduces `fmaf` exactly:
double arithmetic (exact product + one rounding) with a detect-and-redo of
double-rounding hazards via exact rational arithmetic.

## Stream format (v2)

`FX2TFWC2` magic, `u32 n_tensors`, then a single LZMA-style range-coded
stream (11-bit adaptive binary probabilities, shift-5 update, bit-tree symbol
coding). Per tensor: name length + name chars + dtype + ndim + 4-byte dims +
encoding byte, then the payload per the encoding
(`ENC_RAW/INT4/BF16/PLANE4/ROPE_SIN/ROPE_COS`). The Python encoder and both
decoders must stay in exact sync.

Decode time for the full file: < 0.3 s single-threaded (EPYC 7702).
