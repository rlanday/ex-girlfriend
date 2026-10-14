# Theoretical-max bounds (draft; finalized in the report)

Conventions (per the task definition):
- A MAC counts as 2 ops (multiply + accumulate). Peak int8 rate: the single madd
  pipe does 1 vpmaddubsw/cyc = 32 byte-MACs, and the required int16/int32
  accumulate ops fit on other pipes (measured mix 1 madd + 1 add = 2 IPC
  sustained) → **32 int8-MAC/cyc = 64 ops/cyc** is the honest compute ceiling
  counting accumulates. Same for vpmaddwd (16 s16-MAC/cyc) and fp32 FMA
  (2 pipes × 8 lanes = 16 FMA/cyc = 32 ops/cyc).
- Transcendentals are counted at their polynomial mul+acc content
  (exp-class ≈ 13 fp ops/elem); IEEE divisions at the hardware rate
  (vdivps = 1/3.5 cyc). This follows "count multiplies AND accumulates":
  nothing is free, but nothing is charged above its op content.
- Bandwidth bounds use measured best sustained rates at the level each stream
  actually lives: L3 read (prefetched) 23.2 B/cyc; L3 r+w mix 43 B/cyc
  (counting both directions); L2 30-32 B/cyc. Core clock 3.337 GHz.
- Component floor (cycles/token) = max(compute-bound cycles, bandwidth-bound
  cycles); component theoretical-max throughput = min of the two bound
  throughputs. Whole-model theoretical max = 1 / Σ(component floors + measured
  "everything else"), the last term included because a whole-model bound needs
  some time for it but its own max is per instructions not modeled.

Fixed per-token counts (mean over the 2,322,313 predicted positions of the
test set; mean attended keys n̄ = 862.56):

## (1) int4×int8 matmuls
- Dense algorithm: 5,829,504 MAC; unpacked arena 6,035,456 B (w + 8 B/row meta);
  packed arena 3,017,728 B (w/2 + 4 B/row meta).
  - unpacked: BW 6,035,456/23.2 = 260,149 cyc | compute 5,829,504/32 = 182,172
    → floor 260,149 (BW-bound) → 12,828 tok/s
  - packed: BW 3,017,728/23.2 = 130,074 | compute 182,172
    → floor 182,172 (compute-bound) → 18,318 tok/s
- As-executed (exact sparsity: mlp.down mean density 19.6%, prior 11.0%):
  4,371,818 MAC.
  - unpacked(+int4 sparse cols): bytes ≈ 4,440,836 → BW 191,415 | compute
    136,619 → floor 191,415 → 17,434 tok/s
  - packed everywhere: bytes ≈ 2,337,508 → BW 100,755 | compute 136,619
    → floor 136,619 → 24,424 tok/s
- NOTE for the report: the theoretical tables favor packed (it moves the floor
  from bandwidth to compute), but measured packed kernels reach only ~60% of
  the shared 32 MAC/c ALU ceiling (front-end/unpack issue pressure) vs ~64-67%
  for unpacked whose floor is bandwidth — so unpacked is faster in practice
  (measured 252.6K vs 306K+ cyc). We ship UNPACKED (packed = documented fallback).

## (2) vanilla attention mechanism (3 layers, n̄ = 862.56)
- QK: 862.56×192×3 = 496,834 s16-MAC → /16 = 31,052 cyc
- PV: 496,834 FMA → /16 = 31,052 cyc
- softmax: 7,763 exps × 13 ops / 32 = 3,154 cyc + max/sum ≈ 1,941 cyc
- compute bound ≈ 67.2K cyc
- bandwidth (from L3): K int8 496,834 B; V fp32 1,986,617 B (KVF32 variant)
  → 2,483,451/23.2 = 107,045 cyc  [int8-V variant: 993,668/23.2 = 42,830]
- floor: KVF32 107,045 (BW-bound) → 31,175 tok/s; KV8 67,2xx (compute-bound,
  but its real conversion cost is outside the mul/acc model — note in report)

## (3) kimi linear attention mechanism (9 layers, 27 heads)
- state sweeps: 27 × 28,672 ops = 774,144 → /32 = 24,192 cyc
- conv: 41,472 ops → 1,296 cyc; small vector mul/acc ≈ 150 cyc
- transcendentals: ≈ 13,700 exp-class elems × 13/32 ≈ 5,566 cyc;
  IEEE divs ≈ 1,350 ymm-divs × 3.5 ≈ 4,725 cyc
- compute bound ≈ 35.9K cyc
- bandwidth: state r+w 884,736 B / 43 = 20,575 cyc → floor 35.9K (compute)
  → 92,952 tok/s
- (Alternative "practical" floor incl. measured exp256 plateau 1.70 c/elem and
  the measured staging limits: ≈ 51K — quote both, use the strict 35.9K in the
  headline table.)

## (4) everything else — no theoretical max (per instructions); measured only.

## Whole model (fill measured values at report time)
- Σ floors (as-executed, unpacked, KVF32): 191,415 + 107,045 + 35,900 = 334,360
  + else_measured (~22-25K) ≈ ~358K cyc → ≈ 9,320 tok/s at 3.337 GHz
- Σ floors (packed variant): 136,619 + 107,045 + 35,900 + else ≈ ~303K
  → ≈ 11,000 tok/s
- Dense-algorithm variants: unpacked 260,149+107,045+35,900+else ≈ 425K → 7,850
  tok/s; packed 182,172+107,045+35,900+else ≈ 347K → 9,610 tok/s.

L3 hot footprint (for the co-residency section):
weights arenas ~5.9 MB (unpacked; sparse-down int4 cols ~1.2 MB of it) +
KV 2.81 MB (KVF32) or 1.125 MB (KV8) + KDA states 432 KB + conv rings/misc
~50 KB + normed-embedding table 157 KB + rope slice (streamed) ≈ 9.3 MB (KVF32)
/ 7.6 MB (KV8) of 16 MB CCX L3. Expect the thrasher knee near 16 − footprint.
