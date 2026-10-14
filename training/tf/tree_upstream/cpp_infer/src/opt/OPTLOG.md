# Optimization log (whole-project summary)

Baseline → final: naive correctness build **2,475 tok/s (~1.35 M cyc/token)** →
integrated optimized build **6,327 tok/s (527 K cyc/token, f32-V attention)** on
the full 1024-article test set, single thread, EPYC 7702 @ 3.337 GHz.
Correctness at every step: full-set loss 0.9131351253 vs Python reference
0.9131822825 (**−0.0052%**, budget 0.1%; the C++ is slightly *more* accurate
than the reference, which carries GPU tf32 noise in its KDA path).

## Per-workstream results (standalone, realistic-cache harnesses)

| workstream | naive | optimized | how |
|---|---|---|---|
| int4×int8 matmuls (5.83 M MAC/token) | 717 K cyc | **252.6 K** (23.1 eff. MAC/cyc) | unpacked-int8 arenas in consumption order, biased-u8 vpmaddubsw + int16 8-accumulate, per-8/4-row interleaved [corr|scale] metadata, prefetcht0 +2 KB, fused epilogues (relu²+quant+nonzero-idx; residual-add), EXACT sparse mlp.down (int4 columns, 2 lines/col, mean 19.6% density → 3.4×) and sparse prior (~11% density); clang-only inner loops (gcc spills), `#pragma clang loop unroll(disable)` to stop accumulator spills |
| vanilla attention (n̄=863, 3 layers) | 318–327 K | **138.5 K** (f32-V) / 146.6 K (i8-V) | transposed pair-interleaved int8 K per head, vertical vpbroadcastd+vpmovsxbw+vpmaddwd scoring (no h-sums), two kernels (var<1024 / fixed=1024), ring never splits the scan, exp256_ps softmax, fp32-V PV (pure load+FMA, L3-BW-bound); s16-K and skip-threshold variants measured and rejected (skip never fires: score spread ~11 nats) |
| KDA (9 layers, 27 heads) | (in 1.35 M total) | **84.7 K** | fused 1.5-pass state sweep (decay+r accumulation writing decayed rows, then rank-1 update fused with output; bit-identical to naive arithmetic), scheduled L3→L2 prefetch cursor under the nonlinearity passes, staged vectorized exp/softplus/sigmoid/silu (≤2.8 ulp), 70.6% FMA-pipe utilization in sweeps; honest floor ≈67–71 K (transcendental plateau + IEEE divs) |
| glue/vec-math | scalar libm | **~21–25 K** | vec_math.h (exp 1.03 ulp / tanh 1.96 ulp / sigmoid 1.91 ulp, 12–15× libm tanhf), fused rms_norm+m×quantize (IEEE-div bit-exact, div-bound), softcap+softmax head 1.44 K cyc (9–10× naive), F16C prior conversion + sparse index build |

## Integration

- Weight arenas rebuilt at load into exact per-token consumption order (one
  sequential stream for the prefetchers); per-site glue in SPEC §3.3 order.
- Both attention variants integrated behind `--attn f32|int8` (bitwise-identical
  outputs; f32-V faster, i8-V saves 1.7 MB L3).
- Profiling: `-DFX2_PROF` build with raw-rdtsc section accumulators over 24 call
  sites in the 4 report groups; clean build has zero instrumentation.
- Integrated overhead vs sum of standalone parts: 527 K vs ≈500 K ≈ +5%
  (component interference: KV/state streams vs weight stream in L3, plus
  per-call glue).

## Verification history

1. Naive C++ vs Python (GPU): loss −0.0087%; per-component divergence proven
   inherent (tf32 in reference KDA/attention; quantization knife-edges; an
   independent exact-numpy chain reproduces C++ and diverges from the GPU dumps
   identically). clang/gcc builds bit-identical.
2. Optimized kernels vs naive: int paths bit-exact (10 k cases/shape); fp paths
   ≤2 ulp/elem or bit-identical where required (KDA sweep bit-exact; attention
   4.1e-6 max rel from permitted reassociation).
3. Integrated vs naive (64 articles): aggregate loss delta −0.011%; per-row
   diffs are knife-edge chaos (first flip e.g. article 1 t=45), symmetric.
4. Integrated vs Python reference (full 1024 articles): **−0.0052% PASS**.

## Notable rejected/neutral experiments

- Packed-int4 dense weights: theoretical floor is better (compute- not
  BW-bound) but measured kernels are front-end-limited to ~60% of the shared
  madd-pipe ceiling → 306–371 K vs 252–287 K unpacked. Kept as fallback (halves
  footprint; 1.9× better if weights spill to DRAM).
- Attention low-prob skip at any loss-safe threshold: never triggers (diffuse
  attention); machinery kept, off by default.
- s16-stored K: unpack savings < extra L3 bytes.
- In-sweep software prefetch in KDA: net negative (steals AGU/L1-fill from the
  store-saturated pass); replaced by the scheduled front-end cursor.
- lfence-based per-site timing: misattributes overlap; raw rdtsc + clean-stream
  diffs used instead.
