# fx2-cmix transformer — C++ AVX2 single-thread inference: final report

Machine: AMD EPYC 7702 (Zen 2), 1 thread pinned to cpu 42 (CCX = cpus
40-43/104-107 sharing one 16 MB L3). Measured core clock under sustained AVX2
load: **3.337 GHz** (TSC 2.0 GHz; all cycle numbers are core cycles).
Compiler: clang++-17 `-O3 -march=znver2`, no `-ffast-math`.

## 1. Headline results

| | value |
|---|---|
| **Throughput (clean build, shipped config int8-V)** | **6,464 tokens/s** (516,127 cyc/token; median of 3 full-test-set runs; 154.7 µs/token) |
| f32-V attention variant | 6,406 tokens/s (520,979 cyc/token) |
| Instrumented (profiling) build | 6,329 tokens/s (−1.6%) |
| Naive correctness baseline | 2,475 tokens/s (2.61× speedup) |
| **Loss (full 1024-article set, 2,322,313 predicted tokens)** | **0.9131351253 nats/token vs Python reference 0.9131822825 → −0.0052%** (PASS: 19× inside the ≤0.1% budget; the C++ is slightly *more* accurate — the reference carries GPU tf32 noise) |
| **Fraction of theoretical max (shipped config)** | **60.6%** (of 10,669 tok/s; see §4 — the attention bound charitably ignores int→fp conversion, f32-V config achieves 67.7% of its 9,464 bound) |
| Whole-model theoretical max, without / with weight packing | **10,669 / 12,936 tok/s** (as-executed algorithm; we ship WITHOUT packing — see §4 for why measured packed kernels lose despite the higher bound) |
| L3 hot footprint | ≈ 7.2 MB of 16 MB CCX L3 (int8-V); co-residency curve in §5 |

Streaming interface (`cpp_infer/src/opt/model_opt.h`): `TransformerOpt t(weights_path)`;
per article `t.begin_article(rope_offset /*optional, default 0 = positions from
article start*/)`; per token `t.step(token, prior_f16[205], probs_out[205])` —
emits the fp32 distribution over the NEXT token before seeing it. Weights file
(ints + bf16 scales, written by `pysrc/export_weights.py`) per SPEC §4.

## 2. Correctness verification chain

1. **Weights export**: quantization bit-exact vs the model's own `Quantize`
   modules (max diff 0 on ints and dequantized values, all 111 tensors).
2. **Reference data**: subset reconstruction byte-exact; the GPU rerun
   reproduces the saved `transformer-probs-1024-articles` file bit-exactly
   (476,074,165 f16 values, 0 differ). Exact fp32 reference loss 0.9131822825.
3. **Component tests** (3 dump articles, teacher-forced per-block): clean
   upstream components ≤ 5.4e-5 rel-RMS; KDA outputs at the documented tf32
   noise of the reference (≤ 2.3e-4 abs); vanilla q/k/v ints: 132 flips of
   ±1 in 11.1M ints (rounding knife-edges), rope offsets and the t=1024 window
   transition exact. Chained drift proven inherent via an independent
   exact-fp32 numpy chain that reproduces the C++ and diverges from the GPU
   dumps identically.
4. **Optimized vs naive kernels**: integer paths bit-identical (10k random
   cases per shape incl. negative row scales, saturation, half-even ties);
   KDA state sweep bit-identical arithmetic; attention ≤ 4.1e-6 max rel
   (permitted reassociation + ≤1-ulp exp256); glue bit-exact.
5. **Integrated model**: full-set loss −0.0052% vs reference (above); vs the
   naive C++ over 64 articles the aggregate loss agrees to −0.011% (per-row
   diffs are symmetric quantization-knife-edge chaos seeded at ~4e-6, e.g.
   article 1 first flips at t=45; loss-neutral, same phenomenon as
   C++-vs-Python).
6. clang and gcc builds of the naive path bit-identical; ASan/UBSan clean.

## 3. Per-part breakdown (measured INSIDE full-model inference)

Instrumented build (raw-rdtsc sections, overhead-corrected, 24 sites → the 4
required groups; measured over the full test set, shipped int8-V config;
instrumentation costs 1.6% overall — clean-build throughput is reported in §1):

| part | cyc/token | % of runtime | theoretical floor (cyc) | throughput as fraction of its theoretical max |
|---|---|---|---|---|
| int4×int8 matmuls (all 110/token) | 253,116 | 49.7% | 191,415 (BW-bound, unpacked as-executed) | **75.6%** |
| vanilla attention mechanism | 149,337 | 29.3% | 67,156 (compute, int8-V) / 107,045 (BW, f32-V) | **45.0%** (int8-V; the bound excludes its int→fp conversion work) / f32-V measures 149,933 → **71.4%** of its BW bound |
| kimi (KDA) linear attention mechanism | 88,406 | 17.4% | 35,900 strict (compute) / ≈51,000 practical | **40.6%** strict / **57.8%** practical |
| everything else (norms, quants, rope, residual/skip/token connections, embedding+prior input path, softcap+softmax head) | 18,246 | 3.6% | (not computed, per instructions) | — |
| *sections total* | *509,105* | *100%* | | |
| *outside sections (dispatch/loop residue)* | *≈7,000 (1.4% of the clean 516,127)* | | | |

Notes: "measured inside the whole transformer" is exactly what the table shows
(the profiling build runs the full model; standalone-harness numbers differed:
e.g. KDA standalone 84.7K vs 88.4K in-model, attention f32-V standalone 138.5K
vs 149.9K in-model — cache interference between components is real, which is
why kernels were tuned under emulated whole-model cache pressure). The
practical KDA floor prices its irreducible transcendentals at the measured
vector-exp plateau (1.70 c/elem) instead of their polynomial op content.

Hot-site detail (top 5): mlp up+relu²+quant 99.5K; kimi q/k/v projections
51.6K; attention QK+softmax+PV 149.0K; KDA fused step 87.9K; mlp down sparse
35.0K.

## 4. Theoretical maximum (definition and derivation)

Definition per the task: per component, bound throughput = min(bandwidth bound,
compute bound); whole-model max = 1/Σ(component floor times), with "everything
else" included at its measured time (its own max is not computed, per
instructions). Ops count multiplies AND accumulates: a MAC = 2 ops.

Measured machine peaks used (cpp_infer/MACHINE.md): single madd pipe → 32
int8-MAC/cyc (the required int16/int32 accumulates fit on other pipes — the
1 madd + 1 add mix sustains 2 IPC, so 32 MAC/cyc counts both op kinds);
16 s16-MAC/cyc (vpmaddwd); 16 fp32-FMA/cyc (2 pipes; = 32 flop/cyc);
vdivps 1/3.5 cyc; L3 read (prefetched) 23.2 B/cyc; L3 read+write mix 43 B/cyc;
transcendentals priced at polynomial mul+acc content (~13 ops/elem exp-class)
in strict bounds.

Per-token component bounds (as-executed algorithm: exact sparsity skips zero
activations — mlp.down mean density 19.6%, prior 11%):

| component | compute bound | bandwidth bound | floor |
|---|---|---|---|
| matmuls, unpacked int8 (4.372M MAC, 4.44 MB from L3) | 136,619 | **191,415** | 191,415 |
| matmuls, packed int4 (2.34 MB from L3) | **136,619** | 100,755 | 136,619 |
| vanilla attention, int8-V (n̄=862.6 keys: QK 496,834 s16-MAC, PV 496,834 FMA, 7,763 exps; 0.99 MB KV from L3) | **67,156** | 42,830 | 67,156 |
| vanilla attention, f32-V (2.48 MB KV from L3) | 67,156 | **107,045** | 107,045 |
| KDA (774,144 sweep ops + conv 41,472 + transcendentals + ~1,350 divs; 0.885 MB state r+w) | **35,900** | 20,575 | 35,900 |
| everything else | measured 18,246 | | 18,246 |

Whole-model theoretical max (Σ floors, at 3.337 GHz):

| weights | attention | Σ floor cyc/token | theoretical max | achieved |
|---|---|---|---|---|
| **unpacked (shipped)** | **int8-V (shipped)** | 312,717 | **10,669 tok/s** | 6,464 = **60.6%** |
| unpacked | f32-V | 352,561 | 9,464 tok/s | 6,406 = 67.7% |
| packed | int8-V | 257,921 | 12,936 tok/s | (not built end-to-end) |
| packed | f32-V | 297,765 | 11,205 tok/s | — |

Dense-algorithm equivalents (no sparsity; 5.83M MAC, 6.04 MB unpacked /
3.02 MB packed): unpacked floor 260,149 (BW), packed floor 182,172 (compute) →
whole-model 8,747 / 10,995 tok/s (int8-V attention).

**Packing decision**: we ship UNPACKED int8 weights (+ int4 only in the sparse
mlp.down/prior column arenas). The packed-int4 bound is higher on paper (it
turns the matmul floor from bandwidth- to compute-bound), but the bound assumes
the shared single madd pipe can be saturated while also unpacking nibbles;
measured packed kernels reach only ~60% of that ceiling (front-end/issue
pressure) vs unpacked's 75.6% of its bandwidth floor — in cycles: 253K
(unpacked, as-executed) vs 306-371K (packed variants measured in the kernel
harness). Packed remains the documented fallback: it halves the weight
footprint (2.9 vs 5.9 MB) and weight L3 traffic, and is ~1.9× faster if
weights ever spill to DRAM (co-tenancy robustness).

## 5. L3 cache usage

Hot per-token working set (shipped config): weight arenas ≈ 5.41 MB (dense
unpacked arenas 4.21 MB + sparse int4-column arenas 1.20 MB) + KV rings
1.125 MB + KDA states 0.42 MB + normed-embedding table 0.157 MB + conv
rings/buffers/rope-slice ≈ 0.1 MB ≈ **7.2 MB** of the 16 MB CCX L3
(f32-V variant: +1.7 MB → 8.9 MB).

Empirical co-residency (an L3-thrashing reader on another core of the same
CCX while the transformer runs; throughput vs thrasher footprint):

| thrasher footprint (MB) | tok/s (128-article subset) | slowdown |
|---|---|---|
| none | 6,530 (subset baseline) | — |
| 0.5 | 6,809 | ~0 (noise) |
| 1 | 6,660 | ~0 (noise) |
| 2 | 6,430 | −1.5% |
| 3 | 6,228 | −4.6% |
| 4 | 6,041 | −7.5% |
| 5 | 5,300 | −18.8% |
| 6 | 4,853 | −25.7% |
| 8 | 4,016 | −38.5% (worst) |
| 10 | 5,729–5,366 | recovers (thrasher goes DRAM-bound, its L3 pressure drops) |
| SMT sibling, 2–4 MB | 2,913–2,782 | −55…57% (dominated by pipeline sharing, not cache — any same-core co-runner costs ~2× regardless of its footprint) |

Reading: **a worst-case co-tenant (full-speed L3 streamer on another core of
the CCX) can use ≈3 MB with ≤5% impact**; the degradation past 4 MB is L3
*bandwidth* contention as much as capacity (the transformer streams ~21 B/cyc
from L3 in its matmul phase). A low-intensity co-tenant is bounded by capacity
instead: the transformer's hot set is ≈7.2 MB, leaving **≈8–9 MB of the 16 MB
CCX L3** usable without evicting it. Footprint knobs, in order of L3 saved:
packed-int4 weights (−2.9 MB, −4% throughput), int8-V KV (default; −1.7 MB vs
f32-V, +1% throughput). Numbers are single runs (noise ±2–4%); the same-CCX
sweep used a different core (43) than the transformer (42).

## 6. Reproduction

```
cd /root/fx2-cmix-transformer/cpp_infer
make opt                                         # top-level Makefile, clang++-17
./bin/test_e2e_opt                               # full-set loss verdict
./bin/bench_opt --repeats 3 --cpu 42 --attn int8 # clean throughput
./bin/bench_opt_prof --repeats 1 --cpu 42 --attn int8   # per-part profile
./bin/test_xcheck --articles 64 --gate 1         # naive-vs-opt aggregate
```
Python: `pysrc/export_weights.py`, `pysrc/export_testdata.py` (already run;
outputs in `cpp_infer/data/`). Docs: `SPEC.md`, `KIMI_SEMANTICS.md`,
`MACHINE.md`, `THEORY_DRAFT.md`, `src/opt/OPTLOG.md`.
