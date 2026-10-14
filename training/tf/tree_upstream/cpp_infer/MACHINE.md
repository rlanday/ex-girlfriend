# MACHINE.md — measured peaks for the AVX2 inference engine (EPYC 7702, Zen 2)

All numbers measured 2026-07-19 with the microbenchmarks in `cpp_infer/bench/`
(`alu_peaks`, `membw`, `gemv`, `misc`; raw logs in `cpp_infer/bench/results/`).
Every "per-cycle" figure is in **core cycles**.

## Method / environment

- **Pinned to cpu 42** (`taskset -c 42` + in-process `sched_setaffinity`).
  Topology: SMT sibling = **cpu 106**; CCX = cpus **40-43 + 104-107** share one
  **16 MB L3** (`cache/index3/shared_cpu_list = 40-43,104-107`). The whole CCX
  and the SMT sibling were idle during the runs (machine load ~0.5/128 CPUs).
- **TSC = 2.0000 GHz** (vs `CLOCK_MONOTONIC_RAW`), invariant/nonstop.
- **Core clock under sustained AVX2 load = 3.337 GHz** (median; range
  3.3353–3.3389 across 12 calibrations interleaved through the whole suite).
  Measured with a dependent 1-cycle `vpaddd` chain after ≥0.4 s FMA warm-up;
  cross-checked with a dependent scalar `add` chain (agrees to 0.05%). No
  AVX2/memory-load downclock observed (governor schedutil, boost on; cpufreq's
  "max 2183 MHz" is the ACPI token value, real boost is 3.337 GHz).
  Core cycles = rdtsc ticks × 1.6686.
- Compilers `g++-13` and `clang++-17`, `-O3 -march=znver2` (no `-ffast-math`).
  Buffers 2 MB-aligned, `MADV_HUGEPAGE` for ≥2 MB (THP=madvise).
- Each measurement: 1 warm-up + **5 reps, median reported**; spreads were
  ≤ ±0.5% unless noted. GEMV kernels validated bit-exact vs scalar reference.
- **perf is unavailable**: tool not installed AND `perf_event_open` → EPERM
  (`perf_event_paranoid=4`, unprivileged container). rdtsc-only instrumentation.

## 1. ALU peaks (L1-resident, independent chains, asm loops)

| kernel | per core cycle | note |
|---|---|---|
| `vpmaddubsw` ymm | **0.999** | **1/cycle, NOT 2/cycle** — SPEC §8's "2/cycle (FP0/FP1)" is wrong for Zen 2; single-pipe |
| `vpmaddwd` ymm | 0.998 | same single pipe |
| `vpaddw` ymm | 3.00 | three pipes |
| `vpand` ymm | 3.99 | four pipes |
| `vpsrlw $4` ymm | 0.999 | single pipe, but NOT the madd pipe |
| mix 1×madd + 1×srl | 2.00 IPC | madd+shift coexist perfectly (1+1/cyc) |
| mix 1×madd + 1×shufb | 2.00 IPC | shuffle also off the madd pipe |
| mix 2M+2A (madd:paddw) | 2.00 IPC | i.e. **1 madd + 1 add**/cyc; 2+2 NOT sustainable |
| mix 1M+1A | 2.00 IPC | sustained |
| mix 2M+1A | 1.50 IPC | madd-bound |
| `vfmadd231ps` ymm | **2.00** | peak fp32 = 16 MAC/c = 32 flop/c = 106.8 Gflop/s |
| `vdivps` ymm | 0.286 | 1 per 3.50 cycles |
| 32B loads (L1) | **2.00** | 64 B/cyc from L1 |
| combo 2L+2M+2A | 0.50 groups/c | = 1 load + 1 madd + 1 add per cycle; madd-limited |

**Consequence:** every int8-dot kernel is capped by the single `vpmaddubsw`/
`vpmaddwd` pipe at **32 MAC/cycle** (1 ymm madd × 32 byte-MACs), not 64.

## 2. Sequential-read bandwidth (madd-style consumer unless noted)

B/cyc = bytes per core cycle; GB/s at 3.337 GHz. "raw" = loads + `vpor` sink
(2 loads/cyc capable); "madd" = `vpmaddubsw`+`vpaddw` sink (weight-streaming
mimic, itself capped at 32 B/c by the madd pipe); "pf" = best software
`prefetcht0` distance of {256,512,1024,2048} B.

| size | raw B/c | madd B/c | best-pf B/c (dist) | raw GB/s | level |
|---|---|---|---|---|---|
| 16 KB | 63.0 | 26.9 | 26.0 | 210 | L1 |
| 128 KB | 31.8 | 25.6 | 25.3 | 106 | L2 |
| 384 KB | 30.4 | 25.4 | 25.3 | 101 | L2 |
| 1 MB | 23.1 | 22.1 | 23.1 (2K) | 77.0 | L3 |
| 2 MB | 22.7 | 21.9 | 23.2 (2K) | 75.8 | L3 |
| 4 MB | 22.7 | 21.9 | 23.2 (2K) | 75.8 | L3 |
| 8 MB | 22.3 | 21.5 | 22.9 (2K) | 74.5 | L3 |
| 12 MB | 15.8 | 15.0 | 15.8 (2K) | 52.9 | L3 (degrading) |
| 32 MB | 5.66 | 6.01 | 5.61 | 18.9 | DRAM |
| 256 MB | 5.58 | 6.00 | 5.58 | 18.6 | DRAM (~20.0 GB/s w/ madd) |

Software prefetch (+2 KB) is worth ~+2–6% in L3, useless in L2/L1, harmful in
DRAM. Keep the hot working set ≤ ~8–10 MB: the L3 curve has a knee by 12 MB.

Read+write 50/50 (load 64 B line, modify, store back — KDA-state-like); B/c
counts BOTH directions:

| size | 128 KB | 384 KB | 1 MB | 2 MB | 4 MB | 8 MB |
|---|---|---|---|---|---|---|
| B/cyc (r+w) | 53.0 | 50.0 | 43.5 | 42.9 | 42.9 | 42.9 |
| GB/s (r+w) | 177 | 167 | 145 | 143 | 143 | 143 |

## 3. int4×int8 GEMV — THE decision table

Realistic kernels (validated bit-exact): 192-out unit matrices in one
sequential arena, 4-row groups, per-group 32 B meta (per-row fp32 scale +
per-row int32 `128·Σw` correction for unpacked; scales only for packed —
packed uses unsigned nibbles `w+7` so its correction `7·Σa` is one global
scalar). Per 4-row group: unpacked = 192 w-bytes + 32 meta per 32 in-dims;
packed = half the weight bytes. Footprints cycled; streaming verified by the
L2→L3→DRAM scaling. Arrangements:

- **unpacked**: act biased u8 (`a+128`) in regs/L1, `vpmaddubsw(act, w_mem)`
  with memory-operand weight (load folds), `vpaddw` int16 accum, widen via
  `vpmaddwd(·,1)` (+`vpaddd` every 8 chunks for 768-in), 4-row `vphaddd` tree.
- **packed**: per 32 B load: `vpand 0x0F` / `vpsrlw 4`+`vpand` → two u4-in-u8
  regs, 2× `vpmaddubsw(w_u4, act_s8)` + 2× `vpaddw`.

MAC/cyc, clang-17 build (gcc-13 is 8–15% slower on the same source — it spills
in the inner loops; use clang or asm for the production kernel). "pf" =
`prefetcht0` +2 KB. Best variant per cell:

| footprint | u8 192-in | u8+pf 192-in | p4 192-in | u8 768-in | u8+pf 768-in | p4 768-in |
|---|---|---|---|---|---|---|
| ~20–37 KB (L1/L2) | 21.9 | 22.1 | 18.5–19.7 | — | — | — |
| ~256 KB (L2) | 21.9 | **22.6** | 19.1 | 24.3 | 23.9 | 20.8 |
| ~3 MB (L3) | 19.2 | **20.4** | 18.7 (pf hurts: 17.1) | 20.2 | **21.4** | 20.4 |
| ~6 MB (L3) | 19.2 | **20.5** | 18.6 | 20.0 | **21.2** | 20.4 |
| ~48 MB (DRAM) | 4.16 | 4.0 | **7.77** | 5.73 | 4.5 | **8.25** |

Effective weight-stream bandwidth at the 6 MB footprint: unpacked+pf
**21.3 B/cyc (71 GB/s)** — 92% of the pure-stream L3 limit; packed needs only
10.3 B/cyc (35 GB/s), it is compute/front-end-bound at ~58–63% of the 32 MAC/c
ALU ceiling.

**The real model** streams ≈2.92 MB packed / 5.83 MB unpacked (+4–8% metadata)
per token from L3, ~70% of MACs in 192-in shape. At those footprints:

- unpacked+pf: ~20.4 (192-in) / 21.3 (768-in) MAC/cyc → 5.83 M MAC ≈ **287 K cyc**
- packed: ~18.7 (192-in) / 20.4 (768-in) MAC/cyc → ≈ **306 K cyc** (+7%)

**Recommendation: UNPACKED int8 weights with software prefetcht0 ≈2 KB ahead**
for maximum tokens/s — L3 here is fast enough (23 B/c) that halving bytes does
not pay while the unpack ops do cost. Packed int4 is the fallback if L3
residency matters more: it halves footprint (2.9 vs 5.9 MB of the 16 MB CCX
L3) and L3 traffic for ~6–7% matmul-time cost, and is ~1.9× faster if weights
ever spill to DRAM. (Caveat: unpacked's edge assumes the L3 stream is not
contended; under heavy co-residency packed's 10 B/c demand is far more robust.)

## 4. Misc kernels

| item | value | note |
|---|---|---|
| `exp256_ps` (poly, fma) | **0.59 elem/cyc** (1.70 c/elem) | max rel err 8.0e-8 (~0.7 ulp) vs exp(double), range [-30,30] |
| libm `expf` | 10.0 c/elem | glibc scalar |
| libm `tanhf` | 56.2 c/elem | expensive — vectorize (softcap head) |
| libm `sinf` | 11.9 c/elem | rope tables precomputed anyway |
| QK dot `vpmovsxbw`+`vpmaddwd`, 64B k-rows, hsum/row | **6.39 MAC/cyc** (10.0 cyc/row) | same at 256 KB and 1 MB k-footprint (epilogue/latency-bound, not BW). Batch 4 rows per hsum tree to approach ~2× |
| (invalid) `vpmaddubsw` QK | 7.6 MAC/cyc | **saturates**: (q+128)·k pairs reach 64770 > int16 — only usable when one side is ≤4-bit; keep for reference only |
| 192 B fp32 rows from 1.18 MB, sequential | 22.8 B/cyc | KV-cache-like |
| 192 B fp32 rows, random order | 18.2 B/cyc | only ×1.25 penalty (10.6 cyc/row) |
| `lfence;rdtsc` pair | 40 ticks = **66.8 core cyc** | median back-to-back |
| `rdtscp;lfence` pair | 60 ticks = 100.2 core cyc | prefer lfence;rdtsc |

## 5. Implications for the kernels

1. **The single madd pipe rules everything**: 32 MAC/cyc is the hard ALU
   ceiling for all int8×int4 matmuls (SPEC §8's 2/cycle assumption must be
   halved). 5.83 M MAC/token ⇒ ≥182 K cycles/token for matmuls even with
   perfect kernels; realistic best-measured ≈287 K cyc (unpacked+pf).
2. **Store weights UNPACKED int8, stream with prefetcht0 +2 KB, compile the
   kernel with clang (or hand-asm)**: 20.4–21.4 MAC/cyc at the model's L3
   footprint vs 18.6–20.4 packed; keep the packed path only if the 3 MB of
   extra L3 or co-tenant bandwidth becomes a problem (§3 caveat), and never
   let weights fall out of L3 (DRAM: 4–8 MAC/cyc).
3. **Budget bandwidth per level**: L2 101–106 GB/s (30–32 B/c), CCX-L3 74–77
   GB/s (22–23 B/c) read, 143 GB/s (43 B/c) r+w — KDA state (432 KB r+w per
   token) costs only ~20 K cyc; keep total hot footprint under ~8–10 MB (L3
   knee at 12 MB).
4. **Attention**: QK must use `vpmovsxbw`+`vpmaddwd` (maddubs saturates);
   single-row it runs 6.4 MAC/cyc, so batch 4 k-rows per horizontal-sum and
   the score-skip (<max−21) stays valuable; PV in fp32 at 2 FMA/cyc; softmax
   exp costs only 1.7 cyc/elem with `exp256_ps` (0.7 ulp).
5. **Instrumentation**: `lfence;rdtsc` costs ~67 core cycles per stamp — keep
   instrumented sections ≥10 K cycles for <1% distortion; convert ticks with
   the measured 1.6686 ratio (TSC 2.0 GHz, core 3.337 GHz, stable ±0.1%).
