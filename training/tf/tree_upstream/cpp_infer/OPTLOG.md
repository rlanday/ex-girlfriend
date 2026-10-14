# OPTLOG — integration of the optimized transformer (src/opt/model_opt.*)

Machine: EPYC 7702, pinned cpu 42 (CCX {40-43,104-107}, verified idle via
/proc/stat before runs); clang++-17 -O3 -march=znver2, no -ffast-math, MXCSR
default. Core clock measured per run by bench_common.h calibration
(3.330-3.338 GHz across runs). "cyc/token" = core cycles.

## Architecture decisions (initial, from the workstream contracts)

- All dense matmuls: unpacked-int8 qmat arenas (MACHINE.md section 3), built
  at load into ONE contiguous hugepage-advised pool in exact per-token
  consumption order (prior, then per layer [q/k/v proj | gates | kda fp32
  weights | out proj | mlp.up | mlp.down], unembed) so the per-token weight
  read is a single ~5.3 MB sequential stream and every kernel's
  prefetcht0 +2KB runs into the *next* arena.
- mlp.down and prior_embedding: int4-column sparse arenas (qsparse4), index
  lists from the fused mlp.up relu2q epilogue / qsparse_make_idx; dense
  column fallback above the 0.95 density threshold. (The task sheet mentioned
  qsparse_f32 [int8 columns] for the prior; used the int4 variant instead —
  same bit-exact results, it is what bench_qmat's production mode uses, and
  the column fetch is line-bound so 2 lines/col beats 3. Effect either way is
  ~0.1 Kcyc/token on a ~15-nnz prior.)
- KDA: kda_layer_step defaults (ONEFIVE sweep, pf mode 5).
- Vanilla attention: runtime-selected KV variant (constructor arg / --attn):
  AttnKVF32 (fast, 2.81 MB) vs AttnKV int8-V (1.125 MB); low-prob skip
  hardwired OFF (threshold 0 = exact).
- Residual ops via glue (axpby_tok192 / add_scaled192 / qgemv_add epilogues),
  matching the naive "mul-rounded-then-add" contraction shapes bit-exactly.
- beta projection: private inline dot192 (identical reduction to naive
  dot_f32; avoids linking naive kernels into the production binary).

## Correctness gates

1. **e2e loss (THE gate)**: full 1024 articles, f32-V variant:
   mean loss (from logits, double-accumulated) **0.9131351253** vs reference
   0.9131822825 → relative delta **-5.16e-05 (-0.0052%)** → PASS within the
   0.01% target (10x under the 0.1% budget). Naive C++ was -0.0087%; the opt
   engine is *closer* to the Python reference than the naive C++.
   probs vs test_ref_probs.f16: mean row L1 2.878e-2 (naive on an 8-article
   side-by-side: 3.664e-2 vs opt 3.669e-2 — same character, see below).

2. **naive-vs-opt cross-check (articles 0..63)**: max abs prob diff is NOT at
   the ~1e-5 level — and per-component isolation proves this is expected
   seeded divergence, not an integration bug:
   - opt with BOTH recurrent kernels swapped to the naive path
     (-DFX2_XCHECK_NAIVE_KDA -DFX2_XCHECK_NAIVE_ATTN debug builds):
     max abs prob diff **2.26e-6**, max capped-logit diff 1.9e-6 over 4
     articles → the entire integration wiring (arenas/matmuls, glue norms +
     quants, prior, mlp chain, residual/skip plumbing, head) is bit-exact vs
     src/model.cpp; the 2e-6 residue is the documented head tanh256/exp256
     ulps (no feedback).
   - opt attention alone (naive KDA): max 5.0e-2; opt KDA alone (naive
     attention): max 7.5e-2. Mechanism: the documented <=~2-ulp vector-math /
     softmax-reassociation differences occasionally land an activation
     exactly on a round-half-even quantization boundary; the flipped int
     enters the KV ring / KDA state and the divergence persists (first flip:
     article 0 at t=255, article 1 at t=44 — deterministic).
   - The naive C++ itself diverges from the *Python* reference the same way
     and magnitude (8 articles: naive-vs-python mean L1 3.664e-2, max prob
     diff 1.99e-1; opt-vs-python 3.669e-2 / 2.02e-1) — pointwise prob
     agreement at 1e-5 is unattainable for ANY reimplementation that is not
     bit-identical in every fp op; the loss is the meaningful metric and it
     passes with margin.

3. int8-V vs f32-V attention variants: bitwise-identical outputs (attn
   workstream contract, asserted by test_attn); spot-verified e2e (subset
   loss digits identical).

4. Naive suite untouched and still passing (test_components, test_e2e
   --articles 32 spot run — see final report).

## Experiments (accepted / rejected, measured on 48-article subset unless
noted; "idle" = no other load on the machine)

| # | change | result | verdict |
|---|--------|--------|---------|
| 1 | baseline integration, f32-V (idle) | 491.9 Kcyc/tok, 6799 tok/s | reference |
| 2 | attn int8-V variant | 496.4 Kcyc (+4.4K vs f32-V same conditions) | f32-V default (footprint knob kept) |
| 3 | kda pf mode 0 (vs default 5) | +4.3 Kcyc | REJECT: keep pf mode 5 (integration confirms the KDA workstream's choice) |
| 4 | hugepages for weight pool/states | AnonHugePages=0 system-wide: THP=madvise but anon-THP faults never materialize in this container; MADV_HUGEPAGE kept (harmless). The MACHINE.md microbenchmarks ran the same way, so all targets already assume 4K pages | NEUTRAL (no win available) |
| 5 | prefetch first 2KB of each K head stream before attn call | +0.7K under cross-CCX load = noise; retest idle rejected the idea (see 6) | REJECT |
| 6 | cross-CCX co-runner sensitivity | a second instance on cpu 46 (other CCX!) costs ~25 Kcyc/tok (~5%) — DRAM/fabric contention, not L3 | measurement protocol: final numbers on idle machine only |

(continued below with full-set numbers)
