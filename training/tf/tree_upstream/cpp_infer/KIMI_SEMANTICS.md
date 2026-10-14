# KDA / gated-norm / causal-conv exact semantics (fla 0.5.1)

Source of truth: the installed Triton kernels in
`.venv/lib/python3.11/site-packages/fla/`, as invoked by `pysrc/model.py`
`KimiLinearAttention` (fp32 activations, batch 1, varlen `cu_seqlens`,
3 heads, d_head K = V = 64, no initial state). Every formula below was
validated numerically against the kernels by `pysrc/kda_reference.py`
(results in §5). The per-step recurrent form is authoritative from
`fla/ops/kda/fused_recurrent.py` (`fused_recurrent_kda_fwd_kernel`), which the
production `chunk_kda` path reproduces up to kernel-level rounding (§5, §6).

Model call (pysrc/model.py:852):
```
chunk_kda(q, k, v, g=forget_gate_raw, beta=sigmoid(beta_logits),
          A_log=log_baseline_decay_rate,      # (3,)  fp32
          dt_bias=dt_bias,                    # (192,) fp32, viewed (3,64) per head
          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          cu_seqlens=cu_seqlens)              # scale NOT passed
```
`scale` default (fla/ops/kda/chunk.py:414): `K ** -0.5` = **0.125** for K=64.
`use_beta_sigmoid_in_kernel=False` (model applies sigmoid itself),
`safe_gate=False`, `lower_bound=None`, `allow_neg_eigval=False`,
`state_v_first=False` (state is `S[k][v]`), chunk_size=64.

## 1. KDA per-time-step recurrence (C-ready)

Per layer, per head `h` (3 heads), fp32 state `S[64][64]` (first index = k-dim,
second = v-dim), **reset to 0 at every article start** (`cu_seqlens` boundary).
Inputs per token: `q[64], k[64], v[64]` (post conv+silu), `g_raw[64]` (raw
forget-gate pre-activation), scalar `beta` (already sigmoided). All arithmetic
fp32 (fp64 for extra headroom is fine; see §5).

```c
// ---- 1. l2norm of q and k (eps INSIDE sqrt, added to the RAW sum, over 64) ----
float sq = 1e-6f, sk = 1e-6f;                 // fp32(1e-6) = 0x358637BD
for (i) sq += q[i]*q[i];
for (i) sk += k[i]*k[i];
for (i) qn[i] = q[i] / sqrtf(sq);             // kernel divides by sqrt (not *rsqrt)
for (i) kn[i] = k[i] / sqrtf(sk);

// ---- 2. per-channel log-decay gate ----
// A = -exp(A_log[h])  (hoist per head at load; A_log fp32 from checkpoint)
// dt_bias viewed (3,64): row h = dt_bias[64*h .. 64*h+63]
for (i) {
    float x  = g_raw[i] + dt_bias_h[i];
    float sp = (x > 20.0f) ? x : log1pf(expf(x));   // softplus, threshold EXACTLY 20
    decay[i] = expf(A * sp);                        // = exp(-exp(A_log)*softplus(...)) in (0,1]
}

// ---- 3. state update: decay FIRST, then delta rule on the DECAYED state ----
for (i) for (j) S[i][j] *= decay[i];          // decay multiplies k-dim ROWS
for (j) { r[j] = 0; for (i) r[j] += kn[i]*S[i][j]; }   // r = S^T kn  (v-dim vector)
for (j) u[j] = beta * (v[j] - r[j]);
for (i) for (j) S[i][j] += kn[i]*u[j];        // S += outer(kn, u)

// ---- 4. output (scale applied on the q side only; q never touches the state) ----
for (j) { o[j] = 0.0f; for (i) o[j] += (0.125f * qn[i]) * S[i][j]; }   // o = S^T (scale*qn)
```

Equivalently `o = 0.125f * (S^T qn)` (≤1 ulp difference). Matrix form of one step:
`S := diag(exp(g)) S;  S := S + kn (beta (v − S^T kn))^T;  o := S^T (0.125 qn)`
with `g[i] = −exp(A_log[h])·softplus(g_raw[i]+dt_bias[h][i]) ≤ 0`.

Orientation summary (all verified to ~5e-8 against `fused_recurrent_kda`):
- decay is per-**k**-channel: it multiplies the k-indexed rows of `S`, never the v dim;
- the delta-rule correction `r = S^T kn` uses the **already-decayed** `S`;
- the diagonal (current token) IS included in the output: `o_t` is computed
  **after** the time-t rank-1 update, so a token attends to itself;
- `q` is l2-normalized, then multiplied by 0.125; `k` is l2-normalized, NOT scaled;
  `v`, `beta` enter raw.

## 2. FusedRMSNormGated(hidden_size=64, activation="sigmoid")

`fla/modules/fused_norm_gate.py` (`layer_norm_gated_fwd_kernel*`, IS_RMS_NORM=True).
Applied per head vector (module reshapes `[1,T,3,64] -> [T*3, 64]`). Module
default `eps = 1e-5` (fp32(1e-5) = 0x3727C5AC), learnable `weight` shape (64,)
shared across heads (checkpoint key `...output_fused_norm_gate.weight`). All fp32.

```c
// x = kda output o[64] (one head), og = output-gate pre-activation [64], w = weight[64]
float ms = 0.f; for (j) ms += x[j]*x[j];  ms /= 64.0f;   // mean of squares (no eps yet)
float rstd = 1.0f / sqrtf(ms + 1e-5f);                   // eps added to the MEAN, inside sqrt
for (j) y[j] = x[j] * rstd * w[j] * sigmoid(og[j]);      // gate AFTER norm*weight
// sigmoid(z) = 1/(1+expf(-z))
```

It is `rmsnorm(x)*weight*sigmoid(gate)` — the gate does **not** enter the norm
statistics, and the norm statistics do **not** see the gate or weight.

## 3. Causal conv (kernel 4) + SiLU

`fla/modules/conv/triton/kernels.py` (`causal_conv1d_fwd_kernel`), called with
`weight` shape (192, 4), `activation="silu"`, no bias, `cu_seqlens`. Per channel c:

```c
// weight index j multiplies input at t-(W-1)+j:  w[c][3] * NEWEST sample x[t],
// w[c][0] * OLDEST sample x[t-3]. x[<article start] = 0 (reset at every cu_seqlens boundary).
float z = w[c][0]*x[t-3][c] + w[c][1]*x[t-2][c] + w[c][2]*x[t-1][c] + w[c][3]*x[t][c];
y[t][c] = z / (1.0f + expf(-z));              // silu(z) = z * sigmoid(z)
```

The kernel accumulates taps in ascending j order (oldest first), fp32.

## 4. Exact constants and scalar functions (all fp32 literals)

| constant | value | where |
|---|---|---|
| l2norm eps | `1e-6f` added to the **raw sum** of 64 squares, inside sqrt | fused_recurrent.py:155 (hardcoded); l2norm.py `l2norm_fwd` default eps=1e-6 (chunk path) |
| KDA scale | `0.125f` = 64^-0.5 (chunk.py:414 default; model passes none) | multiplies l2-normed q, output path only |
| softplus threshold | `20.0f`: `softplus(x) = x` for `x > 20`, else `log(1+exp(x))` | ops/utils/softplus.py |
| gate | `g = -exp(A_log)*softplus(g_raw + dt_bias)`; decay `= exp(g)` | ops/kda/gate.py, fused_recurrent.py:160-170 |
| gated-norm eps | `1e-5f` added to the **mean** of 64 squares, inside sqrt | FusedRMSNormGated default eps (fused_norm_gate.py:997) |
| sigmoid | `1/(1+exp(-z))` (`tl.sigmoid`) | beta (in model, torch), out-gate, silu |
| silu | `z*sigmoid(z)` | conv activation |
| RCP_LN2 | `1.4426950216` (== fp32(1/ln2) = 0x3FB8AA3B after fp32 rounding) | chunk path only: gates are cumsum-ed in nats then ×RCP_LN2 and exponentiated with `exp2` |

No clamping anywhere in this configuration: `safe_gate=False`, `lower_bound=None`
→ the plain softplus form; decay can underflow toward 0 (with the trained
`exp(A_log) ∈ {1.567, 2.645, 3.129}` and `dt_bias ∈ [-3.26, 0.84]`, per-step
per-channel decays reach ~e-20 for large gate inputs — harmless in fp32).

## 5. Validation results (pysrc/kda_reference.py, RTX 4070 Ti SUPER)

Setup: T=4096, `cu_seqlens=[0,1500,1501,4096]` (includes a length-1 article),
H=3, K=V=64; q,k,v = silu(N(0,1)); g_raw ~ N(0,√2); beta = sigmoid(N(0,1));
A_log/dt_bias loaded from `models/6m-q4-fp32.tch` `blocks.0.attention.*`.
"rel>1e-2" = max relative error over elements with |ref| > 1e-2;
rms(kda ref output) = 9.0e-3, max|ref| = 0.159.

| comparison | max abs | rel>1e-2 |
|---|---|---|
| `fused_recurrent_kda` (fp32 triton) vs reference fp64 | 5.1e-08 | 1.9e-06 |
| `fused_recurrent_kda` vs reference fp32 | 6.0e-08 | 2.3e-06 |
| reference fp32 vs reference fp64 | 2.5e-08 | 6.8e-07 |
| **`chunk_kda` (default = tf32 dots) vs reference fp64** | **2.3e-04** | **3.6e-03** |
| `chunk_kda` with `TRITON_F32_DEFAULT=ieee` vs reference fp64 | 5.9e-06 | 4.1e-04 |
| `chunk_kda` vs `fused_recurrent_kda` (default) | 2.3e-04 | 3.6e-03 |
| FusedRMSNormGated module vs reference (fp64) | 1.2e-06 | 4.2e-07 |
| causal_conv1d module vs reference (fp64) | 8.5e-07 | 1.1e-05 |

chunk(tf32) vs ref64 distribution: mean abs 6.3e-6, median 3.5e-6,
p99.9 7.7e-5, max 2.3e-4; median rel on |ref|>1e-2 elements ≈ 1.1e-3.
Length-1 article closed form `o = 0.125·(qn·kn)·beta·v` matches chunk_kda to
3.1e-5 (tf32) / 2.6e-9 (ieee).

Interpretation: the per-step semantics of §1 are **exact** (the fp32
`fused_recurrent` kernel agrees with the fp64 reference at fp32-rounding level,
5e-8). The remaining `chunk_kda` deviation is the chunk kernel's own arithmetic,
not a semantic difference.

## 6. Kernel-internal dtypes / numerically sensitive spots (for the C++ port)

- **Everything accumulates in fp32** in both kernels: state `S`/`h`, gate,
  cumsum, exp/exp2, l2norm, outputs. Inter-chunk boundary states `h` are stored
  in the input dtype = fp32 (`k.new_empty`, `ops/common/chunk_delta_h.py`).
  No bf16 anywhere in this configuration.
- **TF32 in the production path.** All `tl.dot` matmuls of the chunk kernels
  (intra-chunk Aqk/Akk, UT triangular solve, `w/u`, state propagation, output)
  run on fp32 inputs with Triton's default `input_precision` = **tf32** on
  sm≥80 (`SOLVE_TRIL_DOT_PRECISION='tf32'` explicitly; the rest by default; fla
  only forces `ieee` on pre-Ampere). The repo never overrides
  `TRITON_F32_DEFAULT`, so **the saved reference outputs / dumps were produced
  with tf32 KDA matmuls**. Measured effect vs exact math: ~2.3e-4 max abs
  (~6e-6 mean) on kda raw out whose rms is ~9e-3, i.e. **an exact C++
  implementation will disagree with the Python KDA dumps by up to ~0.3% rel on
  large elements (median ~0.1%)**. This is noise in the *reference*, not error
  in the port; component-test thresholds for `kda raw out` (and anything
  downstream of it pre-quantization) must allow it. The e2e loss criterion is
  the real gate. (`fused_recurrent` uses no `tl.dot` at all — pure fp32 FMAs —
  which is why it matches the reference to 5e-8.)
- Remaining chunk-vs-recurrence gap in ieee mode (~5.9e-6 abs) comes from the
  chunk-form reassociation: per-64-chunk **cumsum of log2 decays** (magnitudes
  up to ~10^2-10^3 in log space; fp32 rounding of the cumsum ⇒ ~3e-5 relative
  error in `exp2` factors) plus `exp2(g·RCP_LN2)` vs `exp(g)`.
- The NVIDIA kernels evaluate softplus's x≤20 branch with **approximate PTX**
  (`ex2.approx.ftz.f32`, `lg2.approx.ftz.f32`:
  `0.6931471805599453 * lg2approx(1 + ex2approx(x * 1.4426950408889634))`) and
  `tl.sigmoid`/`tl.exp` via libdevice. Exact C `log1pf/expf` differ by ~1e-7
  rel; measured overall effect stays inside the 5e-8..5e-6 bands above. Use
  accurate (≤2 ulp) vector implementations; do not replicate the approximations.
- `1e-6f` must be added to the raw sum of squares (NOT the mean) for q/k
  l2norm; `1e-5f` to the mean (NOT the sum) in the gated norm. Don't mix up.
- Order matters: decay → r = S^T kn → u = beta(v−r) → rank-1 update → output.
  Doing the output before the update, or r before decay, is wrong (would show up
  at O(1e-2), not observed).
- Decay factors can underflow to subnormals/0 after long high-decay runs —
  benign; keep flush-to-zero OFF or ON consistently (effect < 1e-30, invisible).
- `beta` is computed by the model in torch (`.sigmoid()` on fp32 matvec output),
  not in the kernel.
- Chunking is per article: `prepare_chunk_indices` restarts 64-token chunks at
  every `cu_seqlens` boundary, so no chunk straddles articles; state resets to 0
  (no initial state passed; `output_final_state` unused).

## 7. Reference implementation

`pysrc/kda_reference.py` — pure-PyTorch (no triton) `kda_recurrent_reference`
(fp32/fp64), `fused_rmsnorm_gated_reference`, `causal_conv1d_silu_reference`,
plus the `__main__` validation that produced §5 (run with `--ieee` for the
`TRITON_F32_DEFAULT=ieee` rows).
