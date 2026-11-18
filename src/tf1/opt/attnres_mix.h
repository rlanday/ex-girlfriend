// AttnRes depth mix of the engine.  AttnRes: Kimi Team (G. Chen,
// Y. Zhang, J. Su, W. Xu et al.), "Attention Residuals", arXiv 2603.15031
// (2026).  Implementation adapted from cmix-neif-beehive (Vladimir
// Malinovskii, 2026, commit 92c07c0b, src/tf1/opt/attnres_mix_v2.h and
// model_opt.cpp, depth-mix form TF_ARV2 = 5, GPL-3.0).
//
// Every sub-layer reads a softmax-weighted mix of the token row, the summed
// outputs of the earlier layer blocks and the running sum of the current
// block, instead of one running residual sum (replace mode; arch.py of
// cmix-neif-beehive):
//
//   l_i = (w . v_i) / rms(v_i),  rms(v) = sqrt(mean_d v_d^2 + 1e-6)     (arch.py _inv_rms, EPS 1e-6)
//   p   = softmax_i(l)                                                 (engine exp: exp256_ps)
//   out = sum_i p_i v_i
//
// The engine evaluates it as an online softmax per read (model_opt.cpp,
// ar_push/ar_read); this header holds its reductions.  D = 192.  Every
// reduction has a fixed order (two FMA accumulators over the 24 vectors, then
// a fixed lane tree), so with the engine's flags (-ffp-model=precise
// -ffp-contract=off -mrecip=none) encoder and decoder compute identical bits.
#pragma once
#include <immintrin.h>

#include <cmath>

#include "vec_math.h"

namespace fx2 {
namespace opt {

// fixed-order horizontal sum of 8 lanes: ((0+4)+(2+6)) + ((1+5)+(3+7))
inline float ar_hsum8(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  const __m128 a = _mm_add_ps(lo, hi);
  const __m128 b = _mm_add_ps(a, _mm_movehl_ps(a, a));
  return _mm_cvtss_f32(_mm_add_ss(b, _mm_movehdup_ps(b)));
}

// a.b over 192 floats (a, b 32-byte aligned)
inline float ar_dot192(const float* a, const float* b) {
  __m256 s0 = _mm256_mul_ps(_mm256_load_ps(a), _mm256_load_ps(b));
  __m256 s1 = _mm256_mul_ps(_mm256_load_ps(a + 8), _mm256_load_ps(b + 8));
  for (int i = 16; i < 192; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_load_ps(a + i), _mm256_load_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_load_ps(a + i + 8), _mm256_load_ps(b + i + 8), s1);
  }
  return ar_hsum8(_mm256_add_ps(s0, s1));
}

// 1 / rms(v)
inline float ar_rstd192(const float* v) {
  return 1.0f / std::sqrt(ar_dot192(v, v) / 192.0f + 1e-6f);
}

// part += f, returns the new partial's 1/rms (same pass)
inline float ar_add_rstd192(float* part, const float* f) {
  __m256 q0 = _mm256_setzero_ps(), q1 = q0;
  for (int d = 0; d < 192; d += 16) {
    const __m256 a = _mm256_add_ps(_mm256_load_ps(part + d), _mm256_load_ps(f + d));
    const __m256 b = _mm256_add_ps(_mm256_load_ps(part + d + 8), _mm256_load_ps(f + d + 8));
    _mm256_store_ps(part + d, a);
    _mm256_store_ps(part + d + 8, b);
    q0 = _mm256_fmadd_ps(a, a, q0);
    q1 = _mm256_fmadd_ps(b, b, q1);
  }
  return 1.0f / std::sqrt(ar_hsum8(_mm256_add_ps(q0, q1)) / 192.0f + 1e-6f);
}

// part += f with the partial's 1/rms and *dot = dot(w, new part) in the same
// pass; the dot has exactly ar_dot192's accumulator layout and FMA order
inline float ar_add_rstd_dot192(float* part, const float* f, const float* w, float* dot) {
  __m256 q0 = _mm256_setzero_ps(), q1 = q0, s0, s1;
  {
    const __m256 a = _mm256_add_ps(_mm256_load_ps(part), _mm256_load_ps(f));
    const __m256 b = _mm256_add_ps(_mm256_load_ps(part + 8), _mm256_load_ps(f + 8));
    _mm256_store_ps(part, a);
    _mm256_store_ps(part + 8, b);
    q0 = _mm256_fmadd_ps(a, a, q0);
    q1 = _mm256_fmadd_ps(b, b, q1);
    s0 = _mm256_mul_ps(_mm256_load_ps(w), a);
    s1 = _mm256_mul_ps(_mm256_load_ps(w + 8), b);
  }
  for (int d = 16; d < 192; d += 16) {
    const __m256 a = _mm256_add_ps(_mm256_load_ps(part + d), _mm256_load_ps(f + d));
    const __m256 b = _mm256_add_ps(_mm256_load_ps(part + d + 8), _mm256_load_ps(f + d + 8));
    _mm256_store_ps(part + d, a);
    _mm256_store_ps(part + d + 8, b);
    q0 = _mm256_fmadd_ps(a, a, q0);
    q1 = _mm256_fmadd_ps(b, b, q1);
    s0 = _mm256_fmadd_ps(_mm256_load_ps(w + d), a, s0);
    s1 = _mm256_fmadd_ps(_mm256_load_ps(w + d + 8), b, s1);
  }
  *dot = ar_hsum8(_mm256_add_ps(s0, s1));
  return 1.0f / std::sqrt(ar_hsum8(_mm256_add_ps(q0, q1)) / 192.0f + 1e-6f);
}

}  // namespace opt
}  // namespace fx2
