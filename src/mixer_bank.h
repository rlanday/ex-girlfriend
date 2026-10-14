#ifndef MIXER_BANK_H_
#define MIXER_BANK_H_

#include <immintrin.h>

namespace mixbank {

// Rows per block (DotRows<kBank>, TrainRows<kBank>).
constexpr int kBank = 4;

static inline float HAdd(__m256 acc) {
  const __m128 d = _mm_add_ps(_mm256_castps256_ps128(acc),
                              _mm256_extractf128_ps(acc, 1));
  const __m128 sw = _mm_castpd_ps(
      _mm_shuffle_pd(_mm_castps_pd(d), _mm_castps_pd(d), 1));
  const __m128 e = _mm_add_ps(d, sw);
  return _mm_cvtss_f32(_mm_add_ss(e, _mm_movehdup_ps(e)));
}

template <int R>
static inline void DotRows(const float* __restrict x,
                           const float* const* __restrict w,
                           unsigned int n, float* __restrict out) {
  __m256 acc[R];
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
  const unsigned int n8 = n & ~7u;
  unsigned int i = 0;
  for (; i < n8; i += 8) {
    const __m256 xv = _mm256_loadu_ps(x + i);
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r)
      acc[r] = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w[r] + i), acc[r]);
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) out[r] = HAdd(acc[r]);
  for (; i < n; ++i) {
    const float xv = x[i];
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) out[r] += xv * w[r][i];
  }
}

template <int R>
static inline void TrainRows(const float* __restrict x,
                             float* const* __restrict w,
                             unsigned int n, const float* __restrict u) {
  __m256 uv[R];
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) uv[r] = _mm256_set1_ps(u[r]);
  const unsigned int n8 = n & ~7u;
  unsigned int i = 0;
  for (; i < n8; i += 8) {
    const __m256 xv = _mm256_loadu_ps(x + i);
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r)
      _mm256_storeu_ps(w[r] + i,
          _mm256_fnmadd_ps(uv[r], xv, _mm256_loadu_ps(w[r] + i)));
  }
  for (; i < n; ++i) {
    const float xv = x[i];
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) w[r][i] -= u[r] * xv;
  }
}

}

#endif
