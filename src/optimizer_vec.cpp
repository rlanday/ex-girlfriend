#include "optimizer.h"

#include "isa_floor.h"
#include <immintrin.h>
#include <string.h>

namespace optimizer {
namespace {

const float kBeta1 = 0.025f, kBeta2 = 0.9999f;

struct Core {
  __m256 b1, c1, b2, c2, eps, alpha, a1, rcp;
};

template <bool MITa1>
inline void Block(const Core& k, const float* g, float* m, float* v, float* w) {
  const __m256 gv = _mm256_loadu_ps(g);
  __m256 mv = _mm256_loadu_ps(m);
  __m256 vv = _mm256_loadu_ps(v);
  /* A + B:  m' = (1-beta1)*g + m*beta1 */
  mv = _mm256_fmadd_ps(k.c1, gv, _mm256_mul_ps(mv, k.b1));
  _mm256_storeu_ps(m, mv);
  /* C + D:  v' = (1-beta2)*(g*g) + v*beta2 */
  vv = _mm256_fmadd_ps(k.c2, _mm256_mul_ps(gv, gv), _mm256_mul_ps(vv, k.b2));
  _mm256_storeu_ps(v, vv);
  /* E:  w' = w - (alpha*m') / (a1 * sqrt(v'*rcp_a2 + eps)) */
  const __m256 num = _mm256_mul_ps(mv, k.alpha);
  __m256 den = _mm256_sqrt_ps(_mm256_fmadd_ps(vv, k.rcp, k.eps));
  if (MITa1) den = _mm256_mul_ps(k.a1, den);
  _mm256_storeu_ps(w, _mm256_sub_ps(_mm256_loadu_ps(w),
                                    _mm256_div_ps(num, den)));
}

template <bool MITa1>
inline void Run(const Core& k, const float* g, float* m, float* v, float* w,
                 unsigned int n) {
  unsigned int j = 0;
  for (; j + 8u <= n; j += 8u) Block<MITa1>(k, g + j, m + j, v + j, w + j);
  if (j < n) {
    const unsigned int r = n - j;
    float bg[8] = {0}, bm[8] = {0}, bv[8] = {0}, bw[8] = {0};
    memcpy(bg, g + j, r * sizeof(float));
    memcpy(bm, m + j, r * sizeof(float));
    memcpy(bv, v + j, r * sizeof(float));
    memcpy(bw, w + j, r * sizeof(float));
    Block<MITa1>(k, bg, bm, bv, bw);
    memcpy(m + j, bm, r * sizeof(float));
    memcpy(v + j, bv, r * sizeof(float));
    memcpy(w + j, bw, r * sizeof(float));
  }
}

}  // namespace

void AdamStep(const float* g, float* m, float* v, float* w, unsigned int n,
              float alpha, float a1, float rcp_a2) {
  Core k;
  k.b1 = _mm256_set1_ps(kBeta1);
  k.c1 = _mm256_set1_ps(1.0f - kBeta1);
  k.b2 = _mm256_set1_ps(kBeta2);
  k.c2 = _mm256_set1_ps(1.0f - kBeta2);
  k.eps = _mm256_set1_ps(1e-6f);
  k.alpha = _mm256_set1_ps(alpha);
  k.a1 = _mm256_set1_ps(a1);
  k.rcp = _mm256_set1_ps(rcp_a2);
  if (a1 == 1.0f) Run<false>(k, g, m, v, w, n);
  else Run<true>(k, g, m, v, w, n);
}

}

#include <immintrin.h>

namespace optimizer {
namespace {

inline float HorizSum(__m256 a) {
  __m128 lo = _mm256_castps256_ps128(a);
  __m128 hi = _mm256_extractf128_ps(a, 1);
  __m128 s = _mm_add_ps(lo, hi);
  __m128 t = _mm_castpd_ps(_mm_shuffle_pd(_mm_castps_pd(s), _mm_castps_pd(s), 1));
  s = _mm_add_ps(s, t);
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

template <unsigned int R>
inline void BwdRowsN(const float* err, const float* const* t, float* dst,
                     unsigned int n) {
  __m256 acc[R];
  for (unsigned int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
  const unsigned int n8 = n & ~7u;
  for (unsigned int j = 0; j < n8; j += 8) {
    const __m256 e = _mm256_loadu_ps(err + j);
    for (unsigned int r = 0; r < R; ++r) {
      acc[r] = _mm256_fmadd_ps(e, _mm256_loadu_ps(t[r] + j), acc[r]);
    }
  }
  for (unsigned int r = 0; r < R; ++r) {
    float f = HorizSum(acc[r]);
    for (unsigned int k = n8; k < n; ++k) f += err[k] * t[r][k];
    dst[r] += f;
  }
}

}  // namespace

void BwdRows(const float* err, const float* const* t, float* dst,
             unsigned int n, unsigned int rows) {
  if (rows == 8) {
    BwdRowsN<8>(err, t, dst, n);
    return;
  }
  for (unsigned int r = 0; r < rows; ++r) BwdRowsN<1>(err, t + r, dst + r, n);
}

}

