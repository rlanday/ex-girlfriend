#include "fast_math.h"

#include "isa_floor.h"
#include <immintrin.h>
#include <stdint.h>
namespace fastmath {
namespace {

inline __m256 Exp8(__m256 x) {
  /* Clamp the range: n stays in [-126, 127], so the exponent field is valid.
   * e^88 overflows float; the original returns inf there, this one
   * 2^127*poly ~ 2.5e38.  Both values are indistinguishable in the softmax
   * after normalization (and unreachable in tanh/logistic). */
  x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
  const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896339e+00f));
  const __m256 n = _mm256_round_ps(
      t, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-6.93359375e-01f), x);
  r = _mm256_fmadd_ps(n, _mm256_set1_ps(2.12194440e-04f), r);
  __m256 p = _mm256_set1_ps(1.0f / 720.0f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
  const __m256i ni = _mm256_cvttps_epi32(n);          /* n is integral */
  const __m256i b = _mm256_slli_epi32(
      _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(b));
}

inline __m256 Logistic8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 neg = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
  return _mm256_div_ps(one, _mm256_add_ps(one, Exp8(neg)));
}

inline __m256 Tanh8(__m256 x) {
  const __m256 one  = _mm256_set1_ps(1.0f);
  const __m256 sgnm = _mm256_set1_ps(-0.0f);
  const __m256 ax   = _mm256_andnot_ps(sgnm, x);
  const __m256 sgn  = _mm256_and_ps(sgnm, x);
  const __m256 t    = Exp8(_mm256_mul_ps(ax, _mm256_set1_ps(-2.0f)));
  __m256 r = _mm256_div_ps(_mm256_sub_ps(one, t), _mm256_add_ps(one, t));
  /* |x| > 20 -> 1 (indistinguishable in float anyway) */
  r = _mm256_blendv_ps(r, one,
      _mm256_cmp_ps(ax, _mm256_set1_ps(20.0f), _CMP_GT_OQ));
  /* Small |x|: (1-t)/(1+t) cancels - at x = 1e-3, 1-t is ~ 2e-3, and one
   * float ulp in t is already 5e-5 relative.  Hence below 0.625 the
   * standard polynomial (Cephes tanhf, Stephen L. Moshier), accurate to
   * 1 ulp.  Both
   * branches are always computed and blended by mask - one fixed path,
   * no branching, no second code path. */
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 q = _mm256_set1_ps(-5.70498872745e-3f);
  q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(2.06390887954e-2f));
  q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(-5.37397155531e-2f));
  q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(1.33314422036e-1f));
  q = _mm256_fmadd_ps(q, z, _mm256_set1_ps(-3.33332819422e-1f));
  const __m256 small = _mm256_fmadd_ps(_mm256_mul_ps(q, z), x, x);
  r = _mm256_blendv_ps(r, _mm256_andnot_ps(sgnm, small),
      _mm256_cmp_ps(ax, _mm256_set1_ps(0.625f), _CMP_LT_OQ));
  return _mm256_or_ps(r, sgn);
}

/* Tail through the same kernel: 8-wide buffer, so that no second code path
 * exists (otherwise the last n%8 elements would round differently). */
template <__m256 (*F)(__m256)>
inline void Apply(float* dst, const float* src, unsigned int n) {
  unsigned int i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(dst + i, F(_mm256_loadu_ps(src + i)));
  if (i < n) {
    float in[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float out[8];
    for (unsigned int k = i; k < n; ++k) in[k - i] = src[k];
    _mm256_storeu_ps(out, F(_mm256_loadu_ps(in)));
    for (unsigned int k = i; k < n; ++k) dst[k] = out[k - i];
  }
}

}  /* namespace */

void VecExp(float* dst, const float* src, unsigned int n) {
  Apply<Exp8>(dst, src, n);
}
void VecLogistic(float* dst, const float* src, unsigned int n) {
  Apply<Logistic8>(dst, src, n);
}
void VecTanh(float* dst, const float* src, unsigned int n) {
  Apply<Tanh8>(dst, src, n);
}

}

