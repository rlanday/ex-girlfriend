#include "adaptive_depth.h"

#include <math.h>

#include <immintrin.h>

#include "isa_floor.h"
#include "fast_math.h"

namespace adaptive_depth {

static inline float DetLogistic(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// IEEE-754 square root without a library call.  vsqrtss is correctly rounded
// and therefore returns exactly what sqrtf() returns for every finite
// non-negative argument (g[] is a sum of squares, so it never is negative).
// Needed because -fno-builtin turns sqrtf() into a PLT call.
static inline float DetSqrt(float x) {
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
}

static inline float ZVec(const State& s) {
  const int n = s.n;
  const float* wp = s.w;
  const float* fp = s.f;
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  const int nb = n & ~7;
  for (; i < nb; i += 8) {
    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(wp + i),
                                           _mm256_loadu_ps(fp + i)));
  }
  // Fixed reduction tree: (0+4)(1+5)(2+6)(3+7) -> (a+c)(b+d) -> (e+f).
  const __m128 vlo = _mm256_castps256_ps128(acc);
  const __m128 vhi = _mm256_extractf128_ps(acc, 1);
  const __m128 s4 = _mm_add_ps(vlo, vhi);
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
  float z = _mm_cvtss_f32(s1);
  for (; i < n; ++i) z += wp[i] * fp[i];
  if (z > 30.0f) z = 30.0f;
  else if (z < -30.0f) z = -30.0f;
  return z;
}
#define DEPTH_Z(s) ZVec(s)

// z = clamp(w . f, +-30).
float ZDet(const State& s) { return DEPTH_Z(s); }

float LogDet(float z) { return DetLogistic(z); }

// Both halves in one call: the bit path pays for one call instead of two.
float Gate(State& s) {
  const float z = DEPTH_Z(s);
  s.p = DetLogistic(z);
  return z;
}

// AdaGrad step on the bit that was just coded -- VALUE-IDENTICAL to
// State::Train(), element for element:
//
//   gr   = er * f[i]                 vmulps   (er broadcast, same operands)
//   g[i] = g[i] + gr*gr              vmulps + vaddps, NEVER a single FMA
//                                    (-ffp-contract=off, and no fma intrinsic)
//   w[i] = w[i] - (kLr*gr) / (sqrt(g[i]) + 1e-8)
//                                    vmulps, vsqrtps, vaddps, vdivps, vsubps
//
// Every one of those is an IEEE-754 basic operation on single precision and is
// element independent, so eight lanes at a time give bit for bit the same
// numbers as the scalar loop.  The remainder (n is 33 today, so exactly one
// element) runs scalar in the SAME order, with vsqrtss instead of the call.
// Nothing is read or written beyond s.n, so the padding of the arrays plays no
// part at all.
void TrainStep(State& s, int bit) {
  const int n = s.n;
  const float er = s.p - (float)bit;
  float* wp = s.w;
  float* gp = s.g;
  const float* fp = s.f;

  const __m256 ver = _mm256_set1_ps(er);
  const __m256 vlr = _mm256_set1_ps(kLr);
  const __m256 vep = _mm256_set1_ps(1.0e-8f);

  int i = 0;
  const int nb = n & ~7;
  for (; i < nb; i += 8) {
    const __m256 vf = _mm256_loadu_ps(fp + i);
    const __m256 gr = _mm256_mul_ps(ver, vf);
    const __m256 vg = _mm256_add_ps(_mm256_loadu_ps(gp + i),
                                    _mm256_mul_ps(gr, gr));
    _mm256_storeu_ps(gp + i, vg);
    const __m256 vd = _mm256_add_ps(_mm256_sqrt_ps(vg), vep);
    const __m256 vw = _mm256_sub_ps(
        _mm256_loadu_ps(wp + i),
        _mm256_div_ps(_mm256_mul_ps(vlr, gr), vd));
    _mm256_storeu_ps(wp + i, vw);
  }
  for (; i < n; ++i) {
    const float gr = er * fp[i];
    gp[i] += gr * gr;
    wp[i] -= kLr * gr / (DetSqrt(gp[i]) + 1.0e-8f);
  }
}

}

