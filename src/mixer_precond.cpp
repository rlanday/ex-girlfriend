// The floating point of the layer-0 column preconditioning (mixer_precond.h) as an
// island translation unit with strict FP semantics (-ffp-model=precise
// -ffp-contract=off, no LTO/PGO), with the same recipe and for the same reason
// as adaptive_depth.cpp: one copy of the arithmetic, shared by encoder and
// decoder, with a fixed summation order.
#include "mixer_precond.h"
#include <immintrin.h>

namespace mixer_precond {

static inline float DetSqrt(float x) {
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
}

const float* L0ColScale(L0Col& c, const float* x, int n) {
  if (n > kMaxN0) n = kMaxN0;
  if (!c.init) {
    for (int i = 0; i < kMaxN0; ++i) { c.v[i] = 1.0f; c.r[i] = 1.0f; }
    c.init = true;
  }
  const float omb = 1.0f / (float)kL0ColN;
  const float beta = 1.0f - omb;
  for (int i = 0; i < n; ++i) c.v[i] = beta * c.v[i] + omb * (x[i] * x[i]);
  if ((c.n & 63) == 0) {
    float sum = 0.0f;   // fixed order: precise TU, no reassociation
    for (int i = 0; i < n; ++i) sum += c.v[i];
    const float ref = DetSqrt(sum / (float)n + 1.0e-12f);
    const float cap = (float)(32);
    const float lo = 1.0f / cap;
    for (int i = 0; i < n; ++i) {
      float r = ref / DetSqrt(c.v[i] + 1.0e-12f);
      if (r > cap) r = cap;
      else if (r < lo) r = lo;
      c.r[i] = r;
    }
  }
  ++c.n;
  for (int i = 0; i < n; ++i) c.xt[i] = x[i] * c.r[i];
  return c.xt;
}

}  // namespace mixer_precond
