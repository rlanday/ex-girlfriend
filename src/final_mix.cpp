// The Bayes term and the SSE blend of the end mixer (final_mix.h) as an island
// translation unit with strict FP semantics (-ffp-model=precise
// -ffp-contract=off): one copy of the arithmetic, shared by encoder and
// decoder.
#include "final_mix.h"
#include <math.h>

namespace final_mix {

static inline double Clamp(double p) { return p < 1e-6 ? 1e-6 : (p > 1.0 - 1e-6 ? 1.0 - 1e-6 : p); }

__attribute__((noinline)) float BayMix(Bay& b, float zrow, const float* s4) {
  double mx = b.lw[0];
  for (int k = 1; k < kExperts; ++k) if (b.lw[k] > mx) mx = b.lw[k];
  double v[kExperts], sum = 0.0, pm = 0.0;
  for (int k = 0; k < kExperts; ++k) { v[k] = exp(b.lw[k] - mx); sum += v[k]; }
  for (int k = 0; k < kExperts; ++k) {
    b.pk[k] = Clamp(1.0 / (1.0 + exp(-(double)s4[k])));
    pm += v[k] / sum * b.pk[k];
  }
  pm = Clamp(pm);
  b.s = (float)log(pm / (1.0 - pm));
  b.z = zrow + b.w * b.s;
  return b.z;
}

__attribute__((noinline)) float BayLearn(Bay& b, int bit, float lr0) {
  const float e = (float)(1.0 / (1.0 + exp(-(double)b.z)) - (double)bit);
  b.w -= lr0 * ((float)kBayLr * 0.001f) * e * b.s;
  for (int k = 0; k < kExperts; ++k) b.lw[k] += log(bit ? b.pk[k] : 1.0 - b.pk[k]);
  double mx = b.lw[0];
  for (int k = 1; k < kExperts; ++k) if (b.lw[k] > mx) mx = b.lw[k];
  double v[kExperts], sum = 0.0;
  for (int k = 0; k < kExperts; ++k) { v[k] = exp(b.lw[k] - mx); sum += v[k]; }
  const double a = (double)100 * 1e-4;
  for (int k = 0; k < kExperts; ++k) b.lw[k] = log((1.0 - a) * v[k] / sum + a / 4.0);
  return e;
}

__attribute__((noinline)) float SseMix(float p_sse, float m1raw) {
  const double a = (double)400 * 1e-3;
  const double p = Clamp((double)p_sse);
  const double z = a * log(p / (1.0 - p)) + (1.0 - a) * (double)m1raw;
  return (float)(1.0 / (1.0 + exp(-z)));
}

}  // namespace final_mix
#include "semantic_cache.h"
#include <stdint.h>
#include <string.h>

namespace semantic_cache {

namespace {
inline uint32_t HighWord(double x) { uint64_t u; memcpy(&u, &x, 8); return (uint32_t)(u >> 32); }
inline double WithHighWord(double x, uint32_t h) {
  uint64_t u;
  memcpy(&u, &x, 8);
  u = (u & 0xffffffffull) | ((uint64_t)h << 32);
  double r;
  memcpy(&r, &u, 8);
  return r;
}
}  // namespace

// log1p after fdlibm s_log1p.c as glibc 2.39 carries it (sysdeps/ieee754/dbl-64/s_log1p.c, with the
// polynomial evaluation for pipelined processors of Naohiko Shimizu, Tokai University, 1997), and with
// the fused multiply-adds of its x86-64 FMA variant (__log1p_fma) at the same places: bit-identical to
// that variant on every float input in (-1, 0) and (0, 1) (exhaustive test).  The only caller,
// TreffLogit in predictor.cpp (the second end stage), passes arguments in (-1, 0).
//
// fdlibm s_log1p.c 5.1 93/09/24:
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
__attribute__((noinline)) double Log1p(double x) {
  const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10;
  const double Lp1 = 6.666666666666735130e-01, Lp2 = 3.999999999940941908e-01, Lp3 = 2.857142874366239149e-01,
               Lp4 = 2.222219843214978396e-01, Lp5 = 1.818357216161805012e-01, Lp6 = 1.531383769920937332e-01,
               Lp7 = 1.479819860511658591e-01;
  double c = 0.0, f = 0.0, u;
  int32_t k = 1, hu = 0;
  const int32_t hx = (int32_t)HighWord(x);
  const int32_t ax = hx & 0x7fffffff;
  if (hx < 0x3FDA827A) {                       // x < 0.41422
    if (ax >= 0x3ff00000) return x == -1.0 ? -__builtin_inf() : __builtin_nan("");   // x <= -1
    if (ax < 0x3e200000) {                     // |x| < 2^-29
      if (ax < 0x3c900000) return x;           // |x| < 2^-54
      return __builtin_fma(-(x * x), 0.5, x);
    }
    if (hx > 0 || hx <= (int32_t)0xbfd2bec3) { k = 0; f = x; hu = 1; }   // -0.2929 < x < 0.41422
  } else if (hx >= 0x7ff00000) {
    return x + x;
  }
  if (k != 0) {
    if (hx < 0x43400000) {
      u = 1.0 + x;
      hu = (int32_t)HighWord(u);
      k = (hu >> 20) - 1023;
      c = (k > 0) ? 1.0 - (u - x) : x - (u - 1.0);   // correction term
      c /= u;
    } else {
      u = x;
      hu = (int32_t)HighWord(u);
      k = (hu >> 20) - 1023;
      c = 0.0;
    }
    hu &= 0x000fffff;
    if (hu < 0x6a09e) {
      u = WithHighWord(u, (uint32_t)(hu | 0x3ff00000));   // normalize u
    } else {
      k += 1;
      u = WithHighWord(u, (uint32_t)(hu | 0x3fe00000));   // normalize u / 2
      hu = (0x00100000 - hu) >> 2;
    }
    f = u - 1.0;
  }
  const double hfsq = (0.5 * f) * f;
  if (hu == 0) {                               // |f| < 2^-20
    if (f == 0.0) {
      if (k == 0) return 0.0;
      const double dk = (double)k;
      return __builtin_fma(dk, ln2_hi, __builtin_fma(dk, ln2_lo, c));
    }
    const double R = __builtin_fma(-f, 0.66666666666666666, 1.0) * hfsq;
    if (k == 0) return f - R;
    const double dk = (double)k;
    return __builtin_fma(dk, ln2_hi, -((R - __builtin_fma(dk, ln2_lo, c)) - f));
  }
  const double s = f / (2.0 + f);
  const double z = s * s;
  const double R2 = __builtin_fma(z, Lp3, Lp2);
  const double R3 = __builtin_fma(z, Lp5, Lp4);
  const double R4 = __builtin_fma(z, Lp7, Lp6);
  const double z2 = z * z;
  const double z4 = z2 * z2;
  const double z6 = z2 * z4;
  const double R = __builtin_fma(z6, R4, __builtin_fma(z4, R3, __builtin_fma(z, Lp1, z2 * R2)));
  const double t = (hfsq + R) * s;
  if (k == 0) return f - (hfsq - t);
  const double dk = (double)k;
  return __builtin_fma(dk, ln2_hi, -((hfsq - (__builtin_fma(dk, ln2_lo, c) + t)) - f));
}

// log2 through the deterministic log (src/detmath.cpp in every package build)
__attribute__((noinline)) double Log2(double x) { return log(x) / 0.69314718055994530942; }

}  // namespace semantic_cache
