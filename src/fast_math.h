
#ifndef FAST_MATH_H_
#define FAST_MATH_H_

namespace fastmath {
void VecExp(float* dst, const float* src, unsigned int n);
void VecLogistic(float* dst, const float* src, unsigned int n);
void VecTanh(float* dst, const float* src, unsigned int n);
}

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

namespace fastmath {

// ---- bfloat16: upper 16 bits of the float, back-cast round-to-nearest-even.
static inline float BF16ToFloat(uint16_t h) {
  uint32_t b = (uint32_t)h << 16;
  float f;
  memcpy(&f, &b, 4);
  return f;
}
static inline uint16_t FloatToBF16(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  uint32_t r = ((b >> 16) & 1u) + 0x7fffu;   // round-to-nearest-even
  return (uint16_t)((b + r) >> 16);
}

// ---- AVX2 blocks: load resp. store 8 weights.
static inline __m256 Load8BF16(const uint16_t* p) {
  __m256i e = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(e, 16));
}
static inline void Store8BF16(uint16_t* p, __m256 v) {
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i bias = _mm256_set1_epi32(0x7fff);
  __m256i b = _mm256_castps_si256(v);
  __m256i r = _mm256_add_epi32(
      _mm256_and_si256(_mm256_srli_epi32(b, 16), one), bias);
  __m256i t = _mm256_srli_epi32(_mm256_add_epi32(b, r), 16);
  _mm_storeu_si128((__m128i*)p,
      _mm_packus_epi32(_mm256_castsi256_si128(t),
                       _mm256_extracti128_si256(t, 1)));
}

static inline float HalfDot(const float* x, const unsigned short* w,
                            unsigned int n) {
  unsigned int i = 0;
  const unsigned int n16 = n & ~15u;
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  for (; i < n16; i += 16) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), Load8BF16(w + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), Load8BF16(w + i + 8), a1);
  }
  __m256 a = _mm256_add_ps(a0, a1);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
  h = _mm_add_ps(h, _mm_movehl_ps(h, h));
  h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
  float p = _mm_cvtss_f32(h);
  for (; i < n; ++i) {
    p += x[i] * BF16ToFloat(w[i]);
  }
  return p;
}

static inline void UpdAxpy4(float* u,
                            const float* x0, const float* x1,
                            const float* x2, const float* x3,
                            float s0, float s1, float s2, float s3,
                            unsigned int n) {
  const __m256 v0 = _mm256_set1_ps(s0), v1 = _mm256_set1_ps(s1);
  const __m256 v2 = _mm256_set1_ps(s2), v3 = _mm256_set1_ps(s3);
  unsigned int j = 0;
  const unsigned int n8 = n & ~7u;
  for (; j < n8; j += 8) {
    __m256 a = _mm256_loadu_ps(u + j);
    a = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x0 + j), a);
    a = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x1 + j), a);
    a = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x2 + j), a);
    a = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x3 + j), a);
    _mm256_storeu_ps(u + j, a);
  }
  if (j < n) {
    const __m256i m = _mm256_cmpgt_epi32(
        _mm256_set1_epi32((int)(n - j)),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256 a = _mm256_maskload_ps(u + j, m);
    a = _mm256_fmadd_ps(v0, _mm256_maskload_ps(x0 + j, m), a);
    a = _mm256_fmadd_ps(v1, _mm256_maskload_ps(x1 + j, m), a);
    a = _mm256_fmadd_ps(v2, _mm256_maskload_ps(x2 + j, m), a);
    a = _mm256_fmadd_ps(v3, _mm256_maskload_ps(x3 + j, m), a);
    _mm256_maskstore_ps(u + j, m, a);
  }
}
// Single epoch (tail, when horizon_ is not divisible by 4).
static inline void UpdAxpy1(float* u, const float* x, float s,
                            unsigned int n) {
  const __m256 v0 = _mm256_set1_ps(s);
  unsigned int j = 0;
  const unsigned int n8 = n & ~7u;
  for (; j < n8; j += 8) {
    _mm256_storeu_ps(u + j, _mm256_fmadd_ps(v0, _mm256_loadu_ps(x + j),
                                            _mm256_loadu_ps(u + j)));
  }
  if (j < n) {
    const __m256i m = _mm256_cmpgt_epi32(
        _mm256_set1_epi32((int)(n - j)),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    _mm256_maskstore_ps(u + j, m,
        _mm256_fmadd_ps(v0, _mm256_maskload_ps(x + j, m),
                        _mm256_maskload_ps(u + j, m)));
  }
}

}
#define LSTMW_CVT(h)      fastmath::BF16ToFloat(h)
#define LSTMW_RCVT(f)     fastmath::FloatToBF16(f)
#define LSTMW_LOAD8(p)    fastmath::Load8BF16(p)
typedef unsigned short lstmw_t;

#endif
