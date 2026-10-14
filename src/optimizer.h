
#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

namespace optimizer {

void BwdRows(const float* err, const float* const* t, float* dst,
             unsigned int n, unsigned int rows);
}

#include <immintrin.h>
namespace optimizer {

static inline void UpdAxpy8(float* u,
                            const float* x0, const float* x1,
                            const float* x2, const float* x3,
                            const float* x4, const float* x5,
                            const float* x6, const float* x7,
                            float s0, float s1, float s2, float s3,
                            float s4, float s5, float s6, float s7,
                            unsigned int n) {
  const __m256 v0 = _mm256_set1_ps(s0), v1 = _mm256_set1_ps(s1);
  const __m256 v2 = _mm256_set1_ps(s2), v3 = _mm256_set1_ps(s3);
  const __m256 v4 = _mm256_set1_ps(s4), v5 = _mm256_set1_ps(s5);
  const __m256 v6 = _mm256_set1_ps(s6), v7 = _mm256_set1_ps(s7);
  unsigned int j = 0;
  const unsigned int n8 = n & ~7u;
  for (; j < n8; j += 8) {
    __m256 a = _mm256_loadu_ps(u + j);
    a = _mm256_fmadd_ps(v0, _mm256_loadu_ps(x0 + j), a);
    a = _mm256_fmadd_ps(v1, _mm256_loadu_ps(x1 + j), a);
    a = _mm256_fmadd_ps(v2, _mm256_loadu_ps(x2 + j), a);
    a = _mm256_fmadd_ps(v3, _mm256_loadu_ps(x3 + j), a);
    a = _mm256_fmadd_ps(v4, _mm256_loadu_ps(x4 + j), a);
    a = _mm256_fmadd_ps(v5, _mm256_loadu_ps(x5 + j), a);
    a = _mm256_fmadd_ps(v6, _mm256_loadu_ps(x6 + j), a);
    a = _mm256_fmadd_ps(v7, _mm256_loadu_ps(x7 + j), a);
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
    a = _mm256_fmadd_ps(v4, _mm256_maskload_ps(x4 + j, m), a);
    a = _mm256_fmadd_ps(v5, _mm256_maskload_ps(x5 + j, m), a);
    a = _mm256_fmadd_ps(v6, _mm256_maskload_ps(x6 + j, m), a);
    a = _mm256_fmadd_ps(v7, _mm256_maskload_ps(x7 + j, m), a);
    _mm256_maskstore_ps(u + j, m, a);
  }
}
}

namespace optimizer {

void AdamStep(const float* g, float* m, float* v, float* w, unsigned int n,
              float alpha, float a1, float rcp_a2);
}

#endif
