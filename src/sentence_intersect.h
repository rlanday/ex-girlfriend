#ifndef SENTENCE_INTERSECT_H_
#define SENTENCE_INTERSECT_H_

#include <immintrin.h>

// Sentence similarity of the text model: how many word codes of the current
// sentence also occur in a stored one (AVX2 compare, 8 codes per step).
namespace sentence_intersect {

static inline int Anfang(int laenge, int num) {
  const int k0 = laenge - num;
  return k0 > 0 ? k0 : 0;
}

static inline unsigned int CountShared(const unsigned int* cur, int curcount,
                                       const unsigned int* test, int testcount) {
  unsigned int codesum = 0;
  if (testcount <= 0) return 0;          // empty list: nothing can be shared
  for (int k = 0; k < curcount; ++k) {
    const unsigned int key = cur[k];
    const __m256i vkey = _mm256_set1_epi32((int)key);
    int j = 0;
    bool hit = false;
    for (; j + 8 <= testcount; j += 8) {
      const __m256i v = _mm256_loadu_si256((const __m256i*)(test + j));
      const __m256i eq = _mm256_cmpeq_epi32(v, vkey);
      if (!_mm256_testz_si256(eq, eq)) { hit = true; break; }
    }
    if (!hit) {
      for (; j < testcount; ++j) {
        if (test[j] == key) { hit = true; break; }
      }
    }
    if (hit) ++codesum;
  }
  return codesum;
}

}

#endif
