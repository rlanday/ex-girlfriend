// SSE2 search for a check value in a context-map bucket.
//
// FindChk<A, B>() compares the first A 16-bit check values of a bucket with
// ch in one or two 128-bit compares and returns the SMALLEST index i with
// chk[i] == ch, or -1.  The result is identical to the linear loop it
// replaces; lanes beyond A are masked, so filler bytes cannot produce a hit.
// If nothing is found, the caller runs its unchanged eviction loop.
//
// The idea follows https://github.com/rohitdwivedula/fx2-cmix, commit
// 468e92b9 (Rohit Dwivedula, GPL); the code here is reimplemented.  In the
// release configuration it is used by the ContextMap4 buckets of
// src/models/fxcmv1.cpp.
#ifndef SIMD_SCAN_H_
#define SIMD_SCAN_H_

#include "isa_floor.h"
#include <immintrin.h>
#include <stdint.h>

namespace simdscan {

template <int A, int B>
static inline int FindChk(const uint16_t* chk, uint16_t ch) {
  static_assert(A >= 1 && A <= 16, "SIMD FindChk: A outside 1..16");
  static_assert(A <= 8 ? B >= 16 : B >= 32,
                "SIMD FindChk: union too small for the block compare");
  const __m128i target = _mm_set1_epi16((short)ch);
  const __m128i c0 = _mm_loadu_si128((const __m128i*)chk);
  unsigned m0 = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi16(c0, target));
  if (A < 8) m0 &= (1u << (A * 2)) - 1u;
  if (m0) return (int)(__builtin_ctz(m0) >> 1);
  if (A <= 8) return -1;
  const __m128i c1 = _mm_loadu_si128((const __m128i*)(chk + 8));
  unsigned m1 = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi16(c1, target));
  if (A < 16) m1 &= (1u << ((A - 8) * 2)) - 1u;
  if (m1) return 8 + (int)(__builtin_ctz(m1) >> 1);
  return -1;
}

}

#endif
