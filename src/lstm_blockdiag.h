#ifndef LSTM_BLOCKDIAG_H_
#define LSTM_BLOCKDIAG_H_

#include "fast_math.h"

// Forward kernel of the block-diagonal recurrence of the byte LSTM: each cell
// reads only the previous state of its own block of cells (16 in the release
// configuration); the weights of the other blocks are zero from construction
// on and are set to zero again after every update (src/mixer/lstm-layer.hpp).
// The form follows Xeloraa, github.com/Xeloraa/Hutter-compression-research
// (lstm-layer-bd.cpp); implemented independently here.
namespace lstm_blockdiag {

static inline float HalfDot2(const float* xa, const lstmw_t* wa,
                             unsigned int na,
                             const float* xb, const lstmw_t* wb,
                             unsigned int nb) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  for (unsigned int i = 0; i < na; i += 16) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xa + i),
                         LSTMW_LOAD8(wa + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xa + i + 8),
                         LSTMW_LOAD8(wa + i + 8), a1);
  }
  for (unsigned int i = 0; i < nb; i += 16) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xb + i),
                         LSTMW_LOAD8(wb + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xb + i + 8),
                         LSTMW_LOAD8(wb + i + 8), a1);
  }
  __m256 a = _mm256_add_ps(a0, a1);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
  h = _mm_add_ps(h, _mm_movehl_ps(h, h));
  h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
  return _mm_cvtss_f32(h);
}
}

#endif
