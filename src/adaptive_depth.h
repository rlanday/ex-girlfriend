#ifndef ADAPTIVE_DEPTH_H_
#define ADAPTIVE_DEPTH_H_

// ---------------------------------------------------------------------------
// ATTRIBUTION. The mechanism implemented here ("adaptive depth": a small online
// logistic model in front of the mixer cascade decides per bit whether the
// cascade and the SSE can be skipped) is taken from the Hutter Prize entry
// zmix 1.0 by James Byrne (GPL-3.0-or-later, submitted 2026-09-14; see its
// build.zig option -Dadaptive-depth and src/predictor_lex.zig lines 660-760,
// 1643-1676, 1864-1889). zmix in turn credits its own instrument "depth4".
// Rewritten in C++ from the Zig source; the feature selection was re-derived for
// this predictor's layer-0 input row.  The entry's documents name the source
// as well.
// ---------------------------------------------------------------------------

// ===========================================================================
// ADAPTIVE DEPTH: a small online
// logistic model AHEAD of the mixer cascade decides, per bit, whether the
// cascade (all layer-0 mixers, the final mixer and the SSE) has to run at all.
//
// ORIGIN: zmix 1.0, `-Dadaptive-depth` (src/predictor_lex.zig there,
// :660-760 and :1643-1676, :1864-1889).  Their figure for their predictor:
// tau = 8.0 skips 42.08 % of all bits for +0.012 % coded length.  What is
// taken over is the MECHANISM, not the code (Zig -> C++), and the feature set
// is translated to our layer-0 input row.
//
// MECHANISM
//   features f  = [1, x[0] (bracket), fxcm final prediction, x[lo..hi-1]] --
//                 that is everything already computed BEFORE the cascade, minus
//                 the 560 individual fxcm columns (zmix likewise takes only the
//                 output of fxcm's own final stage there).
//   z           = w . f, clamped to +-30
//   p_pre       = Logistic(z)
//   SKIP        <=> |z| > tau  (tau = 8.0 by default, set in per mille below)
//   training    AdaGrad, lr 0.01, eps 1e-8, on EVERY coded bit -- including the
//               skipped ones (otherwise the model only ever sees the hard bits).
//
// WHAT HAPPENS ON A SKIPPED BIT
//   skipped:     Mixer::Prime() (and with it the creation/resolution of the
//                weight row), Mix(), Perceive() of all layer-0 mixers and of the
//                final mixer, SSE::Predict()/Perceive(), and the BITLSTM.
//   keeps going: EVERY model (bracket, direct, match, grammar, indirect, PPMd,
//                byte LSTM, transformer) with Predict/Perceive/ByteUpdate, the
//                context manager, the SSE BIT CONTEXT (PerceiveContextOnly --
//                zmix calls it sse.advanceParse), the per-bit cost counters
//                of the other models, and the pre-stage model itself.
//   This is the completeness argument on the time axis: only
//   state that does NOT have to remember anything from the skipped bit is
//   skipped; every state carrier a later bit reads keeps being advanced.  The
//   only carriers that stand still are the mixer weight row and the SSE table --
//   that is exactly WHAT the mechanism is, and it is the same on both coder
//   sides.
//
// WHY THE BITLSTM IS SKIPPED ALONG
//   The BITLSTM reads the STAGE-1 ROW (layers_[1].Inputs()) and the raw final
//   mixer scalar.  On a skipped bit neither exists: the row would still hold the
//   values of the PREVIOUS bit.  Handing it that row is exactly the
//   "plausible but wrong" case -- it would correct a prediction it never saw.
//   So: no Adjust, no Observe, the coder takes p_pre.
//
// DECODER SYMMETRY: z depends exclusively on already coded bytes (all features
// sit in the layer-0 input vector before the bit is resolved).  Coder and
// decoder compute the same number and skip the same bits.
// ===========================================================================

// The pre-stage arithmetic -- z = w . f with a FIXED 8-lane accumulator order
// (AVX2) and the clamp to +-30, the logistic p_pre = 1/(1+exp(-z)), the
// AdaGrad step -- is computed in its own translation unit with strict FP
// semantics (-ffp-model=precise -ffp-contract=off, no LTO, no PGO), like the
// correction model.
// WHY: the gate turns a difference in the last mantissa bit into a model
// decision, so any code generation change that leaves the output untouched
// elsewhere (another PGO profile, another -mtune, another compiler) would
// change the coded output here.  Price: a call per bit instead of inlining.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace adaptive_depth {

static const float kTau = (float)(8000) * 0.001f;   // zmix 1.0: 8.0
static const float kLr  = (float)(1000)  * 0.00001f;   // zmix 1.0 (and depth4): 0.01

struct State {
  float w[64];
  float g[64];
  float f[64];
  int n;                 // features in use (including the bias)
  float p;               // probability of the pre-stage for this bit
  bool skip;             // gate decision for this bit
  unsigned long long bits, skipped, bytes, bytes_all, coded;
  double zabs;
  bool cur_all;

  State() : n(0), p(0.5f), skip(false), bits(0), skipped(0), bytes(0),
            bytes_all(0), coded(0), zabs(0.0), cur_all(true) {
    for (int i = 0; i < 64; ++i) { w[i] = 0.0f; g[i] = 0.0f; f[i] = 0.0f; }
  }

  ~State() {
  }

  inline void Gather(const float* x, unsigned int lo, unsigned int hi,
                     float bracket, float fxfinal) {
    int k = 0;
    f[k++] = 1.0f;
    f[k++] = bracket;
    f[k++] = fxfinal;
    for (unsigned int i = lo; i < hi; ++i) {
      if (k >= 64) {
        fprintf(stderr, "adaptive depth: %u features, capacity %d\n", 3u + (hi - lo), (int)64);
        fflush(stderr);
        exit(4);
      }
      f[k++] = x[i];
    }
    n = k;
  }

  // z = w . f, clamped.  f[0] == 1, so the bias runs in the same loop.
  inline float Z() const {
    float z = 0.0f;
    for (int i = 0; i < n; ++i) z += w[i] * f[i];
    if (z > 30.0f) z = 30.0f;
    else if (z < -30.0f) z = -30.0f;
    return z;
  }

  // AdaGrad step on the bit that was just coded.
  inline void Train(int bit) {
    const float er = p - (float)bit;
    for (int i = 0; i < n; ++i) {
      const float gr = er * f[i];
      g[i] += gr * gr;
      w[i] -= kLr * gr / (sqrtf(g[i]) + 1.0e-8f);
    }
  }
};

inline constexpr bool MetaOff() { return false; }

float ZDet(const State& s);
float LogDet(float z);
float Gate(State& s);          // z AND s.p in one call
void  TrainStep(State& s, int bit);

}

#endif
