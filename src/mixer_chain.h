#ifndef MIXER_CHAIN_H_
#define MIXER_CHAIN_H_

// ===========================================================================
// Exact end-to-end gradient for the layer-0 mixer cascade.
//
//   Origin: zmix 1.0 (James Byrne), GPL-3.0-or-later, build options
//   -Dmixer-e2e / -Dmixer-e2e-norm.  Sources read: src/mixer/shadow_mix.zig
//   (rule table, soundness argument, backward recursion) and
//   src/mixer/mixer_lex.zig (perceiveErr, seedHead), plus the shipped path in
//   src/predictor_lex.zig.  Mechanism taken; the backward recursion in
//   Predictor::E2EGrad is rewritten in C++ from it.  The documents of the
//   entry credit the source as well.
//
// WHAT IT CHANGES
//   Stock (mixer/mixer.cpp, Mixer::Perceive) trains every layer-0 mixer on its
//   OWN error:  update_i = lr_i * decay * mxlr * (sigma(h_i) - y).  Mixer i is
//   therefore trained as if its raw output h_i were the coded probability --
//   but h_i first runs through the extra-input cascade of the mixers i+1..N-1
//   and then through the layer-1 mixer before anything is coded.  The whole
//   intermediate contribution is ignored by the stock rule.
//
//   The stack is AFFINE in h: raw logits are passed between the stages, and
//   the only nonlinearities are the +-logit(1e-4) clamps of
//   MixerInput::SetExtraInput / SetStretchedInput and the final sigmoid.  So
//   dz/dh_i can be written down exactly, as a backward recursion:
//
//     gamma_i   = gate_i * ( W1[i] + sum_{k>i} gamma_k * u_k[i] )
//     gate_i    = 0 if h_i sits in the clamp, else 1
//     update_i  = lr_i * decay * mxlr * (sigma(z) - y) * gamma_i  [* scale]
//
//   W1 is the layer-1 weight row resolved for THIS bit, u_k the extra-weight
//   row of mixer k (u_k[i] is the weight mixer k puts on clamp(h_i)), and z
//   the raw layer-1 output.  Cost: N*(N-1)/2 + N multiply-adds per coded bit,
//   about 1.9 % of one forward cascade.
//
// WHY THE GRADIENT IS EXACT HERE (the load-bearing property)
//   No mixer output feeds any leaf model.  manager_.auxiliary_context_ and
//   manager_.reload_aux are built from LAYER-0 INPUTS and are written before the
//   cascade runs (predictor.cpp, the auxiliary_average block); the SSE
//   consumes the layer-1 probability and feeds nothing back; the coder
//   consumes p and returns only the data bit.  The layer-1 places beyond the N
//   cascade columns (aux_aux_in, the byte-mixer column, the TF1 slot column,
//   the correction-model column) are leaf values too and contribute nothing
//   to dz/dh_i.
//
// MODES (a build-time choice; the release configuration uses mode 1)
//   0  stock     -- the local rule above, unchanged
//   1  e2e       -- exact gradient, unscaled
//   2  e2enorm   -- exact gradient scaled to RMS 1 over the N mixers, so the
//                   direction becomes exact while the step size stays stock
//                   (zmix ships e2e and e2e-norm together)
//   3  finalerr  -- a control: (sigma(z) - y) for every mixer, no chain
//                   factor; it separates "train on the final error" from
//                   "train on the exact chain factor"
//
// DETERMINISM
//   The reciprocal RMS of mode 2 uses _mm_div_ss/_mm_sqrt_ss, not
//   1.0f/sqrtf(): under -ffast-math the compiler may fold the latter into
//   VRSQRTSS, whose result is a VENDOR-DEFINED estimate and therefore differs
//   between AMD and Intel.  Archives would then no longer be identical across
//   machines, and only on some machines, which is the worst kind.  zmix hit
//   the same trap and guards it
//   in src/strict_fp.zig.  Every other operation of the backward pass is a
//   plain float add or multiply: exact per binary, and identical in encoder
//   and decoder because it is the same function fed the same values.
// ===========================================================================

#include <immintrin.h>
#include "fast_math.h"

#define MIXER_CHAIN_W(x) ((float)(x))

namespace mixer_chain {

// Seed of the layer-1 weight row on its N cascade columns, in per mille of
// 1/N: 1000 = 1/N ("start as the plain average").  Reason (zmix,
// shadow_mix.zig rule table): with an all-zero row gamma_i = W1[i] = 0, so
// the exact gradient starts at zero and the cascade only wakes up once the
// layer-1 row has moved on its own.
constexpr int kSeed = 1000;

// Largest cascade the stack arrays accept.  Every known feature set has 23..31
// layer-0 mixers; Predictor::E2EGrad falls back to the stock rule above this
// bound instead of writing past the array.
static const unsigned int kMaxMixers = 64;

// 1/sqrt(ss/n) from exact IEEE single operations only -- no reciprocal
// estimate, no libm call.  See the DETERMINISM note above.
static inline float RmsScale(float ss, float n) {
  const __m128 mean = _mm_div_ss(_mm_set_ss(ss), _mm_set_ss(n));
  const __m128 rms = _mm_sqrt_ss(mean);
  if (!(_mm_cvtss_f32(rms) > 1e-20f)) return 0.0f;
  return _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), rms));
}

}

#endif
