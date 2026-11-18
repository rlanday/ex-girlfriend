#ifndef FINAL_MIX_H_
#define FINAL_MIX_H_


// Two parts at the end of the mixer cascade.
//
// Bayes term: a Bayes mixture of four experts (transformer, fast-weight head,
// PPMd, fxcm; their bit probabilities from the stretched inputs the mixers
// read) enters the final mixer as one more term, z = z_rows + w st(p_B).
// Posterior: lw_k += ln P_k(y), then v = (1 - alpha) softmax(lw) + alpha / 4,
// alpha = 0.01 (fixed share like in M. Herbster and M. K. Warmuth, "Tracking
// the best expert", ICML 1995).  The final-mixer rows learn on the error of z,
// the weight w with kBayLr per mille of the final mixer's step.
//
// SSE blend: the final probability is sigma(a st(p_sse) + (1 - a) z), a = 0.4,
// instead of the SSE output alone; the SSE tables learn as before.
//
// All floating point of both parts runs in an island translation unit of its
// own (precise, no contraction, no LTO/PGO; makefile MXSDET_*), exp and log
// from the deterministic libm, one copy for encoder and decoder.

namespace final_mix {

// step of the Bayes weight w, per mille of the final mixer's step
constexpr int kBayLr = 8000;
// number of experts of the mixture
constexpr int kExperts = 4;

struct Bay {
  double lw[kExperts] = {0, 0, 0, 0};
  double pk[kExperts] = {0.5, 0.5, 0.5, 0.5};
  float w = 0.0f, s = 0.0f, z = 0.0f;
};
// s4 = stretched TF, FW, PPMd, fxcm of this bit; returns z_rows + w st(p_B).
float BayMix(Bay& b, float zrow, const float* s4);
// Error of the sum (sigma(z) - bit), and the update of w (step lr0 x kBayLr /
// 1000) and of the posterior (fixed share alpha).
float BayLearn(Bay& b, int bit, float lr0);
// sigma(a st(p_sse) + (1 - a) m1raw), a = 0.4.
float SseMix(float p_sse, float m1raw);

}  // namespace final_mix

#endif
