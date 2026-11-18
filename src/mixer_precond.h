#ifndef MIXER_PRECOND_H_
#define MIXER_PRECOND_H_
// Column preconditioning of the layer-0 weight step.  The layer-0 input row is ONE
// vector for all layer-0 mixers, so its per-column power v_j (EMA of x_j^2
// over kL0ColN trained bits) is kept once, and the weight step of every
// layer-0 row uses x_j * r_j instead of x_j, with
// r_j = sqrt(mean_k v_k) / sqrt(v_j), clamped to [1/cap, cap] (cap 32 in the
// release configuration).  The forward dot product is unchanged; only the
// update is preconditioned per column (diagonal, Adam-like, without a second
// moment per weight).  The factors r_j are recomputed every 64 trained bits,
// v and the scaled row every trained bit.
// Determinism: the arithmetic runs in the island translation unit
// mixer_precond.cpp (-ffp-model=precise -ffp-contract=off, no LTO/PGO), one copy
// shared by encoder and decoder, with a fixed summation order.

namespace mixer_precond {

constexpr int kL0ColN = 16384;   // window of the column powers, trained bits
constexpr int kMaxN0 = 1024;     // largest layer-0 row (the row has 587 + slots)
struct L0Col {
  float v[kMaxN0];      // per-column power (starts at 1)
  float r[kMaxN0];      // per-column factor (starts at 1)
  float xt[kMaxN0];     // scaled input row of this bit
  bool init = false;
  unsigned long long n = 0;
};
// Updates v from the row x (n columns) and returns the scaled row for the step.
const float* L0ColScale(L0Col& c, const float* x, int n);

}  // namespace mixer_precond

#endif
