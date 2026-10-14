#ifndef BIT_HEAD_H_
#define BIT_HEAD_H_

// A small LSTM that corrects the final bit probability.
//
// The form and the AVX2 kernels follow the bitlstm32 correction head of
// cmix-obias (David Freelan, GPL-3.0):
//
//     x -> in_proj -> SiLU -> LSTMCell(C) -> linear(1) -> delta
//     t = ln(p/(1-p));  p' = sigmoid(t + delta)
//
// applied to the FINAL probability immediately before the arithmetic coder,
// identically in Encoder::Encode and Decoder::Decode.  The weights either
// start from a fixed deterministic initialisation and are learned online, or
// come from a weight blob compiled into the binary (trained offline on a trace,
// as in the model this follows); loaded weights stay frozen, or keep learning
// online from the first bit or from a given bit on.  Encoder and decoder run
// the identical code on identical inputs and stay in lockstep.
//
// Two properties are contractual:
//   * Predictor::Perceive() never sees the corrected probability.  The head
//     is a pure read-only consumer of the predictor state plus the coded bit.
//   * Bits whose value the byte mixer forces (0/1 override) are neither
//     corrected nor learned from.
//
// Determinism: ONE translation unit, -ffp-model=precise -ffp-contract=off,
// noinline entry points, fixed-order reductions (makefile rule `bitl`).
//
// All parameters are compile-time constants (struct Head in the .cpp file);
// nothing is read from the environment.  Learning: Adam, buffered truncated
// BPTT over windows of 1024 coded bits, 16 windows per optimiser step, mean
// gradient, global norm clip 1.0.  Inputs: columns of the stage-1 mixer row,
// the raw mixer output, three shape features of the final logit, a bit
// position feature, the previous bit and fixed projections of the transformer
// and byte-LSTM states.  The release configuration has 16 cells.
//
// The static_asserts below guard the geometry of these input blocks.
//

#include "vocab_frozen.h"
#include "transformer_gate_ctx.h"

static_assert(32 <= 192,
              "the hidden-state block cannot be wider than the "
              "transformer state vector.");

#define STAGE1_COL(c) (c)

static_assert(0 >= 0 && 0 <= 2,
              "only the correction-model form is "
              "supported.");
static_assert(0 == 0 || 0 > 0,
              "this form has exactly one correction "
              "model.");

static_assert(32 >= 0 && 32 >= 0,
              "the optional input blocks are column counts "
              "and must be >= 0 (0 = block off).");
static_assert(1 == 0 ||
                  (STAGE1_COL(7) >= 0 &&
                   STAGE1_COL(20) < STAGE1_COL(28) &&
                   STAGE1_COL(7) <= STAGE1_COL(20)),
              "the first dropped column range must lie "
              "inside the stage-1 row.");

static_assert(1 == 0 ||
                  (STAGE1_COL(27) >= 0 &&
                   STAGE1_COL(27) < STAGE1_COL(28) &&
                   STAGE1_COL(27) <= STAGE1_COL(27)),
              "the second dropped column range must lie "
              "inside the stage-1 row.");
static_assert(1 == 0 || 1 == 0 ||
                  STAGE1_COL(27) > STAGE1_COL(20) ||
                  STAGE1_COL(27) < STAGE1_COL(7),
              "the two dropped column ranges must be "
              "disjoint, otherwise n1keep would count a column twice.");
static_assert(0 == 0 || 0 == 1,
              "the trace output is not part of this "
              "build.");

// Build guards.  A geometry the head cannot honour stops the build instead of
// disabling the head at run time.
static_assert(16 > 0 && (16 % 16) == 0,
              "the cell count must be a positive multiple "
              "of 16 (vector geometry).");
static_assert(3 >= 0 && 3 <= 3,
              "the optimiser must be 0..3 "
              "(0 SGD, 1 momentum, 2 RMSProp, 3 Adam).");
static_assert(16 >= 1,
              "the number of windows per optimiser step "
              "must be >= 1.");
static_assert(16 == 1 || 1 == 1,
              "accumulating several windows needs the "
              "buffered TBPTT path.");
static_assert(1024 >= 0,
              "the window length must be >= 0 (0 = never reset).");
static_assert(300 > 0,
              "the learning rate (x 1e6) must be "
              "positive.");
static_assert(28 > 0,
              "the stage-1 row width must be positive.  It "
              "has to match the row the coder passes; Adjust() checks that "
              "on the first coded bit.");

namespace bit_head {

// 1 from the bit on which the frozen head resumes learning (console mark).
extern int g_unfrozen;

// Feature mask bits.
enum { kFeatSse = 1 };

// Constructed once per compression/decompression run.
bool Init();

inline constexpr bool Enabled() { return true; }

inline bool AuxOff() { return vocab_frozen::g_aux; }

inline bool MetaOff() { return tf_gate::ctx.zone == 3u; }

// One call per coded bit, immediately after the coder computed
// base_p = Discretize(Predict()).  stage1/n_stage1 = the stage-1 mixer input
// row (layers_[1].Inputs()), m1raw = raw mixer_1_[0].Mix(), override_active =
// byte-mixer 0/1 override.  Returns the discretized probability to code with
// (== base_p while the override is active).  Must be followed by exactly one
// Observe() with the bit that was actually coded.
unsigned int Adjust(unsigned int base_p, const float* stage1, int n_stage1,
                    float m1raw, int override_active);

// Learns from the coded bit and advances all causal state.
void Observe(int bit);

// Register the per-byte state vector the head reads as an extra feature
// block: the transformer's hidden vector (192 values) in a transformer build.
// Call it before Adjust(); passing nullptr (or never calling it) leaves the
// head exactly as it was.
void SetAux(const float* hid, int n_hid);

// The transformer's per-byte scalars (not read in the release
// configuration), in this fixed order
//   [0] gate bit    -- 1 = the coded distribution IS the transformer's own
//                      output, 0 = it fell back to the PPMD prior
//   [1] entropy     -- of the 208-softmax, in nats / ln(208), so [0,1]
//   [2] argmax mass -- the largest normalised probability
//   [3] top-2 gap   -- largest minus second largest
void SetTf(const float* tf, int n_tf);

void SetLhid(const float* hid, int n_hid, int pre_projected);

// The gate zone of the byte now beginning (0 = not gated; not read in the
// release configuration).
// Per BYTE, like everything in this channel.
void SetZone(const unsigned int* zone);

// Frees the head.  Optional.
void Shutdown();

// Geometry of the head.
int Cells();
int NumParams();

}

#endif
