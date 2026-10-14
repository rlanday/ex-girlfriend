#ifndef BIT_HEAD_H_
#define BIT_HEAD_H_

// bit_head: the BITLSTM, the small LSTM that corrects the final probability
// behind the mixer, in the form used by the offline replay trainer.  Name and
// design are from David Freelan's bitlstm32 in cmix-obias (GPL-3.0); ported
// and modified for this entry in 2026.  The form and the AVX2 kernels follow
// that model:
//
//     x -> in_proj -> SiLU -> LSTMCell(C) -> linear(1) -> delta
//     t = ln(p/(1-p));  p' = sigmoid(t + delta)
//
// The compressor applies it to the final probability immediately before the
// arithmetic coder (src/bit_head.cpp of the entry).  This copy is the state of
// that code the shipped weights were trained with, with the build options of
// the development tree resolved for the trainer: the inputs, the geometry, the
// learning step and the initialisation are the ones the coder runs; the
// trainer additionally reads a weight file to start from (warm start), writes
// one at the end, and takes its seed and learning rate from the environment.
//
// Invariants:
//   * The predictor never sees the corrected probability.  The model is a
//     read-only consumer of the predictor state plus the coded bit, so a trace
//     of one compression run holds its inputs for any weights.
//   * Bits whose value the byte mixer forces (0/1 override) are neither
//     corrected nor learned from.
//
// Determinism: one translation unit, -ffp-model=precise -ffp-contract=off,
// noinline entry points, fixed-order reductions (see build.sh).

namespace bit_head {

// The configuration the shipped weights were trained with.  These are build
// options in the development tree; here they are constants.
constexpr int kCells = 16;           // LSTM cells, a multiple of 16
constexpr int kRowWidth = 28;        // width of the stage-1 row the coder passes
constexpr int kDropLo = 7;           // stage-1 columns 7..20 are not fed
constexpr int kDropHi = 20;
constexpr int kDrop2Lo = 27;         // nor is column 27, the bit prediction
constexpr int kDrop2Hi = 27;         // of the byte LSTM
constexpr int kHidSketch = 32;       // sketch width of the transformer state
constexpr int kLhidSketch = 32;      // sketch width of the byte-LSTM state
constexpr int kSeq = 1024;           // TBPTT window and state reset, coded bits
constexpr int kTbptt = 1;            // 1 = buffered truncated BPTT per window
constexpr int kAccum = 16;           // windows per optimiser step
constexpr int kMeanGrad = 1;         // 1 = mean loss over the bits of a step
constexpr int kOpt = 3;              // 0 SGD, 1 momentum, 2 RMSProp, 3 Adam
constexpr int kLr1e6 = 3000;         // learning rate x 1e6 (default)
constexpr int kBeta1_1e6 = 900000;   // Adam beta1 x 1e6
constexpr int kBeta2_1e6 = 999000;   // Adam beta2 x 1e6
constexpr int kEpsExp = 8;           // Adam epsilon = 10^-kEpsExp
constexpr int kClip1000 = 1000;      // global L2 clip of the gradient x 1000
constexpr int kClamp1000 = 8000;     // clamp of |delta| x 1000
constexpr int kXScale = 0;           // 0 = stage-1 row and mixer output raw
constexpr int kFeat = 0;             // feature mask, bit 0 = SSE pair (off)
constexpr int kPrec = 0;             // 0 = float weights, 1 = bf16-rounded

static_assert(kCells > 0 && (kCells % 16) == 0,
              "the cell count must be a positive multiple of 16 "
              "(vector geometry).");
static_assert(kDropLo >= 0 && kDropLo <= kDropHi && kDropHi < kRowWidth &&
                  kDrop2Lo >= 0 && kDrop2Lo <= kDrop2Hi &&
                  kDrop2Hi < kRowWidth,
              "the dropped column ranges must lie inside the stage-1 row.");
static_assert(kDrop2Lo > kDropHi || kDrop2Hi < kDropLo,
              "the two dropped column ranges must be disjoint.");

// Feature mask bits.
enum { kFeatSse = 1 };

// Builds the model once per run (reads the weight file to start from, if
// one is named).
bool Init();

// One call per coded bit, immediately after the coder computed
// base_p = Discretize(Predict()).  stage1/n_stage1 = the stage-1 mixer input
// row, m1raw = the raw output of the final mixer, override_active = byte-mixer
// 0/1 override.  Returns the discretized probability to code with (== base_p
// while the override is active).  Must be followed by exactly one Observe()
// with the bit that was actually coded.
unsigned int Adjust(unsigned int base_p, const float* stage1, int n_stage1,
                    float m1raw, int override_active);

// Learns from the coded bit and advances all causal state.
void Observe(int bit);

// The per-byte channel.  Each vector is registered once, before Init(), and
// read at every Adjust(); the caller rewrites the values once per byte.
//   SetAux   the transformer state (or its sketch, see the trace format)
//   SetTf    the transformer's per-byte scalars (not used by this model)
//   SetLhid  the byte-LSTM state; pre_projected = 1 says the vector already
//            is the fixed sketch, so its first kLhidSketch values are taken
//   SetZone  the gate zone of the byte (not used by this model)
void SetAux(const float* hid, int n_hid);
void SetTf(const float* tf, int n_tf);
void SetLhid(const float* hid, int n_hid, int pre_projected);
void SetZone(const unsigned int* zone);

// Writes the weight file (if one is named) and frees the model.
void Shutdown();

// Geometry of the model.
int Cells();
int NumParams();
int NumInputs();   // live input columns, prev_bit included
int Wlen();        // TBPTT window in coded bits

}  // namespace bit_head

#endif  // BIT_HEAD_H_
