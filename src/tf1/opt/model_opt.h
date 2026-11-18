// Change in this file: declaration of set_gelesen() ("set read"); additive
// only.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/model_opt.h.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0,
// among them set_gelesen() for rows whose output is not read.
// model_opt.h - TransformerOpt: the optimized fx2 transformer (SPEC section 6
// interface) assembled from the four kernel workstreams:
//   qmat  (src/opt/qmat*.{h,cpp}) - all int4x int8 matmuls, arena format
//   attn  (src/opt/attn.{h,cpp}) - vanilla sliding-window attention + KV
//   kda   (src/opt/kda.{h,cpp}) - fused Kimi linear-attention step
//   glue  (src/opt/glue.{h,cpp}) - norms/quants/rope/residual/head
// Numerics: bit-exact vs the naive src/model.cpp except the documented
// vector-math ulp divergences (attention softmax/PV reassociation ~1e-7 rel,
// KDA exp-family <= ~2 ulp/elem, head tanh/exp <= ~2 ulp) - expected
// whole-model probability drift <= ~1e-5 absolute.
// The vanilla-attention low-prob skip is hardwired OFF (exact softmax; at the
// SPEC-safe threshold 21 it never fires anyway).
#pragma once

#include <cstdint>
#include <memory>

#include "../tf1_config.h"
#include "../tf1_paket.h"

#include "../tf1_config.h"

namespace fx2 {
namespace opt {

using ::fx2::PaketChannels;

// KV-cache variant (attn.h): outputs are bitwise identical; F32V is fastest,
// I8V saves 1.7 MB of L3 footprint (0.56 vs 2.81 MB KV total).
enum class AttnKind : int { KVF32 = 0, KVI8 = 1 };

struct TransformerOptImpl;

struct TransformerOpt {
  explicit TransformerOpt(const char* weights_path,
                          AttnKind attn = AttnKind::KVF32);
  ~TransformerOpt();
  TransformerOpt(const TransformerOpt&) = delete;
  TransformerOpt& operator=(const TransformerOpt&) = delete;

  // resets KV rings, KDA states and conv histories; rope positions for the
  // article are rope_position_offset + local position (0-based)
  void begin_article(int64_t rope_position_offset = 0);

  // feed input token and its prior (205 float16 values); fills probs_out[205]
  // with the fp32 distribution over the NEXT token.
  // With side-channel tensors this overload reuses the channels of the
  // previous step; use the four-argument form then.
  void step(uint8_t token, const uint16_t* prior_f16, float* probs_out);
  // same with an fp32 prior row (205 values)
  void step(uint8_t token, const float* prior205, float* probs_out);
  // The same step with the side channels of this row.  With a container
  // without side-channel tensors the channels are ignored, so a caller may use
  // this entry point unconditionally.
  void step(uint8_t token, const uint16_t* prior_f16, float* probs_out,
            const PaketChannels& channels);
  // true when the loaded container carries the five extra tensors
  bool paket_head() const;
  int embedding_rows() const;
  // Test hook: the assembled token row for one (token, channels) pair (raw
  // embedding row + last_projection + match_projection, then the embedding
  // norm).  No step is run, no state is touched.
  const float* paket_row(uint8_t token, const PaketChannels& channels);

  // gelesen ("read"): applies to the next step() and only to it.  With false
  // the last (window) layer only computes K and V for its ring; probs_out stays
  // untouched and last_logits() keeps the values of the last step with output;
  // last_hidden() does not.  step() resets the flag to true, so a caller that
  // does not set it right before the step gets the full computation.
  void set_gelesen(bool gelesen);

  // the 208 post-softcap logits of the last step (205 in the original; valid
  // until the next step)
  const float* last_logits() const;

  // The normalised residual stream of the last step, the d_model vector the
  // unembedding reads (the transformer's per-byte state for the BITLSTM
  // head).  Valid until the next step.  Observation only; nothing is written
  // and nothing is copied.
  const float* last_hidden() const;
  static int hidden_size();
  const float* last_key() const;

  AttnKind attn_kind() const;

  // fast weights after ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only,
  // github.com/rlanday/ex-girlfriend, commit e923faec), code adapted from its
  // model_opt.cpp, arena_build.cpp, tf1_engine.cpp and predictor.cpp: every read
  // row also gives a second head on x + sum_b dW_b h_b.  fw_update(y): one step
  // of the dW_b on the LAST read row with target y, before the next read row.
  void fw_update(uint8_t target);
  // the V second-head probabilities of the last read row (valid until the
  // next read row)
  const float* fw_probs() const;

 private:
  std::unique_ptr<TransformerOptImpl> impl;
};

// ---- profiling hooks (functional only when model_opt.cpp is compiled with
// -DFX2_PROF; no-ops otherwise). Accumulators are raw-rdtsc ticks. ----
void prof_reset();
bool prof_enabled();
// Print the per-group / per-site table. tsc_to_core = core cycles per tsc
// tick (bench_common.h Calib::ratio()); tokens = steps accumulated since
// prof_reset. Overhead-corrects each section by the measured median
// back-to-back raw-rdtsc delta.
void prof_print(double tsc_to_core, long tokens);

}  // namespace opt
}  // namespace fx2
