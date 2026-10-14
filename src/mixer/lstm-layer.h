#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "../attributes.h"

#include <valarray>
#include <vector>
#include <stdlib.h>
#include <math.h>

#include "../fast_math.h"
#include "../optimizer.h"
#include "../build_guards.h"

#include "../lstm_aux_contexts.h"
#include "../lstm_row_geometry.h"
#include "../lstm_blockdiag.h"

struct NeuronLayer {
  NeuronLayer(unsigned int input_size, unsigned int num_cells, int horizon,
    int offset) : error_(num_cells), ivar_(horizon), gamma_(1.0, num_cells),
    gamma_u_(num_cells), gamma_m_(num_cells), gamma_v_(num_cells),
    beta_(num_cells), beta_u_(num_cells), beta_m_(num_cells),
    beta_v_(num_cells), weights_(std::valarray<float>(input_size), num_cells),
    state_(std::valarray<float>(num_cells), horizon),
    update_(std::valarray<float>((size_t)0), num_cells),
    m_(std::valarray<float>(input_size), num_cells),
    v_(std::valarray<float>(input_size), num_cells),
    transpose_(std::valarray<float>(num_cells), input_size - offset),
    norm_(std::valarray<float>(num_cells), horizon) {};

  std::valarray<float> error_, ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  std::valarray<std::valarray<float>> weights_, state_, update_, m_, v_,
      transpose_, norm_;
  std::vector<lstmw_t> weights_h_;
  unsigned int wh_stride_ = 0;
  void RefreshHalf() {
    const unsigned int n = (unsigned int)weights_[0].size();
    for (unsigned int i = 0; i < (unsigned int)weights_.size(); ++i) {
      const float* src = &weights_[i][0];
      lstmw_t* dst = &weights_h_[(size_t)i * wh_stride_];
      for (unsigned int j = 0; j < n; ++j) dst[j] = LSTMW_RCVT(src[j]);
    }
  }
  std::vector<float> upd_e_;
  std::vector<float> dx_w_, dx_m_, dx_v_, dx_g_;
  std::vector<unsigned int> dx_touch_;
  std::vector<unsigned char> dx_seen_;
  unsigned int dx_nt_ = 0;
};

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate);
  void ForwardPass(const std::valarray<float>& input, int input_symbol,
      std::valarray<float>* hidden, int hidden_start);
  void BackwardPass(const std::valarray<float>& input, int epoch,
      int layer, int input_symbol, std::valarray<float>* hidden_error);
  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }
  std::vector<std::valarray<std::valarray<float>>*> Weights();

 private:
  std::valarray<float> state_, state_error_, stored_error_;
  std::valarray<std::valarray<float>> tanh_state_, input_gate_state_,
      last_state_;
  float gradient_clip_, learning_rate_;
  static constexpr unsigned int horizon_ = 128;
  static constexpr unsigned int num_cells_ = 80;
  unsigned int nc_div_;
  static constexpr unsigned int input_size_ =
      (unsigned int)208
      + lstm_mixer_inputs::kSlots
      ;
  static constexpr unsigned int output_size_ = (unsigned int)208;
  unsigned int epoch_;
  // Geometry of the block-diagonal state part (form after Xeloraa, see the
  // forward kernel of the recurrence), all of it build time.
  // Coordinates are those of the SLICE (the row minus the one-hot head), i.e.
  // of layer_input_[epoch][layer]:
  //     [ 0, bd_a_ )                    dense auxiliary input
  //     [ bd_a_, bd_a_ + cells )        state of the previous step
  //     [ bd_tail_, n )                 the bias (one column, single layer)
  static constexpr unsigned int bd_w_ =
      (unsigned int)80 / (unsigned int)5;
  static constexpr unsigned int bd_a_ = input_size_;
  static constexpr unsigned int bd_tail_ =
      input_size_ + (unsigned int)80;
  // The forward path reads a 16-ALIGNED SUPERSET of the live columns, so that
  // no HalfDot call ends in a scalar remainder (the auxiliary input is 239
  // columns wide, and 15 scalar bf16 steps per cell and gate would eat the
  // whole saving).  bd_r1_ is the first stretch, rounded up; the columns it
  // adds belong to state block 0 and are exactly zero for every cell whose
  // block is not 0 - and for a cell OF block 0 they are live and correctly
  // read here, which is why BdFwd() then starts the second stretch at bd_r1_.
  static constexpr unsigned int bd_pad_ = (16u - (input_size_ % 16u)) % 16u;
  static constexpr unsigned int bd_r1_ = input_size_ + bd_pad_;
  static_assert(bd_w_ % 16u == 0, "block width must be 16-aligned");
  static_assert(bd_r1_ % 16u == 0,
                "the padded first stretch must be 16-aligned - the "
                "fused kernel steps 16 at a time and has no remainder path");
  static_assert(bd_w_ % (unsigned int)8 == 0,
                "the backward row block must divide the block width");
  static_assert(bd_w_ % (unsigned int)4 == 0,
                "the update row block must divide the block width (a "
                "row block must not straddle two state blocks)");
  static_assert(bd_pad_ < bd_w_, "the alignment padding must stay "
                "inside state block 0");
  // first live state column of cell i, in slice coordinates (update path)
  static constexpr unsigned int BdUpd(unsigned int i) {
    return bd_a_ + (i / bd_w_) * bd_w_;
  }
  // start of the SECOND forward stretch of cell i (see bd_r1_ above)
  static constexpr unsigned int BdFwd(unsigned int i) {
    return BdUpd(i) < bd_r1_ ? bd_r1_ : BdUpd(i);
  }
  unsigned long long update_steps_ = 0;
  std::vector<const float*> upd_x_;
  std::vector<int> upd_sym_;
  std::vector<float> nog_scratch_;
  static constexpr unsigned int upd_epad_ =
      ((unsigned int)80 + 7u) & ~7u;
  static constexpr unsigned int upd_n_ =
      1u + (unsigned int)80 + input_size_;
  void DxAdam(NeuronLayer& neurons);
  unsigned int dx_ex_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;

  void ClipGradients(std::valarray<float>* arr);
  void ForwardPass(NeuronLayer& neurons, const std::valarray<float>& input,
      int input_symbol);
  void BackwardPass(NeuronLayer& neurons, const std::valarray<float>&input,
      int epoch, int layer, int input_symbol,
      std::valarray<float>* hidden_error);
};
#include "lstm-layer.hpp"

#endif
