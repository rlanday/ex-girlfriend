#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include "../attributes.h"

#include <valarray>
#include <vector>
#include <memory>
#include <string>

#include "lstm-layer.h"
#include "../fast_math.h"
#include "../optimizer.h"
#include "../build_guards.h"
#include "../ppm_logit_prior.h"

#include "../ds/emhash_set.hpp"
#include "../ds/SmallVector.h"

#include "../lstm_match_channel.h"

#include "../lstm_aux_contexts.h"

#include "../lstm_row_geometry.h"


class Lstm {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip);
  ~Lstm();

  const std::valarray<float>& Hidden() const { return hidden_; }
  std::valarray<float>& Perceive(unsigned int input);
  std::valarray<float>& Predict(unsigned int input);
  void SetInput(const std::valarray<float>& input);
  // Output-logit bias row of the PPMd prior (hook as in cmix-obias, David
  // Freelan): output_size_ values, already folded to the vocabulary
  // positions.  The buffer belongs to the ByteMixer and outlives the Lstm
  // object; only the pointer is held here.
  void SetOutputBias(const float* bias) { out_bias_ = bias; }
  //void SaveToDisk(const std::string& path);
  //void LoadFromDisk(const std::string& path);

 private:
  llvm::SmallVector<LstmLayer, 1> layers_;
  std::vector<uint8_t> input_history_; // horizon
  std::valarray<float> hidden_, hidden_error_;
  std::valarray<std::valarray<std::valarray<float>>> layer_input_;
  std::valarray<std::valarray<float>> output_;
  float learning_rate_;
  const float* out_bias_ = nullptr;
  static constexpr unsigned int num_cells_ = 80;
  static constexpr unsigned int horizon_   = 128;
  static constexpr unsigned int num_layers_ = 1;
  static constexpr unsigned int input_size_ =
      (unsigned int)208
      + lstm_mixer_inputs::kSlots
      ;
  static constexpr unsigned int output_size_ = (unsigned int)208;
  unsigned int epoch_;
  int last_input_ = -1;
  std::vector<uint32_t> d_hist_[lstm_aux_contexts::kNB];   // horizon each
  unsigned int d_old_[lstm_aux_contexts::kNB] = {};        // overwritten this step
  unsigned int d_cur_[lstm_aux_contexts::kNB] = {};        // step being predicted
  lstm_aux_contexts::Ring d_ring_;

  // Instead of a ring of horizon_ full weight matrices (256*171 float per
  // epoch = 22.5 MB) only two matrices plus two narrow rings:
  //   nc_w_cur_   current matrix W (the only one the hot path needs)
  //   nc_w_snap_  snapshot of W at the time of "ring position 0"; inside the
  //               BPTT block it is reconstructed forward from it (in place)
  //   nc_hidden_  ring of the hidden vectors h_e (for the rank-1 deltas)
  //   nc_u_       ring of the products u_e = W_e^T * err_e used in the BPTT
  // Rows are padded to a multiple of 16 float and 64-B aligned.
  std::vector<float> nc_storage_, nc_hidden_, nc_u_;
  float* nc_w_cur_ = nullptr;
  float* nc_w_snap_ = nullptr;
  static constexpr unsigned int nc_hlen_ =
      80 + 1u;
  static constexpr unsigned int nc_stride_ = (nc_hlen_ + 15u) & ~15u;
};
#include "lstm.hpp"

#endif
