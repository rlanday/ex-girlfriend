#include "lstm.h"

#include "../build_guards.h"
#include "../page_registry.h"
#define REGISTER_VEC(v) PAGE_REGISTRY_ADD((v).data(), (v).size() * sizeof(float))

#include "../attributes.h"
#include "../simd_scan.h"

#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

 inline Lstm::Lstm(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip) : input_history_(horizon),
    hidden_(num_cells * num_layers + 1), hidden_error_(num_cells),
    layer_input_(std::valarray<std::valarray<float>>(std::valarray<float>
    (input_size + 1 + num_cells * 2), num_layers), horizon),
    output_(std::valarray<float>(1.0 / output_size, output_size), horizon),
    learning_rate_(learning_rate), epoch_(0) {
  if (num_cells != num_cells_ || num_layers != num_layers_ ||
      (unsigned int)horizon != horizon_ || input_size != input_size_ ||
      output_size != output_size_) {
    fprintf(stderr, "LSTM geometry differs (cells %u/%u, "
            "layers %u/%u, horizon %d/%u, input %u/%u, output %u/%u)\n",
            num_cells, num_cells_, num_layers, num_layers_, horizon, horizon_,
            input_size, input_size_, output_size, output_size_);
    abort();
  }
  lstm_aux_contexts::SetGeom(output_size);
  d_ring_.Reset();
  for (unsigned int b = 0; b < lstm_aux_contexts::kNB; ++b) {
    unsigned int c0 = 0u;
    if (b == lstm_aux_contexts::kIBig) c0 = lstm_aux_contexts::BigCol(0, 0);
    if (b == lstm_aux_contexts::kITri) c0 = lstm_aux_contexts::TriCol(0, 0, 0);
    // Column 0 of the thin block, i.e. the symbol the match state carries
    // before a single byte has been coded (st.sym starts at 0).  A real
    // column of the block, never one of the base one-hot columns.
    if (b == lstm_aux_contexts::kIMatch) c0 = lstm_aux_contexts::MatchCol(0u);
    d_hist_[b].assign(horizon, c0);
    d_old_[b] = c0;
    d_cur_[b] = c0;
    lstm_aux_contexts::cur.col[b] = c0;
  }
  hidden_[hidden_.size() - 1] = 1;
  for (int epoch = 0; epoch < horizon; ++epoch) {
    layer_input_[epoch][0].resize(1 + num_cells + input_size);
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
  }
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.emplace_back(layer_input_[0][i].size() + output_size, input_size_, output_size_,
        num_cells, horizon, gradient_clip, learning_rate);
  }
  size_t mat = (size_t)output_size * nc_stride_;
  nc_storage_.assign(2 * mat + 16, 0.0f);
  REGISTER_VEC(nc_storage_);
  float* base = nc_storage_.data();
  size_t off = ((64u - ((uintptr_t)base & 63u)) & 63u) / sizeof(float);
  nc_w_cur_ = base + off;
  nc_w_snap_ = nc_w_cur_ + mat;           // mat is a multiple of 16 float
  nc_hidden_.assign((size_t)horizon * nc_stride_, 0.0f);
  REGISTER_VEC(nc_hidden_);
  nc_u_.assign((size_t)horizon * nc_stride_, 0.0f);
  if (hidden_.size() != nc_hlen_) {
    fprintf(stderr, "build check: hidden_ has length %u, compiled nc_hlen_=%u\n",
            (unsigned)hidden_.size(), nc_hlen_);
    fflush(stderr);
    abort();
  }
}

inline Lstm::~Lstm() {
  //SaveToDisk("lstm.dat");
}
/*
inline void Lstm::SaveToDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ofstream os(path, std::ios::binary | std::ios::out);
  if (!os.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    os.write(reinterpret_cast<const char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i].Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        os.write(reinterpret_cast<const char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  os.close();
}

inline void Lstm::LoadFromDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ifstream is(path, std::ios::binary | std::ios::in);
  if (!is.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    is.read(reinterpret_cast<char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i].Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        is.read(reinterpret_cast<char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  is.close();
}
*/
inline void Lstm::SetInput(const std::valarray<float>& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::copy(begin(input), begin(input) + input_size_,
        begin(layer_input_[epoch_][i]));
  }
}

inline std::valarray<float>& Lstm::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  {
    const int s1 = (int)input;
    const int s2 = d_ring_.Raw(0);
    d_cur_[lstm_aux_contexts::kIBig] = lstm_aux_contexts::BigCol(s1, s2);
    d_cur_[lstm_aux_contexts::kITri] = lstm_aux_contexts::TriCol(s1, s2, d_ring_.Raw(1));
    (void)s2;
  }
  d_cur_[lstm_aux_contexts::kIMatch] =
      lstm_aux_contexts::MatchCol(lstm_match_channel::st.sym);
  d_ring_.Push(input);
  for (unsigned int b = 0; b < lstm_aux_contexts::kNB; ++b) {
    d_old_[b] = d_hist_[b][last_epoch];
    d_hist_[b][last_epoch] = d_cur_[b];
  }
  if (epoch_ == 0) {
    // Rank-1 reconstruction.  At epoch_==0 the ring holds the last horizon_
    // matrices M_0..M_{H-1} with
    //     M_{e+1} = M_e - lr * err_e (x) h_{e+1},
    // where err_e is exactly the error vector of epoch e (the same formula as
    // in the BPTT below) and h_{e+1} the hidden vector that was used to
    // update back then.  nc_w_snap_ holds M_0 (snapshot from the last
    // epoch_==0).  We walk FORWARD through the chain, repeating exactly the
    // same float operations as the original code - the reconstructed M_e
    // are therefore bit-equal - and store the only quantities the BPTT
    // needs:  u_e[j] = sum_i M_e[i][j] * err_e[i].
    {
      const unsigned int S = nc_stride_, L = nc_hlen_;
      float* w = nc_w_snap_;
      for (unsigned int e = 0; e < horizon_; ++e) {
        float* u = &nc_u_[(size_t)e * S];
        for (unsigned int j = 0; j < L; ++j) u[j] = 0;
        const float* out = &output_[e][0];
        const unsigned int sym = input_history_[e];
        for (unsigned int i = 0; i < output_size_; ++i) {
          float error = (i == sym) ? (out[i] - 1) : out[i];
          const float* row = w + (size_t)i * S;
          for (unsigned int j = 0; j < L; ++j) u[j] += row[j] * error;
        }
        if (e + 1 < horizon_) {
          const float* h = &nc_hidden_[(size_t)(e + 1) * S];
          for (unsigned int i = 0; i < output_size_; ++i) {
            float error = (i == sym) ? (out[i] - 1) : out[i];
            float c = learning_rate_ * error;
            float* row = w + (size_t)i * S;
            for (unsigned int j = 0; j < L; ++j) row[j] -= c * h[j];
          }
        }
      }
    }
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        // hidden_error_ is provably 0 at this point (LstmLayer::
        // BackwardPass sets *hidden_error = 0 and only adds something back
        // for layer > 0; here num_layers == 1).  Hence
        // hidden_error_[j] = u_e[j+offset] is bit-equal to the original loop.
        const float* u = &nc_u_[(size_t)epoch * nc_stride_];
        for (unsigned int j = 0; j < num_cells_; ++j) {
          hidden_error_[j] += u[j + offset];
        }
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        for (unsigned int b = 0; b < lstm_aux_contexts::kNB; ++b) {
          lstm_aux_contexts::cur.col[b] =
              (epoch == 0) ? d_old_[b] : d_hist_[b][prev_epoch];
        }
        layers_[layer].BackwardPass(layer_input_[epoch][layer], epoch, layer,
            input_symbol, &hidden_error_);
      }
    }
  }

  {
    const unsigned int S = nc_stride_, L = nc_hlen_;
    const float* h = &hidden_[0];
    float* hh = &nc_hidden_[(size_t)epoch_ * S];
    for (unsigned int j = 0; j < L; ++j) hh[j] = h[j];
    const float* out = &output_[last_epoch][0];
    for (unsigned int i = 0; i < output_size_; ++i) {
      float error = (i == input) ? (out[i] - 1) : out[i];
      float c = learning_rate_ * error;
      float* row = nc_w_cur_ + (size_t)i * S;
      for (unsigned int j = 0; j < L; ++j) row[j] -= c * h[j];
    }
    if (epoch_ == 0) {
      memcpy(nc_w_snap_, nc_w_cur_,
          sizeof(float) * (size_t)output_size_ * S);
    }
  }
  return Predict(input);
}

inline std::valarray<float>& Lstm::Predict(unsigned int input) {
  for (unsigned int b = 0; b < lstm_aux_contexts::kNB; ++b) {
    lstm_aux_contexts::cur.col[b] = d_cur_[b];
  }
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_, begin(layer_input_[epoch_][i]) +
        input_size_);
    layers_[i].ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
        num_cells_);
    if (i < layers_.size() - 1) {
      auto start2 = begin(layer_input_[epoch_][i + 1]) + num_cells_ +
          input_size_;
      std::copy(start, start + num_cells_, start2);
    }
  }
  const float* const ob = out_bias_;
  for (unsigned int i = 0; i < output_size_; ++i) {
    float sum = 0;
    const float* row = nc_w_cur_ + (size_t)i * nc_stride_;
    for (unsigned int j = 0; j < nc_hlen_; ++j) {
      sum += hidden_[j] * row[j];
    }
    if (ob != nullptr) sum += ob[i];
    output_[epoch_][i] = sum;
  }
  fastmath::VecExp(&output_[epoch_][0], &output_[epoch_][0], output_size_);
  output_[epoch_] /= output_[epoch_].sum();
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  last_input_ = input;
  return output_[epoch];
}

