#include "lstm-layer.h"

#include "../build_guards.h"
#include "../attributes.h"

#include "sigmoid.h"
#include "../fast_math.h"
#include "../page_registry.h"
#define REGISTER_VEC(v) PAGE_REGISTRY_ADD((v).data(), (v).size() * sizeof(float))

#include "../simd_scan.h"
#include "../optimizer.h"

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <numeric>

#define FAST_TANH tanh //fast_tanh
#define FAST_TANH_VEC tanh //fast_tanh_vec
namespace {

inline void AdamRow(const float* g, float* m, float* v, float* w,
    unsigned int n, float learning_rate, float t) {
  const float beta1 = 0.025, beta2 = 0.9999;
  float alpha, a1, a2;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    a1 = (float)(1.0f - pow(beta1, t));
    a2 = (float)(1.0f - pow(beta2, t));
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
    a1 = (float)(1.0f - pow(beta1, UPDATE_LIMIT));
    a2 = (float)(1.0f - pow(beta2, UPDATE_LIMIT));
  }
  optimizer::AdamStep(g, m, v, w, n, alpha, a1, 1.0f / a2);
}

inline void Adam(std::valarray<float>* g, std::valarray<float>* m,
    std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
    float t) {
  const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
  float alpha;
  if (t < UPDATE_LIMIT) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
  }
  {
    const unsigned int n = (unsigned int)g->size();
    if (n) {
      float a1, a2;
      if (t < UPDATE_LIMIT) {
        a1 = (float)(1.0f - pow(beta1, t));
        a2 = (float)(1.0f - pow(beta2, t));
      } else {
        a1 = (float)(1.0f - pow(beta1, UPDATE_LIMIT));
        a2 = (float)(1.0f - pow(beta2, UPDATE_LIMIT));
      }
      optimizer::AdamStep(&(*g)[0], &(*m)[0], &(*v)[0], &(*w)[0], n,
                          alpha, a1, 1.0f / a2);
    }
  }
  (void)eps;
  return;
  (*m) *= beta1;
  (*m) += (1.0f - beta1) * (*g);
  (*v) *= beta2;
  (*v) += (1.0f - beta2) * (*g) * (*g);
  if (t < UPDATE_LIMIT) {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, t))) /
        (sqrt((*v) / (float)(1.0f - pow(beta2, t)) + eps)));
  } else {
    (*w) -= alpha * (((*m) / (float)(1.0f - pow(beta1, UPDATE_LIMIT))) /
        (sqrt((*v) / (float)(1.0f - pow(beta2, UPDATE_LIMIT)) + eps)));
  }
}

}

 inline LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float gradient_clip, float learning_rate) :
    state_(num_cells), state_error_(num_cells), stored_error_(num_cells),
    tanh_state_(std::valarray<float>(num_cells), horizon),
    input_gate_state_(std::valarray<float>(num_cells), horizon),
    last_state_(std::valarray<float>(num_cells), horizon),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    nc_div_(num_cells), epoch_(0),
    forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    input_node_(input_size, num_cells, horizon, output_size_ + input_size_),
    output_gate_(input_size, num_cells, horizon, output_size_ + input_size_) {
  if (auxiliary_input_size != input_size_ || output_size != output_size_) {
    fprintf(stderr, "abort: LstmLayer receives "
            "input %u (compiled in %u), output %u (compiled in %u).  Those "
            "numbers follow the frozen byte set and the mixer slot count; "
            "changing the byte set or the slot count "
            "changes them too.\n",
            auxiliary_input_size, input_size_, output_size, output_size_);
    fflush(stderr);
    abort();
  }
  {
    dx_ex_ = lstm_aux_contexts::Extra();
    const size_t n = (size_t)dx_ex_ * num_cells_;
    NeuronLayer* dg[3] = {&forget_gate_, &input_node_, &output_gate_};
    for (int g = 0; g < 3; ++g) {
      dg[g]->dx_w_.assign(n, 0.0f);
      dg[g]->dx_m_.assign(n, 0.0f);
      dg[g]->dx_v_.assign(n, 0.0f);
      dg[g]->dx_g_.assign(n, 0.0f);
      REGISTER_VEC(dg[g]->dx_w_); REGISTER_VEC(dg[g]->dx_m_);
      REGISTER_VEC(dg[g]->dx_v_); REGISTER_VEC(dg[g]->dx_g_);
      dg[g]->dx_seen_.assign(dx_ex_, 0);
      dg[g]->dx_touch_.assign(dx_ex_, 0u);
      dg[g]->dx_nt_ = 0;
    }
  }
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_.weights_[i].size(); ++j) {
      forget_gate_.weights_[i][j] = low + Rand() * range;
      input_node_.weights_[i][j] = low + Rand() * range;
      output_gate_.weights_[i][j] = low + Rand() * range;
    }
    forget_gate_.weights_[i][forget_gate_.weights_[i].size() - 1] = 1;
  }
  {
    NeuronLayer* dg[3] = {&forget_gate_, &input_node_, &output_gate_};
    const size_t n = (size_t)dx_ex_ * num_cells_;
    for (size_t j = 0; j < n; ++j) {
      dg[0]->dx_w_[j] = low + Rand() * range;
      dg[1]->dx_w_[j] = low + Rand() * range;
      dg[2]->dx_w_[j] = low + Rand() * range;
    }
  }
  {
    static_assert(bd_r1_ + bd_w_ <= upd_n_,
                  "the padded first stretch plus one block must "
                  "fit into the slice");
    const unsigned int rec = output_size_ + input_size_;
    NeuronLayer* bg[3] = {&forget_gate_, &input_node_, &output_gate_};
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const unsigned int b0 = (i / bd_w_) * bd_w_;
      for (unsigned int j = 0; j < num_cells_; ++j) {
        if (j >= b0 && j < b0 + bd_w_) continue;
        for (int g = 0; g < 3; ++g) bg[g]->weights_[i][rec + j] = 0.0f;
      }
    }
  }
  {
    const unsigned int n = (unsigned int)forget_gate_.weights_[0].size();
    const unsigned int stride = (n + 7u) & ~7u;
    NeuronLayer* gates[3] = {&forget_gate_, &input_node_, &output_gate_};
    for (int g = 0; g < 3; ++g) {
      gates[g]->wh_stride_ = stride;
      gates[g]->weights_h_.assign((size_t)num_cells_ * stride, 0);
      gates[g]->RefreshHalf();
    }
  }
  {
    if (((num_cells_ + 7u) & ~7u) != upd_epad_) {
      fprintf(stderr, "abort: upd_epad_ at run time is "
              "%u, compiled in is %u.\n",
              (num_cells_ + 7u) & ~7u, upd_epad_);
      fflush(stderr); abort();
    }
    if (input_size - output_size_ != upd_n_) {
      fprintf(stderr, "build check: upd_n_ is %u at run time, %u compiled in\n",
              input_size - output_size_, upd_n_);
      fflush(stderr); abort();
    }
    upd_x_.assign(horizon_, (const float*)0);
    upd_sym_.assign(horizon_, 0);
    nog_scratch_.assign((size_t)8 * input_size, 0.0f);
    REGISTER_VEC(nog_scratch_);
    NeuronLayer* ugates[3] = {&forget_gate_, &input_node_, &output_gate_};
    for (int g = 0; g < 3; ++g) {
      ugates[g]->upd_e_.assign((size_t)horizon_ * upd_epad_, 0.0f);
      REGISTER_VEC(ugates[g]->upd_e_);
    }
  }
}

inline void LstmLayer::ForwardPass(const std::valarray<float>& input, int input_symbol,
    std::valarray<float>* hidden, int hidden_start) {
  last_state_[epoch_] = state_;
  ForwardPass(forget_gate_, input, input_symbol);
  ForwardPass(input_node_, input, input_symbol);
  ForwardPass(output_gate_, input, input_symbol);
  fastmath::VecLogistic(&forget_gate_.state_[epoch_][0],
                        &forget_gate_.state_[epoch_][0], num_cells_);
  fastmath::VecTanh(&input_node_.state_[epoch_][0],
                    &input_node_.state_[epoch_][0], num_cells_);
  fastmath::VecLogistic(&output_gate_.state_[epoch_][0],
                        &output_gate_.state_[epoch_][0], num_cells_);
  {
    const float* const fg = &forget_gate_.state_[epoch_][0];
    const float* const nd = &input_node_.state_[epoch_][0];
    float* const igs = &input_gate_state_[epoch_][0];
    float* const st = &state_[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      igs[i] = 1.0f - fg[i];
      st[i] = st[i] * fg[i];
      st[i] = st[i] + nd[i] * igs[i];
    }
  }
  fastmath::VecTanh(&tanh_state_[epoch_][0], &state_[0], num_cells_);
  {
    const float* const og = &output_gate_.state_[epoch_][0];
    const float* const ts = &tanh_state_[epoch_][0];
    float* const hid = &(*hidden)[hidden_start];
    for (unsigned int i = 0; i < num_cells_; ++i) hid[i] = og[i] * ts[i];
  }
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void LstmLayer::ForwardPass(NeuronLayer& neurons,
    const std::valarray<float>& input, int input_symbol) {
  {
    const float* x = &input[0];
    const unsigned int n = (unsigned int)input.size();
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const lstmw_t* w = &neurons.weights_h_[(size_t)i * neurons.wh_stride_];
      float f = LSTMW_CVT(w[input_symbol]);
      for (unsigned int b = 0; b < lstm_aux_contexts::kNB; ++b) {
        f += neurons.dx_w_[(size_t)lstm_aux_contexts::cur.col[b] * num_cells_ + i];
      }
      {
        const unsigned int bo = BdFwd(i);
        f += lstm_blockdiag::HalfDot2(x, w + output_size_, bd_r1_,
                                   x + bo, w + output_size_ + bo, bd_w_);
        f += x[n - 1] * LSTMW_CVT(w[output_size_ + n - 1]);
      }
      neurons.norm_[epoch_][i] = f;
    }
  }
  neurons.ivar_[epoch_] = 1.0f / sqrt(((neurons.norm_[epoch_] *
      neurons.norm_[epoch_]).sum() / nc_div_) + 1e-5f);
  neurons.norm_[epoch_] *= neurons.ivar_[epoch_];
  neurons.state_[epoch_] = neurons.norm_[epoch_] * neurons.gamma_ +
      neurons.beta_;
}

inline void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  {
    const unsigned int n = (unsigned int)arr->size();
    float* const a = n ? &(*arr)[0] : (float*)0;
    const float lo = -gradient_clip_, hi = gradient_clip_;
    for (unsigned int i = 0; i < n; ++i) {
      const float x = a[i];
      a[i] = (x < lo) ? lo : ((x > hi) ? hi : x);
    }
  }
}

inline void LstmLayer::BackwardPass(const std::valarray<float>&input, int epoch,
    int layer, int input_symbol, std::valarray<float>* hidden_error) {
  upd_x_[epoch] = &input[0];
  upd_sym_[epoch] = input_symbol;
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
  } else {
    stored_error_ += *hidden_error;
  }

  output_gate_.error_ = tanh_state_[epoch] * stored_error_ *
      output_gate_.state_[epoch] * (1.0f - output_gate_.state_[epoch]);
  state_error_ += stored_error_ * output_gate_.state_[epoch] * (1.0f -
      (tanh_state_[epoch] * tanh_state_[epoch]));
  input_node_.error_ = state_error_ * input_gate_state_[epoch] * (1.0f -
      (input_node_.state_[epoch] * input_node_.state_[epoch]));
  forget_gate_.error_ = (last_state_[epoch] - input_node_.state_[epoch]) *
      state_error_ * forget_gate_.state_[epoch] * input_gate_state_[epoch];

  *hidden_error = 0;
  if (epoch > 0) {
    state_error_ *= forget_gate_.state_[epoch];
    stored_error_ = 0;
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

inline void LstmLayer::BackwardPass(NeuronLayer& neurons,
    const std::valarray<float>&input, int epoch, int layer, int input_symbol,
    std::valarray<float>* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_ = 0;
    neurons.beta_u_ = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      int offset = output_size_ + input_size_;
      const unsigned int b0 = (i / bd_w_) * bd_w_;
      for (unsigned int j = b0; j < b0 + bd_w_; ++j) {
        neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
      }
    }
  }
  neurons.beta_u_ += neurons.error_;
  neurons.gamma_u_ += neurons.error_ * neurons.norm_[epoch];
  neurons.error_ *= neurons.gamma_ * neurons.ivar_[epoch];
  neurons.error_ -= ((neurons.error_ * neurons.norm_[epoch]).sum() /
      nc_div_) * neurons.norm_[epoch];
  {
    const float* ep = &neurons.error_[0];
    const float* rows[8];
    if (layer > 0) {
      for (unsigned int i = 0; i < num_cells_; i += 8) {
        unsigned int rb = num_cells_ - i;
        if (rb > 8) rb = 8;
        for (unsigned int r = 0; r < rb; ++r) {
          rows[r] = &neurons.transpose_[num_cells_ + i + r][0];
        }
        optimizer::BwdRows(ep, rows, &(*hidden_error)[i], num_cells_, rb);
      }
    }
    if (epoch > 0) {
      for (unsigned int i = 0; i < num_cells_; i += 8) {
        unsigned int rb = num_cells_ - i;
        if (rb > 8) rb = 8;
        const unsigned int b0 = (i / bd_w_) * bd_w_;
        for (unsigned int r = 0; r < rb; ++r) {
          rows[r] = &neurons.transpose_[i + r][b0];
        }
        optimizer::BwdRows(ep + b0, rows, &stored_error_[i], bd_w_, rb);
      }
    }
  }
  {
    float* eb = &neurons.upd_e_[(size_t)epoch * upd_epad_];
    const float* er = &neurons.error_[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      eb[i] = er[i];
    }
    for (unsigned int b = 0; b < lstm_aux_contexts::kNB; ++b) {
      const unsigned int c = lstm_aux_contexts::cur.col[b];
      float* gcol = &neurons.dx_g_[(size_t)c * num_cells_];
      for (unsigned int i = 0; i < num_cells_; ++i) gcol[i] += er[i];
      if (!neurons.dx_seen_[c]) {
        neurons.dx_seen_[c] = 1;
        neurons.dx_touch_[neurons.dx_nt_++] = c;
      }
    }
  }
  if (epoch == 0) {
    {
      const unsigned int n = upd_n_;
      const unsigned int off = output_size_;
      const unsigned int rowlen = off + n;
      const unsigned int pad = upd_epad_;
      const int H = (int)horizon_;
      const float* eb = &neurons.upd_e_[0];
      float* const scr = &nog_scratch_[0];
      float* rows[8];
      for (unsigned int i0 = 0; i0 < num_cells_; i0 += 4) {
        unsigned int rb = num_cells_ - i0;
        if (rb > 4) rb = 4;
        for (unsigned int r = 0; r < rb; ++r) {
          float* g = scr + (size_t)r * rowlen;
          for (unsigned int j = 0; j < rowlen; ++j) g[j] = 0.0f;
          rows[r] = g + off;
        }
        for (int e = H - 1; e >= 0; --e) {
          const float* es = eb + (size_t)e * pad + i0;
          const int sym = upd_sym_[e];
          for (unsigned int r = 0; r < rb; ++r) {
            scr[(size_t)r * rowlen + sym] += es[r];
          }
        }
        int e = H - 1;
        for (; e >= 7; e -= 8) {
          const float* xx[8];
          const float* ss[8];
          for (int k = 0; k < 8; ++k) {
            xx[k] = upd_x_[e - k];
            ss[k] = eb + (size_t)(e - k) * pad + i0;
          }
          for (unsigned int r = 0; r < rb; ++r) {
            optimizer::UpdAxpy8(rows[r], xx[0], xx[1], xx[2], xx[3],
                                xx[4], xx[5], xx[6], xx[7],
                                ss[0][r], ss[1][r], ss[2][r], ss[3][r],
                                ss[4][r], ss[5][r], ss[6][r], ss[7][r], n);
          }
        }
        for (; e >= 3; e -= 4) {
          const float* x0 = upd_x_[e];
          const float* x1 = upd_x_[e - 1];
          const float* x2 = upd_x_[e - 2];
          const float* x3 = upd_x_[e - 3];
          const float* s0 = eb + (size_t)e * pad + i0;
          const float* s1 = eb + (size_t)(e - 1) * pad + i0;
          const float* s2 = eb + (size_t)(e - 2) * pad + i0;
          const float* s3 = eb + (size_t)(e - 3) * pad + i0;
          for (unsigned int r = 0; r < rb; ++r) {
            fastmath::UpdAxpy4(rows[r], x0, x1, x2, x3,
                               s0[r], s1[r], s2[r], s3[r], n);
          }
        }
        for (; e >= 0; --e) {
          const float* x = upd_x_[e];
          const float* es = eb + (size_t)e * pad + i0;
          for (unsigned int r = 0; r < rb; ++r) {
            fastmath::UpdAxpy1(rows[r], x, es[r], n);
          }
        }
        for (unsigned int r = 0; r < rb; ++r) {
          AdamRow(scr + (size_t)r * rowlen, &neurons.m_[i0 + r][0],
                  &neurons.v_[i0 + r][0], &neurons.weights_[i0 + r][0],
                  rowlen, learning_rate_, update_steps_);
        }
        for (unsigned int r = 0; r < rb; ++r) {
          float* wr = &neurons.weights_[i0 + r][output_size_ + input_size_];
          float* mr = &neurons.m_[i0 + r][output_size_ + input_size_];
          float* vr = &neurons.v_[i0 + r][output_size_ + input_size_];
          const unsigned int b0 = ((i0 + r) / bd_w_) * bd_w_;
          for (unsigned int j = 0; j < num_cells_; ++j) {
            if (j >= b0 && j < b0 + bd_w_) continue;
            wr[j] = 0.0f; mr[j] = 0.0f; vr[j] = 0.0f;
          }
        }
      }
    }
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
        &neurons.gamma_, learning_rate_, update_steps_);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
        &neurons.beta_, learning_rate_, update_steps_);
    DxAdam(neurons);
    neurons.RefreshHalf();
    return;
    {
      const unsigned int n = upd_n_;
      const unsigned int off = output_size_;
      const unsigned int pad = upd_epad_;
      const int H = (int)horizon_;
      const float* eb = &neurons.upd_e_[0];
      float* rows[8];
      for (unsigned int i0 = 0; i0 < num_cells_; i0 += 4) {
        unsigned int rb = num_cells_ - i0;
        if (rb > 4) rb = 4;
        for (unsigned int r = 0; r < rb; ++r) {
          rows[r] = &neurons.update_[i0 + r][off];
        }
        int e = H - 1;
        for (; e >= 7; e -= 8) {
          const float* xx[8];
          const float* ss[8];
          for (int k = 0; k < 8; ++k) {
            xx[k] = upd_x_[e - k];
            ss[k] = eb + (size_t)(e - k) * pad + i0;
          }
          for (unsigned int r = 0; r < rb; ++r) {
            optimizer::UpdAxpy8(rows[r], xx[0], xx[1], xx[2], xx[3],
                                xx[4], xx[5], xx[6], xx[7],
                                ss[0][r], ss[1][r], ss[2][r], ss[3][r],
                                ss[4][r], ss[5][r], ss[6][r], ss[7][r], n);
          }
        }
        for (; e >= 3; e -= 4) {
          const float* x0 = upd_x_[e];
          const float* x1 = upd_x_[e - 1];
          const float* x2 = upd_x_[e - 2];
          const float* x3 = upd_x_[e - 3];
          const float* s0 = eb + (size_t)e * pad + i0;
          const float* s1 = eb + (size_t)(e - 1) * pad + i0;
          const float* s2 = eb + (size_t)(e - 2) * pad + i0;
          const float* s3 = eb + (size_t)(e - 3) * pad + i0;
          for (unsigned int r = 0; r < rb; ++r) {
            fastmath::UpdAxpy4(rows[r], x0, x1, x2, x3,
                               s0[r], s1[r], s2[r], s3[r], n);
          }
        }
        for (; e >= 0; --e) {
          const float* x = upd_x_[e];
          const float* es = eb + (size_t)e * pad + i0;
          for (unsigned int r = 0; r < rb; ++r) {
            fastmath::UpdAxpy1(rows[r], x, es[r], n);
          }
        }
      }
    }
    for (unsigned int i = 0; i < num_cells_; ++i) {
      Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i],
          &neurons.weights_[i], learning_rate_, update_steps_);
    }
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
        &neurons.gamma_, learning_rate_, update_steps_);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
        &neurons.beta_, learning_rate_, update_steps_);
    DxAdam(neurons);
    neurons.RefreshHalf();
  }
}

inline void LstmLayer::DxAdam(NeuronLayer& neurons) {
  if (neurons.dx_nt_ == 0) return;
  const float beta1 = 0.025f, beta2 = 0.9999f, eps = 1e-6f;
  const float t = (float)update_steps_;
  const float tt = (t < UPDATE_LIMIT) ? t : (float)UPDATE_LIMIT;
  const float alpha = learning_rate_ * 0.1f / sqrt(5e-5f * tt + 1.0f);
  const float c1 = (float)(1.0f - pow(beta1, tt));
  const float c2 = (float)(1.0f - pow(beta2, tt));
  for (unsigned int k = 0; k < neurons.dx_nt_; ++k) {
    const unsigned int c = neurons.dx_touch_[k];
    const size_t o = (size_t)c * num_cells_;
    float* g = &neurons.dx_g_[o];
    float* m = &neurons.dx_m_[o];
    float* v = &neurons.dx_v_[o];
    float* w = &neurons.dx_w_[o];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      m[i] = m[i] * beta1 + (1.0f - beta1) * g[i];
      v[i] = v[i] * beta2 + (1.0f - beta2) * g[i] * g[i];
      w[i] -= alpha * ((m[i] / c1) / (sqrt(v[i] / c2 + eps)));
      g[i] = 0.0f;
    }
    neurons.dx_seen_[c] = 0;
  }
  neurons.dx_nt_ = 0;
}

inline std::vector<std::valarray<std::valarray<float>>*> LstmLayer::Weights() {
  std::vector<std::valarray<std::valarray<float>>*> weights;
  weights.push_back(&forget_gate_.weights_);
  weights.push_back(&input_node_.weights_);
  weights.push_back(&output_gate_.weights_);
  return weights;
}
