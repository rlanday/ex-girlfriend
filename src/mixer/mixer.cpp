#include "mixer.h"

#include "sigmoid.h"

#include <numeric>
#include <utility>
#include <math.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
Mixer::Mixer(const std::valarray<float>& inputs,
    const std::valarray<float>& extra_inputs,
    const unsigned long long& context, float learning_rate,
    unsigned int extra_input_size) : inputs_(inputs),
    extra_inputs_vec_(extra_inputs), inputs_size_(inputs.size()),
    extra_inputs_size_(extra_input_size),/*extra_inputs_(extra_input_size),*/ p_(0.5),
    learning_rate_(learning_rate),
    mxlr_g_(mixer_lr::Gain(mixer_lr::g_layer)),
    context_(context), /*max_steps_(1),*/ steps_(0),
    context_base_(inputs.size(), extra_inputs_size_)
    {
  if (inputs_size_ != PORT_N0 &&
      inputs_size_ != PORT_N1) {
    fprintf(stderr,
        "\nbuild check: mixer with %u inputs, compiled N0=%d (layer 0), N1=%d (layer 1)\n",
        (unsigned)inputs_size_, (int)PORT_N0,
        (int)PORT_N1);
    fflush(stderr);
    abort();
  }
  const unsigned int full =
      ((unsigned int)inputs_size_ * (unsigned int)sizeof(mixw_t)
       + 63u) >> 6;
  const unsigned int xfull =
      extra_inputs_size_
      ? (((unsigned int)extra_inputs_size_ * (unsigned int)sizeof(mixw_t)
          + 63u) >> 6)
      : 0u;
  mixw_full_ = (unsigned short)full;
  const unsigned int want = (unsigned int)8;
  mixw_lines_ = (unsigned short)(want < full ? want : full);
  {
    const unsigned int xw = kMixwPfExtra;
    mixw_xlines_ = (unsigned short)(xw < xfull ? xw : xfull);
  }
}

// Keeps the row of the last context, so Mix() and Perceive() look it up once.
// The idea follows commit 468e92b9 of github.com/rohitdwivedula/fx2-cmix
// (Rohit Dwivedula, GPL-3); names and the validity flag are this entry's own.
ContextData* Mixer::GetContextData() {
  if (simd_cache_valid_ && simd_cached_ctx_ == context_)
    return simd_cached_data_;
  ContextData* data;
  unsigned long long limit = 10000;
  auto it = context_map_.find(context_);
  if (context_map_.size() >= limit && it == context_map_.end()) {
    data = &context_base_;
    // data = context_map_[0xDEADBEEF].get();
    // if (data == nullptr) {
    //   context_map_[0xDEADBEEF] = std::unique_ptr<ContextData>(
    //       new ContextData(inputs_.size(), extra_inputs_.size()));
    //   data = context_map_[0xDEADBEEF].get();
    // }
  } else {
    if (it != context_map_.end()) {
      data = &it->second;
    } else {
      //auto [it, success] = context_map_.emplace(std::piecewise_construct, std::make_tuple(context_), std::make_tuple(inputs_.size(), extra_inputs_.size()));
      auto [it, success] = context_map_.insert({context_, ContextData(inputs_size_, extra_inputs_size_)});
      data = &it->second;
    }
  }

  simd_cached_ctx_ = context_;
  simd_cached_data_ = data;
  simd_cache_valid_ = true;
  return data;
}

void Mixer::Prime() {
  primed_data_ = GetContextData();
  const char* const w = (const char*)&primed_data_->weights[0];
  if (__builtin_expect(mixw_lines_ == (unsigned short)8, 1)) {
#pragma unroll
    for (unsigned int i = 0; i < (unsigned int)8; ++i)
      __builtin_prefetch(w + (i << 6), 0, 3);
  } else {
    for (unsigned int i = 0; i < mixw_lines_; ++i)
      __builtin_prefetch(w + (i << 6), 0, 3);
  }
  if (extra_inputs_size_ != 0) {
    const char* const x = (const char*)&primed_data_->extra_weights[0];
    if (__builtin_expect(mixw_xlines_ == (unsigned short)kMixwPfExtra, 1)) {
#pragma unroll
      for (unsigned int i = 0; i < kMixwPfExtra; ++i)
        __builtin_prefetch(x + (i << 6), 0, 3);
    } else {
      for (unsigned int i = 0; i < mixw_xlines_; ++i)
        __builtin_prefetch(x + (i << 6), 0, 3);
    }
  }
}

float Mixer::Mix() {
  ContextData* data = primed_data_ ? primed_data_ : GetContextData();
  const float* const inputs = &inputs_[0];
  const mixw_t* const weights = &data->weights[0];
  float p = 0;
  if (inputs_size_ == PORT_N0) {
    for (unsigned int i = 0; i < PORT_N0; ++i) {
      p += inputs[i] * weights[i];
    }
  } else {
    for (unsigned int i = 0; i < inputs_size_; ++i) {
      p += inputs[i] * weights[i];
    }
  }
  p_ = p;
  // for (unsigned int i = 0; i < extra_inputs_.size(); ++i) {
  //   extra_inputs_[i] = extra_inputs_vec_[i];
  // }
  if (extra_inputs_size_ != 0) {
    const float* const extra_inputs = &extra_inputs_vec_[0];
    const mixw_t* const extra_weights = &data->extra_weights[0];
    float e = 0;
    for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
      e += extra_inputs[i] * extra_weights[i];
    }
    p_ += e;
  }
  return p_;
}

#if PORT_PRL
// PRL (port of Z's h3_prl_patch_20260926, mixer_lex.zig perceive): a row that
// has seen few updates keeps a high step even after the mixer's global step
// schedule has decayed: decay = max(global(steps_), row(n)), row(n) = 1.0 /
// 0.7 / 0.3 / floor at n < 1k / 5k / 25k / else.  n counts the updates that
// reached this row (after the early-out) and is never reset.
static inline void PrlDecay(ContextData* data, float* decay) {
  const uint32_t n = data->n;
  data->n = n + 1u;
  float rd = mixdecay::kFinal;
  if (n < 25000u) {
    rd = 0.3f;
    if (n < 5000u) {
      rd = 0.7f;
      if (n < 1000u) rd = 1.0f;
    }
  }
  if (rd > *decay) *decay = rd;
}
#endif

void Mixer::Perceive(int bit) {
  ContextData* primed = primed_data_;
  primed_data_ = nullptr;

  float decay=mixdecay::kFinal;
  if ( steps_ < 25000000) {
      decay = 0.3f;
      if ( steps_ < 5000000) {
          decay = 0.7f;
          if ( steps_ < 1000000)
              decay = 1.0f;
      }
  }
  ++steps_;

  float update = learning_rate_
      * (e2e_use_ ? e2e_err_ : (Sigmoid::Logistic(p_) - bit));
  e2e_use_ = false;
  if(fabsf(update)<0.000000000005f && extra_inputs_size_>0) {
      return;
  }
   // ++data->steps;
#if PORT_PRL
  ContextData* data = primed ? primed : GetContextData();
  PrlDecay(data, &decay);
  update = decay * update;
  update *= mxlr_g_;
#else
  update = decay * update;
  update *= mxlr_g_;
  ContextData* data = primed ? primed : GetContextData();
#endif

  mixw_t* const weights = &data->weights[0];
  const float* const inputs = &inputs_[0];
  if (inputs_size_ == PORT_N0) {
    for (unsigned int i = 0; i < PORT_N0; ++i) {
      weights[i] -= update * inputs[i];
    }
  } else {
    for (unsigned int i = 0; i < inputs_size_; ++i) {
      weights[i] -= update * inputs[i];
    }
  }
  if (extra_inputs_size_ != 0) {
    mixw_t* const extra_weights = &data->extra_weights[0];
    const float* const extra_inputs = &extra_inputs_vec_[0];
    for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
      extra_weights[i] -= update * extra_inputs[i];
    }
  }
 /*if ((data->steps & 1023) == 0) {
    data->weights *= 1.0f - 3.0e-6f;
    data->extra_weights *= 1.0f - 3.0e-6f;
  }*/

}

const mixw_t* Mixer::E2EWRow() {
  ContextData* d = primed_data_;
  if (d == nullptr) d = bank_data_;
  if (d == nullptr) d = GetContextData();
  return &d->weights[0];
}

const mixw_t* Mixer::E2EXRow() {
  if (extra_inputs_size_ == 0) return nullptr;
  ContextData* d = primed_data_;
  if (d == nullptr) d = bank_data_;
  if (d == nullptr) d = GetContextData();
  return &d->extra_weights[0];
}

// End-to-end gradient (zmix 1.0 by James Byrne, GPL-3.0-or-later): the
// final mixer starts as the plain average of its N cascade columns, because
// an all-zero row makes the exact gradient identically zero (gamma_i = W1[i]
// = 0).  Both the row the current context resolves to and the overflow row
// are seeded.  With a context-keyed final mixer the rows of the other zones
// are created later and start at zero; they catch up within a few bits,
// because the final mixer itself keeps training on the stock rule.
void Mixer::E2ESeedHead(unsigned int head, float val) {
  ContextData* d = GetContextData();
  primed_data_ = nullptr;
  simd_cache_valid_ = false;
  const unsigned int n = (head < inputs_size_) ? head : inputs_size_;
  const mixw_t h = val;
  for (unsigned int i = 0; i < n; ++i) d->weights[i] = h;
  for (unsigned int i = 0; i < n; ++i) context_base_.weights[i] = h;
}

const mixw_t* Mixer::BankRow() {
  bank_data_ = primed_data_ ? primed_data_ : GetContextData();
  return &bank_data_->weights[0];
}

float Mixer::BankFinish(float p_main) {
  p_ = p_main;
  if (extra_inputs_size_ != 0) {
    const float* const extra_inputs = &extra_inputs_vec_[0];
    const mixw_t* const extra_weights = &bank_data_->extra_weights[0];
    float e = 0;
    for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
      e += extra_inputs[i] * extra_weights[i];
    }
    p_ += e;
  }
  return p_;
}

bool Mixer::BankPrepUpdate(int bit, float* update_out) {
  ContextData* primed = primed_data_;
  primed_data_ = nullptr;

  float decay = mixdecay::kFinal;
  if ( steps_ < 25000000) {
      decay = 0.3f;
      if ( steps_ < 5000000) {
          decay = 0.7f;
          if ( steps_ < 1000000)
              decay = 1.0f;
      }
  }
  ++steps_;

  float update = learning_rate_
      * (e2e_use_ ? e2e_err_ : (Sigmoid::Logistic(p_) - bit));
  e2e_use_ = false;
  if(fabsf(update)<0.000000000005f && extra_inputs_size_>0) {
      return false;
  }
#if PORT_PRL
  bank_data_ = primed ? primed : GetContextData();
  PrlDecay(bank_data_, &decay);
  update = decay * update;
  update *= mxlr_g_;
#else
  update = decay * update;
  update *= mxlr_g_;
  bank_data_ = primed ? primed : GetContextData();
#endif

  *update_out = update;
  return true;
}

void Mixer::BankTrainExtra(float update) {
  if (extra_inputs_size_ != 0) {
    mixw_t* const extra_weights = &bank_data_->extra_weights[0];
    const float* const extra_inputs = &extra_inputs_vec_[0];
    for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
      extra_weights[i] -= update * extra_inputs[i];
    }
  }
}

// Mixer bank: the dot products of all bank mixers in one pass over the
// shared input vector, a group of weight rows per load of the input.
void MixBankMain(Mixer* m, unsigned int count, float* out) {
  if (count == 0 || count > 64) return;
  const float* rows[64];
  for (unsigned int i = 0; i < count; ++i) rows[i] = m[i].BankRow();
  const float* const x = m[0].BankInputs();
  const unsigned int n = m[0].BankN();
  unsigned int i = 0;
  for (; i + mixbank::kBank <= count; i += mixbank::kBank)
    mixbank::DotRows<mixbank::kBank>(x, rows + i, n, out + i);
  for (; i < count; ++i)
    mixbank::DotRows<1>(x, rows + i, n, out + i);
}

void PerceiveBank(Mixer* m, unsigned int count, int bit) {
  if (count == 0) return;
  if (count > 64) {   // fallback: the plain per-mixer path
    for (unsigned int i = 0; i < count; ++i) m[i].Perceive(bit);
    return;
  }
  float* rows[64];
  float upd[64];
  unsigned int idx[64];
  unsigned int live = 0;
  // First the update value of every mixer, in the original order (step
  // counter, boost, context row); then the weight rows, a group at a time.
  for (unsigned int i = 0; i < count; ++i) {
    float u;
    if (!m[i].BankPrepUpdate(bit, &u)) continue;
    rows[live] = m[i].BankWRow();
    upd[live] = u;
    idx[live] = i;
    ++live;
  }
  if (live == 0) return;
  const float* const x = m[0].BankInputs();
  const unsigned int n = m[0].BankN();
  unsigned int b = 0;
  for (; b + mixbank::kBank <= live; b += mixbank::kBank)
    mixbank::TrainRows<mixbank::kBank>(x, rows + b, n, upd + b);
  for (; b < live; ++b)
    mixbank::TrainRows<1>(x, rows + b, n, upd + b);
  for (unsigned int k = 0; k < live; ++k) m[idx[k]].BankTrainExtra(upd[k]);
}

