#include "../meld.h"
#include "mixer.h"

#include "sigmoid.h"

#include <numeric>
#include <utility>
#include <math.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include "../port_cfg.h"
#if PORT_CMX_MEAS
#include <string.h>
#include <unordered_map>
#include "../cmx_meas.h"   // research CMX_MIXROW / CMX_MIXSKIP / CMX_Z3LR
// the released rows of a re-keyed mixer (CMX_MIXROW 1/2): the same row memory is keyed again first-come
static std::unordered_map<const Mixer*, std::vector<ContextData>>& CmxPool() {
  static std::unordered_map<const Mixer*, std::vector<ContextData>> p;
  return p;
}
#endif
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
#if PORT_CMX_MEAS
  if (inputs_size_ == 26 && cmx_meas::g_semin == 2) {}   // research CMX_SEMIN=2: two more final-mixer columns
  else
  // research CMX_SEMKEYS / CMX_SEMKEY2 (mixers): the n extra caches' layer-0 inputs (563 + n) and final-mixer columns
  // (24 + n, 26 + n with SEMIN=2)
  if (cmx_meas::g_sk2_mix && (inputs_size_ == 563u + (unsigned)cmx_meas::g_nsk ||
                              inputs_size_ == 24u + (unsigned)cmx_meas::g_nsk ||
                              inputs_size_ == 26u + (unsigned)cmx_meas::g_nsk)) {}
  else
#endif
  if (inputs_size_ != 563 &&
      inputs_size_ != 24) {
    msg_note(4)

;
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
#if PORT_CMX_MEAS
  const unsigned long long limit = cmx_meas::g_mix_limit;   // release 10000
#else
  unsigned long long limit = 10000;
#endif
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
    } else
#if PORT_CMX_MEAS
    if (__builtin_expect(cmx_meas::g_mixrow != 0, 0)) {
      data = CmxNewRow();
    } else
#endif
    {
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

#if PORT_CMX_MEAS
// a new row under CMX_MIXROW: a released row of this mixer if there is one (zeroed in mode 1, a copy of the shared
// fallback row in mode 2), else fresh memory (zero; a copy of the fallback row for a layer-0 mixer in mode 3)
__attribute__((noinline)) ContextData* Mixer::CmxNewRow() {
  auto pit = CmxPool().find(this);
  const bool rekeyed = pit != CmxPool().end();
  ContextData* d;
  if (rekeyed && !pit->second.empty()) {
    auto [it, success] = context_map_.insert({context_, pit->second.back()});
    pit->second.pop_back();
    d = &it->second;
    d->cmx_n = 0;     // a re-keyed row is a new row (CMX_ROWDECAY clock)
    d->cmx_ep = 0;
    memset((void*)&d->weights[0], 0, sizeof(mixw_t) * inputs_size_);
    if (extra_inputs_size_) memset((void*)&d->extra_weights[0], 0, sizeof(mixw_t) * extra_inputs_size_);
  } else {
    auto [it, success] = context_map_.insert({context_, ContextData(inputs_size_, extra_inputs_size_)});
    d = &it->second;
  }
  const bool warm = (cmx_meas::g_mixrow == 2 && rekeyed) || (cmx_meas::g_mixrow == 3 && inputs_size_ >= 563);
  if (warm) {
    memcpy((void*)&d->weights[0], (const void*)&context_base_.weights[0], sizeof(mixw_t) * inputs_size_);
    if (extra_inputs_size_)
      memcpy((void*)&d->extra_weights[0], (const void*)&context_base_.extra_weights[0],
             sizeof(mixw_t) * extra_inputs_size_);
  }
  return d;
}
// CMX_MIXROW 1/2 at page 1: every row goes to the pool, the map starts empty (no new memory)
void Mixer::CmxRekey() {
  std::vector<ContextData>& pool = CmxPool()[this];
  for (auto& kv : context_map_) pool.push_back(kv.second);
  context_map_.clear();
  simd_cache_valid_ = false;
  simd_cached_data_ = nullptr;
  primed_data_ = nullptr;
  bank_data_ = nullptr;
}
#endif

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
  if (inputs_size_ == 563) {
    for (unsigned int i = 0; i < 563; ++i) {
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

void Mixer::Perceive(int bit) {
  ContextData* primed = primed_data_;
  primed_data_ = nullptr;

  float decay=mixdecay::kFinal;
  if ( steps_ < mixdecay::kStep3) {
      decay = 0.3f;
      if ( steps_ < mixdecay::kStep2) {
          decay = 0.7f;
          if ( steps_ < mixdecay::kStep1)
              decay = 1.0f;
      }
  }
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_rowdecay != 0, 0) && inputs_size_ >= 563) {   // research CMX_ROWDECAY
    ContextData* rr = primed ? primed : GetContextData();
    if (rr != &context_base_) {   // the shared fallback row keeps the mixer's staircase
      if (rr->cmx_ep != cmx_meas::g_rd_epoch + 1u) {
        // 1: a row born in this restart epoch counts from its birth step, an older row from the restart (= release);
        // 2: the update count, from 0 at the restart
        rr->cmx_n = (cmx_meas::g_rowdecay == 1 && rr->cmx_ep == 0) ? steps_ : 0ull;
        rr->cmx_ep = cmx_meas::g_rd_epoch + 1u;
      }
      const unsigned long long rs = cmx_meas::g_rowdecay == 1 ? steps_ - rr->cmx_n : rr->cmx_n++;
      decay = mixdecay::kFinal;
      if (rs < mixdecay::kStep3) { decay = 0.3f; if (rs < mixdecay::kStep2) { decay = 0.7f; if (rs < mixdecay::kStep1) decay = 1.0f; } }
    }
  }
#endif
  ++steps_;

  float update = learning_rate_
      * (e2e_use_ ? e2e_err_ : (Sigmoid::Logistic(p_) - bit));
  e2e_use_ = false;
  if(fabsf(update)<0.000000000005f && extra_inputs_size_>0) {
      return;
  }
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_skip_on, 0) && cmx_meas::g_region >= cmx_meas::g_skip_from) {   // research CMX_MIXSKIP
    const int ly = inputs_size_ >= 563 ? 0 : 1;
    ++cmx_meas::g_upd[cmx_meas::g_region][ly];
    const float own = fabsf(Sigmoid::Logistic(p_) - (float)bit);
    if (own < (ly == 0 ? cmx_meas::g_skip0 : cmx_meas::g_skip1)) { ++cmx_meas::g_skp[cmx_meas::g_region][ly]; return; }
  }
#endif
   // ++data->steps;
  update = decay * update;
  update *= mxlr_g_;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_z3_on, 0)) update *= cmx_meas::g_z3_mul;   // research CMX_Z3LR
#endif
  ContextData* data = primed ? primed : GetContextData();

  mixw_t* const weights = &data->weights[0];
  const float* const inputs = &inputs_[0];
  if (inputs_size_ == 563) {
    for (unsigned int i = 0; i < 563; ++i) {
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
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)  // one-time table: scalar code is smaller
  for (unsigned int i = 0; i < n; ++i) d->weights[i] = h;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)  // one-time table: scalar code is smaller
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
  if ( steps_ < mixdecay::kStep3) {   // the same thresholds
      decay = 0.3f;
      if ( steps_ < mixdecay::kStep2) {
          decay = 0.7f;
          if ( steps_ < mixdecay::kStep1)
              decay = 1.0f;
      }
  }
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_rowdecay != 0, 0) && inputs_size_ >= 563) {   // research CMX_ROWDECAY
    ContextData* rr = primed ? primed : GetContextData();
    if (rr != &context_base_) {   // the shared fallback row keeps the mixer's staircase
      if (rr->cmx_ep != cmx_meas::g_rd_epoch + 1u) {
        // 1: a row born in this restart epoch counts from its birth step, an older row from the restart (= release);
        // 2: the update count, from 0 at the restart
        rr->cmx_n = (cmx_meas::g_rowdecay == 1 && rr->cmx_ep == 0) ? steps_ : 0ull;
        rr->cmx_ep = cmx_meas::g_rd_epoch + 1u;
      }
      const unsigned long long rs = cmx_meas::g_rowdecay == 1 ? steps_ - rr->cmx_n : rr->cmx_n++;
      decay = mixdecay::kFinal;
      if (rs < mixdecay::kStep3) { decay = 0.3f; if (rs < mixdecay::kStep2) { decay = 0.7f; if (rs < mixdecay::kStep1) decay = 1.0f; } }
    }
  }
#endif
  ++steps_;

  float update = learning_rate_
      * (e2e_use_ ? e2e_err_ : (Sigmoid::Logistic(p_) - bit));
  e2e_use_ = false;
  if(fabsf(update)<0.000000000005f && extra_inputs_size_>0) {
      return false;
  }
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_skip_on, 0) && cmx_meas::g_region >= cmx_meas::g_skip_from) {   // research CMX_MIXSKIP
    const int ly = inputs_size_ >= 563 ? 0 : 1;
    ++cmx_meas::g_upd[cmx_meas::g_region][ly];
    const float own = fabsf(Sigmoid::Logistic(p_) - (float)bit);
    if (own < (ly == 0 ? cmx_meas::g_skip0 : cmx_meas::g_skip1)) { ++cmx_meas::g_skp[cmx_meas::g_region][ly]; return false; }
  }
#endif
  update = decay * update;
  update *= mxlr_g_;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_z3_on, 0)) update *= cmx_meas::g_z3_mul;   // research CMX_Z3LR
#endif
  bank_data_ = primed ? primed : GetContextData();

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

// PerceiveBank verbatim, except that the main step
// of every row reads the preconditioned input row xt instead of the forward
// row (same rows, same update scalars, same extra-row step).
void PerceiveBankX(Mixer* m, unsigned int count, int bit, const float* xt) {
  if (count == 0) return;
  if (count > 64) {   // fallback: the plain per-mixer path
    for (unsigned int i = 0; i < count; ++i) m[i].Perceive(bit);
    return;
  }
  float* rows[64];
  float upd[64];
  unsigned int idx[64];
  unsigned int live = 0;
  for (unsigned int i = 0; i < count; ++i) {
    float u;
    if (!m[i].BankPrepUpdate(bit, &u)) continue;
    rows[live] = m[i].BankWRow();
    upd[live] = u;
    idx[live] = i;
    ++live;
  }
  if (live == 0) return;
  const unsigned int n = m[0].BankN();
  unsigned int b = 0;
  for (; b + mixbank::kBank <= live; b += mixbank::kBank)
    mixbank::TrainRows<mixbank::kBank>(xt, rows + b, n, upd + b);
  for (; b < live; ++b)
    mixbank::TrainRows<1>(xt, rows + b, n, upd + b);
  for (unsigned int k = 0; k < live; ++k) m[idx[k]].BankTrainExtra(upd[k]);
}

