#ifndef MIXER_H
#define MIXER_H

#include <vector>
#include <valarray>
#include "../ds/emhash_map.hpp"
#include "../fast_math.h"
#include "../simd_scan.h"
#include "../mixer_lr.h"
#include "../mixer_decay.h"
#include "../char_rate.h"
#include "../mixer_weights.h"
#include "../mixer_bank.h"
#include "../build_guards.h"
#include "../arena_alloc.h"
#include "../mixer_chain.h"

#include "../mixer_lr_restart.h"
#include "../port_cfg.h"
#include <memory>

typedef float mixw_t;

typedef arena::Row<mixw_t> arena_row_t;

struct ContextData {
  ContextData(unsigned long long input_size,
      unsigned long long extra_input_size) : /*steps(0), */weights(input_size),
      extra_weights(extra_input_size) {
  };
  ContextData& operator=(const ContextData&) = default;
  ContextData(const ContextData&) = default;
  ContextData(ContextData&&) = default;
  ContextData& operator=(ContextData&&) = default;
  //unsigned long long steps;
  arena_row_t weights, extra_weights;
#if PORT_PRL
  uint32_t n = 0;   // PRL: updates applied to this row (never reset)
#endif

};

class Mixer {
 public:
  Mixer(const std::valarray<float>& inputs,
      const std::valarray<float>& extra_inputs, const unsigned long long& context,
      float learning_rate, unsigned int extra_input_size);
  float Mix();
  void Perceive(int bit);
  void Prime();
  void PerceiveSkip() { primed_data_ = nullptr; }

  const mixw_t* BankRow();
  float BankFinish(float p_main);
  bool BankPrepUpdate(int bit, float* update_out);
  void BankTrainExtra(float update);
  mixw_t* BankWRow() { return &bank_data_->weights[0]; }
  const float* BankInputs() const { return &inputs_[0]; }
  unsigned int BankN() const { return inputs_size_; }

  float E2EP() const { return p_; }
  const mixw_t* E2EWRow();
  const mixw_t* E2EXRow();
  void E2ESetErr(float e) { e2e_err_ = e; e2e_use_ = true; }
  void E2ESeedHead(unsigned int head, float val);

  unsigned long long MetalrSteps() const { return steps_; }
  void MetalrSteps0() { steps_ = 0; }

 private:
  ContextData* GetContextData();
  float e2e_err_ = 0.0f;
  bool e2e_use_ = false;
  ContextData* primed_data_ = nullptr;
  ContextData* bank_data_ = nullptr;
  const std::valarray<float>& inputs_;
  const std::valarray<float>& extra_inputs_vec_;
  // std::valarray<float> extra_inputs_;
  uint16_t inputs_size_;
  uint16_t extra_inputs_size_;
  float p_, learning_rate_;
  float mxlr_g_;
  const unsigned long long& context_;
  unsigned long long /*max_steps_,*/ steps_;
  emhash6::HashMap<unsigned int, ContextData> context_map_;
  ContextData context_base_;
  ContextData* simd_cached_data_ = nullptr;
  unsigned long long simd_cached_ctx_ = ~0ULL;
  bool simd_cache_valid_ = false;
  unsigned short mixw_lines_ = 0;
  unsigned short mixw_xlines_ = 0;
  unsigned short mixw_full_ = 0;
};

void MixBankMain(Mixer* m, unsigned int count, float* out);
void PerceiveBank(Mixer* m, unsigned int count, int bit);

#endif
