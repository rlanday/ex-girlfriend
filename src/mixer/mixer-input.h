#ifndef MIXER_INPUT_H
#define MIXER_INPUT_H

#include "sigmoid.h"

#include "../fast_math.h"
#include "../simd_scan.h"
#include "../logit_table.h"

#include <valarray>
#include <vector>

class MixerInput {
 public:
  MixerInput(const Sigmoid& sigmoid, float eps);
  void SetNumModels(int num_models);
  void SetStretchedInputUnchecked(int index, float p) { inputs_[index] = p; }
  // The four Set* functions below are inline in the header in the form of
  // commit 468e92b9 of github.com/rohitdwivedula/fx2-cmix (Rohit Dwivedula,
  // GPL-3); their bodies are the inherited ones.
  inline void SetInput(int index, float p) {
    if (p < min_) p = min_;
    else if (p > max_) p = max_;
    inputs_[index] = sigmoid_.Logit(p);
  }
  inline void SetStretchedInput(int index, float p) {
    if (p > stretched_max_) p = stretched_max_;
    else if (p < stretched_min_) p = stretched_min_;
    inputs_[index] = p;
  }
  inline void SetZero(int index) {
    inputs_[index] = 0.0f;
  }
  inline void SetExtraInput(size_t index, float p) {
    if (p > stretched_max_) p = stretched_max_;
    else if (p < stretched_min_) p = stretched_min_;
    extra_inputs_[index] = p;
  }
  void SetExtraInputSize(size_t size) { extra_inputs_.resize(size);};
  //void ClearExtraInputs() { extra_inputs_.clear(); }
  const std::valarray<float>& Inputs() const { return inputs_; }
  //const std::vector<float>& ExtraInputs() const { return extra_inputs_; }
  const auto& ExtraInputs() const { return extra_inputs_; }

 private:
  std::valarray<float> inputs_;
  //std::vector<float> extra_inputs_;
  std::valarray<float> extra_inputs_;
  const Sigmoid& sigmoid_;
  static constexpr float min_ = (1.0e-4f);
  static constexpr float max_ = 1.0f - min_;
  static constexpr float stretched_min_ = (-0x1.86983p+3f);
  static constexpr float stretched_max_ = (0x1.868cfep+3f);
};

#endif
