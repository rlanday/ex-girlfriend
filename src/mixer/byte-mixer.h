#ifndef BYTE_MIXER_H
#define BYTE_MIXER_H

#include <vector>
#include <memory>

#include "../models/byte-model.h"
#include "lstm.h"
#include "../transformer.h"
#include "../lstm_mixer_inputs.h"

class ByteMixer : public ByteModel {
 public:
  ByteMixer(unsigned int num_models, const unsigned int& bit_context,
      const std::vector<bool>& vocab, unsigned int vocab_size, Lstm* lstm);

  const float* LstmHidden(int* n) const {
    if (lstm_ == nullptr) { *n = 0; return nullptr; }
    const std::valarray<float>& v = lstm_->Hidden();
    *n = (int)v.size();
    return &v[0];
  }
  void SetInput(int index, float val);
  void ByteUpdate();
  void SetProbs(const float* set_probs);
  // Value for a masked (gated) transformer row: not the previous row and not
  // the PPMd prior (already its own model in the mixer, it would count
  // twice), but a defined neutral value: the uniform distribution over the
  // frozen byte set.
  void SetProbsNeutral();
  void SetOutputBias(const float* bias256);

 private:
  std::unique_ptr<Lstm> lstm_;
  const unsigned int& byte_;
  std::valarray<int> byte_map_;
  std::valarray<float> inputs_;
  std::valarray<float> folded_bias_;
  unsigned int num_models_, vocab_size_, offset_;
};

class Tf1Slot : public ByteModel {
 public:
  explicit Tf1Slot(const std::vector<bool>& vocab) : ByteModel(vocab) {}
  // the transformer's row (one float per byte of the frozen set, in set order)
  void SetProbs(const float* set_probs);
  // defined neutral value for gated rows, same form as ByteMixer::SetProbsNeutral
  void SetProbsNeutral();
  void SetProbsOff();
};

#endif
