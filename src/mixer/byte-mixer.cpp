#include "byte-mixer.h"
#include "../vocab_frozen.h"

ByteMixer::ByteMixer(unsigned int num_models, const unsigned int& bit_context,
    const std::vector<bool>& vocab, unsigned int vocab_size, Lstm* lstm) :
    ByteModel(vocab), lstm_(lstm), byte_(bit_context), byte_map_(0, 256),
    inputs_(0.0, vocab_size + lstm_mixer_inputs::st.slots),
    folded_bias_(0.0f, vocab_size),
    num_models_(num_models), vocab_size_(vocab_size),
    offset_(0) {
  for (int i = 0; i < 256; ++i) byte_map_[i] = vocab_frozen::kByteMap[i];
  offset_ = 0;
}

void ByteMixer::SetProbs(const float* set_probs) {
  for (int i = 0; i < 256; ++i) {
    probs_[i] = (vocab_[i] && transformer::InSet(i))
        ? set_probs[transformer::kByteToIndex[i]] : 0.0f;
  }
  {
    int fex = 0;
    float fbest = probs_[0];
    for (int i = 1; i < 256; ++i) {
      if (probs_[i] > fbest) { fbest = probs_[i]; fex = i; }
    }
    lstm_sharpen::state.full_ex = fex;
    lstm_sharpen::state.full_best = fbest;
  }
  ByteModel::ByteUpdate();
}

void ByteMixer::SetProbsNeutral() {
  const float u = 1.0f / (float)transformer::kSetSize;
  for (int i = 0; i < 256; ++i) {
    probs_[i] = (vocab_[i] && transformer::InSet(i)) ? u : 0.0f;
  }
  {
    int fex = 0;
    float fbest = probs_[0];
    for (int i = 1; i < 256; ++i) {
      if (probs_[i] > fbest) { fbest = probs_[i]; fex = i; }
    }
    lstm_sharpen::state.full_ex = fex;
    lstm_sharpen::state.full_best = fbest;
  }
  ByteModel::ByteUpdate();
}

void ByteMixer::SetInput(int index, float val) {
  if (!vocab_frozen::InSet(index)) return;
  inputs_[offset_] += val;
  ++offset_;
  if (offset_ == vocab_size_) offset_ = 0;
}

void ByteMixer::SetOutputBias(const float* bias256) {
  for (int i = 0; i < 256; ++i) {
    if (vocab_frozen::InSet(i)) folded_bias_[vocab_frozen::kByteMap[i]] = bias256[i];
  }
  lstm_->SetOutputBias(&folded_bias_[0]);
}

void ByteMixer::ByteUpdate() {
  inputs_ *= 2 / num_models_;
  for (unsigned int k = 0; k < lstm_mixer_inputs::st.slots; ++k) {
    inputs_[vocab_size_ + k] = lstm_mixer_inputs::st.slot[k];
  }
  lstm_match_channel::st.sym =
      lstm_match_channel::st.ok_
          ? (unsigned int)byte_map_[lstm_match_channel::st.byte_]
          : vocab_size_;
  lstm_->SetInput(inputs_);
  inputs_ = 0;
  const auto& output = lstm_->Perceive(byte_map_[byte_]);
  offset_ = 0;
  if (vocab_frozen::g_aux) {
    for (int i = 0; i < 256; ++i) {
      probs_[i] = output[vocab_frozen::kColClamp(i)];
    }
  } else
  for (int i = 0; i < 256; ++i) {
    probs_[i] = vocab_frozen::InSet(i) ? output[vocab_frozen::kByteMap[i]] : 0;
  }
  offset_ = 0;
  {
    int fex = 0;
    float fbest = probs_[0];
    for (int i = 1; i < 256; ++i) {
      if (probs_[i] > fbest) { fbest = probs_[i]; fex = i; }
    }
    lstm_sharpen::state.full_ex = fex;
    lstm_sharpen::state.full_best = fbest;
  }
  ByteModel::ByteUpdate();
}

void Tf1Slot::SetProbs(const float* set_probs) {
  for (int i = 0; i < 256; ++i) {
    probs_[i] = (vocab_[i] && transformer::InSet(i))
        ? set_probs[transformer::kByteToIndex[i]] : 0.0f;
  }
  ByteModel::ByteUpdate();
}

void Tf1Slot::SetProbsNeutral() {
  const float u = 1.0f / (float)transformer::kSetSize;
  for (int i = 0; i < 256; ++i) {
    probs_[i] = (vocab_[i] && transformer::InSet(i)) ? u : 0.0f;
  }
  ByteModel::ByteUpdate();
}

void Tf1Slot::SetProbsOff() {
  for (int i = 0; i < 256; ++i) probs_[i] = 0.0f;
  ByteModel::ByteUpdate();
}
