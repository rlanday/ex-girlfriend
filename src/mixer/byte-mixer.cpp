#include "byte-mixer.h"
#include "../vocab_frozen.h"
void Tf1Slot::SetProbs(const float* set_probs) {
  for (int i = 0; i < 256; ++i) {
    probs_[i] = (vocab_[i] && transformer::InSet(i))
        ? set_probs[transformer::g_b2i[i]] : 0.0f;
  }
  ktx_live_ = 1u;   // a live transformer row this byte
  ByteModel::ByteUpdate();
}

void Tf1Slot::SetProbsNeutral() {
  const float u = 1.0f / (float)transformer::kSetSize;
  for (int i = 0; i < 256; ++i) {
    probs_[i] = (vocab_[i] && transformer::InSet(i)) ? u : 0.0f;
  }
  ktx_live_ = 0u;   // no transformer row this byte
  ByteModel::ByteUpdate();
}

void Tf1Slot::SetProbsOff() {
  for (int i = 0; i < 256; ++i) probs_[i] = 0.0f;
  ktx_live_ = 0u;   // no transformer row this byte
  ByteModel::ByteUpdate();
}
