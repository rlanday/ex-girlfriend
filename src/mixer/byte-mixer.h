#ifndef BYTE_MIXER_H
#define BYTE_MIXER_H

#include <vector>
#include <memory>

#include "../models/byte-model.h"
#include "../transformer.h"

class Tf1Slot : public ByteModel {
 public:
  explicit Tf1Slot(const std::vector<bool>& vocab) : ByteModel(vocab) {}
  // the transformer's row (one float per byte of the frozen set, in set order)
  void SetProbs(const float* set_probs);
  // defined neutral value for gated rows (uniform over the frozen byte set)
  void SetProbsNeutral();
  void SetProbsOff();
  // 1 = the last setter of this byte was SetProbs (a live transformer row),
  // 0 = neutral or off; read by Predictor::KtxSelect for the transformer slot
  // only (the fast-weight slot is the same class)
  unsigned int KtxLive() const { return ktx_live_; }
 private:
  unsigned int ktx_live_ = 0;
};

#endif
