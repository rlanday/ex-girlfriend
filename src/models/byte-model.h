#ifndef BYTE_MODEL_H
#define BYTE_MODEL_H

#include "model.h"

#include <valarray>
#include <vector>

class ByteModel : public Model {
 public:
  // No virtual destructor: PPMD and Bracket live in std::optional members of
  // the Predictor, never deleted through a ByteModel*.
  ByteModel(const std::vector<bool>& vocab);
  const std::valarray<float>& BytePredict();
   std::valarray<float>& Predict() ;
  void Perceive(int bit);
  // the live byte interval [bot_, top_] (read by Predictor::MlPath)
  int Lx1Bot() const { return bot_; }
  int Lx1Top() const { return top_; }
 protected:
  void ByteUpdate();
  int top_, mid_, bot_;
  const std::vector<bool>& vocab_;
  std::valarray<float> probs_;
};

#endif
