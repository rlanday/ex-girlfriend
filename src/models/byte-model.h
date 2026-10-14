#ifndef BYTE_MODEL_H
#define BYTE_MODEL_H

#include "model.h"
#include "../lstm_sharpen.h"
#include "../lstm_fx_link.h"

#include <valarray>
#include <vector>

class ByteModel : public Model {
 public:
  virtual ~ByteModel() {}
  ByteModel(const std::vector<bool>& vocab);
  const std::valarray<float>& BytePredict();
   std::valarray<float>& Predict() ;
  void Perceive(int bit);
  int ex;
  int Lx1Bot() const { return bot_; }
  int Lx1Top() const { return top_; }
 protected:
  void ByteUpdate();
  int top_, mid_, bot_;
  const std::vector<bool>& vocab_;
  std::valarray<float> probs_;
};

#endif
