//#ifndef FXCM_H
//#define FXCM_H

#include "model.h"
#include "../fast_math.h"
#include <vector>
#include <memory>

namespace fxcmv1 {
  class Predictor{

public:
  Predictor();
  int p() ;
  void update();
  void predict_half();
  void FreeMemory();
  void FreeRest();
};
}

class FXCM : public Model {
 public:
  FXCM();
  const std::valarray<float>& Predict() const;
  const short* RawPredictions() const;
  const unsigned char* PredictionMask() const;
  unsigned int ActivePredictions() const;
  int FinalPrediction() const;
  float RawPredictionProbability(short raw) const;
  unsigned int NumOutputs();
  void Perceive(int bit);
  void PredictHalf();
  void ByteUpdate() {};
  void FreeMemory();
  void FreeRest();

 private:
  std::unique_ptr<fxcmv1::Predictor> predictor_;
};

//#endif
