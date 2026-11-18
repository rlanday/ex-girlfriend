//#ifndef FXCM_H
//#define FXCM_H

#include "model.h"
#include "../port_cfg.h"
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
  void Predict() const {}
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
#if PORT_CMX_MEAS
  // research CMX_VALTEL (src/cmx_meas.h): the mxA[0..16] outputs of this bit (old) and, to first order, without the
  // mxInputs1 slots [lo[k], hi[k]) (new); read-only, between fxcm's forward and its update
  void CmxValFx(const unsigned* lo, const unsigned* hi, int n, short* oldraw, short* newraw) const;
#endif

 private:
  std::unique_ptr<fxcmv1::Predictor> predictor_;
};

//#endif
