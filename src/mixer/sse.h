#ifndef SSE_H
#define SSE_H

#include "../char_rate.h"

namespace SSE_sh {
  struct M_T1;
}

class SSE {
 public:
  SSE();
  ~SSE();
  float Predict(float input);
  void Perceive(int bit);
  void PerceiveContextOnly(int bit);

 private:
  SSE_sh::M_T1* sse_;
};

#endif
