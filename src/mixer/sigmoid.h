#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

#include "../fast_math.h"
#include "../simd_scan.h"
#include "../logit_table.h"

class Sigmoid {
 public:
  Sigmoid(int logit_size);

  // Inline in the header as in commit 468e92b9 of github.com/rohitdwivedula/
  // fx2-cmix (Rohit Dwivedula, GPL-3); the table indexed from the float
  // exponent is this entry's own.
  inline float Logit(float p) const {
    const float q = 1.0f - p;
    const float x = (p < q) ? p : q;
    float v = s52e_[logit_table::EIndex(x)];       // v <= 0
    const float d = 0.5f - p;
    uint32_t uv, ud;
    memcpy(&uv, &v, 4); memcpy(&ud, &d, 4);
    uv ^= (ud & 0x80000000u);
    memcpy(&v, &uv, 4);
    return v;
  }
  float LogitBase(float p) const;
  void S52Activate();
  static float Logistic(float p);
  static float FastLogistic(float p);

 private:
  float SlowLogit(float p);
  int logit_size_;
  std::vector<float> logit_table_;
  float s52e_[logit_table::kESize];
};

#endif
