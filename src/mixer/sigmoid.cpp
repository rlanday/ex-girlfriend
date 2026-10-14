#include "sigmoid.h"

#include <math.h>
#include <string.h>

Sigmoid::Sigmoid(int logit_size) : logit_size_(logit_size),
    logit_table_() {
  (void)0;
}

float Sigmoid::LogitBase(float p) const {
  const int n = 100001;
  int index = p * n;
  if (index >= n) index = n - 1;
  else if (index < 0) index = 0;
  const float q = (index + 0.5f) / n;
  return log(q / (1 - q));
}

// Switches to the small logit table indexed from the float exponent: one
// value per bin, taken at the bin centre.  Called at the end of the
// Predictor constructor, after every table derived from the base table.
void Sigmoid::S52Activate() {
  for (int b = 0; b < logit_table::kESize; ++b) {
    if (b == logit_table::kESize - 1) { s52e_[b] = 0.0f; continue; }  // x = 0.5
    unsigned ul = (unsigned)(logit_table::kEBase + b) << 16;
    unsigned uh = (unsigned)(logit_table::kEBase + b + 1) << 16;
    float xl, xh;
    memcpy(&xl, &ul, 4); memcpy(&xh, &uh, 4);
    double x = 0.5 * ((double)xl + (double)xh);   // centre of the bin
    s52e_[b] = (float)log(x / (1.0 - x));
  }
  logit_table_.clear();
  logit_table_.shrink_to_fit();
  logit_size_ = 0;
}

float Sigmoid::Logistic(float p) {
  return 1.0f / (1.0f + expf(-p));
}

float Sigmoid::FastLogistic(float p) {
  return (0.5f * (p / (1.0f + abs(p)) + 1.0f));
}

float Sigmoid::SlowLogit(float p) {
  return log(p / (1 - p));
}
