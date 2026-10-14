#ifndef LSTM_SHARPEN_H_
#define LSTM_SHARPEN_H_

namespace lstm_sharpen {

struct State {
  int ex = 0;
  float best = 0.0f;      // its mass
  float second = 0.0f;    // second best mass in the interval
  float denom = 1.0f;     // mass of the interval
  int full_ex = 0;
  float full_best = 0.0f; // its mass (share of 1)
  int full_alive = 0;
  unsigned long long bytes = 0;
};

extern State state;

inline unsigned int Q4(float v) {
  if (v < 0.25f) return 0;
  if (v < 0.50f) return 1;
  if (v < 0.80f) return 2;
  return 3;
}

inline unsigned int Q4gap(float v) {
  if (v < 0.02f) return 0;
  if (v < 0.15f) return 1;
  if (v < 0.50f) return 2;
  return 3;
}

}
#endif
