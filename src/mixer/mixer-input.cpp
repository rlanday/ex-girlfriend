#include "../meld.h"
#include "mixer-input.h"

#include <cstdio>
#include <cstdlib>

MixerInput::MixerInput(const Sigmoid& sigmoid, float eps) :
    inputs_(0.5, 1), sigmoid_(sigmoid) {
  if (eps != min_ || (1 - eps) != max_) {
    msg_note(3)
;
    fflush(stderr);
    abort();
  }
  const float lo = sigmoid.LogitBase(0), hi = sigmoid.LogitBase(1);
  if (lo != stretched_min_ || hi != stretched_max_) {
    fprintf(stderr, "abort: Sigmoid::LogitBase returns "
            "(%a, %a), compiled in is (%a, %a).  Changing the logit table, "
            "libm or the ISA flags changes these two bounds too.\n",
            (double)lo, (double)hi,
            (double)stretched_min_, (double)stretched_max_);
    fflush(stderr);
    abort();
  }
}

void MixerInput::SetNumModels(int num_models) {
  inputs_.resize(num_models, 0.5);
}
