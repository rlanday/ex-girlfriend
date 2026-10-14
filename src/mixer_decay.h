
#ifndef MIXER_DECAY_H_
#define MIXER_DECAY_H_

// Final step of the mixer learning-rate schedule (stock value 0.2).  The
// value 0.1 is taken from fx-deepmix by Halvor Yttredal (GPL-3), mixer/mixer.cpp
// there.
namespace mixdecay {

constexpr int kPromille{100};
static_assert(kPromille > 0 && kPromille <= 1000,
              "is PER MILLE (1..1000), e.g. 100 = 0.1");

constexpr float kFinal = (float)(kPromille) / 1000.0f;

}

#endif
