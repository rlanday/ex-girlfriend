
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

static_assert(16000 > 0, "is a factor in per mille");
constexpr unsigned long long kStep1 = 1000000ULL * (16000) / 1000ULL;
constexpr unsigned long long kStep2 = 5000000ULL * (16000) / 1000ULL;
constexpr unsigned long long kStep3 = 25000000ULL * (16000) / 1000ULL;

}  // namespace mixdecay

#endif
