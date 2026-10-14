
#ifndef FX_MIXER_LR_H_
#define FX_MIXER_LR_H_

// The A branch is compiled only away from 1000; at 1000 the original line
// stays and the ELF is unchanged.

namespace fx_mixer_lr {

constexpr int kA{1000};
static_assert(kA > 0 && kA <= 8000,
              "the factor is per mille (1..8000), e.g. 850 = x0.85");

}

#endif
