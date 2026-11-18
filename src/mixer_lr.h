
#ifndef MIXER_LR_H_
#define MIXER_LR_H_

namespace mixer_lr {

inline int g_layer = 0;

inline float Gain(int layer) {
  return (float)(layer == 1 ? (5000) : (2200)) * 0.001f;
}

}  // namespace mixer_lr

#endif
