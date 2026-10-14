#ifndef MIXER_SELECTORS_H_
#define MIXER_SELECTORS_H_

namespace dictreload {

// Not set in this configuration (stays 0); kept for the reader below.
extern unsigned int mtc_len;
inline unsigned int MtcSel() {
  const unsigned int L = mtc_len;
  if (L == 0) return 0;
  if (L < 4) return 1;
  if (L < 8) return 2;
  if (L < 16) return 3;
  if (L < 32) return 4;
  if (L < 64) return 5;
  if (L < 256) return 6;
  return 7;
}

// Not set in this configuration (stays 0); kept for the reader below.
extern int ord_x, ord_w;
inline unsigned int OrdSel() {
  int a = ord_x; if (a < 0) a = 0; if (a > 3) a = 3;
  return (unsigned int)a * 2u + (ord_w >= 2 ? 1u : 0u);
}

inline unsigned int Q8diff(float d) {
  const float a = d < 0.0f ? -d : d;
  unsigned int m;
  if (a < 0.02f) m = 0;
  else if (a < 0.15f) m = 1;
  else if (a < 0.50f) m = 2;
  else m = 3;
  // monotone: 0 = fxcm much lower ... 3/4 = close ... 7 = fxcm much higher
  return d < 0.0f ? 3u - m : 4u + m;
}

}
#endif
