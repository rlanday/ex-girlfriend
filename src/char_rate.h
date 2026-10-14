#ifndef CHAR_RATE_H_
#define CHAR_RATE_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace char_rate {

// Per node of the partial-byte tree: g_det says whether the frozen alphabet
// forces the next bit (+1 = 1, -1 = 0, 0 = open), g_skew is the share of
// alphabet values on the 1 side in 1/16.  A node is only reached when an
// alphabet value lies in its interval, so both sides are never empty.
inline signed char g_det[256];
inline unsigned char g_skew[256];
inline int g_ready = 0;

inline void Init(const std::vector<bool>& vocab) {
  int n_voc = 0;
  for (int i = 0; i < 256; ++i) if (vocab[i]) ++n_voc;
  for (unsigned node = 1; node < 256; ++node) {
    int j = 0;
    while ((node >> (j + 1)) != 0) ++j;          // j = number of bits in the node
    const unsigned prefix = node - (1u << j);
    const unsigned width = 1u << (8 - j);
    const unsigned bot = prefix * width;
    const unsigned mid = bot + width / 2;        // first value of the 1 side
    int nlo = 0, nhi = 0;
    for (unsigned x = bot; x < mid; ++x) if (vocab[x]) ++nlo;
    for (unsigned x = mid; x < bot + width; ++x) if (vocab[x]) ++nhi;
    g_det[node] = (nlo == 0 && nhi > 0) ? 1 : ((nhi == 0 && nlo > 0) ? -1 : 0);
    g_skew[node] = (nlo + nhi) ? (unsigned char)((16 * nhi + (nlo + nhi) / 2)
                                                 / (nlo + nhi)) : 8;
  }
  g_det[0] = 0; g_skew[0] = 8;
  g_ready = 1;
  (void)n_voc;
}

inline int Forced(unsigned int node) {
  return g_det[node & 255];
}

}

#endif
