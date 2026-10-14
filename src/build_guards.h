#ifndef BUILD_GUARDS_H_
#define BUILD_GUARDS_H_

#include <stdio.h>
namespace build_guards {
inline int g_max_mx1 = 0, g_max_mx2 = 0, g_max_mx4 = 0, g_max_pred = 0;
// The maxima go to stderr at process end, except in the package form.
}

#define BUILD_CELLS16_CHECK(n, wo)                                      \
  static_assert((n) % 16 == 0,                                                \
      "" wo " is not a multiple of 16. "              \
      "16 floats = 64 B = one cache line. "           \
      "Allowed: 80 96 112 128 144 160 176 192 208 224 240 256 272 288 "     \
      "304 320 336.")

BUILD_CELLS16_CHECK(80, "LSTM cell count");


#include <stdio.h>
#include <stdlib.h>
namespace build_guards {

inline void PredTailGuard(short* t, int n, int reserve) {
  const int nr = (n + 15) & -16;
  if (nr - n > reserve) {
    fprintf(stderr,
        "\nbuild check: block loop reads up to %d, %d allocated (n=%d, reserve=%d)\n", nr, n + reserve, n,
        reserve);
    fflush(stderr); abort();
  }
  for (int i = n; i < nr; ++i) {
    if (t[i] != 0) {
      fprintf(stderr,
          "\nbuild check: tail element %d is %d, not 0\n", i,
          (int)t[i]);
      fflush(stderr); abort();
    }
    t[i] = 0;   // zero the tail explicitly instead of relying on it
  }
}
}

#endif
