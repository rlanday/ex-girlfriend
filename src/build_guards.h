#ifndef BUILD_GUARDS_H_
#define BUILD_GUARDS_H_
#include "meld.h"

#include <stdio.h>
namespace build_guards {
inline int g_max_mx1 = 0, g_max_mx2 = 0, g_max_mx4 = 0, g_max_pred = 0;
// The maxima go to stderr at process end, except in the package form.
}  // namespace build_guards

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
      msg_note(107)

;
      fflush(stderr); abort();
    }
    t[i] = 0;   // zero the tail explicitly instead of relying on it
  }
}
}  // namespace build_guards

#endif
