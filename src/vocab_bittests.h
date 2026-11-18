
#ifndef VOCAB_BITTESTS_H_
#define VOCAB_BITTESTS_H_

#include "vocab_frozen.h"

namespace vocab_bittests {

struct BlockedTab { unsigned char v[256]; int n; };

constexpr BlockedTab MakeBlocked() {
  BlockedTab t{};
  t.n = 0;
  for (int i = 0; i < 256; ++i) {
    if (!::vocab_frozen::InSet(i)) { t.v[t.n] = (unsigned char)i; ++t.n; }
  }
  return t;
}

constexpr BlockedTab kBlocked = MakeBlocked();

static_assert(kBlocked.n == 256 - 208,
    "the number of blocked byte values does not match the set size; "
    "bitmap and size disagree.");
// The readers take the values from g_blocked, a
// copy of kBlocked.v[0..n) in .bss filled once before main
// (models/byte-model.cpp); the count stays the compile-time kBlocked.n.
extern unsigned char g_blocked[kBlocked.n] __attribute__((visibility("hidden")));

static_assert(kBlocked.n > 0,
    "the frozen byte set blocks nothing; the fast path "
    "would be pointless.");

}  // namespace vocab_bittests

#endif
