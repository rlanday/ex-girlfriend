#ifndef DENSE_INDEX_H_
#define DENSE_INDEX_H_

#include "vocab_frozen.h"

namespace dense_index {

constexpr unsigned long long kN = (unsigned long long)208 + 1ULL;

struct DigitMap { unsigned char m[256]; };

constexpr DigitMap MakeDigits() {
  DigitMap d = {};
  unsigned int rank = 1;
  for (int i = 0; i < 256; ++i) {
    d.m[i] = vocab_frozen::InSet(i) ? (unsigned char)(rank++) : (unsigned char)0;
  }
  return d;
}

constexpr DigitMap kDigits = MakeDigits();

static_assert(kDigits.m[0] == 0,
    "0x00 is inside the frozen byte set; the "
    "virtual pre-stream bytes would collide with a real byte.");

}

#endif
