
#ifndef INVARIANTS_H_
#define INVARIANTS_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace invariants {

// The avalanche step of xxHash32 (Yann Collet, BSD-2-Clause).
inline unsigned int Mix32(unsigned int h) {
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u;
  h ^= h >> 16; return h;
}

// --------------------------------------------------------------- counters --
struct Stat {
  unsigned long long cat_names, cat_bytes, cat_repeat;
  unsigned long long isbn_arm, isbn_fire, isbn_bytes, isbn_ok, isbn_bad;
  unsigned long long head_total, head_dup, head_maskbits;
  unsigned long long tail_bytes, tail_pages, tail_hit;
  unsigned long long bytes;
};
inline Stat stat = {};

struct IsbnState {
  int arm;
  int ndig;
  int sum;
  int gap;          // bytes since the code word without digit/separator
  int expect;       // 0..10 (10 == 'X'), valid only when arm==2

  void Reset() { arm = 0; ndig = 0; sum = 0; gap = 0; expect = 0; }
  void Arm() {
    arm = 1; ndig = 0; sum = 0; gap = 0; expect = 0;
  }
  // One complete byte of the stream.
  void Byte(unsigned int c) {
    if (!arm) return;
    if (c >= '0' && c <= '9') {
      const int d = (int)(c - '0');
      if (arm == 2) {                  // that was the check digit
        Reset();
        return;
      }
      ++ndig; sum += ndig * d; gap = 0;
      if (ndig == 9) {
        arm = 2;
        expect = sum % 11;
      }
      return;
    }
    if (c == 'X' || c == 'x') {
      if (arm == 2) {
        Reset();
        return;
      }
      Reset();
      return;
    }
    if (c == '-' || c == ' ' || c == 0x0c /*ESCAPE*/) {
      if (++gap > 3) Reset();
      return;
    }
    Reset();
  }
  // Expected byte if the rule applies; 0 otherwise.
  inline unsigned int ExpectByte() const {
    if (arm != 2) return 0;
    return (expect == 10) ? (unsigned int)'X' : (unsigned int)('0' + expect);
  }
};
inline IsbnState isbn = {};

// Returns 0 = rule is silent, 1 = rule says bit 0, 2 = rule says bit 1.
// c0 is the partial byte (leading 1 bit), bpos the number of bits set.
// Fires only from bpos>=4 on, once the upper nibble already matches the
// expectation - before that another separator could still follow.
inline unsigned int RuleBit(int c0, int bpos) {
  const unsigned int e = isbn.ExpectByte();
  if (e == 0 || bpos < 4) return 0;
  const int want_hi = (int)(0x100u | e) >> (8 - bpos);   // prefix of e
  if (c0 != want_hi) return 0;
  return (((e >> (7 - bpos)) & 1u) != 0) ? 2u : 1u;
}

inline int tail = 0;      // 1 = article tail section reached

// ------------------------------------------------------- shared outputs --

// fxcmv1::modelPrediction() stores the ISBN rule value here for the bit that
// Predictor::Predict() serves immediately afterwards; the final prediction is
// then pulled towards the expected bit.  Both directions compute the same
// value.
inline int g_isbn_rule = 0;
inline unsigned int ApmExtra() {
  unsigned int v = 0;
  v = (v << 1) | (unsigned int)(tail & 1);
  return v;
}

}

#endif
