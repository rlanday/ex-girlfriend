// A small binary context-mixing coder for the build-side information of comp9
// (dictionary hint package, article-order residual).  Counters per hashed
// context, one logistic mixer, 32-bit binary arithmetic coder.  Integer only,
// so every machine computes the same probabilities.  comp9 compiles the
// reading side; the code lives in dict_derive.cpp (one copy).
// The squash table with its interpolation, stretch as its inverse, the
// binary arithmetic coder and the counter rate 1/(n + 1.5) follow lpaq1 by
// Matt Mahoney (GPL); entry 24 of the table is 4024 here, 4022 in lpaq1.

#ifndef HD_CM_H
#define HD_CM_H
#include <cstddef>
#include <cstdint>
#include "../enc_only.h"

namespace hdcm {

const int kIn = 8;                 // context inputs per bit (plus one bias input)
const int kTabBits = 16;           // counter table: 64 K slots, hashed

struct Cm {
  uint32_t t[1 << kTabBits];       // counter: P(1) in 16 bit << 16 | count
  int32_t w[32][kIn + 1];          // mixer weights (16.16), one set per bit kind
  int st[kIn + 1];                 // stretched inputs of the current bit
  uint32_t ix[kIn];                // counter slots of the current bit
  int n, lim, lr;
  uint32_t x1, x2, x;              // arithmetic coder
  uint8_t* p;                      // reader: next input byte; writer: next output byte
  uint8_t* end;
};

// All of it runs in the compression path only: ENC_ONLY (.encoder_only, once in S).
// buf/len: coded stream (reader) or output buffer (writer).  lim: counter
// adaption limit (<= 255), lr: mixer learning rate.
void Init(Cm* c, uint8_t* buf, size_t len, int lim, int lr);
// context of input i of the next bit (any 32-bit hash).
DIR_INLINE void Ctx(Cm* c, int i, uint32_t h) {
  h = (h + (uint32_t)i * 0x9E3779B1u) * 0x85EBCA6Bu;
  c->ix[i] = h >> (32 - kTabBits);
  if (i >= c->n) c->n = i + 1;
}
// codes one bit with mixer weight set ws (< 32); writer: y is the bit,
// reader: y is ignored.  Returns the bit.
int Bit(Cm* c, int y, int ws);
// value v in [0, 2^nb), nb bits MSB first, binary-tree context under base.
int Bits(Cm* c, int v, int nb, uint32_t base);
// rank r >= 0: unary flags r > i for i < kmax (contexts a, b, c and i, weight
// set ws0 + min(i, 7)), then r - kmax in nbesc bits.
int Rank(Cm* c, int r, int kmax, int nbesc, uint32_t a, uint32_t b, uint32_t d, int ws0);
// one decision with fixed probability pr/4096 of a 1 (no model).
int Code(Cm* c, int y, int pr);
// permutation r[0..n) of 0..n-1, each element uniform among the values still
// free (log2 n! bits); writer: r given, reader: r filled.
void Perm(Cm* c, uint32_t* r, uint32_t n);

// context helpers
DIR_INLINE uint32_t K(uint32_t h, uint32_t v) { return (h ^ v) * 16777619u + 0x9E37u; }
DIR_INLINE int Lg(uint64_t x, int cap) { int r = 0; while (x) { ++r; x >>= 1; } return r > cap ? cap : r; }
DIR_INLINE int Q(uint64_t num, uint64_t den, int n) {   // num/den in n steps
  if (!den) return 0;
  uint64_t v = (uint64_t)n * num / den;
  return v > (uint64_t)(n - 1) ? n - 1 : (int)v;
}
DIR_INLINE int R8(int64_t a, int64_t b) {               // 0, or 1 + 8 b / a in 1..8
  if (a <= 0) return 0;
  int64_t v = 8 * b / a;
  return 1 + (v > 7 ? 7 : (int)v);
}

}  // namespace hdcm
#endif  // HD_CM_H
