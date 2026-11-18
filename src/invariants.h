
#ifndef INVARIANTS_H_
#define INVARIANTS_H_

// Corpus invariants used by the models: the ISBN-10 check digit (applied to
// the final prediction) and a flag for the article tail section (one extra
// bit of the apmA0 context).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace invariants {

// true if the last n chars written into the ring t (mask + 1 places, cnt
// chars written so far) are s[0..n).  Shared by the census numbers below and
// the computed values (computed_values.h); one copy in the image.
__attribute__((noinline, minsize, pure)) inline bool RingEndet(
    const unsigned char* t, unsigned int mask, unsigned int cnt, const char* s, int n) {
  if (cnt < (unsigned int)n) return false;
  for (int i = 0; i < n; ++i)
    if (t[(cnt - (unsigned int)n + (unsigned int)i) & mask] != (unsigned char)s[i]) return false;
  return true;
}

// The avalanche step of xxHash32 (Yann Collet, BSD-2-Clause).
inline unsigned int Mix32(unsigned int h) {
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u;
  h ^= h >> 16; return h;
}

// --------------------------------------------------------------- counters --
struct Stat {
  unsigned long long cat_names, cat_bytes, cat_repeat;
  unsigned long long head_total, head_dup, head_maskbits;
  unsigned long long tail_bytes, tail_pages, tail_hit;
  unsigned long long bytes;
};
inline Stat stat = {};

struct Isbn2State {
  int cw;                  // the WRT code word "isbn" (2 stream bytes, lead << 8 | final), -1 = none
  int pair;                // the last two stream bytes
  // rule state; with arm == 0 the other fields are dead (arming sets them)
  unsigned char arm;       // 0 off, 1 digits, 2 at P (nine digits seen), 3 'x' behind the flag at P
  unsigned char ndig, gap, sep;   // gap: separators in a row (behind the ninth digit: 0 or 1)

  // The code word index of "isbn" in the dictionary (fxcmv1 cwISBN); called
  // once per model construction, so it is also the reset of every stream.
  // Two-byte band of the code plan (src/preprocess/dictionary.cpp): index
  // 80..3919 is 0xD0+(i-80)/80, 0x80+(i-80)%80; with any other index (none,
  // 1 or 3 bytes) the state never arms.
  // noinline/minsize (also Byte()): kept out of the hot,
  // profiled callers, whose control flow and PGO profile stay unchanged
  // (inlined into Predictor::Perceive(), the state cost about 1.0 KB).
  __attribute__((noinline, minsize)) void SetCodeword(unsigned int idx) {
    arm = 0; pair = 0; cw = -1;   // pair 0: no code word (its lead is >= 0xD0)
    if (idx - 80u < 3840u)
      cw = (int)((0xD0u + (idx - 80u) / 80u) << 8 | (0x80u + (idx - 80u) % 80u));
  }
  __attribute__((always_inline)) inline bool Due() const {   // at P
    return arm == 3 || (arm == 2 && gap == sep);
  }
  // One complete byte of the coder stream.  The code word is recognized as
  // the byte pair (lead, final); a pair inside a longer code word arms in
  // vain (the state clears on the next byte that is no digit/separator).
  __attribute__((noinline, minsize)) void Byte(unsigned int c) {
    const bool due = Due();
    Spur((int)c);
    pair = (pair << 8 | (int)c) & 0xFFFF;
    if (pair == cw) {
      arm = 1; ndig = 0; gap = 0; sep = 0;
      Spur(-1);
      return;
    }
    if (arm == 0) return;
    if (arm == 3) { arm = 0; return; }
    const bool sepc = (c == 0x2Du || c == 0x20u || c == 0x0Cu);  // '-' ' ' escape
    if (arm == 2) {
      if (c == 0x40u && due) arm = 3;
      else if (sepc && sep && !gap) gap = 1;   // the separator in front of P
      else arm = 0;
      return;
    }
    if (c - 0x30u < 10u) {
      gap = 0;
      if (++ndig == 9) arm = 2;
      return;
    }
    // separators; before the first digit also ':' '=' tab (stream bytes, the
    // alphabet is permuted: ':' = 0x4A, '=' = 0x4D)
    if (sepc || (ndig == 0 && (c == 0x4Au || c == 0x4Du || c == 0x09u))) {
      sep |= (ndig != 0);
      if (++gap > 3) arm = 0;
      return;
    }
    arm = 0;
  }
  inline void Spur(int) {}
};
inline Isbn2State isbn2 = {};

struct UmrState {
  unsigned char ph;        // 0 scan, 1 number X, 2 unit text, 3 predicting Y
  unsigned char nd, dx, dot, ul, esc, plen, prevc, prevraw, ex;
  unsigned short alive;    // candidates still consistent with the chars of Y
  unsigned long long m;    // digits of X
  char u[32];              // text between X and '('
  char cand[16][24];       // candidate strings of Y (NUL-terminated)

  __attribute__((noinline, minsize)) void Reset() { ph = 0; esc = 0; prevc = 0; prevraw = 0; ex = 0; }

  // unit text -> conversion type (0 = none).  One pass: a '|' starts the
  // label of a link again ("[square kilometer|km&&]"), blanks, brackets, the
  // nbsp entity "&&!" before the word and the capital flag '@' are skipped;
  // "&&" (entity of the superscript two), a '2' or the two escaped high
  // bytes ("^^", UTF-8) behind the word mark km/mi as square, "^^" with the
  // word "c" is the degree sign of deg C.  The word itself by its hash.
  __attribute__((noinline, minsize)) int Classify() {
    unsigned int h = 0; int n = 0, sq = 0, car = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int i = 0; i < ul; ++i) {
      const int ch = (unsigned char)u[i];
      if (ch == '|') { h = 0; n = 0; sq = 0; car = 0; continue; }
      if (ch == '^') { ++car; continue; }
      if (ch == '&') { if (n) ++sq; continue; }
      if (ch == '2' && n) { sq += 2; continue; }
      if (ch == ' ' || ch == '[' || ch == ']' || ch == '!' || ch == '@') continue;
      h = h * 31u + (unsigned int)ch; ++n;
    }
    if (h == 'c') return car >= 2 ? 9 : 0;
    // hashes (h = h*31 + c) of: km mi mile miles m metre meter metres meters
    // ft feet foot kg lb lbs pound pounds
    static const unsigned int kH[17] = {3426u, 3484u, 3351573u, 103898878u, 109u, 103787791u,
                                        103787401u, 3217421636u, 3217409546u, 3278u, 3138990u,
                                        3148910u, 3420u, 3446u, 106941u, 106857100u, 3312570215u};
    static const char kTyp[] = "34445555566678888";
#pragma clang loop vectorize(disable) interleave(disable)
    for (int k = 0; k < 17; ++k) {
      if (h != kH[k]) continue;
      const int t = kTyp[k] - '0';
      if (sq >= 2) return (t == 3) ? 1 : (t == 4) ? 2 : 0;
      if (car >= 2) return (t == 3) ? 1 : (t == 4) ? 2 : 0;
      return t;
    }
    return 0;
  }

  // Y*10^3 = m*num*10^(3-dx)/den exactly in 64 bit: factors as reduced
  // fractions (km2->mi2 = 5^12/(2^2*3^4*11^2*127^2), 1 mile = 1609.344 m,
  // 1 ft = 0.3048 m, 1 lb = 0.45359237 kg), X at most 10 digits, so m*num
  // < 2^63.  The other candidates follow from V3 = floor(Y*10^3) and the
  // remainder: d < 3 truncation V3/k, rounding (V3+k/2)/k with k = 10^(3-d)
  // (exact, k/2 is an integer); d = 3 rounding V3 + (2r >= den').
  __attribute__((noinline, minsize)) void Start(int t) {
    static const unsigned int kNum[9] = {244140625u, 632321316u, 15625u, 25146u, 1250u, 381u,
                                         100000000u, 45359237u, 9u};
    static const unsigned int kDen[9] = {632321316u, 244140625u, 25146u, 15625u, 381u, 1250u,
                                         45359237u, 100000000u, 5u};
    ph = 0; ex = 0;
    if (nd > 10 || dx > 6) return;
    const unsigned long long mn = m * kNum[t - 1];
    unsigned long long den = kDen[t - 1], q, r;
    static const unsigned int kP10[4] = {1u, 10u, 100u, 1000u};
    if (dx <= 3) {
      const unsigned long long e = kP10[3 - dx];
      const unsigned long long r1 = (mn % den) * e;
      q = (mn / den) * e + r1 / den;
      r = r1 % den;
    } else {
      den *= kP10[dx - 3];
      q = mn / den;
      r = mn % den;
    }
    if (t == 9) q += 32000u;
    const int up3 = 2 * r >= den;
    unsigned long long k = 1000;
    int c = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int d = 0; d < 4; ++d) {
      const unsigned long long tr = q / k;
      const unsigned long long ro = (d < 3) ? (q + k / 2) / k : q + (unsigned long long)up3;
      Format(ro, d, 0, cand[c++]); Format(ro, d, 1, cand[c++]);
      Format(tr, d, 0, cand[c++]); Format(tr, d, 1, cand[c++]);
      k /= 10u;
    }
    alive = 0xFFFFu; plen = 0; ph = 3;
  }
  // V/10^d with d decimals, thousands commas if k; truncated to 23 chars
  __attribute__((noinline, minsize)) static void Format(unsigned long long V, int d, int k, char* o) {
    char g[24]; int n = 0;
    do { g[n++] = (char)('0' + (int)(V % 10u)); V /= 10u; } while (V);
    while (n < d + 1) g[n++] = '0';
    int j = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int i = n - 1; i >= 0 && j < 21; --i) {
      if (i == d - 1) o[j++] = '.';
      o[j++] = g[i];
      if (k && i > d && ((i - d) % 3) == 0) o[j++] = ',';
    }
    o[j] = 0;
  }
  __attribute__((noinline, minsize)) void Next() {
    int e = -1;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int i = 0; i < 16; ++i) {
      if (!((alive >> i) & 1u)) continue;
      const int ch = (unsigned char)cand[i][plen];
      if (ch == 0 || (e >= 0 && e != ch)) { ex = 0; return; }
      e = ch;
    }
    ex = (unsigned char)(e < 0 ? 0 : e);
  }
  // One decoded text char of fxcmv1 (setbuf).
  __attribute__((noinline, minsize)) void Text(unsigned int c) {
    const int ec = esc; esc = 0;
    if (ph == 3) { prevc = (unsigned char)c; return; }
    const bool dig = c - 0x30u < 10u;
    if (ph == 1) {
      if (dig) {
        if (nd < 15) { m = m * 10u + (c - 0x30u); ++nd; if (dot) ++dx; } else ph = 0;
        prevc = (unsigned char)c; return;
      }
      if ((c == ',' || c == '.') && !dot) { if (c == '.') dot = 1; prevc = (unsigned char)c; return; }
      if (prevc == ',' || prevc == '.') ph = 0; else { ph = 2; ul = 0; }
    }
    if (ph == 2) {
      if (ec && ul < 31) u[ul++] = '^';
      if (ec > 1 && ul < 31) u[ul++] = '^';
      if (c == '(') { const int t = Classify(); if (t) Start(t); else ph = 0; prevc = (unsigned char)c; return; }
      if (ul < 31 && c != 10 && (!dig || (prevc >= 'a' && prevc <= 'z'))) { u[ul++] = (char)c; prevc = (unsigned char)c; return; }
      ph = 0;
    }
    if (dig && !(prevc - 0x30u < 10u || prevc == '.' || prevc == ',' || prevc == '/')) {
      ph = 1; m = c - 0x30u; nd = 1; dx = 0; dot = 0;
    }
    prevc = (unsigned char)c;
  }
  // One raw byte of the coder stream (before the text of the byte).
  __attribute__((noinline, minsize)) void Byte(unsigned int c) {
    const unsigned int pr = prevraw; prevraw = (unsigned char)c;
    if (c >= 0x80u && pr == 0x0Cu && esc < 2) ++esc;   // escaped high byte (fxcmv1: c1>127 && c2==ESCAPE)
    if (ph != 3) return;
    if (c - 0x30u < 10u || c == ',' || c == '.') {
      unsigned short a = 0;
#pragma clang loop vectorize(disable) interleave(disable)
      for (int i = 0; i < 16; ++i)
        if (((alive >> i) & 1u) && (unsigned char)cand[i][plen] == c) a |= (unsigned short)(1u << i);
      alive = a; ++plen;
      if (!a || plen >= 23) { ph = 0; ex = 0; return; }
      Next();
    } else { ph = 0; ex = 0; }
  }
  inline unsigned int Rule(int c0, int bpos) {
    if (ph != 3 || !ex) return 0;
    if (c0 != (int)((0x100u | ex) >> (8 - bpos))) return 0;
    return (((ex >> (7 - bpos)) & 1u) != 0) ? 2u : 1u;
  }
};
inline UmrState umr = {};

// Rule value for this bit (0 silent, 1 bit 0, 2 bit 1).  At P: "no digit";
// behind the flag 0x40 at P: the 'x'.  Predictor::Predict() blends it in with
// eps = 2^-5 (both rules).
inline unsigned int RuleBit2(int c0, int bpos) {
  if (!isbn2.Due()) return umr.Rule(c0, bpos);
  if (isbn2.arm == 2) return c0 == 9 ? 1u : 0u;   // node 001: bit 4 is 0 but for a digit
  if (c0 != (int)((0x100u | 0x78u) >> (8 - bpos))) return 0;
  return ((0x78u >> (7 - bpos)) & 1u) ? 2u : 1u;
}

inline int g_injected_byte = 0;   // 1 while Encoder/Decoder::ObserveKnownByte runs

struct CensusVz {
  enum { kCtx = 32, kTail = 32, kPct = 16 };
  char tail[kTail];                 // the last decoded chars (ring)
  unsigned int tc;                  // decoded chars so far
  // number in the text: value without separators, decimals, state
  unsigned long long nv;
  unsigned char innum, dot, dec, nthere, nintro;
  unsigned long long intro, cand, pop;
  unsigned int cand_end;
  unsigned char hintro, hcand, hpop;
  // sentence: 0 none, 1 race (2 decimals), 2 age (1 decimal); percentages
  unsigned char mode, np, bad;
  unsigned int pct[kPct];
  // field of the next stream byte: context (-1 none); pending bit
  int ctx, pctx, pc0;
  unsigned short q[kCtx][256];      // P(bit 1) per context and partial byte
  unsigned char qn[kCtx][256];
  int w[kCtx][2];                   // end mixer, 16.16
  int s1, s2, po;                   // inputs and output of the pending bit
  short st[4096];

  __attribute__((noinline, minsize)) static int Sq(int d) {            // squash of paq8, 12 bit
    static const short kSq[33] = {1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546,
                                  2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4024, 4050, 4068,
                                  4079, 4085, 4089, 4092, 4093, 4094};
    if (d > 2047) d = 2047;
    if (d < -2047) d = -2047;
    const int k = d & 127, j = (d >> 7) + 16;
    return (kSq[j] * (128 - k) + kSq[j + 1] * k + 64) >> 7;
  }
  __attribute__((noinline, minsize)) void Reset() {
    memset(this, 0, sizeof(*this));
    ctx = -1; pctx = -1;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int i = 0; i < kCtx; ++i) {
      w[i][0] = 65536;
      for (int j = 0; j < 256; ++j) q[i][j] = 32768;
    }
    int pi = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int x = -2047; x <= 2047; ++x) {
      const int v = Sq(x);
      for (int i = pi; i <= v; ++i) st[i] = (short)x;
      pi = v + 1;
    }
    for (int i = pi; i < 4096; ++i) st[i] = 2047;
  }
  // Before kTail chars the ring holds zeros, which no literal contains, so
  // the count check of RingEndet changes no answer.
  __attribute__((noinline, minsize)) bool Endet(const char* s, int n) const {
    static_assert((kTail & (kTail - 1)) == 0, "ring size is a power of two");
    return RingEndet((const unsigned char*)tail, kTail - 1, tc, s, n);
  }
  // One decoded char of fxcm (setbuf).
  __attribute__((noinline, minsize)) void Text(unsigned int c) {
    if (g_injected_byte) return;
    if (c - 0x30u < 10u && !innum) {   // a number starts: what stands in front of it
      innum = 1; nv = 0; dot = 0; dec = 0;
      nthere = Endet("there are ", 10) || Endet("there were ", 11);
      nintro = Endet("total population of ", 20);
    }
    tail[tc % kTail] = (char)c;
    ++tc;
    if (c - 0x30u < 10u) {
      if (nv < 100000000000ull) nv = nv * 10u + (c - 0x30u);
      dec += dot;
      return;
    }
    if (innum && c == ',' && !dot) return;
    if (innum && c == '.' && !dot) { dot = 1; return; }
    if (innum) {
      innum = 0;
      if (nintro && !hintro) { intro = nv; hintro = 1; }   // the first one of the page
      if (nthere && c == ' ') { cand = nv; hcand = 1; cand_end = tc + 6; }
      if (mode && c == '%' && dot) {   // a percentage with other decimals: no field
        if (dec == mode && np < kPct) pct[np++] = (unsigned int)nv; else bad = 1;
      }
    }
    if (c == 0x0Au) { mode = 0; return; }
    if (c == ' ') {
      if (Endet("there are  ", 11) || Endet("there were  ", 12)) {
        cand = intro; hcand = hintro; cand_end = tc + 6;
      } else if (Endet("racial makeup of the ", 21)) {
        mode = 2; np = 0; bad = 0;
      } else if (Endet("population is spread out with ", 30)) {
        mode = 1; np = 0; bad = 0;
      } else if (mode && np >= 4 && !bad && Endet(", and ", 6)) {
        Field();
        mode = 0;
      }
    } else if (c == 'e' && hcand && tc == cand_end && Endet("people", 6)) {
      pop = cand; hpop = 1;
    } else if (c == '>' && Endet("</page>", 7)) {
      hintro = hcand = hpop = 0; mode = 0;
    }
  }
  // The field is the next stream byte: its context.
  __attribute__((noinline, minsize)) void Field() {
    const unsigned long long unit = (mode == 2) ? 10000u : 1000u;   // 100 % in units of the last decimal
    int conf = 3, pred = 3;
    if (hpop && pop) {
      const unsigned long long N = pop;
      unsigned long long s = 0, sp = 0;
#pragma clang loop vectorize(disable) interleave(disable)
      for (int i = 0; i < np; ++i) {
        s += (2u * pct[i] * N + unit) / (2u * unit);
        sp += pct[i];
      }
      conf = (N < unit) ? 0 : (N < 4u * unit) ? 1 : 2;
      if (s <= N) {
        const long long Q = (long long)((2u * (N - s) * unit + N) / (2u * N));
        const long long d = Q - ((long long)unit - (long long)sp);
        pred = (d == 0) ? 0 : (d == 1) ? 1 : (d == -1) ? 2 : 3;
      }
    }
    ctx = ((mode == 2 ? 0 : 1) << 4) | (conf << 2) | pred;
  }
  // Every perceived bit (Predictor::Perceive, in front of UpdateContexts).
  __attribute__((noinline, minsize)) void Perceive(int bit, unsigned int c0) {
    if (pctx >= 0) {
      if (!g_injected_byte) {
        unsigned short& a = q[pctx][pc0];
        unsigned char& n = qn[pctx][pc0];
        a = (unsigned short)(a + ((bit ? 65535 : 0) - (int)a) / (n + 2));
        if (n < 255) ++n;
        const int err = (bit << 12) - po;
        int* wc = w[pctx];
        wc[0] += (s1 * err) >> 14;
        wc[1] += (s2 * err) >> 14;
        for (int i = 0; i < 2; ++i) wc[i] = wc[i] > (8 << 16) ? (8 << 16) : wc[i] < -(8 << 16) ? -(8 << 16) : wc[i];
      }
      pctx = -1;
    }
    if (c0 >= 128u && !g_injected_byte) {
      ctx = -1;   // the field is one byte
    }
  }
  // The final prediction p (= P(bit 1)) for the bit with partial byte c0.
  __attribute__((noinline, minsize)) float End(float p, unsigned int c0) {
    if (ctx < 0 || g_injected_byte) return p;
    int p12 = (int)(p * 4096.0f);
    p12 = p12 < 1 ? 1 : p12 > 4095 ? 4095 : p12;
    const int q12 = q[ctx][c0] >> 4;
    s1 = st[p12];
    s2 = st[q12 < 1 ? 1 : q12 > 4095 ? 4095 : q12];
    const int* wc = w[ctx];
    const long long d = ((long long)wc[0] * s1 + (long long)wc[1] * s2) >> 16;
    po = Sq((int)d);
    pctx = ctx; pc0 = (int)c0;
    return (float)po * (1.0f / 4096.0f);
  }
};
inline CensusVz cvz = {};

inline int tail = 0;      // 1 = article tail section reached

// ------------------------------------------------------- shared outputs --

// fxcmv1::modelPrediction() stores the ISBN rule value here for the bit that
// Predictor::Predict() serves immediately afterwards; the final prediction is
// then pulled towards the expected bit.  Both directions compute the same
// value.
inline int g_isbn_rule = 0;    // 0 = rule is silent, 1 = bit 0, 2 = bit 1
inline unsigned int ApmExtra() {
  unsigned int v = 0;
  v = (v << 1) | (unsigned int)(tail & 1);
  return v;
}

}

#endif
