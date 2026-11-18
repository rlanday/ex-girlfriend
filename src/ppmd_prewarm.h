#ifndef PPMD_PREWARM_H_
#define PPMD_PREWARM_H_
#include "meld.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ppmd_meta.h"
#include "preprocess/dictionary.h"   // kWrtSubMin, the partial-word rule of the WRT stage

namespace ppmd_prewarm {

// Word book: runs of a-z of ./.dict in file order, first 65,360 (code plan of
// bands 1-4), FNV-1a table with linear probing (same as FastWordMap).
struct Buch {
  const unsigned char* t;
  unsigned* off;
  unsigned* len;
  unsigned* tab;
  unsigned mask;
  int Find(const unsigned char* p, unsigned n) const {
    unsigned h = 2166136261u;
    for (unsigned i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    for (h &= mask; tab[h]; h = (h + 1) & mask) {
      const unsigned i = tab[h] - 1;
      if (len[i] == n && !memcmp(t + off[i], p, n)) return (int)i;
    }
    return -1;
  }
};

static const unsigned kBandKap4 = 80 + 3840 + 40960 + 20480;   // 65,360
// the number of derived endings the stems are cut by
static const unsigned kKX = 16;

// Code bytes of dictionary rank r: preprocessor::CodeWord (dictionary.cpp).

// The stems of ./.dict (warm-up at the end of slot 1); caller frees the buffer.
__attribute__((minsize, noinline))
static unsigned char* Material(size_t* n_out) {
  FILE* f = fopen(".dict", "rb");
  if (!f) { msg_note(104); exit(1); }
  fseek(f, 0L, SEEK_END);
  const unsigned L = (unsigned)ftell(f);
  fseek(f, 0L, SEEK_SET);
  unsigned char* t = (unsigned char*)malloc(L + 1);
  if (fread(t, 1, L, f) != L) { msg_note(104); exit(1); }
  fclose(f);
  Buch b;
  b.t = t;
  b.off = (unsigned*)malloc(sizeof(unsigned) * (L / 2 + 64));
  b.len = (unsigned*)malloc(sizeof(unsigned) * (L / 2 + 64));
  unsigned nw = 0, s = 0;
  for (unsigned p = 0; p < L; ++p) {          // a word ends at a non-letter
    if (t[p] >= 'a' && t[p] <= 'z') continue;
    if (p > s && nw < kBandKap4) { b.off[nw] = s; b.len[nw] = p - s; ++nw; }
    s = p + 1;
  }
  for (b.mask = 1; b.mask < 8 * nw; b.mask <<= 1) {}
  b.tab = (unsigned*)calloc(b.mask, sizeof(unsigned));
  unsigned* seen = (unsigned*)calloc(b.mask, sizeof(unsigned));
  --b.mask;
  for (unsigned i = 0; i < nw; ++i) {
    const unsigned char* p = t + b.off[i];
    unsigned h = 2166136261u;
    for (unsigned k = 0; k < b.len[i]; ++k) { h ^= p[k]; h *= 16777619u; }
    for (h &= b.mask; b.tab[h]; h = (h + 1) & b.mask) {}
    b.tab[h] = i + 1;
  }
  unsigned char* out = (unsigned char*)malloc(6 * (size_t)L + 16);
  unsigned char* o = out;
  // Endings: counts over (w, w+X), then the KX best by count, ties smaller X.
  unsigned* cnt = (unsigned*)calloc(27 * 27 * 27 * 27, sizeof(unsigned));
  for (unsigned i = 0; i < nw; ++i) {
    const unsigned char* p = t + b.off[i];
    const unsigned n = b.len[i];
    for (unsigned k = 1; k <= 4 && n >= k + 3; ++k) {
      if (b.Find(p, n - k) < 0) continue;
      unsigned c = 0;
      for (unsigned j = 0; j < 4; ++j) c = c * 27 + (j < k ? p[n - k + j] - 'a' + 1 : 0);
      ++cnt[c];
    }
  }
  unsigned char xs[kKX][4];
  unsigned xl[kKX], nx = 0;
  for (; nx < kKX; ++nx) {
    unsigned best = 0, bc = 0;
    for (unsigned c = 0; c < 27 * 27 * 27 * 27; ++c)
      if (cnt[c] > bc) { bc = cnt[c]; best = c; }
    if (!bc) break;
    cnt[best] = 0;
    xl[nx] = 0;
    for (unsigned j = 0, d = 27 * 27 * 27; j < 4; ++j, d /= 27)
      if (best / d % 27) xs[nx][xl[nx]++] = (unsigned char)('a' - 1 + best / d % 27);
  }
  free(cnt);
  // Stems in dictionary order; every stem once, only if not a dictionary word.
  for (unsigned i = 0; i < nw; ++i) {
    const unsigned char* p = t + b.off[i];
    for (unsigned x = 0; x < nx; ++x) {
      const unsigned n = b.len[i], sl = n - xl[x];
      if (n < xl[x] + 3 || memcmp(p + sl, xs[x], xl[x]) || b.Find(p, sl) >= 0) continue;
      unsigned h = 2166136261u, e;
      for (unsigned k = 0; k < sl; ++k) { h ^= p[k]; h *= 16777619u; }
      for (h &= b.mask; (e = seen[h]); h = (h + 1) & b.mask)
        if ((e >> 17) == sl && !memcmp(t + b.off[(e & 0x1FFFF) - 1], p, sl)) break;
      if (e) continue;
      seen[h] = (i + 1) | (sl << 17);
      *o++ = ' ';
      int r = -1;
      unsigned z = 0;                         // code(r) [z,sl)
      if (sl > preprocessor::kWrtSubMin)
        for (z = sl - 1; z >= preprocessor::kWrtSubMin && (r = b.Find(p, z)) < 0; --z) {}
      if (r < 0) z = 0;
      else o = preprocessor::CodeWord((unsigned)r, o);
      memcpy(o, p + z, sl - z); o += sl - z;
    }
  }
  free(seen); free(b.tab); free(b.len); free(b.off); free(t);
  *n_out = (size_t)(o - out);
  return out;
}

__attribute__((minsize, noinline))
static unsigned char* CodesMaterial(size_t* n_out) {
  static_assert(54857 <= kBandKap4, "exceeds the bands CodeWord() knows");
  unsigned char* out = (unsigned char*)malloc(4 * (size_t)(54857) + 16);
  unsigned char* o = out;
  for (unsigned r = 0; r < (unsigned)(54857); ++r) {
    *o++ = ' ';
    o = preprocessor::CodeWord(r, o);
  }
  *n_out = (size_t)(o - out);
  return out;
}

}  // namespace ppmd_prewarm

#endif
