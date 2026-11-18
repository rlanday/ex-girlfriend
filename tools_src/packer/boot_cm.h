/* boot_cm.h - bootstrap coder of the two-stage packer stub.
 * Context models after Crinkler (Aske Simon Christensen, Rune L. H. Stubbe;
 * the form, code written anew), like in oneKpaq (Teemu Suutari).  Each model
 * is a byte mask over the last 8 bytes and a weight; a context keeps a pair of
 * bit counters (the seen bit counts up, the other one is halved); a pair that
 * has seen only one bit value gets a boost; the counts, shifted by the model
 * weights, are summed and turned into a probability by one division.  Own
 * choices: the halving keeps n/2+1,
 * the boost doubles, the hash and its table size, the base count BOOT_EPS.
 * Binary arithmetic coder of cm_core.h (12-bit, after Matt Mahoney's paq7,
 * GPL).  The host tool encodes with boot_compress() and checks the round trip
 * with boot_decompress(); stage 1 (stub_boot.c) carries the matching decoder
 * written in assembly, and the probe build of build_packer.sh runs that code
 * on the real stream.
 *
 * The model list (boot_models.h) was found for the stage-2 image by a greedy
 * search that is not part of this package; the decoder is one generic loop
 * over that list.  Integer only; stage 1 keeps no table beyond the hash table
 * it maps at run time.  */
#ifndef BOOT_CM_H
#define BOOT_CM_H
#include <stdint.h>
#include <stddef.h>
#include "boot_models.h"   /* BOOT_NM, boot_mask[], boot_wt[], BOOT_TB, BOOT_EPS, BOOT_CAP */

static const uint64_t BOOT_C1 = 0x100000001B3ull;
static const uint64_t BOOT_C2 = 0x2545F4914F6CDD1Dull;
static const uint64_t BOOT_CS = 0x9E3779B97F4A7C15ull;

static uint8_t *boot_T;   /* 2 << BOOT_TB bytes */
static uint64_t boot_hist;

static inline uint64_t boot_slot(int m, unsigned c0) {
  uint64_t h = c0, hv = boot_hist; uint8_t mk = boot_mask[m];
  while (mk) { if (mk & 1) h = h * BOOT_C1 + (hv & 0xff); hv >>= 8; mk >>= 1; }
  h = (h + (uint64_t)m * BOOT_CS) * BOOT_C2;
  return (h >> (64 - BOOT_TB)) * 2;
}
static inline int boot_pr(unsigned c0, uint64_t *sl) {
  uint64_t s0 = BOOT_EPS, s1 = BOOT_EPS; int m, p;
  for (m = 0; m < BOOT_NM; ++m) {
    uint64_t q = boot_slot(m, c0); sl[m] = q;
    unsigned n0 = boot_T[q], n1 = boot_T[q + 1];
    if (!n0 && n1) n1 <<= 1; else if (!n1 && n0) n0 <<= 1;
    s0 += (uint64_t)n0 << boot_wt[m]; s1 += (uint64_t)n1 << boot_wt[m];
  }
  p = (int)((s1 << 12) / (s0 + s1)); if (p < 1) p = 1; if (p > 4095) p = 4095; return p;
}
static inline void boot_up(int bit, const uint64_t *sl) {
  int m; for (m = 0; m < BOOT_NM; ++m) { uint64_t q = sl[m]; int y = q + bit, o = q + !bit;
    if (boot_T[y] < BOOT_CAP) boot_T[y]++; if (boot_T[o] > 1) boot_T[o] = boot_T[o] / 2 + 1; }
}

#ifdef BOOT_ENCODER
static size_t boot_compress(const unsigned char *src, size_t n, unsigned char *out) {
  uint32_t x1 = 0, x2 = 0xffffffffu; size_t on = 0, i; int b; uint64_t sl[32];
  boot_hist = 0; boot_T = calloc((size_t)2 << BOOT_TB, 1);
  for (i = 0; i < n; ++i) {
    unsigned c0 = 1;
    for (b = 7; b >= 0; --b) {
      int bit = (src[i] >> b) & 1, p = boot_pr(c0, sl);
      uint32_t xm = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)p) >> 12);
      if (bit) x2 = xm; else x1 = xm + 1;
      while (((x1 ^ x2) & 0xff000000u) == 0) { out[on++] = (unsigned char)(x2 >> 24); x1 <<= 8; x2 = (x2 << 8) | 255; }
      boot_up(bit, sl); c0 = c0 << 1 | bit;
    }
    boot_hist = boot_hist << 8 | src[i];
  }
  for (b = 24; b >= 0; b -= 8) out[on++] = (unsigned char)(x1 >> b);
  free(boot_T); return on;
}
#endif

/* reference decoder (C), for the tool's round-trip check only; the shipped
 * decoder is the assembly in stub_boot.c. */
static void boot_decompress(const unsigned char *in, unsigned char *out, size_t n) {
  uint32_t x1 = 0, x2 = 0xffffffffu, x = 0; size_t i; int b; uint64_t sl[32];
  boot_hist = 0; boot_T = calloc((size_t)2 << BOOT_TB, 1);
  for (b = 0; b < 4; ++b) x = x << 8 | *in++;
  for (i = 0; i < n; ++i) {
    unsigned c0 = 1;
    for (b = 0; b < 8; ++b) {
      int bit, p = boot_pr(c0, sl);
      uint32_t xm = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)p) >> 12);
      bit = x <= xm; if (bit) x2 = xm; else x1 = xm + 1;
      while (((x1 ^ x2) & 0xff000000u) == 0) { x1 <<= 8; x2 = (x2 << 8) | 255; x = x << 8 | *in++; }
      boot_up(bit, sl); c0 = c0 << 1 | bit;
    }
    out[i] = (unsigned char)(c0 & 255); boot_hist = boot_hist << 8 | (c0 & 255);
  }
  free(boot_T);
}
#endif
