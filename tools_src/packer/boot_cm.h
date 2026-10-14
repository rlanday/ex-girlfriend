/* boot_cm.h - bootstrap coder of the two-stage packer stub.
 * Order-1 bitwise model, one 16-bit probability per (previous byte, bit
 * prefix) with a count-driven adaptive rate, binary arithmetic coder of
 * cm_core.h (after Matt Mahoney's paq7, GPL).  Used by the host tool (encode)
 * and stage 1 (decode); both
 * sides share this file, so they are bit-exact mirror images.
 * Stage 1 must stay tiny: no tables, no multiplications beyond the coder. */
#ifndef BOOT_CM_H
#define BOOT_CM_H
#include <stdint.h>
#include <stddef.h>
#ifndef BOOT_RATE
#define BOOT_RATE 3
#endif
static uint16_t boot_p[65536];
static uint8_t boot_n[65536];

/* probability (12 bit) of a 1 in context i */
static int boot_pr(unsigned i) { int p = boot_p[i] >> 4; return p ? p : 1; }
static void boot_upd(unsigned i, int bit) {
  int r = boot_n[i] < BOOT_RATE ? boot_n[i] + 1 : BOOT_RATE;
  if (boot_n[i] < BOOT_RATE) boot_n[i]++;
  boot_p[i] += ((bit ? 65535 : 0) - (int)boot_p[i]) >> r;
}
static void boot_init(void) { unsigned i; for (i = 0; i < 65536; ++i) { boot_p[i] = 32768; boot_n[i] = 0; } }

#ifdef BOOT_ENCODER
static size_t boot_compress(const unsigned char *src, size_t n, unsigned char *out) {
  uint32_t x1 = 0, x2 = 0xffffffffu; size_t on = 0, i; int b, c1 = 0;
  boot_init();
  for (i = 0; i < n; ++i) {
    int c0 = 1;
    for (b = 7; b >= 0; --b) {
      int bit = (src[i] >> b) & 1; unsigned ix = (unsigned)(c1 << 8 | c0);
      uint32_t xm = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)boot_pr(ix)) >> 12);
      if (bit) x2 = xm; else x1 = xm + 1;
      while (((x1 ^ x2) & 0xff000000u) == 0) { out[on++] = (unsigned char)(x2 >> 24); x1 <<= 8; x2 = (x2 << 8) | 255; }
      boot_upd(ix, bit); c0 = c0 << 1 | bit;
    }
    c1 = src[i];
  }
  for (b = 24; b >= 0; b -= 8) out[on++] = (unsigned char)(x1 >> b);
  return on;
}
#endif

/* decode n bytes from in (reading past the end yields zeros is NOT needed:
 * the encoder flushes 4 bytes) */
static void boot_decompress(const unsigned char *in, unsigned char *out, size_t n) {
  uint32_t x1 = 0, x2 = 0xffffffffu, x = 0; size_t i; int b, c1 = 0;
  boot_init();
  for (b = 0; b < 4; ++b) x = x << 8 | *in++;
  for (i = 0; i < n; ++i) {
    int c0 = 1;
    for (b = 0; b < 8; ++b) {
      unsigned ix = (unsigned)(c1 << 8 | c0); int bit;
      uint32_t xm = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)boot_pr(ix)) >> 12);
      bit = x <= xm;
      if (bit) x2 = xm; else x1 = xm + 1;
      while (((x1 ^ x2) & 0xff000000u) == 0) { x1 <<= 8; x2 = (x2 << 8) | 255; x = x << 8 | *in++; }
      boot_upd(ix, bit); c0 = c0 << 1 | bit;
    }
    out[i] = (unsigned char)(c1 = c0 & 255);
  }
}
#endif
