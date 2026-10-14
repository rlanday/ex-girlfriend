#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "cm_core.h"
#include "x86t.h"
#define BOOT_ENCODER
#include "boot_cm.h"
#ifndef MAGIC
#define MAGIC "B5JPAYLOAD_V1__"
#endif
static const char MAG[15] = MAGIC;

/* forward (dir>0) / inverse (dir<0) address transform over [0,n) */
static void xform(unsigned char *b, size_t n, int dir) {
  int j;
  cm_elf_ranges(b, n);
  for (j = 0; j < cm_ncr; ++j)
    if (cm_cr_lo[j] < (long)n)
      x86t_range(b, cm_cr_lo[j], cm_cr_hi[j] < (long)n ? cm_cr_hi[j] : (long)n, cm_cr_va[j], (int)PV(xt) >> 1, dir);
}

static unsigned char *rd(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb"); long L; unsigned char *b;
  if (!f) { perror(p); exit(2); }
  fseek(f, 0, SEEK_END); L = ftell(f); fseek(f, 0, SEEK_SET);
  b = malloc(L + 64); if (fread(b, 1, L, f) != (size_t)L) exit(2); fclose(f);
  memset(b + L, 0, 64); *n = L; return b;
}
static void wr(const char *p, const unsigned char *b, size_t n) {
  FILE *f = fopen(p, "wb"); if (!f) { perror(p); exit(2); }
  fwrite(b, 1, n, f); fclose(f); chmod(p, 0755);
}
static size_t findh(unsigned char *d, size_t n) {
  size_t i;
  for (i = 0; i + 16 + 48 <= n && i < (1u << 20); i++) if (!memcmp(d + i, MAG, 15)) return i + 16;
  fprintf(stderr, "mark missing\n"); exit(2);
}
static unsigned long long ld(const unsigned char *p) { unsigned long long v = 0; int b; for (b = 7; b >= 0; b--) v = (v << 8) | p[b]; return v; }
static void st(unsigned char *p, unsigned long long v) { int b; for (b = 0; b < 8; b++) p[b] = (v >> (8 * b)) & 0xff; }

int main(int c, char **v) {
  unsigned char *g_out = malloc(CM_MAXBUF + 65536);
  if (c < 3) { fprintf(stderr, "boot|xform|compress|pack|cut|unpack|eval\n"); return 2; }
  if (!strcmp(v[1], "boot")) {
    size_t n, m, i; unsigned char *b = rd(v[2], &n), *o = malloc(2 * n + 64), *k = malloc(n + 64);
    m = boot_compress(b, n, o);
    boot_decompress(o, k, n);
    for (i = 0; i < n; ++i) if (k[i] != b[i]) { fprintf(stderr, "boot round trip FAILED at %zu\n", i); return 4; }
    wr(v[3], o, m); fprintf(stderr, "boot stage2 %zu -> %zu\n", n, m); return 0;
  }
  if (!strcmp(v[1], "xform")) {
    size_t n; unsigned char *b = rd(v[2], &n), *r = malloc(n);
    memcpy(r, b, n); xform(b, n, +1);
    { unsigned char *k = malloc(n); memcpy(k, b, n); xform(k, n, -1);
      if (memcmp(k, r, n)) { fprintf(stderr, "xform not invertible\n"); return 4; } }
    wr(v[3], b, n); fprintf(stderr, "xform Nf=%zu ranges=%d\n", n, cm_ncr); return 0;
  }
  if (!strcmp(v[1], "compress")) {
    size_t n, cl; unsigned char *b = rd(v[2], &n);
    cl = cm_compress(b, n, g_out); if (!cl) return 5;
    wr(v[3], g_out, cl); fprintf(stderr, "orig=%zu cm=%zu\n", n, cl); return 0;
  }
  if (!strcmp(v[1], "pack")) {
    size_t sl, pl, nf, nd, h, K; unsigned char *s = rd(v[2], &sl), *p = rd(v[3], &pl), *o;
    nf = strtoull(v[4], 0, 10); nd = strtoull(v[5], 0, 10); h = findh(s, sl);
    K = cm_decompress(p, pl, g_out, nd);
    if (!K || K > pl) { fprintf(stderr, "K>stream\n"); return 2; }
    st(s + h + 0 * 8, sl); st(s + h + 1 * 8, pl); st(s + h + 2 * 8, nf); st(s + h + 3 * 8, nf); st(s + h + 4 * 8, nd); st(s + h + 5 * 8, K);
    o = malloc(sl + pl); memcpy(o, s, sl); memcpy(o + sl, p, pl); wr(v[6], o, sl + pl);
    printf("stub=%zu strom=%zu paket=%zu Nf=%zu Nd=%zu K=%zu decomp_bin=%zu\n", sl, pl, sl + pl, nf, nd, K, sl + K);
    return 0;
  }
  if (!strcmp(v[1], "cut")) {
    size_t n, h, off, split, K, nl; unsigned char *b = rd(v[2], &n);
    h = findh(b, n); off = ld(b + h); split = ld(b + h + 32); K = ld(b + h + 40); nl = off + K;
    if (nl > n) { fprintf(stderr, "K beyond end\n"); return 2; }
    st(b + h + 8, K); st(b + h + 16, split); wr(v[3], b, nl);
    printf("cut %zu->%zu K=%zu Nd=%zu\n", n, nl, K, split); return 0;
  }
  if (!strcmp(v[1], "unpack")) {
    size_t n, h, off, clen, dlen, ilen, split; unsigned char *b = rd(v[2], &n);
    h = findh(b, n); off = ld(b + h); clen = ld(b + h + 8); dlen = ld(b + h + 16); ilen = ld(b + h + 24); split = ld(b + h + 32);
    if (!cm_decompress(b + off, clen, g_out, dlen)) return 5;
    xform(g_out, dlen, -1);
    if (ilen > dlen) memset(g_out + dlen, 0, ilen - dlen);
    wr(v[3], g_out, ilen);
    fprintf(stderr, "unpack off=%zu clen=%zu dlen=%zu ilen=%zu split=%zu\n", off, clen, dlen, ilen, split); return 0;
  }
  if (!strcmp(v[1], "eval")) {
    size_t n, nd = strtoull(v[3], 0, 10), cl, K; unsigned char *b = rd(v[2], &n), *r = malloc(n), *cm;
    memcpy(r, b, n); if (!getenv("NOXFORM")) xform(b, n, +1);
    if (getenv("KEST")) {   /* fast bench: K estimated from the encoder (bytes out at Nd + 4) */
      cm_ac a; size_t i, ke = 0; int bb;
      if (!cm_model_init()) return 5;
      cm_ac_enc_init(&a, g_out);
      for (i = 0; i < n; ++i) {
        if (i == nd) ke = a.outn + 4;
        for (bb = 7; bb >= 0; --bb) { int bit = (b[i] >> bb) & 1; cm_ac_encode(&a, bit, cm_predict()); cm_update(bit); }
      }
      cl = cm_ac_enc_flush(&a);
      printf("%zu %zu %zu\n", cl, ke, cl + ke);
      return 0;
    }
    cl = cm_compress(b, n, g_out); cm = malloc(cl + 64); memcpy(cm, g_out, cl);
    if (getenv("EVAL_FULL")) {
      cm_decompress(cm, cl, g_out, n); if (!getenv("NOXFORM")) xform(g_out, n, -1);
      if (memcmp(g_out, r, n)) { fprintf(stderr, "ROUNDTRIP FAILED\n"); return 6; }
      fprintf(stderr, "roundtrip ok\n");
    }
    K = cm_decompress(cm, cl, g_out, nd);
    printf("%zu %zu %zu\n", cl, K, cl + K);
    return 0;
  }
  return 2;
}
