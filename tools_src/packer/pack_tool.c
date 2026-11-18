/* pack_tool.c - offline tool for the truncatable x86-CM stub.
 * Does NOT count in S.  Header = six u32 at file offset 120 of stage 1,
 * the place the cut of the self-extraction reads and rewrites.
 *   boot <stage2.bin> <out>            bootstrap coder for stage 2 (boot_cm.h), with round trip
 *   xform <img> <out>                  exact x86 address transform (forward)
 *   compress <ximg> <out.cm> [W]       CM stream (no length prefix)
 *   pack <stub> <cm> <Nf> <Nd> <out> [W]   patch the 6-field header + append stream
 *   cut <package> <out>                like the -e path: cut at payload_off+K
 *   unpack <package> <out> [W]         reference unpacker (CM + inverse transform + zero fill)
 *   eval <img> <Nd>                    bench: xform+compress+exact K+full decode check
 * W = <s2.bin> <s2text>: warm-up of the x86-CM on the flat stage-2 image
 * (as the stub does it, cm_core.h cm_warm); s2text = length of its code.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "cm_core.h"
#include "x86t.h"
#define BOOT_ENCODER
#include "boot_cm.h"

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
/* header: six u32 behind the ELF header (64 B) and the one program header (56 B) */
#define HOFF 120
static size_t findh(unsigned char *d, size_t n) {
  if (n < HOFF + 24 || memcmp(d, "\177ELF", 4) || d[56] != 1 || d[57] != 0) { fprintf(stderr, "no stage-1 header\n"); exit(2); }
  return HOFF;
}
/* warm-up data: the flat stage-2 image, code address-transformed like the stub does it */
static void warm(const char *s2p, const char *s2t) {
  size_t n; unsigned char *b = rd(s2p, &n);
  cm_wt = strtol(s2t, 0, 0); cm_wn = n;
  x86t_range(b, 0, cm_wt, CM_WARM_VA, (int)PV(xt) >> 1, +1);
  cm_wp = b;
}
static unsigned long long ld(const unsigned char *p) { unsigned long long v = 0; int b; for (b = 3; b >= 0; b--) v = (v << 8) | p[b]; return v; }
static void st(unsigned char *p, unsigned long long v) { int b; if (v >> 32) { fprintf(stderr, "header field >= 4 GB\n"); exit(2); } for (b = 0; b < 4; b++) p[b] = (v >> (8 * b)) & 0xff; }

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
    if (c > 5) warm(v[4], v[5]);
    cl = cm_compress(b, n, g_out); if (!cl) return 5;
    wr(v[3], g_out, cl); fprintf(stderr, "orig=%zu cm=%zu\n", n, cl); return 0;
  }
  if (!strcmp(v[1], "pack")) {
    size_t sl, pl, nf, nd, h, K; unsigned char *s = rd(v[2], &sl), *p = rd(v[3], &pl), *o;
    nf = strtoull(v[4], 0, 10); nd = strtoull(v[5], 0, 10); h = findh(s, sl);
    if (c > 8) warm(v[7], v[8]);
    K = cm_decompress(p, pl, g_out, nd);
    if (!K || K > pl) { fprintf(stderr, "K>stream\n"); return 2; }
    st(s + h + 0 * 4, sl); st(s + h + 1 * 4, pl); st(s + h + 2 * 4, nf); st(s + h + 3 * 4, nf); st(s + h + 4 * 4, nd); st(s + h + 5 * 4, K);
    o = malloc(sl + pl); memcpy(o, s, sl); memcpy(o + sl, p, pl); wr(v[6], o, sl + pl);
    printf("stub=%zu strom=%zu paket=%zu Nf=%zu Nd=%zu K=%zu decomp_bin=%zu\n", sl, pl, sl + pl, nf, nd, K, sl + K);
    return 0;
  }
  if (!strcmp(v[1], "cut")) {
    size_t n, h, off, split, K, nl; unsigned char *b = rd(v[2], &n);
    h = findh(b, n); off = ld(b + h); split = ld(b + h + 16); K = ld(b + h + 20); nl = off + K;
    if (nl > n) { fprintf(stderr, "K beyond end\n"); return 2; }
    st(b + h + 4, K); st(b + h + 8, split); wr(v[3], b, nl);
    printf("cut %zu->%zu K=%zu Nd=%zu\n", n, nl, K, split); return 0;
  }
  if (!strcmp(v[1], "unpack")) {
    size_t n, h, off, clen, dlen, ilen, split; unsigned char *b = rd(v[2], &n);
    h = findh(b, n); off = ld(b + h); clen = ld(b + h + 4); dlen = ld(b + h + 8); ilen = ld(b + h + 12); split = ld(b + h + 16);
    if (c > 5) warm(v[4], v[5]);
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
