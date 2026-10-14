/* cm_core.h - x86-aware context-mixing codec for the packer stub.
 *
 * Successor of an earlier context-mixing packer of this entry.  Encoder (host
 * tool) and decoder (stub)
 * share THIS file, so both sides are bit-exact mirror images.  Purely integer.
 *
 * What is new against cm_core.h:
 *  - x86-64 length decoder (x86p.h) run over the already coded bytes: role of
 *    the next byte (opcode / ModRM / SIB / displacement / immediate, index in
 *    the field), opcode, ModRM, prefixes, previous instructions.  It only
 *    sees bytes both sides already have, so it needs no side information.
 *  - code/data split taken from the ELF program headers inside the stream
 *    (executable PT_LOADs), read as soon as they have been coded.
 *  - contexts from that parse; statemaps split by instruction role;
 *    several mixers selected by role / order-1 / match state, second mixer
 *    layer; two match models; up to four APM/SSE stages.
 *  - exact address transform (x86t.h) instead of BCJ: RIP-relative disp32
 *    and rel32 of call/jmp/jcc become absolute addresses.
 *
 * Parameters: CM2_LAB = runtime array P[] (experiment bench), otherwise the
 * CM2_* constants from cm_params.h (one tuned set, stamped into the stub).
 * squash() with its 33-point table, the inverse stretch table, the binary
 * arithmetic coder and the APM form follow Matt Mahoney's paq7 (GPL); the
 * x86 parse, the contexts, the mixer selection, the match models and the
 * address transform are written for this entry.
 */
#ifndef CM_CORE_H
#define CM_CORE_H
#include <stdint.h>
#include <stddef.h>
#include "x86p.h"

#ifndef CM_ALLOC
#include <stdlib.h>
#define CM_ALLOC(n) calloc(1, (n))
#endif

#include "cm_params.h"
#define PV(n) (CM2_##n)

static int cm_squash(int d) {
  static const int t[33] = {1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,1546,2047,
    2549,2994,3348,3607,3785,3901,3975,4022,4050,4068,4079,4085,4089,4092,4093,4094};
  int w, i;
  if (d > 2047) return 4095;
  if (d < -2047) return 0;
  w = d & 127; i = (d >> 7) + 16;
  return (t[i] * (128 - w) + t[i + 1] * w + 64) >> 7;
}
static int16_t cm_st_tab[4096];
#define STR(p) ((int)cm_st_tab[(p) & 4095])

static uint8_t cm_ns(uint8_t s, int bit) {
  int n0 = s >> 4, n1 = s & 15;
  if (PV(nsd) == 1) {        /* stronger discount of the opposite count */
    if (bit) { if (n1 < 15) n1++; if (n0 > 1) n0 = 1 + ((n0 - 1) >> 1); }
    else     { if (n0 < 15) n0++; if (n1 > 1) n1 = 1 + ((n1 - 1) >> 1); }
  } else if (PV(nsd) == 2) { /* weaker discount */
    if (bit) { if (n1 < 15) n1++; if (n0 > 3) n0 = 3 + ((n0 - 3) >> 1); }
    else     { if (n0 < 15) n0++; if (n1 > 3) n1 = 3 + ((n1 - 3) >> 1); }
  } else if (PV(nsd) == 3) { /* discount by a quarter */
    if (bit) { if (n1 < 15) n1++; if (n0 > 2) n0 -= (n0 + 2) >> 2; }
    else     { if (n0 < 15) n0++; if (n1 > 2) n1 -= (n1 + 2) >> 2; }
  } else {
  if (bit) { if (n1 < 15) n1++; if (n0 > 2) n0 = 2 + ((n0 - 2) >> 1); }
  else     { if (n0 < 15) n0++; if (n1 > 2) n1 = 2 + ((n1 - 2) >> 1); }
  }
  return (uint8_t)((n0 << 4) | n1);
}
typedef struct { uint16_t p; uint16_t n; } cm_sm_e;
static void cm_sm_init(cm_sm_e *m, int nstates) {
  int s;
  for (s = 0; s < nstates; ++s) {
    int n0 = (s >> 4) & 15, n1 = s & 15;
    int p = ((n1 * 2 + 1) * 65536) / ((n0 + n1) * 2 + 2);
    m[s].p = (uint16_t)p; m[s].n = 0;
  }
}
static void cm_sm_upd(cm_sm_e *m, int s, int bit, int limit) {
  int target = bit ? 65535 : 0;
  m[s].p = (uint16_t)((int)m[s].p + (target - (int)m[s].p) / (m[s].n + 2));
  if (m[s].n < limit) m[s].n++;
}

#define CM_MAXCTX 48
#define CM_SMX 16
#define CM_NMM 2
#define CM_MAXIN (3 * CM_MAXCTX + CM_NMM + 2)
#define CM_NMIX 6
#define CM_MAXBUF (1u << 21)
#if defined(CM2_LAB) || CM2_dp
typedef struct { uint8_t state, chk; uint16_t p; } cm_node;
#else
typedef struct { uint8_t state, chk; } cm_node;
#endif

static int cm_nctx, cm_nin, cm_nmm, cm_nmix;
static unsigned char cm_ctxid[CM_MAXCTX];
static cm_node *cm_t;
static uint32_t cm_hmask;
static cm_sm_e cm_sms[CM_MAXCTX][256 * CM_SMX], cm_smf[CM_MAXCTX][256 * CM_SMX];
static unsigned char *cm_hist;
static uint32_t *cm_mmh[CM_NMM]; static uint32_t cm_mmptr[CM_NMM];
static int cm_mmlen[CM_NMM], cm_mmpred[CM_NMM], cm_mmsi[CM_NMM], cm_mmL[CM_NMM];
static cm_sm_e cm_mmsm[CM_NMM][128];
static int32_t *cm_mw[CM_NMIX]; static int cm_msel[CM_NMIX], cm_mcur[CM_NMIX], cm_mp[CM_NMIX], cm_mst[CM_NMIX + 1];
static int32_t cm_w2[4096][CM_NMIX + 1]; static int cm_w2s;
static uint16_t cm_apm1[256 * 33], cm_apm2[4096 * 33], cm_apm3[8192 * 33], cm_apm4[8192 * 33];
static int cm_a1i, cm_a1w, cm_a2i, cm_a2w, cm_a3i, cm_a3w, cm_a4i, cm_a4w;
static uint32_t cm_c4, cm_c8, cm_word; static int cm_c0, cm_bpos; static size_t cm_pos;
static cm_node *cm_np[CM_MAXCTX]; static int cm_stt[CM_MAXCTX]; static int cm_st[CM_MAXIN];
static int cm_mixp;
static uint32_t cm_base[CM_MAXCTX]; static uint8_t cm_chk[CM_MAXCTX];
static int cm_smc;
static uint32_t cm_fld; static unsigned char cm_lastf[1024][8]; static int cm_incode;
/* executable PT_LOAD ranges: file offset lo/hi, vaddr - offset */
static long cm_cr_lo[8], cm_cr_hi[8], cm_cr_va[8]; static int cm_ncr, cm_crdone;
static long cm_curva;
static uint32_t *cm_sph; static int cm_spb, cm_spb2;   /* structural match: predicted byte (+1, 0 = none) */

/* primes and the first avalanche steps of xxHash32 (Yann Collet, BSD-2-Clause) */
static uint32_t cm_hashn(uint32_t a, uint32_t b) {
  uint32_t h = a * 2654435761u + b * 2246822519u + 0x9E3779B1u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return h;
}
static uint32_t cm_h3(uint32_t a, uint32_t b, uint32_t c) { return cm_hashn(cm_hashn(a, b), c); }

static uint64_t cm_le(const unsigned char *p, int n) {
  uint64_t v = 0; while (n--) v = v << 8 | p[n]; return v;
}
/* Read the executable PT_LOADs from an ELF64 header in buf[0..n). 1 = done. */
static int cm_elf_ranges(const unsigned char *b, size_t n) {
  size_t phoff, phn, j;
  cm_ncr = 0;
  if (n < 64) return 0;
  phoff = (size_t)cm_le(b + 32, 8); phn = (size_t)cm_le(b + 56, 2);
  if (phoff > (1u << 20) || phn > 64) return 1;      /* not ours: no code ranges */
  if (n < phoff + 56 * phn) return 0;
  for (j = 0; j < phn && cm_ncr < 8; ++j) {
    const unsigned char *ph = b + phoff + 56 * j;
    if (cm_le(ph, 4) == 1 && (cm_le(ph + 4, 4) & 1) && cm_le(ph + 32, 8) > 0) {
      cm_cr_lo[cm_ncr] = (long)cm_le(ph + 8, 8);
      cm_cr_hi[cm_ncr] = cm_cr_lo[cm_ncr] + (long)cm_le(ph + 32, 8);
      cm_cr_va[cm_ncr] = (long)cm_le(ph + 16, 8) - cm_cr_lo[cm_ncr];
      cm_ncr++;
    }
  }
  return 1;
}

static int cm_xrole(void) { if (!cm_incode) return 9; return xr_role == XR_OP && (xr_op & 0x400) ? 8 : xr_role; }

#define CU(id) ((PV(ctx) >> (id)) & 1)
static uint32_t cm_ctxhash(int id) {
  uint32_t c4 = cm_c4, c8 = cm_c8, c1 = c4 & 0xff, c2 = c4 & 0xffff, c3 = c4 & 0xffffff;
  int r = cm_xrole(), ix = xr_idx > 7 ? 7 : xr_idx;
  uint32_t rr = (uint32_t)(r << 3 | ix);
  int fk = (xr_idx & 3) + (xr_role == XR_IMM ? 4 : 0);
  uint32_t pins = (uint32_t)(xr_lop << 9 | xr_lmrm);
  uint32_t om = rr << 20 | (uint32_t)xr_op << 9 | (uint32_t)xr_mrm;
  unsigned char *lf = cm_lastf[xr_op & 0x3ff];
  switch (id) {
  case 0: return 0x30000000u;
  case 1: if (CU(1)) return cm_hashn(1, c1); break;
  case 2: if (CU(2)) return cm_hashn(2, c2); break;
  case 3: if (CU(3)) return cm_hashn(3, c3); break;
  case 4: if (CU(4)) return cm_hashn(4, c4); break;
  case 5: if (CU(5)) return cm_hashn(5, c4 ^ ((c8 & 0xff) << 4)); break;
  case 6: if (CU(6)) return cm_hashn(6, c4 ^ (c8 << 3)); break;
  case 7: if (CU(7)) return cm_hashn(7, c4 ^ (c8 << 5) ^ 0x7000u); break;
  case 8: if (CU(8)) return cm_hashn(8, c4 ^ c8); break;
  case 9: if (CU(9)) return cm_hashn(9, (c4 & 0xff00) ^ (c4 & 0xff000000)); break;
  case 10: if (CU(10)) return cm_hashn(10, (c4 & 0xff) ^ ((c4 & 0xff0000) >> 8)); break;
  case 11: if (CU(11)) return cm_hashn(11, cm_word); break;
  case 12: if (CU(12)) return cm_hashn(12, cm_word ^ (c1 << 24)); break;
  case 13: if (CU(13)) return cm_h3(13, c3, c8 & 0xffff); break;
  case 14: if (CU(14)) return cm_h3(14, c4, c8 & 0xffffff); break;
  case 15: if (CU(15)) return cm_h3(15, (uint32_t)r, c2 & 0xff00); break;
  case 16: if (CU(16)) return cm_h3(16, rr, (uint32_t)(xr_op | xr_pfx << 12)); break;
  case 17: if (CU(17)) return cm_h3(17, rr << 20 | (uint32_t)xr_op, (uint32_t)(xr_mrm << 8 | (int)c1)); break;
  case 18: if (CU(18)) return cm_h3(18, rr << 12 | (uint32_t)xr_op, xr_hist & 0xffffff); break;
  case 19: if (CU(19)) return cm_h3(19, om, c2); break;
  case 20: if (CU(20)) return cm_h3(20, rr << 12 | (uint32_t)xr_op, xr_hist & 0xffff) ^ cm_hashn(c1, (uint32_t)xr_mrm); break;
  case 21: if (CU(21)) return cm_h3(21, rr, xr_hist); break;
  case 22: if (CU(22)) return cm_h3(22, om, c3 | (uint32_t)xr_pfx << 24); break;
  case 23: if (CU(23)) return cm_h3(23, rr, c1); break;
  case 24: if (CU(24)) return cm_h3(24, rr, c2); break;
  case 25: if (CU(25)) return cm_h3(25, om, pins); break;
  case 26: if (CU(26)) return cm_h3(26, om, cm_fld); break;
  case 27: if (CU(27)) return cm_h3(27, om, cm_fld ^ (uint32_t)lf[fk] << 24); break;
  case 28: if (CU(28)) return cm_h3(28, om, lf[fk] | (uint32_t)lf[fk ^ 1] << 8); break;
  case 29: if (CU(29)) return cm_h3(29, rr << 20 | (uint32_t)xr_op << 9, pins ^ (xr_hist & 0xff0000)); break;
  case 30: if (CU(30)) return cm_h3(30, (uint32_t)cm_incode, c2 | (c4 >> 16 & 0xf0) << 16); break;
  case 31: if (CU(31)) return cm_h3(31, rr << 12 | (uint32_t)xr_op, cm_fld ^ xr_hist << 24); break;
  case 32: if (CU(32)) return cm_h3(32, rr << 12 | (uint32_t)xr_op, cm_fld ^ (uint32_t)(((cm_pos + cm_curva) >> (8 * (xr_idx & 3))) & 0xff) << 24); break;
  case 33: if (CU(33)) return cm_h3(33, om, (uint32_t)(((cm_pos + cm_curva) >> (8 * (xr_idx & 3) + 4)) & 0xfff)); break;
  case 34: if (CU(34)) return cm_h3(34, (uint32_t)xr_op << 9 | (uint32_t)xr_mrm, pins ^ (uint32_t)cm_c4 << 16 ^ rr); break;
  case 35: if (CU(35)) return cm_h3(35, (uint32_t)r, (uint32_t)(cm_pos & 7) << 16 | c1); break;
  case 36: if (CU(36)) return cm_h3(36, c2, (uint32_t)(cm_pos & 7)); break;
  case 37: if (CU(37)) return cm_h3(37, om, (uint32_t)cm_spb); break;
  case 38: if (CU(38)) return cm_h3(38, (uint32_t)(cm_mmlen[0] ? cm_hist[cm_mmptr[0]] + 1 : 0) << 8 | (cm_mmlen[0] > 15 ? 15 : cm_mmlen[0]) << 16, c1); break;
  case 39: if (CU(39)) return cm_h3(39, om, (uint32_t)cm_spb | (uint32_t)cm_spb2 << 9); break;
  case 40: if (CU(40)) return cm_h3(40, rr, (uint32_t)cm_spb << 8 | c1); break;
  case 41: if (CU(41)) return cm_h3(41, (uint32_t)(cm_pos & 3) | (uint32_t)cm_incode << 2, cm_pos >= 4 ? cm_hist[cm_pos - 4] : 0); break;
  case 42: if (CU(42)) return cm_h3(42, (uint32_t)(cm_pos & 7) | (uint32_t)cm_incode << 3, cm_pos >= 16 ? (uint32_t)cm_hist[cm_pos - 8] | (uint32_t)cm_hist[cm_pos - 16] << 8 : 0); break;
  case 43: if (CU(43)) return cm_h3(43, (uint32_t)(cm_pos & 3), cm_pos >= 8 ? (uint32_t)cm_hist[cm_pos - 4] | (uint32_t)cm_hist[cm_pos - 8] << 8 | c1 << 16 : 0); break;
  case 45: if (CU(45)) return cm_h3(45, om, (uint32_t)xr_sib << 8 | c1); break;
  case 46: if (CU(46)) return cm_h3(46, (uint32_t)(xr_ipos > 15 ? 15 : xr_ipos) << 8 | (uint32_t)r, (uint32_t)xr_op << 8 | c1); break;
  case 47: if (CU(47)) return cm_h3(47, (uint32_t)(xr_ipos > 15 ? 15 : xr_ipos), c2 ^ (uint32_t)xr_pfx << 16); break;
  case 48: if (CU(48)) return cm_h3(48, om, (uint32_t)(xr_rex & 15) << 8 | (uint32_t)xr_pfx); break;
  case 49: if (CU(49)) return cm_h3(49, om ^ (uint32_t)(xr_rex & 15) << 28, (uint32_t)xr_lmrm << 9 | (uint32_t)xr_llmrm); break;
  case 50: if (CU(50)) return cm_h3(50, rr << 12 | (uint32_t)xr_op, (uint32_t)xr_lop << 20 ^ (uint32_t)xr_llop << 8 ^ (uint32_t)xr_lmrm << 16 ^ (uint32_t)(xr_rex & 15)); break;
  case 51: if (CU(51)) return cm_h3(51, rr << 12 | (uint32_t)xr_op, (xr_rh & 0xffffff) ^ (uint32_t)(xr_rex & 15) << 24); break;
  case 52: if (CU(52)) return cm_h3(52, om, (uint32_t)xr_vexp); break;
  case 53: if (CU(53)) return cm_h3(53, rr << 12 | (uint32_t)xr_op, (xr_rh & 0xffff) << 8 | c1); break;
  case 44: if (CU(44)) return cm_h3(44, (uint32_t)(cm_pos & 7), cm_pos >= 8 ? (uint32_t)cm_hist[cm_pos - 8] | c1 << 8 : 0); break;
  }
  return 0;
}

static void cm_byte_ctx(void) {
  int i, r = cm_xrole();
  for (i = 0; i < cm_nctx; ++i) { cm_base[i] = cm_ctxhash(cm_ctxid[i]); cm_chk[i] = (uint8_t)(cm_base[i] >> 24); }
  cm_smc = 0;
  if (PV(smx) == 1) cm_smc = r;
  if (PV(smx) == 2) {
    cm_smc = r < 6 ? r : 6 + (xr_idx > 3 ? 3 : xr_idx) + (r == XR_IMM ? 4 : 0);
    if (r >= 8) cm_smc = 14 + (r - 8);
  }
  if (cm_smc >= CM_SMX) cm_smc = CM_SMX - 1;
}

static int cm_lenb(int l) { return l == 0 ? 0 : l < 16 ? 1 : l < 32 ? 2 : 3; }
static int cm_msel_of(int k) {
  int r = cm_xrole(), ix = xr_idx > 3 ? 3 : xr_idx, bp = cm_bpos;
  switch (cm_msel[k]) {
  case 1: return (int)(cm_c4 & 0xff) << 3 | bp;
  case 2: return ((r << 2 | ix) << 3) | bp;
  case 3: return (((xr_op & 0xff) ^ (xr_op >> 8) * 0x3b) & 0xff) << 3 | bp;
  case 4: return (cm_lenb(cm_mmlen[0]) << 3 | bp) | (cm_mmpred[0] < 0 ? 0 : 32 + (cm_mmpred[0] << 5));
  case 5: return ((r << 2 | ix) * 8 + (cm_mmlen[0] == 0 ? 0 : cm_mmlen[0] < 16 ? 1 : 2)) << 3 | bp;
  case 6: return (int)(cm_c0 & 0xff) << 4 | (r < 8 ? r : 8);
  case 7: return ((r << 2 | ix) * 8 + cm_lenb(cm_mmlen[0]) * 2 + (cm_mmpred[0] > 0)) << 3 | bp;
  }
  return 0;
}

static void cm_apm_init(uint16_t *t, int n) {
  int i, j;
  for (i = 0; i < n; ++i) for (j = 0; j < 33; ++j) t[i * 33 + j] = (uint16_t)(cm_squash(j * 128 - 2048) * 16);
}

/* 0 = out of memory. */
static int cm_model_init(void) {
  int i, j, k, pi = 0, x;
  long ms;
  for (x = -2047; x <= 2047; ++x) { i = cm_squash(x); for (; pi <= i; ++pi) cm_st_tab[pi] = (int16_t)x; }
  for (; pi < 4096; ++pi) cm_st_tab[pi] = 2047;
  x86_init();
  cm_hmask = PV(sh) ? (1u << PV(sh)) - 1 : (1u << PV(hb)) - 1;
  cm_nctx = 0;
  for (i = 0; i < 64; ++i) if ((PV(ctx) >> i & 1) && cm_nctx < CM_MAXCTX) cm_ctxid[cm_nctx++] = (unsigned char)i;
  cm_nmm = 0;
  if (PV(mm1)) cm_mmL[cm_nmm++] = (int)PV(mm1);
  if (PV(mm2)) cm_mmL[cm_nmm++] = (int)PV(mm2);
  cm_nin = (PV(dp) ? 3 : 2) * cm_nctx + cm_nmm + 1;
  cm_t = (cm_node *)CM_ALLOC(PV(sh) ? sizeof(cm_node) << PV(sh) : (size_t)cm_nctx * sizeof(cm_node) << PV(hb));
  cm_hist = (unsigned char *)CM_ALLOC(CM_MAXBUF);
  cm_sph = (uint32_t *)CM_ALLOC((size_t)4 << 20);
  if (!cm_t || !cm_hist || !cm_sph) return 0;
  cm_spb = cm_spb2 = 0;
  for (k = 0; k < cm_nmm; ++k) {
    cm_mmh[k] = (uint32_t *)CM_ALLOC((size_t)4 << PV(mmhb));
    if (!cm_mmh[k]) return 0;
    cm_sm_init(cm_mmsm[k], 128); cm_mmlen[k] = 0; cm_mmpred[k] = -1;
  }
  ms = PV(msel); cm_nmix = 0;
  while (ms && cm_nmix < CM_NMIX) { cm_msel[cm_nmix++] = (int)(ms & 15) - 1; ms >>= 4; }
  if (!cm_nmix) { cm_nmix = 1; cm_msel[0] = 0; }
  for (k = 0; k < cm_nmix; ++k) {
    cm_mw[k] = (int32_t *)CM_ALLOC(sizeof(int32_t) * 4096 * (size_t)cm_nin);
    if (!cm_mw[k]) return 0;
    for (i = 0; i < 4096 * cm_nin; ++i) cm_mw[k][i] = (int32_t)PV(winit);
  }
  for (i = 0; i < 4096; ++i) for (j = 0; j <= CM_NMIX; ++j) cm_w2[i][j] = 65536 / cm_nmix;
  for (i = 0; i < cm_nctx; ++i) { cm_sm_init(cm_sms[i], 256 * CM_SMX); cm_sm_init(cm_smf[i], 256 * CM_SMX); }
  cm_apm_init(cm_apm1, 256); cm_apm_init(cm_apm2, 4096); cm_apm_init(cm_apm3, 8192); cm_apm_init(cm_apm4, 8192);
  cm_c4 = cm_c8 = cm_word = 0; cm_c0 = 1; cm_bpos = 0; cm_pos = 0;
  cm_incode = 1; cm_ncr = 0; cm_crdone = 0; cm_fld = 0;
  for (i = 0; i < 1024 * 8; ++i) cm_lastf[i >> 3][i & 7] = 0;
  cm_byte_ctx();
  return 1;
}

static int cm_apm_pp(uint16_t *tab, int ctx, int pr, int *oi, int *ow) {
  int s = STR(pr) + 2048, lo = s >> 7, w = s & 127, i = ctx * 33 + lo;
  *oi = i; *ow = w;
  return (tab[i] * (128 - w) + tab[i + 1] * w) >> 11;
}
static void cm_apm_upd(uint16_t *tab, int i, int w, int bit) {
  int t = bit ? 65535 : 0, wn = w >= 64 ? i + 1 : i;
  tab[wn] = (uint16_t)((int)tab[wn] + ((t - (int)tab[wn]) >> PV(apmr)));
}
static int cm_dot(const int32_t *w, int n) {
  long long d = 0; int i;
  for (i = 0; i < n; ++i) d += (long long)w[i] * cm_st[i];
  d >>= 16; if (d < -2047) d = -2047; else if (d > 2047) d = 2047;
  return (int)d;
}

static int cm_predict(void) {
  int i, pr, k, j = 2 * cm_nctx;
  uint32_t c0h = cm_hashn(0x51ed, (uint32_t)cm_c0);
  int so = cm_smc << 8;
  for (i = 0; i < cm_nctx; ++i) {
    cm_node *c = &cm_t[(PV(sh) ? 0 : (size_t)i << PV(hb)) + ((cm_base[i] ^ c0h) & cm_hmask & ~(uint32_t)(PV(bk) - 1))];
    int s;
    if (PV(bk) > 1) {       /* bucket search; miss -> replace the entry with the fewest counts */
      int w, bw = 0, bp = 99;
      for (w = 0; w < PV(bk); ++w) {
        int pr;
        if (c[w].chk == cm_chk[i]) { bw = w; bp = -1; break; }
        pr = (c[w].state >> 4) + (c[w].state & 15);
        if (pr < bp) { bp = pr; bw = w; }
      }
      c += bw;
    }
    if (c->chk != cm_chk[i]) { c->state = 0; c->chk = cm_chk[i];
#if defined(CM2_LAB) || CM2_dp
      c->p = 32768;
#endif
    }
    s = c->state | so; cm_np[i] = c; cm_stt[i] = s;
    cm_st[2 * i] = STR(cm_sms[i][s].p >> 4);
    cm_st[2 * i + 1] = STR(cm_smf[i][s].p >> 4);
  }
  for (k = 0; k < cm_nmm; ++k) {
    if (cm_mmlen[k] > 0 && cm_mmptr[k] < cm_pos) {
      int l = cm_mmlen[k], pc;
      cm_mmpred[k] = (cm_hist[cm_mmptr[k]] >> (7 - cm_bpos)) & 1;
      cm_mmsi[k] = (l > 15 ? 12 + (l > 31) + (l > 63) + (l > 127) : l) * 2 + (PV(mmb) ? (cm_bpos == 0) : 0);
      pc = cm_mmsm[k][cm_mmsi[k]].p >> 4;
      cm_st[j++] = cm_mmpred[k] ? STR(pc) : -STR(pc);
    } else { cm_mmpred[k] = -1; cm_st[j++] = 0; }
  }
#if defined(CM2_LAB) || CM2_dp
  if (PV(dp)) for (i = 0; i < cm_nctx; ++i) cm_st[j++] = STR(cm_np[i]->p >> 4);
#endif
  cm_st[j] = 256;
  for (k = 0; k < cm_nmix; ++k) {
    cm_mcur[k] = cm_msel_of(k);
    cm_mst[k] = cm_dot(cm_mw[k] + (size_t)cm_mcur[k] * cm_nin, cm_nin);
    cm_mp[k] = cm_squash(cm_mst[k]);
  }
  if (cm_nmix == 1) pr = cm_mp[0];
  else {
    long long d = 0;
    cm_mst[cm_nmix] = 256;
    switch (PV(w2sel)) {
    case 0: cm_w2s = cm_bpos; break;
    case 1: cm_w2s = cm_xrole() << 3 | cm_bpos; break;
    case 2: cm_w2s = cm_c0 & 255; break;
    default: cm_w2s = (cm_xrole() << 3 | cm_bpos) * 4 + cm_lenb(cm_mmlen[0]); break;
    }
    for (k = 0; k <= cm_nmix; ++k) d += (long long)cm_w2[cm_w2s][k] * cm_mst[k];
    d >>= 16; if (d < -2047) d = -2047; else if (d > 2047) d = 2047;
    pr = cm_squash((int)d);
  }
  cm_mixp = pr;
  {
    int p1, p2, p3 = 0, p4 = 0;
    p1 = cm_apm_pp(cm_apm1, (int)(cm_c4 & 0xff), pr, &cm_a1i, &cm_a1w);
    p2 = cm_apm_pp(cm_apm2, (int)(cm_hashn(99, cm_c4 & 0xffff) & 4095), pr, &cm_a2i, &cm_a2w);
    if (PV(aw3))
      p3 = cm_apm_pp(cm_apm3, (int)(cm_hashn(98, (uint32_t)(cm_xrole() << 16 | xr_op << 4 | (xr_idx & 15)) ^ (uint32_t)cm_c0 << 20) & 8191),
                     pr, &cm_a3i, &cm_a3w);
    if (PV(aw4)) {
      int ml = cm_mmlen[0] > 15 ? 15 : cm_mmlen[0];
      p4 = cm_apm_pp(cm_apm4, (int)(cm_hashn(97, (uint32_t)(ml << 10 | (cm_mmpred[0] + 1) << 8 | cm_c0)) & 8191),
                     pr, &cm_a4i, &cm_a4w);
    }
    pr = (int)((pr * PV(aw0) + p1 * PV(aw1) + p2 * PV(aw2) + p3 * PV(aw3) + p4 * PV(aw4)) /
               (PV(aw0) + PV(aw1) + PV(aw2) + PV(aw3) + PV(aw4)));
  }
  if (pr < 1) pr = 1; else if (pr > 4094) pr = 4094;
  return pr;
}

static void cm_update(int bit) {
  int i, k, err, lim = 1 << PV(clip);
  int em = (int)PV(errmul);
  if (PV(eboost)) em += (int)(PV(eboost) * 16 / ((long)(cm_pos >> 10) + 16));
  for (i = 0; i < cm_nctx; ++i) {
    int s = cm_stt[i];
    cm_sm_upd(cm_sms[i], s, bit, 1 << PV(sms));
    cm_sm_upd(cm_smf[i], s, bit, 1 << PV(smf));
    cm_np[i]->state = cm_ns((uint8_t)s, bit);
#if defined(CM2_LAB) || CM2_dp
    if (PV(dp)) cm_np[i]->p += (((bit << 16) - bit) - cm_np[i]->p) >> PV(dp);
#endif
  }
  for (k = 0; k < cm_nmm; ++k) if (cm_mmpred[k] >= 0) cm_sm_upd(cm_mmsm[k], cm_mmsi[k], bit == cm_mmpred[k], 1 << PV(sms));
  for (k = 0; k < cm_nmix; ++k) {
    int32_t *w = cm_mw[k] + (size_t)cm_mcur[k] * cm_nin;
    err = ((bit << 12) - cm_mp[k]) * em;
    for (i = 0; i < cm_nin; ++i) {
      w[i] += (cm_st[i] * err) >> PV(mixlr);
      if (w[i] < -lim) w[i] = -lim; else if (w[i] > lim) w[i] = lim;
    }
  }
  if (cm_nmix > 1) {
    err = ((bit << 12) - cm_mixp) * em;
    for (k = 0; k <= cm_nmix; ++k) cm_w2[cm_w2s][k] += (cm_mst[k] * err) >> PV(mix2lr);
  }
  cm_apm_upd(cm_apm1, cm_a1i, cm_a1w, bit);
  cm_apm_upd(cm_apm2, cm_a2i, cm_a2w, bit);
  if (PV(aw3)) cm_apm_upd(cm_apm3, cm_a3i, cm_a3w, bit);
  if (PV(aw4)) cm_apm_upd(cm_apm4, cm_a4i, cm_a4w, bit);
  cm_c0 = (cm_c0 << 1) | bit;
  if (++cm_bpos == 8) {
    unsigned char by = (unsigned char)cm_c0;
    size_t pos;
    cm_hist[cm_pos] = by; pos = ++cm_pos;
    cm_c8 = (cm_c8 << 8) | (cm_c4 >> 24); cm_c4 = (cm_c4 << 8) | by;
    if ((by >= 'a' && by <= 'z') || (by >= 'A' && by <= 'Z') || (by >= '0' && by <= '9') || by == '_')
      cm_word = cm_word * 0x6f + by + 1;
    else cm_word = 0;
    if (!cm_crdone) cm_crdone = cm_elf_ranges(cm_hist, pos);
    {
      int orole = xr_role;
      if (cm_incode && (orole == XR_DISP || orole == XR_IMM)) {
        cm_fld = (cm_fld << 8) | by;
        cm_lastf[xr_op & 0x3ff][(xr_idx & 3) + (orole == XR_IMM ? 4 : 0)] = by;
      } else cm_fld = 0;
      if (PV(code)) {
        int j; cm_incode = 0;
        for (j = 0; j < cm_ncr; ++j) if ((long)pos >= cm_cr_lo[j] && (long)pos < cm_cr_hi[j]) { cm_incode = 1; cm_curva = cm_cr_va[j]; }
      }
      if (cm_incode) x86_step(by); else { xr_role = XR_OP; xr_idx = 0; xr_op = 0; xr_mrm = 0; xr_pfx = 0; }
    }
    for (k = 0; k < cm_nmm; ++k) {
      int L = cm_mmL[k];
      if (cm_mmlen[k] > 0 && cm_mmptr[k] < pos && cm_hist[cm_mmptr[k]] == by) { cm_mmptr[k]++; cm_mmlen[k]++; }
      else cm_mmlen[k] = 0;
      if ((long)pos >= L) {
        uint32_t h = 0, slot; int q;
        for (q = 1; q <= L; ++q) h = (h + cm_hist[pos - q] + 1) * 0x2f0b4c6du;
        slot = (h ^ h >> 15) >> (32 - PV(mmhb));
        if (cm_mmlen[k] == 0) { uint32_t cand = cm_mmh[k][slot]; if (cand != 0 && cand < pos) { cm_mmptr[k] = cand; cm_mmlen[k] = 1; } }
        cm_mmh[k][slot] = (uint32_t)pos;
      }
    }
    if (PV(ctx) >> 37 & 15) {   /* structural match: last position with the same instruction structure */
      uint32_t hs = cm_h3(0x5157, (uint32_t)(cm_xrole() << 3 | (xr_idx > 7 ? 7 : xr_idx)) << 20 | (uint32_t)xr_op << 9 | (uint32_t)xr_mrm, xr_hist & 0xffffff) >> 12;
      uint32_t cand = cm_sph[hs];
      cm_spb = cand ? cm_hist[cand] + 1 : 0;
      cm_spb2 = cand ? cm_hist[cand - 1] + 1 : 0;
      cm_sph[hs] = (uint32_t)pos;
    }
    cm_c0 = 1; cm_bpos = 0;
    cm_byte_ctx();
  }
}

/* ---- binary arithmetic coder (unchanged from cm_core.h) ---------------- */
typedef struct {
  uint32_t x1, x2, x;
  unsigned char *out; size_t outn;
  const unsigned char *in; size_t inn, inpos;
} cm_ac;
static void cm_ac_enc_init(cm_ac *a, unsigned char *out) { a->x1 = 0; a->x2 = 0xffffffffu; a->out = out; a->outn = 0; }
static void cm_ac_encode(cm_ac *a, int bit, int p12) {
  uint32_t xmid = a->x1 + (uint32_t)(((uint64_t)(a->x2 - a->x1) * (uint32_t)p12) >> 12);
  if (bit) a->x2 = xmid; else a->x1 = xmid + 1;
  while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
    a->out[a->outn++] = (unsigned char)(a->x2 >> 24);
    a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
  }
}
static size_t cm_ac_enc_flush(cm_ac *a) {
  int i;
  for (i = 24; i >= 0; i -= 8) a->out[a->outn++] = (unsigned char)(a->x1 >> i);
  return a->outn;
}
static int cm_ac_inbyte(cm_ac *a) { return a->inpos < a->inn ? a->in[a->inpos++] : 0; }
static void cm_ac_dec_init(cm_ac *a, const unsigned char *in, size_t inn) {
  int i; a->x1 = 0; a->x2 = 0xffffffffu; a->in = in; a->inn = inn; a->inpos = 0; a->x = 0;
  for (i = 0; i < 4; ++i) a->x = (a->x << 8) | (uint32_t)cm_ac_inbyte(a);
}
static int cm_ac_decode(cm_ac *a, int p12) {
  uint32_t xmid = a->x1 + (uint32_t)(((uint64_t)(a->x2 - a->x1) * (uint32_t)p12) >> 12);
  int bit = (a->x <= xmid);
  if (bit) a->x2 = xmid; else a->x1 = xmid + 1;
  while (((a->x1 ^ a->x2) & 0xff000000u) == 0) {
    a->x1 <<= 8; a->x2 = (a->x2 << 8) | 255;
    a->x = (a->x << 8) | (uint32_t)cm_ac_inbyte(a);
  }
  return bit;
}

static size_t cm_compress(const unsigned char *src, size_t n, unsigned char *out) {
  cm_ac a; size_t i; int b;
  if (!cm_model_init()) return 0;
  cm_ac_enc_init(&a, out);
  for (i = 0; i < n; ++i)
    for (b = 7; b >= 0; --b) { int bit = (src[i] >> b) & 1; cm_ac_encode(&a, bit, cm_predict()); cm_update(bit); }
  return cm_ac_enc_flush(&a);
}
/* Decodes n bytes; returns the number of input bytes consumed (K), 0 = no memory. */
static size_t cm_decompress(const unsigned char *src, size_t srcn, unsigned char *out, size_t n) {
  cm_ac a; size_t i; int b;
  if (!cm_model_init()) return 0;
  cm_ac_dec_init(&a, src, srcn);
  for (i = 0; i < n; ++i) {
    int byte = 1;
    for (b = 0; b < 8; ++b) { int bit = cm_ac_decode(&a, cm_predict()); cm_update(bit); byte = (byte << 1) | bit; }
    out[i] = (unsigned char)byte;
  }
  return a.inpos ? a.inpos : 1;
}
#endif /* CM_CORE_H */
