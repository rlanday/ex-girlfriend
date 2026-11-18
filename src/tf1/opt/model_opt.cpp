// Changes in this file: van_layer_kv(), the `gelesen` ("read") flag with its
// early exit in step() and set_gelesen(), for rows whose output is not read;
// and the mlp.down density threshold, which can follow the layer's neuron
// count instead of DMLP; the side-channel token row (proj_add,
// paket_token_row) and last_hidden(); the transformer form of the beehive
// entry (AttnRes depth mix, one K/V head for the three query heads, value
// embeddings, prior logit head); in the AttnRes form the queue of
// unread rows that runs them as one block of up to 16 rows, with one layer
// code for read and unread rows.  All additive, except that the AttnRes
// members and steps moved into a type of their own (ArRow) and, in the block
// form, the row buffers into ZRow.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/model_opt.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0,
// among them a K/V-only last layer for rows whose output is not read.
// TransformerOpt implementation. Wiring order matches SPEC section 3.3 and
// the arena stream order of arena_build.cpp exactly.
//
// Profiling (-DFX2_PROF): raw rdtsc (no lfence/rdtscp - the ~67-cycle
// serialized stamp would distort short sections; raw back-to-back stamps
// measure ~10-20 core cycles) section accumulators, one per call site,
// aggregated into the four SPEC section-9 groups at print time and
// overhead-corrected by the measured empty-section delta.

#include "../../meld.h"
#include "model_opt.h"
#include "../../cold_size.h"
#include "../../semantic_cache.h"
#include "../../port_cfg.h"
#if PORT_CMX_MEAS
#include "../../cmx_meas.h"   // research CMX_FWK_MUL (kappa of the fast-weight head), CMX_FW_BLOCKS, CMX_FW_TICKS
#include <x86intrin.h>
#endif

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include <sys/mman.h>

#include "arena_build.h"
#include "attnres_mix.h"
#include "attn.h"
#include "glue.h"
#include "kda.h"
#include "kda_math.h"
#include "qmat_dense.h"
#include "qmat_sparse.h"

// Debug-only component isolation for the naive-vs-opt cross-check (never
// defined in the production build targets): swap the fused KDA step and/or
// the attention kernel for the naive path (bit-exact vs src/model.cpp) to
// attribute divergence per component.
#if defined(FX2_XCHECK_NAIVE_KDA) || defined(FX2_XCHECK_NAIVE_ATTN)
#include "../kernels.h"
#include "../tf1_config.h"
#endif

namespace fx2 {
namespace opt {

namespace {
constexpr int V = 208, D = 192, NL = 16, DH = 64, NH = 3,
              DMLP = 1024;
constexpr int NKIMI = 12, NVAN = 4,
              NSKIP = 8;

#if !defined(FX2_XCHECK_NAIVE_ATTN)
constexpr int KVP = 1;
#else
constexpr int KVP = NH;
#endif

constexpr int WIN = ATTN_WIN, ROPE_LEN = 131072;
static_assert(WIN <= ATTN_WIN,
              "the ring cannot be wider than the scan.");
static_assert(WIN % ATTN_BLK == 0 && WIN >= 256,
              "the ring follows the block width of the scan.");
// density thresholds for the sparse column kernels (qmat.h)
constexpr int PRIOR_DENSE_NNZ =
    static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * V);  // 194

// beta projection: 3x192 fp32 matvec on xn. Bit-identical to the naive
// fx2::dot_f32(xn, w, 192) (same 8-wide fmadd chain + hsum tree).  Out of
// line: inlined, each of its call sites carried the unrolled chain.
__attribute__((noinline)) float dot192(const float* a, const float* b) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < 192; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  return hsum256v(acc);
}
}  // namespace

// ---------------------------------------------------------------------------
// profiling sites
// ---------------------------------------------------------------------------
namespace {

enum PSite : int {
  // group 0: int4 x int8 matmuls (ALL arenas)
  PS_MM_PRIOR = 0,
  PS_MM_KQKV,
  PS_MM_GUP,
  PS_MM_GDN,
  PS_MM_KOP,
  PS_MM_VQKV,
  PS_MM_VOP,
  PS_MM_MLP_UP,
  PS_MM_MLP_DOWN,
  PS_MM_UNEMB,
  // group 1: vanilla attention mechanism
  PS_AT_INSERT,
  PS_AT_STEP,
  // group 2: kimi linear attention mechanism
  PS_KD_BETA,
  PS_KD_STEP,
  // group 3: everything else
  PS_EL_PRIORQ,
  PS_EL_EMBED,
  PS_EL_RESID,
  PS_EL_NQ_KIMI,
  PS_EL_NQ_VAN,
  PS_EL_NQ_MLP,
  PS_EL_NQ_FIN,
  PS_EL_HEADPREP,
  PS_EL_OQ,
  PS_EL_HEAD,
#if 16 && defined(FX2_PROF)
  PS_ZB_RUN,
#endif
  PS_N
};

#ifdef FX2_PROF
struct PSiteInfo {
  const char* name;
  int group;
};
const PSiteInfo kPSites[PS_N] = {
    {"prior spmv (int4 cols)", 0},
    {"kimi q/k/v proj (27)", 0},
    {"gate up (fg+og, 18)", 0},
    {"gate down (fg+og, 18)", 0},
    {"kimi out proj (9)", 0},
    {"van q/k/v proj (9)", 0},
    {"van out proj (3)", 0},
    {"mlp up + relu2q (12)", 0},
    {"mlp down spmv (12)", 0},
    {"unembed (1)", 0},
    {"kv insert (3)", 1},
    {"attn step qk+sm+pv (3)", 1},
    {"beta matvec (9)", 2},
    {"kda fused step (9)", 2},
    {"prior quant+idx", 3},
    {"embed combine", 3},
    {"resid/skip/tok ops", 3},
    {"norm+quant kimi (9)", 3},
    {"norm+quant van (3)", 3},
    {"norm+quant mlp (12)", 3},
    {"final norm+quant", 3},
    {"head-prep norm/rope/q (3)", 3},
    {"attnout quant (12)", 3},
    {"softcap+softmax head", 3},
    {"zblock run (spans sites)", 3},
};
const char* kGroups[4] = {"int4xint8 matmuls", "vanilla attention mechanism",
                          "kimi (KDA) mechanism", "everything else"};

inline uint64_t prof_rdtsc() {
  unsigned lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi)::"memory");
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
uint64_t g_pacc[PS_N] = {};
uint64_t g_pcalls[PS_N] = {};
struct PScope {
  int s;
  uint64_t t0;
  explicit PScope(int s_) : s(s_), t0(prof_rdtsc()) {}
  ~PScope() {
    g_pacc[s] += prof_rdtsc() - t0;
    g_pcalls[s]++;
  }
};
#define PSCOPE(site) PScope pscope_(site)
#else
#define PSCOPE(site) \
  do {               \
  } while (0)
#endif

}  // namespace

void prof_reset() {
#ifdef FX2_PROF
  std::memset(g_pacc, 0, sizeof(g_pacc));
  std::memset(g_pcalls, 0, sizeof(g_pcalls));
#endif
}

bool prof_enabled() {
#ifdef FX2_PROF
  return true;
#else
  return false;
#endif
}

void prof_print(double tsc_to_core, long tokens) {
#ifndef FX2_PROF
  (void)tsc_to_core;
  (void)tokens;
  std::printf("prof: not compiled in (build with -DFX2_PROF)\n");
#else
  // empty-section overhead: median back-to-back raw-rdtsc delta
  const int NS = 4001;
  std::vector<uint64_t> st(NS);
  for (int i = 0; i < NS; i++) st[i] = prof_rdtsc();
  std::vector<double> dd(NS - 1);
  for (int i = 0; i + 1 < NS; i++) dd[i] = double(st[i + 1] - st[i]);
  std::nth_element(dd.begin(), dd.begin() + (NS - 1) / 2, dd.end());
  const double stamp_ticks = dd[(NS - 1) / 2];
  std::printf("\n== per-part profile (raw rdtsc, overhead-corrected by %.1f "
              "ticks/section = %.1f core cyc) ==\n",
              stamp_ticks, stamp_ticks * tsc_to_core);
  double gcyc[4] = {}, gcalls[4] = {};
  double site_cyc[PS_N];
  double total = 0;
  for (int s = 0; s < PS_N; s++) {
    double ticks = double(g_pacc[s]) - double(g_pcalls[s]) * stamp_ticks;
    if (ticks < 0) ticks = 0;
    site_cyc[s] = ticks * tsc_to_core / double(tokens > 0 ? tokens : 1);
    gcyc[kPSites[s].group] += site_cyc[s];
    gcalls[kPSites[s].group] += double(g_pcalls[s]);
    total += site_cyc[s];
  }
  std::printf("%-34s %12s %6s %9s\n", "group", "cyc/token", "%", "calls/tok");
  for (int g = 0; g < 4; g++)
    std::printf("%-34s %12.0f %5.1f%% %9.1f\n", kGroups[g], gcyc[g],
                100.0 * gcyc[g] / total,
                gcalls[g] / double(tokens > 0 ? tokens : 1));
  std::printf("%-34s %12.0f %5.1f%%\n", "TOTAL (instrumented sections)", total,
              100.0);
  std::printf("\n%-34s %12s %6s %9s\n", "  site", "cyc/token", "%",
              "calls/tok");
  for (int g = 0; g < 4; g++) {
    std::printf("  -- %s --\n", kGroups[g]);
    for (int s = 0; s < PS_N; s++) {
      if (kPSites[s].group != g) continue;
      std::printf("  %-32s %12.0f %5.1f%% %9.1f\n", kPSites[s].name,
                  site_cyc[s], 100.0 * site_cyc[s] / total,
                  double(g_pcalls[s]) / double(tokens > 0 ? tokens : 1));
    }
  }
#endif
}

struct ArRow {
  static constexpr int AR_NQ = 2 * NL + 1;
  alignas(64) float ar_x0[D];
  alignas(64) float ar_qA[AR_NQ][D];
  float ar_qM[AR_NQ], ar_qZ[AR_NQ];
  int ar_n;               // completed sources (x0 + frozen block sums)
  int ar_qnext;           // first read that has not happened yet
  alignas(64) float ar_part[D];
  float ar_part_rstd;  // 1/rms of ar_part, kept current by take_f
  float ar_part_dot;   // dot(w_q, ar_part) for q == ar_part_dot_q
  int ar_part_dot_q = -1;
  bool ar_has_part;
  alignas(64) float ar_f[D];
  static const float* qw(const OptModel& M, int q) {
    return q == 2 * NL ? M.ar_q_final : ((q & 1) ? M.ar_q_mlp[q >> 1] : M.ar_q_mix[q >> 1]);
  }
  void push_x0(const OptModel& M, const float* x);
  void push_part(const OptModel& M);
  void read(const OptModel& M, float* x);
  void take_f(const OptModel& M, int next_q);
};

// source 0 = x0: M = l, Z = 1; A_q stays x0 (lazy) until the first block sum
__attribute__((noinline)) void ArRow::push_x0(const OptModel& M, const float* x) {
  std::memcpy(ar_x0, x, sizeof(float) * D);
  const float rs = ar_rstd192(ar_x0);
  for (int q = ar_qnext; q < AR_NQ; q++) {
    ar_qM[q] = ar_dot192(qw(M, q), ar_x0) * rs;
    ar_qZ[q] = 1.0f;
  }
  ar_n = 1;
}
// freeze the partial as a completed source for all remaining reads: logits,
// then both exps 8 reads at a time, then A
__attribute__((noinline)) void ArRow::push_part(const OptModel& M) {
  alignas(32) float ea[AR_NQ + 7], eb[AR_NQ + 7];
  const int q0 = ar_qnext, nq = AR_NQ - q0;
  for (int k = 0; k < nq; k++) {
    const int q = q0 + k;
    const float l = ar_dot192(qw(M, q), ar_part) * ar_part_rstd;
    const float m2 = l > ar_qM[q] ? l : ar_qM[q];
    ea[k] = ar_qM[q] - m2;  // old-sum scale exponent (<= 0)
    eb[k] = l - m2;         // new-source exponent (<= 0)
    ar_qM[q] = m2;
  }
  for (int k = nq; k < ((nq + 7) & ~7); k++) ea[k] = eb[k] = 0.0f;
  for (int k = 0; k < nq; k += 8) {
    _mm256_store_ps(ea + k, exp256_ps(_mm256_load_ps(ea + k)));
    _mm256_store_ps(eb + k, exp256_ps(_mm256_load_ps(eb + k)));
  }
  for (int k = 0; k < nq; k++) {
    const int q = q0 + k;
    const float* Aold = ar_n == 1 ? ar_x0 : ar_qA[q];
    float* Aq = ar_qA[q];
    const __m256 va = _mm256_set1_ps(ea[k]), vb = _mm256_set1_ps(eb[k]);
    for (int d = 0; d < D; d += 8)
      _mm256_store_ps(Aq + d, _mm256_fmadd_ps(vb, _mm256_load_ps(ar_part + d),
                                              _mm256_mul_ps(va, _mm256_load_ps(Aold + d))));
    ar_qZ[q] = ea[k] * ar_qZ[q] + eb[k];
  }
  ar_n++;
  ar_has_part = false;
}
// x = depth_mix(x0, completed block sums, partial) for the next read
__attribute__((noinline)) void ArRow::read(const OptModel& M, float* x) {
  const int q = ar_qnext++;
  const float* Aq = ar_n == 1 ? ar_x0 : ar_qA[q];
  if (!ar_has_part) {  // completed sources only: out = A / Z
    if (ar_n == 1) {   // x0 alone: exactly x0
      for (int d = 0; d < D; d += 8) _mm256_store_ps(x + d, _mm256_load_ps(Aq + d));
    } else {
      const __m256 r = _mm256_set1_ps(1.0f / ar_qZ[q]);
      for (int d = 0; d < D; d += 8) _mm256_store_ps(x + d, _mm256_mul_ps(_mm256_load_ps(Aq + d), r));
    }
  } else {
    const float lp = (ar_part_dot_q == q ? ar_part_dot : ar_dot192(qw(M, q), ar_part)) * ar_part_rstd;
    const float m = lp > ar_qM[q] ? lp : ar_qM[q];
    alignas(32) float ex[8] = {ar_qM[q] - m, lp - m, 0, 0, 0, 0, 0, 0};
    _mm256_store_ps(ex, exp256_ps(_mm256_load_ps(ex)));
    const float den = ex[0] * ar_qZ[q] + ex[1];
    const __m256 ra = _mm256_set1_ps(ex[0] / den), rb = _mm256_set1_ps(ex[1] / den);
    for (int d = 0; d < D; d += 8)
      _mm256_store_ps(x + d, _mm256_fmadd_ps(rb, _mm256_load_ps(ar_part + d),
                                             _mm256_mul_ps(ra, _mm256_load_ps(Aq + d))));
  }
}
// partial += f (the block's running sum), f = the sub-layer output;
// next_q: the read that uses this partial next (its logit's dot is fused
// into the sum), -1 when a block boundary freezes the partial first
__attribute__((noinline)) void ArRow::take_f(const OptModel& M, int next_q) {
  if (ar_has_part && next_q >= 0) {
    ar_part_rstd = ar_add_rstd_dot192(ar_part, ar_f, qw(M, next_q), &ar_part_dot);
    ar_part_dot_q = next_q;
    return;
  }
  ar_part_dot_q = -1;
  if (ar_has_part) {
    ar_part_rstd = ar_add_rstd192(ar_part, ar_f);
  } else {
    std::memcpy(ar_part, ar_f, sizeof(float) * D);
    ar_part_rstd = ar_rstd192(ar_part);
    ar_has_part = true;
  }
}

struct ZRow : ArRow {
  alignas(64) float x[D];
  alignas(64) float xn[D];
  alignas(64) float a[D];
  alignas(64) float b[D];
  alignas(64) float c[D];
  alignas(64) float graw[D];
  alignas(64) float og192[D];
  alignas(64) float gn[D];
  alignas(64) uint8_t q5[5 * D];  // up to 5 biased-u8 copies of xn
  alignas(64) uint8_t q64b[DH];
  alignas(64) uint8_t q768[DMLP];
  alignas(64) uint16_t idx768[DMLP + 16];
  alignas(64) float prior_f32[208];
  alignas(32) float braw[8];
  int nnz;
  int tok;
  PaketChannels pk;
};
struct TransformerOptImpl : ZRow {
  OptModel M;
  AttnKind kind;

  // streaming state (hugepage pool: KDA states, then KV caches)
  HugeBuf spool;
  KdaState* kst;   // [9]
  AttnKVF32* kvf;  // [3] iff kind == KVF32
  AttnKV* kvi;     // [3] iff kind == KVI8
  int64_t t;
  int64_t rope_off;

#ifdef FX2_XCHECK_NAIVE_KDA
  struct NaiveKst {
    alignas(64) float S[NH][DH * DH];
    alignas(64) float hist[3][3][D];
  };
  std::vector<NaiveKst> nkst = std::vector<NaiveKst>(NKIMI);
#endif
#ifdef FX2_XCHECK_NAIVE_ATTN
  struct NaiveVst {
    alignas(64) int8_t kring[WIN][D];
    alignas(64) int8_t vring[WIN][D];
  };
  std::vector<NaiveVst> nvst = std::vector<NaiveVst>(3);
#endif

  // per-token skip stack (written by layers 0..5, read by 6..11 of the SAME
  // token) - transient within step
  alignas(64) float skip_store[NSKIP][D];

  // scratch
  alignas(64) float yb[D];
  // AttnRes replace mode (Kimi Team, "Attention Residuals", arXiv
  // 2603.15031; engine implementation adapted from cmix-neif-beehive,
  // Vladimir Malinovskii, 2026, 92c07c0b, model_opt.cpp, depth-mix form
  // TF_ARV2 = 5).  Sources of a read: x0, the completed block sums and the
  // running partial sum of the current block.  `acc` is where a sub-layer
  // adds its output: ar_f inside the layer loop.
  // Online softmax per future read q over the completed sources, kept at push
  // time: M_q = max logit, Z_q = sum exp(l - M_q), A_q = sum exp(l - M_q) v.
  // A read only adds the partial: m = max(M_q, l_p), a = exp(M_q - m),
  // b = exp(l_p - m), out = (a A_q + b part) / (a Z_q + b).  Read index q:
  // 2l = mixer read of layer l, 2l+1 = its MLP read, 2 NL = the final read.
  static constexpr int ARB = 2;
  const float* ar_qw(int q) const { return ArRow::qw(M, q); }
  void ar_push_x0() { push_x0(M, x); }
  void ar_push_part() { push_part(M); }
  void ar_read() { read(M, x); }
  void ar_take_f(int next_q) { take_f(M, next_q); }
  // key of the semantic cache: staged during the step (in the AttnRes form
  // the MLP read of the last layer, else the residual behind layer NL-2), and
  // the key of the last read row
  alignas(64) float sc_l2_stage[D];
  alignas(64) float sc_key[D];
  alignas(64) float pre[D];   // the row buffers are in ZRow
  alignas(64) int8_t qq[D], kk[D], vv[D];
  alignas(64) uint8_t q8p[224];  // V rounded to 32
  alignas(64) uint16_t idxp[224 + 16];
  alignas(64) float logits[208];
  alignas(64) float probs208[208];
  alignas(32) float sfall[32], cfall[32];  // rope fallback row
  // The assembled token row with side channels.  `tok_paket` is the normed
  // sum, `pk` the channels of the row that step() is about to consume.
  alignas(64) float tok_paket[D];
  // fast-weight state, taken over from ex-girlfriend 1.0 (Ryan
  // Landay, GPL-3.0-only; model_opt.cpp FW-DUAL + FW-ROW + FW-RULES).  One dW
  // per block for the last FWK MLP blocks (block b = layer NL-FWK+b), column
  // per code index: dW_b[(b*DMLP + i)*D + j]; zero-initialised, never reset.
  // All blocks feed the ONE second head on x + sum_b dW_b h_b.
  // dW lives in the state pool behind the KDA states and the
  // KV rings (64-byte aligned columns, huge pages, zeroed by the pool).  The
  // dW step of update u is not applied at once: it stays pending PER COLUMN
  // until that column is read again (fw_apply) or until the slot of gx_u in
  // the ring of the last FWR updates is reused.  A column read in a row has no
  // pending step when that row's update arrives (the read applied it), so every
  // column carries at most one and its steps keep their order; nothing but
  // fw_apply reads dW.  Every element gets exactly the arithmetic of the
  // immediate step (col - c*gx, then dx + col*h).
#if PORT_CMX_MEAS
  // research CMX_FW_BLOCKS (F8): FWK is the run-time number of blocks (release 4), arrays sized for 12
  static constexpr int FWK_MAX = 12;
  static_assert(FWK_MAX >= 1 && FWK_MAX <= NL, "more blocks than layers");
  int FWK = 4;
#define CMX_FWK_ARR FWK_MAX
#else
  static constexpr int FWK = 4;
  static_assert(FWK >= 1 && FWK <= NL, "more blocks than layers");
#define CMX_FWK_ARR FWK
#endif
  static constexpr int FWR = 64;   // ring of the last gx vectors (power of two)
  // Only the FW_TOPK largest codes of a block take part in forward and
  // update, and the mean RMS msd of a block is renewed every FW_MSD-th
  // update (v itself every update).
  static constexpr int FW_TOPK = 32;
  static constexpr int FW_MSD = 16;
  float* fw_dW;
  int fw_nnz[CMX_FWK_ARR];
  uint16_t fw_idx[CMX_FWK_ARR][DMLP];
  alignas(64) float fw_h[CMX_FWK_ARR][DMLP];
  bool fw_gel;     // the current step() is a read row (captures h); true from the ctor
  float fw_denom;
  uint64_t fw_t;      // update counter of the step schedule
  alignas(64) float fw_v[CMX_FWK_ARR][DMLP];   // EMA of h_i^2 per block
  alignas(64) float fw_sq[CMX_FWK_ARR][DMLP];  // sqrt(fw_v) of the last msd renewal
  float fw_sh[CMX_FWK_ARR][DMLP];
  alignas(64) float fw_gxr[FWR][D];    // gx of the last FWR updates
  alignas(64) int32_t fw_pt[CMX_FWK_ARR][DMLP];     // update whose step is pending, or -1
  float fw_pc[CMX_FWK_ARR][DMLP];              // its coefficient step * sh
  int32_t fw_u;                         // updates so far
  float fw_msd[CMX_FWK_ARR];                   // msd of the last renewal
  uint16_t fw_sel[DMLP];                    // the FW_TOPK codes of a block
  alignas(64) float fw_dx[D], fw_x2[D], fw_xn[D];
  alignas(64) uint8_t fw_q[256];
  alignas(64) float fw_logits[208],
                    fw_probs208[208];

  // The object (0.9 MB, members aligned to 64) comes from
  // MapNull (arena_build.h): page aligned and zero-filled, so neither the
  // aligned new/delete of the library nor member initialisers are needed;
  // the non-zero starts are set in the constructor.
  static void* operator new(size_t n, std::align_val_t) { return MapNull(n); }
  static void operator delete(void* p, size_t n, std::align_val_t) { ::munmap(p, n); }

  COLD_SIZE_ATTR explicit TransformerOptImpl(const char* path, AttnKind k) : kind(k) {
    gelesen = true;
    fw_gel = true;
#if PORT_CMX_MEAS
    FWK = NL - cmx_meas::g_fw_first;   // research CMX_FW_BLOCKS: blocks g_fw_first .. NL-1
#endif
    M.load(path);
    const size_t kda_sz = sizeof(KdaState) * NKIMI;
    const size_t kv_sz =
        (kind == AttnKind::KVF32 ? sizeof(AttnKVF32) : sizeof(AttnKV)) * NVAN;
    // dW behind the states (kda_sz and kv_sz are multiples of 64)
    const size_t fw_sz = sizeof(float) * size_t(FWK) * DMLP * D;
    spool.alloc(kda_sz + kv_sz + fw_sz);
    spool.trim(kda_sz + kv_sz + fw_sz);
    fw_dW = reinterpret_cast<float*>(spool.p + kda_sz + kv_sz);
    for (int b = 0; b < FWK; b++)
      for (int j = 0; j < DMLP; j++) fw_pt[b][j] = -1;
    kst = reinterpret_cast<KdaState*>(spool.p);
    if (kind == AttnKind::KVF32)
      kvf = reinterpret_cast<AttnKVF32*>(spool.p + kda_sz);
    else
      kvi = reinterpret_cast<AttnKV*>(spool.p + kda_sz);
    for (int i = 0; i < NKIMI; i++) kda_layer_reset(kst[i]);
  }

  void begin(int64_t rope_position_offset) {
    zb_run();
    for (int i = 0; i < NKIMI; i++) kda_layer_reset(kst[i]);
#ifdef FX2_XCHECK_NAIVE_KDA
    std::memset(nkst.data(), 0, sizeof(NaiveKst) * NKIMI);
#endif
    t = 0;  // invalidates the KV rings (validity is [0, t] within an article)
    rope_off = rope_position_offset;
  }

  static constexpr ptrdiff_t ZS = static_cast<ptrdiff_t>(sizeof(ZRow));
  // a matmul over the rows: one row (a read row) on the single-row kernel,
  // several fused
  static void mm_f32(const QDense& m, const uint8_t* act, float* out, int n) {
    if (n == 1)
      qgemv_f32(m, act, out);
    else
      qgemm(m, act, ZS, out, ZS, n, false);
  }
  static void mm_add(const QDense& m, const uint8_t* act, float* out, int n) {
    if (n == 1)
      qgemv_add(m, act, out);
    else
      qgemm(m, act, ZS, out, ZS, n, true);
  }
  static void mm_quant_bias(const QDense& m, const uint8_t* act, float s, uint8_t* q, int n) {
    if (n == 1)
      qgemv_quant_bias(m, act, s, q);
    else
      qgemm_quant_bias(m, act, ZS, s, q, ZS, n);
  }

  // the rope rows of position pos; the libm fallback beyond the shipped table
  // out of line
  __attribute__((noinline)) void zb_rope_fall(int64_t pos) {
    const float fpos = static_cast<float>(pos);
#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 32; i++) {
      const float ang = fpos * M.inv_freq[i];
      sfall[i] = std::sin(ang);
      cfall[i] = std::cos(ang);
    }
  }
  void zb_rope(int64_t pos, const float*& sp, const float*& cp) {
    if (pos < ROPE_LEN) {
      sp = M.rope_sin + size_t(pos) * 32;
      cp = M.rope_cos + size_t(pos) * 32;
    } else {  // beyond the shipped table: libm fallback (same as naive)
      zb_rope_fall(pos);
      sp = sfall;
      cp = cfall;
    }
  }

  // after kv_expand of cmix-neif-beehive: k and v of row r come
  // out of the projections with KVH heads; v gets c0 v + c1 VE[token row],
  // then the K/V heads are repeated for the NH query heads, so the kernels and
  // the KV ring stay as they are.
  void kv_expand(const VanArenas& L, ZRow& r) {
    float* kf = r.b;
    float* vf = r.c;
    {
      constexpr int W = 1 * DH;
      int row = r.tok;
      if (row >= V) row = 1 == 1 ? V - 1 : (1 == 2 ? row - V : -1);
      if (row >= 0) {
        const float* e = L.ve + size_t(row) * W;
        for (int i = 0; i < W; i++) vf[i] = L.ve_c0 * vf[i] + L.ve_c1 * e[i];
      } else {
        for (int i = 0; i < W; i++) vf[i] = L.ve_c0 * vf[i];
      }
    }
    for (int h = 1; h < KVP; h++) {
      std::memcpy(kf + h * DH, kf + (h % 1) * DH, sizeof(float) * DH);
      std::memcpy(vf + h * DH, vf + (h % 1) * DH, sizeof(float) * DH);
    }
  }

  // kimi (KDA) layer ki for the rows r[0, n); the output goes to each row's ar_f
  void kimi_layer(int ki, ZRow* r, int n) {
    KimiArenas& L = M.kimi[ki];
    {
      PSCOPE(PS_EL_NQ_KIMI);
      const float s5[5] = {L.qp.s_act, L.kp.s_act, L.vp.s_act, L.fg_up.s_act,
                           L.og_up.s_act};
      for (int j = 0; j < n; j++) rms_norm_quant192_multi(r[j].x, r[j].xn, s5, 5, r[j].q5);
    }
    {
      PSCOPE(PS_MM_KQKV);
      mm_f32(L.qp.m, r->q5, r->a, n);
      mm_f32(L.kp.m, r->q5 + D, r->b, n);
      mm_f32(L.vp.m, r->q5 + 2 * D, r->c, n);
    }
    {
      PSCOPE(PS_MM_GUP);
      mm_quant_bias(L.fg_up.m, r->q5 + 3 * D, L.fg_down.s_act, r->q64b, n);
    }
    {
      PSCOPE(PS_MM_GDN);
      mm_f32(L.fg_down.m, r->q64b, r->graw, n);
    }
    {
      PSCOPE(PS_MM_GUP);
      mm_quant_bias(L.og_up.m, r->q5 + 4 * D, L.og_down.s_act, r->q64b, n);
    }
    {
      PSCOPE(PS_MM_GDN);
      mm_f32(L.og_down.m, r->q64b, r->og192, n);
    }
    {
      PSCOPE(PS_KD_BETA);
      for (int j = 0; j < n; j++)
        for (int h = 0; h < NH; h++)
          r[j].braw[h] = dot192(r[j].xn, L.beta_w + size_t(h) * D);
    }
    {
      PSCOPE(PS_KD_STEP);
      for (int j = 0; j < n; j++)   // the recurrence: rows in stream order
        kda_layer_step(L.kw, kst[ki], r[j].a, r[j].b, r[j].c, r[j].graw, r[j].og192, r[j].braw,
                       r[j].gn);
    }
    {
      PSCOPE(PS_EL_OQ);
      for (int j = 0; j < n; j++) quant192_u8(r[j].gn, L.op.s_act, r[j].q5);
    }
    {
      PSCOPE(PS_MM_KOP);
      mm_add(L.op.m, r->q5, r->ar_f, n);
    }
  }

  // window layer vi for the rows r[0, n) at positions t0 + j.  kv_only: an
  // unread row in the last layer, which needs only its K and V in the ring
  // (everything after the insert only feeds the output head).
  void van_layer(int vi, ZRow* r, int n, bool kv_only, int64_t t0) {
    VanArenas& L = M.van[vi];
    {
      PSCOPE(PS_EL_NQ_VAN);
      const float s3[3] = {L.qp.s_act, L.kp.s_act, L.vp.s_act};
      for (int j = 0; j < n; j++)
        rms_norm_quant192_x3(r[j].x, r[j].xn, s3, r[j].q5, r[j].q5 + D, r[j].q5 + 2 * D);
    }
    {
      PSCOPE(PS_MM_VQKV);
      if (!kv_only) mm_f32(L.qp.m, r->q5, r->a, n);
      mm_f32(L.kp.m, r->q5 + D, r->b, n);
      mm_f32(L.vp.m, r->q5 + 2 * D, r->c, n);
    }
    for (int j = 0; j < n; j++)   // KV ring and window scan: rows in stream order
      van_row(vi, r[j], t0 + j, kv_only);
    if (!kv_only) {
      PSCOPE(PS_MM_VOP);
      mm_add(L.op.m, r->q5, r->ar_f, n);
    }
  }

  // the part of a window layer that runs row by row: head prep, KV insert
  // and window scan of row w at position tj
  __attribute__((noinline)) void van_row(int vi, ZRow& w, int64_t tj, bool kv_only) {
    VanArenas& L = M.van[vi];
    float* qf = w.a;
    float* kf = w.b;
    float* vf = w.c;
    kv_expand(L, w);
    {
      PSCOPE(PS_EL_HEADPREP);
      if (!kv_only)
        for (int h = 0; h < NH; h++) rms_norm64(qf + h * DH, qf + h * DH);
      for (int h = 0; h < KVP; h++) rms_norm64(kf + h * DH, kf + h * DH);
      const float *sp, *cp;
      zb_rope(rope_off + tj, sp, cp);
      if (!kv_only) rope_apply_192(qf, sp, cp);
      if constexpr (KVP == 1) {
        rope_apply_64(kf, sp, cp);
        if (!kv_only)
          for (int h = 0; h < NH; h++) quant64_i8(qf + h * DH, L.sq[h], qq + h * DH);
        quant64_i8(kf, L.sk[0], kk);
        quant64_i8(vf, L.sv[0], vv);
      } else {
        rope_apply_192(kf, sp, cp);
        for (int h = 0; h < NH; h++) {
          if (!kv_only) quant64_i8(qf + h * DH, L.sq[h], qq + h * DH);
          quant64_i8(kf + h * DH, L.sk[h], kk + h * DH);
          quant64_i8(vf + h * DH, L.sv[h], vv + h * DH);
        }
      }
    }
    const int slot = static_cast<int>(tj % WIN);
    const int64_t nn = tj + 1;
    if (!((3 & 2) && 1) && kind == AttnKind::KVF32) {
      {
        PSCOPE(PS_AT_INSERT);
        attn_kv_insert(kvf[vi], slot, kk, vv);
      }
      if (kv_only) return;
      PSCOPE(PS_AT_STEP);
      if (nn < WIN)
        attn_step_var(kvf[vi], qq, L.coef, L.sv, static_cast<int>(nn), pre, 0.0f);
      else
        attn_step_fixed(kvf[vi], qq, L.coef, L.sv, pre, 0.0f);
    } else {
      {
        PSCOPE(PS_AT_INSERT);
        attn_kv_insert(kvi[vi], slot, kk, vv);
      }
      if (kv_only) return;
      PSCOPE(PS_AT_STEP);
      if (nn < WIN)
        attn_step_var(kvi[vi], qq, L.coef, L.sv, static_cast<int>(nn), pre, 0.0f);
      else
        attn_step_fixed(kvi[vi], qq, L.coef, L.sv, pre, 0.0f);
    }
    PSCOPE(PS_EL_OQ);
    quant192_u8(pre, L.op.s_act, w.q5);
  }

  // the MLP of layer l for the rows r[0, n)
  void mlp_block(int l, ZRow* r, int n) {
    MlpArenas& mm = M.mlp[l];
    {
      PSCOPE(PS_EL_NQ_MLP);
      for (int j = 0; j < n; j++) rms_norm_quant192(r[j].x, r[j].xn, mm.up.s_act, r[j].q5);
    }
    {
      PSCOPE(PS_MM_MLP_UP);
      if (n == 1)
        r->nnz = qgemv_relu2q(mm.up.m, r->q5, mm.down_s_act, r->q768, r->idx768);
      else
        qgemm_relu2q(mm.up.m, r->q5, ZS, mm.down_s_act, r->q768, r->idx768, &r->nnz, ZS, n);
    }
    {
      PSCOPE(PS_MM_MLP_DOWN);
      for (int j = 0; j < n; j++) {
        ZRow& w = r[j];
        if (w.nnz > static_cast<int>(QMAT_SPARSE_DENSITY_THRESHOLD * mm.down.d_in))
          qsparse4_dense_add(mm.down, w.q768, w.ar_f);
        else
          qsparse4_add(mm.down, w.q768, w.idx768, w.nnz, w.ar_f);
      }
    }
    // The relu^2 code of the last FWK MLPs, on read rows only (fw_gel; then
    // r is this object, n = 1, and fw_apply reads its q768 and idx768).
#if PORT_CMX_MEAS
    if (fw_gel && l >= NL - FWK && !cmx_meas::g_fwoff) {   // research CMX_FWOFF: no fast-weight forward
      const unsigned long long cmx_t = cmx_meas::g_fw_ticks ? __rdtsc() : 0ull;   // research CMX_FW_TICKS
      fw_apply(l - (NL - FWK), r->nnz, mm.down_s_act);
      if (cmx_meas::g_fw_ticks) cmx_meas::g_fw_cyc_apply += __rdtsc() - cmx_t;
    }
#else
    if (fw_gel && l >= NL - FWK)
      fw_apply(l - (NL - FWK), r->nnz, mm.down_s_act);
#endif
  }

  // the embedding of row r (token r.tok, prior row r.prior_f32, side channels
  // pk) and source 0 of its depth mix
  __attribute__((noinline)) void zb_embed(ZRow& r) {
    const float* tok_row = M.tok_table + size_t(r.tok) * D;
    if (M.paket.present) tok_row = paket_token_row(r.tok);
    int nnzp;
    {
      PSCOPE(PS_EL_PRIORQ);
      prior_quant_raw(r.prior_f32, M.prior_s_act, q8p);  // raw u8; q8p[208..224)=0
      nnzp = qsparse_make_idx(q8p, V, idxp);
      const uint16_t fillv = nnzp ? idxp[nnzp - 1] : 0;
      for (int k = nnzp; k < nnzp + 8; k++) idxp[k] = fillv;
    }
    {
      PSCOPE(PS_MM_PRIOR);
      if (nnzp > PRIOR_DENSE_NNZ)
        qsparse4_dense_f32(M.prior, q8p, yb);
      else
        qsparse4_f32(M.prior, q8p, idxp, nnzp, yb);
    }
    {
      PSCOPE(PS_EL_EMBED);
      embed_combine192(tok_row, yb, r.x);  // x0 = tok_row + rms_norm(prior_y)
    }
    r.ar_qnext = 0;
    r.ar_has_part = false;
    r.ar_part_dot_q = -1;
    r.push_x0(M, r.x);                     // source 0 = x0
  }

  // AttnRes replace mode (after model_opt.cpp of cmix-neif-beehive).
  // No U-Net skips, no residual/token coefficients: every sub-layer reads
  // h = depth_mix(x0, completed block sums, partial) and its output f goes
  // into the partial sum of the current block.  Layers [l0, l1) of the rows
  // r[0, n) at positions t0 + j; kv_last: unread rows, whose last layer only
  // writes K and V.
  __attribute__((noinline)) void zb_layers(ZRow* r, int n, bool kv_last, int l0, int l1,
                                           int64_t t0) {
    for (int l = l0; l < l1; l++) {
      {
        PSCOPE(PS_EL_RESID);
        for (int j = 0; j < n; j++) {
          if (l % ARB == 0 && r[j].ar_has_part)   // block boundary: freeze its sum
            r[j].push_part(M);
          r[j].read(M, r[j].x);                  // x = h for the mixer (q 2l, M.ar_q_mix[l])
        }
      }
#if PORT_CMX_MEAS
      // research CMX_SEMKEYS / CMX_SEMKEY2: the extra semantic caches' keys, the mixer read of layer l (q 2l) of a read
      // row for every layer of the list (the only call with kv_last false is step()'s, one row: this object)
      static_assert(D == 192 && NL <= 16, "CMX_SEMKEYS key buffers");
      if (__builtin_expect((cmx_meas::g_sk_lmask >> l) & 1u, 0) && !kv_last)
        std::memcpy(cmx_meas::g_sk_key[l], r->x, sizeof(float) * D);
#endif
      if (kv_last && l == NL - 1) {              // unread rows: K/V of the last layer only
        van_layer(M.layer2van[l], r, n, true, t0);
        return;
      }
      for (int j = 0; j < n; j++) std::memset(r[j].ar_f, 0, sizeof(r[j].ar_f));
      if (OptModel::KIMI_L[l])
        kimi_layer(M.layer2kimi[l], r, n);
      else
        van_layer(M.layer2van[l], r, n, false, t0);
      {
        PSCOPE(PS_EL_RESID);
        for (int j = 0; j < n; j++) {
          r[j].take_f(M, 2 * l + 1);
          r[j].read(M, r[j].x);                  // x = h for the MLP (q 2l+1, M.ar_q_mlp[l])
        }
      }
      // semantic-cache key: the MLP read of the last layer (q 2 NL - 1);
      // only a read row gets here
      if (l == NL - 1) std::memcpy(sc_l2_stage, r->x, sizeof(float) * D);
      for (int j = 0; j < n; j++) std::memset(r[j].ar_f, 0, sizeof(r[j].ar_f));
      mlp_block(l, r, n);
      {
        PSCOPE(PS_EL_RESID);
        // next read of this partial: the final read after the last layer,
        // else layer l+1's mixer read unless a block boundary freezes it first
        const int nq = l == NL - 1 ? 2 * NL : ((l + 1) % ARB != 0 ? 2 * l + 2 : -1);
        for (int j = 0; j < n; j++) r[j].take_f(M, nq);
      }
    }
  }

  // The queue of unread rows (ZB rows): step() with an unread row only queues
  // it; the queue runs when the next read row arrives, at begin_article() or
  // when it is full.  What the rows leave (KV rings, KDA states, t) is written
  // before anything reads it; xn holds the last row's, as after an unread step.
  static constexpr int ZB = 16;
  ZRow zr[ZB];
  int zn;

  // The two functions below only call other functions; compiled for size.
  __attribute__((minsize)) void zb_push(uint8_t token) {
    ZRow& r = zr[zn];
    r.tok = token;
    std::memcpy(r.prior_f32, prior_f32, sizeof(r.prior_f32));
    r.pk = pk;
    if (++zn == ZB) zb_run();
  }

  __attribute__((minsize, noinline)) void zb_run() {
    const int n = zn;
    if (n == 0) return;
    zn = 0;
#ifdef FX2_PROF
    PSCOPE(PS_ZB_RUN);
#endif
    fw_gel = false;   // unread rows capture no MLP code
    const PaketChannels pk_keep = pk;   // paket_token_row() reads pk
    for (int j = 0; j < n; j++) {
      pk = zr[j].pk;
      zb_embed(zr[j]);
    }
    pk = pk_keep;
    zb_layers(zr, n, true, 0, NL, t);
    std::memcpy(xn, zr[n - 1].xn, sizeof(float) * D);
    t += n;
  }

  // Only the FW_TOPK largest codes of a block take part
  // (forward and update; ties by index), in ascending index order.
  const uint16_t* fw_topk(int& nnz) {
    constexpr int K = FW_TOPK;
    if (nnz <= K) return idx768;
    int hist[256] = {};
    for (int k = 0; k < nnz; k++) hist[q768[idx768[k]]]++;
    int cnt = 0, t = 255;
    while (t > 0 && cnt + hist[t] <= K) cnt += hist[t--];
    int room = K - cnt, m = 0;
    for (int k = 0; k < nnz; k++) {
      const int v = q768[idx768[k]];
      if (v > t || (v == t && room-- > 0)) fw_sel[m++] = idx768[k];
    }
    nnz = m;
    return fw_sel;
  }

  // One column's pending dW step, col -= c * gx.
  __attribute__((noinline)) void fw_col_step(float* col, float c, const float* gx) {
    const __m256 vc = _mm256_set1_ps(c);
    for (int j = 0; j < D; j += 8)
      _mm256_store_ps(col + j, _mm256_sub_ps(_mm256_load_ps(col + j),
                                             _mm256_mul_ps(vc, _mm256_load_ps(gx + j))));
  }

  // after ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only), fast-weight
  // forward of block b.  h_k = q768[i_k] * s over the nonzero codes of this
  // MLP; their ascending index list is the one the up
  // projection just wrote (idx768, nnz; rows past d_out are zero there),
  // cut to the FW_TOPK largest (fw_topk).
  // x stays untouched, the delta goes to fw_dx; block 0 runs first in every
  // read row and starts it from zero.  64 columns at a time with the
  // accumulators in registers; a column with a pending dW step gets it first.
  __attribute__((noinline)) void fw_apply(int b, int nnz, float s) {
    const uint16_t* const lst = fw_topk(nnz);
    float* const h = fw_h[b];
    for (int k = 0; k < nnz; k++) h[k] = float(q768[lst[k]]) * s;
    float* const dWb = fw_dW + size_t(b) * DMLP * D;
    int32_t* const pt = fw_pt[b];
    const float* const pc = fw_pc[b];
    for (int jc = 0; jc < D; jc += 64) {
      __m256 a[8];
      for (int u = 0; u < 8; u++)
        a[u] = b == 0 ? _mm256_setzero_ps() : _mm256_load_ps(fw_dx + jc + 8 * u);
      for (int k = 0; k < nnz; k++) {
        const int j = lst[k];
        float* col = dWb + size_t(j) * D + jc;
        const __m256 vh = _mm256_set1_ps(h[k]);
        const int32_t t = pt[j];
        if (t >= 0) {   // the pending step, then the read
          const float* g = fw_gxr[t & (FWR - 1)] + jc;
          const __m256 vc = _mm256_set1_ps(pc[j]);
          for (int u = 0; u < 8; u++) {
            const __m256 w = _mm256_sub_ps(_mm256_load_ps(col + 8 * u),
                                           _mm256_mul_ps(vc, _mm256_load_ps(g + 8 * u)));
            _mm256_store_ps(col + 8 * u, w);
            a[u] = _mm256_add_ps(a[u], _mm256_mul_ps(w, vh));
          }
        } else {
          for (int u = 0; u < 8; u++)
            a[u] = _mm256_add_ps(a[u], _mm256_mul_ps(_mm256_load_ps(col + 8 * u), vh));
        }
      }
      for (int u = 0; u < 8; u++) _mm256_store_ps(fw_dx + jc + 8 * u, a[u]);
    }
    for (int k = 0; k < nnz; k++) pt[lst[k]] = -1;
    std::memcpy(fw_idx[b], lst, sizeof(uint16_t) * size_t(nnz));
    fw_nnz[b] = nnz;
  }

  // The second head from x + dW h into its own buffers; the static head
  // (xn, q5, logits, probs208, last_hidden()) stays untouched.
  __attribute__((noinline)) void fw_head() {
    for (int i = 0; i < D; i++) fw_x2[i] = x[i] + fw_dx[i];
    rms_norm_quant192(fw_x2, fw_xn, M.unembed.s_act, fw_q);
    float sx = 0.0f, sn = 0.0f;
    for (int i = 0; i < D; i++) {
      sx += fw_x2[i] * fw_x2[i];
      sn += fw_xn[i] * fw_xn[i];
    }
    fw_denom = sx > 0.0f ? std::sqrt(sn / sx) : 0.0f;
    qgemv_f32(M.unembed.m, fw_q, fw_logits);
    alignas(64) float raw[208];
    std::memcpy(raw, fw_logits, sizeof(raw));
    head_softcap_softmax(fw_logits, fw_probs208);  // fw_logits -> 15 tanh(l/15)
    head_plm_softcap_softmax(raw, prior_f32, dot192(M.plm_w, fw_xn), fw_probs208);
  }

  // Backward and update on the last read row's second-head prediction with
  // target y: cross-entropy gradient on the soft-capped logits, back through
  // the frozen unembedding (float copy) and the RMS norm to x, then one NLMS
  // step on all dW_b with per-feature RMS preconditioning and the step
  // schedule kappa_t = 1.5e-3 + 4.5e-3 / (1 + t / 8000).
  __attribute__((noinline)) void fw_update(uint8_t y) {
    // The ring slot of this update; steps of update u - FWR
    // that are still pending need its gx, so they go now.
    const int32_t u = fw_u++;
    float* const gx = fw_gxr[u & (FWR - 1)];
    if (u >= FWR) {
      const __m256i vo = _mm256_set1_epi32(u - FWR);
      for (int b = 0; b < FWK; b++)
        for (int j0 = 0; j0 < DMLP; j0 += 8) {
          unsigned mk = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
              _mm256_load_si256(reinterpret_cast<const __m256i*>(&fw_pt[b][j0])), vo))));
          while (mk) {
            const int j = j0 + __builtin_ctz(mk);
            mk &= mk - 1;
            fw_col_step(fw_dW + (size_t(b) * DMLP + j) * D, fw_pc[b][j], gx);
            fw_pt[b][j] = -1;
          }
        }
    }
    alignas(64) float dl[V];
    for (int o = 0; o < V; o++) {
      const float d = fw_probs208[o] - (o == int(y) ? 1.0f : 0.0f);
      const float c = fw_logits[o] * (1.0f / 15.0f);
      dl[o] = d * (1.0f - c * c);
    }
    // per-feature RMS preconditioning with per-block statistics (msd = mean
    // RMS over the block's DMLP features); one NLMS step over the joined code
    // of all blocks, joint denominator 1 + sum_b h_b . sh_b.
    // msd is renewed every FW_MSD-th update only (in between
    // the last one); then the square roots once per feature (vectorised,
    // exact), and the FWK sums of msd are independent chains, each in its own
    // order i = 0..DMLP-1, interleaved with the gxn loop below, which hides
    // their latency.
    const float fb = 999000.0e-6f;
    const bool msd_neu = fw_t % FW_MSD == 0;
    for (int b = 0; b < FWK; b++) {
      float* const v = fw_v[b];
      const uint16_t* const idx = fw_idx[b];
      const float* const h = fw_h[b];
      const int nnz = fw_nnz[b];
      for (int i = 0; i < DMLP; i++) v[i] *= fb;
      for (int k = 0; k < nnz; k++) v[idx[k]] += (1.0f - fb) * h[k] * h[k];
      if (msd_neu)
        for (int i = 0; i < DMLP; i++) fw_sq[b][i] = std::sqrt(v[i]);
    }
    float msd[CMX_FWK_ARR];
    for (int b = 0; b < FWK; b++) msd[b] = 0.0f;
    int ii = msd_neu ? 0 : DMLP;   // no renewal: the chains stay off
    // gxn = U^T dl, 64 columns at a time with the accumulators in registers;
    // every gxn[i] sums over o in the original order.
    static_assert(DMLP % 2 == 0 && (D / 64) * V * 2 >= DMLP, "msd chains fit the gxn loop");
    alignas(64) float gxn[D];
    const float* const U = M.unembed_f32;
    for (int jc = 0; jc < D; jc += 64) {
      __m256 a[8];
      for (int k = 0; k < 8; k++) a[k] = _mm256_setzero_ps();
      for (int o = 0; o < V; o++) {
        const float* r = U + size_t(o) * D + jc;
        const __m256 vg = _mm256_set1_ps(dl[o]);
        for (int k = 0; k < 8; k++)
          a[k] = _mm256_add_ps(a[k], _mm256_mul_ps(_mm256_loadu_ps(r + 8 * k), vg));
        if (ii < DMLP) {   // two steps of every msd chain
          for (int b = 0; b < FWK; b++) msd[b] += fw_sq[b][ii];
          for (int b = 0; b < FWK; b++) msd[b] += fw_sq[b][ii + 1];
          ii += 2;
        }
      }
      for (int k = 0; k < 8; k++) _mm256_store_ps(gxn + jc + 8 * k, a[k]);
    }
    float m = 0.0f;
    for (int i = 0; i < D; i++) m += gxn[i] * fw_xn[i];
    m *= (1.0f / float(D));
    for (int i = 0; i < D; i++) gx[i] = fw_denom * (gxn[i] - fw_xn[i] * m);
#if PORT_CMX_MEAS
    const double kap = __builtin_expect(cmx_meas::g_fwk_mul != 1.0, 0)
        ? cmx_meas::g_fwk_mul * (1500.0e-6 + (6000.0e-6 - 1500.0e-6) / (1.0 + double(fw_t) / 8000.0))
        : (1500.0e-6 + (6000.0e-6 - 1500.0e-6) / (1.0 + double(fw_t) / 8000.0));
#else
    const double kap =
        (1500.0e-6 + (6000.0e-6 - 1500.0e-6) / (1.0 + double(fw_t) / 8000.0))
        ;
#endif
    fw_t++;
    float hsh = 0.0f;
    for (int b = 0; b < FWK; b++) {
      if (msd_neu) fw_msd[b] = msd[b] * (1.0f / float(DMLP));
      const float ms = fw_msd[b];
      const float* const vv = fw_v[b];
      const uint16_t* const idx = fw_idx[b];
      const float* const h = fw_h[b];
      float* const sh = fw_sh[b];
      const int nnz = fw_nnz[b];
      for (int k = 0; k < nnz; k++) {
        // the root of the CURRENT v at this feature (msd may be older)
        const float sk = ms / (std::sqrt(vv[idx[k]]) + 1e-3f * ms + 1e-12f);
        sh[k] = sk * h[k];
        hsh += h[k] * sh[k];
      }
    }
    const float step = float(kap) / (1.0f + hsh);
    // the dW step itself stays pending (see the members)
    for (int b = 0; b < FWK; b++) {
      const uint16_t* const idx = fw_idx[b];
      const float* const sh = fw_sh[b];
      for (int k = 0; k < fw_nnz[b]; k++) {
        const int j = idx[k];
        if (fw_pt[b][j] >= 0)   // an update without a read row in between
          fw_col_step(fw_dW + (size_t(b) * DMLP + j) * D, fw_pc[b][j],
                      fw_gxr[fw_pt[b][j] & (FWR - 1)]);
        fw_pt[b][j] = u;
        fw_pc[b][j] = step * sh[k];
      }
    }
  }

  static_assert(PaketArenas::DIN_LAST == 32 && PaketArenas::DIN_MATCH == 32 && D % 8 == 0,
                "is written for d_in 32 and D a multiple of 8");
  // out of line: paket_token_row() calls it for both side channels
  __attribute__((noinline)) static void proj_add(const float* f, int d_in, const int8_t* q,
                       const float* fold, float s_act, float* acc_out) {
    (void)d_in;
    alignas(32) int16_t qa[32];
    for (int i = 0; i < 32; i++) {
      float v = f[i] / s_act;   // as below: true division, clamp, round half to even
      if (v < -128.0f) v = -128.0f;
      if (v > 127.0f) v = 127.0f;
      qa[i] = static_cast<int16_t>(std::nearbyintf(v));
    }
    const __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qa));
    const __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qa + 16));
    for (int o = 0; o < D; o += 8) {
      __m256i r[8];
      for (int k = 0; k < 8; k++) {
        const int8_t* w = q + size_t(o + k) * 32;
        const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
        const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16)));
        r[k] = _mm256_add_epi32(_mm256_madd_epi16(a0, w0), _mm256_madd_epi16(a1, w1));
      }
      // lane-wise pair sums: h0123 = [r0..r3 low halves | r0..r3 high halves]
      const __m256i h0123 = _mm256_hadd_epi32(_mm256_hadd_epi32(r[0], r[1]), _mm256_hadd_epi32(r[2], r[3]));
      const __m256i h4567 = _mm256_hadd_epi32(_mm256_hadd_epi32(r[4], r[5]), _mm256_hadd_epi32(r[6], r[7]));
      const __m256i s = _mm256_add_epi32(_mm256_permute2x128_si256(h0123, h4567, 0x20),
                                         _mm256_permute2x128_si256(h0123, h4567, 0x31));
      const __m256 t = _mm256_mul_ps(_mm256_loadu_ps(fold + o), _mm256_cvtepi32_ps(s));
      _mm256_storeu_ps(acc_out + o, _mm256_add_ps(_mm256_loadu_ps(acc_out + o), t));
    }
  }

  // The token row with side channels: raw embedding row + the two projection
  // terms, then the embedding norm (the order of the training model).
  const float* paket_token_row(int token) {
    const PaketArenas& P = M.paket;
    if (token >= P.emb_rows ||
        pk.prev1 >= P.emb_rows + 1 ||
        pk.prev2 >= P.emb_rows + 1 ||
        pk.match_byte >= PaketArenas::NMATCH ||
        (pk.match_class & ~15) != 0) {
      // A row index out of range would read past a table and produce numbers
      // that look plausible.  The engine never generates one; a caller that
      // does gets a message instead of a silent wrong distribution.
      std::fprintf(stderr, "E194 %d %d %d %d %d %d\n",
          token, pk.prev1, pk.prev2, pk.match_byte, pk.match_class, P.emb_rows);
      std::exit(1);
    }
    alignas(64) float t[D];
    std::memcpy(t, P.emb_raw + size_t(token) * D, sizeof(float) * D);
    if (P.has_last) {
      alignas(64) float f[PaketArenas::DIN_LAST];
      static_assert(PaketArenas::X == 2,
          "the predecessor list below is written out for two "
          "slots.  With a different slot count the remaining slots would "
          "silently read row 0 -- widen PaketChannels and this initializer "
          "together, and re-export the last_table (its row count is X*rows).");
      const int prev[PaketArenas::X] = {pk.prev1, pk.prev2};
      for (int k = 0; k < PaketArenas::X; k++) {
        const size_t row = size_t(k) * (P.emb_rows + 1)
                         + size_t(prev[k]);
        std::memcpy(f + k * PaketArenas::DSLOT,
                    P.last_rows + row * PaketArenas::DSLOT,
                    sizeof(float) * PaketArenas::DSLOT);
      }
      proj_add(f, PaketArenas::DIN_LAST, P.last_q, P.last_fold, P.last_s_act,
               t);
    }
    if (P.has_match) {
      alignas(64) float f[PaketArenas::DIN_MATCH];
      // as zctx/zpmodel.py match_features: rows [match_byte] and [NMATCH + class]
      std::memcpy(f, P.match_rows + size_t(pk.match_byte) * PaketArenas::DSLOT,
                  sizeof(float) * PaketArenas::DSLOT);
      std::memcpy(f + PaketArenas::DSLOT,
                  P.match_rows +
                      size_t(PaketArenas::NMATCH + pk.match_class) *
                          PaketArenas::DSLOT,
                  sizeof(float) * PaketArenas::DSLOT);
      proj_add(f, PaketArenas::DIN_MATCH, P.match_q, P.match_fold,
               P.match_s_act, t);
    }
    rms_norm192(t, tok_paket);
    return tok_paket;
  }

  // gelesen ("read"): the output of the next step() is used.  Applies to
  // that step only.  true from the ctor.
  bool gelesen;
  static_assert((16 % 4) == 0,
      "the K/V-only last layer needs (NL-1) % 4 == 3 (a window layer)");

  // prior must be the cap-208 internal buffer with pads [205..208) zeroed
  void step(uint8_t token, float* probs_out) {
    if (!gelesen) {
      gelesen = true;
      zb_push(token);
      return;
    }
    zb_run();
    fw_gel = true;
    tok = token;
    zb_embed(*this);
    zb_layers(this, 1, false, 0, NL, t);
    {
      PSCOPE(PS_EL_RESID);
      ar_read();                                  // x = the final read (q 2 NL, M.ar_q_final)
    }

    {
      PSCOPE(PS_EL_NQ_FIN);
      rms_norm_quant192(x, xn, M.unembed.s_act, q5);
    }
    std::memcpy(sc_key, sc_l2_stage, sizeof(float) * D);
    {
      PSCOPE(PS_MM_UNEMB);
      qgemv_f32(M.unembed.m, q5, logits);  // rows_padded 208
    }
    {
      PSCOPE(PS_EL_HEAD);
      head_plm_softcap_softmax(logits, prior_f32, dot192(M.plm_w, xn), probs208);
      std::memcpy(probs_out, probs208, sizeof(float) * V);
#if PORT_CMX_MEAS
      { const unsigned long long cmx_t = cmx_meas::g_fw_ticks ? __rdtsc() : 0ull;
        if (!cmx_meas::g_fwoff) fw_head();   // research CMX_FWOFF: no second head (its row is never read then)
        if (cmx_meas::g_fw_ticks) cmx_meas::g_fw_cyc_head += __rdtsc() - cmx_t; }
#else
      fw_head();
#endif
    }
    t++;
  }

};

COLD_SIZE_ATTR TransformerOpt::TransformerOpt(const char* weights_path, AttnKind attn)
    : impl(new TransformerOptImpl(weights_path, attn)) {}

COLD_SIZE_ATTR TransformerOpt::~TransformerOpt() = default;

void TransformerOpt::begin_article(int64_t rope_position_offset) {
  impl->begin(rope_position_offset);
}

void TransformerOpt::step(uint8_t token, const uint16_t* prior_f16,
                          float* probs_out) {
  prior_f16_to_f32(prior_f16, impl->prior_f32);  // exact; pads zeroed
  impl->step(token, probs_out);
}

void TransformerOpt::step(uint8_t token, const float* prior205,
                          float* probs_out) {
  std::memcpy(impl->prior_f32, prior205, sizeof(float) * V);
  for (int i = 208; i < 208; ++i) impl->prior_f32[i] = 0.0f;
  impl->step(token, probs_out);
}

void TransformerOpt::step(uint8_t token, const uint16_t* prior_f16,
                          float* probs_out, const PaketChannels& channels) {
  impl->pk = channels;
  prior_f16_to_f32(prior_f16, impl->prior_f32);  // exact; pads zeroed
  impl->step(token, probs_out);
}

bool TransformerOpt::paket_head() const { return impl->M.paket.present; }

const float* TransformerOpt::paket_row(uint8_t token,
                                       const PaketChannels& channels) {
  impl->pk = channels;
  return impl->paket_token_row(token);
}

int TransformerOpt::embedding_rows() const { return impl->M.paket.emb_rows; }

void TransformerOpt::set_gelesen(bool gelesen) { impl->gelesen = gelesen; }

const float* TransformerOpt::last_logits() const { return impl->logits; }

// `xn` is x after the final rms_norm, i.e. exactly the vector fed to
// the unembedding matmul one line later in step().  Nothing is written here.
const float* TransformerOpt::last_hidden() const { return impl->xn; }
int TransformerOpt::hidden_size() { return D; }
const float* TransformerOpt::last_key() const { return impl->sc_key; }

AttnKind TransformerOpt::attn_kind() const { return impl->kind; }

#if PORT_CMX_MEAS
void TransformerOpt::fw_update(uint8_t target) {
  const unsigned long long cmx_t = cmx_meas::g_fw_ticks ? __rdtsc() : 0ull;   // research CMX_FW_TICKS
  impl->fw_update(target);
  if (cmx_meas::g_fw_ticks) { cmx_meas::g_fw_cyc_update += __rdtsc() - cmx_t; ++cmx_meas::g_fw_n_update; }
}
#else
void TransformerOpt::fw_update(uint8_t target) { impl->fw_update(target); }
#endif
const float* TransformerOpt::fw_probs() const { return impl->fw_probs208; }

}  // namespace opt
}  // namespace fx2
