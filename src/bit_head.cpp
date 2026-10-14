// Per-bit LSTM correction head.  The contract is described in the header of
// the same name.
//
// This translation unit MUST be compiled with value-safe floating point
// (-ffp-model=precise -ffp-contract=off, see the makefile rule `bitl`) and
// its entry points are noinline: encode and decode then execute the identical
// machine code, so the head stays in lockstep by construction.
//
// ATTRIBUTION.  The AVX2 kernels Exp8/Sigmoid8/Tanh8 and the fixed-order
// DotPadded reduction below are ported line by line from
// src/models/bitlstm32-head.cpp of cmix-obias (David Freelan, GPL-3.0);
// Exp8 is an altered version of exp256_ps of avx_mathfun.h (Giovanni
// Garberoglio, 2012), itself based on sse_mathfun.h (Julien Pommier, 2007);
// zlib licence, notice in third_party/mathfun-LICENSE.txt of the source
// package; its polynomial is the Cephes expf of Stephen L. Moshier.  The forward
// architecture (in_proj -> SiLU -> LSTMCell -> linear, logit-additive
// correction after Discretize), several input features and the offline
// training on a trace follow the same model.  The online backward pass, the
// optimiser, the state projections and the parameters are our own.

#include "bit_head.h"
#include "cold_size.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "isa_floor.h"
#include <immintrin.h>

#include "bit_head_weights.h"

namespace bit_head {
namespace {

// Logit-space normaliser.  Same constant as the reference head: it only has
// to bring the stage-1 row, the terminal logit and the SSE correction into
// the same O(1) range so a single learning rate fits all of them.
constexpr float kTScale = 12.203f;

// Number of |sd| samples in the SSE-correction window (reference: 256).
constexpr int kSdCap = 256;

// Feature block that is always present, on top of the stage-1 row:
//   +0        m1raw / T
//   +1..+3    t/T, (t/T)^2, sgn(t)
//   +4..+11   bit position one-hot(8)
constexpr int kBaseExtra = 12;
// Optional SSE block (feature bit 0, off in the release configuration): sd/T and
// mean|sd| over 256 / T.
constexpr int kSseExtra = 2;

constexpr int kNTf = 0;

inline float Sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float Sgn(float x) { return (x > 0.0f) ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

// ---------------------------------------------------------------------------
// AVX2 kernels (ported, see ATTRIBUTION above)
// ---------------------------------------------------------------------------
// 8-lane expf: altered exp256_ps of avx_mathfun.h (see the attribution
// above), Cephes polynomial.  Fixed coefficients and fixed operation order,
// so it is deterministic.
inline __m256 Exp8(__m256 x) {
  const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
  const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
  const __m256 c1 = _mm256_set1_ps(0.693359375f);
  const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);
  const __m256 one = _mm256_set1_ps(1.0f);
  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);
  __m256 fx = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f));
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, c1, x);
  x = _mm256_fnmadd_ps(fx, c2, x);
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = p0;
  y = _mm256_fmadd_ps(y, x, p1);
  y = _mm256_fmadd_ps(y, x, p2);
  y = _mm256_fmadd_ps(y, x, p3);
  y = _mm256_fmadd_ps(y, x, p4);
  y = _mm256_fmadd_ps(y, x, p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);
  const __m256i imm = _mm256_add_epi32(_mm256_cvttps_epi32(fx),
                                       _mm256_set1_epi32(0x7f));
  const __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(imm, 23));
  return _mm256_mul_ps(y, pow2n);
}
inline __m256 Sigmoid8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_div_ps(one, _mm256_add_ps(one,
      Exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}
inline __m256 Tanh8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 e = Exp8(_mm256_mul_ps(x, _mm256_set1_ps(-2.0f)));
  return _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
}

// Deterministic dense dot product with a fixed-order horizontal reduction.
inline float DotPadded(const float* w, const float* x, int n_padded) {
  __m256 acc = _mm256_setzero_ps();
  for (int j = 0; j < n_padded; j += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(x + j), acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

inline void AxpyNoFma(float* __restrict acc, const float* __restrict v,
                      float sc, int n) {
  const __m256 vs = _mm256_set1_ps(sc);
  for (int j = 0; j < n; j += 8) {
    _mm256_storeu_ps(acc + j,
        _mm256_add_ps(_mm256_loadu_ps(acc + j),
                      _mm256_mul_ps(vs, _mm256_loadu_ps(v + j))));
  }
}

// gwih += d*a ; gwhh += d*hprev ; da += d*wih ; dhn += d*whh, in one pass.
inline void GateRowBack(float* __restrict gwih_k, float* __restrict gwhh_k,
                        float* __restrict dav, float* __restrict dhnv,
                        const float* __restrict wih_k,
                        const float* __restrict whh_k,
                        const float* __restrict a_t,
                        const float* __restrict hprev_t, float d, int n) {
  const __m256 vd = _mm256_set1_ps(d);
  for (int r = 0; r < n; r += 8) {
    _mm256_storeu_ps(gwih_k + r,
        _mm256_add_ps(_mm256_loadu_ps(gwih_k + r),
                      _mm256_mul_ps(vd, _mm256_loadu_ps(a_t + r))));
    _mm256_storeu_ps(gwhh_k + r,
        _mm256_add_ps(_mm256_loadu_ps(gwhh_k + r),
                      _mm256_mul_ps(vd, _mm256_loadu_ps(hprev_t + r))));
    _mm256_storeu_ps(dav + r,
        _mm256_add_ps(_mm256_loadu_ps(dav + r),
                      _mm256_mul_ps(_mm256_loadu_ps(wih_k + r), vd)));
    _mm256_storeu_ps(dhnv + r,
        _mm256_add_ps(_mm256_loadu_ps(dhnv + r),
                      _mm256_mul_ps(_mm256_loadu_ps(whh_k + r), vd)));
  }
}

// ---------------------------------------------------------------------------
// bf16 weight rounding (PREC != 0; the release configuration keeps float
// weights, prec == 0)
//
// Weights are kept in float arrays but rounded to the bf16 grid
// (round-to-nearest-even) after every update, so the forward pass sees
// exactly the values a bf16 weight store would hold.
// ---------------------------------------------------------------------------
inline float Bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t rnd = (u >> 16) & 1u;
  u = (u + 0x7fffu + rnd) & 0xffff0000u;
  float o;
  std::memcpy(&o, &u, sizeof(o));
  return o;
}
inline __m256 Bf16_8(__m256 x) {
  __m256i u = _mm256_castps_si256(x);
  const __m256i rnd = _mm256_and_si256(_mm256_srli_epi32(u, 16),
                                       _mm256_set1_epi32(1));
  u = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), rnd));
  u = _mm256_and_si256(u, _mm256_set1_epi32((int)0xffff0000u));
  return _mm256_castsi256_ps(u);
}

// ---------------------------------------------------------------------------
// Weight update kernels.  OPT and PREC are compile-time so the hot loops carry
// no branch; the eight instantiations are dispatched once per bit.
//
//   OPT 0  plain SGD          w -= lr*g
//   OPT 1  SGD + momentum     m = mom*m + g;                 w -= lr*m
//   OPT 2  RMSProp            s = rho*s + (1-rho)*g^2;       w -= lr*g/(sqrt(s)+eps)
//   OPT 3  Adam               m,s as usual, bias corrected;  w -= lrt*m/(sqrt(s)+eps)
//
// The release configuration uses OPT 3 (Adam).  The whole backward pass plus update costs
// about 1.2 to 1.7 times the forward pass.
//
// RowUpdateAccum additionally accumulates acc[j] += w_old[j]*d, the gradient
// flowing into the layer below.  It MUST read w before writing it.
// ---------------------------------------------------------------------------
struct Hyp {
  float lr;      // OPT 0/1/2: learning rate; OPT 3: bias-corrected rate
  float mom;     // OPT 1: momentum; OPT 2: rho; OPT 3: beta1
  float b2;      // OPT 3: beta2
  float eps;
};

// Turns the raw gradient g into the weight decrement, updating the optimiser
// state in place.  One definition, used by all three call sites.
template <int OPT>
inline __m256 Step8(__m256 g, float* m1, float* m2, int j, const Hyp& hp) {
  if (OPT == 0) return g;
  if (OPT == 1) {
    const __m256 s = _mm256_fmadd_ps(_mm256_set1_ps(hp.mom),
                                     _mm256_loadu_ps(m1 + j), g);
    _mm256_storeu_ps(m1 + j, s);
    return s;
  }
  if (OPT == 2) {
    const __m256 rho = _mm256_set1_ps(hp.mom);
    const __m256 s = _mm256_fmadd_ps(rho, _mm256_loadu_ps(m2 + j),
        _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), rho),
                      _mm256_mul_ps(g, g)));
    _mm256_storeu_ps(m2 + j, s);
    return _mm256_div_ps(g, _mm256_add_ps(_mm256_sqrt_ps(s),
                                          _mm256_set1_ps(hp.eps)));
  }
  const __m256 b1 = _mm256_set1_ps(hp.mom);
  const __m256 b2 = _mm256_set1_ps(hp.b2);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 mv = _mm256_fmadd_ps(b1, _mm256_loadu_ps(m1 + j),
                                    _mm256_mul_ps(_mm256_sub_ps(one, b1), g));
  const __m256 sv = _mm256_fmadd_ps(b2, _mm256_loadu_ps(m2 + j),
      _mm256_mul_ps(_mm256_sub_ps(one, b2), _mm256_mul_ps(g, g)));
  _mm256_storeu_ps(m1 + j, mv);
  _mm256_storeu_ps(m2 + j, sv);
  return _mm256_div_ps(mv, _mm256_add_ps(_mm256_sqrt_ps(sv),
                                         _mm256_set1_ps(hp.eps)));
}
template <int OPT>
inline float Step1(float g, float* m1, float* m2, const Hyp& hp) {
  if (OPT == 0) return g;
  if (OPT == 1) { const float s = hp.mom * (*m1) + g; *m1 = s; return s; }
  if (OPT == 2) {
    const float s = hp.mom * (*m2) + (1.0f - hp.mom) * g * g;
    *m2 = s;
    return g / (std::sqrt(s) + hp.eps);
  }
  const float mv = hp.mom * (*m1) + (1.0f - hp.mom) * g;
  const float sv = hp.b2 * (*m2) + (1.0f - hp.b2) * g * g;
  *m1 = mv; *m2 = sv;
  return mv / (std::sqrt(sv) + hp.eps);
}

template <int OPT, int PREC>
inline void RowUpdate(float* w, float* m1, float* m2, const float* v, float d,
                      const Hyp& hp, int n) {
  const __m256 vd = _mm256_set1_ps(d);
  const __m256 vlr = _mm256_set1_ps(hp.lr);
  for (int j = 0; j < n; j += 8) {
    __m256 wv = _mm256_loadu_ps(w + j);
    const __m256 step = Step8<OPT>(_mm256_mul_ps(_mm256_loadu_ps(v + j), vd),
                                   m1, m2, j, hp);
    wv = _mm256_fnmadd_ps(vlr, step, wv);
    if (PREC) wv = Bf16_8(wv);
    _mm256_storeu_ps(w + j, wv);
  }
}

template <int OPT, int PREC>
inline void RowUpdateAccum(float* w, float* m1, float* m2, const float* v,
                           float d, const Hyp& hp, float* acc, int n) {
  const __m256 vd = _mm256_set1_ps(d);
  const __m256 vlr = _mm256_set1_ps(hp.lr);
  for (int j = 0; j < n; j += 8) {
    __m256 wv = _mm256_loadu_ps(w + j);
    _mm256_storeu_ps(acc + j,
                     _mm256_fmadd_ps(wv, vd, _mm256_loadu_ps(acc + j)));
    const __m256 step = Step8<OPT>(_mm256_mul_ps(_mm256_loadu_ps(v + j), vd),
                                   m1, m2, j, hp);
    wv = _mm256_fnmadd_ps(vlr, step, wv);
    if (PREC) wv = Bf16_8(wv);
    _mm256_storeu_ps(w + j, wv);
  }
}

template <int OPT, int PREC>
inline void ScalarUpdate(float* w, float* m1, float* m2, float g,
                         const Hyp& hp) {
  const float step = Step1<OPT>(g, m1, m2, hp);
  float wv = *w - hp.lr * step;
  if (PREC) wv = Bf16(wv);
  *w = wv;
}

// Deterministic splitmix64 (Steele, Lea and Flood; public-domain reference
// code by Sebastiano Vigna): the initial weights must be identical in encoder
// and decoder, so no library RNG and no time/address dependence.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t Next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  // Uniform in [-b, b], 24 mantissa bits.
  float Uniform(float b) {
    const uint32_t r = (uint32_t)(Next() >> 40);  // 24 bits
    const float u = (float)r * (1.0f / 16777216.0f);  // [0,1)
    return (2.0f * u - 1.0f) * b;
  }
};

constexpr double PowTenNeg(int n) {
  return (n <= 0) ? 1.0 : (PowTenNeg(n - 1) / 10.0);
}
constexpr float kEps = (float)PowTenNeg(8);

struct Head {
  static constexpr int cells = 16;
  static constexpr int n1 = STAGE1_COL(28);
  static constexpr int nsse = ((0) & kFeatSse) ? kSseExtra : 0;
  static constexpr int ndrop =
      1 ? (STAGE1_COL(20) - STAGE1_COL(7) + 1) : 0;
  static constexpr int ndrop2 =
      1 ? (STAGE1_COL(27) - STAGE1_COL(27) + 1) : 0;
  static constexpr int n1keep = n1 - ndrop - ndrop2;
  static constexpr int nhid = 32;
  static constexpr int ntf = kNTf;
  static constexpr int nlhid = 32;
  static constexpr int nzone = 0;   // gate-zone one-hot: not used
  // Layout of the feature row:
  //   [0 .. n1keep-1]              stage-1 row (selected columns)
  //   [n1keep+0]                   m1raw
  //   [n1keep+1 .. n1keep+3]       t/T, (t/T)^2, sgn(t)
  //   [n1keep+4 .. n1keep+11]      bit-position one-hot(8)
  //   [n1keep+12, n1keep+13]       SSE pair (only with feat & kFeatSse)
  //   [kPrev]                      prev_bit
  //   [kHid ..] [kTf ..] [kLhid ..] [kZone ..]   optional blocks, empty here
  static constexpr int kPrev = n1keep + kBaseExtra + nsse;
  static constexpr int kHid = kPrev + 1;
  static constexpr int kTf = kHid + nhid;
  static constexpr int kLhid = kTf + ntf;
  static constexpr int kZone = kLhid + nlhid;
  static constexpr int nin =                // features without prev_bit
      n1keep + kBaseExtra + nsse + nhid + ntf + nlhid + nzone;
  static constexpr int inw = ((nin + 1) + 7) & ~7;   // padded to 8

  static constexpr float lr = (float)(300 / 1000000.0);
  static constexpr float mom = (float)(900000 / 1000000.0);
  static constexpr float b2 = (float)(999000 / 1000000.0);
  static constexpr float eps = kEps;
  static constexpr float clamp = (float)(8000 / 1000.0);
  static constexpr int opt = 3;
  static constexpr int prec = (0) ? 1 : 0;
  static constexpr int seq = 1024;
  static constexpr int feat = 0;
  double b1t = 1.0, b2t = 1.0;
  static constexpr int accum = 16;
  static constexpr int meang = (1) ? 1 : 0;
  static constexpr float clip = (float)(1000 / 1000.0);
  static constexpr int xscale = (0) ? 1 : 0;
  int agrp = 0;               // windows accumulated in the current group
  unsigned long long nvalid = 0;  // non-override bits in the current group

  // ---- weights ----
  float* win = nullptr;    // cells x inw
  float* bin = nullptr;    // cells
  float* wih = nullptr;    // 4*cells x cells
  float* whh = nullptr;    // 4*cells x cells
  float* bg = nullptr;     // 4*cells
  float* wout = nullptr;   // cells
  float bout = 0.0f;

  // ---- optimiser state ----
  // m1 = first moment / momentum, m2 = second moment.  Both are allocated
  // together: the memory is a few dozen KB and one branch fewer in Build().
  float* m1win = nullptr;  float* m2win = nullptr;
  float* m1bin = nullptr;  float* m2bin = nullptr;
  float* m1wih = nullptr;  float* m2wih = nullptr;
  float* m1whh = nullptr;  float* m2whh = nullptr;
  float* m1bg = nullptr;   float* m2bg = nullptr;
  float* m1wout = nullptr; float* m2wout = nullptr;
  float m1bout = 0.0f;     float m2bout = 0.0f;

  // ---- recurrent state ----
  float* h = nullptr;
  float* c = nullptr;
  float* hprev = nullptr;
  float* cprev = nullptr;
  float prev_bit = 0.5f;
  unsigned long long i = 0;   // coded-bit index

  // ---- per-bit scratch, live between Adjust() and Observe() ----
  float* x = nullptr;      // inw, feature row incl. prev_bit
  float* z = nullptr;      // cells, in_proj pre-activation
  float* sg = nullptr;     // cells, sigmoid(z) (for the SiLU derivative)
  float* a = nullptr;      // cells, SiLU(z)
  float* gi = nullptr;
  float* gf = nullptr;
  float* gg = nullptr;
  float* go = nullptr;
  float* tc = nullptr;     // cells, tanh(c)
  float* pre = nullptr;    // 4*cells
  float* dpre = nullptr;   // 4*cells
  float* da = nullptr;     // cells
  float delta = 0.0f;
  float t = 0.0f;
  int skip = 0;            // override bit: forward only, no learning

  // ---- per-byte channel ----
  // Registered by the coder before Adjust(); constant across the eight bits of
  // a byte.  Pure observation - nothing the head computes from them re-enters
  // the predictor.
  const float* aux_hid = nullptr;   // TF residual stream / LSTM cell state
  int aux_n_hid = 0;
  const float* aux_tf = nullptr;    // kTfAll per-byte scalars
  int aux_n_tf = 0;
  const float* aux_lhid = nullptr;  // state of the byte-LSTM beside the TF
  int aux_n_lhid = 0;
  int aux_lhid_pre = 0;             // the vector IS already the sketch
  // A pointer, not a value: the zone is written once per byte by the gate
  // automaton and read live here, so the head needs no per-bit setter call
  // across the translation-unit boundary (this unit is built without -flto).
  const unsigned int* aux_zone = nullptr;   // gate zone of this byte (not read in the release configuration)
  int hid_proj = 0;                 // 1 = fixed +-1/sqrt(n) sketch
  float* hidP = nullptr;            // nhid x aux_n_hid
  float* hidC = nullptr;            // nhid, the sketch OF THIS BYTE
  unsigned long long hid_byte = ~0ULL;   // byte hidC belongs to
  int lhid_proj = 0;
  float* lhidP = nullptr;
  float* lhidC = nullptr;
  unsigned long long lhid_byte = ~0ULL;
  int frozen = 0;
  const char* load_path = nullptr;

  // ---- SSE-correction window ----
  float* sd_ring = nullptr;
  double sd_sum = 0.0;
  float pend_sd = 0.0f;

  static constexpr int tbptt = (1) ? 1 : 0;
  // window length in coded bits
  static constexpr int wlen = (seq > 0) ? seq : 64;
  int nbuf = 0;      // steps buffered so far
  float* bx = nullptr;    // wlen x inw
  float* bz = nullptr;    // wlen x C, in_proj pre-activation
  float* bsg = nullptr;   // wlen x C, sigmoid(z)
  float* ba = nullptr;    // wlen x C, SiLU(z)
  float* bgi = nullptr; float* bgf = nullptr;
  float* bgg = nullptr; float* bgo = nullptr;
  float* btc = nullptr;   // wlen x C, tanh(c)
  float* bh = nullptr;    // wlen x C, h AFTER step t
  float* bc = nullptr;    // wlen x C, c AFTER step t
  float* bdl = nullptr;   // wlen, dL/d(delta); 0 on override bits
  // gradient accumulators
  float* gwin = nullptr; float* gbin = nullptr;
  float* gwih = nullptr; float* gwhh = nullptr;
  float* gbg = nullptr;  float* gwout = nullptr;
  float gbout = 0.0f;
  float* dhn = nullptr;  float* dcn = nullptr;   // carried-back gradients
  float* zbuf = nullptr;  // C zeros, the h_prev of the first window step
  // The state the window STARTED from, i.e. the predecessor of step 0.  With
  // seq > 0 that is the reset and therefore exactly zero; with seq == 0 the
  // state carries across the window boundary and the backward pass has to
  // see it.
  float* wsh = nullptr;  float* wsc = nullptr;

  // ---- learning-curve counters (measurement only) ----
  // Costs of the same bits under the coder's base probability and under the
  // head's corrected one.  Nothing here feeds back into the model.
  const char* log_path = nullptr;
  unsigned long long log_n = 8388608ULL;   // report every N coded bits
  double log_base = 0.0, log_head = 0.0;   // bits, current window
  double log_base_tot = 0.0, log_head_tot = 0.0;
  unsigned int pend_base_p = 32768, pend_head_p = 32768;

  int nparams = 0;

  void Build();
  void LoadBlob();
  void Forward();
  void Backward(float y);
  template <int OPT, int PREC> void BackwardT(float y);
  void BackwardWindow();
  void FinishGroup();
  template <int OPT, int PREC> void ApplyT();
  void Apply();
  void Free();
};

Head g_head;
bool g_init_done = false;

float* Alloc(int n) {
  float* p = (float*)std::calloc((size_t)n > 0 ? (size_t)n : 1, sizeof(float));
  return p;
}

void Head::Free() {
  std::free(win); std::free(bin); std::free(wih); std::free(whh);
  std::free(bg); std::free(wout);
  std::free(m1win); std::free(m1bin); std::free(m1wih); std::free(m1whh);
  std::free(m1bg); std::free(m1wout);
  std::free(m2win); std::free(m2bin); std::free(m2wih); std::free(m2whh);
  std::free(m2bg); std::free(m2wout);
  std::free(h); std::free(c); std::free(hprev); std::free(cprev);
  std::free(x); std::free(z); std::free(sg); std::free(a);
  std::free(gi); std::free(gf); std::free(gg); std::free(go); std::free(tc);
  std::free(pre); std::free(dpre); std::free(da); std::free(sd_ring);
  std::free(bx); std::free(bz); std::free(bsg); std::free(ba);
  std::free(bgi); std::free(bgf); std::free(bgg); std::free(bgo);
  std::free(btc); std::free(bh); std::free(bc); std::free(bdl);
  std::free(gwin); std::free(gbin); std::free(gwih); std::free(gwhh);
  std::free(gbg); std::free(gwout); std::free(dhn); std::free(dcn);
  std::free(zbuf); std::free(wsh); std::free(wsc);
  std::free(hidP); std::free(hidC);
  std::free(lhidP); std::free(lhidC);
  std::memset(this, 0, sizeof(*this));
}

void Head::Build() {
  const int C = cells;

  // Build guard for the two state-vector blocks.  A block that is switched
  // on but has nothing registered, or is WIDER than the vector it reads, must
  // stop the run - it would otherwise read past the vector, or read a column
  // of constant zeros, and a full run started from a silently dead arm costs
  // days and reads as a null result.  Same principle as the row-width check
  // in Adjust() below.
  {
    const char* what = nullptr;
    int want = 0, have = 0;
    if (nhid > aux_n_hid) { what = "transformer state"; want = nhid; have = aux_n_hid; }
    if (what == nullptr && nlhid > aux_n_lhid) {
      what = "lstm state"; want = nlhid; have = aux_n_lhid;
    }
    if (what != nullptr) {
      std::fprintf(stderr,
          "\ncorrection model: %s width %d, %d values registered\n",
          what, want, have);
      std::fflush(stderr);
      std::abort();
    }
  }
  hid_proj = (nhid < aux_n_hid) ? 1 : 0;
  if (hid_proj) {
    hidP = Alloc(nhid * aux_n_hid);
    hidC = Alloc(nhid);
    Rng prng(0x4849445052ULL);   // "HIDPR"
    const float sc = 1.0f / std::sqrt((float)aux_n_hid);
    for (int r = 0; r < nhid; ++r) {
      for (int cc = 0; cc < aux_n_hid; ++cc) {
        hidP[r * aux_n_hid + cc] = (prng.Uniform(1.0f) >= 0.0f) ? sc : -sc;
      }
    }
  }
  // The same sketch for the second state vector, but an OWN draw
  // stream -- the two vectors must not get the same projection, or their
  // comparison would be co-determined by it.
  lhid_proj = (!aux_lhid_pre && nlhid < aux_n_lhid) ? 1 : 0;
  if (lhid_proj) {
    lhidP = Alloc(nlhid * aux_n_lhid);
    lhidC = Alloc(nlhid);
    Rng prng(0x4c484450ULL);   // "LHDP"
    const float sc = 1.0f / std::sqrt((float)aux_n_lhid);
    for (int r = 0; r < nlhid; ++r) {
      for (int cc = 0; cc < aux_n_lhid; ++cc) {
        lhidP[r * aux_n_lhid + cc] = (prng.Uniform(1.0f) >= 0.0f) ? sc : -sc;
      }
    }
  }

  win = Alloc(C * inw);   bin = Alloc(C);
  wih = Alloc(4 * C * C); whh = Alloc(4 * C * C);
  bg = Alloc(4 * C);      wout = Alloc(C);
  // Always allocated, even for plain SGD: the update kernels form
  // `m1wih + k*C` unconditionally, and pointer arithmetic on a null pointer is
  // undefined behaviour even when the result is never dereferenced.  A few
  // dozen KB is not worth a class of bug that only shows up under -O3.
  m1win = Alloc(C * inw);   m1bin = Alloc(C);
  m1wih = Alloc(4 * C * C); m1whh = Alloc(4 * C * C);
  m1bg = Alloc(4 * C);      m1wout = Alloc(C);
  m2win = Alloc(C * inw);   m2bin = Alloc(C);
  m2wih = Alloc(4 * C * C); m2whh = Alloc(4 * C * C);
  m2bg = Alloc(4 * C);      m2wout = Alloc(C);
  h = Alloc(C); c = Alloc(C); hprev = Alloc(C); cprev = Alloc(C);
  x = Alloc(inw); z = Alloc(C); sg = Alloc(C); a = Alloc(C);
  gi = Alloc(C); gf = Alloc(C); gg = Alloc(C); go = Alloc(C); tc = Alloc(C);
  pre = Alloc(4 * C); dpre = Alloc(4 * C); da = Alloc(C);
  sd_ring = Alloc(kSdCap);
  if (tbptt) {
    const int W = wlen;
    bx = Alloc(W * inw);
    bz = Alloc(W * C); bsg = Alloc(W * C); ba = Alloc(W * C);
    bgi = Alloc(W * C); bgf = Alloc(W * C);
    bgg = Alloc(W * C); bgo = Alloc(W * C);
    btc = Alloc(W * C); bh = Alloc(W * C); bc = Alloc(W * C);
    bdl = Alloc(W);
    gwin = Alloc(C * inw); gbin = Alloc(C);
    gwih = Alloc(4 * C * C); gwhh = Alloc(4 * C * C);
    gbg = Alloc(4 * C); gwout = Alloc(C);
    dhn = Alloc(C); dcn = Alloc(C); zbuf = Alloc(C);
    wsh = Alloc(C); wsc = Alloc(C);
  }

  // Initialisation.  PyTorch-style uniform for the hidden layers (symmetry
  // breaking is essential: with all-zero weights h stays 0, no gradient ever
  // reaches in_proj and the head is dead), but the OUTPUT layer starts at
  // exactly zero.  delta is therefore 0 on the first bit and the head begins
  // as an exact no-op on the coder - it can only move away from the base
  // prediction once it has seen evidence.
  Rng rng(0x42495431ULL);  // fixed seed ("BIT1" in ASCII)
  const float bin_lim = 1.0f / std::sqrt((float)inw);
  for (int r = 0; r < C; ++r) {
    for (int j = 0; j < inw; ++j) win[r * inw + j] = rng.Uniform(bin_lim);
    // Columns beyond nin+1 are never fed anything but 0; keep them at 0 so
    // the padding cannot contribute a bias term.
    for (int j = nin + 1; j < inw; ++j) win[r * inw + j] = 0.0f;
    bin[r] = rng.Uniform(bin_lim);
  }
  const float hid_lim = 1.0f / std::sqrt((float)C);
  for (int k = 0; k < 4 * C * C; ++k) wih[k] = rng.Uniform(hid_lim);
  for (int k = 0; k < 4 * C * C; ++k) whh[k] = rng.Uniform(hid_lim);
  for (int k = 0; k < 4 * C; ++k) bg[k] = rng.Uniform(hid_lim);
  // Forget-gate bias +1: the classic Gers/Schmidhuber initialisation.  With
  // backpropagation truncated to short windows the recurrent path learns
  // slowly, so the cell has to default to remembering rather than forgetting.
  for (int r = 0; r < C; ++r) bg[1 * C + r] += 1.0f;
  for (int r = 0; r < C; ++r) wout[r] = 0.0f;
  bout = 0.0f;
  if (prec) {
    for (int k = 0; k < C * inw; ++k) win[k] = Bf16(win[k]);
    for (int k = 0; k < C; ++k) bin[k] = Bf16(bin[k]);
    for (int k = 0; k < 4 * C * C; ++k) wih[k] = Bf16(wih[k]);
    for (int k = 0; k < 4 * C * C; ++k) whh[k] = Bf16(whh[k]);
    for (int k = 0; k < 4 * C; ++k) bg[k] = Bf16(bg[k]);
  }

  nparams = C * (nin + 1) + C + 8 * C * C + 4 * C + C + 1;
  LoadBlob();
}

void Head::LoadBlob() {
  // The blob is part of the binary (.rodata, which the decoder keeps),
  // read through a memory stream so the checks below are those of the file form.
  const char* path = "the embedded weights";
  FILE* f = fmemopen(const_cast<unsigned char*>(kBlob), sizeof(kBlob), "r");
  if (f == nullptr) {
    std::fprintf(stderr, "correction model: cannot open %s\n", path);
    std::exit(3);
  }
  char mg[4]; std::uint32_t ver = 0; std::int32_t g[4] = {0, 0, 0, 0};
  if (std::fread(mg, 1, 4, f) != 4 || std::memcmp(mg, "NBLB", 4) != 0) {
    std::fprintf(stderr, "correction model: %s is not a weight file\n", path);
    std::exit(3);
  }
  if (std::fread(&ver, 4, 1, f) != 1 ||
      std::fread(g, sizeof(std::int32_t), 4, f) != 4) {
    std::fprintf(stderr, "correction model: %s: header too short\n", path);
    std::exit(3);
  }
  if (ver != 1u && ver != 2u && ver != 4u && ver != 5u) {
    std::fprintf(stderr, "correction model: weight file version %u, expected 1, 2, 4 or 5\n",
                 (unsigned)ver);
    std::exit(3);
  }
  if (ver == 4u || ver == 5u) {
    // versions 4 and 5 carry a learned projection of the transformer state,
    // this head reads a fixed one
    std::fprintf(stderr,
        "correction model: weight file version %u has a learned projection, "
        "this build a fixed one\n", (unsigned)ver);
    std::exit(3);
  }
  if (g[0] != cells || g[1] != nin || g[2] != inw || g[3] != n1) {
    std::fprintf(stderr,
        "correction model: weight file shape %d/%d/%d/%d, this build "
        "%d/%d/%d/%d (cells/inputs/input width/stage-1 width)\n",
        g[0], g[1], g[2], g[3], cells, nin, inw, n1);
    std::exit(3);
  }
  const int C = cells;
  const size_t nw = (size_t)C * (size_t)inw;
  const size_t ng = (size_t)4 * (size_t)C * (size_t)C;
  const size_t nb = (size_t)4 * (size_t)C;
  const bool blob_f32 = (ver == 1u || ver == 4u);
  auto rd = [f, blob_f32](float* dst, size_t n) -> bool {
    if (blob_f32) return std::fread(dst, sizeof(float), n, f) == n;
    for (size_t i = 0; i < n; ++i) {
      std::uint16_t h = 0;
      if (std::fread(&h, sizeof(h), 1, f) != 1) return false;
      const std::uint32_t u = (std::uint32_t)h << 16;
      std::memcpy(dst + i, &u, sizeof(u));
    }
    return true;
  };
  bool ok = true;
  ok = ok && rd(win, nw);
  ok = ok && rd(bin, (size_t)C);
  ok = ok && rd(wih, ng);
  ok = ok && rd(whh, ng);
  ok = ok && rd(bg, nb);
  ok = ok && rd(wout, (size_t)C);
  ok = ok && rd(&bout, 1);
  std::fclose(f);
  if (!ok) {
    std::fprintf(stderr, "correction model: %s: short read\n", path);
    std::exit(3);
  }
  // Loaded weights start frozen: forward pass only.  In the release
  // configuration Observe() switches the head to online learning at the first
  // 1,024-bit window boundary after the end of the training range.  The
  // trainer's warm start (the other load mode) keeps learning from the first
  // bit, which is how one epoch hands over to the next.
  frozen = (1 == 2) ? 0 : 1;
}

void Head::Forward() {
  const int C = cells;
  for (int r = 0; r < C; ++r) z[r] = DotPadded(win + r * inw, x, inw) + bin[r];
  for (int r = 0; r < C; r += 8) {
    const __m256 zv = _mm256_loadu_ps(z + r);
    const __m256 s = Sigmoid8(zv);
    _mm256_storeu_ps(sg + r, s);
    _mm256_storeu_ps(a + r, _mm256_mul_ps(zv, s));
  }
  for (int k = 0; k < 4 * C; ++k) {
    pre[k] = DotPadded(wih + k * C, a, C) + DotPadded(whh + k * C, hprev, C) +
             bg[k];
  }
  for (int r = 0; r < C; r += 8) {
    const __m256 vi = Sigmoid8(_mm256_loadu_ps(pre + 0 * C + r));
    const __m256 vf = Sigmoid8(_mm256_loadu_ps(pre + 1 * C + r));
    const __m256 vg = Tanh8(_mm256_loadu_ps(pre + 2 * C + r));
    const __m256 vo = Sigmoid8(_mm256_loadu_ps(pre + 3 * C + r));
    const __m256 cn = _mm256_fmadd_ps(vf, _mm256_loadu_ps(cprev + r),
                                      _mm256_mul_ps(vi, vg));
    const __m256 th = Tanh8(cn);
    _mm256_storeu_ps(gi + r, vi);
    _mm256_storeu_ps(gf + r, vf);
    _mm256_storeu_ps(gg + r, vg);
    _mm256_storeu_ps(go + r, vo);
    _mm256_storeu_ps(c + r, cn);
    _mm256_storeu_ps(tc + r, th);
    _mm256_storeu_ps(h + r, _mm256_mul_ps(vo, th));
  }
  delta = DotPadded(wout, h, C) + bout;
  if (tbptt) {
    const size_t n = sizeof(float) * (size_t)C;
    const int o0 = nbuf * C;
    std::memcpy(bx + (size_t)nbuf * (size_t)inw, x, sizeof(float) * (size_t)inw);
    std::memcpy(bz + o0, z, n);   std::memcpy(bsg + o0, sg, n);
    std::memcpy(ba + o0, a, n);   std::memcpy(bgi + o0, gi, n);
    std::memcpy(bgf + o0, gf, n); std::memcpy(bgg + o0, gg, n);
    std::memcpy(bgo + o0, go, n); std::memcpy(btc + o0, tc, n);
    std::memcpy(bh + o0, h, n);   std::memcpy(bc + o0, c, n);
  }
}

// Per-bit path (unused in the release configuration, tbptt == 1): one optimiser step on
// the binary cross entropy of the bit that was just coded.  The recurrent
// path is truncated after ONE step, h_prev and c_prev are treated as
// constants.
template <int OPT, int PREC>
void Head::BackwardT(float y) {
  const int C = cells;
  // dL/d(delta) of -[y log p + (1-y) log(1-p)] with p = sigmoid(t + delta).
  // The clamp on delta is treated as straight-through: it only guards the
  // coder, it must not switch the learning off.
  const float dl = Sigmoidf(t + delta) - y;

  Hyp hp;
  hp.lr = lr; hp.mom = mom; hp.b2 = b2; hp.eps = eps;
  if (OPT == 3) {  // Adam bias correction, folded into the step size
    b1t *= (double)mom;
    b2t *= (double)b2;
    hp.lr = (float)((double)lr * std::sqrt(1.0 - b2t) / (1.0 - b1t));
  }

  // Output layer.  dh must read the OLD wout.
  for (int r = 0; r < C; ++r) da[r] = dl * wout[r];   // reused as dh here
  RowUpdate<OPT, PREC>(wout, m1wout, m2wout, h, dl, hp, C);
  ScalarUpdate<OPT, PREC>(&bout, &m1bout, &m2bout, dl, hp);

  // LSTM cell, one step.
  for (int r = 0; r < C; ++r) {
    const float dh = da[r];
    const float th = tc[r];
    const float o = go[r];
    const float dc = dh * o * (1.0f - th * th);
    dpre[3 * C + r] = dh * th * o * (1.0f - o);
    dpre[0 * C + r] = dc * gg[r] * gi[r] * (1.0f - gi[r]);
    dpre[2 * C + r] = dc * gi[r] * (1.0f - gg[r] * gg[r]);
    dpre[1 * C + r] = dc * cprev[r] * gf[r] * (1.0f - gf[r]);
  }
  for (int r = 0; r < C; ++r) da[r] = 0.0f;
  for (int k = 0; k < 4 * C; ++k) {
    const float d = dpre[k];
    RowUpdateAccum<OPT, PREC>(wih + k * C, m1wih + k * C, m2wih + k * C, a, d,
                              hp, da, C);
    RowUpdate<OPT, PREC>(whh + k * C, m1whh + k * C, m2whh + k * C, hprev, d,
                         hp, C);
    ScalarUpdate<OPT, PREC>(bg + k, m1bg + k, m2bg + k, d, hp);
  }

  // Input projection.  d/dz SiLU(z) = s + z*s*(1-s).
  for (int r = 0; r < C; ++r) {
    const float s = sg[r];
    const float dz = da[r] * (s + z[r] * s * (1.0f - s));
    RowUpdate<OPT, PREC>(win + r * inw, m1win + r * inw, m2win + r * inw, x, dz,
                         hp, inw);
    ScalarUpdate<OPT, PREC>(bin + r, m1bin + r, m2bin + r, dz, hp);
  }
}

// ---------------------------------------------------------------------------
// Buffered truncated BPTT over the whole window of seq coded bits (the path
// of the release configuration).
//
// The weights were CONSTANT for every step in the buffer, and the window
// starts from a real state reset (h = c = 0), so this gradient is the exact
// gradient of the window loss - not the one-step approximation the per-bit
// path uses.  This is the form the reference head was TRAINED with (seq-64
// TBPTT); the per-bit path is our cheap approximation of it.
//
// Cost: the same per-bit work as the per-bit path, plus one optimiser step
// per W bits instead of per bit.
// ---------------------------------------------------------------------------
void Head::BackwardWindow() {
  const int C = cells;
  const int W = nbuf;
  if (W <= 0) return;
  if (agrp == 0) {
    std::memset(gwin, 0, sizeof(float) * (size_t)(C * inw));
    std::memset(gbin, 0, sizeof(float) * (size_t)C);
    std::memset(gwih, 0, sizeof(float) * (size_t)(4 * C * C));
    std::memset(gwhh, 0, sizeof(float) * (size_t)(4 * C * C));
    std::memset(gbg, 0, sizeof(float) * (size_t)(4 * C));
    std::memset(gwout, 0, sizeof(float) * (size_t)C);
    gbout = 0.0f;
  }
  // Truncation at the window edge: no gradient flows back past step W-1.
  for (int r = 0; r < C; ++r) { dhn[r] = 0.0f; dcn[r] = 0.0f; }

  for (int t = W - 1; t >= 0; --t) {
    const int o0 = t * C;
    const float dl = bdl[t];
    // Output layer.
    for (int r = 0; r < C; ++r) gwout[r] += dl * bh[o0 + r];
    gbout += dl;
    for (int r = 0; r < C; ++r) da[r] = dl * wout[r] + dhn[r];

    // Cell.  hprev/cprev are the states AFTER step t-1; at t == 0 they are the
    // state the WINDOW started from.  With seq > 0 that is the reset, i.e.
    // exactly zero; with seq == 0 it is the carried-over state.
    const float* hprev_t = (t > 0) ? (bh + o0 - C) : wsh;
    const float* cprev_t = (t > 0) ? (bc + o0 - C) : wsc;
    for (int r = 0; r < C; ++r) {
      const float th = btc[o0 + r];
      const float go_ = bgo[o0 + r];
      const float gf_ = bgf[o0 + r];
      const float gi_ = bgi[o0 + r];
      const float gg_ = bgg[o0 + r];
      const float dc = da[r] * go_ * (1.0f - th * th) + dcn[r];
      dpre[3 * C + r] = da[r] * th * go_ * (1.0f - go_);
      dpre[0 * C + r] = dc * gg_ * gi_ * (1.0f - gi_);
      dpre[2 * C + r] = dc * gi_ * (1.0f - gg_ * gg_);
      dpre[1 * C + r] = dc * cprev_t[r] * gf_ * (1.0f - gf_);
      dcn[r] = dc * gf_;
    }
    for (int r = 0; r < C; ++r) { da[r] = 0.0f; dhn[r] = 0.0f; }
    const float* a_t = ba + o0;
    const float* hp_t = hprev_t;
    for (int k = 0; k < 4 * C; ++k) {
      const float d = dpre[k];
      GateRowBack(gwih + k * C, gwhh + k * C, da, dhn, wih + k * C,
                  whh + k * C, a_t, hp_t, d, C);
      gbg[k] += d;
    }
    // Input projection.
    const float* x_t = bx + (size_t)t * (size_t)inw;
    for (int r = 0; r < C; ++r) {
      const float sv = bsg[o0 + r];
      const float dz = da[r] * (sv + bz[o0 + r] * sv * (1.0f - sv));
      AxpyNoFma(gwin + (size_t)r * (size_t)inw, x_t, dz, inw);
      gbin[r] += dz;
    }
  }
  ++agrp;
  if (agrp >= accum) {
    FinishGroup();
    agrp = 0;
    nvalid = 0;
  }
}

void Head::FinishGroup() {
  const int C = cells;
  const int nw = C * inw, ng = 4 * C * C, nb = 4 * C;
  if (meang && nvalid > 0) {
    // Mean over the bits that actually carried a loss.  Override bits are
    // masked out and must not dilute the average.
    const float sc = 1.0f / (float)nvalid;
    for (int k = 0; k < nw; ++k) gwin[k] *= sc;
    for (int k = 0; k < C; ++k) gbin[k] *= sc;
    for (int k = 0; k < ng; ++k) gwih[k] *= sc;
    for (int k = 0; k < ng; ++k) gwhh[k] *= sc;
    for (int k = 0; k < nb; ++k) gbg[k] *= sc;
    for (int k = 0; k < C; ++k) gwout[k] *= sc;
    gbout *= sc;
  }
  if (clip > 0.0f) {
    // Global L2 norm over every parameter gradient, in one fixed order.  The
    // accumulation is double so that the norm does not depend on how the
    // compiler happened to schedule the additions.
    double ss = 0.0;
    for (int k = 0; k < nw; ++k) ss += (double)gwin[k] * (double)gwin[k];
    for (int k = 0; k < C; ++k) ss += (double)gbin[k] * (double)gbin[k];
    for (int k = 0; k < ng; ++k) ss += (double)gwih[k] * (double)gwih[k];
    for (int k = 0; k < ng; ++k) ss += (double)gwhh[k] * (double)gwhh[k];
    for (int k = 0; k < nb; ++k) ss += (double)gbg[k] * (double)gbg[k];
    for (int k = 0; k < C; ++k) ss += (double)gwout[k] * (double)gwout[k];
    ss += (double)gbout * (double)gbout;
    const double nrm = std::sqrt(ss);
    // A non-finite group gradient (inf/NaN from a diverging step) would turn
    // every weight into NaN: inf gives sc = 0 and 0*inf = NaN, NaN passes the
    // clip unchanged.  Drop that one step instead; the accumulators are
    // cleared at the next group start.  Tested on the exponent bits so that
    // no floating-point flag can fold the check away.  Never taken on a
    // healthy run, so archives stay bitidentical.
    std::uint64_t nb_bits;
    std::memcpy(&nb_bits, &ss, sizeof nb_bits);
    if (((nb_bits >> 52) & 0x7FFu) == 0x7FFu) return;
    if (nrm > (double)clip) {
      const float sc = (float)((double)clip / nrm);
      for (int k = 0; k < nw; ++k) gwin[k] *= sc;
      for (int k = 0; k < C; ++k) gbin[k] *= sc;
      for (int k = 0; k < ng; ++k) gwih[k] *= sc;
      for (int k = 0; k < ng; ++k) gwhh[k] *= sc;
      for (int k = 0; k < nb; ++k) gbg[k] *= sc;
      for (int k = 0; k < C; ++k) gwout[k] *= sc;
      gbout *= sc;
    }
  }
  Apply();
}

// One optimiser step on the accumulated window gradients.
template <int OPT, int PREC>
void Head::ApplyT() {
  const int C = cells;
  Hyp hp;
  hp.lr = lr; hp.mom = mom; hp.b2 = b2; hp.eps = eps;
  if (OPT == 3) {
    b1t *= (double)mom;
    b2t *= (double)b2;
    hp.lr = (float)((double)lr * std::sqrt(1.0 - b2t) / (1.0 - b1t));
  }
  RowUpdate<OPT, PREC>(wout, m1wout, m2wout, gwout, 1.0f, hp, C);
  ScalarUpdate<OPT, PREC>(&bout, &m1bout, &m2bout, gbout, hp);
  for (int k = 0; k < 4 * C; ++k) {
    RowUpdate<OPT, PREC>(wih + k * C, m1wih + k * C, m2wih + k * C,
                         gwih + k * C, 1.0f, hp, C);
    RowUpdate<OPT, PREC>(whh + k * C, m1whh + k * C, m2whh + k * C,
                         gwhh + k * C, 1.0f, hp, C);
    ScalarUpdate<OPT, PREC>(bg + k, m1bg + k, m2bg + k, gbg[k], hp);
  }
  for (int r = 0; r < C; ++r) {
    RowUpdate<OPT, PREC>(win + r * inw, m1win + r * inw, m2win + r * inw,
                         gwin + r * inw, 1.0f, hp, inw);
    ScalarUpdate<OPT, PREC>(bin + r, m1bin + r, m2bin + r, gbin[r], hp);
  }
}

void Head::Apply() { ApplyT<opt, prec>(); }

void Head::Backward(float y) { BackwardT<opt, prec>(y); }

}  // namespace

COLD_SIZE_ATTR bool Init() {
  if (g_init_done) return true;
  g_init_done = true;
  Head& s = g_head;
  s.log_path = nullptr;
  s.log_n = 8388608ULL;
  s.Build();
  return true;
}

int Cells() { return g_head.cells; }
int NumParams() { return g_head.nparams; }

// The per-byte channel.  All four are registered ONCE, in the coder's
// constructor and BEFORE Init() - every pointer handed over is stable for the
// whole run (the transformer's residual stream, the LSTM's hidden valarray,
// the predictor's scalar staging, the gate's context state), and the head
// sizes its projections from the widths.  A per-bit setter would be a call
// across the translation-unit boundary, and this unit is built WITHOUT -flto.
void SetAux(const float* hid, int n_hid) {
  g_head.aux_hid = hid;
  g_head.aux_n_hid = (hid != nullptr) ? n_hid : 0;
}
void SetTf(const float* tf, int n_tf) {
  g_head.aux_tf = tf;
  g_head.aux_n_tf = (tf != nullptr) ? n_tf : 0;
}
void SetLhid(const float* hid, int n_hid, int pre_projected) {
  g_head.aux_lhid = hid;
  g_head.aux_n_lhid = (hid != nullptr) ? n_hid : 0;
  g_head.aux_lhid_pre = pre_projected ? 1 : 0;
}
void SetZone(const unsigned int* zone) { g_head.aux_zone = zone; }

void Shutdown() {
  if (g_init_done) g_head.Free();
  g_init_done = false;
}

__attribute__((noinline))
unsigned int Adjust(unsigned int base_p, const float* stage1, int n_stage1,
                    float m1raw, int override_active) {
  Head& s = g_head;
  if (__builtin_expect(s.i == 0, 0) && n_stage1 != Head::n1) {
    std::fprintf(stderr,
        "\ncorrection model: stage-1 row width %d, compiled %d (rebuild with %d)\n",
        n_stage1, (int)Head::n1, n_stage1);
    std::fflush(stderr);
    std::abort();
  }

  // Sequence reset, as in the reference head: the recurrent state is zeroed
  // at the start of every window of seq bits (seq == 0 would mean never).
  if (s.seq > 0 && (s.tbptt ? (s.nbuf == 0)
                            : ((s.i % (unsigned long long)s.seq) == 0))) {
    for (int r = 0; r < s.cells; ++r) { s.h[r] = 0.0f; s.c[r] = 0.0f; }
  }
  // Remember the state the window starts from: the h_prev / c_prev of step 0
  // in the backward pass.
  // wsh/wsc are read by BackwardWindow() alone - a frozen head has no
  // backward pass and must not pay for them.
  if (s.tbptt && s.nbuf == 0 && !s.frozen) {
    std::memcpy(s.wsh, s.h, sizeof(float) * (size_t)s.cells);
    std::memcpy(s.wsc, s.c, sizeof(float) * (size_t)s.cells);
  }
  std::memcpy(s.hprev, s.h, sizeof(float) * (size_t)s.cells);
  std::memcpy(s.cprev, s.c, sizeof(float) * (size_t)s.cells);

  // ---- features ----
  const float p1 = (float)base_p / 65536.0f;
  const float t = std::log(p1 / (1.0f - p1));
  s.t = t;
  const float inv_t = 1.0f / kTScale;
  float* x = s.x;
  for (int j = 0; j < s.inw; ++j) x[j] = 0.0f;
  const int n1 = Head::n1keep;
  const float xs = s.xscale ? inv_t : 1.0f;
  // Dropped stage-1 columns (in the release configuration column 27, the
  // appended bit prediction of the byte LSTM) are not fed.  The row is
  // compacted, so the input matrix C*(nin+1) shrinks with it instead of
  // carrying a masked column.
  {
    int w = 0;
    for (int j = 0; j < Head::n1; ++j) {
      if (j >= STAGE1_COL(7) && j <= STAGE1_COL(20)) continue;
      if (j >= STAGE1_COL(27) && j <= STAGE1_COL(27)) continue;
      const float v = (j < n_stage1) ? stage1[j] : 0.0f;
      x[w++] = v * xs;
    }
  }
  x[n1 + 0] = m1raw * xs;
  const float tl = t * inv_t;
  x[n1 + 1] = tl;
  x[n1 + 2] = tl * tl;
  x[n1 + 3] = Sgn(t);
  x[n1 + 4 + (int)(s.i & 7)] = 1.0f;
  // sd is the SSE/APM correction in logit space: logit(sigmoid(m1raw)) is
  // m1raw itself, so sd = t - m1raw costs one subtraction.
  const float sd = t - m1raw;
  s.pend_sd = sd;
  if (s.feat & kFeatSse) {
    x[n1 + 12] = sd * inv_t;
    const double den = (double)((s.i < (unsigned long long)kSdCap)
                                    ? (s.i ? s.i : 1) : (unsigned long long)kSdCap);
    x[n1 + 13] = (float)(s.sd_sum / den) * inv_t;
  }
  x[Head::kPrev] = s.prev_bit;
  // The registered state vector.  Its sketch is recomputed once every eight
  // bits the head sees and held in between; without this cache the n-to-k
  // sketch costs n*k multiplications per bit and the head becomes several
  // times more expensive (measured: factor 3.0 at k = 16 without the cache,
  // 1.4 with it).
  if (s.aux_hid != nullptr) {
    if (s.hid_proj) {
      const unsigned long long b = s.i >> 3;
      if (b != s.hid_byte) {
        s.hid_byte = b;
        for (int r = 0; r < Head::nhid; ++r) {
          const float* pr = s.hidP + (size_t)r * s.aux_n_hid;
          float acc = 0.0f;
          for (int cc = 0; cc < s.aux_n_hid; ++cc) acc += pr[cc] * s.aux_hid[cc];
          s.hidC[r] = acc;
        }
      }
      for (int r = 0; r < Head::nhid; ++r) x[Head::kHid + r] = s.hidC[r];
    } else {
      for (int r = 0; r < Head::nhid; ++r) x[Head::kHid + r] = s.aux_hid[r];
    }
  } else {
    for (int r = 0; r < Head::nhid; ++r) x[Head::kHid + r] = 0.0f;
  }
  // The state vector of the byte-LSTM beside the transformer, same
  // cache (every eight bits the head sees).  `aux_lhid_pre` says the vector already IS the
  // sketch (a replay on a pre-projected trace): then its first nlhid columns
  // are exactly the nlhid-sketch, by the prefix property.
  if (s.aux_lhid != nullptr) {
    if (s.lhid_proj) {
      const unsigned long long b = s.i >> 3;
      if (b != s.lhid_byte) {
        s.lhid_byte = b;
        for (int r = 0; r < Head::nlhid; ++r) {
          const float* pr = s.lhidP + (size_t)r * s.aux_n_lhid;
          float acc = 0.0f;
          for (int cc = 0; cc < s.aux_n_lhid; ++cc) {
            acc += pr[cc] * s.aux_lhid[cc];
          }
          s.lhidC[r] = acc;
        }
      }
      for (int r = 0; r < Head::nlhid; ++r) x[Head::kLhid + r] = s.lhidC[r];
    } else {
      for (int r = 0; r < Head::nlhid; ++r) x[Head::kLhid + r] = s.aux_lhid[r];
    }
  } else {
    for (int r = 0; r < Head::nlhid; ++r) x[Head::kLhid + r] = 0.0f;
  }

  s.Forward();

  s.skip = override_active ? 1 : 0;
  if (override_active) {
    s.pend_base_p = base_p;
    s.pend_head_p = base_p;
    return base_p;
  }

  float d = s.delta;
  if (d > s.clamp) d = s.clamp;
  else if (d < -s.clamp) d = -s.clamp;
  const float pnew = Sigmoidf(t + d);
  unsigned int q = (unsigned int)(1.0f + 65534.0f * pnew);
  if (q < 1) q = 1;
  if (q > 65535) q = 65535;
  s.pend_base_p = base_p;
  s.pend_head_p = q;
  return q;
}

__attribute__((noinline))
void Observe(int bit) {
  Head& s = g_head;
  if (s.tbptt) {
    if (!s.frozen) {
      s.bdl[s.nbuf] = s.skip ? 0.0f
                             : (Sigmoidf(s.t + s.delta) - (float)bit);
      if (!s.skip) ++s.nvalid;
    }
    ++s.nbuf;
    if (s.nbuf >= s.wlen) {
      if (!s.frozen) s.BackwardWindow();
      s.nbuf = 0;
    }
  } else if (!s.skip && !s.frozen) {
    s.Backward((float)bit);
  }

  // |sd| window, kept only when the SSE feature bit is on.
  if (s.feat & kFeatSse) {
    const float v = std::fabs(s.pend_sd);
    if (s.i >= (unsigned long long)kSdCap) {
      s.sd_sum -= (double)s.sd_ring[(s.i - kSdCap) % kSdCap];
    }
    s.sd_sum += (double)v;
    s.sd_ring[s.i % kSdCap] = v;
  }
  s.prev_bit = (float)bit;
  s.i = s.i + 1;
  // Thaw the loaded head after the training range.  Only at a window
  // boundary (nbuf == 0): the next Adjust() then saves wsh/wsc and the window
  // buffers fill from step 0, so the first backward pass sees a complete
  // window.  Everything the backward pass and Adam read is still in its
  // initial state - moments, gradient accumulators, agrp/nvalid and the bias
  // correction powers b1t/b2t were never touched while frozen - so the head
  // continues exactly like an online head started from the blob weights.
  // `i` and `nbuf` run through identical values in encoder and decoder.
  if (s.frozen && s.nbuf == 0 &&
      s.i >= (unsigned long long)418837176) {
    s.frozen = 0;
    g_unfrozen = 1;
  }
}

int g_unfrozen = 0;

}

