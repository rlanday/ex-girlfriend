// bit_head.cpp: the BITLSTM in the form of the replay trainer.  See
// bit_head.h for what it is and how this copy relates to the compressor's.
//
// This translation unit MUST be compiled with value-safe floating point
// (-ffp-model=precise -ffp-contract=off, see build.sh) and its entry points
// are noinline: encoder, decoder and trainer then execute the identical
// arithmetic, so a model trained here behaves the same in the coder.
//
// ATTRIBUTION.  The AVX2 kernels Exp8/Sigmoid8/Tanh8 (his Silu8 is inlined as
// z * Sigmoid8(z)) and the fixed-order DotPadded reduction below are ported
// line by line from src/models/bitlstm32-head.cpp of cmix-obias (David
// Freelan, GPL-3.0); Exp8 is itself the classic Cephes expf polynomial in 8
// lanes.  The forward architecture (in_proj -> SiLU -> LSTMCell -> linear,
// logit-additive correction after Discretize) follows the same file.
// Everything else here (the backward pass, the optimiser, the feature set,
// the parameters) is our own.  Ported and modified for this entry in 2026.


#include "bit_head.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "isa_floor.h"
#include <immintrin.h>

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
//   +4..+11   one-hot(8) of the coded-bit count modulo 8
constexpr int kBaseExtra = 12;
// Optional SSE block (feature bit 0, off here): sd/T and mean|sd| over
// 256 / T.
constexpr int kSseExtra = 2;

constexpr int kNTf = 0;   // the transformer's per-byte scalars are not fed

// Ring length of the permutation control (REPLAY_CTRL_*, a diagnostic that
// was not used for the shipped weights): the value of a column is replaced by
// a pseudo-randomly drawn earlier value of the same column.  That keeps the
// column's distribution and removes its information about this bit.
constexpr int kSurRing = 1 << 20;

// splitmix64 finaliser, used as a counter-based PRNG for the control.
inline uint64_t Mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

int EnvInt(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  return (int)std::strtol(v, nullptr, 10);
}
unsigned long long EnvU64(const char* name, unsigned long long dflt) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  return std::strtoull(v, nullptr, 10);
}

inline float Sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float Sgn(float x) { return (x > 0.0f) ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

// ---------------------------------------------------------------------------
// AVX2 kernels (ported, see ATTRIBUTION above)
// ---------------------------------------------------------------------------
// 8-lane expf, Cephes polynomial. Fixed coefficients and fixed operation
// order -> deterministic.
// Altered version of exp256_ps from avx_mathfun.h (Giovanni Garberoglio, 2012),
// itself based on sse_mathfun.h (Julien Pommier, 2007); zlib licence, notices in
// third_party/mathfun-LICENSE.txt of the package.  Polynomial: Cephes expf.
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

// Backward-window kernels.  Deliberately mul-then-add and NOT fmadd: that is
// exactly what the scalar form `acc[j] += s*v[j]` computes under
// -ffp-contract=off, so vectorising here changes the speed and not a single
// bit of the result.  Fused into one pass over the row because the rows are
// short (C = 16..64).
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
// bf16 weight rounding (kPrec != 0; the trainer keeps float weights)
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
// no branch.
//
//   OPT 0  plain SGD          w -= lr*g
//   OPT 1  SGD + momentum     m = mom*m + g;                 w -= lr*m
//   OPT 2  RMSProp            s = rho*s + (1-rho)*g^2;       w -= lr*g/(sqrt(s)+eps)
//   OPT 3  Adam               m,s as usual, bias corrected;  w -= lrt*m/(sqrt(s)+eps)
//
// The trainer uses OPT 3 (Adam).
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

// Deterministic splitmix64.  The initial weights must be identical in encoder
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

// eps = 10^-kEpsExp, folded at compile time.
constexpr double PowTenNeg(int n) {
  return (n <= 0) ? 1.0 : (PowTenNeg(n - 1) / 10.0);
}
constexpr float kEps = (float)PowTenNeg(kEpsExp);

struct Head {
  // ---- geometry (compile time) ----
  static constexpr int cells = kCells;
  static constexpr int n1 = kRowWidth;   // stage-1 row width
  static constexpr int nsse = ((kFeat) & kFeatSse) ? kSseExtra : 0;
  //   n1keep  stage-1 columns actually fed (two ranges are dropped)
  //   nhid    columns of the transformer-state sketch
  //   ntf     columns of the transformer's per-byte scalars (none)
  //   nlhid   columns of the byte-LSTM-state sketch
  //   nzone   columns of a gate-zone one-hot (none)
  static constexpr int ndrop = kDropHi - kDropLo + 1;
  static constexpr int ndrop2 = kDrop2Hi - kDrop2Lo + 1;
  static constexpr int n1keep = n1 - ndrop - ndrop2;
  static constexpr int nhid = kHidSketch;
  static constexpr int ntf = kNTf;
  static constexpr int nlhid = kLhidSketch;
  static constexpr int nzone = 0;
  // Layout of the feature row:
  //   [0 .. n1keep-1]              stage-1 row (selected columns)
  //   [n1keep+0]                   m1raw
  //   [n1keep+1 .. n1keep+3]       t/T, (t/T)^2, sgn(t)
  //   [n1keep+4 .. n1keep+11]      coded-bit count mod 8, one-hot(8)
  //   [n1keep+12, n1keep+13]       SSE pair (only with feat & kFeatSse)
  //   [kPrev]                      prev_bit
  //   [kHid ..] [kTf ..] [kLhid ..] [kZone ..]
  static constexpr int kPrev = n1keep + kBaseExtra + nsse;
  static constexpr int kHid = kPrev + 1;
  static constexpr int kTf = kHid + nhid;
  static constexpr int kLhid = kTf + ntf;
  static constexpr int kZone = kLhid + nlhid;
  static constexpr int nin =                // features without prev_bit
      n1keep + kBaseExtra + nsse + nhid + ntf + nlhid + nzone;
  static constexpr int inw = ((nin + 1) + 7) & ~7;   // padded to 8

  // ---- parameters (compile time, except the learning rate) ----
  static constexpr float lr = (float)(kLr1e6 / 1000000.0);
  // The trainer follows a learning-rate schedule over several passes, so the
  // rate is read at run time (REPLAY_LR1E6); the constant is the default.
  float lr_rt = lr;
  static constexpr float mom = (float)(kBeta1_1e6 / 1000000.0);
  static constexpr float b2 = (float)(kBeta2_1e6 / 1000000.0);
  static constexpr float eps = kEps;
  static constexpr float clamp = (float)(kClamp1000 / 1000.0);
  static constexpr int opt = kOpt;
  static constexpr int prec = kPrec;
  static constexpr int seq = kSeq;
  static constexpr int feat = kFeat;
  double b1t = 1.0, b2t = 1.0;   // OPT3 bias-correction powers
  // The gradients of `accum` windows are summed and applied in one step.  The
  // weights do not move inside such a group, so the sum is the exact gradient
  // of the group loss.
  static constexpr int accum = kAccum;   // windows per step
  static constexpr int meang = kMeanGrad;  // 1 = mean loss
  // global L2 norm clip over the group gradient
  static constexpr float clip = (float)(kClip1000 / 1000.0);
  // 1 = stage-1 row and m1raw through T, 0 = raw
  static constexpr int xscale = kXScale;
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

  // ---- the per-byte channel ----
  // Registered before Init(); constant across the eight bits of a byte.
  // Nothing computed from these re-enters the predictor.
  const float* aux_hid = nullptr;   // transformer state (or its sketch)
  int aux_n_hid = 0;
  const float* aux_tf = nullptr;    // transformer per-byte scalars
  int aux_n_tf = 0;
  const float* aux_lhid = nullptr;  // byte-LSTM state (or its sketch)
  int aux_n_lhid = 0;
  int aux_lhid_pre = 0;             // the vector IS already the sketch
  const unsigned int* aux_zone = nullptr;   // gate zone of this byte
  int hid_proj = 0;                 // 1 = fixed +-1/sqrt(n) sketch here
  float* hidP = nullptr;            // nhid x aux_n_hid
  float* hidC = nullptr;            // nhid, the sketch OF THIS BYTE
  unsigned long long hid_byte = ~0ULL;   // byte hidC belongs to
  int lhid_proj = 0;
  float* lhidP = nullptr;
  float* lhidC = nullptr;
  unsigned long long lhid_byte = ~0ULL;
  // 1 = forward only, no learning (REPLAY_FREEZE_AT)
  int frozen = 0;
  const char* load_path = nullptr;  // REPLAY_LOAD: weights to start from
  const char* dump_path = nullptr;  // REPLAY_DUMP: weights written at the end
  // ---- run-time options of the trainer ----
  uint64_t seed = 0x42495431ULL;    // REPLAY_SEED, init draw ("BIT1")
  unsigned long long freeze_at = 0ULL;   // REPLAY_FREEZE_AT
  // Permutation control of the two state blocks (REPLAY_CTRL_*): the block
  // keeps its width and distribution and loses its information.
  int ctrlhid = 0, ctrllhid = 0, ctrlzone = 0;
  float* sur_ring = nullptr;        // nring x kSurRing
  int nring = 0;
  int sur_idx[(nhid + nlhid) > 0 ? (nhid + nlhid) : 1] = {0};
  int* zone_ring = nullptr;         // kSurRing, control of the zone

  // ---- SSE-correction window ----
  float* sd_ring = nullptr;
  double sd_sum = 0.0;
  float pend_sd = 0.0f;

  // ---- TBPTT window ----
  // Buffered truncated BPTT over the window: the weights stay constant for
  // the whole window, so the gradient is EXACT for it (the window boundary is
  // a real state reset, not an approximation), and the optimiser step is paid
  // once per W bits instead of once per bit.
  static constexpr int tbptt = kTbptt;
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

  // ---- cost counters of an optional report (unused here) ----
  const char* log_path = nullptr;
  unsigned long long log_n = 8388608ULL;   // report every N coded bits
  double log_base = 0.0, log_head = 0.0;   // bits, current window
  double log_base_tot = 0.0, log_head_tot = 0.0;
  unsigned int pend_base_p = 32768, pend_head_p = 32768;

  int nparams = 0;

  void Build();
  void LoadBlob();
  void DumpBlob();
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
  std::free(sur_ring); std::free(zone_ring);
  std::memset(this, 0, sizeof(*this));
}

void Head::Build() {
  const int C = cells;

  // A state block wider than the vector registered for it would read past
  // that vector's end; with nothing registered it would read zeros.  Both
  // stop the run.
  {
    const char* what = nullptr;
    int want = 0, have = 0;
    if (nhid > aux_n_hid) { what = "transformer state"; want = nhid; have = aux_n_hid; }
    if (what == nullptr && nlhid > aux_n_lhid) {
      what = "byte-LSTM state"; want = nlhid; have = aux_n_lhid;
    }
    if (what != nullptr) {
      std::fprintf(stderr,
          "\nBITLSTM: the %s block needs %d values, %d are "
          "registered.\n",
          what, want, have);
      std::fflush(stderr);
      std::abort();
    }
  }
  // The fixed +-1/sqrt(n) sketch (Johnson-Lindenstrauss) of the transformer
  // state.  Its draw stream is its own and does not depend on the seed: the
  // projection is part of the feature set, not of the initialisation.  Rows
  // are drawn one after another, so the k-sketch is the prefix of every wider
  // one.  Not needed when the registered vector already is the sketch.
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
  // The same for the byte-LSTM state, with an own draw stream, so that the
  // two vectors do not share a projection.
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
  // Only a control run with a permuted column pays for the ring.
  {
    nring = 0;
    if (ctrlhid) for (int k = 0; k < nhid; ++k) sur_idx[nring++] = kHid + k;
    if (ctrllhid) for (int k = 0; k < nlhid; ++k) sur_idx[nring++] = kLhid + k;
    if (nring > 0) sur_ring = Alloc(nring * kSurRing);
    if (ctrlzone && nzone > 0) {
      zone_ring = (int*)std::calloc((size_t)kSurRing, sizeof(int));
    }
  }

  win = Alloc(C * inw);   bin = Alloc(C);
  wih = Alloc(4 * C * C); whh = Alloc(4 * C * C);
  bg = Alloc(4 * C);      wout = Alloc(C);
  // Always allocated, even for plain SGD: the update kernels form
  // `m1wih + k*C` unconditionally, and pointer arithmetic on a null pointer is
  // undefined behaviour even when the result is never dereferenced.
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
  // reaches in_proj and the model is dead), but the OUTPUT layer starts at
  // exactly zero.  delta is therefore 0 on the first bit and the model begins
  // as an exact no-op on the coder; it can only move away from the base
  // prediction once it has seen evidence.  The trainer takes the seed from
  // REPLAY_SEED; a weight file named in REPLAY_LOAD overwrites the draw.
  Rng rng(seed);
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
  // Forget-gate bias +1 (the classic Gers/Schmidhuber initialisation), so
  // that the cell starts out remembering rather than forgetting.
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
  // Warm start: overwrite the drawn weights with those of the previous pass.
  // It stands after the draw, so the draw stream is the same with and
  // without a weight file.
  LoadBlob();
}

// The weight file, `NBLB`.  Native little-endian, uncompressed:
//   char[4] "NBLB" | u32 version | i32 {cells, nin, inw, n1} |
//   win[C*inw] bin[C] wih[4CC] whh[4CC] bg[4C] wout[C] bout
// Version 1 stores the values as f32, version 2 as bf16 (the upper half of
// the f32 pattern, expanded exactly by a 16-bit shift).  The trainer writes
// version 1; the shipped weights are version 2 (tools/nblb.py).  Versions 4
// and 5 (with a learned projection) belong to another form of the model.  The
// four geometry words must match the model EXACTLY: a file of a different
// input form is not a worse model, it is a different feature set.
void Head::LoadBlob() {
  // Without a file the first pass starts from the drawn weights.
  if (load_path == nullptr) return;
  const char* path = load_path;
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "BITLSTM: cannot open %s\n", path);
    std::exit(3);
  }
  char mg[4]; std::uint32_t ver = 0; std::int32_t g[4] = {0, 0, 0, 0};
  if (std::fread(mg, 1, 4, f) != 4 || std::memcmp(mg, "NBLB", 4) != 0) {
    std::fprintf(stderr, "BITLSTM: %s is not a weight file\n", path);
    std::exit(3);
  }
  if (std::fread(&ver, 4, 1, f) != 1 ||
      std::fread(g, sizeof(std::int32_t), 4, f) != 4) {
    std::fprintf(stderr, "BITLSTM: %s: header too short\n", path);
    std::exit(3);
  }
  if (ver != 1u && ver != 2u && ver != 4u && ver != 5u) {
    std::fprintf(stderr, "BITLSTM: weight file version %u, expected 1, 2, 4 or 5\n",
                 (unsigned)ver);
    std::exit(3);
  }
  if (ver == 4u || ver == 5u) {
    std::fprintf(stderr,
        "BITLSTM: weight file version %u carries a learned "
        "projection; this model uses the fixed sketch.\n", (unsigned)ver);
    std::exit(3);
  }
  if (g[0] != cells || g[1] != nin || g[2] != inw || g[3] != n1) {
    std::fprintf(stderr,
        "BITLSTM: weight file geometry %d/%d/%d/%d, model %d/%d/%d/%d "
        "(cells/nin/inw/n1)\n",
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
    std::fprintf(stderr, "BITLSTM: %s: short read\n", path);
    std::exit(3);
  }
  // Warm start: the weights are loaded and learning continues.  Only the
  // weights carry over between passes, the optimiser moments start at zero.
  frozen = 0;
  std::fprintf(stderr,
      "BITLSTM: loaded %s (C=%d nin=%d inw=%d n1=%d)\n",
      path, cells, nin, inw, n1);
}

// The counterpart, written at Shutdown, always as version 1 (f32).
// dump -> load -> dump is byte-identical.
void Head::DumpBlob() {
  if (dump_path == nullptr) return;
  FILE* f = std::fopen(dump_path, "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "BITLSTM: cannot write %s\n", dump_path);
    return;
  }
  const int C = cells;
  const size_t nw = (size_t)C * (size_t)inw;
  const size_t ng = (size_t)4 * (size_t)C * (size_t)C;
  const size_t nb = (size_t)4 * (size_t)C;
  std::uint32_t ver = 1;
  std::int32_t g[4] = {cells, nin, inw, n1};
  std::fwrite("NBLB", 1, 4, f);
  std::fwrite(&ver, 4, 1, f);
  std::fwrite(g, sizeof(std::int32_t), 4, f);
  std::fwrite(win, sizeof(float), nw, f);
  std::fwrite(bin, sizeof(float), (size_t)C, f);
  std::fwrite(wih, sizeof(float), ng, f);
  std::fwrite(whh, sizeof(float), ng, f);
  std::fwrite(bg, sizeof(float), nb, f);
  std::fwrite(wout, sizeof(float), (size_t)C, f);
  std::fwrite(&bout, sizeof(float), 1, f);
  std::fclose(f);
  std::fprintf(stderr, "BITLSTM: wrote %s (version %u, %d params)\n",
               dump_path, (unsigned)ver, nparams);
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

// Per-bit path (not used here, tbptt == 1): one optimiser step on the binary
// cross entropy of the bit that was just coded.  The recurrent path is
// truncated after ONE step, h_prev and c_prev are treated as constants.
template <int OPT, int PREC>
void Head::BackwardT(float y) {
  const int C = cells;
  // dL/d(delta) of -[y log p + (1-y) log(1-p)] with p = sigmoid(t + delta).
  // The clamp on delta is treated as straight-through: it only guards the
  // coder, it must not switch the learning off.
  const float dl = Sigmoidf(t + delta) - y;

  Hyp hp;
  const float lr = lr_rt;           // the run-time rate shadows the constant
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
// used here).
//
// The weights were CONSTANT for every step in the buffer, and the window
// starts from a real state reset (h = c = 0), so this gradient is the exact
// gradient of the window loss, not the one-step approximation of the per-bit
// path.
// ---------------------------------------------------------------------------
void Head::BackwardWindow() {
  const int C = cells;
  const int W = nbuf;
  if (W <= 0) return;
  // The accumulators are cleared once per GROUP of `accum` windows.
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
  // One optimiser step per group of `accum` windows.
  ++agrp;
  if (agrp >= accum) {
    FinishGroup();
    agrp = 0;
    nvalid = 0;
  }
}

// Turns the accumulated group gradient into the one it is stepped with (mean
// over the valid bits, global L2 norm clip) and takes the step.
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
  const float lr = lr_rt;           // the run-time rate shadows the constant
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

bool Init() {
  if (g_init_done) return true;
  g_init_done = true;
  Head& s = g_head;
  s.log_path = nullptr;
  s.log_n = 8388608ULL;
  // Run-time options of the trainer.  None of them changes the geometry.
  s.seed = EnvU64("REPLAY_SEED", 0x42495431ULL);
  s.freeze_at = EnvU64("REPLAY_FREEZE_AT", 0ULL);
  s.ctrlhid = EnvInt("REPLAY_CTRL_HID", 0) ? 1 : 0;
  s.ctrllhid = EnvInt("REPLAY_CTRL_LHID", 0) ? 1 : 0;
  s.ctrlzone = EnvInt("REPLAY_CTRL_ZONE", 0) ? 1 : 0;
  { const char* lr_s = std::getenv("REPLAY_LR1E6");
    if (lr_s != nullptr && *lr_s != '\0')
      s.lr_rt = (float)(std::atof(lr_s) / 1000000.0); }
  { const char* lp = std::getenv("REPLAY_LOAD");
    s.load_path = (lp != nullptr && *lp != '\0') ? lp : nullptr; }
  { const char* dp = std::getenv("REPLAY_DUMP");
    s.dump_path = (dp != nullptr && *dp != '\0') ? dp : nullptr; }
  s.Build();
  return true;
}

int Cells() { return g_head.cells; }
int NumParams() { return g_head.nparams; }

// The per-byte channel.  All four pointers are registered once, before
// Init(), and stay valid for the whole run; Build() sizes the projections
// from the widths.
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

int NumInputs() { return Head::nin + 1; }
int Wlen() { return g_head.wlen; }

void Shutdown() {
  if (g_init_done) g_head.DumpBlob();   // before Free()
  if (g_init_done) g_head.Free();
  g_init_done = false;
}

__attribute__((noinline))
unsigned int Adjust(unsigned int base_p, const float* stage1, int n_stage1,
                    float m1raw, int override_active) {
  Head& s = g_head;
  // The row width is compiled in; the caller passes the width it really
  // has.  They must agree, or the feature row would be read past its end.
  // Checked on the first coded bit only.
  if (__builtin_expect(s.i == 0, 0) && n_stage1 != Head::n1) {
    std::fprintf(stderr,
        "\nBITLSTM: the stage-1 row has width %d, the model is "
        "built for %d (a trace of width %d needs a matching build).\n",
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
  // in the backward pass.  A frozen model has no backward pass.
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
  const int n1 = Head::n1keep;   // base offset of the block behind the row
  const float xs = s.xscale ? inv_t : 1.0f;
  // The dropped stage-1 columns are not fed at all: the row is compacted, so
  // the input matrix C*(nin+1) shrinks with it instead of carrying masked
  // columns.
  {
    int w = 0;
    for (int j = 0; j < Head::n1; ++j) {
      if (j >= kDropLo && j <= kDropHi) continue;
      if (j >= kDrop2Lo && j <= kDrop2Hi) continue;
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
  // The transformer state and its sketch: recomputed once every eight bits
  // the head sees and held in between.
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
  // The byte-LSTM state, same once-per-byte cache.  `aux_lhid_pre` says the
  // vector already IS the sketch: then its first nlhid values are exactly the
  // nlhid-sketch, by the prefix property.
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
  // Permutation control of the state blocks (diagnostic, off unless
  // REPLAY_CTRL_HID / REPLAY_CTRL_LHID is set).  It runs on the finished
  // column values, so the ring holds the column's own distribution.
  if (s.sur_ring != nullptr) {
    for (int k = 0; k < s.nring; ++k) {
      const int col = s.sur_idx[k];
      const float v = x[col];
      float* r = s.sur_ring + (size_t)k * kSurRing;
      const int pos = (int)(s.i % (unsigned long long)kSurRing);
      r[pos] = v;
      const unsigned long long fill =
          (s.i + 1 < (unsigned long long)kSurRing)
              ? (s.i + 1) : (unsigned long long)kSurRing;
      if (fill <= 1) continue;
      // Drawn from the ring EXCLUDING the value just pushed, so the surrogate
      // never accidentally equals the true feature of this bit.
      unsigned long long d =
          Mix64(s.i * 0x9E3779B97F4A7C15ULL +
                (unsigned long long)(col + 1) * 0xC2B2AE3D27D4EB4FULL) %
          (fill - 1);
      if (d >= (unsigned long long)pos) ++d;
      x[col] = r[d];
    }
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
  // `frozen` switches the entire learning path off.  nbuf keeps cycling
  // because it is also the window position that drives the state reset in
  // Adjust(): a frozen model keeps the reset rhythm it was trained under.
  // Override bits are loss-masked: they contribute a zero gradient but still
  // occupy a window slot.
  if (s.tbptt) {
    if (!s.frozen) {
      s.bdl[s.nbuf] = s.skip ? 0.0f
                             : (Sigmoidf(s.t + s.delta) - (float)bit);
      if (!s.skip) ++s.nvalid;   // denominator of the mean-gradient reduction
    }
    ++s.nbuf;
    if (s.nbuf >= s.wlen) {
      if (!s.frozen) s.BackwardWindow();
      s.nbuf = 0;
    }
  } else if (!s.skip && !s.frozen) {
    s.Backward((float)bit);
  }

  // |sd| window (only fed when the SSE feature is on).
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
  // REPLAY_FREEZE_AT: stop learning after that many coded bits (a diagnostic,
  // not used for the shipped weights).  A group still accumulating at that
  // point is dropped.
  if (s.freeze_at != 0ULL && !s.frozen && s.i >= s.freeze_at) s.frozen = 1;
}

}  // namespace bit_head
