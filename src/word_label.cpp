// Implementation of the log-bilinear word model; the header describes the
// mechanism, the declared divergences from the zmix instrument and the memory.
//
//   Origin: zmix 1.0 (James Byrne), GPL-3.0-or-later, build option -Dwordlbl
//   (src/models/wordlbl.zig).  Mechanism and swept hyper-parameters taken;
//   the code below is a port of that source, stamped onto OUR codeword geometry.
//
// The makefile rule `wordlbl` builds this file as a separate translation unit
// with strict FP semantics (WORDLBL_CXXFLAGS: -ffp-model=precise
// -ffp-contract=off -fno-math-errno -fno-builtin, no PGO, no LTO), so that
// the order of operations is fixed and encoder and decoder compute identical
// values.

#include "word_label.h"
#include "page_registry.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace word_label {
namespace {

// --------------------------------------------------------------------------
// Geometry and constants of the fixed form (header: FIXED FORM).
// --------------------------------------------------------------------------
constexpr int kD = 96;            // embedding / weight-row width (zmix sweep:
                                  // 97.7 % of the value of 192 at half the RAM)
constexpr int kNodeBits = 17;     // log2 of the trie-node table
constexpr uint32_t kNodeMask = (1u << kNodeBits) - 1u;
constexpr uint32_t kNodeCount = kNodeMask + 1u;
// Byte-0 nodes are 1..127, byte-1 nodes 128..16639, byte-2 nodes start here.
constexpr uint32_t kNodeBase2 = 16640u;
constexpr int kCtx = 16;          // words in the tied-decay context
// The literals are the correctly rounded floats of 60/1000, 30/1000 and
// 850/1000, i.e. the very values (float)N / 1000.0f had as build parameters.
constexpr float kEtaN = 0.060f;   // node rows and biases (all depths)
constexpr float kEtaE = 0.030f;   // embedding rows
constexpr float kLambda = 0.85f;  // decay per context word
static_assert(kD % 8 == 0, "the elementwise loops have no scalar tail");

// The three numbers the instrument does not expose as build options; they are
// constants here, not switches, because nothing was ever swept over them.
constexpr float kTopicRate = 0.004f;   // topic EMA (wordlbl.zig tokenDone)
constexpr float kTopicW = 0.25f;       // topic weight inside cvec
constexpr float kGradClamp = 0.5f;     // elementwise embedding-gradient clamp
constexpr float kLogitClamp = 20.0f;   // node logit clamp before the squash
constexpr double kInitScale = 0.05;    // embedding init half-width (mode 1)
constexpr uint64_t kSeed = 0x243F6A8885A308D3ull;  // instrument rnd() seed
inline float SquashF(float x) {
  // No +-30 clamp as in the instrument: the one caller (Forward) clamps the
  // logit to +-kLogitClamp (20) first, so it could never act.
  const float e = (float)std::exp((double)(-x));
  return 1.0f / (1.0f + e);
}

// splitmix-style avalanche with two constants of splitmix64 (public domain),
// then an AND with a NON-power-of-2 mask; the zmix source does exactly this
// at the byte-2 call site, so the port keeps it.
inline uint32_t Hsh(uint64_t x, uint32_t mask) {
  x *= 0x9E3779B97F4A7C15ull;
  x ^= x >> 29;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 32;
  return (uint32_t)x & mask;
}

// Node id: exact for codeword byte 0 and byte 1, hashed for byte 2.
inline uint32_t NodeId(int bytepos, uint32_t b0, uint32_t b1, uint32_t nodeix) {
  if (bytepos == 0) return nodeix;                        // 1..127
  if (bytepos == 1) return 128u + (b0 - 128u) * 128u + nodeix;  // <= 16639
  return kNodeBase2 +
      Hsh(((uint64_t)b0 << 20) ^ ((uint64_t)b1 << 8) ^ (uint64_t)nodeix,
          kNodeMask - kNodeBase2);
}

}  // namespace

// --------------------------------------------------------------------------
// The state.
// --------------------------------------------------------------------------
// The scalars stand first: their offsets fit a one-byte displacement on the
// per-bit path.  The state heads the one block of Create(); the tables follow.
struct State {
  // learned tables (inside the block, behind the state)
  float* nd;      // [kNodeCount][kD]  trie-node weight rows
  float* nb;      // [kNodeCount]      trie-node biases
  float* e1;      // [kVMax][kD]       the shared embedding table

  // per-bit state carried from Forward() to Perceive()
  uint32_t nodeix;
  uint32_t nid;
  float plbl;
  bool live;

  bool cvec_valid;
  bool escape_next;

  // causal parse of the coded byte stream (see the geometry note below)
  uint32_t breg;      // partial-byte register, 1 = byte boundary, >= 256 = full
  int first_bit;      // MSB of the current byte
  int expecting;      // 0 = token boundary, 1/2 = awaiting continuation byte
  uint32_t pfx0, pfx1;

  // word history
  uint32_t ring[kCtx];            // ring[0] = most recent
  float lampow[kCtx];             // lambda^s by iterated f32 multiply

  // per-token working state
  float topic[kD];
  float cvec[kD];
  float gacc[kD];

  // glibc hands a large calloc out at mmap + 16; with sizeof(State) = 48 mod
  // 64 the node and embedding rows (384 B) start on a cache line: 6 lines per
  // row instead of 7.  Layout only, no value depends on it.
  char pad_[48];
};
static_assert(sizeof(State) % 64 == 48, "rows on cache lines (see pad_)");

namespace {

// The instrument rnd(): a fixed-seed xorshift64 mapped to [0,1).
inline double Rnd(uint64_t* st) {
  *st ^= *st << 13;
  *st ^= *st >> 7;
  *st ^= *st << 17;
  return (double)(*st >> 11) / 9007199254740992.0;
}

inline void BuildCvec(State* s) {
  // tied decay: topic term first, then the ring, s ascending -- the same f32
  // order as the instrument.  Rebuilt once per token, lazily.
  for (int k = 0; k < kD; ++k) s->cvec[k] = kTopicW * s->topic[k];
  for (int sx = 0; sx < kCtx; ++sx) {
    const float w = s->lampow[sx];
    const float* e = s->e1 + (size_t)s->ring[sx] * kD;
    for (int k = 0; k < kD; ++k) s->cvec[k] += w * e[k];
  }
  s->cvec_valid = true;
}

// Back to the token boundary; the token gradient is spent.  The common end of
// TokenDone() and of a parse abort, therefore not inlined.
__attribute__((noinline)) void TokenEnde(State* s) {
  s->expecting = 0;
  s->cvec_valid = false;
  memset(s->gacc, 0, sizeof(s->gacc));
}

inline void ParseAbort(State* s) {
  // Invalid continuation: treat the bytes as literals and never reach token
  // end; discard the aborted token gradient and reset.  No abort occurred on
  // the measured inputs, so this is armour, not a hot path.
  TokenEnde(s);
}

void TokenDone(State* s, uint32_t r) {
  const size_t ri = (r < (uint32_t)kVMax) ? (size_t)r : (size_t)(kVMax - 1);
  // Each ring word descends the lambda^s-WEIGHTED gradient, clamped per
  // element AFTER the weighting; duplicates in the ring update sequentially.
  for (int sx = 0; sx < kCtx; ++sx) {
    const float w = s->lampow[sx];
    float* e = s->e1 + (size_t)s->ring[sx] * kD;
    for (int k = 0; k < kD; ++k) {
      float gk = kEtaE * w * s->gacc[k];
      gk = gk > kGradClamp ? kGradClamp : gk;     // two selections: the same
      gk = gk < -kGradClamp ? -kGradClamp : gk;   // values as if / else if
      e[k] -= gk;
    }
  }
  {   // topic EMA from the completed token row, read AFTER the updates
    const float* er = s->e1 + ri * kD;
    for (int k = 0; k < kD; ++k) s->topic[k] += kTopicRate * (er[k] - s->topic[k]);
  }
  memmove(&s->ring[1], &s->ring[0], (kCtx - 1) * sizeof(s->ring[0]));
  s->ring[0] = (uint32_t)ri;
  TokenEnde(s);   // also clears expecting, which nothing above reads
}

void ByteDone(State* s, uint32_t byte) {
  if (s->escape_next) {           // the escaped literal: consumed, no effect
    s->escape_next = false;
    return;
  }
  if (s->expecting == 0) {
    if (byte == 0x0Cu) {
      s->escape_next = true;
    } else if (byte < 0x80u) {
      // literal: the model uses only word history
    } else if (byte <= 0xCFu) {
      TokenDone(s, byte - 0x80u);                       // band 1
    } else {
      s->pfx0 = byte;                                   // continuation follows
      s->expecting = 1;
    }
    return;
  }
  if (s->expecting == 1) {
    if (byte >= 0x80u && byte <= 0xCFu) {
      TokenDone(s, 80u + (s->pfx0 - 0xD0u) * 80u + (byte - 0x80u));   // band 2
    } else if (byte >= 0xD0u) {
      s->pfx1 = byte;
      s->expecting = 2;
    } else {
      ParseAbort(s);
    }
    return;
  }
  // expecting == 2
  if (byte >= 0x80u && byte <= 0xCFu) {
    if (s->pfx0 >= 0xF0u && s->pfx1 <= 0xEFu) {
      TokenDone(s, 3920u + 80u * (32u * (s->pfx0 - 0xF0u) + (s->pfx1 - 0xD0u))
                + (byte - 0x80u));                                    // band 3
    } else if (s->pfx0 >= 0xF0u) {
      TokenDone(s, 44880u + (s->pfx0 - 0xF0u) * 1280u
                + (s->pfx1 - 0xF0u) * 80u + (byte - 0x80u));           // band 4
    } else {
      ParseAbort(s);
    }
  } else {
    ParseAbort(s);
  }
  // every branch above ends in TokenEnde(): expecting is 0 here
}

}  // namespace

// --------------------------------------------------------------------------
// Public interface.
// --------------------------------------------------------------------------

State* Create() {
  // ONE allocation: the state, then node rows, embeddings and biases.  calloc
  // hands out zero pages that stay lazy until touched (header, divergence C),
  // and a reset gives the whole block back with one free().
  constexpr size_t kNd = (size_t)kNodeCount * kD;
  constexpr size_t kE = (size_t)kVMax * kD;
  constexpr size_t kBytes = sizeof(State) + (kNd + kE + kNodeCount) * sizeof(float);
  State* s = (State*)calloc(kBytes, 1);
  PAGE_REGISTRY_ADD(s, kBytes);
  if (s == nullptr) abort();   // no 69 MiB: a host fault, SIGABRT is loud
  float* t = (float*)(s + 1);
  s->nd = t;
  s->e1 = t + kNd;
  s->nb = t + kNd + kE;
  // Fixed-seed xorshift, the instrument draw order, no burn-in.
  uint64_t st = kSeed;
  // The factor 2.0 * kInitScale folds to the double 0.1 exactly and x * 2.0
  // is exact, so the product is the instrument's (x * 2.0) * kInitScale.
  for (size_t i = 0; i < kE; ++i)
    s->e1[i] = (float)((Rnd(&st) - 0.5) * (2.0 * kInitScale));
  float w = 1.0f;
  for (int i = 0; i < kCtx; ++i) { s->lampow[i] = w; w *= kLambda; }
  s->breg = 1;
  s->nodeix = 1;
  return s;
}

State* Neu(State* alt) {
  free(alt);
  return Create();
}

float Forward(State* s) {
  s->live = false;
  if (s->breg == 1) return 0.0f;       // the MSB: codeword / literal flag
  if (s->escape_next) return 0.0f;     // the escaped literal byte
  const int bp = s->expecting;
  if (bp == 0 && s->first_bit == 0) return 0.0f;   // literal byte
  if (!s->cvec_valid) BuildCvec(s);
  const uint32_t nid = NodeId(bp, s->pfx0, s->pfx1, s->nodeix);
  const float* nv = s->nd + (size_t)nid * kD;
  float zl = s->nb[nid];
  for (int k = 0; k < kD; ++k) zl += s->cvec[k] * nv[k];
  zl = zl > kLogitClamp ? kLogitClamp : zl;
  zl = zl < -kLogitClamp ? -kLogitClamp : zl;
  s->nid = nid;
  s->plbl = SquashF(zl);
  s->live = true;
  return zl;
}

#if PORT_CMX_MEAS
void PerceiveG(State* s, int bit, float extra) {
  const int y = bit & 1;
  if (s->live) {
    const float g = (s->plbl - (float)y) + extra;
    float* nv = s->nd + (size_t)s->nid * kD;
    const float ge = kEtaN * g;
    for (int k = 0; k < kD; ++k) {
      s->gacc[k] += g * nv[k];
      nv[k] -= ge * s->cvec[k];
    }
    s->nb[s->nid] -= ge;
    s->nodeix = s->nodeix * 2u + (uint32_t)y;
    s->live = false;
  }
  if (s->breg == 1) s->first_bit = y;
  s->breg = (s->breg << 1) | (uint32_t)y;
  if (s->breg < 256u) return;
  const uint32_t byte = s->breg & 0xFFu;
  s->breg = 1;
  s->nodeix = 1;
  ByteDone(s, byte);
}
#endif

void Perceive(State* s, int bit) {
  const int y = bit & 1;
  if (s->live) {
    const float g = s->plbl - (float)y;
    float* nv = s->nd + (size_t)s->nid * kD;
    const float ge = kEtaN * g;
    // Instrument order: gacc from the OLD node row, then descend that row.
    for (int k = 0; k < kD; ++k) {
      s->gacc[k] += g * nv[k];
      nv[k] -= ge * s->cvec[k];
    }
    s->nb[s->nid] -= ge;
    s->nodeix = s->nodeix * 2u + (uint32_t)y;
    s->live = false;
  }
  if (s->breg == 1) s->first_bit = y;
  s->breg = (s->breg << 1) | (uint32_t)y;
  if (s->breg < 256u) return;
  const uint32_t byte = s->breg & 0xFFu;
  s->breg = 1;
  s->nodeix = 1;
  ByteDone(s, byte);
}

}  // namespace word_label
