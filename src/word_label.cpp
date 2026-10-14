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
#include "port_cfg.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace word_label {
namespace {

// --------------------------------------------------------------------------
// Compile-time geometry and constants.
// --------------------------------------------------------------------------
constexpr int kD = 96;
constexpr int kNodeBits = 17;
constexpr uint32_t kNodeMask = (1u << kNodeBits) - 1u;
constexpr uint32_t kNodeCount = kNodeMask + 1u;
// Byte-0 nodes are 1..127, byte-1 nodes 128..16639, byte-2 nodes start here.
constexpr uint32_t kNodeBase2 = 16640u;
constexpr int kVMax = 54857;
constexpr int kCtx = 16;

constexpr float kEtaN = (float)60 / 1000.0f;
constexpr float kEtaE = (float)30 / 1000.0f;
#if PORT_ETA0
// WL2 eta0 (Z: -Dwordlbl-eta0=0.0225): its own eta for byte-0 trie nodes.
constexpr float kEta0 = 0.0225f;
constexpr bool kEta0Live = true;
#else
constexpr float kEta0 = (float)60 / 1000.0f;
constexpr bool kEta0Live = (60 != 60);
#endif
constexpr float kLambda = (float)850 / 1000.0f;

// The three numbers the instrument does not expose as build options; they are
// constants here, not switches, because nothing was ever swept over them.
constexpr float kTopicRate = 0.004f;   // topic EMA (wordlbl.zig tokenDone)
constexpr float kTopicW = 0.25f;       // topic weight inside cvec
constexpr float kGradClamp = 0.5f;     // elementwise embedding-gradient clamp
constexpr float kLogitClamp = 20.0f;   // node logit clamp before the squash
constexpr double kInitScale = 0.05;    // embedding init half-width (mode 1)
constexpr uint64_t kSeed = 0x243F6A8885A308D3ull;
inline float SquashF(float x) {
  if (x < -30.0f) x = -30.0f;
  else if (x > 30.0f) x = 30.0f;
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
struct State {
  // learned tables
  float* nd;      // [kNodeCount][kD]  trie-node weight rows
  float* nb;      // [kNodeCount]      trie-node biases
  float* e1;      // [kVMax][kD]       (CTX>0: THE shared table)
  float* e2;      // [kVMax][kD]       v1 only
  float* e3;      // [kVMax][kD]       v1 only
  float topic[kD];

  // per-token working state
  float cvec[kD];
  float gacc[kD];
  bool cvec_valid;

  // word history
  uint32_t w1, w2, w3;            // v1
  uint32_t ring[kCtx];            // v2, ring[0] = most recent
  float lampow[kCtx];             // lambda^s by iterated f32 multiply

  // causal parse of the coded byte stream (see the geometry note below)
  uint32_t breg;      // partial-byte register, 1 = byte boundary
  int nbits_in;       // bits received for the current byte
  int first_bit;      // MSB of the current byte
  int expecting;      // 0 = token boundary, 1/2 = awaiting continuation byte
  uint32_t pfx0, pfx1;
  bool escape_next;

  // per-bit state carried from Forward() to Perceive()
  uint32_t nodeix;
  bool live;
  uint32_t nid;
  float plbl;

};

namespace {

void* Xalloc(size_t n, bool zero, const char* what) {
  void* p = zero ? calloc(n, 1) : malloc(n);
  PAGE_REGISTRY_ADD(p, n);
  if (p == nullptr) {
    fprintf(stderr, "\nword label: %zu B for %s not available (D=%d, nodes=%d, vmax=%d)\n", n, what, kD, kNodeBits, kVMax);
    fflush(stderr);
    abort();
  }
  return p;
}

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

inline void ParseAbort(State* s) {
  // Invalid continuation: treat the bytes as literals and never reach token
  // end; discard the aborted token gradient and reset.  No abort occurred on
  // the measured inputs, so this is armour, not a hot path.
  s->expecting = 0;
  s->cvec_valid = false;
  memset(s->gacc, 0, sizeof(s->gacc));
}

void TokenDone(State* s, uint32_t r) {
  s->expecting = 0;
  const size_t ri = (r < (uint32_t)kVMax) ? (size_t)r : (size_t)(kVMax - 1);
  // Each ring word descends the lambda^s-WEIGHTED gradient, clamped per
  // element AFTER the weighting; duplicates in the ring update sequentially.
  for (int sx = 0; sx < kCtx; ++sx) {
    const float w = s->lampow[sx];
    float* e = s->e1 + (size_t)s->ring[sx] * kD;
    for (int k = 0; k < kD; ++k) {
      float gk = kEtaE * w * s->gacc[k];
      if (gk > kGradClamp) gk = kGradClamp;
      else if (gk < -kGradClamp) gk = -kGradClamp;
      e[k] -= gk;
    }
  }
  {   // topic EMA from the completed token row, read AFTER the updates
    const float* er = s->e1 + ri * kD;
    for (int k = 0; k < kD; ++k) s->topic[k] += kTopicRate * (er[k] - s->topic[k]);
  }
  for (int sx = kCtx - 1; sx > 0; --sx) s->ring[sx] = s->ring[sx - 1];
  s->ring[0] = (uint32_t)ri;
  memset(s->gacc, 0, sizeof(s->gacc));
  s->cvec_valid = false;
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
  s->expecting = 0;
}

}  // namespace

// --------------------------------------------------------------------------
// Public interface.
// --------------------------------------------------------------------------

State* Create() {
  State* s = (State*)Xalloc(sizeof(State), true, "State");
  s->nd = (float*)Xalloc((size_t)kNodeCount * kD * sizeof(float), true, "nd");
  s->nb = (float*)Xalloc((size_t)kNodeCount * sizeof(float), true, "nb");
  const size_t etab = (size_t)kVMax * kD * sizeof(float);
  s->e1 = (float*)Xalloc(etab, false, "e1");
  s->e2 = nullptr;
  s->e3 = nullptr;
  // Fixed-seed xorshift, the instrument draw order: one table after the other,
  // sequentially, no burn-in.
  uint64_t st = kSeed;
  const size_t n = (size_t)kVMax * kD;
  for (size_t i = 0; i < n; ++i)
    s->e1[i] = (float)((Rnd(&st) - 0.5) * 2.0 * kInitScale);
  float w = 1.0f;
  for (int i = 0; i < kCtx; ++i) { s->lampow[i] = w; w *= kLambda; }
  s->breg = 1;
  s->nodeix = 1;
  s->cvec_valid = false;
  return s;
}

void Destroy(State* s) {
  if (s == nullptr) return;
  free(s->nd);
  free(s->nb);
  free(s->e1);
  free(s->e2);
  free(s->e3);
  free(s);
}

float Forward(State* s) {
  s->live = false;
  if (s->nbits_in == 0) return 0.0f;
  if (s->escape_next) return 0.0f;     // the escaped literal byte
  const int bp = s->expecting;
  if (bp == 0 && s->first_bit == 0) return 0.0f;   // literal byte
  if (!s->cvec_valid) BuildCvec(s);
  const uint32_t nid = NodeId(bp, s->pfx0, s->pfx1, s->nodeix);
  const float* nv = s->nd + (size_t)nid * kD;
  float zl = s->nb[nid];
  for (int k = 0; k < kD; ++k) zl += s->cvec[k] * nv[k];
  if (zl > kLogitClamp) zl = kLogitClamp;
  else if (zl < -kLogitClamp) zl = -kLogitClamp;
  s->nid = nid;
  s->plbl = SquashF(zl);
  s->live = true;
  return zl;
}

void Perceive(State* s, int bit) {
  const int y = bit & 1;
  if (s->live) {
    const float g = s->plbl - (float)y;
    float* nv = s->nd + (size_t)s->nid * kD;
    // Per-depth eta: byte-0 trie nodes are EXACTLY ids 1..127, so nid < 128 is
    // the instrument test (bp == 0).  Dead when ETA0 == ETAN (the default).
    const float eta = (kEta0Live && s->nid < 128u) ? kEta0 : kEtaN;
    const float ge = eta * g;
    // Instrument order: gacc from the OLD node row, then descend that row.
    for (int k = 0; k < kD; ++k) {
      s->gacc[k] += g * nv[k];
      nv[k] -= ge * s->cvec[k];
    }
    s->nb[s->nid] -= ge;
    s->nodeix = s->nodeix * 2u + (uint32_t)y;
    s->live = false;
  }
  if (s->nbits_in == 0) s->first_bit = y;
  s->breg = (s->breg << 1) | (uint32_t)y;
  ++s->nbits_in;
  if (s->nbits_in < 8) return;
  const uint32_t byte = s->breg & 0xFFu;
  s->breg = 1;
  s->nbits_in = 0;
  s->nodeix = 1;
  ByteDone(s, byte);
}

}

