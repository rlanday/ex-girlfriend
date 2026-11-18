// Floor stage: a terminal correction for nearly certain bits.
//
// The coder codes every bit with a 16-bit probability p16 in [1, 65535]; a bit
// that is certain still costs at least -log2(65535/65536) = 2.2e-5 bit, and the
// adaptive-depth pre-stage hands on its logistic p for every bit it skips
// (|z| > tau), which is measurably under-confident between |z| tau and tau + 3.
// This stage takes the probability the coder would use (p16, after the
// BITLSTM) and, for the bits of the kinds below only, replaces the error
// probability by an adaptive count of how often bits of the same kind were
// really wrong, coded with 32-bit precision.  Nothing upstream reads its
// output: models, SSE, BITLSTM and the pre-stage walk exactly the same
// trajectory, only the interval split changes.  The end stages of the
// semantic cache act behind it.
//
// Kinds (contexts), each per predicted side and per metablock/rest:
//   bits skipped by the pre-stage, by |z| (steps of 1/zstep above the gate
//   tau) and bit position;
//   mixer-path bits whose floor class f = min(p16, 65536 - p16) <= fmax, by f,
//   by |m1raw| (steps of 2) and by whether the BITLSTM corrected the bit.
// Estimate: err32 = (k * 2^32 + C * f * 2^16) / (n + C), a pseudo-count prior
// of C bits at the probability the coder would use, so a young context codes
// as without this stage; integer arithmetic only, identical in coder and
// decoder.  Counts halve at a fixed n.
// Bits outside these kinds get p32 = p16 << 16, which splits the interval exactly
// as the 16-bit formula does (floor(r * p16 / 65536) either way).  A build
// without the pre-stage therefore codes byte for byte as without this stage.
//
// Release configuration: prior C = 2^12 bits, skipped bits only (fmax 0),
// |z| in steps of 0.25, per bit position, own tables inside the metablock,
// counts halve at 2^22.
#ifndef FLOOR_STAGE_H_
#define FLOOR_STAGE_H_

#include <cstdint>

namespace floor_stage {

constexpr int kCLog = 12;          // prior weight C = 2^12 bits
constexpr int kFMax = 0;           // mixer-path floor classes 1..kFMax take part (0 = none)
constexpr int kHalfLog = 22;       // counts halve at n = 2^22
constexpr int kSkipOn = 1;         // bits skipped by the pre-stage take part
constexpr int kZStep = 4;          // |z| bins per unit (4 = steps of 0.25)
constexpr int kUseBp = 1;          // skip contexts per bit position
constexpr int kUseMeta = 1;        // own tables inside the metablock

constexpr int kZBins = 96;
constexpr float kZMin = (float)(7000) / 1000.0f;
constexpr int kSkip = 8 * kZBins;  // x bit position
constexpr int kMixBins = 16;       // |m1raw| in [0, 32) by 2
constexpr int kFMaxCap = 64;
constexpr int kHalf = kSkip + 2 * kFMaxCap * kMixBins;
constexpr int kCtx = 4 * kHalf;    // x side x metablock
static_assert(kFMax >= 0 && kFMax <= kFMaxCap, "kFMax in 0..64");
static_assert(kZStep >= 1 && kZStep <= 4, "kZStep in 1..4");

struct Cell { uint32_t n, k; };

struct Stage {
  Cell c[kCtx];
  int cur = -1;
  int side = 0;
  // Build constants: unused kinds fold away (fmax 0 removes the mixer branch).
  static constexpr int c_log = kCLog, fmax = kFMax;
  static constexpr int half_log = kHalfLog, skip_on = kSkipOn;
  static constexpr int zstep = kZStep, use_bp = kUseBp;
  static constexpr int use_meta = kUseMeta;

  // noinline (constructor, P32, Update): Encoder::Encode/Decoder::Decode and
  // the two constructors are profiled; inlined, the branches and the loop of
  // this stage would change their control flow and void their PGO profile
  // (a hash mismatch in the profiled build).  As calls, their CFG stays the same.
  __attribute__((noinline, minsize)) Stage() {
    for (int i = 0; i < kCtx; ++i) { c[i].n = 0; c[i].k = 0; }
  }

  // p16: what the coder would use; ovr/skip/bitl: path of the bit; z: logit
  // of the adaptive-depth pre-stage (skip path); m1raw: end-mixer logit (mixer path); meta: inside the
  // metablock; bp: bits of this byte already seen (0 = MSB).  Returns p32, the
  // probability of a 1 in units of 2^-32.
  __attribute__((noinline)) uint32_t P32(unsigned int p16, int ovr, int skip, int bitl, float z, float m1raw,
                      int meta, int bp) {
    cur = -1;
    const uint32_t base = (uint32_t)p16 << 16;
    if (ovr) return base;
    const unsigned int f = p16 < 32768u ? p16 : 65536u - p16;
    side = p16 >= 32768u ? 1 : 0;
    int idx;
    if (skip) {
      if (!skip_on) return base;
      const float az = z < 0.0f ? -z : z;
      int b = (int)((az - kZMin) * (float)zstep);
      if (!(b >= 0)) b = 0;
      if (b > kZBins - 1) b = kZBins - 1;
      idx = (use_bp ? (bp & 7) : 0) * kZBins + b;
    } else {
      if ((int)f > fmax) return base;
      const float am = m1raw < 0.0f ? -m1raw : m1raw;
      int b = (int)(am * 0.5f);
      if (!(b >= 0)) b = kMixBins - 1;   // NaN guard
      if (b > kMixBins - 1) b = kMixBins - 1;
      idx = kSkip + ((bitl ? 1 : 0) * kFMaxCap + (int)(f - 1)) * kMixBins + b;
    }
    idx += side * kHalf + ((use_meta && meta) ? 2 * kHalf : 0);
    cur = idx;
    const Cell& e = c[idx];
    const uint64_t cw = (uint64_t)1 << c_log;
    uint64_t err = (((uint64_t)e.k << 32) + cw * ((uint64_t)f << 16)) / ((uint64_t)e.n + cw);
    if (err < 1) err = 1;
    if (err > ((uint64_t)1 << 31)) err = (uint64_t)1 << 31;
    return side ? (uint32_t)(((uint64_t)1 << 32) - err) : (uint32_t)err;
  }

  __attribute__((noinline)) void Update(int bit) {
    if (cur < 0) return;
    Cell& e = c[cur];
    ++e.n;
    if (bit != side) ++e.k;
    if (e.n >= ((uint32_t)1 << half_log)) { e.n >>= 1; e.k >>= 1; }
  }
};

// Interval split with a 32-bit probability; for p32 = p16 << 16 this is exactly
// x1 + floor(r * p16 / 65536), the 16-bit formula of the coder.
inline unsigned int Mid(unsigned int x1, unsigned int x2, uint32_t p32) {
  return x1 + (unsigned int)(((uint64_t)(x2 - x1) * (uint64_t)p32) >> 32);
}

}  // namespace floor_stage
#endif
