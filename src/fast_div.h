#ifndef FAST_DIV_H_
#define FAST_DIV_H_

namespace fastdiv {

// Replacement for `n % d` with runtime-constant d.  Init() once in the
// constructor, Mod() on the hot path.
struct Div64 {
  typedef unsigned long long u64;

  u64 d_ = 1;      // the divisor itself (for n - q*d)
  u64 M_ = 0;      // magic multiplier
  u64 mask_ = 0;   // d-1, used only in the power-of-two branch
  int s_ = 0;      // shift amount (magic branch)
  int p2s_ = 0;    // log2(d), only in the power-of-two branch (for Div())
  int a_ = 0;      // 1 = add branch needed (65-bit partial sum)
  bool pot2_ = true;

  // Granlund and Montgomery (1994), in the 64-bit form of magicu from
  // Hacker's Delight (H. S. Warren), section 10-9.  Runs once per model object.
  void Init(u64 d) {
    d_ = d;
    // d == 0 can only be a build accident; we then emulate the harmless
    // identity n % 1 == 0 instead of running into a division by zero.
    if (d <= 1 || (d & (d - 1)) == 0) {   // power of two (including 1)
      pot2_ = true;
      mask_ = (d == 0) ? 0 : (d - 1);
      p2s_ = (d <= 1) ? 0 : __builtin_ctzll(d);
      return;
    }
    pot2_ = false;
    const u64 kTwo63 = 0x8000000000000000ULL;
    const u64 kMax   = 0xFFFFFFFFFFFFFFFFULL;
    u64 nc = kMax - (kMax - d + 1) % d;   // = largest n with n % d == d-1
    int p = 63;
    u64 q1 = kTwo63 / nc,        r1 = kTwo63 - q1 * nc;
    u64 q2 = (kTwo63 - 1) / d,   r2 = (kTwo63 - 1) - q2 * d;
    u64 delta;
    a_ = 0;
    do {
      ++p;
      if (r1 >= nc - r1) { q1 = 2 * q1 + 1; r1 = 2 * r1 - nc; }
      else               { q1 = 2 * q1;     r1 = 2 * r1;      }
      if (r2 + 1 >= d - r2) {
        if (q2 >= kTwo63 - 1) a_ = 1;
        q2 = 2 * q2 + 1; r2 = 2 * r2 + 1 - d;
      } else {
        if (q2 >= kTwo63) a_ = 1;
        q2 = 2 * q2;     r2 = 2 * r2 + 1;
      }
      delta = d - 1 - r2;
    } while (p < 128 && (q1 < delta || (q1 == delta && r1 == 0)));
    M_ = q2 + 1;
    s_ = p - 64;
  }

  static inline u64 MulHi(u64 a, u64 b) {
    return (u64)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
  }

  inline u64 Div(u64 n) const {
    if (pot2_) return n >> p2s_;   // d==1 -> p2s_==0 -> n/1 == n
    u64 q = MulHi(n, M_);
    if (a_) {
      // 65-bit sum q + (n-q)/2 without overflow; then s-1 instead of s.
      q = ((n - q) >> 1) + q;
      return q >> (s_ - 1);
    }
    return q >> s_;
  }

  inline u64 Mod(u64 n) const {
    if (pot2_) return n & mask_;
    return n - Div(n) * d_;
  }
};

}

#endif
