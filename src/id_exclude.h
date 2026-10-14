
#ifndef ID_EXCLUDE_H_
#define ID_EXCLUDE_H_

#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include "metablock.h"
#include "neif_levers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// Exclusion model over the identifiers in part A of the metadata block.
// A part-A row is "<page id delta><revision id, 4 base-100 digits>[	<restrictions>]".
// Page ids and revision ids are MediaWiki primary keys, so a value that was
// seen cannot occur again.  The model tracks the still-free values (page ids
// in a Fenwick tree, revision ids in a digit tree) and weights every
// candidate digit of the current byte by the number of free ids behind it.
// It arms itself after 64 well-formed rows and resets on a malformed row.
namespace id_exclude {

// Corpus invariants of enwik9 (build-time constants).  A guard aborts if a
// value lies outside, instead of a silent fallback.
static const int kPgLo = 1, kPgHi = 326490;          // page id range
static const long long kRevLo = 15898943LL, kRevHi = 42164216LL;
struct Tree {
  int D = 0;
  long long B = 100;               // place-value base (base 100)
  long long lo = 0, hi = 0;
  long long pow10[19];
  long long off[19];
  int       *free_[19];
  unsigned char *leaf_ = 0;
  long long  size_[19];

  void Init(int digits, long long a, long long b, long long basis = 0) {
    D = digits; lo = a; hi = b;
    if (basis > 0) B = basis;
    pow10[0] = 1;
    for (int i = 1; i <= 18; ++i)
      pow10[i] = (i <= D) ? pow10[i-1] * B : pow10[i-1];
    if (hi >= pow10[D]) {
      fprintf(stderr, "id model: %lld does not fit %d digits\n", hi, D);
      exit(1);
    }
    for (int k = 0; k <= D; ++k) {
      long long s = pow10[D - k];
      off[k] = lo / s;
      size_[k] = hi / s - off[k] + 1;
      if (k == D) {
        leaf_ = (unsigned char*)malloc((size_t)((size_[k] + 7) / 8));
        if (!leaf_) { fprintf(stderr, "id model: out of memory\n"); exit(1); }
        memset(leaf_, 0xFF, (size_t)((size_[k] + 7) / 8));
        continue;
      }
      free_[k] = (int*)malloc((size_t)size_[k] * sizeof(int));
      if (!free_[k]) { fprintf(stderr, "id model: out of memory\n"); exit(1); }
      for (long long p = 0; p < size_[k]; ++p) free_[k][p] = (int)All(k, p + off[k]);
    }
  }
  long long All(int k, long long p) const {
    long long s = pow10[D - k];
    long long a = p * s, b = a + s - 1;
    if (a < lo) a = lo;
    if (b > hi) b = hi;
    return b >= a ? b - a + 1 : 0;
  }
  long long Free(int k, long long p) const {
    long long i = p - off[k];
    if (i < 0 || i >= size_[k]) return 0;
    if (k == D) return (leaf_[i >> 3] >> (i & 7)) & 1u;
    return free_[k][i];
  }
  void Mark(long long v) {
    if (v < lo || v > hi) {
      fprintf(stderr, "id model: value %lld outside [%lld,%lld]\n",
              v, lo, hi);
      exit(1);
    }
    if (Free(D, v) == 0) {
      fprintf(stderr, "id model: duplicate value %lld\n", v);
      exit(1);
    }
    for (int k = 0; k < D; ++k) {
      long long i = v / pow10[D - k] - off[k];
      if (i >= 0 && i < size_[k] && free_[k][i] > 0) --free_[k][i];
    }
    { long long i = v - off[D];
      if (i >= 0 && i < size_[D]) leaf_[i >> 3] &= (unsigned char)~(1u << (i & 7)); }
  }
  void Reset() {
    for (int k = 0; k < D; ++k)
      for (long long p = 0; p < size_[k]; ++p) free_[k][p] = (int)All(k, p + off[k]);
    memset(leaf_, 0xFF, (size_t)((size_[D] + 7) / 8));
  }
  void Release() {
    for (int k = 0; k < D; ++k) { free(free_[k]); free_[k] = 0; }
    free(leaf_); leaf_ = 0;
  }
};

static const long long kRevBase = 0;

struct Fen {
  int n = 0;
  int *t = 0;
  void Init(int N) {
    n = N;
    t = (int*)malloc((size_t)(n + 1) * sizeof(int));
    if (!t) { fprintf(stderr, "id model: out of memory\n"); exit(1); }
    Fill();
  }
  void Fill() {
    for (int i = 1; i <= n; ++i) t[i] = i & (-i);
  }
  void Add(int i, int v) { for (; i <= n; i += i & (-i)) t[i] += v; }
  int Pref(int i) const {
    if (i > n) i = n;
    int s = 0; for (; i > 0; i -= i & (-i)) s += t[i]; return s;
  }
  int Range(long long a, long long b) const {
    if (a < 1) a = 1;
    if (b > n) b = n;
    if (b < a) return 0;
    return Pref((int)b) - Pref((int)(a - 1));
  }
};

inline Tree revt;
inline Fen  pgf;
inline int inited = 0;

inline void Init() {
  if (inited) return;
  inited = 1;
  pgf.Init(kPgHi);
  revt.Init(4, kRevBase ? 0 : kRevLo, kRevHi - kRevBase, 100);
}

inline int  armed = 0;          // well-formed rows in a row so far
inline int  col = 0;            // byte position in the current row
inline int  fld = 1;            // 1 = page id delta, 2 = revision id, 3 = rest,
                                // 4 = B1 revid residual (E1: after '+')
inline int  fcol = 0;           // byte position inside the field
inline int  bad = 0;            // the row has already broken the form
inline int  neg = 0;            // field 1 starts with '-'
inline long long pfx = 0;       // digit prefix of the current field
inline long long revpfx = 0;
#if NEIF_E1
// E1: a B1 row is "<delta> '+' [-]<residual>[\t<restr>]".  '+' ends field 1;
// the residual field carries no exclusion (its value depends on the
// interpolation, not on the free set), the row marks only its page id.
static const unsigned int kB1Mark = 0x2b;
// Share of the field-end weight given to '+' - the same factor on the free
// and on the total count, so the page-id evidence (free/total) is unchanged.
static const double kB1Share = 1.0 / 101.0;
// B2 revids are >= 17,000,000: first base-100 digit >= 17.
static const int kB2FirstMin = 17;
inline int b1row = 0;           // the row is a B1 row ('+' seen)
inline int rneg = 0;            // the residual starts with '-'
#endif
inline long long cur_id = 0;    // absolute page id of the previous row
inline long long asc = 0;       // length of the current ascending run
inline unsigned long long stat_fire = 0, stat_hard = 0, stat_lines = 0,
                          stat_reset = 0, stat_mark = 0, stat_det = 0,
                          stat_bytes = 0;

// Candidate table of the current byte (computed once per byte, not per bit).
inline int    cw_n = 0;
// A base-100 alphabet needs at least 102 entries (E1: 1 + 100 + 10 = 111).
inline unsigned char cw_b[112];
inline double cw_w[112], cw_u[112];
inline int    cw_valid = 0;

inline void MarkId(long long v) {
  if (v < kPgLo || v > kPgHi) {
    fprintf(stderr, "id model: page id %lld outside [%d,%d]\n",
            v, kPgLo, kPgHi);
    exit(1);
  }
  if (pgf.Range(v, v) == 0) {
    fprintf(stderr, "id model: duplicate page id %lld\n", v);
    exit(1);
  }
  pgf.Add((int)v, -1);
}
inline void ResetId() { pgf.Fill(); }

inline int released = 0;
inline void Release() {
  free(pgf.t); pgf.t = 0;
  revt.Release();
  released = 1;
}
inline void Revive() {
  if (!released) return;
  released = 0;
  pgf.Init(kPgHi);
  revt.Init(8, kRevBase ? 0 : kRevLo, kRevHi - kRevBase);
}

inline bool MpAlpha(unsigned int c) {
  return c >= (unsigned int)0x80 &&
         c <= (unsigned int)(0x80 + 99);
}
inline void Byte(unsigned int ch) {
#if AUX_BUILD
  (void)ch;
  return;
#else

  Init();
  ++stat_bytes;
  cw_valid = 0;
  int ok = 1;
  if (ch == '\t') {
    if (fld == 2) { ok = (fcol == 4); fld = 3; fcol = 0; }
#if NEIF_E1
    else if (fld == 4) { ok = (fcol >= 1 + rneg); fld = 3; fcol = 0; }
#endif
    else if (fld == 1) { ok = 0; }
    else { ++fcol; }
  } else if (ch == '\n') {
    // row end, handled below
  } else if (fld == 1 && MpAlpha(ch)) {
    ok = (fcol >= 1 + neg);
#if NEIF_E1
    if ((int)(ch - (unsigned int)0x80) < kB2FirstMin) ok = 0;
#endif
    fld = 2; fcol = 0;
    revpfx = revpfx * 100 + (long long)(ch - (unsigned int)0x80);
    ++fcol;
#if NEIF_E1
  } else if (fld == 1 && ch == kB1Mark) {
    ok = (fcol >= 1 + neg);
    fld = 4; fcol = 0; b1row = 1;
  } else if (fld == 4) {
    if (fcol == 0 && ch == '-') { rneg = 1; }
    else ok = (ch >= '0' && ch <= '9');
    ++fcol;
#endif
  } else if (fld == 1) {
    if (fcol == 0 && ch == '-') { neg = 1; }
    else {
      ok = (ch >= '0' && ch <= '9') && fcol < 6 + neg;
      if (ok) pfx = pfx * 10 + (long long)(ch - '0');
    }
    ++fcol;
  } else if (fld == 2) {
    ok = MpAlpha(ch) && fcol < 4;
    if (ok) revpfx = revpfx * 100 + (long long)(ch - (unsigned int)0x80);
    ++fcol;
  } else {
    ++fcol;                    // restrictions: arbitrary bytes (WRT-coded)
  }
  if (!ok) bad = 1;

  if (ch == '\n') {
    ++stat_lines;
    const long long id = cur_id + (neg ? -pfx : pfx);
#if NEIF_E1
    const int good = b1row
        ? (!bad && ((fld == 4 && fcol >= 1 + rneg) || fld == 3) &&
           id >= kPgLo && id <= kPgHi)
        : (!bad && ((fld == 2 && fcol == 4) || fld == 3) &&
           id >= kPgLo && id <= kPgHi &&
           revpfx >= kRevLo - kRevBase && revpfx <= kRevHi - kRevBase);
#else
    const int good = (!bad && ((fld == 2 && fcol == 4) || fld == 3) &&
                      id >= kPgLo && id <= kPgHi &&
                      revpfx >= kRevLo - kRevBase && revpfx <= kRevHi - kRevBase);
#endif
    if (good) {
      if (armed >= 64) {
        Revive();
        ++stat_mark;
        MarkId(id);
#if NEIF_E1
        if (!b1row) revt.Mark(revpfx);
#else
        revt.Mark(revpfx);
#endif
      }
      asc = (id > cur_id) ? asc + 1 : 0;
      cur_id = id;
      if (armed < 1000000000) ++armed;
    } else {
      if (armed >= 64) {
        ++stat_reset;
        Release();
      }
      armed = 0; cur_id = 0; asc = 0;
    }
    col = 0; fld = 1; fcol = 0; bad = 0; neg = 0; pfx = 0; revpfx = 0;
#if NEIF_E1
    b1row = 0; rneg = 0;
#endif
  } else {
    ++col;
  }
#endif
}

// ------------------------ candidate table of the current byte ---------------
inline void AddCand(unsigned char b, double w, double u) {
  cw_b[cw_n] = b; cw_w[cw_n] = w; cw_u[cw_n] = u; ++cw_n;
}
// Free/all page ids in [lo,hi] (absolute).  With the monotonic preference on
// (shift 2), ids below the previous one count only 1/4 during a long
// ascending run.
inline void SpanWAbs(long long lo, long long hi, double *w, double *u) {
  if (lo < kPgLo) lo = kPgLo;
  if (hi > kPgHi) hi = kPgHi;
  if (hi < lo) { *w = 0; *u = 0; return; }
  if (asc >= 1024) {
    const double f = 1.0 / (double)(1 << 2);
    double wa = 0, ua = 0, wb = 0, ub = 0;
    if (lo <= cur_id) {
      long long h2 = hi < cur_id ? hi : cur_id;
      wa = (double)pgf.Range(lo, h2); ua = (double)(h2 - lo + 1);
    }
    if (hi > cur_id) {
      long long l2 = lo > cur_id + 1 ? lo : cur_id + 1;
      wb = (double)pgf.Range(l2, hi); ub = (double)(hi - l2 + 1);
    }
    *w = wa * f + wb; *u = ua * f + ub;
    return;
  }
  *w = (double)pgf.Range(lo, hi);
  *u = (double)(hi - lo + 1);
}

// Free/all page ids in cur_id + s*[a,b]  (a <= b, both >= 0).
inline void SpanW(int s, long long a, long long b, double *w, double *u) {
  if (s > 0) SpanWAbs(cur_id + a, cur_id + b, w, u);
  else       SpanWAbs(cur_id - b, cur_id - a, w, u);
}

inline void PrepareByte() {
  cw_n = 0; cw_valid = 1;
  if (armed < 64 || bad) return;
  Revive();
  if (fld == 2) {                          // revision id, 4 base-100 digits
    if (fcol >= 4) return;
    for (int d = 0; d < 100; ++d) {
      const long long q = revpfx * 100 + d;
      cw_b[cw_n] = (unsigned char)((unsigned)0x80 + (unsigned)d);
      cw_w[cw_n] = (double)revt.Free(fcol + 1, q);
      cw_u[cw_n] = (double)revt.All (fcol + 1, q);
      ++cw_n;
    }
    return;
  }
  if (fld != 1) return;                    // (E1: no candidates in field 4)
  // Field 1: signed delta, variable length, at most 6 digits.  The field end
  // is the first revision id byte (one of 100 alphabet bytes), not a tab.
  {
    const int nd = fcol - neg;             // digits so far
    if (nd > 6) return;
    const int s = neg ? -1 : 1;
    double w, u;
    if (fcol == 0) {
      SpanW(-1, 1, 999999, &w, &u);
      AddCand('-', w, u);
    }
    if (nd >= 1) {                         // field end possible
      SpanW(s, pfx, pfx, &w, &u);
#if NEIF_E1
      // E1: the field ends with '+' (B1 row) or with the first byte of a B2
      // revid; the end weight is split between them by the same factor for
      // free and total.  A B2 revid's first digit is >= 17, so the bytes
      // below are excluded (no free value, full total).
      AddCand((unsigned char)kB1Mark, w * kB1Share, u * kB1Share);
      {
        const double wa = w * (1.0 - kB1Share) / 100.0;
        const double ua = u * (1.0 - kB1Share) / 100.0;
        for (int d = 0; d < 100; ++d)
          AddCand((unsigned char)((unsigned)0x80 + (unsigned)d),
                  d < kB2FirstMin ? 0.0 : wa, ua);
      }
#else
      for (int d = 0; d < 100; ++d)
        AddCand((unsigned char)((unsigned)0x80 + (unsigned)d),
                w / 100.0, u / 100.0);
#endif
    }
    if (nd < 6) {
      for (int d = 0; d < 10; ++d) {
        const long long q = pfx * 10 + d;
        if (nd >= 1 && pfx == 0) {         // the stream has no leading zeros
          AddCand((unsigned char)('0' + d), 0, 0);
          continue;
        }
        const int rmax = (nd == 0 && d == 0) ? 0 : (6 - nd - 1);
        double ww = 0, uu = 0;
        long long p10 = 1;
        for (int r = 0; r <= rmax; ++r) {
          double w2, u2;
          SpanW(s, q * p10, q * p10 + p10 - 1, &w2, &u2);
          ww += w2; uu += u2;
          p10 *= 10;
        }
        AddCand((unsigned char)('0' + d), ww, uu);
      }
    }
  }
}

inline void CountDet() {
  if (cw_n == 0) return;
  int frei = 0;
  for (int i = 0; i < cw_n; ++i) if (cw_w[i] > 0) ++frei;
  if (frei == 1) ++stat_det;
}

inline int Weights(unsigned int c0, int bpos,
                   double *W1, double *W0, double *U1, double *U0) {
  if (!cw_valid) {
    if (bpos != 0) return 0;
    PrepareByte();
    CountDet();
  }
  if (cw_n == 0) return 0;
  double w1 = 0, w0 = 0, u1 = 0, u0 = 0;
  int nl = 0;
  for (int i = 0; i < cw_n; ++i) {
    const unsigned int by = cw_b[i];
    if ((int)((0x100u | by) >> (8 - bpos)) != (int)c0) continue;
    ++nl;
    if ((by >> (7 - bpos)) & 1u) { w1 += cw_w[i]; u1 += cw_u[i]; }
    else                         { w0 += cw_w[i]; u0 += cw_u[i]; }
  }
  if (u1 <= 0 || u0 <= 0) return 0;
  if (w1 == u1 && w0 == u0) return 0;
  *W1 = w1; *W0 = w0; *U1 = u1; *U0 = u0;
  return 1;
}

inline float Hard(float p, unsigned int c0, int bpos) {
  double W1, W0, U1, U0;
  if (!Weights(c0, bpos, &W1, &W0, &U1, &U0)) return p;
  if (W1 > 0.0 && W0 > 0.0) return p;
  if (W1 <= 0.0 && W0 <= 0.0) return p;
  ++stat_hard;
  return (W1 > 0.0) ? (float)(65534.0 / 65535.0) : (float)(1.0 / 65535.0);
}

inline float Correct(float p, unsigned int c0, int bpos) {
  double W1, W0, U1, U0;
  if (!Weights(c0, bpos, &W1, &W0, &U1, &U0)) return p;
  ++stat_fire;
  double r1 = W1 / U1, r0 = W0 / U0;
  if (r1 <= 0.0 && r0 <= 0.0) return p;
  if (r1 <= 0.0 || r0 <= 0.0) ++stat_hard;
  const double a = (double)p * r1, b = (1.0 - (double)p) * r0;
  if (a + b <= 0.0) return p;
  double q = a / (a + b);
  if (q < 1.0 / 65535.0) q = 1.0 / 65535.0;
  if (q > 65534.0 / 65535.0) q = 65534.0 / 65535.0;
  return (float)q;
}

inline float MixDelta(unsigned int c0, int bpos) {
  double W1, W0, U1, U0;
  if (!Weights(c0, bpos, &W1, &W0, &U1, &U0)) return 0.0f;
  ++stat_fire;
  const double kLo = 1.0 / 65535.0, kHi = 65534.0 / 65535.0;
  double q = (W1 + W0 > 0.0) ? W1 / (W1 + W0) : 0.5;
  double u = U1 / (U1 + U0);
  if (q < kLo) q = kLo; if (q > kHi) q = kHi;
  if (u < kLo) u = kLo; if (u > kHi) u = kHi;
  double d = log(q / (1.0 - q)) - log(u / (1.0 - u));
  if (d >  12.0) d =  12.0;
  if (d < -12.0) d = -12.0;
  return (float)d;
}

inline float MixDeltaP(unsigned int c0, int bpos) {
  double W1, W0, U1, U0;
  if (!Weights(c0, bpos, &W1, &W0, &U1, &U0)) return 0.0f;
  if (W1 + W0 <= 0.0) return 0.0f;
  const double kLo = 1.0 / 65535.0, kHi = 65534.0 / 65535.0;
  double q = W1 / (W1 + W0);
  if (q < kLo) q = kLo;
  if (q > kHi) q = kHi;
  double d = log(q / (1.0 - q));
  if (d >  12.0) d =  12.0;
  if (d < -12.0) d = -12.0;
  return (float)d;
}

inline float MixPBit(unsigned int c0, int bpos) {
  double W1, W0, U1, U0;
  if (!Weights(c0, bpos, &W1, &W0, &U1, &U0)) return 0.5f;
  ++stat_fire;
  if (W1 + W0 <= 0.0) return 0.5f;
  double q = W1 / (W1 + W0);
  if (q < 1.0 / 65535.0) q = 1.0 / 65535.0;
  if (q > 65534.0 / 65535.0) q = 65534.0 / 65535.0;
  return (float)q;
}

}

#endif
