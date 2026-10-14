#ifndef TIMESTAMP_ALGEBRA_H_
#define TIMESTAMP_ALGEBRA_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace ts_algebra {

// drev threshold of the mixed estimator: above it the window slope.
constexpr long long kSchwelle = 1000;

// Floor division (towards -inf); C rounds towards 0, which gives another
// value for negative slopes.
static inline long long Fdiv(long long a, long long b) {
  long long q = a / b, r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) --q;
  return q;
}

// days_from_civil and civil_from_days of Howard Hinnant, "chrono-Compatible
// Low-Level Date Algorithms" (howardhinnant.github.io/date_algorithms.html),
// public domain; integer only, adapted to the timestamp form below.
static inline long long DaysFromCivil(long long y, unsigned m, unsigned d) {
  y -= m <= 2;
  const long long era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153u * (m + (m > 2 ? -3 : 9)) + 2u) / 5u + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097LL + (long long)doe - 719468LL;   // days since 1970-01-01
}
static inline void CivilFromDays(long long z, long long* y, unsigned* m, unsigned* d) {
  z += 719468LL;
  const long long era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = (unsigned)(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const long long yy = (long long)yoe + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  *d = doy - (153 * mp + 2) / 5 + 1;
  *m = mp + (mp < 10 ? 3 : -9);
  *y = yy + (*m <= 2);
}

static const long long kEpocheTage = 11323LL;
static inline long long TsZuSek(int yy, long long mt, long long sek) {
  const long long jahr = 2001 + yy;
  const unsigned monat = (unsigned)(mt / 31 + 1);
  const unsigned tag = (unsigned)(mt % 31 + 1);
  return (DaysFromCivil(jahr, monat, tag) - kEpocheTage) * 86400LL + sek;
}
static inline void SekZuTs(long long t, int* yy, long long* mt, long long* sek) {
  long long tage = Fdiv(t, 86400LL);
  *sek = t - tage * 86400LL;
  long long y; unsigned m, d;
  CivilFromDays(tage + kEpocheTage, &y, &m, &d);
  *yy = (int)(y - 2001);
  *mt = (long long)m * 31 + (long long)d - 32;
}

struct Schaetzer {
  static int Breite() {
    return 32;
  }

  void Beobachte(long long rev, long long t) {
    if (!leer) {
      long long dr = rev - rev_letzt;
      if (dr <= 0) dr = 1;
      Push(Fdiv((t - t_letzt) << 20, dr));
    } else {
      t0 = t; rev0 = rev; leer = false;
    }
    t_letzt = t; rev_letzt = rev;
  }

  long long Prognose(long long rev) const {
    if (leer) return 0;
    if (fen.empty()) return t_letzt;
    const long long drev = rev - rev_letzt;
    return t_letzt + Fdiv(drev * Median(), 1LL << 20);
  }

  long long LetztesT() const { return t_letzt; }
  bool Leer() const { return leer; }

 private:
  long long GlobalAnteil(long long drev) const {
    const long long dt = t_letzt - t0;
    long long dr = rev_letzt - rev0;
    if (dr <= 0) dr = 1;
    return Fdiv(drev * dt, dr);
  }
  void Push(long long r) {
    const size_t W = (size_t)Breite();
    if (fen.size() < W) {
      fen.push_back(r);
    } else {
      fen[pos] = r;
      pos = (pos + 1) % W;
    }
    ema = (nsl == 0) ? r : (ema - Fdiv(ema, 8) + Fdiv(r, 8));
    ++nsl;
  }
  long long Median() const {
    hilf = fen;
    std::sort(hilf.begin(), hilf.end());
    return hilf[hilf.size() / 2];             // element n/2 of the sorted window (upper middle)
  }

  std::vector<long long> fen;
  mutable std::vector<long long> hilf;
  size_t pos = 0;
  unsigned long long nsl = 0;
  long long ema = 0, t_letzt = 0, t0 = 0, rev0 = 0, rev_letzt = 0;
  bool leer = true;
};

static inline void RohFail(const char* what) {
  fprintf(stderr, "timestamps: %s\n", what);
  fflush(stderr);
  exit(1);
}

// "<yy><mt>:<sek>" -> seconds since 2001-01-01
static inline long long TsRohZuSek(const char* p, int n) {
  if (n < 4) RohFail("raw timestamp too short");
  int yy = 0;
  for (int i = 0; i < 2; ++i) {
    if (p[i] < '0' || p[i] > '9') RohFail("raw timestamp without two year digits");
    yy = yy * 10 + (p[i] - '0');
  }
  int i = 2;
  long long mt = 0;
  for (; i < n && p[i] >= '0' && p[i] <= '9'; ++i) mt = mt * 10 + (p[i] - '0');
  if (i == 2) RohFail("raw timestamp without month field");
  if (i >= n || p[i] != ':') RohFail("raw timestamp without ':' - the stream is not in the raw form");
  ++i;
  if (i >= n) RohFail("raw timestamp without seconds field");
  long long sek = 0;
  for (; i < n; ++i) {
    if (p[i] < '0' || p[i] > '9') RohFail("seconds field not numeric");
    sek = sek * 10 + (p[i] - '0');
  }
  if (sek > 86399) RohFail("seconds field over 86399");
  return TsZuSek(yy, mt, sek);
}

static inline int SekZuTsRoh(char* out, long long t) {
  int yy; long long mt, sek;
  SekZuTs(t, &yy, &mt, &sek);
  if (yy < 0 || yy > 99) RohFail("reconstructed year outside 2001..2100 - prediction and residual do not match");
  if (mt < 0 || sek < 0 || sek > 86399) RohFail("reconstructed timestamp outside the raw form");
  int n = 0;
  out[n++] = (char)('0' + yy / 10);
  out[n++] = (char)('0' + yy % 10);
  char z[24]; int m = 0;
  long long v = mt;
  do { z[m++] = (char)('0' + (int)(v % 10)); v /= 10; } while (v);
  while (m > 0) out[n++] = z[--m];
  out[n++] = ':';
  v = sek; m = 0;
  do { z[m++] = (char)('0' + (int)(v % 10)); v /= 10; } while (v);
  while (m > 0) out[n++] = z[--m];
  return n;
}

static inline int DezimalRoh(char* out, long long v) {
  int n = 0;
  if (v < 0) { out[n++] = '-'; v = -v; }
  char z[24]; int m = 0;
  do { z[m++] = (char)('0' + (int)(v % 10)); v /= 10; } while (v);
  while (m > 0) out[n++] = z[--m];
  return n;
}
static inline long long ZahlRoh(const char* p, int n) {
  long long v = 0; int i = 0, neg = 0;
  if (n > 0 && p[0] == '-') { neg = 1; i = 1; }
  if (i >= n) RohFail("empty residual field");
  for (; i < n; ++i) {
    if (p[i] < '0' || p[i] > '9') RohFail("residual not numeric");
    v = v * 10 + (p[i] - '0');
  }
  return neg ? -v : v;
}

}

#endif
