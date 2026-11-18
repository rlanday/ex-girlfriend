#ifndef TS_PROGRESS_H_
#define TS_PROGRESS_H_
#include "meld.h"

#include "metablock.h"
#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include "timestamp_algebra.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace ts_progress {

using ts_algebra::Fdiv;
using ts_algebra::DaysFromCivil;
using ts_algebra::CivilFromDays;
using ts_algebra::kEpocheTage;
using ts_algebra::TsZuSek;
using ts_algebra::SekZuTs;
using ts_algebra::Schaetzer;
using ts_algebra::RohFail;
using ts_algebra::TsRohZuSek;
using ts_algebra::SekZuTsRoh;
using ts_algebra::DezimalRoh;
using ts_algebra::ZahlRoh;

// Signed decimal appended to o, the form of DezimalRoh (no '+', no leading
// zero but for 0): the injection here and the metadata block codec
__attribute__((noinline)) static void DezimalAn(std::vector<unsigned char>* o, long long v) {
  unsigned char z[24];
  int n = 0;
  if (v < 0) { o->push_back(0x2d); v = -v; }
  do { z[n++] = (unsigned char)(0x30 + (int)(v % 10)); v /= 10; } while (v);
  while (n > 0) o->push_back(z[--n]);
}

struct Stats {   // statistics for Report (empty in the release build)
};

static const unsigned char kNL = 0x0a;
static const unsigned char kTab = 0x09;

// After E3b of ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only, ts_progress.h
// B2Basis): forecast of a record k above the edge from the long
// baseline over the last 9 records above the edge:
//   t_{k-1} + (r_k - r_{k-1}) * (t_{k-1} - t_j) / (r_{k-1} - r_j),
//   j = the oldest record in the ring,
// floor division, integer only.  False (the caller keeps the median
// estimator) while fewer than two records are known or the baseline has no
// revid span.  The ring holds 8 intervals as there; 4/6/8/10/12/16/32
// intervals gave -1.5/-2.1/-2.2/-2.1/-2.0/-1.6/-0.3 KB on enwik9.
class Steigung {
 public:
  enum { kN = 9 };
  __attribute__((noinline)) void Beobachte(long long rev, long long t) {
    rev_[pos_] = rev; t_[pos_] = t;
    pos_ = (pos_ + 1) % kN;
    if (n_ < kN) ++n_;
  }
  __attribute__((noinline)) bool Prognose(long long rev, long long* f) const {
    if (n_ < 2) return false;
    const int neu = (pos_ + kN - 1) % kN;              // record k-1
    const int alt = (pos_ + kN - n_) % kN;             // record j
    const long long dr = rev_[neu] - rev_[alt];
    if (dr <= 0) return false;
    const long long drev = rev - rev_[neu];
    const long long dt = t_[neu] - t_[alt];
    // No overflow guard: revids have 26 bits (E91 in the unit) and the
    // seconds since 2001 stay below 2^32 (year < 2100, E63), so the product
    // stays below 2^58.
    *f = t_[neu] + Fdiv(drev * dt, dr);
    return true;
  }

 private:
  long long rev_[kN] = {0}, t_[kN] = {0};
  int pos_ = 0, n_ = 0;
};

class Aut {
 public:
  void Observe(unsigned char b) {
    have_block_ = false;
#if AUX_BUILD
    (void)b; return;
#else
    if (phase_ == kFertig) return;
    if (!block_opened_ &&
        ++guard_pos_ > (unsigned long long)4096)
      __builtin_unreachable();
    if (metablock::mk.Offen()) block_opened_ = true;
    if (metablock::mk.BZeileFertig()) {
      const size_t k = metablock::mk.BRang();
      const long long t_k = metablock::mk.BT();
      sch_.Beobachte(Rev(k), t_k);
      if (Oben(k)) stg_.Beobachte(Rev(k), t_k);
      const size_t kn = k + 1;
      if (kn < metablock::mk.RangZahl()) Einspeisen(kn);
    }
    if (metablock::mk.NachlaufEndet()) {
      part_b_done_ = true;
      phase_ = kFertig;
      Bericht("Blockende");
    }
    return;
#endif
  }

  int TakeBlock(const unsigned char** p) {
    if (!have_block_) return 0;
    have_block_ = false;
    *p = block_.data();
    return (int)block_.size();
  }

  void Finish() {}

  void Report(const char* page_index) const {
    (void)page_index;
  }

  const Stats& GetStats() const { return st_; }

 private:
  enum Phase { kBlock = 0, kFertig = 1 };

  [[noreturn]] __attribute__((cold, noinline)) static void Fail(const char* what) {
    fprintf(stderr, "%s\n", what);
    fflush(stderr);
    exit(1);
  }

  static bool Oben(size_t k) {
    return (long long)k >= (long long)87813;
  }

  long long Rev(size_t k) const {
    return metablock::mk.RevidNachRang(k);
  }

  // ---- injection ---------------------------------------------------------
  void Einspeisen(size_t kn) {
    block_.clear();
    letzte_prognose_ = sch_.Prognose(Rev(kn));
    if (Oben(kn)) {
      long long f = 0;
      if (stg_.Prognose(Rev(kn), &f)) letzte_prognose_ = f;
    }
    if (Oben(kn)) {
      Dezimal(letzte_prognose_ - (sch_.Leer() ? 0LL : sch_.LetztesT()));
      block_.push_back(kTab);
      have_block_ = !block_.empty();
      return;
    }
    if (!Oben(kn)) {
      long long thk = letzte_prognose_;
      if (thk < 0) thk = 0;
      {
        int yk; long long mk, sk;
        SekZuTs(thk, &yk, &mk, &sk);
        if (yk < 0) yk = 0;
        if (yk > 99) yk = 99;
        if (mk < 0) mk = 0;
        if (mk > 9999) mk = 9999;
        if (sk < 0) sk = 0;
        if (sk > 999999) sk = 999999;
        TsFest((long long)yk, 1);
        TsFest(mk, 2);
        TsFest(sk, 3);
      }
      block_.push_back(kTab);
      have_block_ = !block_.empty();
      return;
    }
    long long th = letzte_prognose_;
    int yy; long long mt, sek;
    if (th < 0) th = 0;
    SekZuTs(th, &yy, &mt, &sek);
    if (yy < 0) yy = 0;
    if (yy > 99) yy = 99;
    if (mt < 0) mt = 0;
    if (mt > 9999) mt = 9999;
    if (sek < 0) sek = 0;
    if (sek > 999999) sek = 999999;
    MpFest((long long)yy, 1);
    MpFest(mt, 2);
    MpFest(sek, 3);
    block_.push_back(kTab);
    have_block_ = !block_.empty();
  }

  void TsFest(long long v, int n) {
    unsigned char z[8];
    for (int i = n - 1; i >= 0; --i) {
      z[i] = (unsigned char)((unsigned)0x80 + (unsigned)(v % 100));
      v /= 100;
    }
    for (int i = 0; i < n; ++i) block_.push_back(z[i]);
  }

  void MpFest(long long v, int n) {
    unsigned char z[8];
    for (int i = n - 1; i >= 0; --i) {
      z[i] = (unsigned char)((unsigned)0x80 + (unsigned)(v % 100));
      v /= 100;
    }
    for (int i = 0; i < n; ++i) block_.push_back(z[i]);
  }

  void Dezimal(long long v) { DezimalAn(&block_, v); }

  void Bericht(const char*) {}

  Phase phase_ = kBlock;
  bool block_opened_ = false, line_has_tab_ = false, part_b_done_ = false, have_block_ = false;
  unsigned long long guard_pos_ = 0;
  std::vector<unsigned char> block_;
  Schaetzer sch_;
  Steigung stg_;                 // slope above the edge
  long long letzte_prognose_ = 0;
  Stats st_;
};

inline Aut aut;

static inline void KalenderWacht() {
  if (DaysFromCivil(2001, 1, 1) != kEpocheTage) {
    fprintf(stderr, "guard - calendar constant wrong\n");
    exit(1);
  }
}

}  // namespace ts_progress

#endif
