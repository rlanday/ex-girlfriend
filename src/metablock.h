#ifndef METABLOCK_H_
#define METABLOCK_H_

#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "neif_levers.h"
#include "ppmd_meta.h"
#include "timestamp_algebra.h"
#if NEIF_E1
#include "b1_interp.h"
#endif

namespace metablock {

static const unsigned char kNL = 0x0a;
static const unsigned char kTab = 0x09;

enum {
  kRevBits = 26, kLenBits = 6,
  kOffBits = 20, kTagBits = 11,
  kTagJahr = 372          // mt = month*31 + day - 32, so 0..371 per year
};
static_assert(kRevBits + kLenBits == 32, "s0 holds revid and length exactly");
static_assert(kOffBits + kTagBits <= 32, "s1 holds offset and day number");

static_assert((1u << kOffBits) > 1000000u,
              "the offset field covers the measured text area with headroom");

class Grenze {
 public:
  enum { kKein = 0, kKopfzeile = 1, kBlockzeile = 2, kHinterBlock = 3 };

  int Byte(unsigned char b) {
    endet_ = false;
    ++pos_;
    if (b != kNL) return kKein;
    ++lf_;
    const unsigned long long kz = (unsigned long long)3;
    const unsigned long long nn = (unsigned long long)243425;
    if (lf_ <= kz) {
      if (lf_ == kz) { ab_ = pos_; offen_ = true; }
      return kKopfzeile;
    }
    if (fertig_) return kHinterBlock;
    idx_ = (long long)(lf_ - kz - 1);        // 0-basiert im Block
    if (lf_ == kz + 2ull * nn) { fertig_ = true; endet_ = true; bis_ = pos_; }
    return kBlockzeile;
  }

  bool Offen() const { return offen_; }
  bool ImBlock() const { return offen_ && !fertig_; }
  bool Fertig() const { return fertig_; }
  bool Endet() const { return endet_; }
  long long ZeileIdx() const { return idx_; }
  unsigned long long Ab() const { return ab_; }
  unsigned long long Bis() const { return bis_; }
  unsigned long long Pos() const { return pos_; }
  unsigned long long Zeilenvorschuebe() const { return lf_; }

 private:
  unsigned long long pos_ = 0, lf_ = 0, ab_ = 0, bis_ = 0;
  long long idx_ = -1;
  bool offen_ = false, fertig_ = false, endet_ = false;
};

struct Stats {
  unsigned long long block_bytes = 0;  // bytes of the block
  unsigned long long a_zeilen = 0, b_zeilen = 0;
  unsigned long long namen = 0;        // distinct names in the text area
  unsigned long long namen_bytes = 0;  // text area
  unsigned long long namen_roh = 0;
  unsigned long long kdiff = 0;
  unsigned long long maxtag = 0, maxrev = 0, maxlen = 0, maxid = 0;
  unsigned long long nachlauf_bytes = 0;
  unsigned long long b_delta = 0;
  unsigned long long b_kurz = 0;
};

class Einheit {
 public:
  void Byte(unsigned char b) {
#if AUX_BUILD
    (void)b;
    return;
#else
    nach_endet_ = false;
    bzeile_ = false;
    switch (g_.Byte(b)) {
      case Grenze::kKein:
        if (g_.ImBlock()) {
          if (zl_n_ < (int)kZeile) zl_[zl_n_] = b;
          ++zl_n_;
        } else if (g_.Fertig() && !tail_closed_) {
          ++st_.nachlauf_bytes;
        }
        return;
      case Grenze::kKopfzeile:
        return;
      case Grenze::kHinterBlock:
        if (!tail_closed_) {
          ++st_.nachlauf_bytes;
          tail_closed_ = true;
          nach_endet_ = true;
          ppmd_meta::block_fertig = true;
          Freigabe();
        }
        return;
      default:
        break;
    }
    Zeile(g_.ZeileIdx());
    zl_n_ = 0;
    if (g_.Endet()) {
      st_.block_bytes = g_.Bis() - g_.Ab();
      Abschluss();
    }
#endif
  }

  // ---- the boundary ------------------------------------------------------
  bool Offen() const { return g_.Offen(); }
  bool ImBlock() const { return g_.ImBlock(); }
  bool Fertig() const { return g_.Fertig(); }
  bool Endet() const { return g_.Endet(); }
  bool NachlaufEndet() const { return nach_endet_; }
  unsigned long long Ab() const { return g_.Ab(); }
  unsigned long long Bis() const { return g_.Bis(); }
  unsigned long long Pos() const { return g_.Pos(); }
  unsigned long long Zeilenvorschuebe() const { return g_.Zeilenvorschuebe(); }

  bool BZeileFertig() const { return bzeile_; }
  size_t BRang() const { return bk_; }
  int BYY() const { return byy_; }
  long long BMT() const { return bmt_; }
  long long BSek() const { return bsek_; }
  long long BT() const { return bt_; }

  size_t Seiten() const { return g_.Fertig() ? (size_t)243425 : (size_t)0; }
  long long RevidNachRang(size_t k) const {
    if (k >= rang_.size()) return -1;
    return (long long)(satz_[3 * (size_t)rang_[k]] >> kLenBits);
  }
  size_t RangZahl() const { return rang_.size(); }
  const unsigned int* Rang() const { return rang_.data(); }

  unsigned int IdVon(size_t page_index) const { return satz_[3 * page_index + 2]; }
  unsigned int RevidVon(size_t page_index) const { return satz_[3 * page_index] >> kLenBits; }
  unsigned int TagnrVon(size_t page_index) const {
    return satz_[3 * page_index + 1] & ((1u << kTagBits) - 1u);
  }
  const unsigned char* AutorVon(size_t page_index, int* n) const {
    const unsigned int len = satz_[3 * page_index] & ((1u << kLenBits) - 1u);
    *n = (int)len;
    return text_.data() + (satz_[3 * page_index + 1] >> kTagBits);
  }

  const unsigned char* Feld(long long page_index, int f, int* n) const {
    if (page_index < 0 || (size_t)page_index >= Seiten()) { *n = 0; return nullptr; }
    const size_t i = (size_t)page_index;
    if (f == 1) return AutorVon(i, n);
    if (f == 0) {
      unsigned int v = satz_[3 * i + 2];
      int nz = 0;
      unsigned char z[12];
      do { z[nz++] = (unsigned char)(0x30 + (int)(v % 10u)); v /= 10u; } while (v);
      for (int j = 0; j < nz; ++j) kratz_id_[j] = z[nz - 1 - j];
      *n = nz;
      return kratz_id_;
    }
    const unsigned int tg = TagnrVon(i);
    const long long jahr = 2001 + (long long)(tg / (unsigned int)kTagJahr);
    const long long mt = (long long)(tg % (unsigned int)kTagJahr);
    const long long monat = mt / 31 + 1;
    kratz_dat_[0] = (unsigned char)(0x30 + (int)((jahr / 1000) % 10));
    kratz_dat_[1] = (unsigned char)(0x30 + (int)((jahr / 100) % 10));
    kratz_dat_[2] = (unsigned char)(0x30 + (int)((jahr / 10) % 10));
    kratz_dat_[3] = (unsigned char)(0x30 + (int)(jahr % 10));
    kratz_dat_[4] = 0x2d;
    kratz_dat_[5] = (unsigned char)(0x30 + (int)(monat / 10));
    kratz_dat_[6] = (unsigned char)(0x30 + (int)(monat % 10));
    int nz = 7;
    {
      const long long tag = mt % 31 + 1;
      kratz_dat_[7] = 0x2d;
      kratz_dat_[8] = (unsigned char)(0x30 + (int)(tag / 10));
      kratz_dat_[9] = (unsigned char)(0x30 + (int)(tag % 10));
      nz = 10;
    }
    *n = nz;
    return kratz_dat_;
  }

  const Stats& GetStats() const { return st_; }

  // Self-report of the unit, left out of the package form.
  void Report(const char* page_index) const {
    (void)page_index;
  }

  void ByteWacht(const char* page_index, unsigned long long strombytes) const {
    if (g_.Offen() && !g_.Fertig()) {
      fprintf(stderr,
          "\nmetablock (%s): opened but never closed, %llu newlines of %llu (%d head lines + 2 x %d records), %llu A and %llu B lines\n",
          page_index, g_.Zeilenvorschuebe(),
          (unsigned long long)3 +
              2ull * (unsigned long long)243425,
          (int)3, (int)243425,
          st_.a_zeilen, st_.b_zeilen);
      fflush(stderr);
      exit(1);
    }
    if (g_.Pos() != strombytes) {
      fprintf(stderr,
          "metablock (%s): byte count differs from the stream\n", page_index);
      fflush(stderr);
      exit(1);
    }
  }

 private:
  enum { kZeile = 256, kHash = 1 << 17 };
  static constexpr unsigned int kLeer = 0xffffffffu;

  static bool Oben(size_t k) {
    return (long long)k >= (long long)87813;
  }
  static bool MpAlpha(unsigned char c) {
    return c >= (unsigned char)0x80 &&
           c <= (unsigned char)(0x80 + 99);
  }
  static bool FestAlpha(unsigned char c) {
    return c >= (unsigned char)0x80 &&
           c <= (unsigned char)(0x80 + 99);
  }

  static void Fail(const char* what) {
    fprintf(stderr,
        "\nmetablock: %s\n", what);
    fflush(stderr);
    exit(1);
  }

  static int KlasseHeuristik(const unsigned char* p, int n) {
    enum { kFenster = 64 };
    if (n <= 0) return 0;
    const int m = n < (int)kFenster ? n : (int)kFenster;
    {
      int i = 0;
      if (i < m && p[i] == 0x2d) ++i;
      const int d0 = i;
      while (i < m && p[i] >= 0x30 && p[i] <= 0x39) ++i;
      int e = m;
      for (int j = i; j < m; ++j) if (p[j] == kTab) { e = j; break; }
      if (i > d0 && e - i == 4) {
        bool alpha = true;
        for (int j = i; j < e; ++j) if (!FestAlpha(p[j])) { alpha = false; break; }
        if (alpha) return 1;
      }
#if NEIF_E1
      // B1 line: delta '+' [-]residual
      if (i > d0 && i < e && p[i] == b1_interp::kMark) {
        int j = i + 1;
        if (j < e && p[j] == 0x2d) ++j;
        const int r0 = j;
        while (j < e && p[j] >= 0x30 && p[j] <= 0x39) ++j;
        if (j > r0 && j == e) return 1;
      }
#endif
    }
    return 2;
  }

  void Zeile(long long idx) {
    if (zl_n_ > (int)kZeile) Fail("metablock line longer than the buffer");
    const int n = zl_n_;
    if (n <= 0) Fail("empty metablock line");
    const int art = (idx < (long long)243425) ? 1 : 2;
    if (KlasseHeuristik(zl_, n) != art) ++st_.kdiff;
    if (art == 1) ZeileA((size_t)idx, zl_, n);
    else ZeileB((size_t)(idx - (long long)243425), zl_, n);
  }

  void ZeileA(size_t i, const unsigned char* p, int n) {
    if (satz_.empty()) {
      satz_.assign(3u * (size_t)243425, 0u);
      text_.reserve(600000);
      hoff_.assign((size_t)kHash, kLeer);
      hlen_.assign((size_t)kHash, 0u);
    }
    const unsigned char* e = p + n;
    const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
    if (!t) t = e;
#if NEIF_E1
    const unsigned char* const fe = t;    // end of field 1
    const unsigned char* mp_rv = nullptr;
    bool b1 = false;
    {
      const unsigned char* dz = p;
      if (dz < fe && *dz == 0x2d) ++dz;
      while (dz < fe && *dz >= 0x30 && *dz <= 0x39) ++dz;
      b1 = dz < fe && *dz == b1_interp::kMark;
      mp_rv = b1 ? dz : fe - 4;
    }
#else
    const unsigned char* mp_rv = t - 4;
#endif
    if (mp_rv < p) Fail("A line too short for the revid");
    t = mp_rv;                            // the page id delta ends here
    long long d = 0;
    int vz = 1;
    const unsigned char* q = p;
    if (q < t && *q == 0x2d) { vz = -1; ++q; }      // `-`: ids may go back
    if (q == t) Fail("page id delta empty");
    for (; q < t; ++q) {
      if (*q < 0x30 || *q > 0x39) Fail("page id delta not numeric");
      d = d * 10 + (*q - 0x30);
    }
    id_lauf_ += vz * d;
    if (id_lauf_ < 0) Fail("running page id negative");
    long long rv = 0;
#if NEIF_E1
    if (b1) {                             // '+' [-]residual vs the interpolation
      const unsigned char* q2 = mp_rv + 1;
      long long res = 0;
      int vz2 = 1;
      if (q2 < fe && *q2 == 0x2d) { vz2 = -1; ++q2; }
      if (q2 == fe) Fail("B1 line without revid residual");
      for (; q2 < fe; ++q2) {
        if (*q2 < 0x30 || *q2 > 0x39) Fail("revid residual not numeric");
        res = res * 10 + (*q2 - 0x30);
      }
      rv = b1_.Prognose(id_lauf_) + vz2 * res;
      if (rv < 0) Fail("B1 revid negative");
      b1_.Add(id_lauf_, rv);
    } else
#endif
    for (int mq = 0; mq < 4; ++mq) {
      if (!MpAlpha(mp_rv[mq]))
        Fail("revid digit outside the base-100 alphabet");
      rv = rv * 100 + (long long)((unsigned)mp_rv[mq] - (unsigned)0x80);
    }
    if (rv >= (long long)(1u << kRevBits)) Fail("revid exceeds 26 bits");
    if (id_lauf_ > 0xffffffffLL) Fail("page id exceeds 32 bits");
    satz_[3 * i] = (unsigned int)rv << kLenBits;
    satz_[3 * i + 2] = (unsigned int)id_lauf_;
    if ((unsigned long long)rv > st_.maxrev) st_.maxrev = (unsigned long long)rv;
    if ((unsigned long long)id_lauf_ > st_.maxid)
      st_.maxid = (unsigned long long)id_lauf_;
    ++st_.a_zeilen;
  }

  void RangBauen() {
    const size_t N = (size_t)243425;
    rang_.resize(N);
    for (size_t i = 0; i < N; ++i) rang_[i] = (unsigned int)i;
    const unsigned int* s = satz_.data();
    std::sort(rang_.begin(), rang_.end(),
        [s](unsigned int a, unsigned int b) {
          const unsigned int ra = s[3 * (size_t)a] >> kLenBits;
          const unsigned int rb = s[3 * (size_t)b] >> kLenBits;
          return ra != rb ? ra < rb : a < b;
        });
    for (size_t k = 1; k < N; ++k)
      if ((s[3 * (size_t)rang_[k]] >> kLenBits) ==
          (s[3 * (size_t)rang_[k - 1]] >> kLenBits))
        Fail("duplicate revid");
  }

  // ---- part B: `timestamp TAB name [TAB uid]` -----------------------------
  void ZeileB(size_t k, const unsigned char* p, int n) {
    if (rang_.empty()) RangBauen();
    if (k >= rang_.size()) Fail("more B records than revids");
    const size_t i = (size_t)rang_[k];
    const unsigned char* e = p + n;
    const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
    if (!t) Fail("B line without tab");
    {
      const int m = (int)(t - p);
      int yy = 0;
      long long mt = 0, sek = 0;
      long long t_k = 0;
      bool gerechnet = false;
      if (Oben(k)) {
        if (m <= 0 || m > 24) Fail("B field 0 delta of impossible length");
        const long long dv = ts_algebra::ZahlRoh((const char*)p, m);
        t_k = (k ? t_vor_ : 0LL) + dv;
        gerechnet = true;
        ++st_.b_delta;
      }
      if (!gerechnet && !Oben(k)) {
        const int br = 6;                 // short form: six base-100 bytes
        if (m != br) Fail("B field 0 not short-form width");
        for (int j = 0; j < br; ++j)
          if (!MpAlpha(p[j])) Fail("short form: byte outside the alphabet");
        const int a0 = (int)(unsigned)0x80;
        {
          yy = (int)p[0] - a0;
          mt = (long long)((int)p[1] - a0) * 100 + ((int)p[2] - a0);
          sek = ((long long)((int)p[3] - a0) * 100 + ((int)p[4] - a0)) * 100 +
                ((int)p[5] - a0);
          if (sek > 86399) Fail("short form: seconds over 86399");
          t_k = ts_algebra::TsZuSek(yy, mt, sek);
        }
        ++st_.b_kurz;
      }
      if (gerechnet) {
        int yy2 = 0; long long mt2 = 0, sek2 = 0;
        ts_algebra::SekZuTs(t_k, &yy2, &mt2, &sek2);
        yy = yy2; mt = mt2; sek = sek2;
        if (yy < 0 || yy > 99)
          Fail("year outside 2001..2100");
      }
      if (mt < 0 || mt > (long long)kTagJahr - 1)
        Fail("timestamp month/day outside 0..371");
      const long long tg = (long long)yy * (long long)kTagJahr + mt;
      if (tg >= (long long)(1u << kTagBits))
        Fail("day number exceeds 11 bits");
      byy_ = yy; bmt_ = mt; bsek_ = sek;
      bt_ = t_k;
      tag_ = (unsigned int)tg;
      if ((unsigned long long)tg > st_.maxtag) st_.maxtag = (unsigned long long)tg;
      t_vor_ = t_k;
    }
    const unsigned char* n2 =
        (const unsigned char*)memchr(t + 1, kTab, (size_t)(e - t - 1));
    const unsigned char* ne = n2 ? n2 : e;
    const int len = (int)(ne - t - 1);
    if (len >= (1 << kLenBits)) Fail("author name over 63 bytes");
    const unsigned int off = InternName(t + 1, len);
    if (off >= (1u << kOffBits)) Fail("author offset exceeds 20 bits");
    satz_[3 * i] |= (unsigned int)len;
    satz_[3 * i + 1] = (off << kTagBits) | tag_;
    st_.namen_roh += (unsigned long long)len;
    if ((unsigned long long)len > st_.maxlen) st_.maxlen = (unsigned long long)len;
    ++st_.b_zeilen;
    bzeile_ = true;
    bk_ = k;
  }

  static unsigned long long Fnv(const unsigned char* p, int n) {
    unsigned long long h = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) {
      h ^= (unsigned long long)p[i];
      h *= 1099511628211ull;
    }
    return h;
  }

  unsigned int InternName(const unsigned char* p, int len) {
    if (len <= 0) return 0u;
    if (st_.namen * 2ull >= (unsigned long long)kHash)
      Fail("name hash table half full");
    size_t m = (size_t)(Fnv(p, len) & (unsigned long long)(kHash - 1));
    while (hoff_[m] != kLeer) {
      if ((int)hlen_[m] == len &&
          memcmp(text_.data() + hoff_[m], p, (size_t)len) == 0)
        return hoff_[m];
      m = (m + 1) & (size_t)(kHash - 1);
    }
    const unsigned int o = (unsigned int)text_.size();
    text_.insert(text_.end(), p, p + len);
    hoff_[m] = o;
    hlen_[m] = (unsigned int)len;
    ++st_.namen;
    st_.namen_bytes = (unsigned long long)text_.size();
    return o;
  }

  void Abschluss() {
    if (st_.a_zeilen != (unsigned long long)243425 ||
        st_.b_zeilen != (unsigned long long)243425)
      Fail("A or B line count differs from the page count");
    text_.shrink_to_fit();
#if NEIF_E1
    b1_.Clear();
#endif
    std::vector<unsigned int>().swap(hoff_);
    std::vector<unsigned int>().swap(hlen_);
  }

  void Freigabe() {
    std::vector<unsigned int>().swap(rang_);
  }

  unsigned char zl_[kZeile] = {0};
  int zl_n_ = 0;
  Grenze g_;
  bool tail_closed_ = false, nach_endet_ = false, bzeile_ = false;

  long long id_lauf_ = 0;
#if NEIF_E1
  b1_interp::Interp b1_;             // E1: B1 page id -> revid, stream order
#endif
  size_t bk_ = 0;
  int byy_ = 0;
  long long bmt_ = 0, bsek_ = 0;
  long long bt_ = 0;
  long long t_vor_ = 0;
  unsigned int tag_ = 0;
  std::vector<unsigned int> satz_;   // 3 columns per page
  std::vector<unsigned char> text_;  // Name1Name2Name3...
  std::vector<unsigned int> rang_;   // rank -> page (temporary)
  std::vector<unsigned int> hoff_, hlen_;
  static const int kKratz = 12;
  static_assert(kKratz >= 10,
      "the scratch buffers of Feld() hold up to 10 characters - "
      "a ten-digit id (unsigned int) or YYYY-MM-DD");
  mutable unsigned char kratz_id_[kKratz] = {0};   // Feld() f == 0 (id)
  mutable unsigned char kratz_dat_[kKratz] = {0};  // Feld() f == 2 (date)
  Stats st_;
};

inline Einheit mk;

}

#endif
