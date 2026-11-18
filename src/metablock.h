// Metadata block reader.  One unit parses the metadata block once (part A
// "page id delta, revid, restrictions", part B "timestamp, author") into a
// table per page (revid, day number, author) and answers the author channel,
// the timestamp forecast, the transformer gate and the metadata prompt from
// it.  The block end is found by counting line feeds (3 + 2N lines for N
// pages, then the census line), never by a stored byte length.  Tables needed
// only while parsing (ranks, the name hash table, the seconds of the
// timestamps) are released at the block end.  The unit also tracks the text
// start behind the block (slots and the name grammar of the author article).
#ifndef METABLOCK_H_
#define METABLOCK_H_
#include "meld.h"
#include "u8_append.h"

#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif
// Two levers of the block form are taken over from ex-girlfriend 1.0 (Ryan
// Landay, GPL-3.0-only, a modification of cmix-neif 1.0; files
// b1_interp.h, metablock_form.h, metablock.h, id_exclude.h, ts_progress.h of
// github.com/rlanday/ex-girlfriend e923faec):
//   part A: a revid below 17,000,000 (MediaWiki 1.5 conversion, assigned in
//     page-id order) is written as `<dpageid>+[-]<residual>` against a linear
//     interpolation over the nearest page ids already seen in the block
//     (their E1; the forecast below, the block codec, the exclusion model).
//   part B: above the edge the injected timestamp forecast is the slope over
//     the last 9 records above the edge (their E3b); below it the median
//     estimator stays.
// On enwik9 they change the coded block by -21.4 KB (part A) and -2.2 KB (part B).

// ---------------------------------------------------------------------------
//


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "ppmd_meta.h"
#include "timestamp_algebra.h"

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

// ---------------------------------------------------------------------------
// Taken over from ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only), b1_interp.h
// and the E1 hunks of metablock_form.h / metablock.h: the MediaWiki 1.5
// conversion assigned the revids below 17,000,000 in page-id order, so such a
// revid is forecast by linear interpolation between the nearest pages (by
// page id) already seen among these lines (78 % exact, 92 % within +-1 on
// enwik9).  Same rule as there (floor division, one neighbour: slope 1, none:
// 15,898,943); their std::map is replaced by one page-id array (2 MiB, one
// allocation, released at the block end).  The writer (Pack of the block
// codec), the reader (Unpack) and this unit (ZeileA) all feed it in stream
// order, so coder and decoder see the same set at every line.

// The one abort of the unit (the revid forecast uses it as well).
__attribute__((noinline)) static void Fail(const char* what) {
  fprintf(stderr, "\n%s\n", what);
  fflush(stderr);
  exit(1);
}

namespace meta_fields {

static const long long kMax = 17000000LL;
static const long long kLeer = 15898943LL;  // forecast of an empty set
static const unsigned char kMarke = 0x2b;   // '+' ends the page id delta
static const long long kPg = 1LL << 19;     // page ids 0 .. 524,287
static_assert(kMarke != 0x09 && kMarke != 0x0a && kMarke != 0x2d &&
              (kMarke < 0x30 || kMarke > 0x39) && kMarke < 0x80,
    "the B1 mark must be neither TAB, LF, '-', a digit nor a base-100 byte");

__attribute__((noinline)) static long long Zahl(const unsigned char* p,
                                                const unsigned char* e) {
  long long vz = 1, v = 0;
  if (p < e && *p == 0x2d) { vz = -1; ++p; }
  if (p == e) __builtin_unreachable();
  for (; p < e; ++p) {
    if (*p < 0x30 || *p > 0x39)
      __builtin_unreachable();
    v = v * 10 + (*p - 0x30);
  }
  return vz * v;
}

// `[-]digits` from p, then the mark: its position, else nullptr.
__attribute__((noinline)) static const unsigned char* Marke(
    const unsigned char* p, const unsigned char* e) {
  if (p < e && *p == 0x2d) ++p;
  const unsigned char* d0 = p;
  while (p < e && *p >= 0x30 && *p <= 0x39) ++p;
  return (p > d0 && p < e && *p == kMarke) ? p : nullptr;
}

class Interp {
 public:
  // The part-A line of page id p with a forecast revid.  Writer (residual =
  // false): x is the revid, the result the residual to write.  Reader
  // (residual = true): x is the
  // residual read, the result the revid.  Either way the pair goes into the
  // set, so both sides grow it identically.
  __attribute__((noinline)) long long Paar(long long p, long long x,
                                           bool residual) {
    if (p < 0 || p >= kPg)
      __builtin_unreachable();
    if (rv_.empty()) rv_.assign((size_t)kPg, 0u);
    long long q2 = p, q1 = p - 1;   // nearest set page id >= p and < p
    while (q2 <= hi_ && !rv_[q2]) ++q2;
    while (q1 >= lo_ && !rv_[q1]) --q1;
    const bool h1 = q1 >= lo_, h2 = q2 <= hi_;
    long long f = kLeer;
    if (h1 && h2) {
      const long long r1 = rv_[q1], r2 = rv_[q2];
      f = r1 + ts_algebra::Fdiv((p - q1) * (r2 - r1), q2 - q1);
    } else if (h1) {
      f = (long long)rv_[q1] + (p - q1);
    } else if (h2) {
      f = (long long)rv_[q2] - (q2 - p);
    }
    const long long r = residual ? f + x : x;
    // r > 0: 0 marks a free page id
    if (r <= 0 || r >= kMax) __builtin_unreachable();
    if (rv_[p]) __builtin_unreachable();
    rv_[p] = (unsigned int)r;
    if (p < lo_) lo_ = p;
    if (p > hi_) hi_ = p;
    return residual ? r : r - f;
  }
  void Frei() { std::vector<unsigned int>().swap(rv_); }

 private:
  std::vector<unsigned int> rv_;
  long long lo_ = kPg, hi_ = -1;
};

}  // namespace meta_fields
class Grenze {
 public:
  enum { kKein = 0, kKopfzeile = 1, kBlockzeile = 2, kHinterBlock = 3 };
  enum { kZensuszeile = 4 };
  enum { kTextanfang = 5 };
  static const unsigned char kEnde = 0x0B;   // end mark of a slot: 0x0B 0x0B

  int Byte(unsigned char b) {
    endet_ = false;
    meta_endet_ = false;
    ta_slotende_ = false;
    ++pos_;
    if (fertig_ && ta_rest_) {
      if (b == kEnde && ta_vor_ == kEnde) {
        --ta_rest_;
        ta_vor_ = 0;
        ta_slotende_ = true;
      } else {
        ta_vor_ = b;
      }
      return kTextanfang;
    }
    if (b != kNL) return kKein;
    ++lf_;
    const unsigned long long kz = (unsigned long long)3;
    const unsigned long long nn = (unsigned long long)243425;
    if (lf_ <= kz) {
      if (lf_ == kz) { ab_ = pos_; offen_ = true; }
      return kKopfzeile;
    }
    if (fertig_) return kHinterBlock;
    if (zens_) { zens_ = false; fertig_ = true; endet_ = true; bis_ = pos_; return kZensuszeile; }
    idx_ = (long long)(lf_ - kz - 1);        // 0-based inside the block
    if (lf_ == kz + 2ull * nn) {
      meta_endet_ = true;
      if (zensus_) { zens_ = true; return kBlockzeile; }   // the census line follows
      fertig_ = true; endet_ = true; bis_ = pos_;
    }
    return kBlockzeile;
  }

  bool Offen() const { return offen_; }
  bool ImBlock() const { return offen_ && !fertig_; }
  bool Fertig() const { return fertig_; }
  bool Endet() const { return endet_; }
  void Textanfang() { ta_rest_ = 3; ta_vor_ = 0; }
  // true while the NEXT byte is a slot byte (state after the bytes fed so far)
  bool ImTextanfang() const { return fertig_ && ta_rest_ > 0; }
  // slots still open
  int TaRest() const { return ta_rest_; }
  // true for exactly one byte: the last byte of a slot's end mark
  bool TaSlotEndet() const { return ta_slotende_; }
  // true for exactly one byte: the last byte of the text start
  bool TaFertig() const { return ta_slotende_ && ta_rest_ == 0; }
  // would byte b close the current slot?  (Readers that see a byte before the
  // unit does: the transformer's document cut.)
  bool TaEndmarke(unsigned char b) const {
    return ImTextanfang() && ta_vor_ == kEnde && b == kEnde;
  }
  // would byte b be the line feed that ends the whole metablock?  (Same
  // readers: the document cut in front of the first slot.)
  bool SchliesstMit(unsigned char b) const {
    if (fertig_ || b != kNL) return false;
    if (zensus_) return zens_;
    return lf_ + 1 == (unsigned long long)3 +
                      2ull * (unsigned long long)243425;
  }
  void Zensus() { zensus_ = true; }
  bool ImZensus() const { return zens_; }
  // true for exactly one byte: the line feed of the last block line
  bool MetaEndet() const { return meta_endet_; }
  long long ZeileIdx() const { return idx_; }
  unsigned long long Ab() const { return ab_; }
  unsigned long long Bis() const { return bis_; }
  unsigned long long Pos() const { return pos_; }
  unsigned long long Zeilenvorschuebe() const { return lf_; }

 private:
  unsigned long long pos_ = 0, lf_ = 0, ab_ = 0, bis_ = 0;
  long long idx_ = -1;
  bool offen_ = false, fertig_ = false, endet_ = false;
  int ta_rest_ = 0;              // slots not yet closed (0 = no text start)
  unsigned char ta_vor_ = 0;     // previous byte inside the current slot
  bool ta_slotende_ = false;
  bool zensus_ = false, zens_ = false, meta_endet_ = false;
};

struct Stats {
  unsigned long long a_zeilen = 0, b_zeilen = 0;
  unsigned long long namen = 0;        // distinct names in the text area
  unsigned long long ta_bytes = 0;
};

namespace zw {
inline bool an = false;              // the next stream byte carries forced bits
inline int art = 0;                  // 1 the byte behind a copy mark, 2 name grammar
inline signed char line_buf[256];
inline unsigned long long bytes[5] = {0, 0, 0, 0, 0}, bits[5] = {0, 0, 0, 0, 0};   // per kind
// the row for the set of allowed bytes v[0..n): a bit is forced where every
// allowed byte with the same prefix agrees; true if any bit is forced
__attribute__((noinline)) inline bool Setze(const unsigned char* v, int n) {
  unsigned char z[256];              // 1: a 0 follows this prefix, 2: a 1 follows
  memset(z, 0, sizeof z);
  for (int i = 0; i < n; ++i) {
    unsigned int ctx = 1;
    for (int j = 7; j >= 0; --j) {
      const int bit = (v[i] >> j) & 1;
      z[ctx] |= (unsigned char)(1 + bit);
      ctx += ctx + (unsigned int)bit;
    }
  }
  bool fest = false;
  for (int c = 0; c < 256; ++c) {
    line_buf[c] = (signed char)(z[c] == 1 ? 0 : (z[c] == 2 ? 1 : -1));
    fest = fest || line_buf[c] >= 0;
  }
  return fest;
}
}  // namespace zw

class Einheit {
 public:
  Einheit() {
    g_.Zensus();
    g_.Textanfang();
  }
  void Byte(unsigned char b) {
#if AUX_BUILD
    (void)b;
    return;
#else
    nach_endet_ = false;
    bzeile_ = false;
    ta_byte_ = false;
    ta_zeile_folgt_ = -1;
    switch (g_.Byte(b)) {
      case Grenze::kTextanfang:
        TaByte(b);
        return;
      case Grenze::kKein:
        if (g_.ImBlock()) {
          if (zl_n_ < (int)kZeile) zl_[zl_n_] = b;
          ++zl_n_;
        }
        return;
      case Grenze::kKopfzeile:
        return;
      case Grenze::kHinterBlock:
        if (!tail_closed_) {
          tail_closed_ = true;
          nach_endet_ = true;
          ppmd_meta::block_fertig = true;
          Freigabe();
        }
        return;
      case Grenze::kZensuszeile:
        zl_n_ = 0;
        TaStart();
        return;
      default:
        break;
    }
    TitleLine(g_.ZeileIdx());
    zl_n_ = 0;
    if (g_.MetaEndet()) {
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
  bool MetaEndet() const { return g_.MetaEndet(); }
  size_t SatzZahl() const { return satz_.size() / 3; }
  unsigned long long Ab() const { return g_.Ab(); }
  unsigned long long Bis() const { return g_.Bis(); }
  unsigned long long Pos() const { return g_.Pos(); }
  unsigned long long Zeilenvorschuebe() const { return g_.Zeilenvorschuebe(); }

  bool ImTextanfang() const { return g_.ImTextanfang(); }
  bool TaByte() const { return ta_byte_; }
  bool TaSlotEndet() const { return g_.TaSlotEndet(); }
  bool TaFertig() const { return g_.TaFertig(); }
  bool TaEndmarke(unsigned char b) const { return g_.TaEndmarke(b); }
  // would byte b close the LAST slot (the byte behind it is page 0)?
  bool TaSchliesstLetzten(unsigned char b) const { return g_.TaEndmarke(b) && g_.TaRest() == 1; }
  bool SchliesstMit(unsigned char b) const { return g_.SchliesstMit(b); }
  // the slot of the byte just fed as its bit (1, 2, 4 = slots 1, 2, 3), 0
  // outside the text start
  int TaSlot() const { return ta_byte_ ? ta_slot_ : 0; }
  // the slot (bit 1, 2 or 4) that begins behind the byte just fed: behind the end
  // of the metablock the first one, behind a slot end the next present one,
  // 0 otherwise
  int TaNaechsterSlot() const {
    if (g_.Endet()) return TaSlotBit(0);
    if (!g_.TaSlotEndet() || g_.TaRest() == 0) return 0;
    return TaSlotBit(3 - g_.TaRest());
  }
  // the content of slot bit s (1, 2 or 4) without its end mark; valid from the
  // end of that slot until the line behind the text start (then released)
  const unsigned char* TaPuffer(int s, size_t* n) const {
    const int i = s == 1 ? 0 : (s == 2 ? 1 : 2);
    *n = ta_buf_[i].size();
    return ta_buf_[i].data();
  }
  // article line j begins behind the byte just fed (-1: none)
  int TaZeileFolgt() const { return ta_zeile_folgt_; }
  __attribute__((noinline)) bool TaZwSetze() {
    const bool in2 = (ta_byte_ && ta_slot_ == 2 && !g_.TaSlotEndet()) || TaNaechsterSlot() == 2;
    if (!in2 || ta_lf_ >= ta_named_n_ || ta_art_[ta_lf_] == 1) return false;
    unsigned char v[257];
    int n = 0;
    for (unsigned int c = gr_[3 * gr_k_]; c; c = gr_[3 * c + 1]) v[n++] = (unsigned char)gr_[3 * c + 2];
    if ((gr_[3 * gr_k_ + 2] >> 8)
        )
      v[n++] = kNL;
    zw::art = 2;
    return zw::Setze(v, n);
  }
  // the context injected in front of article line j (day number, kind, tab)
  int TaOkb(int j, const unsigned char** p) {
    if (j < 0 || (size_t)j + 1 >= ta_okb_off_.size()) return 0;
    const unsigned int a = ta_okb_off_[(size_t)j], e = ta_okb_off_[(size_t)j + 1];
    *p = ta_okb_.data() + a;
    return (int)(e - a);
  }

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
    if (ta_namen_offen_) __builtin_unreachable();
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
      __builtin_unreachable()

;
      fflush(stderr);
      exit(1);
    }
    if (g_.Pos() != strombytes) {
      __builtin_unreachable()
;
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

  void TitleLine(long long idx) {
    if (zl_n_ > (int)kZeile) __builtin_unreachable();
    const int n = zl_n_;
    if (n <= 0) __builtin_unreachable();
    const int art = (idx < (long long)243425) ? 1 : 2;
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
    const unsigned char* const fe = t;
    const unsigned char* mp_rv = meta_fields::Marke(p, fe);
    const bool b1 = mp_rv != nullptr;
    if (!b1) mp_rv = fe - 4;
    if (mp_rv < p) __builtin_unreachable();
    // the page id delta ends at mp_rv; `-`: ids may go back
    id_lauf_ += meta_fields::Zahl(p, mp_rv);
    if (id_lauf_ < 0) __builtin_unreachable();
    long long rv = 0;
    if (b1) rv = b1_.Paar(id_lauf_, meta_fields::Zahl(mp_rv + 1, fe), true);
    else
    for (int mq = 0; mq < 4; ++mq) {
      if (!MpAlpha(mp_rv[mq]))
        __builtin_unreachable();
      rv = rv * 100 + (long long)((unsigned)mp_rv[mq] - (unsigned)0x80);
    }
    if (rv >= (long long)(1u << kRevBits)) __builtin_unreachable();
    if (id_lauf_ > 0xffffffffLL) __builtin_unreachable();
    satz_[3 * i] = (unsigned int)rv << kLenBits;
    satz_[3 * i + 2] = (unsigned int)id_lauf_;
    ++st_.a_zeilen;
  }

  static inline const unsigned int* rang_satz_ = nullptr;
  static int RangVergleich(const void* x, const void* y) {
    const unsigned int a = *(const unsigned int*)x, b = *(const unsigned int*)y;
    const unsigned int ra = rang_satz_[3 * (size_t)a] >> kLenBits;
    const unsigned int rb = rang_satz_[3 * (size_t)b] >> kLenBits;
    return ra != rb ? (ra < rb ? -1 : 1) : (a < b ? -1 : (a > b ? 1 : 0));
  }
  void RangBauen() {
    const size_t N = (size_t)243425;
    std::vector<unsigned int>(N, 0u).swap(rang_);   // rang_ is empty: as resize(N), without its instance
    for (size_t i = 0; i < N; ++i) rang_[i] = (unsigned int)i;
    const unsigned int* s = satz_.data();
    rang_satz_ = s;
    if (rang_.begin() != rang_.end())
      qsort(rang_.data(), rang_.size(), sizeof(unsigned int), RangVergleich);
    for (size_t k = 1; k < N; ++k)
      if ((s[3 * (size_t)rang_[k]] >> kLenBits) ==
          (s[3 * (size_t)rang_[k - 1]] >> kLenBits))
        __builtin_unreachable();
  }

  // ---- part B: `timestamp TAB name [TAB uid]` -----------------------------
  void ZeileB(size_t k, const unsigned char* p, int n) {
    if (rang_.empty()) RangBauen();
    if (k >= rang_.size()) __builtin_unreachable();
    const size_t i = (size_t)rang_[k];
    const unsigned char* e = p + n;
    const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
    if (!t) __builtin_unreachable();
    {
      const int m = (int)(t - p);
      int yy = 0;
      long long mt = 0, sek = 0;
      long long t_k = 0;
      bool gerechnet = false;
      if (Oben(k)) {
        if (m <= 0 || m > 24) __builtin_unreachable();
        const long long dv = meta_fields::Zahl(p, p + m);
        t_k = (k ? t_vor_ : 0LL) + dv;
        gerechnet = true;
      }
      if (!gerechnet && !Oben(k)) {
        const int br = 6;                 // short form: six base-100 bytes
        if (m != br) __builtin_unreachable();
        for (int j = 0; j < br; ++j)
          if (!MpAlpha(p[j])) __builtin_unreachable();
        const int a0 = (int)(unsigned)0x80;
        {
          yy = (int)p[0] - a0;
          mt = (long long)((int)p[1] - a0) * 100 + ((int)p[2] - a0);
          sek = ((long long)((int)p[3] - a0) * 100 + ((int)p[4] - a0)) * 100 +
                ((int)p[5] - a0);
          if (sek > 86399) __builtin_unreachable();
          t_k = ts_algebra::TsZuSek(yy, mt, sek);
        }
      }
      if (gerechnet) {
        int yy2 = 0; long long mt2 = 0, sek2 = 0;
        ts_algebra::SekZuTs(t_k, &yy2, &mt2, &sek2);
        yy = yy2; mt = mt2; sek = sek2;
        if (yy < 0 || yy > 99)
          __builtin_unreachable();
      }
      if (mt < 0 || mt > (long long)kTagJahr - 1)
        __builtin_unreachable();
      const long long tg = (long long)yy * (long long)kTagJahr + mt;
      if (tg >= (long long)(1u << kTagBits))
        __builtin_unreachable();
      byy_ = yy; bmt_ = mt; bsek_ = sek;
      bt_ = t_k;
      tag_ = (unsigned int)tg;
      t_vor_ = t_k;
    }
    const unsigned char* n2 =
        (const unsigned char*)memchr(t + 1, kTab, (size_t)(e - t - 1));
    const unsigned char* ne = n2 ? n2 : e;
    const int len = (int)(ne - t - 1);
    TaSatz(i, p, t, ne, e, len);
    ++st_.b_zeilen;
    bzeile_ = true;
    bk_ = k;
  }

  static bool IstTaMarke(unsigned char c) {
    return c == (unsigned char)0x05 || c == (unsigned char)0x03 ||
           c == (unsigned char)0x03;
  }
  void TaSatz(size_t i, const unsigned char* p, const unsigned char* t,
              const unsigned char* ne, const unsigned char* e, int len) {
    const bool uid = ne < e && ne + 1 < e;          // a third field with content
    int art = 0;
    if (len == 1 && IstTaMarke(t[1])) {
      art = t[1] == (unsigned char)0x05 ? 1 : 3;
    }
    (void)uid;
    // the deferred record: page, day number, named / address (length,
    // offset of its bytes)
    unsigned int w1 = tag_;
    unsigned int w2 = 0;
    if (art) {
      w1 |= 1u << kTagBits;
      ++ta_named_n_;
      ta_art_.push_back((unsigned char)art);
    } else {
      if (len >= (1 << kLenBits)) __builtin_unreachable();
      w1 |= (unsigned int)len << (kTagBits + 1);
      w2 = (unsigned int)ta_ip_.size();
      u8_append::Anh(ta_ip_, t + 1, ne);
    }
    ta_rec_.push_back((unsigned int)i);
    ta_rec_.push_back(w1);
    ta_rec_.push_back(w2);
    // The day number has readers in front of slot 2 (the census context,
    // KontextBytes, is coded in the census line): it enters the table at
    // once; offset and length follow at the end of slot 2.  (Entered later,
    // the census context would read day digit 0x80 on every record, and the
    // table checksum, taken at the end, would not see it.)
    satz_[3 * i + 1] = tag_;
    // the context in front of the article line of this record: the day
    // number (the key the name had at its place in part B) and the kind (new
    // or repeated); the kind next to the name measured better than in front
    // of the day
    if (art) {
      if (ta_okb_off_.empty()) ta_okb_off_.push_back(0u);
      const unsigned char a0 = (unsigned char)0x80;
      ta_okb_.push_back((unsigned char)(a0 + tag_ / 100u));
      ta_okb_.push_back((unsigned char)(a0 + tag_ % 100u));
      ta_okb_.push_back(art == 1 ? (unsigned char)0x05 : (art == 2 ? (unsigned char)0x01 : (unsigned char)0x03));
      ta_okb_.push_back(kTab);
      ta_okb_off_.push_back((unsigned int)ta_okb_.size());
    }
    (void)p;
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
      __builtin_unreachable();
    size_t m = (size_t)(Fnv(p, len) & (unsigned long long)(kHash - 1));
    while (hoff_[m] != kLeer) {
      if ((int)hlen_[m] == len &&
          memcmp(text_.data() + hoff_[m], p, (size_t)len) == 0)
        return hoff_[m];
      m = (m + 1) & (size_t)(kHash - 1);
    }
    const unsigned int o = (unsigned int)text_.size();
    u8_append::Anh(text_, p, p + len);
    hoff_[m] = o;
    hlen_[m] = (unsigned int)len;
    ++st_.namen;
    return o;
  }

  void Abschluss() {
    if (st_.a_zeilen != (unsigned long long)243425 ||
        st_.b_zeilen != (unsigned long long)243425)
      Fail("E103");
    b1_.Frei();
  }

  void Freigabe() {
    std::vector<unsigned int>().swap(rang_);
    for (int i = 0; i < 3; ++i) std::vector<unsigned char>().swap(ta_buf_[i]);
  }

  static int TaSlotBit(int ord) {
    for (int bit = 1; bit <= 4; bit <<= 1) {
      if (!((7) & bit)) continue;
      if (ord-- == 0) return bit;
    }
    return 0;
  }
  // the line feed that ends the whole metablock: the slots follow
  void TaStart() {
    // the PPMd restart in front of the slots (with the census line both
    // restarts are set already; without it the restart of the line behind
    // the block would fall behind the slots)
    ppmd_meta::block_fertig = true;
    if (TaSlotBit(0) == 2 && ta_named_n_ > 0) ta_zeile_folgt_ = 0;
    for (int i = 0; i < 3; ++i) gr_.push_back(0u);   // stage 2: the root of the name grammar
  }
  // one slot byte: into the buffer of its slot; at the end mark its first
  // byte leaves the buffer again
  void TaByte(unsigned char b) {
    const int ord = 3 - g_.TaRest() - (g_.TaSlotEndet() ? 1 : 0);
    const int bit = TaSlotBit(ord);
    const int i = bit == 1 ? 0 : (bit == 2 ? 1 : 2);
    ta_byte_ = true;
    ta_slot_ = bit;
    ++st_.ta_bytes;
    std::vector<unsigned char>& v = ta_buf_[i];
    if (!g_.TaSlotEndet()) {
      v.push_back(b);
      if (bit == 2) GrByte(b);
      if (bit == 2 && b == kNL && ++ta_lf_ < ta_named_n_) ta_zeile_folgt_ = (int)ta_lf_;
      return;
    }
    v.pop_back();                          // the first 0x0B of the end mark
    if (bit == 2) TaNamenEintragen();
    if (TaNaechsterSlot() == 2 && ta_named_n_ > 0) ta_zeile_folgt_ = 0;
  }
  // the second phase of the table: at the end of slot 2 every deferred
  // part-B record enters the text area in part-B order - a named one with
  // the next article line, an address with its own bytes - exactly the
  // InternName() sequence of the form without the article
  void TaNamenEintragen() {
    const std::vector<unsigned char>& v = ta_buf_[1];
    std::vector<unsigned int> z;
    z.reserve((size_t)ta_named_n_ + 1u);
    z.push_back(0u);
    for (size_t k = 0; k < v.size(); ++k)
      if (v[k] == kNL) z.push_back((unsigned int)(k + 1));
    if (!v.empty() && v.back() != kNL)
      __builtin_unreachable();
    const size_t zeilen = z.size() - 1;
    if (zeilen != (size_t)ta_named_n_)
      __builtin_unreachable();
    size_t j = 0;
    for (size_t r = 0; r + 2 < ta_rec_.size(); r += 3) {
      const size_t page_index = ta_rec_[r];
      const unsigned int w1 = ta_rec_[r + 1];
      const unsigned char* nb;
      int len;
      if ((w1 >> kTagBits) & 1u) {
        size_t jj = j;
        nb = v.data() + z[jj];
        len = (int)(z[jj + 1] - z[jj] - 1);
        ++j;
      } else {
        nb = ta_ip_.data() + ta_rec_[r + 2];
        len = (int)(w1 >> (kTagBits + 1));
      }
      if (len >= (1 << kLenBits)) __builtin_unreachable();
      const unsigned int off = InternName(nb, len);
      if (off >= (1u << kOffBits)) __builtin_unreachable();
      satz_[3 * page_index] |= (unsigned int)len;
      satz_[3 * page_index + 1] = (off << kTagBits) | (w1 & ((1u << kTagBits) - 1u));
    }
    text_.shrink_to_fit();
    std::vector<unsigned int>().swap(hoff_);
    std::vector<unsigned int>().swap(hlen_);
    std::vector<unsigned int>().swap(ta_rec_);
    std::vector<unsigned char>().swap(ta_ip_);
    std::vector<unsigned int>().swap(gr_);
    std::vector<unsigned char>().swap(ta_art_);
    std::vector<unsigned char>().swap(ta_okb_);
    std::vector<unsigned int>().swap(ta_okb_off_);
    ta_namen_offen_ = false;
  }
  __attribute__((noinline)) void GrByte(unsigned char b) {
    if (ta_lf_ >= ta_named_n_) return;   // the first byte of the end mark
    const unsigned char art = ta_art_[ta_lf_];
    if (b == kNL) {
      if (art == 1) {
        gr_[3 * gr_k_ + 2] |= 0x100u;    // a new author enters the grammar
      } else {
        if (!(gr_[3 * gr_k_ + 2] >> 8)) Fail("E543");
      }
      gr_k_ = 0;
      return;
    }
    unsigned int c = gr_[3 * gr_k_];
    while (c && (gr_[3 * c + 2] & 255u) != b) c = gr_[3 * c + 1];
    if (!c) {
      if (art != 1) Fail("E543");
      const unsigned int bruder = gr_[3 * gr_k_];
      c = (unsigned int)(gr_.size() / 3);
      gr_.push_back(0u);
      gr_.push_back(bruder);
      gr_.push_back(b);
      gr_[3 * gr_k_] = c;
    }
    gr_k_ = c;
  }
  std::vector<unsigned int> gr_;
  std::vector<unsigned char> ta_art_;   // kind per article line: 1 new, 2 same, 3 repeated
  unsigned int gr_k_ = 0;               // node of the current line
  std::vector<unsigned char> ta_buf_[3];
  int ta_slot_ = 0;
  bool ta_byte_ = false;
  int ta_zeile_folgt_ = -1;
  bool ta_namen_offen_ = true;        // until the end of slot 2
  unsigned int ta_named_n_ = 0;       // named records (copy marks) in part B
  unsigned int ta_lf_ = 0;            // line feeds of slot 2 so far
  std::vector<unsigned int> ta_rec_;  // deferred part-B records, 3 words each
  std::vector<unsigned char> ta_ip_;  // the address bytes of part B
  std::vector<unsigned char> ta_okb_;      // the injections, article-line order
  std::vector<unsigned int> ta_okb_off_;   // their offsets (+1 end)

  unsigned char zl_[kZeile] = {0};
  int zl_n_ = 0;
  Grenze g_;
  bool tail_closed_ = false, nach_endet_ = false, bzeile_ = false;

  long long id_lauf_ = 0;
  meta_fields::Interp b1_;
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

}  // namespace metablock

#endif
