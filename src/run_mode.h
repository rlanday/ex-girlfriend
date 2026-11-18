
#ifndef RUN_MODE_H_
#define RUN_MODE_H_
#include "meld.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "vocab_frozen.h"

// Grammar automaton over the coded stream.  It follows the word and codeword
// structure of the WRT form and marks, per bit position, bits whose value the
// grammar already fixes; those bits are not coded (the models still see
// them).  Encoder and decoder run the same automaton.
namespace run_mode {

constexpr unsigned long long kSegHeader =
    1 + (2 == 2 ? 5 : 8) + 1;

const unsigned char kEndUpper = 0x06;
const unsigned char kUppercase = 0x07;
const unsigned char kEscape = 0x0C;
const unsigned char kCapitalized = 0x40;

enum {
  S_START = 0, S_FLAG_U = 2, S_FLAG_F = 3,
  S_LET_N = 4, S_LET_U = 5, S_LET_F = 6,
  S_CODE_N = 7, S_CODE_U = 8, S_CODE_F = 9,
  S_ESC = 10, S_CW2 = 11, S_CW3 = 12
};

// mask classes (sets of allowed byte values per state)
enum {
  M_START = 0, M_FLAG = 1, M_LET_N = 2, M_LET_U = 3, M_CODE_N = 4,
  M_CODE_U = 5, M_CODE_N_T = 6, M_CODE_U_T = 7, M_ESC = 8, M_CW2_LO = 9,
  M_CW2_HI = 10, M_CW2_FF = 11, M_CW3 = 12, M_CW3_LAST = 13, M_LET_U_T = 14,
  M_CW2_F0 = 15,
  M_LET_N_NS = M_CW2_F0 + 16, M_LET_U_NS, M_LETONLY, M_CODE_N_SC,
  M_CODE_U_SC, M_UTF_ESC, M_UTF_CONT, M_UTF_FRESH = M_UTF_CONT + 5,
  N_MASK = M_UTF_FRESH + 1
};

static const int kBand1 = 80;
static const int kBand2 = kBand1 + 3840;      // 2-byte codes, lead byte 0xD0..0xFF
static const int kBand3 = kBand2 + 40960;     // 3-byte codes, b1 0xF0..0xFF, b2 0xD0..0xEF
static const int kBand4 = kBand3 + 20480;
static inline int Idx2(int b1, int b2) {
  return kBand1 + (b1 - 0xD0) * 80 + (b2 - 0x80);
}
static inline int Idx3(int b1, int b2, int b3) {
  if (b2 >= 0xF0)
    return kBand3 + (b1 - 0xF0) * 1280 + (b2 - 0xF0) * 80 + (b3 - 0x80);
  return kBand2 + ((b1 - 0xF0) * 32 + (b2 - 0xD0)) * 80 + (b3 - 0x80);
}

class Gram {
 public:
  void Init(FILE* dictionary, unsigned long long stream_bytes,
            unsigned long long hi_hint) {
    active_ = false;
    if (!LoadDict(dictionary)) {
      return;
    }
    BuildMasks();
    BuildForced();
    lo_ = kSegHeader;
    hi_ = hi_hint ? hi_hint : stream_bytes;
    if (hi_ > stream_bytes) hi_ = stream_bytes;
    Reset();
    active_ = true;
  }

  void ZoneSperren() {
    if (!active_) return;
    lo_ = kSperre;
  }

  void ZoneOeffnen(unsigned long long ab) {
    if (lo_ != kSperre) return;
    lo_ = ab;
  }

  bool active() const { return active_; }

  inline bool ByteActive(unsigned long long pos) const {
    return active_ && pos >= lo_ && pos < hi_;
  }

  inline int Forced(unsigned int ctx) const { return forced_[mask_][ctx & 255]; }

  inline void CountFree(unsigned int ctx) {
    ++n_free_;
    ++by_pos_free_[BitIndex(ctx)];
  }
  inline void CountKnown(unsigned int ctx) {
    ++n_known_;
    ++by_pos_known_[BitIndex(ctx)];
  }

  void Violation(unsigned long long pos, unsigned int ctx, int forced,
                 int actual) {
    msg_note(105)

;
    msg_stop(106);
    exit(3);
  }

  void NextByte(unsigned int stream_byte) {
    ++n_bytes_;
    const int r = perm_[stream_byte & 255];
    const int st0 = st_;
    switch (st_) {
      case S_CW2:
        if (r >= 0xD0) { b2_ = r; st_ = S_CW3; }
        else {
          int lc = 80 + (b1_ - 0xD0) * 80 + (r - 0x80);
          wl_ += (lc >= 0 && lc < ndict_) ? wlen_[lc] : 0;
          st_ = (flag_ == 2) ? S_CODE_U : (flag_ == 1 ? S_CODE_F : S_CODE_N);
        }
        break;
      case S_CW3: {
        int lc = Idx3(b1_, b2_, r);
        wl_ += (lc >= 0 && lc < ndict_) ? wlen_[lc] : 0;
        st_ = (flag_ == 2) ? S_CODE_U : (flag_ == 1 ? S_CODE_F : S_CODE_N);
        break;
      }
      case S_ESC:
        st_ = S_START; flag_ = 0; wl_ = 0;
        break;
      default:
        if (r == kEscape) { st_ = S_ESC; }
        else if (r == kUppercase) { flag_ = 2; wl_ = 0; st_ = S_FLAG_U; }
        else if (r == kCapitalized) { flag_ = 1; wl_ = 0; st_ = S_FLAG_F; }
        else if (r == kEndUpper) { flag_ = 0; wl_ = 0; st_ = S_START; }
        else if (r < 0x80 && !IsLet(r)) { flag_ = 0; wl_ = 0; st_ = S_START; }
        else if (IsLet(r)) {
          if (st_ == S_START) { flag_ = 0; wl_ = 0; }
          ++wl_;
          if (st_ != S_CODE_N && st_ != S_CODE_U && st_ != S_CODE_F)
            st_ = (flag_ == 2) ? S_LET_U : (flag_ == 1 ? S_LET_F : S_LET_N);
        } else {                                   // lead byte of a codeword
          if (st_ == S_CODE_N || st_ == S_CODE_U || st_ == S_CODE_F) {
            flag_ = 0; wl_ = 0;                    // truncation (word > 58)
          }
          b1_ = r;
          if (r <= 0xCF) {
            int lc = r - 0x80;
            wl_ += (lc < ndict_) ? wlen_[lc] : 0;
            st_ = (flag_ == 2) ? S_CODE_U : (flag_ == 1 ? S_CODE_F : S_CODE_N);
          } else st_ = S_CW2;
        }
        break;
    }
    SetMask();
    ErwNach(r, st0);
  }

  void Reset() {
    st_ = S_START; flag_ = 0; wl_ = 0; b1_ = 0; b2_ = 0;
    n_bytes_ = 0; n_known_ = 0; n_free_ = 0; n_viol_ = 0;
    for (int i = 0; i < 8; ++i) { by_pos_known_[i] = 0; by_pos_free_[i] = 0; }
    u_pend_ = 0; u_cls_ = 0; u_esc_ = 0; lw_ = 0; lw_n_ = 0;
    SetMask();
  }

  void Report(const char* tag) const {
  }

  int DbgSt() const { return st_; }
  int DbgMask() const { return mask_; }
  int DbgFlag() const { return flag_; }
  int DbgWl() const { return wl_; }

  unsigned long long known_bits() const { return n_known_; }
  unsigned long long total_bits() const { return n_known_ + n_free_; }
  unsigned long long violations() const { return n_viol_; }

 private:
  static inline int BitIndex(unsigned int ctx) {
    int k = 0;
    while ((ctx >> (k + 1)) != 0) ++k;
    return k;
  }
  static inline bool IsLet(int r) { return r >= 'a' && r <= 'z'; }
  static inline bool IsUpLet(int r) { return r >= 'A' && r <= 'Z'; }
  static inline bool IsNonLet(int r) {
    return r < 0x80 && !IsLet(r) && !IsUpLet(r) && r != kEndUpper &&
           r != kUppercase && r != kEscape && r != kCapitalized;
  }

  void BuildPerm() {
    for (int i = 0; i < 256; ++i) {
      int c = i;
      if (c >= '{' && c < 127) c += 'P' - '{';
      else if (c >= 'P' && c < 'T') c -= 'P' - '{';
      else if ((c >= ':' && c <= '?') || (c >= 'J' && c <= 'O')) c ^= 0x70;
      if (c == 'X' || c == '`') c ^= 'X' ^ '`';
      perm_[i] = (unsigned char)c;
    }
  }

  // Word length per dictionary line, parsed as in Dictionary::Dictionary().
  bool LoadDict(FILE* dictionary) {
    FILE* f = dictionary;
    FILE* own = NULL;
    if (!f) {
      own = fopen("./.dict", "rb");
      f = own;
    }
    if (!f) return false;
    fseek(f, 0L, SEEK_END);
    long n = ftell(f);
    fseek(f, 0L, SEEK_SET);
    if (n <= 0) { if (own) fclose(own); return false; }
    ndict_ = 0;
    int run = 0;
    for (long i = 0; i < n; ++i) {
      int c = getc(f);
      if (c >= 'a' && c <= 'z') { if (run < 255) ++run; }
      else if (run) {
        if (ndict_ < kMaxDict) wlen_[ndict_++] = (unsigned char)run;
        run = 0;
      }
    }
    if (run && ndict_ < kMaxDict) wlen_[ndict_++] = (unsigned char)run;
    ErwDict(f);
    fseek(f, 0L, SEEK_SET);
    if (own) fclose(own);
    return ndict_ > 0;
  }

  void Set(int m, int rawbyte) { allow_[m][perm_[rawbyte & 255]] = 1; }

  void BuildMasks() {
    BuildPerm();
    memset(allow_, 0, sizeof allow_);
    // the classes of a raw byte (disjoint; an upper-case letter is in none):
    // non-letter, letter, code byte (>= 0x80) and the four flag bytes; K_U8
    // (a UTF-8 lead byte 0xC2..0xF4) lies inside K_LD
    enum { K_NL = 1, K_LT = 2, K_LD = 4, K_ES = 8, K_UP = 16, K_CA = 32, K_EU = 64,
           K_U8 = 128 };
    static const unsigned char kErlaubt[17][2] = {
      {M_START, K_NL | K_ES | K_UP | K_CA | K_LT | K_LD},
      {M_FLAG, K_LT | K_LD},
      {M_LET_N, K_NL | K_ES | K_UP | K_CA | K_LT | K_LD},
      {M_LET_U, K_NL | K_ES | K_EU | K_LT | K_LD},
      {M_LET_U_T, K_NL | K_ES | K_EU | K_UP | K_CA | K_LT | K_LD},
      {M_CODE_N, K_NL | K_ES | K_UP | K_CA | K_LT},
      {M_CODE_U, K_NL | K_ES | K_EU | K_LT},
      {M_CODE_N_T, K_NL | K_ES | K_UP | K_CA | K_LT | K_LD},
      {M_CODE_U_T, K_NL | K_ES | K_EU | K_LT | K_LD},
      {M_ESC, K_EU | K_UP | K_ES | K_CA | K_LD},
      {M_LET_N_NS, K_NL | K_ES | K_UP | K_CA | K_LT},   // M_LET_N without code bytes
      {M_LET_U_NS, K_NL | K_ES | K_EU | K_LT},          // M_LET_U without code bytes
      {M_LETONLY, K_LT},
      {M_CODE_N_SC, K_NL | K_ES | K_UP | K_CA},         // M_CODE_N without letters
      {M_CODE_U_SC, K_NL | K_ES | K_EU},                // M_CODE_U without letters
      {M_UTF_ESC, K_ES},
      {M_UTF_FRESH, K_EU | K_UP | K_ES | K_CA | K_U8}};
    for (int r = 0; r < 256; ++r) {
      const int k = (IsNonLet(r) ? K_NL : 0) | (IsLet(r) ? K_LT : 0) |
                    (r >= 0x80 ? K_LD : 0) | (r == kEscape ? K_ES : 0) |
                    (r == kUppercase ? K_UP : 0) | (r == kCapitalized ? K_CA : 0) |
                    (r == kEndUpper ? K_EU : 0) | (r >= 0xC2 && r <= 0xF4 ? K_U8 : 0);
      for (const auto& e : kErlaubt)
        if (e[1] & k) Set(e[0], r);
    }
    for (int b2 = 0x80; b2 <= 0xCF; ++b2) { Set(M_CW2_LO, b2); Set(M_CW3, b2); }
    for (int b1 = 0xF0; b1 <= 0xFF; ++b1) {
      const int m = M_CW2_F0 + (b1 - 0xF0);
      // (a) second byte < 0xD0: 2-byte codeword (band 2, lead byte up to 0xFF).
      for (int b2 = 0x80; b2 <= 0xCF; ++b2)
        if (Idx2(b1, b2) < ndict_) Set(m, b2);
      for (int b2 = 0xD0; b2 <= 0xFF; ++b2)
        if (Idx3(b1, b2, 0x80) < ndict_) Set(m, b2);
    }
    part_b1_ = -1; part_b2_ = -1;
    if (ndict_ > kBand2) {
      const int last = ndict_ - 1;
      int lb1 = -1, lb2 = -1;
      if (last < kBand3) {
        const int g = (last - kBand2) / 80;
        lb1 = 0xF0 + g / 32; lb2 = 0xD0 + g % 32;
      } else if (last < kBand4) {
        const int j = last - kBand3;
        lb1 = 0xF0 + j / 1280; lb2 = 0xF0 + (j / 80) % 16;
      }
      if (lb1 >= 0) {
        bool teil = false;   // last codeword group only partly used
        for (int b3 = 0x80; b3 <= 0xCF; ++b3) {
          if (Idx3(lb1, lb2, b3) < ndict_) Set(M_CW3_LAST, b3);
          else teil = true;
        }
        if (teil) { part_b1_ = lb1; part_b2_ = lb2; }
      }
    }
    ErwMasks();
  }

  void BuildForced() {
    memset(forced_, -1, sizeof forced_);
    for (int m = 0; m < N_MASK; ++m) {
      for (int ctx = 1; ctx < 256; ++ctx) {
        const int k = BitIndex((unsigned int)ctx);  // number of bits already known
        const int j = 7 - k;                        // index of the open bit
        const int pref = ctx - (1 << k);
        int n0 = 0, n1 = 0;
        for (int v = 0; v < 256; ++v) {
          if (!allow_[m][v]) continue;
          if ((v >> (j + 1)) != pref) continue;
          if ((v >> j) & 1) ++n1; else ++n0;
        }
        if (n0 && !n1) forced_[m][ctx] = 0;
        else if (n1 && !n0) forced_[m][ctx] = 1;
        else forced_[m][ctx] = -1;
      }
    }
  }

  void SetMask() {
    // the mask of a state below S_CW2 (st_ only takes the parser states);
    // row 1 is a word of >= 59 letters (truncation)
    static const unsigned char kMaske[2][S_CW2] = {
      {M_START, M_START, M_FLAG, M_FLAG, M_LET_N, M_LET_U, M_LET_N,
       M_CODE_N, M_CODE_U, M_CODE_N, M_ESC},
      {M_START, M_START, M_FLAG, M_FLAG, M_LET_N, M_LET_U_T, M_LET_N,
       M_CODE_N_T, M_CODE_U_T, M_CODE_N_T, M_ESC}};
    if (st_ == S_CW2)
      mask_ = (b1_ <= 0xEF) ? M_CW2_LO : (M_CW2_F0 + (b1_ - 0xF0));
    else if (st_ == S_CW3)
      mask_ = (b1_ == part_b1_ && b2_ == part_b2_) ? M_CW3_LAST : M_CW3;
    else
      mask_ = kMaske[wl_ >= 59][st_];
  }

  __attribute__((noinline, cold)) void ErwMasks() {
    // the continuation byte behind an escaped lead byte: 80..BF; as the first
    // one behind E0 A0..BF, ED 80..9F, F0 90..BF, F4 80..8F (RFC 3629)
    static const unsigned char lo[5] = {0x80, 0xA0, 0x80, 0x90, 0x80};
    static const unsigned char hi[5] = {0xBF, 0xBF, 0x9F, 0xBF, 0x8F};
    for (int k = 0; k < 5; ++k)
      for (int r = lo[k]; r <= hi[k]; ++r) Set(M_UTF_CONT + k, r);
    if (!vocab_frozen::g_aux)
    for (int m = 0; m < N_MASK; ++m)
      for (int b = 0; b < 256; ++b)
        if (!vocab_frozen::InSet(b)) allow_[m][b] = 0;
  }

  // the dictionary words with < 7 letters as keys (5 bits per letter,
  // letter + 1), an open hash set; only lines with a code word (< kBand4)
  __attribute__((noinline, cold)) void ErwDict(FILE* f) {
    memset(kurz_, 0, sizeof kurz_);
    n_kurz_ = 0;
    fseek(f, 0L, SEEK_SET);
    unsigned int key = 0;
    int run = 0, line = 0, c;
    while ((c = getc(f)) != EOF) {
      if (c >= 'a' && c <= 'z') {
        if (run < 6) key = key * 32 + (unsigned int)(c - 'a' + 1);
        ++run;
      } else if (run) {
        if (run < 7 && line < kBand4) KurzRein(key);
        ++line; key = 0; run = 0;
      }
    }
    if (run && run < 7 && line < kBand4) KurzRein(key);
  }
  static unsigned int KurzHash(unsigned int key) {
    return (key * 2654435761u) >> (32 - kKurzLog);
  }
  void KurzRein(unsigned int key) {
    if (n_kurz_ >= kKurzN / 2) return;   // never full (a lookup ends at a 0)
    unsigned int h = KurzHash(key);
    while (kurz_[h] && kurz_[h] != key) h = (h + 1) & (kKurzN - 1);
    if (!kurz_[h]) { kurz_[h] = key; ++n_kurz_; }
  }
  bool Kurz(unsigned int key) const {
    for (unsigned int h = KurzHash(key); kurz_[h]; h = (h + 1) & (kKurzN - 1))
      if (kurz_[h] == key) return true;
    return false;
  }

  // Behind SetMask(): the state of the further kinds (from the byte just
  // seen and the state before it, the same on both sides) and the narrower
  // mask where one applies.  Called for every byte of the zone.
  __attribute__((noinline)) void ErwNach(int r, int st0) {
    const bool let0 = (st0 == S_LET_N || st0 == S_LET_U || st0 == S_LET_F);
    const bool let1 = (st_ == S_LET_N || st_ == S_LET_U || st_ == S_LET_F);
    // a word in letters: its letters so far (up to 6; 7 = longer)
    if (!let1) lw_n_ = 0;
    else if (!let0) { lw_ = (unsigned int)(r - 'a' + 1); lw_n_ = 1; }
    else if (lw_n_ && lw_n_ < 7) {
      if (lw_n_ < 6) lw_ = lw_ * 32 + (unsigned int)(r - 'a' + 1);
      ++lw_n_;
    }
    // UTF-8: an escaped lead byte opens 1..3 continuation bytes, each one
    // behind its own escape
    if (u_esc_) u_esc_ = 0;
    else if (st0 == S_ESC) {
      if (u_pend_) { --u_pend_; u_cls_ = 0; u_esc_ = u_pend_ ? 1 : 0; }
      else if (r >= 0xC2 && r <= 0xF4) {
        u_pend_ = r <= 0xDF ? 1 : (r <= 0xEF ? 2 : 3);
        u_cls_ = r == 0xE0 ? 1 : (r == 0xED ? 2 : (r == 0xF0 ? 3 : (r == 0xF4 ? 4 : 0)));
        u_esc_ = 1;
      }
    }
    if (u_esc_) { mask_ = M_UTF_ESC; return; }
    if (st_ == S_ESC) { mask_ = u_pend_ ? M_UTF_CONT + u_cls_ : M_UTF_FRESH; return; }
    if (r == kEndUpper && st0 != S_ESC) { mask_ = M_FLAG; return; }
    if (st_ == S_LET_U && wl_ == 1) { mask_ = M_LETONLY; return; }
    if (lw_n_ && lw_n_ < 7 && Kurz(lw_)) { mask_ = M_LETONLY; return; }
    if ((st_ == S_LET_N || st_ == S_LET_F) && wl_ < 59) { mask_ = M_LET_N_NS; return; }
    if (st_ == S_LET_U && wl_ < 59) { mask_ = M_LET_U_NS; return; }
    if ((st_ == S_CODE_N || st_ == S_CODE_F) && wl_ < 7) { mask_ = M_CODE_N_SC; return; }
    if (st_ == S_CODE_U && wl_ < 7) { mask_ = M_CODE_U_SC; return; }
  }

  static const int kMaxDict = 1 << 17;
  static const unsigned long long kSperre = ~0ull;

  bool active_ = false;
  unsigned long long lo_ = 6, hi_ = 0;
  int st_ = S_START, flag_ = 0, wl_ = 0, b1_ = 0, b2_ = 0, mask_ = M_START;
  int part_b1_ = -1, part_b2_ = -1;
  int ndict_ = 0;
  unsigned long long n_bytes_ = 0, n_known_ = 0, n_free_ = 0, n_viol_ = 0;
  unsigned long long by_pos_known_[8] = {0}, by_pos_free_[8] = {0};
  unsigned char perm_[256];
  unsigned char allow_[N_MASK][256];
  signed char forced_[N_MASK][256];
  unsigned char wlen_[kMaxDict];
  int u_pend_ = 0, u_cls_ = 0, u_esc_ = 0;   // UTF-8 of escaped bytes
  unsigned int lw_ = 0;                      // letters of a word in letters
  int lw_n_ = 0;
  static const int kKurzLog = 16, kKurzN = 1 << kKurzLog;
  int n_kurz_ = 0;
  unsigned int kurz_[kKurzN];                // short dictionary words
};

inline Gram gram;

}  // namespace run_mode

#endif
