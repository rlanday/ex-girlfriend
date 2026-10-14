
#ifndef RUN_MODE_H_
#define RUN_MODE_H_

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

// parser states
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
  N_MASK = M_CW2_F0 + 16
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
    fprintf(stderr,
            "\ngrammar violation @%llu mask=%d ctx=%u forced=%d actual=%d (st=%d flag=%d wl=%d)\n",
            pos, mask_, ctx, forced, actual, st_, flag_, wl_);
    fprintf(stderr, "grammar: abort (archive would not decode)\n");
    exit(3);
  }

  void NextByte(unsigned int stream_byte) {
    ++n_bytes_;
    const int r = perm_[stream_byte & 255];
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
  }

  void Reset() {
    st_ = S_START; flag_ = 0; wl_ = 0; b1_ = 0; b2_ = 0;
    n_bytes_ = 0; n_known_ = 0; n_free_ = 0; n_viol_ = 0;
    for (int i = 0; i < 8; ++i) { by_pos_known_[i] = 0; by_pos_free_[i] = 0; }
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
    fseek(f, 0L, SEEK_SET);
    if (own) fclose(own);
    return ndict_ > 0;
  }

  void Set(int m, int rawbyte) { allow_[m][perm_[rawbyte & 255]] = 1; }

  void BuildMasks() {
    BuildPerm();
    memset(allow_, 0, sizeof allow_);
    for (int r = 0; r < 256; ++r) {
      const bool nl = IsNonLet(r), lt = IsLet(r), ld = (r >= 0x80);
      if (nl || r == kEscape || r == kUppercase || r == kCapitalized || lt || ld)
        Set(M_START, r);
      if (lt || ld) Set(M_FLAG, r);
      if (nl || r == kEscape || r == kUppercase || r == kCapitalized || lt || ld)
        Set(M_LET_N, r);
      if (nl || r == kEscape || r == kEndUpper || lt || ld) Set(M_LET_U, r);
      if (nl || r == kEscape || r == kEndUpper || r == kUppercase ||
          r == kCapitalized || lt || ld) Set(M_LET_U_T, r);
      if (nl || r == kEscape || r == kUppercase || r == kCapitalized || lt)
        Set(M_CODE_N, r);
      if (nl || r == kEscape || r == kEndUpper || lt) Set(M_CODE_U, r);
      if (nl || r == kEscape || r == kUppercase || r == kCapitalized || lt || ld)
        Set(M_CODE_N_T, r);
      if (nl || r == kEscape || r == kEndUpper || lt || ld) Set(M_CODE_U_T, r);
      if (r == kEndUpper || r == kUppercase || r == kEscape ||
          r == kCapitalized || ld) Set(M_ESC, r);
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
    switch (st_) {
      case S_START:  mask_ = M_START; break;
      case S_FLAG_U: case S_FLAG_F: mask_ = M_FLAG; break;
      case S_LET_N:  case S_LET_F:  mask_ = M_LET_N; break;
      case S_LET_U:  mask_ = (wl_ >= 59) ? M_LET_U_T : M_LET_U; break;
      case S_CODE_N: case S_CODE_F:
        mask_ = (wl_ >= 59) ? M_CODE_N_T : M_CODE_N; break;
      case S_CODE_U: mask_ = (wl_ >= 59) ? M_CODE_U_T : M_CODE_U; break;
      case S_ESC:    mask_ = M_ESC; break;
      case S_CW2:
        mask_ = (b1_ <= 0xEF) ? M_CW2_LO : (M_CW2_F0 + (b1_ - 0xF0));
        break;
      case S_CW3:
        mask_ = (b1_ == part_b1_ && b2_ == part_b2_) ? M_CW3_LAST : M_CW3;
        break;
      default: mask_ = M_START; break;
    }
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
};

inline Gram gram;

}

#endif
