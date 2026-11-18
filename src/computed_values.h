// Computed values for the transformer prompt.
// The stream stays as it is.  When the model side has decoded a source that
// determines a value further on, the value is computed and fed to the
// transformer behind its trigger, after a copy mark:
//   * unit pairs "<X> <unit> (", " or ", " /", " =", " ||": Y = X in the
//     default target unit of the source unit, formatted like X (rounding rule
//     per unit pair); behind "(" followed by the ending (target unit and ")")
//     last written in that target unit behind the same source spelling;
//   * adjacent infobox fields "<stem>_<unit1> = X | <stem>_<unit2> =" (also
//     "hectares = X | km2 ="): Y behind the second "=";
//   * the life years of the lead (first parenthesis behind
//     the first bold) behind the first "[[Category:" of the page, the age
//     behind "aged " / "at age " / "the age of ", the year behind the persondata
//     fields "DATE OF BIRTH=" / "DATE OF DEATH=".
// Census pages are left out (the census block treats them).
//
// Consumer: the transformer only, as a prompt block behind the trigger token
// (tf1_prompt.cpp): the sentinel row of the copy mark (class 7), then the
// bytes of the value as tokens, closed by the end mark of a copy block; fed
// in, never coded.  The automaton reads the coded stream tokens of the page
// bodies (fed by the prompt automaton, which sees every coded token once, on
// both sides), so encoder and decoder build the same injection behind the
// same token.  All arithmetic is integer (exact fractions, 64-bit long
// division).
#ifndef COMPUTED_VALUES_H_
#define COMPUTED_VALUES_H_

#include <stdint.h>
#include <stdio.h>
#include "meld.h"
#include "invariants.h"    // RingEndet, shared with the census numbers
#include <stdlib.h>
#include <string.h>

#ifdef __clang__
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#endif
namespace computed_values {

constexpr int kMarke = 7;

// The word list of the WRT stage (runs of a-z, as Dictionary::Dictionary()).
inline const char* g_dict = ".dict";

inline bool g_dict_da = false;

static const int kEinheiten = 41;
// factor to the base of the dimension: kFm * 10^kFe (speed base km/h; temperature in the code);
// mantissas rounded to 9 significant digits (hp, lbf, lbft, psi, oz, acre, mi2: relative error < 1e-8)
static const uint32_t kFm[41] = {1u, 1u, 1u, 1u, 404685642u, 1u, 1u, 3048u, 9290304u, 109728u, 1u, 1u, 745699872u, 254u, 16387064u, 1u, 1u, 1u, 1u, 1u, 1852u, 1u, 1u, 1u, 45359237u, 444822162u, 135581795u, 1u, 1u, 1609344u, 258998811u, 1u, 1609344u, 36u, 1852u, 283495231u, 73549875u, 689475729u, 1u, 90718474u, 9144u};
static const signed char kFe[41] = {0, 0, 0, 0, -5, -2, -6, -4, -8, -5, -3, 4, -6, -4, -12, 3, 0, 3, 6, 0, -3, 3, 3, -3, -8, -8, -8, 0, 0, -3, -2, -3, -6, -1, 0, -10, -5, -5, 3, -5, -4};
static const unsigned char kDim[41] = {8, 8, 8, 0, 2, 5, 9, 5, 2, 3, 7, 2, 6, 5, 9, 4, 7, 5, 2, 3, 3, 1, 6, 9, 7, 4, 0, 5, 2, 5, 2, 5, 3, 3, 5, 7, 6, 1, 7, 7, 5};
static const signed char kZiel[41] = {1, 0, 0, 26, 18, 13, 14, 27, 28, 33, 35, 18, 22, 31, 23, 25, 24, 29, 30, 32, 19, 37, 12, 6, 16, 15, 3, 7, 8, 17, 18, 13, 19, 9, 17, 10, 22, 21, 39, 38, 27};
// rounding rule per pair where it is not `dec` (0 dec 1 sig 2 sig2 3 sig3 4 d0 5 d1)
static const unsigned char kRegel[38][3] = {{27, 7, 4}, {13, 31, 4}, {16, 24, 2}, {13, 27, 3}, {32, 19, 2}, {13, 5, 4}, {5, 13, 4}, {20, 19, 4}, {34, 17, 2}, {1, 0, 4}, {4, 28, 2}, {30, 18, 2}, {8, 28, 2}, {11, 18, 1}, {7, 17, 1}, {14, 23, 5}, {22, 12, 2}, {17, 34, 2}, {15, 25, 3}, {39, 38, 2}, {27, 40, 2}, {28, 8, 2}, {27, 17, 1}, {27, 13, 4}, {24, 38, 2}, {33, 9, 3}, {18, 4, 2}, {38, 39, 2}, {17, 7, 3}, {29, 27, 2}, {35, 10, 2}, {7, 5, 1}, {23, 6, 1}, {40, 17, 2}, {11, 4, 2}, {27, 31, 1}, {30, 4, 4}, {27, 29, 5}};
// spellings: [length][unit id][text], normalised as Normal() writes them
static const unsigned char kSchreib[] =
    "\x01\x1b\x6d\x05\x1d\x6d\x69\x6c\x65\x73\x02\x07\x66\x74\x02\x11\x6b\x6d\x04\x07\x66\x65\x65\x74\x02"
    "\x0d\x69\x6e\x05\x04\x61\x63\x72\x65\x73\x04\x00\x64\x65\x67\x63\x02\x18\x6c\x62\x02\x0c\x68\x70\x02"
    "\x10\x6b\x67\x03\x20\x6d\x70\x68\x06\x0d\x69\x6e\x63\x68\x65\x73\x04\x0d\x69\x6e\x63\x68\x02\x1f\x6d"
    "\x6d\x06\x1b\x6d\x65\x74\x65\x72\x73\x02\x05\x63\x6d\x04\x13\x6b\x6d\x2f\x68\x03\x12\x6b\x6d\x32\x05"
    "\x14\x6b\x6e\x6f\x74\x73\x06\x1b\x6d\x65\x74\x72\x65\x73\x06\x18\x70\x6f\x75\x6e\x64\x73\x0e\x22\x6e"
    "\x61\x75\x74\x69\x63\x61\x6c\x20\x6d\x69\x6c\x65\x73\x04\x01\x64\x65\x67\x66\x04\x1d\x6d\x69\x6c\x65"
    "\x01\x0d\x22\x04\x04\x61\x63\x72\x65\x02\x1d\x6d\x69\x04\x07\x66\x6f\x6f\x74\x0a\x11\x6b\x69\x6c\x6f"
    "\x6d\x65\x74\x65\x72\x73\x0a\x11\x6b\x69\x6c\x6f\x6d\x65\x74\x72\x65\x73\x0c\x1e\x73\x71\x75\x61\x72"
    "\x65\x20\x6d\x69\x6c\x65\x73\x03\x19\x6c\x62\x66\x01\x02\x6b\x0a\x0c\x68\x6f\x72\x73\x65\x70\x6f\x77"
    "\x65\x72\x05\x28\x79\x61\x72\x64\x73\x02\x16\x6b\x77\x04\x09\x66\x74\x2f\x73\x03\x08\x66\x74\x32\x02"
    "\x22\x6e\x6d\x07\x1a\x66\x74\xc2\xb7\x6c\x62\x66\x05\x18\x70\x6f\x75\x6e\x64\x02\x24\x70\x73\x07\x18"
    "\x70\x6f\x75\x6e\x64\x65\x72\x03\x21\x6d\x2f\x73\x02\x0f\x6b\x6e\x02\x1c\x6d\x32\x04\x27\x74\x6f\x6e"
    "\x73\x03\x0e\x69\x6e\x33\x02\x28\x79\x64\x04\x14\x6b\x6e\x6f\x74\x01\x17\x6c\x05\x1b\x6d\x65\x74\x72"
    "\x65\x05\x1b\x6d\x65\x74\x65\x72\x03\x0c\x73\x68\x70\x03\x0d\x69\x6e\x2e\x03\x18\x6c\x62\x73\x01\x0a"
    "\x67\x03\x07\x66\x74\x2e\x03\x1e\x6d\x69\x32\x01\x00\x63\x03\x0c\x62\x68\x70\x01\x07\x27\x03\x15\x6b"
    "\x70\x61\x02\x0b\x68\x61\x02\x23\x6f\x7a";
static const int kSchreibN = 66;
// end of the generated tables
enum { kTempDim = 8, kC = 0, kF = 1, kK = 2, kFt = 7, kM = 27, kIn = 13, kLb = 24, kOz = 35, kMi = 29, kYd = 40,
       kCm = 5 };
// compound sources "<a> <big> <b> <small>": big unit, small unit, factor
static const unsigned char kKomp[4][2] = {{kFt, kIn}, {kLb, kOz}, {kMi, kYd}, {kM, kCm}};
static const unsigned short kKompF[4] = {12, 16, 1760, 100};

// Numerator and denominator of a value stay below kGrenze, so a remainder
// times 10 (long division) stays below 2^64.
static const uint64_t kGrenze = 1800000000000000000ull;

// ------------------------------------------------------------ numbers ----
struct Zahl {             // a number of the decoded text
  uint64_t m;             // its digits without separators
  uint32_t a, e;          // stream positions: sign or first digit, behind the last digit
  uint8_t nd, dx, sig;    // digits, decimals, significant digits as written
  uint8_t tk, dk, neg;    // thousands commas, decimal comma, minus sign
};

// Reads the written number at z[0, n) (digits, thousands commas, a decimal
// point or comma).  false: no plain number.
__attribute__((noinline, minsize)) inline bool Lesen(const unsigned char* z, int n, Zahl* r) {
  int teil[12], nt = 0, len = 0, punkte = 0, kommas = 0;
  char sep[12];
  for (int i = 0; i < n; ++i) {
    if (z[i] == ',' || z[i] == '.') {
      if (nt >= 11) return false;
      sep[nt] = (char)z[i];
      teil[nt++] = len;
      len = 0;
      if (z[i] == '.') ++punkte; else ++kommas;
    } else {
      ++len;
    }
  }
  teil[nt] = len;
  if (punkte > 1 || (punkte && sep[nt - 1] != '.')) return false;
  r->tk = 0;
  r->dk = 0;
  int fp = 0;
  if (punkte) {
    fp = teil[nt];
    if (kommas) {
      r->tk = 1;
      if (teil[0] > 3) return false;
      for (int i = 1; i < nt; ++i) if (teil[i] != 3) return false;
    }
  } else if (kommas == 1 && teil[1] != 3) {
    r->dk = 1;
    fp = teil[1];
  } else if (kommas) {
    r->tk = 1;
    if (teil[0] > 3) return false;
    for (int i = 1; i <= nt; ++i) if (teil[i] != 3) return false;
  }
  uint64_t m = 0;
  int nd = 0, lead = 1, sig = 0, nz = 0;
  for (int i = 0; i < n; ++i) {
    if (z[i] == ',' || z[i] == '.') continue;
    if (++nd > 15) return false;
    m = m * 10 + (uint64_t)(z[i] - '0');
    if (lead && z[i] == '0') continue;
    lead = 0;
    ++sig;
    nz = z[i] == '0' ? nz + 1 : 0;
  }
  if (!fp) sig -= nz;   // trailing zeros of an integer do not count
  r->m = m;
  r->nd = (uint8_t)nd;
  r->dx = (uint8_t)fp;
  r->sig = (uint8_t)(sig < 1 ? 1 : sig);
  return true;
}

// Y = (yneg ? -1 : 1) * N / D for X = m / 10^dx in unit u1, target unit u2.
__attribute__((noinline)) inline bool ValueOf(uint64_t m, int dx, int neg, int u1, int u2, uint64_t* N, uint64_t* D,
                                           int* yneg) {
  uint64_t n, d;
  *yneg = neg;
  if (kDim[u1] == kTempDim) {
    // temperature, (a*X + b) / c exactly: C->F, F->C, C->K, K->C
    static const short kT[4][5] = {{kC, kF, 9, 160, 5}, {kF, kC, 5, -160, 9}, {kC, kK, 20, 5463, 20},
                                   {kK, kC, 20, -5463, 20}};
    int i = 0;
    while (i < 4 && !(kT[i][0] == u1 && kT[i][1] == u2)) ++i;
    if (i == 4 || m >= kGrenze / 20) return false;
    int64_t b = kT[i][3];
    d = (uint64_t)kT[i][4];
    for (int k = 0; k < dx; ++k) { b *= 10; d *= 10; }
    const int64_t v = (neg ? -(int64_t)m : (int64_t)m) * kT[i][2] + b;
    *yneg = v < 0;
    n = (uint64_t)(v < 0 ? -v : v);
  } else {
    if (kDim[u2] != kDim[u1] || u2 == u1 || __builtin_mul_overflow(m, (uint64_t)kFm[u1], &n)) return false;
    d = kFm[u2];
    for (int k = kFe[u1] - kFe[u2] - dx; k; k += k > 0 ? -1 : 1) {
      if (n >= kGrenze || d >= kGrenze) return false;
      if (k > 0) n *= 10; else d *= 10;
    }
  }
  if (n >= kGrenze || d >= kGrenze) return false;
  *N = n;
  *D = d;
  return true;
}

// |Y| = N / D rounded half up to dd decimals (dd < 0: to 10^-dd), by long
// division; written with thousands commas (tk) and a decimal comma (dk) like
// X.  Returns the length, 0 if it does not fit.
__attribute__((noinline)) inline int Schreiben(uint64_t N, uint64_t D, int dd, int tk, int dk, unsigned char* o) {
  uint64_t v = N / D, r = N % D;
  int up;
  if (dd >= 0) {
    for (int k = 0; k < dd; ++k) {
      if (v >= kGrenze / 10) return 0;
      r *= 10;
      v = v * 10 + r / D;
      r %= D;
    }
    up = 2 * r >= D;
  } else {
    uint64_t q = 1;
    for (int k = 0; k < -dd; ++k) {
      if (q >= kGrenze / 10) return 0;
      q *= 10;
    }
    const uint64_t rest = v % q;
    v /= q;
    up = 2 * rest >= q || (2 * rest + 1 == q && 2 * r >= D);
  }
  v += (uint64_t)up;
  unsigned char g[24];
  int n = 0;
  if (dd < 0 && v) for (int k = 0; k < -dd && n < 20; ++k) g[n++] = '0';
  do { g[n++] = (unsigned char)('0' + (int)(v % 10)); v /= 10; } while (v && n < 20);
  if (v) return 0;
  while (dd > 0 && n < dd + 1) g[n++] = '0';
  int j = 0;
  const int fr = dd > 0 ? dd : 0;
  for (int i = n - 1; i >= 0; --i) {
    if (i == fr - 1) o[j++] = dk ? ',' : '.';
    o[j++] = g[i];
    if (tk && i > fr && ((i - fr) % 3) == 0) o[j++] = ',';
  }
  return j;
}

// floor(log10(N / D)) for N > 0
inline int Exponent(uint64_t N, uint64_t D) {
  uint64_t q = N / D;
  int e = 0;
  if (q) {
    while (q >= 10) { q /= 10; ++e; }
    return e;
  }
  while (N < D && e > -30) { N *= 10; --e; }
  return e;
}

// ------------------------------------------------------------ automaton ----
struct Aut {
  // word list of the WRT stage
  unsigned char* wt_;   // the words, concatenated
  uint32_t* wo_;        // offset of word i, wo_[n] = end
  uint32_t wn_;
  // stream bytes and the WRT decoder
  unsigned char sr_[256];         // the last stream bytes (position & 255)
  uint32_t sp_;               // stream bytes read
  uint8_t esc_, cwn_;
  uint8_t cw_[2];
  uint32_t cwa_, esca_;   // stream position of the code word / escape
  // decoded text (lower case): the last 64 characters and the stream
  // position of the byte (code word, escape) each comes from
  unsigned char t_[64];
  uint32_t tp_[64];
  uint32_t tn_;
  long long seite_;
  int frei_;
  // numbers: the one being read, the last one and the text behind it
  unsigned char nz_[24];
  int nzn_, innum_;
  uint32_t nsa_, nze_, nti_;
  int nneg_, nfeld_;
  Zahl cur_;
  int cur_ok_;
  Zahl lo_;                       // low end of a range "lo <sep> cur"
  int range_;
  uint32_t rsa_, rse_;    // stream bytes of the range separator
  uint64_t komp_m_;           // compound "<a> <big> <b> <small>": a and big
  int komp_u_, komp_nd_;
  unsigned char ub_[48];          // text behind the last number
  int ubn_;
  // parenthesis form: the spelling waiting for its written value and ending
  int kw_, kwu_, kwy_;
  uint32_t kwt_;
  struct Lern { uint8_t n; unsigned char b[16]; };   // ending of the default target
  Lern lern_[kSchreibN];
  // infobox fields: the line part since "|", the last field with a value
  unsigned char zl_[40];
  int zln_;
  uint32_t fnr_;
  struct Feld { uint32_t stamm, nr; int u; Zahl x; int ok; };
  Feld fw_;   // field waiting for its value
  Feld fv_;   // last field with a value
  // year reader: lead state (0 bold?, 1 parenthesis?, 2 inside, 3 done),
  // years of the parenthesis, life years
  int ls_, ltief_, lstrich_, lgeb_, ljn_;
  uint32_t lti_;
  int lj_[2];
  int jb_, jd_, kat_;
  int jlast_;
  uint32_t jlast_ti_, absatz_ti_, died_ti_;
  // output (at most kMax bytes, longer values are not fed)
  enum { kMax = 40 };
  unsigned char ob_[96];
  int obn_;

  __attribute__((noinline, minsize)) void Laden() {
    FILE* f = fopen(g_dict, "rb");
    if (!f) { msg_stop(552); exit(1); }
    fseek(f, 0L, SEEK_END);
    const long L = ftell(f);
    fseek(f, 0L, SEEK_SET);
    unsigned char* b = (unsigned char*)malloc((size_t)L + 1);
    wt_ = (unsigned char*)malloc((size_t)L + 1);
    wo_ = (uint32_t*)malloc(sizeof(uint32_t) * ((size_t)L / 2 + 2));
    if (!b || !wt_ || !wo_ || fread(b, 1, (size_t)L, f) != (size_t)L) exit(1);
    fclose(f);
    // runs of a-z: wo_[i]..wo_[i+1] is word i (a word ends at a non-letter, as
    // Dictionary::Dictionary() reads the file)
    uint32_t k = 0, w = 0;
    wo_[0] = 0;
    for (long i = 0; i < L; ++i) {
      if (b[i] >= 'a' && b[i] <= 'z') wt_[k++] = b[i];
      else if (k > wo_[w]) wo_[++w] = k;
    }
    wn_ = w;
    free(b);
    seite_ = -1;
    komp_u_ = -1;
    kw_ = -1;
  }

  inline unsigned char T(int k) const { return t_[(tn_ - (uint32_t)k) & 63]; }
  __attribute__((noinline)) bool Endet(const char* s, int n) const {
    return invariants::RingEndet(t_, 63, tn_, s, n);
  }
  // the list l of texts ([length][text]...), k of them: the first one the
  // decoded text ends with, -1 if none
  __attribute__((noinline)) int EndetAuf(const char* l, int k) const {
    for (int i = 0; i < k; ++i, l += 1 + l[0]) if (Endet(l + 1, l[0])) return i;
    return -1;
  }
  // the list l of texts: the index of the one equal to u[0..n), -1 if none
  __attribute__((noinline)) static int Gleich(const char* l, int k, const unsigned char* u, int n) {
    for (int i = 0; i < k; ++i, l += 1 + l[0]) if (l[0] == n && !memcmp(l + 1, u, (size_t)n)) return i;
    return -1;
  }

  // normalised unit text (lower case, links, entities, degree and
  // superscripts folded, blanks collapsed) -> spelling index, -1 if none
  __attribute__((noinline, minsize)) static int Schreibung(const unsigned char* s, int n) {
    unsigned char o[48];
    int j = 0, lk = -1, sp = 1;
    for (int i = 0; i < n && j < 44; ++i) {
      unsigned c = s[i];
      if (c == '[') { lk = j; continue; }
      if (c == ']') continue;
      if (c == '|') { if (lk >= 0) { j = lk; lk = -1; } else return -1; continue; }
      if (c == '&' && i + 2 < n && s[i + 1] == '&') {
        i += 2;
        c = s[i];
        if (c == 0x0B) c = '2';            // &sup2;
        else if (c == '!') c = ' ';        // &nbsp;
      }
      if (c == 0xC2 && i + 1 < n) {
        const unsigned d = s[i + 1];
        if (d == 0xB0 || d == 0xBA) { ++i; if (j + 3 <= 44) { o[j++] = 'd'; o[j++] = 'e'; o[j++] = 'g'; sp = 0; } continue; }
        if (d == 0xB2 || d == 0xB3) { ++i; c = d == 0xB2 ? '2' : '3'; }
        else if (d == 0xA0) { ++i; c = ' '; }
      }
      if (c == ' ' || c == '\t') { if (!sp) o[j++] = ' '; sp = 1; continue; }
      if (j == 0 && c == '-') continue;
      o[j++] = (unsigned char)c;
      sp = 0;
    }
    if (j && o[j - 1] == ' ') --j;
    // "deg c" -> "degc"
    if (j == 5 && o[0] == 'd' && o[1] == 'e' && o[2] == 'g' && o[3] == ' ') { o[3] = o[4]; j = 4; }
    const unsigned char* p = kSchreib;
    for (int i = 0; i < kSchreibN; ++i) {
      if (p[0] == j && !memcmp(p + 2, o, (size_t)j)) return i;
      p += 2 + p[0];
    }
    return -1;
  }
  static int Einheit(int s) {
    const unsigned char* p = kSchreib;
    while (s-- > 0) p += 2 + p[0];
    return p[1];
  }

  // ---- one stream byte -------------------------------------------------
  // `seite` changes at every page start; `frei`: an injection may stand behind
  // this token.  Returns the length of the bytes to feed (0: none).
  int Byte(unsigned char b, long long page_index, int frei, const unsigned char** p) {
    if (!wt_ && g_dict_da) Laden();
    if (page_index != seite_) NeueSeite(page_index);
    const uint32_t pos = sp_++;
    sr_[pos & 255] = b;
    obn_ = 0;
    frei_ = frei;
    // the byte permutation of the WRT stage (fxcmv1 charSwap)
    unsigned w = b;
    if (w >= '{' && w < 127) w += 'P' - '{';
    else if (w >= 'P' && w < 'T') w -= 'P' - '{';
    else if ((w >= ':' && w <= '?') || (w >= 'J' && w <= 'O')) w ^= 0x70;
    if (w == 'X' || w == '`') w ^= 'X' ^ '`';
    if (esc_) {
      esc_ = 0;
      Zeichen(w, esca_);
    } else if (w == 0x0C) {
      esc_ = 1;
      esca_ = pos;
      cwn_ = 0;
    } else if (w >= 0x80) {
      if (!cwn_) cwa_ = pos;
      if (w <= 0xCF) {   // last byte of a code word
        uint32_t i;
        if (cwn_ == 0) i = w - 0x80;
        else if (cwn_ == 1) i = 80 + (cw_[0] - 0xD0) * 80 + (w - 0x80);
        else if (cw_[1] >= 0xF0) i = 44880 + (cw_[0] - 0xF0) * 1280 + (cw_[1] - 0xF0) * 80 + (w - 0x80);
        else if (cw_[0] >= 0xF0) i = 3920 + ((cw_[0] - 0xF0) * 32 + (cw_[1] - 0xD0)) * 80 + (w - 0x80);
        else i = ~0u;
        cwn_ = 0;
        if (i < wn_) for (uint32_t k = wo_[i]; k < wo_[i + 1]; ++k) Zeichen(wt_[k], cwa_);
        else Zeichen('?', cwa_);
      } else if (cwn_ < 2) {
        cw_[cwn_++] = (uint8_t)w;
      } else {
        cwn_ = 0;
        Zeichen('?', cwa_);
      }
    } else {
      cwn_ = 0;
      if (w != 0x40 && w != 0x07 && w != 0x06) Zeichen(w, pos);   // case flags carry no text
    }
    *p = ob_;
    return obn_;
  }

  __attribute__((noinline, minsize)) void NeueSeite(long long page_index) {
    seite_ = page_index;
    innum_ = 0;
    cur_ok_ = 0;
    range_ = 0;
    komp_u_ = -1;
    ubn_ = 0;
    kw_ = -1;
    zln_ = 0;
    fw_.ok = 0;
    fv_.ok = 0;
    ls_ = 0; ljn_ = 0; jb_ = 0; jd_ = 0; kat_ = 0; jlast_ = 0;
    absatz_ti_ = died_ti_ = tn_;
  }

  // ---- one decoded character (s: stream position of its byte) -----------
  void Zeichen(unsigned c, uint32_t s) {
    const bool dig = c - '0' < 10u;
    int ende = 0;
    if (innum_) {
      if (dig || ((c == ',' || c == '.') && nz_[nzn_ - 1] - '0' < 10u)) {
        if (nzn_ < 24) nz_[nzn_++] = (unsigned char)c; else innum_ = 0;
        if (dig) nze_ = s + 1;
      } else {
        ZahlEnde();
        ende = 1;
      }
    }
    if (!innum_ && dig && !ende) {
      const unsigned v = T(1);
      if (!(v - '0' < 10u || (v >= 'a' && v <= 'z') || v == '.' || v == ',')) ZahlStart(c, s);
    }
    t_[tn_ & 63] = (unsigned char)c;
    tp_[tn_ & 63] = s;
    ++tn_;
    if (!innum_ && cur_ok_) {
      if (c == '\n' || ubn_ >= 48) {
        cur_ok_ = 0;
      } else {
        ub_[ubn_++] = (unsigned char)c;
        if (c == '(') Ausloesen(0, ubn_ - 1);
        else if (c == ' ' && ubn_ >= 4 && !memcmp(ub_ + ubn_ - 4, " or ", 4)) Ausloesen(1, ubn_ - 4);
        else if (c == '/' && ubn_ >= 2 && ub_[ubn_ - 2] == ' ') Ausloesen(2, ubn_ - 2);
        else if (c == '=' && ubn_ >= 2 && ub_[ubn_ - 2] == ' ') Ausloesen(3, ubn_ - 2);
        else if (c == '|' && ubn_ >= 3 && ub_[ubn_ - 2] == '|' && ub_[ubn_ - 3] == ' ') Ausloesen(4, ubn_ - 3);
        else if (c == ')' && kw_ >= 0 && kwy_) Lernen(s);
      }
    }
    if (c == '\n' || tn_ - kwt_ > 64) kw_ = -1;
    // infobox fields
    if (fw_.ok && !innum_) {
      if (c != ' ' && !(dig && nfeld_)) fw_.ok = 0;
    }
    if (c == '|' || c == '\n' || c == '{') {
      zln_ = 0;
    } else if (c == '=') {
      FeldName();
      zln_ = 0;
    } else if (zln_ < 40) {
      zl_[zln_++] = (unsigned char)c;
    }
    Jahre(c);
  }

  void ZahlStart(unsigned c, uint32_t s) {
    innum_ = 1;
    nzn_ = 1;
    nz_[0] = (unsigned char)c;
    nsa_ = s;
    nze_ = s + 1;
    nti_ = tn_;
    nneg_ = 0;
    // a minus sign in front: "-" (not behind a digit or letter), U+2212, "&&minus;"
    if (T(1) == '-' && !(T(2) - '0' < 10u || (T(2) >= 'a' && T(2) <= 'z'))) nneg_ = 1;
    else if (EndetAuf("\x03" "\xe2\x88\x92" "\x08" "&&minus;", 2) >= 0) nneg_ = 1;
    nfeld_ = fw_.ok;
    // the text since the last number: a range separator or the big unit of
    // a compound
    int r = 0, k = -1;
    if (cur_ok_ && !cur_.neg && !nneg_) {
      const unsigned char* u = ub_;
      const int n = ubn_;
      // range separators: "-", " - ", " to ", " and ", en dash, " en dash ", "&&*", "&&^"
      r = Gleich("\x01" "-" "\x03" " - " "\x04" " to " "\x05" " and " "\x03" "\xe2\x80\x93" "\x05" " \xe2\x80\x93 "
                 "\x03" "&&*" "\x03" "&&^", 8, u, n) >= 0;
      if (!r && cur_.dx == 0 && !cur_.tk && n > 1) {
        int m = n;
        if (m > 5 && !memcmp(u + m - 5, " and ", 5)) m -= 4;
        else if (m > 2 && u[m - 2] == ',' && u[m - 1] == ' ') m -= 2;
        k = Schreibung(u, m);
        if (k >= 0) k = Einheit(k);
      }
    }
    range_ = r;
    if (r) { lo_ = cur_; rsa_ = cur_.e; rse_ = nsa_; }
    komp_u_ = k;
    if (k >= 0) { komp_m_ = cur_.m; komp_nd_ = cur_.nd; }
    cur_ok_ = 0;
    ubn_ = 0;
  }

  void ZahlEnde() {
    innum_ = 0;
    if (nz_[nzn_ - 1] == ',' || nz_[nzn_ - 1] == '.') --nzn_;
    cur_ok_ = Lesen(nz_, nzn_, &cur_);
    ubn_ = 0;
    if (!cur_ok_) { range_ = 0; komp_u_ = -1; nfeld_ = 0; return; }
    cur_.a = nsa_;
    cur_.e = nze_;
    cur_.neg = (uint8_t)nneg_;
    if (nfeld_ && fw_.ok && !range_) { fv_ = fw_; fv_.x = cur_; }
    nfeld_ = 0;
    fw_.ok = 0;
    if (kw_ >= 0 && nti_ > kwt_) kwy_ = 1;
    if (!range_ && cur_.dx == 0 && !cur_.tk && !cur_.neg && cur_.nd >= 3 && cur_.nd <= 4 && cur_.m >= 100 &&
        cur_.m <= 2010) Jahr((int)cur_.m);
  }

  // ---- the value ---------------------------------------------------------
  int Regel(int u1, int u2) const {
    for (int i = 0; i < (int)(sizeof(kRegel) / 3); ++i) if (kRegel[i][0] == u1 && kRegel[i][1] == u2) return kRegel[i][2];
    return 0;
  }
  // the number X (+ compound) in u2 by the rule of the pair, into o
  __attribute__((noinline, minsize)) int Zahltext(const Zahl& x, int u1, int u2, int komp, unsigned char* o) {
    uint64_t m = x.m;
    int sig = x.sig;
    if (komp) {   // "<a> <big> <x> <small>": X = a * f + x in the small unit
      int f = 0;
      for (int i = 0; i < 4; ++i) if (kKomp[i][0] == komp_u_ && kKomp[i][1] == u1) f = kKompF[i];
      if (f) {
        uint64_t v;
        if (__builtin_mul_overflow(komp_m_, (uint64_t)f, &v)) return 0;
        for (int k = 0; k < x.dx; ++k) {
          if (v >= kGrenze / 10) return 0;
          v *= 10;
        }
        m += v;
        sig += komp_nd_;
      }
    }
    uint64_t N, D;
    int yneg;
    if (!ValueOf(m, x.dx, x.neg, u1, u2, &N, &D, &yneg)) return 0;
    const int r = Regel(u1, u2);
    int dd;
    if (r == 0) dd = x.dx < 4 ? x.dx : 4;
    else if (r == 4) dd = 0;
    else if (r == 5) dd = 1;
    else {
      const int s = r == 1 ? sig : (r == 2 ? (sig > 2 ? sig : 2) : (sig > 3 ? sig : 3));
      dd = N ? s - 1 - Exponent(N, D) : 0;
    }
    // thousands commas as X, or from 10.000 on unless X is a 4-digit number written without
    const int intx = x.nd - x.dx;
    const int tk = x.tk || (Exponent(N ? N : 1, D) >= 4 && !(intx >= 4 && !x.tk));
    int n = 0;
    if (yneg && N) o[n++] = '-';
    const int k = Schreiben(N, D, dd, tk, x.dk, o + n);
    if (!k) return 0;
    // "-0" is no number
    bool null = true;
    for (int i = 0; i < k; ++i) if (o[n + i] - '1' < 9u) null = false;
    if (null && n) { memmove(o, o + n, (size_t)k); n = 0; }
    return n + k;
  }

  // a unit trigger of form f (0 "(", 1 " or ", 2 " /", 3 " =", 4 " ||"); the
  // unit text is ub_[0..n)
  __attribute__((noinline, minsize)) void Ausloesen(int f, int n) {
    while (n && ub_[n - 1] == ' ') --n;
    if (f == 1 && n && ub_[n - 1] == ',') --n;
    const int s = Schreibung(ub_, n);
    if (s < 0) return;
    const int u1 = Einheit(s);
    if (f == 0) { kw_ = s; kwu_ = u1; kwy_ = 0; kwt_ = tn_; }
    if (!frei_) return;
    // the target is the fixed default of the unit (harness, whole stream: 12.086 exact numbers
    // against 11.009 with the last target written behind the spelling)
    const int u2 = kZiel[u1];
    if (u2 < 0) return;
    unsigned char* o = ob_;
    int k;
    if (range_) {
      k = Zahltext(lo_, u1, u2, 0, o);
      const int ns = (int)(rse_ - rsa_);
      if (!k || ns < 1 || ns > 8) return;
      for (int i = 0; i < ns; ++i) o[k + i] = sr_[(rsa_ + (uint32_t)i) & 255];
      k += ns;
      const int k2 = Zahltext(cur_, u1, u2, 0, o + k);
      if (!k2) return;
      k += k2;
    } else {
      k = Zahltext(cur_, u1, u2, komp_u_ >= 0, o);
      if (!k) return;
    }
    if (f == 0 && k + lern_[s].n <= kMax) {
      memcpy(o + k, lern_[s].b, lern_[s].n);
      k += lern_[s].n;
    }
    Ausgabe(f, k);
  }

  // the parenthesis behind a recognised source is closed: written in the
  // default target unit, its ending becomes the ending of the source spelling
  __attribute__((noinline, minsize)) void Lernen(uint32_t s) {
    const int sp = kw_;
    kw_ = -1;
    const int n = (int)(s + 1 - cur_.e);   // stream bytes from behind Y through ")"
    int k = ubn_ - 1;
    while (k && ub_[k - 1] == ' ') --k;
    const int t = Schreibung(ub_, k);
    if (t < 0) return;
    const int u2 = Einheit(t);
    if (u2 != kZiel[kwu_] || n < 2 || n > 16) return;
    lern_[sp].n = (uint8_t)n;
    for (int i = 0; i < n; ++i) lern_[sp].b[i] = sr_[(cur_.e + (uint32_t)i) & 255];
  }

  // a field name "<stem>_<unit>" (or a unit) closed by "="
  __attribute__((noinline, minsize)) void FeldName() {
    int n = zln_;
    while (n && zl_[n - 1] == ' ') --n;
    int a = 0;
    while (a < n && zl_[a] == ' ') ++a;
    if (n - a < 1) { fw_.ok = 0; return; }
    unsigned char w[40];
    for (int i = a; i < n; ++i) w[i - a] = zl_[i] == '_' ? ' ' : zl_[i];
    n -= a;
    ++fnr_;
    int u = -1;
    uint32_t stamm = 0;
    int s = Schreibung(w, n);
    if (s >= 0) {
      u = Einheit(s);
    } else {
      int k = n - 1;
      while (k > 0 && w[k] != ' ') --k;
      if (k > 0) {
        if (n - k - 1 == 1 && w[k + 1] == 'f') u = kFt;
        else if ((s = Schreibung(w + k + 1, n - k - 1)) >= 0) u = Einheit(s);
        for (int i = 0; i < k; ++i) stamm = stamm * 31u + w[i];
      }
    }
    fw_.ok = 0;
    if (u < 0) return;
    if (fv_.ok && fv_.nr + 1 == fnr_ && fv_.stamm == stamm && fv_.u != u && kDim[fv_.u] == kDim[u]) {
      if (frei_) {
        const int k = Zahltext(fv_.x, fv_.u, u, 0, ob_);
        if (k) Ausgabe(5, k);
      }
    }
    fw_.ok = 1;
    fw_.u = u;
    fw_.stamm = stamm;
    fw_.nr = fnr_;
  }

  // the k bytes in ob_ are the injection of form f
  void Ausgabe(int f, int k) {
    if (k > kMax) return;
    obn_ = k;
    (void)f;
  }

  // ---- year reader -------------------------------------------------------
  void Jahr(int y) {
    if (ls_ == 2) { if (ljn_ < 2) lj_[ljn_] = y; else lj_[1] = y; ++ljn_; }
    jlast_ = y;
    jlast_ti_ = tn_;
  }
  __attribute__((noinline, minsize)) void Jahre(unsigned c) {
    // lead: the first bold, the first parenthesis behind it (400 characters),
    // the years in it (400 characters); a dash, "died", "d.", " to " make it a
    // life span, "born", "b." a birth
    if (ls_ == 0) {
      if (c == '\'' && Endet("'''", 3)) { ls_ = 1; lti_ = tn_; }
    } else if (ls_ == 1) {
      if (c == '(') { ls_ = 2; ltief_ = 1; lti_ = tn_; ljn_ = 0; lstrich_ = 0; lgeb_ = 0; }
      else if (tn_ - lti_ > 400 || (c == '=' && T(2) == '\n')) ls_ = 3;
    } else if (ls_ == 2) {
      if (c == '(') ++ltief_;
      else if (c == ')' && --ltief_ == 0) {
        ls_ = 3;
        // a life span "B - D" or "born B" (a single year alone is mostly no
        // life date: taxon authors "(Linnaeus, 1758)", founding years)
        if (ljn_ >= 2 && lstrich_ && !lgeb_ && lj_[1] > lj_[0] && lj_[1] - lj_[0] <= 120) { jb_ = lj_[0]; jd_ = lj_[1]; }
        else if (ljn_ == 1 && lgeb_) jb_ = lj_[0];
      } else if (tn_ - lti_ > 400) {
        ls_ = 3;
      } else {
        const int w = EndetAuf("\x01" "-" "\x02" "\x80\x93" "\x02" "\x80\x94" "\x03" "&&*" "\x03" "&&^" "\x04" "died"
                                 "\x02" "d." "\x04" " to " "\x04" "born" "\x02" "b.", 10);
        if (w >= 8) lgeb_ = 1;
        else if (w >= 0) lstrich_ = 1;
      }
    }
    if (c == '\n' && T(2) == '\n') absatz_ti_ = tn_;
    if (c == 'd' && Endet("died", 4)) died_ti_ = tn_;
    if (!frei_ || (c != ':' && c != ' ' && c != '=')) return;
    // triggers: 0 the first category, 1-3 the age, 4-5 birth, 6-7 death (persondata)
    const int w = EndetAuf("\x0a" "[category:" "\x05" "aged " "\x07" "at age " "\x0b" "the age of "
                           "\x0e" "date of birth=" "\x0f" "date of birth =" "\x0e" "date of death="
                           "\x0f" "date of death =", 8);
    if (w < 0 || (w == 0 && kat_)) return;
    // the age triggers start a word ("managed ", "engaged ", "that age " are none)
    if (w >= 1 && w <= 3 && (T(w == 1 ? 6 : w == 2 ? 8 : 12) | 0x20) - 'a' < 26u) return;
    int n = 0, f = w >= 1 && w <= 3 ? 7 : 6;
    unsigned char* o = ob_;
    if (w == 0) {   // the life years as a line behind the first "[[Category:"
      kat_ = 1;
      if (jb_) {
        n = Ziffern(jb_, o);
        if (jd_) { o[n++] = ' '; n += Ziffern(jd_, o + n); }
        o[n++] = '\n';
      }
    } else if (f == 7) {
      // the reference year: the last year of the paragraph (300 characters), else the death
      int y = 0;
      if (jb_ && jlast_ > jb_ && jlast_ - jb_ <= 120 && jlast_ti_ > absatz_ti_ && tn_ - jlast_ti_ <= 300) y = jlast_;
      else if (jb_ && jd_ && tn_ - died_ti_ <= 150) y = jd_;
      if (y) n = Ziffern(y - jb_, o);
    } else {
      const int j = w <= 5 ? jb_ : jd_;
      if (j) n = Ziffern(j, o);
    }
    if (n) Ausgabe(f, n);
  }
  static int Ziffern(int v, unsigned char* o) {
    unsigned char g[8];
    int n = 0;
    do { g[n++] = (unsigned char)('0' + v % 10); v /= 10; } while (v && n < 8);
    for (int i = 0; i < n; ++i) o[i] = g[n - 1 - i];
    return n;
  }

  void Report(const char* page_index) {
    (void)page_index;
  }
};
inline Aut aut;   // zero-initialised (no constructor)

}  // namespace computed_values
#ifdef __clang__
#pragma clang attribute pop
#endif
#endif
