// Census block, number coder.  A port of a Python prototype that is not part
// of this package ("v2" in the comments below).
//
// The values of a census record are written as residual tokens against an
// online estimate from what is known before: the values of the record coded
// earlier, and per census type ridge models over transformed features, state
// offsets, the 8 nearest places (kNN over the coordinates).  Tokens:
//   ""       exact hit of a rule (residual 0)      "*"   special value hit
//   "<r>"    residual in steps of the last digit   "g<r>" residual of the count
//   "<k>/<o>" income: bracket residual / position  "#<raw>" not reconstructable
// each with an optional format prefix "_<decimals>[c][p]" when the number format
// of the field changes (c thousands comma, p County 2-decimal padding).
// Encoder and decoder run the SAME function on a record: the encoder
// turns values into tokens, the decoder tokens into values, both update the
// models with the values afterwards.  The whole file is compiled with precise
// floating point, and exp/log are the deterministic ones of detmath.cpp, so
// both directions compute bit-equal estimates on any machine.
#ifndef CENSBLOCK_ZAHL_H_
#define CENSBLOCK_ZAHL_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "../port_cfg.h"

#if defined(__clang__)
#pragma float_control(precise, on, push)
#endif

// The Mapit/Geolinks repetition of a decimal coordinate
// without the format head: token "" = the coordinate with the trailing zeros cut
// (39,537 of 39,556 on enwik9), "*k" another of the four forms, "#" + value
// else; the format of the field is written by the primary values only.
// County prediction: the county of a place starts its candidate order with the
// counties of the nearest records of earlier types (same state) below and above
// its page; the caller sets anl4_u / anl4_o per record.

namespace census_block {
namespace zahl {

// The decimal text of v, the same bytes as std::to_string (one helper for all
// census call sites).
__attribute__((noinline)) inline std::string Dez(long v) {
  char b[24];
  return std::string(b, (size_t)snprintf(b, sizeof b, "%ld", v));
}

// A key >= 0 (page index, context key) -> dense index 0, 1, ... in insertion
// order, open addressing.  One table type for every census map keyed by a long;
// the values live in vectors of the owner, indexed by the dense index.
struct LTab {
  unsigned long* t = nullptr;   // slot: key + 1 (0 = empty), dense index
  unsigned long cap = 0, n = 0;
  LTab() = default;
  LTab(const LTab&) = delete;
  LTab& operator=(const LTab&) = delete;
  // the dense index of k, or -1
  __attribute__((noinline)) long Find(long k) const {
    if (!n) return -1;
    const unsigned long x = (unsigned long)k + 1;
    for (unsigned long i = (x * 0x9E3779B97F4A7C15ull) >> 32 & (cap - 1);; i = (i + 1) & (cap - 1)) {
      if (t[2 * i] == x) return (long)t[2 * i + 1];
      if (!t[2 * i]) return -1;
    }
  }
  // the dense index of k; absent: inserted as the next index, *neu = true
  __attribute__((noinline)) long Neu(long k, bool* neu) {
    const long f = Find(k);
    *neu = f < 0;
    if (f >= 0) return f;
    if (2 * (n + 1) > cap) {
      const unsigned long c = cap ? 2 * cap : 64;
      unsigned long* a = (unsigned long*)calloc(2 * c, sizeof(unsigned long));
      if (!a) abort();
      unsigned long* o = t;
      const unsigned long oc = cap;
      t = a;
      cap = c;
      for (unsigned long j = 0; j < oc; ++j) if (o[2 * j]) Setze(o[2 * j], o[2 * j + 1]);
      free(o);
    }
    Setze((unsigned long)k + 1, n);
    return (long)n++;
  }
  void Setze(unsigned long x, unsigned long v) {
    unsigned long i = (x * 0x9E3779B97F4A7C15ull) >> 32 & (cap - 1);
    while (t[2 * i]) i = (i + 1) & (cap - 1);
    t[2 * i] = x;
    t[2 * i + 1] = v;
  }
  void Clear() {
    free(t);
    t = nullptr;
    cap = n = 0;
  }
  ~LTab() { free(t); }
};

// field labels (stored per value slot in the variant table of the block)
enum Feld {
  F_X = 0,  // no field: the value is written as it is
  F_POP_I, F_POP, F_PV, F_LAT, F_LON, F_LAT_D, F_LAT_M, F_LAT_S, F_LON_D, F_LON_M, F_LON_S,
  F_MLAT, F_MLON, F_TOT_KM, F_TOT_MI, F_LAND_KM, F_LAND_MI, F_WAT_KM, F_WAT_MI, F_WAT_PCT,
  F_WAT_PCT2, F_DENS_KM, F_DENS_MI, F_HH, F_HH2, F_FAM, F_HU, F_HUD_KM, F_HUD_MI,
  F_R0, F_R1, F_R2, F_R3, F_R4, F_R5, F_R6, F_HISP,
  F_HH_KID, F_HH_MAR, F_HH_FEM, F_HH_NF, F_HH_IND, F_HH_65, F_HHS, F_FS,
  F_A0, F_A1, F_A2, F_A3, F_A4, F_MED, F_M100, F_M18,
  F_INC_PC, F_INC_HH, F_INC_FAM, F_INC_M, F_INC_F, F_POV, F_POV_FAM, F_POV_U18, F_POV_65,
  F_KREIS,  // county of a place: page index of its County record
  F_GLIED,  // place of a county list: members skipped (never coded, written as is)
  F_NACHBAR,  // adjacent county: rank in the neighbour order (written as is)
  kFelder
};
// model keys besides the fields
enum { M_DENS_KM_ALT = kFelder, M_TOT_KM_ALT, M_HU_ALT, M_MED2, M_FS2, kModelle };
#if PORT_CMX_MEAS
// research number-coder variants (PORT_CMX_MEAS; ported from the side agent's probe.patch, 2026-10-31; 0 = neif 2.0):
//   bit 2   V2: the County record's transformed value of the same field is a feature of the place models
//           (changes the tokens: encoder and model mirror must both run it; a bed is recoded with runner -r)
//   bit 128 RF: the scale byte of a rule value is its rounding-boundary class (model side only)
//   bit 256 with 128: RF without WAT_MI
// (bit 4 RK lives in census_block.h)
inline int g_var = 0;
inline long g_cur_page = -1;   // the page of the record being coded (County: its own page index)
struct KreisY { LTab i; std::vector<double> w; };
inline KreisY& kreis_y() { static KreisY k; return k; }
#endif
// fields written in count space with an injected estimate: none in this form;
// the zf branches below are folded away by the compiler
inline bool ZaehlFeld(int) { return false; }

static const double kKm2 = 2.589988110336;
static const double kNan = __builtin_nan("");
inline bool Ok(double x) { return x == x && x - x == 0.0; }   // finite, not NaN
// The scale byte: the estimate a preview finds for the token it stops at
#if PORT_CMX_MEAS
struct SkalaEst { int f = -1; double pred = kNan, q = kNan, chat = kNan; bool ok = false; double raw = kNan; int art = -1; };
#else
struct SkalaEst { int f = -1; double pred = kNan, q = kNan, chat = kNan; bool ok = false; };
#endif
// the scale byte of that token: half-octave class of the estimate magnitude, -1 = none
inline int SkalaByte(const SkalaEst& e) {
  if (!e.ok || e.f == F_MLAT || e.f == F_MLON) return -1;
#if PORT_CMX_MEAS
  if ((g_var & 128) && e.art == 1 && e.raw == e.raw && e.q > 0.0 && !((g_var & 256) && e.f == F_WAT_MI)) {
    // research RF: rounding-boundary class of a rule value
    const double r = e.raw / e.q, fr = r - floor(r), d = fr - 0.5;
    const int c = (int)((fabs(d) < 0.4999 ? fabs(d) : 0.4999) * 20.0);
    return 0x80 + 1 + (d >= 0 ? 10 + c : c);
  }
#endif
  double x;
  if (e.chat == e.chat) x = fabs(e.chat);   // count estimate on a coarse count grid
  else if (e.pred != e.pred || !(e.q > 0.0)) return -1;
  else if (e.f == F_INC_HH || e.f == F_INC_FAM || e.f == F_INC_M || e.f == F_INC_F) x = fabs(e.pred) / 2500.0;
  else x = fabs(e.pred) / e.q;
  const double l = 2.0 * log2(x + 1.0);
  if (!Ok(l)) return -1;
  const double sc = floor(l) + 1.0;
  return 0x80 + (sc > 99.0 ? 99 : (int)sc);
}
// XFI: per type and state (the field the previous X token matched) the counts of the next match, in the order
// of first appearance; the prediction is the most frequent one, the first of equals
struct XfiTafel {
  enum { kStart = -3, kLeer = -2, kKein = -1 };
  std::vector<std::pair<int, long>> n[8][kFelder + 3];
  int vor = kStart;
  std::vector<std::pair<int, long>>& TitleLine(int typ) { return n[typ & 7][vor + 3]; }
  int Vorhersage(int typ) {
    const auto& z = TitleLine(typ);
    int best = kKein;
    long bn = 0;
    for (const auto& e : z) if (e.second > bn) { bn = e.second; best = e.first; }
    return best;
  }
  void Lerne(int typ, int nf) {
    auto& z = TitleLine(typ);
    bool da = false;
    for (auto& e : z) if (e.first == nf) { ++e.second; da = true; break; }
    if (!da) z.push_back(std::make_pair(nf, 1L));
    vor = nf;
  }
};
// the labelled values known in front of a token: per label its first slot with a known non-empty value
inline const std::string* XfiWert(const std::vector<unsigned char>& lab, const std::vector<std::string>& w,
                                  const std::vector<char>& bek, int f) {
  for (size_t i = 0; i < lab.size() && i < w.size(); ++i) {
    if (lab[i] == F_X || !bek[i] || w[i].empty()) continue;
    // an earlier slot of the same label with a known non-empty value wins (the loop meets it first)
    if (lab[i] == f) return &w[i];
  }
  return nullptr;
}
// the field a decoded X token x matched: the first label (in the order of first appearance) whose value is x
inline int XfiTreffer(const std::vector<unsigned char>& lab, const std::vector<std::string>& w,
                      const std::vector<char>& bek, const std::string& x) {
  if (x.empty()) return XfiTafel::kLeer;
  // labels already met (a fixed table: a push of an int rvalue would instantiate a vector function that the
  // linker script keeps in the encoder block)
  unsigned char gesehen[256] = {0};
  for (size_t i = 0; i < lab.size() && i < w.size(); ++i) {
    if (lab[i] == F_X || !bek[i] || w[i].empty()) continue;
    if (gesehen[lab[i]]) continue;
    gesehen[lab[i]] = 1;
    if (w[i] == x) return lab[i];
  }
  return XfiTafel::kKein;
}

// ------------------------------------------------------------- formatting --
// komma: the thousands comma DEVIATES from the rule "comma when the integer
// part has four or more digits" (so it rarely changes from record to record)
struct Fmt { int nd = 0; bool komma = false, pad = false; bool gesetzt = false; };
// integer digits of a number string (sign and commas not counted)
inline int Ganzstellen(const std::string& s) {
  int n = 0;
  for (char c : s) {
    if (c == '.') break;
    n += (c >= '0' && c <= '9');
  }
  return n;
}

inline bool Parse(const std::string& s, double* x) {
  char buf[64];
  size_t m = 0;
  int ziff = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == ',') continue;
    if (!((c >= '0' && c <= '9') || c == '.' || (c == '-' && m == 0))) return false;
    ziff += (c >= '0' && c <= '9');
    if (m + 1 >= sizeof buf) return false;
    buf[m++] = c;
  }
  if (!ziff) return false;
  buf[m] = 0;
  *x = strtod(buf, 0);
  return true;
}
inline int Nachkomma(const std::string& s) {
  const size_t p = s.find('.');
  return p == std::string::npos ? 0 : (int)(s.size() - p - 1);
}
// '%.<nd>f' with thousands commas on demand; "" for non-finite or too long
inline std::string Formatiere(double x, const Fmt& f) {
  if (!Ok(x) || fabs(x) > 1e15) return std::string();
  char b[64];
  snprintf(b, sizeof b, "%.*f", f.nd, x + 0.0);
  std::string s(b, strlen(b));
  if (s[0] == '-') {   // no "-0.0": only '0' and '.' behind the sign
    size_t i = 1;
    while (i < s.size() && (s[i] == '0' || s[i] == '.')) ++i;
    if (i == s.size()) s.erase(0, 1);
  }
  if ((Ganzstellen(s) >= 4) == f.komma) return s;
  const size_t a = (s[0] == '-') ? 1 : 0;
  size_t e = s.find('.');
  if (e == std::string::npos) e = s.size();
  std::string o = s.substr(0, a);
  for (size_t i = a; i < e; ++i) {
    o.push_back(s[i]);
    const size_t rest = e - i - 1;
    if (rest > 0 && rest % 3 == 0) o.push_back(',');
  }
  o += s.substr(e);
  return o;
}
inline double Rnd(double x, int d) {
  const double m = pow(10.0, (double)d);
  return floor(x * m + 0.5 + 1e-9) / m;
}
inline double Rund(double x) { return nearbyint(x); }   // Python round(): half to even

static const int kStaatenZ = 52;   // states (kStaaten of the page level)
#if PORT_CMX_MEAS
static const int kMaxD = 28;       // features + 1 of a model (research: room for the V2 feature, as the probe)
#else
static const int kMaxD = 24;       // features + 1 of a model (the largest one has 21 features)
#endif
// features of a model: a braced list at the call or an array on the stack (no heap)
struct Mk {
  const double* p;
  int n;
  Mk(std::initializer_list<double> l) : p(l.begin()), n((int)l.size()) {}
  Mk(const double* q, int m) : p(q), n(m) {}
};

// ---------------------------------------------------------------- the model --
// online ridge regression with a periodic solve, feature means for missing
// features, and a decaying offset per state (v2 Cont without the probability part)
struct Cont {
  int d = 0;               // features + 1
  double A[kMaxD * kMaxD], b[kMaxD], w[kMaxD], fm[kMaxD], fn[kMaxD];
  long n = 0;
  double ysum = 0.0;
  // the offset of state s: str[s] over stn[s] records (stn 0: none yet)
  double str[kStaatenZ];
  long stn[kStaatenZ];
  void Init(int nf) {
    d = nf + 1;
    for (int i = 0; i < d * d; ++i) A[i] = 0.0;
    for (int i = 0; i < d; ++i) { A[i * d + i] = 1.0; b[i] = w[i] = fm[i] = fn[i] = 0.0; }
    for (int i = 0; i < kStaatenZ; ++i) stn[i] = 0;
  }
  void Vec(const double* f, int nf, double* v) const {
    for (int i = 0; i < d - 1; ++i) {
      const double x = i < nf ? f[i] : kNan;
      v[i] = Ok(x) ? x : (fn[i] > 0 ? fm[i] / fn[i] : 0.0);
    }
    v[d - 1] = 1.0;
  }
  void Predict(const double* v, int staat, double* base, double* mu) const {
    double s = 0.0;
    if (n >= 16) { for (int i = 0; i < d; ++i) s += w[i] * v[i]; }
    else s = n ? ysum / (double)n : 0.0;
    *base = s;
    double off = 0.0;
    if ((unsigned)staat < (unsigned)kStaatenZ && stn[staat])
      off = str[staat] * (double)stn[staat] / ((double)stn[staat] + 4.0);
    *mu = s + off;
  }
  // Gaussian elimination with partial pivoting on a copy of A
  void Loese() {
    double M[kMaxD * kMaxD], r[kMaxD], x[kMaxD];
    for (int i = 0; i < d * d; ++i) M[i] = A[i];
    for (int i = 0; i < d; ++i) { r[i] = b[i]; x[i] = 0.0; }
    for (int c = 0; c < d; ++c) {
      int p = c;
      for (int i = c + 1; i < d; ++i) if (fabs(M[i * d + c]) > fabs(M[p * d + c])) p = i;
      if (M[p * d + c] == 0.0) return;   // singular: keep w
      if (p != c) {
        for (int j = 0; j < d; ++j) std::swap(M[p * d + j], M[c * d + j]);
        std::swap(r[p], r[c]);
      }
      for (int i = c + 1; i < d; ++i) {
        const double f = M[i * d + c] / M[c * d + c];
        if (f == 0.0) continue;
        for (int j = c; j < d; ++j) M[i * d + j] -= f * M[c * d + j];
        r[i] -= f * r[c];
      }
    }
    for (int i = d - 1; i >= 0; --i) {
      double s = r[i];
      for (int j = i + 1; j < d; ++j) s -= M[i * d + j] * x[j];
      x[i] = s / M[i * d + i];
    }
    for (int i = 0; i < d; ++i) if (!Ok(x[i])) return;
    for (int i = 0; i < d; ++i) w[i] = x[i];
  }
  void Update(const double* v, const double* f, int nf, double y, double base, int staat) {
    if (!Ok(y)) return;
    for (int i = 0; i < d; ++i) if (!Ok(v[i])) return;
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) A[i * d + j] += v[i] * v[j];
      b[i] += y * v[i];
    }
    ++n;
    ysum += y;
    for (int i = 0; i < d - 1 && i < nf; ++i) {
      if (Ok(f[i])) { fm[i] += f[i]; fn[i] += 1.0; }
    }
    if (n == 16 || n == 24 || n == 32 || n == 48 || (n < 1024 && n % 32 == 0) || n % 128 == 0) Loese();
    if ((unsigned)staat >= (unsigned)kStaatenZ) return;
    const double r = y - base;
    if (!stn[staat]) { str[staat] = r; stn[staat] = 1; }
    else { str[staat] = 0.85 * str[staat] + 0.15 * r; ++stn[staat]; }
  }
};

// transforms and their inverses (v2: tlog1p, tlog, tlogit, tid)
enum Tr { T_LOG1P, T_LOG, T_LOGIT, T_ID, T_HU };
struct Trafo {
  int art = T_ID;
  double off = 0.0;     // t(x) - off
  double h = 0.0;       // households (T_HU)
  double T(double x) const {
    double y;
    switch (art) {
      case T_LOG1P: y = log(1.0 + (x > -0.999999 ? x : -0.999999)); break;
      case T_LOG: y = log(x > 1e-9 ? x : 1e-9); break;
      case T_LOGIT: { const double p = x < -0.4999 ? -0.4999 : (x > 100.4999 ? 100.4999 : x);
                      y = log((p + 0.5) / (100.5 - p)); break; }
      case T_HU: { const double z = x - h + 1.0; y = log(z > 0.5 ? z : 0.5); break; }
      default: y = x;
    }
    return y - off;
  }
  double Inv(double y) const {
    y += off;
    switch (art) {
      case T_LOG1P: return exp(y) - 1.0;
      case T_LOG: return exp(y);
      case T_LOGIT: { const double e = exp(y < 50.0 ? y : 50.0); return (100.5 * e - 0.5) / (1.0 + e); }
      case T_HU: return h - 1.0 + exp(y < 50.0 ? y : 50.0);
      default: return y;
    }
  }
};
inline double L(double x) { return Ok(x) ? log(1.0 + (x > 0.0 ? x : 0.0)) : kNan; }
inline double Lg(double p) {
  if (!Ok(p)) return kNan;
  p = p < -0.49 ? -0.49 : (p > 100.49 ? 100.49 : p);
  return log((p + 0.5) / (100.5 - p));
}

// County records per state in record order (page indices): the candidates of
// F_KREIS.  The encoder fills it from a dry pass over the County pages, the
// decoder while it reads the County records (they come first in the line).
inline std::vector<std::vector<long>> kreisliste;
inline long anl4_u = -1, anl4_o = -1;   // the two candidates of the running record, -1 none

// per census type: models, kNN store, format state, carried context
static const int kNv = kFelder + 2;   // kNN store: per record lat, lon, y per field
// the up to 8 nearest records of NbRechnen: a plain array, copied by value (no vector copy instances)
struct Nb8 {
  int v[8] = {};
  int n = 0;
  bool empty() const { return n == 0; }
  const int* begin() const { return v; }
  const int* end() const { return v + n; }
  void clear() { n = 0; }
  void push_back(int x) { v[n++] = x; }
};
struct Typmodell {
  Cont M[kModelle];
  bool mok[kModelle] = {false};
  Fmt fmt[kFelder];
  std::vector<double> nv;   // kNN store: per record lat, lon, then y per field (kNv values, NaN if absent)
  Nb8 nbi;
  long nb_m = -1;                    // NbSetzen: the last query and its answer (the previews repeat it)
  double nb_la = 0.0, nb_lo = 0.0;
  Nb8 nb_c;
  double cur_y[kFelder];
  int prevstate = -1;
  double prev_lat = kNan, prev_lon = kNan;
  std::vector<double> prevg[kStaatenZ];   // per state: growth by offset from 2000 (NaN none)
  LTab kzi;                          // county page -> index into kzw
  std::vector<double> kzw;           // per county: coordinate sums lat, lon and count of the places of this type
  std::vector<long> kmtf[kStaatenZ];   // per state: counties, the last used first
  Typmodell() {
    for (int f = 0; f < kFelder; ++f) cur_y[f] = kNan;
  }
  Cont& Modell(int k, int nf) {
    if (!mok[k]) { M[k].Init(nf); mok[k] = true; }
    return M[k];
  }
};

// ---------------------------------------------------------- one record --
// R.s[f] holds the value strings (encoder: input; decoder: output), R.da[f]
// whether the field is present.  pv: (year, value) rows of the table.
struct Rec {
  bool da[kFelder] = {false};
  std::string s[kFelder];
  std::vector<int> pvj;           // year of the pv row
  std::vector<std::string> pvs;   // value of the pv row
};

inline Trafo Tf(int art, double off = 0.0) { Trafo t; t.art = art; t.off = Ok(off) ? off : 0.0; return t; }
struct Gitter { char art = 0; double N = kNan; };   // 'p' percent of N, 'r' ratio k/N, 'm' 100 M/(N-M)
inline Gitter Gt(char a, double N) { Gitter g; g.art = a; g.N = N; return g; }
// the prediction of a rule: a constant, a value rounded to the decimals of the
// field, or the remainder share (100 % minus the other shares, from the counts
// when the grid allows it).  One body for all rules (code size).
struct Vor {
  int art = 0;                  // 0 x, 1 Rnd(x, decimals), 2 remainder share
  double x = kNan;
  double pop = kNan, cnt = 0.0, sum = 0.0;
  bool allg = false;            // all shares came from counts
  bool pad1 = false;            // one decimal on the County padding grid
};

class Run;
struct StetigKod {   // the locals of Stetig that its encoder branch reads
  int f;
  const std::string* pre;
  Fmt fm;
  double q, spezial, chat, pr, N;
  char art;
  bool zf, grob, pok;
  int dq;
};
struct KodZweige {
  void (*head)(Run&, int, Fmt*, std::string*, bool*, Fmt&) = nullptr;
  void (*regel)(Run&, int, const std::string&, const Fmt&, double, double) = nullptr;
  void (*stetig)(Run&, const StetigKod&) = nullptr;
  void (*einkommen)(Run&, int, const std::string&, const Fmt&, double, bool) = nullptr;
  void (*kreis)(Run&, const std::string&, const std::vector<long>&) = nullptr;
};
inline KodZweige kodzweige;

class Run {
 public:
  Run(Typmodell& T, Rec& R, bool cnty, int staat, bool enc, std::vector<std::string>* tok, size_t ti = 0)
      : T_(T), R_(R), cnty_(cnty), staat_(staat), enc_(enc), tok_(tok), ti_(ti) {
    for (int f = 0; f < kFelder; ++f) V_[f] = kNan;
  }
  bool Gut() const { return !fehler_; }
  // dry run: only the order of the fields (the model side needs it to inject
  // the field labels); no token, no model, no state is touched
  void Trocken(std::vector<unsigned char>* folge) { folge_ = folge; }
  bool Genutzt(int f) const { return genutzt_[f]; }
  size_t Gelesen() const { return ti_; }
  // preview (decoder only): decode the tokens before index `bis` without
  // changing the type model (formats in a local copy, no model update, no
  // state), stop at token `bis` and keep the estimate of its field
  void Vorschau(size_t bis, std::string* est) {
    vorschau_ = (long)bis;
    est_ = est;
    for (int f = 0; f < kFelder; ++f) fmtl_[f] = T_.fmt[f];
  }
  // preview: stop at any field with a format and keep its estimate (SKALA)
  void Skala(SkalaEst* s) { sk_ = s; }
  // the encoder sets the pointers to its branches (see KodZweige)
  ENC_ONLY static void ZweigeSetzen() {
    kodzweige.head = &KopfW;
    kodzweige.regel = &RegelW_;
    kodzweige.stetig = &StetigW;
    kodzweige.einkommen = &EinkommenW;
    kodzweige.kreis = &KreisW;
  }
  void Alles() {
    Bevoelkerung();
    Koordinaten();
    Kreis();
    Geografie();
    Rest();
    if (folge_ || vorschau_ >= 0) return;
    T_.prevstate = staat_;
    NbCommit(lat_, lon_);
#if PORT_CMX_MEAS
    if ((g_var & 2) && cnty_ && g_cur_page >= 0) {   // research V2: keep the county's values
      bool neu;
      const long zi = kreis_y().i.Neu(g_cur_page, &neu);
      if (neu) kreis_y().w.resize(kreis_y().w.size() + kFelder, kNan);
      for (int f = 0; f < kFelder; ++f) kreis_y().w[(size_t)zi * kFelder + (size_t)f] = Has(f) ? T_.cur_y[f] : kNan;
    }
#endif
  }

 private:
  Typmodell& T_;
  Rec& R_;
  bool cnty_;
  int staat_;
  bool enc_;
  std::vector<std::string>* tok_;
  size_t ti_;
  bool fehler_ = false;
  double V_[kFelder];
  std::string pvs_;                  // value of the running pv row
  double pop_ = kNan, dmi_ = kNan, lkc_ = kNan, wv_ = kNan, lat_ = kNan, lon_ = kNan;
  int wp_ = -1;
  std::vector<unsigned char>* folge_ = nullptr;
  bool genutzt_[kFelder] = {false};
  long vorschau_ = -1;               // preview: stop at this token index
  bool halt_ = false;                // preview reached its token
  std::string* est_ = nullptr;       // preview: the estimate of the stop field
  Fmt fmtl_[kFelder];                // preview: formats (the model keeps its own)
  SkalaEst* sk_ = nullptr;           // preview: the estimate of the stop field (SKALA)
  bool Lernt() const { return !folge_ && vorschau_ < 0; }

  // a token as a copy (push_back(const&): one instance for all token sites)
  void Tok(const std::string& t) { tok_->push_back(t); }
  // the token a + m + b, built in one place instead of two operator+ temporaries per site
  __attribute__((noinline)) void Tok3(const std::string& a, const char* m, const std::string& b) {
    std::string t(a);
    t += m;
    t += b;
    Tok(t);
  }
  bool Has(int f) const { return R_.da[f]; }
  double G(int f) const { return Has(f) ? V_[f] : kNan; }
  // Lg / L of a field value: one call instead of the field test and the call at every site
  __attribute__((noinline)) double LgF(int f) const { return Lg(G(f)); }
  __attribute__((noinline)) double LF(int f) const { return L(G(f)); }
  std::string& Str(int f) { return f == F_PV ? pvs_ : R_.s[f]; }
  void ValueOf(int f) {
    if (f == F_PV) return;
    double x;
    V_[f] = Parse(R_.s[f], &x) ? x : kNan;
  }
  double WertVon(int f) {
    double x;
    return Parse(Str(f), &x) ? x : kNan;
  }
  static double Q(const Fmt& fm) { return fm.pad ? 0.1 : pow(10.0, -(double)fm.nd); }

  // ---- token head.  Encoder: format of s, prefix "_<nd>[c][p]" if it changed;
  // a value whose format cannot be written is escaped (esc).  Decoder: prefix
  // parsed from the token, else the last format of the field.  Returns the body.
  bool FieldHead(int f, Fmt* fm, std::string* pre, std::string* body, bool* esc) {
    genutzt_[f] = true;
    if (folge_) { folge_->push_back((unsigned char)f); return false; }
    Fmt& last = vorschau_ >= 0 ? fmtl_[f] : T_.fmt[f];
    *esc = false;
    if (vorschau_ >= 0 && (long)ti_ >= vorschau_) {
      // preview: the token of this field is not known yet.  A count-space
      // field computes its estimate with the last format; everything else stops.
      // SKALA: once the stop token failed (no format), no later field answers for it
      if (!halt_ && !(sk_ != nullptr && fehler_) && (ZaehlFeld(f) || sk_ != nullptr) && last.gesetzt) {
        halt_ = true;
        *fm = last;
        pre->clear();
        body->clear();
        return true;
      }
      fehler_ = true;
      return false;
    }
    if (enc_) {
      kodzweige.head(*this, f, fm, pre, esc, last);
      return true;
    }
    if (ti_ >= tok_->size()) { fehler_ = true; return false; }
    const std::string& t = (*tok_)[ti_++];
    size_t i = 0;
    if (!t.empty() && t[0] == '_') {
      if (t.size() < 2 || t[1] < '0' || t[1] > '9') { fehler_ = true; return false; }
      Fmt k;
      k.nd = t[1] - '0';
      i = 2;
      if (i < t.size() && t[i] == 'c') { k.komma = true; ++i; }
      if (i < t.size() && t[i] == 'p') { k.pad = true; ++i; }
      k.gesetzt = true;
      last = k;
    }
    *fm = last;
    *body = t.substr(i);
    if (!body->empty() && (*body)[0] == '#') { *esc = true; Str(f) = body->substr(1); return true; }
    if (!last.gesetzt) { fehler_ = true; return false; }
    return true;
  }
  ENC_ONLY void KopfK(int f, Fmt* fm, std::string* pre, bool* esc, Fmt& last) {
    const std::string& s = Str(f);
    Fmt k;
    k.nd = Nachkomma(s);
    k.komma = (s.find(',') != std::string::npos) != (Ganzstellen(s) >= 4);
    k.pad = cnty_ && k.nd == 2 && !s.empty() && s.back() == '0' && !(f >= F_R0 && f <= F_HISP) &&
            f != F_WAT_PCT && f != F_WAT_PCT2;
    k.gesetzt = true;
    pre->clear();
    if (k.nd > 9) { *esc = true; *fm = last; return; }
    if (!last.gesetzt || last.nd != k.nd || last.komma != k.komma || last.pad != k.pad) {
      pre->push_back('_');
      pre->push_back((char)('0' + k.nd));
      if (k.komma) pre->push_back('c');
      if (k.pad) pre->push_back('p');
    }
    last = k;
    *fm = k;
  }
  ENC_ONLY static void KopfW(Run& L, int f, Fmt* fm, std::string* pre, bool* esc, Fmt& last) {
    L.KopfK(f, fm, pre, esc, last);
  }
  // encoder: the token is written if it restores s, else the escape
  ENC_ONLY void Schreib(int f, const std::string& pre, const std::string& body, const std::string& rekon) {
    const std::string& s = Str(f);
    if (!rekon.empty() && rekon == s) Tok3(pre, "", body);
    else Tok3(pre, "#", s);
  }
  static bool Ganz(const std::string& b, size_t a, long* r) {
    if (a >= b.size()) return false;
    char* e = nullptr;
    *r = strtol(b.c_str() + a, &e, 10);
    return *e == 0 && !(b.size() == a + 1 && b[a] == '-');
  }

  static double Vorhersage(const Vor& p, const Fmt& fm, double q) {
    if (p.art == 0) return p.x;
    if (p.art == 1) return Ok(p.x) ? Rnd(p.x, fm.nd) : kNan;
    const int dd = (p.pad1 && q == 0.1) ? 1 : fm.nd;
    if (Ok(p.pop) && p.pop != 0.0 && p.allg && 100.0 / p.pop > q) return Rnd(100.0 * (p.pop - p.cnt) / p.pop, dd);
    return Rnd(100.0 - p.sum, dd);
  }
  // ---- rule: value = round(pred / q) * q + r * q; pred from the format (rounding
  // to the decimals of the field), q fixed (qfest > 0) or from the format
  void Regel(int f, const Vor& vor, double qfest = 0.0) {
    Fmt fm;
    std::string pre, body;
    bool esc;
    if (!FieldHead(f, &fm, &pre, &body, &esc)) return;
    if (halt_ && sk_) {   // preview stop: the estimate with the last format
      const double q = qfest > 0 ? qfest : Q(fm);
      sk_->f = f; sk_->pred = Vorhersage(vor, fm, q); sk_->q = q; sk_->chat = kNan; sk_->ok = true;
#if PORT_CMX_MEAS
      sk_->raw = vor.x; sk_->art = vor.art;
#endif
      fehler_ = true;
      return;
    }
    if (!esc) {
      const double q = qfest > 0 ? qfest : Q(fm);
      const double pred = Vorhersage(vor, fm, q);
      if (enc_) {
        kodzweige.regel(*this, f, pre, fm, q, pred);
      } else {
        long r = 0;
        if (!Ok(pred) || (!body.empty() && !Ganz(body, 0, &r))) { fehler_ = true; return; }
        const double prq = Rund(pred / q) * q;
        Str(f) = Formatiere(prq + (double)r * q, fm);
        if (Str(f).empty()) { fehler_ = true; return; }
      }
    } else if (enc_) {
      Tok3(pre, "#", Str(f));
    }
    ValueOf(f);
  }
  // encoder branch of Regel
  ENC_ONLY void RegelK(int f, const std::string& pre, const Fmt& fm, double q, double pred) {
    double x;
    if (!Ok(pred) || !Parse(Str(f), &x)) {
      Tok3(pre, "#", Str(f));
    } else {
      const double prq = Rund(pred / q) * q;
      const long r = (long)Rund((x - prq) / q);
      Schreib(f, pre, r ? Dez(r) : std::string(), Formatiere(prq + (double)r * q, fm));
    }
  }
  ENC_ONLY static void RegelW_(Run& L, int f, const std::string& pre, const Fmt& fm, double q, double pred) {
    L.RegelK(f, pre, fm, q, pred);
  }
  void RegelW(int f, double pred, double qfest = 0.0) {
    Vor v;
    v.x = pred;
    Regel(f, v, qfest);
  }
  void RegelRnd(int f, double x) {   // v2: rnd(x, decimals of the field)
    Vor v;
    v.art = 1;
    v.x = x;
    Regel(f, v);
  }
  void RegelRest(int f, double pop, double cnt, bool allg, const double* teile, int nt, bool pad1) {
    Vor v;
    v.art = 2;
    v.pop = pop;
    v.cnt = cnt;
    v.allg = allg;
    v.pad1 = pad1;
    for (int i = 0; i < nt; ++i) v.sum += teile[i];
    Regel(f, v);
  }

  // encoder branch of Stetig (the same expressions as before, from the locals in a)
  ENC_ONLY void StetigK(const StetigKod& a) {
    const int f = a.f;
    const std::string& pre = *a.pre;
    const Fmt& fm = a.fm;
    const double q = a.q, spezial = a.spezial, chat = a.chat, pr = a.pr, N = a.N;
    const bool zf = a.zf, grob = a.grob, pok = a.pok;
    const int dq = a.dq;
    Gitter gt;
    gt.art = a.art;
    gt.N = N;
    auto gw = [&](double c) { return gt.art == 'p' ? 100.0 * c / N : (gt.art == 'r' ? c / N : 100.0 * c / (N - c)); };
    auto zaehl_ok = [&](double x, const std::string& s, double* c) {
      if (!grob || gt.art != 'p') return false;
      *c = Rund(x * N / 100.0);
      return fabs(100.0 * *c / N - x) <= q / 2 + 1e-9 && Formatiere(Rnd(gw(*c), dq), fm) == s;
    };
    const std::string& s = Str(f);
    double x;
    if (!Parse(s, &x)) {
      Tok3(pre, "#", s);
    } else if (Ok(spezial) && fabs(x - spezial) < q / 2 && Formatiere(spezial, fm) == s) {
      Tok3(pre, "*", std::string());
    } else if (zf) {
      double c;
      double y;
      if (zaehl_ok(x, s, &c)) Tok3(pre, "k", Dez((long)c));
      else if (!s.empty() && s[0] != '*' && s[0] != 'k' && s[0] != '#' && s[0] != '_' && Parse(s, &y) &&
               Formatiere(y, fm) == s) Tok3(pre, "", s);
      else Tok3(pre, "#", s);
    } else {
      bool g = false;
      if (grob && Ok(chat)) {
        double c = kNan;
        if (gt.art == 'p') { c = Rund(x * N / 100.0); if (!(fabs(100.0 * c / N - x) <= q / 2 + 1e-9)) c = kNan; }
        else if (gt.art == 'r') { c = Rund(x * N); if (!(fabs(c / N - x) <= q / 2 + 1e-9)) c = kNan; }
        else {
          const double c0 = Rund(N * x / (100.0 + x));
          const double cs[3] = {c0, c0 - 1.0, c0 + 1.0};
          for (double cc : cs)
            if (cc > 0.0 && cc < N && fabs(100.0 * cc / (N - cc) - x) <= q / 2 + 1e-9) { c = cc; break; }
        }
        if (Ok(c) && Formatiere(Rnd(gw(c), dq), fm) == s) {
          Tok3(pre, "g", Dez((long)(c - Rund(chat))));
          g = true;
        }
      }
      if (!g) {
        if (pok) {
          const long r = (long)Rund((x - pr) / q);
          Schreib(f, pre, Dez(r), Formatiere(pr + (double)r * q, fm));
        } else {
          Tok3(pre, "#", s);
        }
      }
    }
  }
  ENC_ONLY static void StetigW(Run& L, const StetigKod& a) { L.StetigK(a); }

  // the model input of a value: the features, the kNN estimate behind them (x, n),
  // the input vector v of model mk and its estimate (base without, mu with the
  // state offset)
  struct EstInput { double x[kMaxD], v[kMaxD], base, mu; int n; };
  Cont& Schaetze(int f, int mk, Mk ft, EstInput* e) {
#if PORT_CMX_MEAS
    e->n = ft.n < kMaxD - 6 ? ft.n : kMaxD - 6;   // = 22 as released
    for (int i = 0; i < e->n; ++i) e->x[i] = ft.p[i];
    if ((g_var & 2) && !cnty_) {   // research V2: the county's value of this field
      double cy = kNan;
      if (Has(F_KREIS) && Ok(V_[F_KREIS]) && f < kFelder) {
        const long zi = kreis_y().i.Find((long)V_[F_KREIS]);
        if (zi >= 0) cy = kreis_y().w[(size_t)zi * kFelder + (size_t)f];
      }
      e->x[e->n++] = cy;
    }
#else
    e->n = ft.n < kMaxD - 2 ? ft.n : kMaxD - 2;
    for (int i = 0; i < e->n; ++i) e->x[i] = ft.p[i];
#endif
    e->x[e->n++] = Nb(f);   // kNN
    Cont& m = T_.Modell(mk, e->n);
    m.Vec(e->x, e->n, e->v);
    m.Predict(e->v, staat_, &e->base, &e->mu);
    return m;
  }

  // ---- continuous value: estimate from a ridge model on transformed features,
  // count grid (coarse grid: one count step above the last digit), special value
  // (nfq: fam / hh, the special value 100 (1 - fam / hh) of HH_NF on the grid
  // of the format, 1 decimal with the County padding)
  // Stetig with the transform Tf(art) (no offset, no special value): one call form for most sites
  __attribute__((noinline)) void StetigT(int f, int mk, Mk feats, int art, double qfest, Gitter gt = Gitter(),
                                        bool null = false) {
    Stetig(f, mk, feats, Tf(art), qfest, gt, null);
  }
  void Stetig(int f, int mk, Mk feats, Trafo tr, double qfest, Gitter gt = Gitter(),
              bool null = false, double spezial = kNan, double nfq = kNan) {
    Fmt fm;
    std::string pre, body;
    bool esc;
    if (!FieldHead(f, &fm, &pre, &body, &esc)) return;
    const double q = qfest > 0 ? qfest : Q(fm);
    double off = 0.0;
    if (Ok(nfq)) {
      spezial = Rnd(100.0 * (1.0 - nfq), q == 0.1 ? 1 : fm.nd);
      off = Lg(spezial);
      if (!Ok(off)) off = 0.0;
    }
    tr.off += off;
    EstInput e;
    Cont& m = Schaetze(f, mk, feats, &e);
    const double pred = tr.Inv(e.mu);
    const bool pok = Ok(pred);
    const double pr = pok ? Rund(pred / q) * q : kNan;
    const double N = gt.N;
    bool grob = false;
    if (gt.art && Ok(N) && N > 0.0) {
      if (gt.art == 'p') grob = 100.0 / N > q;
      else if (gt.art == 'r') grob = 1.0 / N > q;
      else grob = 400.0 / N > q;
    }
    double chat = kNan;
    if (grob && pok) chat = gt.art == 'p' ? pred * N / 100.0 : (gt.art == 'r' ? pred * N : N * pred / (100.0 + pred));
    const double lq = -Rund(log10(q));
    const int dq = lq > 0 ? (int)lq : 0;
    auto gw = [&](double c) { return gt.art == 'p' ? 100.0 * c / N : (gt.art == 'r' ? c / N : 100.0 * c / (N - c)); };
    const bool zf = ZaehlFeld(f);
    if (halt_ && sk_) {   // preview stop: estimate, step and count estimate with the last format
      sk_->f = f; sk_->pred = pred; sk_->q = q; sk_->chat = chat; sk_->ok = true;
      fehler_ = true;
      return;
    }
    if (halt_ && vorschau_ >= 0 && est_) {
      // preview stop: the estimate in the form of the token; =2 clamps it to
      // the possible range (count 0..N, share 0..100), measured worse on the
      // whole census line (+237 B)
      if (grob && Ok(chat)) {
        const double c = chat;
        *est_ = "k" + Dez((long)Rund(c));
      } else if (pok) {
        const double x = pr;
        *est_ = Formatiere(x, fm);
      }
      fehler_ = true;
      est_ = nullptr;
      return;
    }
    // count space (zf): the decoder needs no estimate - "k<count>" on the grid,
    // the special value, or the value as written (encoder: StetigK)
    if (!esc) {
      if (enc_) {
        StetigKod a;
        a.f = f; a.pre = &pre; a.fm = fm; a.q = q; a.spezial = spezial; a.chat = chat; a.pr = pr; a.N = N;
        a.art = gt.art; a.zf = zf; a.grob = grob; a.pok = pok; a.dq = dq;
        kodzweige.stetig(*this, a);
      } else {
        std::string& s = Str(f);
        if (body.size() == 1 && body[0] == '*') {
          if (!Ok(spezial)) { fehler_ = true; return; }
          s = Formatiere(spezial, fm);
        } else if (zf) {
          long c;
          if (!body.empty() && body[0] == 'k') {
            if (!grob || gt.art != 'p' || !Ganz(body, 1, &c)) { fehler_ = true; return; }
            s = Formatiere(Rnd(gw((double)c), dq), fm);
          } else {
            s = body;
          }
        } else if (!body.empty() && body[0] == 'g') {
          long r;
          if (!grob || !Ok(chat) || !Ganz(body, 1, &r)) { fehler_ = true; return; }
          s = Formatiere(Rnd(gw(Rund(chat) + (double)r), dq), fm);
        } else {
          long r;
          if (!pok || !Ganz(body, 0, &r)) { fehler_ = true; return; }
          s = Formatiere(pr + (double)r * q, fm);
        }
        if (s.empty()) { fehler_ = true; return; }
      }
    } else if (enc_) {
      Tok3(pre, "#", Str(f));
    }
    ValueOf(f);
    const double x = WertVon(f);
    const double y = Ok(x) ? tr.T(x) : kNan;
    // preview (SKALA): the population rows update their shared model as in the
    // coder, row by row (Zahlen puts the model back afterwards)
    const bool pvlernt = vorschau_ >= 0 && sk_ && mk == F_PV;
    if (!Lernt() && !pvlernt) return;
    if (!pvlernt && f != F_PV) T_.cur_y[f] = y;
    if (null && Ok(x) && x == 0.0) m.Update(e.v, e.x, e.n, tr.T(q / 4), e.base, staat_);
    else m.Update(e.v, e.x, e.n, y, e.base, staat_);
  }

  // encoder branch of Einkommen
  ENC_ONLY void EinkommenK(int f, const std::string& pre, const Fmt& fm, double pred, bool pc) {
    const std::string& s = Str(f);
    double xd;
    if (!Ok(pred) || !Parse(s, &xd) || xd < 0 || xd != floor(xd) || xd > 1e12) {
      Tok3(pre, "#", s);
    } else {
      const long x = (long)xd, rp = (long)Rund(pred);
      if (pc || x >= 200000) {
        Schreib(f, pre, Dez(x - rp), Formatiere((double)(rp + (x - rp)), fm));
      } else {
        const long lo = x / 2500 * 2500, o = x - lo;
        const long kp = (long)floor((pred > 0 ? pred : 0.0) / 2500.0), kr = lo / 2500 - kp;
        std::string kb = Dez(kr);   // kr "/" o without the operator+(string&&, string&&) instance
        kb += '/';
        kb += Dez(o);
        Schreib(f, pre, kb, Formatiere((double)((kp + kr) * 2500 + o), fm));
      }
    }
  }
  ENC_ONLY static void EinkommenW(Run& L, int f, const std::string& pre, const Fmt& fm, double pred, bool pc) {
    L.EinkommenK(f, pre, fm, pred, pc);
  }

  // ---- income: bracket of 2500 $ residual + position, or a direct residual
  void Einkommen(int f, Mk feats, bool pc) {
    Fmt fm;
    std::string pre, body;
    bool esc;
    if (!FieldHead(f, &fm, &pre, &body, &esc)) return;
    EstInput e;
    Cont& m = Schaetze(f, f, feats, &e);
    const Trafo tr = Tf(T_LOG1P);
    const double pred = tr.Inv(e.mu);
    if (halt_ && sk_) {   // preview stop: the estimate (no format in it)
      sk_->f = f; sk_->pred = pred; sk_->q = 1.0; sk_->chat = kNan; sk_->ok = true;
      fehler_ = true;
      return;
    }
    if (!esc) {
      if (enc_) {
        kodzweige.einkommen(*this, f, pre, fm, pred, pc);
      } else {
        std::string& s = Str(f);
        const size_t sl = body.find('/');
        long a, o;
        if (!Ok(pred)) { fehler_ = true; return; }
        if (sl == std::string::npos) {
          if (!Ganz(body, 0, &a)) { fehler_ = true; return; }
          s = Formatiere(Rund(pred) + (double)a, fm);
        } else {
          if (!Ganz(body.substr(0, sl), 0, &a) || !Ganz(body, sl + 1, &o)) { fehler_ = true; return; }
          const long kp = (long)floor((pred > 0 ? pred : 0.0) / 2500.0);
          s = Formatiere((double)((kp + a) * 2500 + o), fm);
        }
        if (s.empty()) { fehler_ = true; return; }
      }
    } else if (enc_) {
      Tok3(pre, "#", Str(f));
    }
    ValueOf(f);
    const double x = V_[f];
    if (!Lernt()) return;   // preview: the model stays as it is
    T_.cur_y[f] = Ok(x) ? tr.T(x) : kNan;
    m.Update(e.v, e.x, e.n, T_.cur_y[f], e.base, staat_);
  }

  // ---- Mapit / Geolinks: the decimal coordinate again, in one of four forms
  void Wiederholung(int f, int von) {
    genutzt_[f] = true;
    if (folge_) { folge_->push_back((unsigned char)f); return; }
    if (vorschau_ >= 0 && (long)ti_ >= vorschau_) { fehler_ = true; return; }
    std::string form[4];
    if (Has(von)) {
      const std::string& x = R_.s[von];
      auto kurz = [](std::string y) {
        if (y.find('.') != std::string::npos) {
          size_t n = y.size();
          while (n && y[n - 1] == '0') --n;
          if (n && y[n - 1] == '.') --n;
          y.erase(n);
        }
        return y;
      };
      form[0] = x;
      form[1] = kurz(x);
      char b[64];
      snprintf(b, sizeof b, "%.4f", V_[von]);
      form[2] = b;
      form[3] = kurz(form[2]);
    }
    if (enc_) {
      const std::string& v = Str(f);
      int k = -1;
      for (int i = 0; i < 4 && Has(von); ++i) if (form[i] == v) { k = i; break; }
      if (Has(von) && form[1] == v) Tok(std::string());
      else if (k >= 0) Tok3(std::string(), "*", Dez(k));
      else Tok3(std::string(), "#", v);
    } else {
      if (ti_ >= tok_->size()) { fehler_ = true; return; }
      const std::string& t = (*tok_)[ti_++];
      if (t.empty()) {
        if (!Has(von)) { fehler_ = true; return; }
        Str(f) = form[1];
      } else if (t[0] == '#') {
        Str(f) = t.substr(1);
      } else if (t.size() == 2 && t[0] == '*' && t[1] >= '0' && t[1] <= '3' && Has(von)) {
        Str(f) = form[t[1] - '0'];
      } else {
        fehler_ = true;
        return;
      }
    }
    ValueOf(f);
  }

  // ---- kNN over the places of the type coded before (8 nearest, mean of y)
  double Nb(int f) const {
    if (T_.nbi.empty() || f == F_PV) return kNan;
    double s = 0.0;
    int k = 0;
    for (int i : T_.nbi) {
      const double x = T_.nv[(size_t)i * kNv + 2 + (size_t)f];
      if (Ok(x)) { s += x; ++k; }
    }
    return k ? s / (double)k : kNan;
  }
  void NbSetzen(double lat, double lon) {
    const long mq = (long)(T_.nv.size() / kNv);
    if (T_.nb_m == mq && memcmp(&T_.nb_la, &lat, sizeof lat) == 0 && memcmp(&T_.nb_lo, &lon, sizeof lon) == 0) {
      T_.nbi = T_.nb_c;
      return;
    }
    NbRechnen(lat, lon);
    T_.nb_m = mq;
    T_.nb_la = lat;
    T_.nb_lo = lon;
    T_.nb_c = T_.nbi;
  }
  void NbRechnen(double lat, double lon) {
    T_.nbi.clear();
    const int m = (int)(T_.nv.size() / kNv);
    if (!Ok(lat) || !Ok(lon) || m < 8) return;
    const double cl = cos(lat * 3.14159265358979323846 / 180.0);
    // the 8 smallest (distance, index), ascending: insertion into a short list
    // (i grows, so an equal distance stays behind; equal to std::partial_sort)
    double bd[8];
    int bi[8];
    int k = 0;
    for (int i = 0; i < m; ++i) {
      const double* r = &T_.nv[(size_t)i * kNv];
      if (!Ok(r[0]) || !Ok(r[1])) continue;
      const double dx = r[0] - lat, dy = (r[1] - lon) * cl;
      const double d = dx * dx + dy * dy;
      if (k == 8 && !(d < bd[7])) continue;
      int j = k < 8 ? k++ : 7;
      for (; j > 0 && d < bd[j - 1]; --j) { bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; }
      bd[j] = d;
      bi[j] = i;
    }
    for (int j = 0; j < k; ++j) T_.nbi.push_back(bi[j]);
  }
  void NbCommit(double lat, double lon) {
    T_.nv.push_back(Ok(lat) ? lat : kNan);
    T_.nv.push_back(Ok(lon) ? lon : kNan);
    for (int f = 0; f < kFelder; ++f) T_.nv.push_back(Has(f) ? T_.cur_y[f] : kNan);
  }

  // ---- county of a place (county index): rank among the County records
  // of its state, ordered by the distance of the place to the mean coordinate of
  // the places of this type already in that county, then by the last use (move
  // to front).  Token: "" rank 0, "<rank>", "#<page index>" not a candidate.
  std::vector<long> KreisFolge() const {
    std::vector<long> c;
    if (staat_ < 0 || staat_ >= (int)kreisliste.size()) return c;
    const std::vector<long>& L = kreisliste[(size_t)staat_];
    const std::vector<long>& M = T_.kmtf[staat_];   // kreisliste has kStaatenZ states at most
    const bool ko = Ok(lat_) && Ok(lon_);
    const double cl = ko ? cos(lat_ * 3.14159265358979323846 / 180.0) : 1.0;
    struct E { double d; long m; long s; };
    std::vector<E> v;
    for (size_t i = 0; i < L.size(); ++i) {
      E e;
      e.s = L[i];
      e.d = 1e300;
      if (ko) {
        const long zi = T_.kzi.Find(L[i]);
        const double* z = zi >= 0 ? &T_.kzw[(size_t)zi * 3] : nullptr;
        if (z && z[2] > 0.0) {
          const double dx = z[0] / z[2] - lat_, dy = (z[1] / z[2] - lon_) * cl;
          e.d = dx * dx + dy * dy;
        }
      }
      e.m = (long)(M.size() + i);
      for (size_t j = 0; j < M.size(); ++j) if (M[j] == L[i]) { e.m = (long)j; break; }
      v.push_back(e);
    }
    // (d, m) has no ties between distinct counties (m is unique per county)
    qsort(v.data(), v.size(), sizeof(E), [](const void* x, const void* y) {
      const E& a = *(const E*)x;
      const E& b = *(const E*)y;
      if (a.d != b.d) return a.d < b.d ? -1 : 1;
      return (a.m > b.m) - (a.m < b.m);
    });
    for (const E& e : v) c.push_back(e.s);
    // 1: below, above, rest; 2: above, below, rest; 3: only a county both agree on
    const long vz[2] = {anl4_o, anl4_u};
    for (long x : vz) {
      if (x < 0) continue;
      for (size_t j = 0; j < c.size(); ++j) {
        if (c[j] != x) continue;
        long* d = c.data();   // to the front
        memmove(d + 1, d, j * sizeof *d);
        d[0] = x;
        break;
      }
    }
    return c;
  }
  ENC_ONLY void KreisK(const std::string& pre, const std::vector<long>& c) {
    const std::string& s = Str(F_KREIS);
    long x;
    long r = -1;
    if (Ganz(s, 0, &x)) for (size_t i = 0; i < c.size(); ++i) if (c[i] == x) { r = (long)i; break; }
    if (r < 0) Tok3(pre, "#", s);
    else Tok3(pre, "", r ? Dez(r) : std::string());
  }
  ENC_ONLY static void KreisW(Run& L, const std::string& pre, const std::vector<long>& c) { L.KreisK(pre, c); }
  void Kreis() {
    if (!Has(F_KREIS)) return;
    Fmt fm;
    std::string pre, body;
    bool esc;
    if (!FieldHead(F_KREIS, &fm, &pre, &body, &esc)) return;
    if (halt_) { fehler_ = true; return; }   // preview stop: no estimate on this path
    const std::vector<long> c = KreisFolge();
    if (!esc) {
      if (enc_) {
        kodzweige.kreis(*this, pre, c);
      } else {
        long r = 0;
        if (!body.empty() && !Ganz(body, 0, &r)) { fehler_ = true; return; }
        if (r < 0 || r >= (long)c.size()) { fehler_ = true; return; }
        Str(F_KREIS) = Dez(c[(size_t)r]);
      }
    } else if (enc_) {
      Tok3(pre, "#", Str(F_KREIS));
    }
    ValueOf(F_KREIS);
    if (!Lernt() || !Ok(V_[F_KREIS]) || V_[F_KREIS] < 0) return;
    const long k = (long)V_[F_KREIS];
    if (Ok(lat_) && Ok(lon_)) {
      bool neu;
      const long zi = T_.kzi.Neu(k, &neu);
      if (neu) {
        const double z0 = 0.0;
        T_.kzw.push_back(z0);
        T_.kzw.push_back(z0);
        T_.kzw.push_back(z0);
      }
      double* w = &T_.kzw[(size_t)zi * 3];
      w[0] += lat_;
      w[1] += lon_;
      w[2] += 1.0;
    }
    if ((unsigned)staat_ >= (unsigned)kStaatenZ) return;   // KreisFolge reads states < kStaatenZ only
    std::vector<long>& M = T_.kmtf[staat_];   // move to front
    size_t j = 0;
    while (j < M.size() && M[j] != k) ++j;
    if (j == M.size()) M.push_back(k);
    long* d = M.data();
    memmove(d + 1, d, j * sizeof *d);
    d[0] = k;
  }

  // ---- the record, in the order of v2
  void Bevoelkerung() {
    double pop = kNan;
    if (Has(F_POP_I)) { StetigT(F_POP_I, F_POP_I, {}, T_LOG1P, 1.0); pop = V_[F_POP_I]; }
    if (Has(F_POP)) {
      if (Ok(pop)) RegelW(F_POP, pop, 1.0);
      else StetigT(F_POP, F_POP_I, {}, T_LOG1P, 1.0);
      if (!Ok(pop)) pop = V_[F_POP];
    }
    pop_ = pop;
    // population table: from the last row backwards; the row of 2000 repeats pop
    const int n = (int)R_.pvj.size();
    if (!n) return;
    std::vector<double> leer;
    std::vector<double>& prevg = folge_ || (unsigned)staat_ >= (unsigned)kStaatenZ ? leer : T_.prevg[staat_];
    std::vector<double> g_now((size_t)n, kNan);
    double nxt = kNan;
    for (int i = n - 1; i >= 0; --i) {
      pvs_ = R_.pvs[(size_t)i];
      if (R_.pvj[(size_t)i] == 2000 && Ok(pop)) {
        RegelW(F_PV, pop, 1.0);
      } else {
        const size_t k = (size_t)(n - 1 - i);
        const double gp = k < prevg.size() ? prevg[k] : kNan;
        Stetig(F_PV, F_PV, {L(nxt), Ok(gp) ? gp : 0.0, Ok(gp) ? 1.0 : 0.0, (double)(n - 1 - i) / 10.0},
               Tf(T_LOG1P), 1.0);
      }
      R_.pvs[(size_t)i] = pvs_;
      const double x = WertVon(F_PV);
      if (Ok(nxt) && Ok(x)) g_now[(size_t)(n - 1 - i)] = L(x) - L(nxt);
      nxt = x;
    }
    if (Lernt()) prevg.swap(g_now);
  }

  void Koordinaten() {
    const bool same = T_.prevstate == staat_;
    const int f2[2] = {F_LAT, F_LON};
    // without the decimal coordinate the Mapit value is the coordinate (v2)
    const int qu[2] = {Has(F_LAT) ? F_LAT : (Has(F_MLAT) ? F_MLAT : -1),
                       Has(F_LON) ? F_LON : (Has(F_MLON) ? F_MLON : -1)};
    for (int j = 0; j < 2; ++j) {
      const int f = qu[j];
      if (f < 0) continue;
      const double p = j == 0 ? T_.prev_lat : T_.prev_lon;
      StetigT(f, f2[j], {same && Ok(p) ? p : kNan, same ? 1.0 : 0.0}, T_ID, 0.0);
      if (!Lernt()) continue;
      if (j == 0) T_.prev_lat = V_[f]; else T_.prev_lon = V_[f];
    }
    // degrees, minutes, seconds from the decimal coordinate
    const int d3[2][3] = {{F_LAT_D, F_LAT_M, F_LAT_S}, {F_LON_D, F_LON_M, F_LON_S}};
    for (int j = 0; j < 2; ++j) {
      const double x = qu[j] >= 0 ? G(qu[j]) : kNan;
      double pr[3] = {kNan, kNan, kNan};
      if (Ok(x)) {
        const double a = fabs(x), dgr = floor(a), fr = a - dgr, mn = floor(fr * 60.0);
        double sc = Rund(fr * 3600.0 - mn * 60.0);
        if (sc > 59.0) sc = 59.0;
        pr[0] = dgr; pr[1] = mn; pr[2] = sc;
      }
      for (int k = 0; k < 3; ++k) if (Has(d3[j][k])) RegelW(d3[j][k], pr[k], 1.0);
    }
    if (Has(F_MLAT) && qu[0] != F_MLAT) Wiederholung(F_MLAT, F_LAT);
    if (Has(F_MLON) && qu[1] != F_MLON) Wiederholung(F_MLON, F_LON);
    lat_ = qu[0] >= 0 ? G(qu[0]) : kNan;
    lon_ = qu[1] >= 0 ? G(qu[1]) : kNan;
    if (!folge_) NbSetzen(lat_, lon_);
  }

  void Geografie() {
    const double pop = pop_, lpop = L(pop);
    const int wp = Has(F_WAT_PCT) ? F_WAT_PCT : (Has(F_WAT_PCT2) ? F_WAT_PCT2 : -1);
    double dmi = kNan, Lk_c = kNan;
    auto nz = [](double x) { return Ok(x) && x != 0.0; };
    if (cnty_) {
      if (Has(F_TOT_KM)) StetigT(F_TOT_KM, F_TOT_KM, {lpop}, T_LOG1P, 0.0);
      const double tk = G(F_TOT_KM);
      double p = kNan;
      if (wp >= 0) { StetigT(wp, F_WAT_PCT, {L(tk)}, T_LOGIT, 0.0, Gitter(), true); p = V_[wp]; }
      const double p0 = Ok(p) ? p : 0.0;
      if (Has(F_WAT_KM)) RegelRnd(F_WAT_KM, nz(tk) ? p0 / 100.0 * tk : kNan);
      const double wk = Has(F_WAT_KM) && Ok(V_[F_WAT_KM]) ? V_[F_WAT_KM] : 0.0;
      if (Has(F_LAND_KM)) RegelW(F_LAND_KM, nz(tk) ? tk - wk : kNan);
      const double lk = Has(F_LAND_KM) ? V_[F_LAND_KM] : tk;
      if (Has(F_TOT_MI)) RegelRnd(F_TOT_MI, nz(tk) ? tk / kKm2 : kNan);
      if (Has(F_LAND_MI)) RegelRnd(F_LAND_MI, nz(lk) ? lk / kKm2 : kNan);
      const double lmi = G(F_LAND_MI);
      if (Has(F_WAT_MI)) RegelRnd(F_WAT_MI, nz(tk) ? p0 / 100.0 * tk / kKm2 : kNan);
      if (Has(F_DENS_KM)) RegelRnd(F_DENS_KM, nz(pop) && nz(lk) ? pop / lk : kNan);
      if (Has(F_DENS_MI)) RegelRnd(F_DENS_MI, nz(pop) && nz(lmi) ? pop / lmi : kNan);
      dmi = G(F_DENS_MI);
      Lk_c = lk;
    } else {
      double Lm = kNan;
      if (Has(F_DENS_MI)) {
        StetigT(F_DENS_MI, F_DENS_MI, {lpop}, T_LOG1P, 0.0);
        dmi = V_[F_DENS_MI];
        if (nz(pop) && nz(dmi)) Lm = pop / dmi;
      }
      double L2 = Ok(Lm) ? Rnd(Lm, 2) : kNan;
      if (Has(F_DENS_KM)) {
        if (nz(pop) && nz(L2)) RegelRnd(F_DENS_KM, pop / (L2 * kKm2));
        else StetigT(F_DENS_KM, M_DENS_KM_ALT, {lpop}, T_LOG1P, 0.0);
      }
      if (!Ok(Lm) && Has(F_DENS_KM) && nz(pop) && nz(V_[F_DENS_KM])) {
        Lm = pop / V_[F_DENS_KM] / kKm2;
        L2 = Rnd(Lm, 2);
      }
      if (Has(F_LAND_MI)) RegelRnd(F_LAND_MI, nz(Lm) ? Lm : kNan);
      if (Has(F_LAND_KM)) RegelRnd(F_LAND_KM, nz(L2) ? L2 * kKm2 : kNan);
      double p = kNan;
      if (wp >= 0) { StetigT(wp, F_WAT_PCT, {L(Lm)}, T_LOGIT, 0.0, Gitter(), true); p = V_[wp]; }
      const double pp = Ok(p) ? p : 0.0;
      const double T = (nz(Lm) && pp < 100.0) ? Lm / (1.0 - pp / 100.0) : kNan;
      if (Has(F_TOT_MI)) RegelRnd(F_TOT_MI, T);
      if (Has(F_TOT_KM)) {
        const double pr = Ok(T) ? Rnd(T, 2) * kKm2 : kNan;
        if (!Ok(pr) && !nz(Lm)) StetigT(F_TOT_KM, M_TOT_KM_ALT, {lpop}, T_LOG1P, 0.0);
        else RegelRnd(F_TOT_KM, pr);
      }
      if (Has(F_WAT_MI)) RegelRnd(F_WAT_MI, Ok(T) ? T - Lm : kNan);
      if (Has(F_WAT_KM)) RegelRnd(F_WAT_KM, Ok(T) ? Rnd(T - Lm, 2) * kKm2 : kNan);
      Lk_c = nz(L2) ? L2 * kKm2 : kNan;
    }
    dmi_ = dmi;
    lkc_ = Lk_c;
    wv_ = wp >= 0 ? Lg(V_[wp]) : kNan;
    wp_ = wp;
  }

  void Rest() {
    const double pop = pop_, lpop = L(pop), dmi = dmi_, ldmi = L(dmi);
    auto nz = [](double x) { return Ok(x) && x != 0.0; };
    const bool hatwp = wp_ >= 0;
    // ---- households (places carry one more, always missing feature, as v2)
    double hh = kNan, fam = kNan, hu = kNan;
    if (Has(F_HH)) {
      const double f[] = {lpop, ldmi, hatwp ? 1.0 : 0.0, wv_, kNan};
      StetigT(F_HH, F_HH, Mk(f, cnty_ ? 4 : 5), T_LOG1P, 1.0);
      hh = V_[F_HH];
    }
    if (Has(F_HH2)) RegelW(F_HH2, hh, 1.0);
    if (Has(F_FAM)) {
      const double f[] = {lpop, L(hh), ldmi, kNan};
      StetigT(F_FAM, F_FAM, Mk(f, cnty_ ? 3 : 4), T_LOG1P, 1.0);
      fam = V_[F_FAM];
    }
    if (Has(F_HU)) {
      const double f0[] = {lpop, L(hh), L(fam), ldmi, hatwp ? 1.0 : 0.0, wv_, kNan};
      const Mk f(f0, cnty_ ? 6 : 7);
      if (nz(hh)) { Trafo t; t.art = T_HU; t.h = hh; Stetig(F_HU, F_HU, f, t, 1.0); }
      else StetigT(F_HU, M_HU_ALT, f, T_LOG1P, 1.0);
      hu = V_[F_HU];
    }
    if (Has(F_HUD_MI)) {
      double pr = kNan;
      if (cnty_) { const double lmi = G(F_LAND_MI); if (nz(hu) && nz(lmi)) pr = hu / lmi; }
      else if (nz(hu) && nz(dmi) && nz(pop)) pr = hu * dmi / pop;
      RegelRnd(F_HUD_MI, pr);
    }
    if (Has(F_HUD_KM)) RegelRnd(F_HUD_KM, nz(hu) && nz(lkc_) ? hu / lkc_ : kNan);
    // ---- race and hispanic
    double rl[6];
    int nrl = 0;
    double cnt = 0.0;
    bool allg = true;
    for (int i = 0; i < 6; ++i) {
      const int f = F_R0 + i;
      if (!Has(f)) continue;
      double ft[8];
      int k = 0;
      ft[k++] = lpop;
      for (int j = 0; j < i; ++j) ft[k++] = LgF(F_R0 + j);
      ft[k++] = ldmi;
      StetigT(f, f, Mk(ft, k), T_LOGIT, 0.0, Gt('p', pop), true);
      rl[nrl++] = V_[f];
      const double q = Q(T_.fmt[f]);
      if (nz(pop) && 100.0 / pop > q) cnt += Rund(V_[f] * pop / 100.0);
      else allg = false;
    }
    if (Has(F_R6)) RegelRest(F_R6, pop, cnt, allg, rl, nrl, false);
    if (Has(F_HISP))
      StetigT(F_HISP, F_HISP, {lpop, LgF(F_R0), LgF(F_R5), LgF(F_R1), LgF(F_R2), ldmi}, T_LOGIT, 0.0,
             Gt('p', pop), true);
    // ---- household composition
    const double fr = (nz(fam) && nz(hh)) ? Lg(100.0 * fam / hh) : kNan;
    const double ph = (nz(pop) && nz(hh)) ? L(pop / hh) : kNan;
    const Gitter gh = Gt('p', hh);
    const double lhh = L(hh);
    if (Has(F_HH_KID)) StetigT(F_HH_KID, F_HH_KID, {lhh, fr, ph, LgF(F_R0), LgF(F_HISP)}, T_LOGIT, 0.0, gh, true);
    if (Has(F_HH_MAR))
      StetigT(F_HH_MAR, F_HH_MAR, {lhh, fr, LgF(F_HH_KID), ph, LgF(F_R1), LgF(F_R0)}, T_LOGIT, 0.0, gh, true);
    if (Has(F_HH_FEM))
      StetigT(F_HH_FEM, F_HH_FEM, {lhh, fr, LgF(F_HH_MAR), LgF(F_HH_KID), LgF(F_R1), ph}, T_LOGIT, 0.0, gh,
             true);
    if (Has(F_HH_NF))   // special value 100 (1 - fam / hh) rounded to the decimals (1 with the County padding)
      Stetig(F_HH_NF, F_HH_NF, {lhh, LgF(F_HH_MAR), LgF(F_HH_FEM)}, Tf(T_LOGIT), 0.0, gh, false, kNan,
             (nz(fam) && nz(hh)) ? fam / hh : kNan);
    if (Has(F_HH_IND))
      StetigT(F_HH_IND, F_HH_IND, {lhh, LgF(F_HH_NF), LgF(F_HH_MAR), ph, LgF(F_HH_KID)}, T_LOGIT, 0.0, gh,
             true);
    if (Has(F_HH_65))
      StetigT(F_HH_65, F_HH_65, {lhh, LgF(F_HH_IND), LgF(F_HH_KID), LgF(F_HH_MAR), ph}, T_LOGIT, 0.0, gh,
             true);
    // ---- sizes
    double phh = kNan;
    if (Has(F_HHS)) {
      const bool ok = nz(pop) && nz(hh);
      const double sp = ok ? Rnd(pop / hh, 2) : kNan;
      Stetig(F_HHS, F_HHS,
             {lpop, LgF(F_HH_KID), LgF(F_HH_IND), LgF(F_HH_MAR), (nz(hu) && ok) ? L(hu / hh) : kNan},
             Tf(T_LOG, ok ? log(pop / hh) : 0.0), 0.0, Gt('r', hh), false, sp);
      phh = (Ok(sp) && fabs(V_[F_HHS] - sp) < 0.005) ? pop : (Ok(hh) ? V_[F_HHS] * hh : kNan);
    }
    if (Has(F_FS)) {
      double fsmax = kNan, alone = 0.0, nonf = 0.0;
      if (Ok(phh) && nz(fam) && nz(hh) && Ok(G(F_HH_IND))) {
        alone = Rund(V_[F_HH_IND] * hh / 100.0);
        nonf = hh - fam;
        fsmax = (phh - alone - 2.0 * (nonf - alone > 0 ? nonf - alone : 0.0)) / fam;
      }
      const bool ok = Ok(fsmax) && fsmax > 0.5;
      const double hv = G(F_HHS);
      if (ok) {
        const double multi = nonf - alone > 0 ? nonf - alone : 0.0;
        Stetig(F_FS, M_FS2, {multi / fam, 1.0 / fam, G(F_HH_KID), G(F_HH_FEM), G(F_HH_65), G(F_HH_MAR), G(F_HH_IND), hv, lpop},
               Tf(T_ID, fsmax), 0.0, Gt('r', fam));
      } else {
        Stetig(F_FS, F_FS,
               {(nz(hh) && nz(fam)) ? L(hh / fam) : kNan, LgF(F_HH_KID), lpop, LgF(F_HH_FEM), kNan, 0.0},
               Tf(T_LOG, Ok(hv) && hv > 0 ? log(hv) + 0.2 : 0.0), 0.0, Gt('r', fam));
      }
    }
    // ---- age
    double al[4];
    int nal = 0;
    cnt = 0.0;
    allg = true;
    const double hhx[6] = {LgF(F_HH_KID), LF(F_HHS), LF(F_FS), LgF(F_HH_65), LgF(F_HH_IND),
                           LgF(F_HH_MAR)};
    for (int i = 0; i < 4; ++i) {
      const int f = F_A0 + i;
      if (!Has(f)) continue;
      double ft[12];
      int k = 0;
      ft[k++] = lpop;
      for (int j = 0; j < 6; ++j) ft[k++] = hhx[j];
      for (int j = 0; j < i; ++j) ft[k++] = LgF(F_A0 + j);
      StetigT(f, f, Mk(ft, k), T_LOGIT, 0.0, Gt('p', pop), true);
      al[nal++] = V_[f];
      const double q = Q(T_.fmt[f]);
      if (nz(pop) && 100.0 / pop > q) cnt += Rund(V_[f] * pop / 100.0);
      else allg = false;
    }
    if (Has(F_A4)) RegelRest(F_A4, pop, cnt, allg, al, nal, true);
    if (Has(F_MED)) {
      double a[5];
      bool alle = true;
      for (int i = 0; i < 5; ++i) { a[i] = G(F_A0 + i); alle = alle && Ok(a[i]); }
      double mi = kNan, mq = kNan, u = kNan;
      int jb = 5;
      if (alle) {
        static const double lo[5] = {0, 18, 25, 45, 65}, hi[5] = {18, 25, 45, 65, 90};
        double cum = 0.0;
        for (int jj = 0; jj < 5; ++jj) {
          if (cum + a[jj] >= 50.0 && a[jj] > 0.0) { u = (50.0 - cum) / a[jj]; mi = lo[jj] + u * (hi[jj] - lo[jj]); jb = jj; break; }
          cum += a[jj];
        }
        static const double xs[6] = {0, 18, 25, 45, 65, 90};
        double cs[6];
        cs[0] = 0.0;
        for (int i = 0; i < 5; ++i) cs[i + 1] = cs[i] + a[i];
        int j = -1;
        for (int i = 1; i < 6; ++i) if (cs[i] >= 50.0) { j = i; break; }
        if (j >= 0) {
          int i0 = j - 1 < 0 ? 0 : j - 1;
          if (i0 > 3) i0 = 3;
          const double x0 = xs[i0], x1 = xs[i0 + 1], x2 = xs[i0 + 2], y0 = cs[i0], y1 = cs[i0 + 1], y2 = cs[i0 + 2];
          const double d0 = (y0 - y1) * (y0 - y2), d1 = (y1 - y0) * (y1 - y2), d2 = (y2 - y0) * (y2 - y1);
          if (d0 != 0.0 && d1 != 0.0 && d2 != 0.0)
            mq = x0 * (50 - y1) * (50 - y2) / d0 + x1 * (50 - y0) * (50 - y2) / d1 + x2 * (50 - y0) * (50 - y1) / d2;
        }
      }
      if (!Ok(mi)) {
        StetigT(F_MED, F_MED, {mi, mq, Lg(a[0]), Lg(a[1]), Lg(a[2]), Lg(a[3]), Lg(a[4]), LF(F_HHS), LgF(F_HH_65)},
               T_ID, 0.0);
      } else {
        Stetig(F_MED, M_MED2,
               {Ok(mq) ? mq - mi : kNan, u, u * u, mi, jb == 1 ? 1.0 : 0.0, jb == 3 ? 1.0 : 0.0, Lg(a[0]), Lg(a[1]),
                Lg(a[2]), Lg(a[3]), Lg(a[4]), LF(F_HHS), LgF(F_HH_65)},
               Tf(T_ID, mi), 0.0);
      }
    }
    if (Has(F_M100))
      StetigT(F_M100, F_M100, {lpop, LgF(F_A4), LgF(F_A0), LgF(F_A1), LgF(F_A2), LF(F_HHS), LgF(F_HH_MAR)},
             T_LOG, 0.0, Gt('m', Ok(pop) ? floor(pop) : kNan));
    if (Has(F_M18)) {
      const double A = (nz(pop) && Ok(G(F_A0))) ? Rund(pop - pop * V_[F_A0] / 100.0) : kNan;
      const double mv = G(F_M100);
      Stetig(F_M18, F_M18, {lpop, LgF(F_A4), LgF(F_A1), LgF(F_A0)}, Tf(T_LOG, Ok(mv) && mv > 0 ? log(mv) : 0.0),
             0.0, Gt('m', A));
    }
    // ---- income (per capita first), then poverty
    static const int INC[5] = {F_INC_PC, F_INC_HH, F_INC_FAM, F_INC_M, F_INC_F};
    for (int i = 0; i < 5; ++i) {
      const int f = INC[i];
      if (!Has(f)) continue;
      double ft[20] = {lpop, LgF(F_HH_MAR), LgF(F_HH_FEM), LgF(F_HH_KID), LgF(F_HH_65), LgF(F_A4),
                       LgF(F_A1), LgF(F_A2), LF(F_HHS), LF(F_MED), LgF(F_R0), LgF(F_R1),
                       LgF(F_HISP), ldmi, (nz(hu) && nz(hh)) ? L(hu / hh) : kNan, LF(F_M100)};
      int k = 16;
      for (int j = 0; j < i; ++j) ft[k++] = LF(INC[j]);
      Einkommen(f, Mk(ft, k), f == F_INC_PC);
    }
    static const int POV[4] = {F_POV, F_POV_FAM, F_POV_U18, F_POV_65};
    for (int i = 0; i < 4; ++i) {
      const int f = POV[i];
      if (!Has(f)) continue;
      double ft[16] = {lpop, LF(F_INC_HH), LF(F_INC_PC), LF(F_INC_FAM), LgF(F_HH_FEM), LgF(F_HH_MAR),
                       LgF(F_R1), LgF(F_HISP), LgF(F_A4), LgF(F_A1), LgF(F_HH_KID), LF(F_HHS),
                       LgF(F_A0)};
      int k = 13;
      for (int j = 0; j < i; ++j) ft[k++] = LgF(POV[j]);
      StetigT(f, f, Mk(ft, k), T_LOGIT, 0.0, Gitter(), true);
    }
  }
};

}  // namespace zahl
}  // namespace census_block

#if defined(__clang__)
#pragma float_control(pop)
#endif
#endif
