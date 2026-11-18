
#ifndef ORDER_CODEC_H_
#define ORDER_CODEC_H_

#ifndef ORD_V3
#define ORD_V3 1
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../preprocess/dict_derive.h"   // SortU64, SortByScore, ReadVarint
#include "../preprocess/hd_cm.h"

// Article order as rule plus residual.  From features of every page (title
// tokens, wikilinks, content words of the lead, categories) a deterministic
// cluster vote predicts the article order; the residual file (the compressor's
// ".new_article_order" block) corrects it.  Used by the compressor only: the
// decompressor restores the original order by sorting on the page id.
// The base order is the fx2-cmix article order (Byron Knoll; inherited
// through cmix-lex in the revision of Kaido Orav's fxcm_v26 archive).  The
// page classes and their detection strings follow the manual sort step of
// that pipeline (src/article_order/manual-sort.ipynb in fx2-cmix): census,
// image, disambiguation, geographic and personal-name disambiguation pages
// are moved to the end.
// Plain arrays instead of std containers: the code runs once per run, and the
// template instances cost more than the code itself in the packed binary.
namespace ord {

// The whole file in one malloc'd buffer; 0 if it cannot be read.
static uint8_t* ReadFileBytes(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) return 0;
  fseek(f, 0, SEEK_END); long m = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t* b = m < 0 ? 0 : (uint8_t*)malloc((size_t)m + 1);
  if (b && fread(b, 1, (size_t)m, f) != (size_t)m) { free(b); b = 0; }
  fclose(f);
  *n = (size_t)m;
  return b;
}

// ------------------------------------------------ integer fixed-point ln
// round(2^57 * ln(x)) for integer x >= 1, pure integer arithmetic.
static int64_t IlnQ57(uint64_t x) {
  static const int64_t LN2_Q57 = 99893036290645744LL;
  static const int64_t LN1P_Q62[53] = { 0LL,
    1869877770022571008LL,1029067995697975552LL,543178378744478080LL,279581720803300544LL,
    141909228037287936LL,71500440275001120LL,35888788270063140LL,17979305495762816LL,
    8998414598213862LL,4501402034722952LL,2251250236762820LL,1125762490254678LL,
    562915596478891LL,281466387125573LL,140735340915370LL,70368207312213LL,
    35184237871787LL,17592152490069LL,8796084633611LL,4398044413953LL,
    2199022731264LL,1099511496704LL,549755781120LL,274877898752LL,
    137438951424LL,68719476224LL,34359738240LL,17179869152LL,
    8589934584LL,4294967294LL,2147483648LL,1073741824LL,
    536870912LL,268435456LL,134217728LL,67108864LL,
    33554432LL,16777216LL,8388608LL,4194304LL,
    2097152LL,1048576LL,524288LL,262144LL,
    131072LL,65536LL,32768LL,16384LL,
    8192LL,4096LL,2048LL,1024LL };
  int k = 63; while (!((x >> k) & 1)) --k;          // floor(log2 x)
  uint64_t m = x << (62 - k);                       // mantissa in Q62, [1,2)
  uint64_t r = 1ULL << 62; int64_t acc = 0;
  for (int j = 1; j <= 52; ++j) {
    uint64_t t = r + (r >> j);
    if (t <= m) { r = t; acc += LN1P_Q62[j]; }
  }
  return (int64_t)k * LN2_Q57 + ((acc + 16) >> 5);  // Q62 -> Q57
}
// IDF = round(256*ln(n/df)) via Q57 difference (256/2^57 = 2^-49).
static int64_t Idf(uint64_t n, uint64_t df) {
  int64_t d = IlnQ57(n) - IlnQ57(df);
  return (d + (1LL << 48)) >> 49;
}

// -------------------------------------------------------- feature hash
static uint64_t Fnv1a(uint8_t type, const char* s, size_t len) {
  uint64_t h = 1469598103934665603ULL ^ type;
  h *= 1099511628211ULL;
  for (size_t i = 0; i < len; ++i) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
  return h;
}
static void StripSpan(const char*& a, const char*& b) {   // ASCII-strip
  while (a < b && (uint8_t)*a <= ' ') ++a;
  while (b > a && (uint8_t)b[-1] <= ' ') --b;
}

// locale-free ASCII classes (identical to isalpha/isalnum under LC_ALL=C;
// the compressor never calls setlocale, but determinism is not guessed at).
static inline bool IsAlphaA(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static inline bool IsAlnumA(char c) { return IsAlphaA(c) || (c >= '0' && c <= '9'); }
static const size_t ORD_LEADCHARS = 600;
static const int ORD_MAXCW = 160;           // content words per page

// The opening of the page text (33 bytes) and the redirect marks behind it
// (one array with its own section, read by the encoder only).
#define ORD_TEXT_OPEN "      <text xml:space=\"preserve\">"
static const char kRedir[] = "#REDIRECT\0#redirect\0#Redirect\0#REdirect\0{{softredirect";
// The census sentences of Byron Knoll's manual sort step (fx2-cmix), one
// NUL-separated array.
static const char kCen[] =
  "According to the [[United States Census Bureau]],\0"
  "According to the [[U.S. Census Bureau]],\0"
  "As of the [[2000]] census,\0"
  "According to the [[2000]] census, \0"
  "According to the [[United States Census Bureau|U.S. Census Bureau]],\0"
  "According to the [[US Census Bureau]], \0"
  "According to the U.S. Census,\0"
  "As of the 2000 \0"
  "As of the census of 2000,";

// true if the line opens the page text with a redirect mark.
static bool IsRedirLine(const char* s, size_t n) {
  if (n < 33 || memcmp(s, ORD_TEXT_OPEN, 33)) return false;
  for (const char* w = kRedir; w < kRedir + sizeof kRedir; ) {
    const size_t l = strlen(w);
    if (n - 33 >= l && !memcmp(s + 33, w, l)) return true;
    w += l + 1;
  }
  return false;
}

static const char kKalTxt[] =
  "January\0February\0March\0April\0May\0June\0July\0August\0September\0"
  "October\0November\0December\0Deaths in \0Births in \0 century\0 millennium";
static bool KalNum(const char* p, size_t m, int64_t* v) {
  if (m == 0 || m > 6) return false;
  int64_t x = 0;
  for (size_t i = 0; i < m; ++i) {
    if (p[i] < '0' || p[i] > '9') return false;
    x = x * 10 + (p[i] - '0');
  }
  *v = x;
  return true;
}
static inline int64_t KalMk(int64_t g, int64_t v) {
  return (g << 32) | (int64_t)(uint32_t)(v + (1 << 30));
}
static int64_t KalKey(const char* s, size_t n) {
  int64_t v;
  const char* w = kKalTxt;
  for (int mo = 0; mo < 12; ++mo) {                  // month, day, month-year
    const size_t l = strlen(w);
    if (n >= l && memcmp(s, w, l) == 0) {
      if (n == l) return KalMk(5, mo);
      if (s[l] != ' ' || !KalNum(s + l + 1, n - l - 1, &v)) return -1;
      if (n - l - 1 <= 2) return KalMk(4, (mo + 1) * 32 + v);
      if (n - l - 1 == 4) return KalMk(6, v * 16 + mo);
      return -1;
    }
    w += l + 1;
  }
  // w: "Deaths in ", w+11: "Births in ", w+22: " century", w+31: " millennium"
  if (n > 10 && (memcmp(s, w, 10) == 0 || memcmp(s, w + 11, 10) == 0))
    return (n <= 14 && KalNum(s + 10, n - 10, &v)) ? KalMk(7, 2 * v + (s[0] == 'B')) : -1;
  const bool bc = n > 3 && s[n - 3] == ' ' && s[n - 2] == 'B' && s[n - 1] == 'C';
  const size_t m = bc ? n - 3 : n;
  size_t i = 0;
  while (i < m && s[i] >= '0' && s[i] <= '9') ++i;
  if (i == 0 || !KalNum(s, i, &v)) return -1;
  const int64_t sv = bc ? -2 * v - 1 : 2 * v;
  if (i == m) return i <= 4 ? KalMk(0, sv) : -1;     // year
  if (i + 1 == m && s[i] == 's' && s[i - 1] == '0') return KalMk(1, sv);   // decade
  if (m - i > 2 && ((s[i] == 's' && s[i + 1] == 't') || (s[i] == 'n' && s[i + 1] == 'd') ||
                    (s[i] == 'r' && s[i + 1] == 'd') || (s[i] == 't' && s[i + 1] == 'h'))) {
    const char* r = s + i + 2;
    const size_t rl = m - i - 2;
    if (rl == 8 && memcmp(r, w + 22, 8) == 0) return KalMk(2, sv);
    if (rl == 11 && memcmp(r, w + 31, 11) == 0) return KalMk(3, sv);
  }
  return -1;
}

struct PageScan {
  bool is_redirect = false;
  int klass = 0;                        // 0 normal, 1 census .. 5 hndis
  int64_t kal = -1;                     // KalKey of the title, -1 = none
};

// The features of all pages go into ONE growing array: those of the current
// page from p0 on.  At the page end they are sorted and deduplicated there;
// the caller keeps them (p0 = fn) or drops them (fn = p0).
struct Extractor {
  // state of the current page
  bool in_page = false, in_text = false;
  bool f_cen = false, f_dis = false, f_geo = false, f_hnd = false;
  bool f_img = false, f_distitle = false, redirect = false;
  int n_links = 0;
  uint64_t* fa = 0;             // features
  size_t fn = 0, fcap = 0, p0 = 0;
  int n_cw = 0;                 // content words in the lead (cap ORD_MAXCW)
  size_t lead_used = 0;         // body characters consumed
  int64_t kal = -1;             // KalKey of the title of the current page

  void Push(uint64_t h) {
    if (fn == fcap) { fcap = fcap * 2 + 65536; fa = (uint64_t*)realloc(fa, fcap * 8); }
    fa[fn++] = h;
  }

  void Title(const char* a, const char* b) {           // title bytes
    size_t n = (size_t)(b - a);
    kal = KalKey(a, n);
    if (n >= 6 && memcmp(a, "Image:", 6) == 0) f_img = true;
    for (size_t i = 0; i + 16 <= n; ++i)
      if (memcmp(a + i, "(disambiguation)", 16) == 0) { f_distitle = true; break; }
    // channel 'W': title tokens (alnum runs, lowercase, len>=2)
    {
      char buf[64];
      size_t i = 0;
      while (i < n) {
        while (i < n && !IsAlnumA(a[i])) ++i;
        size_t s = i;
        while (i < n && IsAlnumA(a[i])) ++i;
        size_t len = i - s;
        if (len >= 2) {
          if (len > 63) len = 63;
          for (size_t t = 0; t < len; ++t) {
            char c = a[s + t];
            buf[t] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
          }
          Push(Fnv1a('W', buf, len));
        }
      }
    }
  }

  // channels 'l' (wikilinks) and 'w' (content words len>=5) from the first
  // ORD_LEADCHARS characters of the body text.
  void LeadLine(const char* s, size_t n) {
    if (lead_used >= ORD_LEADCHARS) return;
    size_t budget = ORD_LEADCHARS - lead_used;
    size_t m = (n < budget) ? n : budget;
    for (size_t i = 0; i + 2 <= m; ) {
      const char* hit = (const char*)memmem(s + i, m - i, "[[", 2);
      if (!hit) break;
      size_t a = (size_t)(hit - s) + 2, b = a;
      while (b < m && b - a < 81 && s[b] != ']' && s[b] != '|' && s[b] != ':' && s[b] != '#') ++b;
      size_t rl = b - a;
      if (rl >= 1 && rl <= 80 && b < m && (s[b] == ']' || s[b] == '|')) {
        const char* x = s + a; const char* y = s + b;
        StripSpan(x, y);
        size_t len = (size_t)(y - x);
        if (len) { if (len > 60) len = 60; Push(Fnv1a('l', x, len)); }
        i = b + 1;
      } else i = a - 1;
    }
    char buf[64];
    size_t i = 0;
    while (i < m && n_cw < ORD_MAXCW) {
      while (i < m && !IsAlphaA(s[i])) ++i;
      size_t st = i;
      while (i < m && IsAlphaA(s[i])) ++i;
      size_t len = i - st;
      if (len >= 5) {
        if (len > 63) len = 63;
        for (size_t t = 0; t < len; ++t) {
          char c = s[st + t];
          buf[t] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
        }
        Push(Fnv1a('w', buf, len));
        ++n_cw;
      }
    }
    lead_used += m;
  }

  // The census sentences (kCen) and the disambiguation templates below are
  // the detection strings of Byron Knoll's manual sort step (fx2-cmix).
  void FlagsLine(const char* s, size_t n) {            // body text only
    if (!f_cen)
      for (const char* w = kCen; w < kCen + sizeof kCen; ) {
        const size_t l = strlen(w);
        if (memmem(s, n, w, l)) { f_cen = true; break; }
        w += l + 1;
      }
    if (!f_hnd && memmem(s, n, "{{hndis", 7)) f_hnd = true;
    if (!f_geo && memmem(s, n, "{{geodis}}", 10)) f_geo = true;
    if (!f_dis) {                                      // \{\{[Dd]isam[^}]*\}\}
      for (size_t i = 0; i + 7 <= n; ++i) {
        if (s[i] == '{' && s[i+1] == '{' && (s[i+2] == 'D' || s[i+2] == 'd')
            && i + 6 <= n && memcmp(s + i + 3, "isam", 4) == 0) {
          size_t j = i + 7;
          while (j < n && s[j] != '}') ++j;
          if (j + 1 < n && s[j] == '}' && s[j+1] == '}') { f_dis = true; break; }
          i = j;
        }
      }
    }
  }

  void FeatsLine(const char* s, size_t n) {            // all page lines
    // categories: \[\[Category:([^\]\|#]{1,120})
    for (size_t i = 0; i + 11 <= n; ) {
      const char* hit = (const char*)memmem(s + i, n - i, "[[Category:", 11);
      if (!hit) break;
      size_t a = (size_t)(hit - s) + 11, b = a;
      while (b < n && b - a < 120 && s[b] != ']' && s[b] != '|' && s[b] != '#') ++b;
      if (b > a) {
        const char* x = s + a; const char* y = s + b;
        StripSpan(x, y);
        Push(Fnv1a('C', x, (size_t)(y - x)));
      }
      i = (b > a) ? b : a;
    }
    // Templates: \{\{([A-Za-z][^\}\|\n]{0,60})  -> strip, lower, split('{')[0], [:40]
    for (size_t i = 0; i + 3 <= n; ) {
      const char* hit = (const char*)memmem(s + i, n - i, "{{", 2);
      if (!hit) break;
      size_t a = (size_t)(hit - s) + 2;
      if (a >= n || !((s[a] >= 'A' && s[a] <= 'Z') || (s[a] >= 'a' && s[a] <= 'z'))) {
        i = (size_t)(hit - s) + 1;                     // like re.finditer: next start position
        continue;
      }
      size_t b = a + 1;
      while (b < n && b - a < 61 && s[b] != '}' && s[b] != '|') ++b;
      const char* x = s + a; const char* y = s + b;
      StripSpan(x, y);
      const char* z = x;                               // split('{')[0]
      while (z < y && *z != '{') ++z;
      size_t len = (size_t)(z - x);
      if (len > 40) len = 40;
      if (len) {
        char buf[40];
        for (size_t t = 0; t < len; ++t) {
          char c = x[t];
          buf[t] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
        }
        Push(Fnv1a('T', buf, len));
      }
      i = b;
    }
    // wikilinks: \[\[([^\]\|:#\n]{1,80})[\]\|]  -> strip, [:60], max 200/page
    for (size_t i = 0; i + 2 <= n && n_links < 200; ) {
      const char* hit = (const char*)memmem(s + i, n - i, "[[", 2);
      if (!hit) break;
      size_t a = (size_t)(hit - s) + 2, b = a;
      while (b < n && b - a < 81 && s[b] != ']' && s[b] != '|' && s[b] != ':' && s[b] != '#') ++b;
      size_t rl = b - a;
      if (rl >= 1 && rl <= 80 && b < n && (s[b] == ']' || s[b] == '|')) {
        const char* x = s + a; const char* y = s + b;
        StripSpan(x, y);
        size_t len = (size_t)(y - x);
        if (len) {
          if (len > 60) len = 60;
          Push(Fnv1a('L', x, len));
          ++n_links;
        }
        i = b + 1;
      } else {
        i = a - 1;                                     // continue past first '['
      }
    }
  }

  // processes one line s[0..n); returns true when a page has been completed.
  bool Line(const char* s, size_t n, PageScan* out) {
    if (n >= 8 && memcmp(s, "  <page>", 8) == 0) {
      in_page = true; in_text = false;
      f_cen = f_dis = f_geo = f_hnd = f_img = f_distitle = redirect = false;
      n_links = 0; fn = p0;
      n_cw = 0; lead_used = 0;
      kal = -1;
      return false;
    }
    if (!in_page) return false;
    if (n >= 9 && memcmp(s, "  </page>", 9) == 0) {
      in_page = false;
      out->is_redirect = redirect;
      out->kal = kal;
      int k = 0;
      bool dis = f_distitle || f_dis || f_hnd || f_geo;
      if (f_cen) k = 1;
      if (f_img) k = 2;
      if (dis)   k = 3;
      if (f_geo) k = 4;
      if (f_hnd) k = 5;
      out->klass = k;
      dict_derive::SortU64(fa + p0, fn - p0);          // sorted, deduplicated
      size_t w = p0;
      for (size_t i = p0; i < fn; ++i)
        if (w == p0 || fa[i] != fa[w - 1]) fa[w++] = fa[i];
      fn = w;
      return true;
    }
    if (IsRedirLine(s, n)) redirect = true;
    if (const char* t = (const char*)memmem(s, n, "<title>", 7)) {
      const char* e = (const char*)memmem(t + 7, n - (size_t)(t + 7 - s), "</title>", 8);
      if (e) Title(t + 7, e);
    }
    bool text_open = n >= 33 && memcmp(s, ORD_TEXT_OPEN, 33) == 0;
    if (text_open) in_text = true;
    if (in_text) {
      FlagsLine(s, n);
      {
        const char* e = (const char*)memmem(s, n, "</text>", 7);
        size_t from = text_open ? 33 : 0;
        size_t to = e ? (size_t)(e - s) : n;
        if (to > from) LeadLine(s + from, to - from);
        if (e) in_text = false;
      }
    }
    FeatsLine(s, n);
    return false;
  }
};

static const int   ORDV3_DFMAX = 175;
static const int   ORDV3_AXISK = 8;      // axis neighbors per side
static const int   ORDV3_AXISW = 2;      // smoothing weight in 1/16
static const int   ORDV3_NORMK = 16;
static const int64_t ORDV3_R_ESC = 4096;
static const int64_t ORDV4_TAG = 0x1FFFFE;
struct Residual {
  int32_t* excl = 0;                               // unit IDs, ascending
  int32_t* corr = 0;                               // (unit ID, class) pairs, ascending
  int32_t* starts = 0;                             // cluster sequence
  int64_t ne = 0, nc = 0, nr = 0;
  int64_t n_split = 0;
  int64_t n_kal = 0;
  uint8_t* buf = 0;                                // the file
  size_t len = 0;
  size_t hdr = 0;
};

// The v3 header with n_split between corr and n_runs.  Varints of the
// dictionary derivation (one reader); a count beyond the file length is a
// broken file.
static bool ParseResidualV3(const char* path, Residual* r) {
  using dict_derive::ReadVarint;
  uint8_t* b = r->buf = ReadFileBytes(path, &r->len);
  if (!b) return false;
  const size_t n = r->len;
  size_t q = 0;
  const uint64_t ne = ReadVarint(b, q);
  if (ne > n) return false;
  r->excl = (int32_t*)malloc(4 * ne + 4); r->ne = (int64_t)ne;
  int64_t last = -1;
  for (uint64_t i = 0; i < ne; ++i) { last += (int64_t)ReadVarint(b, q); r->excl[i] = (int32_t)last; }
  const uint64_t nc = ReadVarint(b, q);
  if (nc > n) return false;
  r->corr = (int32_t*)malloc(8 * nc + 8); r->nc = (int64_t)nc;
  last = -1;
  for (uint64_t i = 0; i < nc; ++i) {
    last += (int64_t)ReadVarint(b, q);
    r->corr[2 * i] = (int32_t)last;
    r->corr[2 * i + 1] = (uint8_t)ReadVarint(b, q);
  }
  r->n_split = (int64_t)ReadVarint(b, q);
  r->n_kal = (int64_t)ReadVarint(b, q);
  const int64_t nr = (int64_t)ReadVarint(b, q);
  if (nr <= 0 || nr > (1 << 20) || (int64_t)ReadVarint(b, q) != ORDV4_TAG) return false;
  r->starts = (int32_t*)calloc((size_t)nr, 4); r->nr = nr;
  r->hdr = q;
  return q <= n;
}

// vote of one article: fills cltouch/clscore (score order = rank).
// Encoder and decoder call the same function - the roundtrip is thus
// consistent by construction.
struct ORDV3Vote {
  int32_t nrun = 0;
  int32_t nt = 0;                        // number of voted clusters in cltouch
  int64_t* clscore;                      // the int64 block: clscore, smear
  int64_t* smear;
  int32_t* clstamp;                      // the int32 block: clstamp, smstamp,
  int32_t* smstamp;                      //   axis_of_seq, fill, cltouch
  int32_t* axis_of_seq;                  // seq id -> axis rank
  int32_t* fill;
  int32_t* cltouch;
  const int32_t* seq_of_axis;            // axis rank -> seq id

  bool Init(int32_t nr, const int32_t* starts) {
    nrun = nr;
    const size_t m = (size_t)nrun + 2;
    clscore = (int64_t*)calloc(2 * m, 8);
    smear = clscore + m;
    clstamp = (int32_t*)calloc(5 * m, 4);
    memset(clstamp, 0xff, 12 * m);       // clstamp, smstamp, axis_of_seq = -1
    smstamp = clstamp + m;
    axis_of_seq = smstamp + m;
    fill = axis_of_seq + m;
    cltouch = fill + m;
    for (int32_t a = 0; a < nrun; ++a) {
      int32_t s = starts[a];
      if (s < 0 || s >= nrun || axis_of_seq[s] != -1) return false;
      axis_of_seq[s] = a;
    }
    seq_of_axis = starts;
    return true;
  }

  // t: dense id of the current article; nfeat[0..nn): its feature indices;
  // post[poff[f]..poff[f+1]): dense ids with feature f, ascending;
  // clus_of_dense: seq clusters of the already decoded articles; n_seen: number
  // of seq ids handed out so far.
  void Score(int32_t t, const uint64_t* nfeat, size_t nn, const int32_t* post,
             const uint32_t* poff, const int64_t* idf,
             const int32_t* clus_of_dense, int32_t n_seen) {
    nt = 0;
    for (size_t i = 0; i < nn; ++i) {
      int32_t f = (int32_t)nfeat[i];
      int64_t w = idf[f];
      if (w <= 0) continue;
      for (uint32_t k = poff[f]; k < poff[f + 1]; ++k) {
        int32_t j = post[k];
        if (j >= t) break;
        int32_t cs = clus_of_dense[j];
        if (clstamp[cs] != t) {
          clstamp[cs] = t; clscore[cs] = 0; cltouch[nt++] = cs;
        }
        clscore[cs] += w;
      }
    }
    // axis smoothing: smear a cluster's own score onto its axis neighbors.
    // Only the originally voted clusters are sources (prefix of cltouch);
    // newly touched neighbors are appended at the end.
    if (nt) {
      const int32_t nbase = nt;
      for (int32_t bi = 0; bi < nbase; ++bi) {
        int32_t cs = cltouch[bi];
        int32_t a = axis_of_seq[cs];
        int64_t w = (clscore[cs] * ORDV3_AXISW) >> 4;
        for (int d = 1; d <= ORDV3_AXISK && w > 0; ++d, w >>= 1) {
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            int32_t ax = a + sgn * d;
            if (ax < 0 || ax >= nrun) continue;
            int32_t nb = seq_of_axis[ax];
            if (nb >= n_seen) continue;              // cluster not seen yet
            if (clstamp[nb] != t) {
              clstamp[nb] = t; clscore[nb] = 0; cltouch[nt++] = nb;
            }
            if (smstamp[nb] != t) { smstamp[nb] = t; smear[nb] = 0; }
            smear[nb] += w;
          }
        }
      }
      for (int32_t k = 0; k < nt; ++k) {
        int32_t cs = cltouch[k];
        if (smstamp[cs] == t) { clscore[cs] += smear[cs]; smstamp[cs] = -1; }
      }
    }
    // score normalization by cluster fill level (integer division)
    for (int32_t k = 0; k < nt; ++k) {
      int32_t cs = cltouch[k];
      clscore[cs] = clscore[cs] * 64 / (fill[cs] + ORDV3_NORMK);
    }
  }
  void Commit(int32_t cl) { ++fill[cl]; }
  void Recent(int32_t t, const int32_t* clus_of_dense) {
    if (t == 0) return;
    int64_t mx = 0;
    for (int32_t k = 0; k < nt; ++k) if (clscore[cltouch[k]] > mx) mx = clscore[cltouch[k]];
    const int32_t p1 = clus_of_dense[t - 1];
    Add(p1, mx * 3 / 16);
    if (t > 1 && clus_of_dense[t - 2] != p1) Add(clus_of_dense[t - 2], mx * 2 / 16);
  }
  void Add(int32_t c, int64_t b) {
    bool have = false;
    for (int32_t k = 0; k < nt; ++k) if (cltouch[k] == c) { have = true; break; }
    if (!have) { cltouch[nt++] = c; clscore[c] = 0; }
    clscore[c] += b > 1 ? b : 1;
  }
};

static ENC_ONLY int64_t OrdSym(hdcm::Cm* cm, const ORDV3Vote& V, int64_t sym, int32_t n_seen,
                      int64_t prev, int nf) {
  using namespace hdcm;
  int64_t s1 = 0, s2 = 0, s3 = 0;                  // the three best scores
  for (int32_t k = 0; k < V.nt; ++k) {
    const int64_t v = V.clscore[V.cltouch[k]];
    if (v > s1) { s3 = s2; s2 = s1; s1 = v; }
    else if (v > s2) { s3 = s2; s2 = v; }
    else if (v > s3) s3 = v;
  }
  const size_t nt = (size_t)V.nt;
  const int r8 = R8(s1, s2), lnt = Lg(nt, 6), ls = Lg((uint64_t)s1, 50);
  const int pv = prev == 0 ? 0 : (prev == 1 ? 1 : 2);
  Ctx(cm, 0, K(K(31, r8), lnt)); Ctx(cm, 1, K(32, ls / 2)); Ctx(cm, 2, K(K(33, pv), nt == 0));
  if (Bit(cm, sym == 0, 20)) return 0;
  Ctx(cm, 0, K(K(34, r8), lnt)); Ctx(cm, 1, K(35, pv));
  if (Bit(cm, sym >= ORDV3_R_ESC, 21))
    return ORDV3_R_ESC + Bits(cm, (int)(sym - ORDV3_R_ESC), Lg((uint64_t)n_seen, 20), 9);
  return 1 + Rank(cm, (int)sym - 1, 24, 12, K(K(36, r8), lnt), K(K(37, R8(s1, s3)), ls / 3),
                  K(K(38, pv), Lg((uint64_t)nf, 8)), 22);
}

// ------------------------------------------------ main entry: decoding
// residual_path: residual file; text[0..len): the page stream (.main).
// Result: unit IDs in target order (as the lines of the plain
// .new_article_order) in order[0..return value); at most cap units.
// Returns -1 on any consistency error.
static int DecodeOrderV3(const char* residual_path, const char* text, size_t len,
                         int* order, int cap) {
  Residual res;
  if (!ParseResidualV3(residual_path, &res)) return -1;

  // pass over the page stream: class per unit, features of the normal units.
  // The exclusion and correction lists are ascending (varint deltas), so one
  // cursor each finds the entry of a unit; of two corrections of one unit the
  // first counts.
  uint8_t* klass = (uint8_t*)malloc((size_t)cap + 1);   // per unit (after corrections)
  int32_t* uni = (int32_t*)malloc(4 * (size_t)cap + 4); // normal units, ascending
  size_t* fo = (size_t*)malloc(8 * (size_t)cap + 8);    // their features: fa[fo[t]..fo[t+1])
  uint64_t* kal = (uint64_t*)malloc(8 * (size_t)cap + 8);   // block entries: title key << 18 | unit
  size_t nkal = 0;
  int32_t nkl = 0, n = 0;
  {
    Extractor ex;
    PageScan ps;
    int64_t ei = 0, ci = 0;
    fo[0] = 0;
    for (const char* p = text, *end = text + len; p < end; ) {
      const char* e = (const char*)memchr(p, '\n', (size_t)(end - p));
      const size_t ln = e ? (size_t)(e - p) : (size_t)(end - p);
      const bool done = ex.Line(p, ln, &ps);
      p += ln + 1;
      if (!done) continue;
      if (ps.is_redirect) { ex.fn = ex.p0; continue; }
      if (nkl >= cap) return -1;
      const int32_t unit = nkl;
      int k = ps.klass;
      while (ci < res.nc && res.corr[2 * ci] < unit) ++ci;
      if (ci < res.nc && res.corr[2 * ci] == unit) k = res.corr[2 * ci + 1];
      while (ei < res.ne && res.excl[ei] < unit) ++ei;
      if (ei < res.ne && res.excl[ei] == unit) k = 6;
      if (k == 0 && ps.kal >= 0) { k = 7; kal[nkal++] = (uint64_t)ps.kal << 18 | (uint64_t)unit; }
      klass[nkl++] = (uint8_t)k;
      if (k == 0) { uni[n++] = unit; fo[n] = ex.p0 = ex.fn; }
      else ex.fn = ex.p0;
    }
    uint64_t* fa = ex.fa;

    // inverted index: document frequency = run length in the sorted list of
    // all features (each page holds a feature once), DF<=ORDV3_DFMAX kept,
    // integer IDF.  Feature ids in ascending hash order; they are only
    // indices, every result reads them through post/idf.
    const size_t F = fo[n];
    uint64_t* uh = (uint64_t*)malloc(8 * F + 8);
    memcpy(uh, fa, 8 * F);
    dict_derive::SortU64(uh, F);
    int32_t* ufid = (int32_t*)malloc(4 * F + 4);
    size_t U = 0;
    for (size_t i = 0; i < F; ) {
      size_t j = i;
      while (j < F && uh[j] == uh[i]) ++j;
      uh[U] = uh[i]; ufid[U++] = (int32_t)(j - i);
      i = j;
    }
    uint32_t* poff = (uint32_t*)malloc(4 * U + 8);
    int32_t nf = 0;
    poff[0] = 0;
    for (size_t u = 0; u < U; ++u) {
      if (ufid[u] <= ORDV3_DFMAX) { poff[nf + 1] = poff[nf] + (uint32_t)ufid[u]; ufid[u] = nf++; }
      else ufid[u] = -1;
    }
    int32_t* post = (int32_t*)malloc(4 * (size_t)poff[nf] + 4);
    int64_t* idf = (int64_t*)malloc(8 * (size_t)nf + 8);
    uint32_t* pcur = (uint32_t*)malloc(4 * (size_t)nf + 4);
    for (int32_t f = 0; f < nf; ++f) {
      idf[f] = Idf((uint64_t)n, (uint64_t)(poff[f + 1] - poff[f]));
      pcur[f] = poff[f];
    }
    // postings in ascending dense id; each page's features become its kept
    // feature ids, in place and in the same order
    size_t w = 0;
    for (int32_t t = 0; t < n; ++t) {
      const size_t a = fo[t], b = fo[t + 1];
      fo[t] = w;
      for (size_t i = a; i < b; ++i) {
        size_t lo = 0, hi = U;
        while (lo < hi) { const size_t m = (lo + hi) / 2; if (uh[m] < fa[i]) lo = m + 1; else hi = m; }
        const int32_t f = ufid[lo];
        if (f >= 0) { post[pcur[f]++] = t; fa[w++] = (uint64_t)f; }
      }
    }
    fo[n] = w;
    free(uh); free(ufid); free(pcur);

    const int32_t nrun = (int32_t)res.nr;
    if (nrun <= 0) return -1;
    hdcm::Cm* cm = (hdcm::Cm*)malloc(sizeof(hdcm::Cm));
    hdcm::Init(cm, res.buf + res.hdr, res.len - res.hdr, 255, 6);
    hdcm::Perm(cm, (uint32_t*)res.starts, (uint32_t)nrun);
    int64_t prevsym = -1;
    ORDV3Vote V;
    if (!V.Init(nrun, res.starts)) return -1;
    int32_t* clus = (int32_t*)malloc(4 * (size_t)n + 4);
    int32_t n_seen = 0;
    for (int32_t t = 0; t < n; ++t) {
      V.Score(t, fa + fo[t], fo[t + 1] - fo[t], post, poff, idf, clus, n_seen);
      V.Recent(t, clus);
      int64_t sym = OrdSym(cm, V,
                           0,
                           n_seen, prevsym, (int)(fo[t + 1] - fo[t]));
      prevsym = sym;
      int32_t cl;
      if (sym == 0) {
        cl = n_seen++;
        if (cl >= nrun) return -1;
      } else if (sym < ORDV3_R_ESC) {
        int64_t r = sym - 1;
        if (r >= (int64_t)V.nt) return -1;
        // rank r in the vote (score descending, cluster ascending: a total
        // order, so the sorted list has the same element at r as any selection)
        dict_derive::SortByScore((uint32_t*)V.cltouch, (uint32_t)V.nt, V.clscore);
        cl = V.cltouch[r];
      } else {
        cl = (int32_t)(sym - ORDV3_R_ESC);
        if (cl >= n_seen) return -1;
      }
      clus[t] = cl;
      V.Commit(cl);
    }
    free(cm);
    if (n_seen != nrun) return -1;
    free(ex.fa); free(post); free(poff); free(idf);
    free(V.clscore); free(V.clstamp);

    // normal part in target order: cluster sequence (a permutation, checked
    // in Init), inside dense id descending; cnt[c] = start of cluster c.
    int32_t* cnt = (int32_t*)calloc((size_t)nrun + 1, 4);
    for (int32_t t = 0; t < n; ++t) ++cnt[clus[t]];
    int32_t acc = 0;
    for (int32_t a = 0; a < nrun; ++a) {
      const int32_t cs = res.starts[a], c = cnt[cs];
      cnt[cs] = acc; acc += c;
    }
    int32_t* nrm = (int32_t*)malloc(4 * (size_t)n + 4);
    for (int32_t t = n - 1; t >= 0; --t) nrm[cnt[clus[t]]++] = uni[t];
    free(cnt); free(clus); free(uni);
    uni = nrm;
  }

  // assemble with n_split: norm[0..s) | 1 | 2 | norm[s..) | 3 | 4 | 5.
  // n_split == 0 (or >= n) = schema v2 behavior: norm | 1 | 2 | 3 | 4 | 5.
  int64_t sp = res.n_split;
  if (sp < 0 || sp > (int64_t)n) return -1;
  const bool split = (sp > 0 && sp < (int64_t)n);
  int no = 0;
  dict_derive::SortU64(kal, nkal);
  const int64_t q = res.n_kal;
  if (q < 0 || q > (int64_t)n || nkl > 0x40000) return -1;
  for (int64_t i = 0; i <= (int64_t)n; ++i) {
    if (i == q)
      for (size_t j = 0; j < nkal; ++j) order[no++] = (int)(kal[j] & 0x3FFFF);
    if (split && i == sp)
      for (int k = 1; k <= 2; ++k)
        for (int32_t u = nkl - 1; u >= 0; --u)
          if (klass[u] == k) order[no++] = u;
    if (i < (int64_t)n) order[no++] = uni[i];
  }
  for (int k = split ? 3 : 1; k <= 5; ++k)
    for (int32_t u = nkl - 1; u >= 0; --u)
      if (klass[u] == k) order[no++] = u;
  free(kal);
  free(uni); free(klass); free(fo);
  free(res.excl); free(res.corr); free(res.starts); free(res.buf);
  return no;
}

}  // namespace ord
#endif  // ORDER_CODEC_H_
