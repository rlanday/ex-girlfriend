
#ifndef ORDER_CODEC_H_
#define ORDER_CODEC_H_

#ifndef ORD_V3
#define ORD_V3 1
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>

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
namespace ord {

// ---------------------------------------------------------------- Varint
static bool ReadFileBytes(const char* path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  out.resize((size_t)n);
  bool ok = (n == 0) || fread(out.data(), 1, (size_t)n, f) == (size_t)n;
  fclose(f);
  return ok;
}

struct VarintReader {
  const uint8_t* p; const uint8_t* end;
  bool ok = true;
  int64_t Get() {
    int64_t x = 0; int s = 0;
    while (p < end) {
      uint8_t c = *p++;
      x |= (int64_t)(c & 0x7f) << s;
      if (!(c & 0x80)) return x;
      s += 7;
    }
    ok = false; return -1;
  }
  bool Done() const { return p >= end; }
};

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
static const int ORD_MAXCW = 160;
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
  std::vector<uint64_t> feats;          // deduplicated, sorted
  int64_t kal = -1;                     // KalKey of the title, -1 = none
};

struct Extractor {
  // state of the current page
  bool in_page = false, in_text = false;
  bool f_cen = false, f_dis = false, f_geo = false, f_hnd = false;
  bool f_img = false, f_distitle = false, redirect = false;
  int n_links = 0;
  std::vector<uint64_t> feats;
  int n_cw = 0;                 // content words in the lead (cap ORD_MAXCW)
  size_t lead_used = 0;         // body characters consumed
  int64_t kal = -1;             // KalKey of the title of the current page

  static bool StartsWith(const std::string& l, const char* p) {
    size_t n = strlen(p);
    return l.size() >= n && memcmp(l.data(), p, n) == 0;
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
          feats.push_back(Fnv1a('W', buf, len));
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
        if (len) { if (len > 60) len = 60; feats.push_back(Fnv1a('l', x, len)); }
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
        feats.push_back(Fnv1a('w', buf, len));
        ++n_cw;
      }
    }
    lead_used += m;
  }

  // The census sentences and the disambiguation templates below are the
  // detection strings of Byron Knoll's manual sort step (fx2-cmix).
  void FlagsLine(const char* s, size_t n) {            // body text only
    static const char* CEN[9] = {
      "According to the [[United States Census Bureau]],",
      "According to the [[U.S. Census Bureau]],",
      "As of the [[2000]] census,",
      "According to the [[2000]] census, ",
      "According to the [[United States Census Bureau|U.S. Census Bureau]],",
      "According to the [[US Census Bureau]], ",
      "According to the U.S. Census,",
      "As of the 2000 ",
      "As of the census of 2000," };
    if (!f_cen)
      for (int i = 0; i < 9; ++i)
        if (memmem(s, n, CEN[i], strlen(CEN[i]))) { f_cen = true; break; }
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
        feats.push_back(Fnv1a('C', x, (size_t)(y - x)));
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
        feats.push_back(Fnv1a('T', buf, len));
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
          feats.push_back(Fnv1a('L', x, len));
          ++n_links;
        }
        i = b + 1;
      } else {
        i = a - 1;                                     // continue past first '['
      }
    }
  }

  // processes one line; returns true when a page has been completed.
  bool Line(const std::string& l, PageScan* out) {
    if (StartsWith(l, "  <page>")) {
      in_page = true; in_text = false;
      f_cen = f_dis = f_geo = f_hnd = f_img = f_distitle = redirect = false;
      n_links = 0; feats.clear();
      n_cw = 0; lead_used = 0;
      kal = -1;
      return false;
    }
    if (!in_page) return false;
    if (StartsWith(l, "  </page>")) {
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
      std::sort(feats.begin(), feats.end());
      feats.erase(std::unique(feats.begin(), feats.end()), feats.end());
      out->feats.swap(feats);
      return true;
    }
    static const char* REDIR[5] = {
      "      <text xml:space=\"preserve\">#REDIRECT",
      "      <text xml:space=\"preserve\">#redirect",
      "      <text xml:space=\"preserve\">#Redirect",
      "      <text xml:space=\"preserve\">#REdirect",
      "      <text xml:space=\"preserve\">{{softredirect" };
    for (int i = 0; i < 5; ++i)
      if (StartsWith(l, REDIR[i])) { redirect = true; break; }
    const char* s = l.data(); size_t n = l.size();
    if (const char* t = (const char*)memmem(s, n, "<title>", 7)) {
      const char* e = (const char*)memmem(t + 7, n - (size_t)(t + 7 - s), "</title>", 8);
      if (e) Title(t + 7, e);
    }
    bool text_open = StartsWith(l, "      <text xml:space=\"preserve\">");
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

// ------------------------------------------------------- residual file
struct Residual {
  std::vector<int32_t> excl;                       // unit IDs, ascending
  std::vector<std::pair<int32_t, uint8_t>> corr;   // (unit ID, class)
  std::vector<int32_t> starts;                     // dense run starts
  std::vector<int64_t> syms;                       // symbol stream
  int64_t n_split = 0;                             // schema v3 only
  int64_t n_kal = 0;
};

static bool ParseResidual(const char* path, Residual* r) {
  std::vector<uint8_t> buf;
  if (!ReadFileBytes(path, buf)) return false;
  VarintReader v{buf.data(), buf.data() + buf.size()};
  int64_t ne = v.Get();
  int64_t last = -1;
  for (int64_t i = 0; i < ne; ++i) { last += v.Get(); r->excl.push_back((int32_t)last); }
  int64_t nc = v.Get();
  last = -1;
  for (int64_t i = 0; i < nc; ++i) {
    last += v.Get();
    int64_t k = v.Get();
    r->corr.push_back({(int32_t)last, (uint8_t)k});
  }
  int64_t nr = v.Get();
  for (int64_t i = 0; i < nr; ++i) r->starts.push_back((int32_t)v.Get());
  while (!v.Done()) {
    int64_t s = v.Get();
    if (!v.ok) break;
    r->syms.push_back(s);
  }
  return v.ok;
}

// ------------------------------------------------ main entry: decoding
// main_path: the page stream (.main); residual_path: residual file.
// Result: unit IDs in target order (like the lines of the former
// .new_article_order). Returns false on any consistency error.
static bool DecodeOrder(const char* residual_path, const char* main_path,
                        std::vector<int>* order) {
  const int CAP = 512, DFMAX = 800;
  Residual res;
  if (!ParseResidual(residual_path, &res)) return false;

  // pass over the page stream: class per unit, features of the normal units.
  std::vector<uint8_t> klass;                  // per unit (after corrections)
  std::vector<std::vector<uint64_t>> feats;    // normal units only (index parallel to uni)
  std::vector<int32_t> uni;                    // normal units, ascending
  {
    std::unordered_map<int32_t, uint8_t> corr(res.corr.begin(), res.corr.end());
    std::vector<char> excluded;
    for (int32_t e : res.excl) {
      if ((size_t)e >= excluded.size()) excluded.resize((size_t)e + 1, 0);
      excluded[(size_t)e] = 1;
    }
    std::ifstream in(main_path);
    if (!in) return false;
    std::string line;
    Extractor ex;
    PageScan ps;
    int32_t unit = 0;
    while (std::getline(in, line)) {
      if (!ex.Line(line, &ps)) continue;
      if (ps.is_redirect) continue;
      int k = ps.klass;
      auto it = corr.find(unit);
      if (it != corr.end()) k = it->second;
      if ((size_t)unit < excluded.size() && excluded[(size_t)unit]) k = 6;  // not listed
      klass.push_back((uint8_t)k);
      if (k == 0) { uni.push_back(unit); feats.push_back(std::move(ps.feats)); }
      ++unit;
    }
  }
  const int32_t n = (int32_t)uni.size();
  if ((int64_t)res.syms.size() != (int64_t)n) return false;

  // inverted index: count DF, keep DF<=800, integer IDF.
  std::unordered_map<uint64_t, int32_t> fid;   // feature -> index
  {
    std::unordered_map<uint64_t, int32_t> df;
    df.reserve((size_t)n * 8);
    for (int32_t t = 0; t < n; ++t)
      for (uint64_t f : feats[t]) ++df[f];
    fid.reserve(df.size());
    for (auto& kv : df)
      if (kv.second <= DFMAX) fid.emplace(kv.first, (int32_t)fid.size());
  }
  std::vector<std::vector<int32_t>> post(fid.size());   // sorted ascending
  std::vector<int64_t> idf(fid.size());
  for (int32_t t = 0; t < n; ++t) {
    std::vector<uint64_t> kept;
    for (uint64_t f : feats[t]) {
      auto it = fid.find(f);
      if (it != fid.end()) { post[it->second].push_back(t); kept.push_back((uint64_t)it->second); }
    }
    feats[t].swap(kept);                               // from here on: feature indices
  }
  for (auto& kv : fid) idf[kv.second] = Idf((uint64_t)n, (uint64_t)post[kv.second].size());

  // decode the symbol stream (rank/gap code, candidates via score).
  std::vector<char> alive(n, 1);
  std::vector<int64_t> score(n, 0);
  std::vector<int32_t> stamp(n, -1);
  std::vector<int32_t> touched;
  std::unordered_map<int32_t, int32_t> cur;            // feature -> weight
  std::vector<int32_t> dense;
  dense.reserve(n);
  size_t si = 0;
  int32_t prev = -1;
  for (int32_t i = 0; i < n; ++i) {
    int64_t s = res.syms[i];
    int32_t t;
    if (s == 0) {
      if (si >= res.starts.size()) return false;
      t = res.starts[si++];
      cur.clear();
      prev = -1;
    } else if (s <= CAP) {
      touched.clear();
      for (auto& kv : cur) {
        int64_t wt = (int64_t)kv.second * idf[kv.first];
        for (int32_t j : post[kv.first]) {
          if (j >= prev) break;
          if (!alive[j]) continue;
          if (stamp[j] != i) { stamp[j] = i; score[j] = 0; touched.push_back(j); }
          score[j] += wt;
        }
      }
      int32_t r = (int32_t)s - 1;
      if ((size_t)r >= touched.size()) return false;
      std::nth_element(touched.begin(), touched.begin() + r, touched.end(),
        [&](int32_t a, int32_t b) {
          return score[a] != score[b] ? score[a] > score[b] : a > b;
        });
      t = touched[r];
    } else {
      t = prev - (int32_t)(s - CAP - 1);
    }
    if (t < 0 || t >= n || !alive[t]) return false;
    dense.push_back(t);
    alive[t] = 0;
    for (uint64_t f : feats[t]) ++cur[(int32_t)f];
    prev = t;
  }
  if (si != res.starts.size()) return false;

  // assemble: normal part, then the 5 class blocks by descending ID.
  order->clear();
  order->reserve(klass.size());
  for (int32_t t : dense) order->push_back(uni[t]);
  for (int k = 1; k <= 5; ++k)
    for (int32_t u = (int32_t)klass.size() - 1; u >= 0; --u)
      if (klass[u] == k) order->push_back(u);
  return true;
}

static bool DecodeOrderV2(const char* residual_path, const char* main_path,
                          std::vector<int>* order) {
  const int DFMAX = 800;
  const int64_t R_ESC = 4096;
  Residual res;                       // starts == cluster sequence, syms == symbols
  if (!ParseResidual(residual_path, &res)) return false;

  std::vector<uint8_t> klass;
  std::vector<std::vector<uint64_t>> feats;
  std::vector<int32_t> uni;
  {
    std::unordered_map<int32_t, uint8_t> corr(res.corr.begin(), res.corr.end());
    std::vector<char> excluded;
    for (int32_t e : res.excl) {
      if ((size_t)e >= excluded.size()) excluded.resize((size_t)e + 1, 0);
      excluded[(size_t)e] = 1;
    }
    std::ifstream in(main_path);
    if (!in) return false;
    std::string line;
    Extractor ex;
    PageScan ps;
    int32_t unit = 0;
    while (std::getline(in, line)) {
      if (!ex.Line(line, &ps)) continue;
      if (ps.is_redirect) continue;
      int k = ps.klass;
      auto it = corr.find(unit);
      if (it != corr.end()) k = it->second;
      if ((size_t)unit < excluded.size() && excluded[(size_t)unit]) k = 6;
      klass.push_back((uint8_t)k);
      if (k == 0) { uni.push_back(unit); feats.push_back(std::move(ps.feats)); }
      ++unit;
    }
  }
  const int32_t n = (int32_t)uni.size();
  if ((int64_t)res.syms.size() != (int64_t)n) return false;
  const int32_t nrun = (int32_t)res.starts.size();

  std::unordered_map<uint64_t, int32_t> fid;
  {
    std::unordered_map<uint64_t, int32_t> df;
    df.reserve((size_t)n * 8);
    for (int32_t t = 0; t < n; ++t)
      for (uint64_t f : feats[t]) ++df[f];
    fid.reserve(df.size());
    for (auto& kv : df)
      if (kv.second <= DFMAX) fid.emplace(kv.first, (int32_t)fid.size());
  }
  std::vector<std::vector<int32_t>> post(fid.size());
  std::vector<int64_t> idf(fid.size());
  for (int32_t t = 0; t < n; ++t) {
    std::vector<uint64_t> kept;
    for (uint64_t f : feats[t]) {
      auto it = fid.find(f);
      if (it != fid.end()) { post[it->second].push_back(t); kept.push_back((uint64_t)it->second); }
    }
    feats[t].swap(kept);
  }
  for (auto& kv : fid) idf[kv.second] = Idf((uint64_t)n, (uint64_t)post[kv.second].size());

  // sequential vote: cluster per dense id
  std::vector<int32_t> clus(n, -1);
  int32_t n_seen = 0;
  std::vector<int64_t> clscore((size_t)nrun + 2, 0);
  std::vector<int32_t> clstamp((size_t)nrun + 2, -1);
  std::vector<int32_t> cltouch;
  for (int32_t t = 0; t < n; ++t) {
    cltouch.clear();
    for (uint64_t ff : feats[t]) {
      int32_t f = (int32_t)ff;
      int64_t w = idf[f];
      for (int32_t j : post[f]) {
        if (j >= t) break;
        int32_t cs = clus[j];
        if (clstamp[(size_t)cs] != t) { clstamp[(size_t)cs] = t; clscore[(size_t)cs] = 0; cltouch.push_back(cs); }
        clscore[(size_t)cs] += w;
      }
    }
    int64_t sym = res.syms[t];
    int32_t cl;
    if (sym == 0) {
      cl = n_seen++;
      if (cl >= nrun) return false;
    } else if (sym < R_ESC) {
      int64_t r = sym - 1;
      if ((size_t)r >= cltouch.size()) return false;
      std::nth_element(cltouch.begin(), cltouch.begin() + r, cltouch.end(),
        [&](int32_t a, int32_t b) {
          return clscore[(size_t)a] != clscore[(size_t)b]
                 ? clscore[(size_t)a] > clscore[(size_t)b] : a < b;
        });
      cl = cltouch[r];
    } else {
      cl = (int32_t)(sym - R_ESC);
      if (cl >= n_seen) return false;
    }
    clus[t] = cl;
  }
  if (n_seen != nrun) return false;

  // assemble: cluster sequence, inside by dense id descending (= unit ID
  // descending), then the 5 class blocks as in schema v1.
  std::vector<std::vector<int32_t>> members((size_t)nrun);
  for (int32_t t = 0; t < n; ++t) members[(size_t)clus[t]].push_back(t);
  order->clear();
  order->reserve(klass.size());
  for (int32_t cs : res.starts) {
    if (cs < 0 || cs >= nrun) return false;
    std::vector<int32_t>& m = members[(size_t)cs];
    for (size_t i = m.size(); i-- > 0;) order->push_back(uni[m[i]]);
  }
  for (int k = 1; k <= 5; ++k)
    for (int32_t u = (int32_t)klass.size() - 1; u >= 0; --u)
      if (klass[u] == k) order->push_back(u);
  return true;
}

static const int   ORDV3_DFMAX = 175;
static const int   ORDV3_AXISK = 8;      // axis neighbors per side
static const int   ORDV3_AXISW = 2;      // smoothing weight in 1/16
static const int   ORDV3_NORMK = 16;
static const int64_t ORDV3_R_ESC = 4096;

static const int64_t ORDV3_TUNE_TAG = 0x1FFFFF;

// like ParseResidual, but with the v3 header n_split between corr and n_runs.
// A separate function on purpose: an extra parameter on ParseResidual made
// the packed binary 8 B larger.
static bool ParseResidualV3(const char* path, Residual* r) {
  std::vector<uint8_t> buf;
  if (!ReadFileBytes(path, buf)) return false;
  VarintReader v{buf.data(), buf.data() + buf.size()};
  int64_t ne = v.Get();
  int64_t last = -1;
  for (int64_t i = 0; i < ne; ++i) { last += v.Get(); r->excl.push_back((int32_t)last); }
  int64_t nc = v.Get();
  last = -1;
  for (int64_t i = 0; i < nc; ++i) {
    last += v.Get();
    int64_t k = v.Get();
    r->corr.push_back({(int32_t)last, (uint8_t)k});
  }
  r->n_split = v.Get();
  r->n_kal = v.Get();
  int64_t nr = v.Get();
  for (int64_t i = 0; i < nr; ++i) r->starts.push_back((int32_t)v.Get());
  while (!v.Done()) {
    int64_t s = v.Get();
    if (!v.ok) break;
    r->syms.push_back(s);
  }
  if (r->syms.empty() || r->syms.back() != ORDV3_TUNE_TAG) return false;
  r->syms.pop_back();
  return v.ok;
}

// vote of one article: fills cltouch/clscore (score order = rank).
// Encoder and decoder call the same function - the roundtrip is thus
// consistent by construction.
struct ORDV3Vote {
  int32_t nrun = 0;
  std::vector<int64_t> clscore;
  std::vector<int32_t> clstamp;
  std::vector<int64_t> smear;
  std::vector<int32_t> smstamp;
  std::vector<int32_t> fill;
  std::vector<int32_t> cltouch;
  std::vector<int32_t> axis_of_seq;      // seq id -> axis rank
  const std::vector<int32_t>* seq_of_axis = 0;   // axis rank -> seq id

  bool Init(int32_t nr, const std::vector<int32_t>& starts) {
    nrun = nr;
    clscore.assign((size_t)nrun + 2, 0);
    clstamp.assign((size_t)nrun + 2, -1);
    smear.assign((size_t)nrun + 2, 0);
    smstamp.assign((size_t)nrun + 2, -1);
    fill.assign((size_t)nrun + 2, 0);
    axis_of_seq.assign((size_t)nrun, -1);
    if ((int32_t)starts.size() != nrun) return false;
    for (int32_t a = 0; a < nrun; ++a) {
      int32_t s = starts[(size_t)a];
      if (s < 0 || s >= nrun || axis_of_seq[(size_t)s] != -1) return false;
      axis_of_seq[(size_t)s] = a;
    }
    seq_of_axis = &starts;
    return true;
  }

  // t: dense id of the current article; nfeat: its feature indices;
  // clus_of_dense: seq clusters of the already decoded articles; n_seen: number
  // of seq ids handed out so far.
  void Score(int32_t t, const std::vector<uint64_t>& nfeat,
             const std::vector<std::vector<int32_t>>& post,
             const std::vector<int64_t>& idf,
             const std::vector<int32_t>& clus_of_dense, int32_t n_seen) {
    cltouch.clear();
    for (uint64_t ff : nfeat) {
      int32_t f = (int32_t)ff;
      int64_t w = idf[(size_t)f];
      if (w <= 0) continue;
      for (int32_t j : post[(size_t)f]) {
        if (j >= t) break;
        int32_t cs = clus_of_dense[(size_t)j];
        if (clstamp[(size_t)cs] != t) {
          clstamp[(size_t)cs] = t; clscore[(size_t)cs] = 0; cltouch.push_back(cs);
        }
        clscore[(size_t)cs] += w;
      }
    }
    // axis smoothing: smear a cluster's own score onto its axis neighbors.
    // Only the originally voted clusters are sources (prefix of cltouch);
    // newly touched neighbors are appended at the end.
    if (!cltouch.empty()) {
      const size_t nbase = cltouch.size();
      for (size_t bi = 0; bi < nbase; ++bi) {
        int32_t cs = cltouch[bi];
        int32_t a = axis_of_seq[(size_t)cs];
        int64_t w = (clscore[(size_t)cs] * ORDV3_AXISW) >> 4;
        for (int d = 1; d <= ORDV3_AXISK && w > 0; ++d, w >>= 1) {
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            int32_t ax = a + sgn * d;
            if (ax < 0 || ax >= nrun) continue;
            int32_t nb = (*seq_of_axis)[(size_t)ax];
            if (nb >= n_seen) continue;              // cluster not seen yet
            if (clstamp[(size_t)nb] != t) {
              clstamp[(size_t)nb] = t; clscore[(size_t)nb] = 0; cltouch.push_back(nb);
            }
            if (smstamp[(size_t)nb] != t) { smstamp[(size_t)nb] = t; smear[(size_t)nb] = 0; }
            smear[(size_t)nb] += w;
          }
        }
      }
      for (size_t k = 0; k < cltouch.size(); ++k) {
        int32_t cs = cltouch[k];
        if (smstamp[(size_t)cs] == t) { clscore[(size_t)cs] += smear[(size_t)cs]; smstamp[(size_t)cs] = -1; }
      }
    }
    // score normalization by cluster fill level (integer division)
    for (int32_t cs : cltouch)
      clscore[(size_t)cs] = clscore[(size_t)cs] * 64 / (fill[(size_t)cs] + ORDV3_NORMK);
  }
  void Commit(int32_t cl) { ++fill[(size_t)cl]; }
  void Recent(int32_t t, const std::vector<int32_t>& clus_of_dense) {
    if (t == 0) return;
    int64_t mx = 0;
    for (int32_t x : cltouch) if (clscore[(size_t)x] > mx) mx = clscore[(size_t)x];
    const int32_t p1 = clus_of_dense[(size_t)t - 1];
    Add(p1, mx * 3 / 16);
    if (t > 1 && clus_of_dense[(size_t)t - 2] != p1) Add(clus_of_dense[(size_t)t - 2], mx * 2 / 16);
  }
  void Add(int32_t c, int64_t b) {
    bool have = false;
    for (int32_t x : cltouch) if (x == c) { have = true; break; }
    if (!have) { cltouch.push_back(c); clscore[(size_t)c] = 0; }
    clscore[(size_t)c] += b > 1 ? b : 1;
  }
};

static bool DecodeOrderV3(const char* residual_path, const char* main_path,
                          std::vector<int>* order) {
  Residual res;
  if (!ParseResidualV3(residual_path, &res)) return false;

  std::vector<uint8_t> klass;
  std::vector<std::vector<uint64_t>> feats;
  std::vector<int32_t> uni;
  std::vector<uint64_t> kal;    // block entries: title key << 18 | unit
  {
    std::unordered_map<int32_t, uint8_t> corr(res.corr.begin(), res.corr.end());
    std::vector<char> excluded;
    for (int32_t e : res.excl) {
      if ((size_t)e >= excluded.size()) excluded.resize((size_t)e + 1, 0);
      excluded[(size_t)e] = 1;
    }
    std::ifstream in(main_path);
    if (!in) return false;
    std::string line;
    Extractor ex;
    PageScan ps;
    int32_t unit = 0;
    while (std::getline(in, line)) {
      if (!ex.Line(line, &ps)) continue;
      if (ps.is_redirect) continue;
      int k = ps.klass;
      auto it = corr.find(unit);
      if (it != corr.end()) k = it->second;
      if ((size_t)unit < excluded.size() && excluded[(size_t)unit]) k = 6;
      if (k == 0 && ps.kal >= 0) { k = 7; kal.push_back((uint64_t)ps.kal << 18 | (uint64_t)unit); }
      klass.push_back((uint8_t)k);
      if (k == 0) { uni.push_back(unit); feats.push_back(std::move(ps.feats)); }
      ++unit;
    }
  }
  const int32_t n = (int32_t)uni.size();
  if ((int64_t)res.syms.size() != (int64_t)n) return false;
  const int32_t nrun = (int32_t)res.starts.size();
  if (nrun <= 0) return false;

  std::unordered_map<uint64_t, int32_t> fid;
  {
    std::unordered_map<uint64_t, int32_t> df;
    df.reserve((size_t)n * 16);
    for (int32_t t = 0; t < n; ++t)
      for (uint64_t f : feats[t]) ++df[f];
    fid.reserve(df.size());
    for (auto& kv : df)
      if (kv.second <= ORDV3_DFMAX) fid.emplace(kv.first, (int32_t)fid.size());
  }
  std::vector<std::vector<int32_t>> post(fid.size());
  std::vector<int64_t> idf(fid.size());
  for (int32_t t = 0; t < n; ++t) {
    std::vector<uint64_t> kept;
    for (uint64_t f : feats[t]) {
      auto it = fid.find(f);
      if (it != fid.end()) { post[it->second].push_back(t); kept.push_back((uint64_t)it->second); }
    }
    feats[t].swap(kept);
  }
  for (auto& kv : fid) idf[kv.second] = Idf((uint64_t)n, (uint64_t)post[kv.second].size());

  ORDV3Vote V;
  if (!V.Init(nrun, res.starts)) return false;
  std::vector<int32_t> clus(n, -1);
  int32_t n_seen = 0;
  for (int32_t t = 0; t < n; ++t) {
    V.Score(t, feats[t], post, idf, clus, n_seen);
    V.Recent(t, clus);
    int64_t sym = res.syms[t];
    int32_t cl;
    if (sym == 0) {
      cl = n_seen++;
      if (cl >= nrun) return false;
    } else if (sym < ORDV3_R_ESC) {
      int64_t r = sym - 1;
      if ((size_t)r >= V.cltouch.size()) return false;
      std::nth_element(V.cltouch.begin(), V.cltouch.begin() + r, V.cltouch.end(),
        [&](int32_t a, int32_t b) {
          return V.clscore[(size_t)a] != V.clscore[(size_t)b]
                 ? V.clscore[(size_t)a] > V.clscore[(size_t)b] : a < b;
        });
      cl = V.cltouch[(size_t)r];
    } else {
      cl = (int32_t)(sym - ORDV3_R_ESC);
      if (cl >= n_seen) return false;
    }
    clus[t] = cl;
    V.Commit(cl);
  }
  if (n_seen != nrun) return false;

  // normal part in target order: cluster sequence, inside dense id descending.
  std::vector<std::vector<int32_t>> members((size_t)nrun);
  for (int32_t t = 0; t < n; ++t) members[(size_t)clus[t]].push_back(t);
  std::vector<int> nrm;
  nrm.reserve((size_t)n);
  for (int32_t cs : res.starts) {
    if (cs < 0 || cs >= nrun) return false;
    std::vector<int32_t>& m = members[(size_t)cs];
    for (size_t i = m.size(); i-- > 0;) nrm.push_back(uni[m[i]]);
  }
  if ((int32_t)nrm.size() != n) return false;

  // assemble with n_split: norm[0..s) | 1 | 2 | norm[s..) | 3 | 4 | 5.
  // n_split == 0 (or >= n) = schema v2 behavior: norm | 1 | 2 | 3 | 4 | 5.
  int64_t sp = res.n_split;
  if (sp < 0 || sp > (int64_t)n) return false;
  const bool split = (sp > 0 && sp < (int64_t)n);
  order->clear();
  order->reserve(klass.size());
  std::sort(kal.begin(), kal.end());
  const int64_t q = res.n_kal;
  if (q < 0 || q > (int64_t)n || klass.size() > 0x40000) return false;
  for (int64_t i = 0; i <= (int64_t)n; ++i) {
    if (i == q)
      for (uint64_t e : kal) order->push_back((int)(e & 0x3FFFF));
    if (split && i == sp)
      for (int k = 1; k <= 2; ++k)
        for (int32_t u = (int32_t)klass.size() - 1; u >= 0; --u)
          if (klass[u] == k) order->push_back(u);
    if (i < (int64_t)n) order->push_back(nrm[(size_t)i]);
  }
  for (int k = split ? 3 : 1; k <= 5; ++k)
    for (int32_t u = (int32_t)klass.size() - 1; u >= 0; --u)
      if (klass[u] == k) order->push_back(u);
  return true;
}

}  // namespace ord
#endif  // ORDER_CODEC_H_
