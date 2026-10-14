
#include "dict_derive.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

// Derives the word dictionary at run time from enwik9 plus the hint package:
// count words, keep the most frequent candidates, apply the hint parts
// (H1 deletions, H2 insertions, H3..H5 groups and ranks) and order bands 3+4
// with an online segment model and co-occurrence clustering.  Integer
// arithmetic only, so the result is the same on every machine.
namespace dict_derive {
namespace {

const int kCand = 145000;           // candidate words kept after counting
const int kG1 = 80, kG2 = 3840;     // code-length groups (dictionary.cpp)
const size_t kCountLimit = 0;       // 0 = count the whole stream
const uint32_t kHashBits = 22;      // 4,194,304 slots for about 1.13 M words
const uint32_t kHashMask = (1u << kHashBits) - 1;

char* arena;                        // all words, NUL-separated
size_t arena_len, arena_cap;
uint32_t* slot;                     // hash -> entry index+1
uint32_t* ent_off;                  // entry -> offset in arena
uint32_t* ent_cnt;                  // entry -> frequency
uint32_t n_ent;

const size_t kSnapBytes = 100000000ULL;
uint32_t* ent_c100;
int c100_on;

inline const char* W(uint32_t off) { return arena + off; }

uint32_t Hash(const char* s, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t)s[i]) * 16777619u;
  return h;
}

uint32_t ArenaPut(const char* s, size_t n) {
  if (arena_len + n + 1 > arena_cap) {
    arena_cap = arena_cap * 2 + n + 1;
    arena = (char*)realloc(arena, arena_cap);
  }
  uint32_t off = (uint32_t)arena_len;
  memcpy(arena + arena_len, s, n);
  arena_len += n;
  arena[arena_len++] = 0;
  return off;
}

// full == 0: only the counter of the first 100 MB (the partial word at the
// 100 MB mark is counted there, as the hint builder counts it).
void BumpEx(const char* s, size_t n, int full) {
  uint32_t h = Hash(s, n) & kHashMask;
  for (;;) {
    uint32_t e = slot[h];
    if (!e) {
      uint32_t off = ArenaPut(s, n);
      ent_off[n_ent] = off;
      ent_cnt[n_ent] = full ? 1u : 0u;
      ent_c100[n_ent] = c100_on ? 1u : 0u;
      slot[h] = ++n_ent;
      return;
    }
    const char* w = W(ent_off[e - 1]);
    if (strlen(w) == n && !memcmp(w, s, n)) {
      if (full) ++ent_cnt[e - 1];
      if (c100_on) ++ent_c100[e - 1];
      return;
    }
    h = (h + 1) & kHashMask;
  }
}
inline void Bump(const char* s, size_t n) { BumpEx(s, n, 1); }
// count of a word; as in the hint builder's count tables only values >= 2
// count (counts below 2 are not written there).
uint32_t CountOf(const char* s, uint32_t* tab) {
  size_t n = strlen(s);
  uint32_t h = Hash(s, n) & kHashMask;
  for (;;) {
    uint32_t e = slot[h];
    if (!e) return 0;
    const char* w = W(ent_off[e - 1]);
    if (strlen(w) == n && !memcmp(w, s, n)) {
      uint32_t c = tab[e - 1];
      return c >= 2 ? c : 0;
    }
    h = (h + 1) & kHashMask;
  }
}

// step 1: counting with the WRT tokenization from Dictionary::Encode.
void CountWords(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "dict_derive: %s missing\n", path); exit(1); }
  const size_t kBuf = 1 << 20;
  char* buf = (char*)malloc(kBuf);
  char word[80];
  int wl = 0, nu = 0, nl = 0;
  size_t done = 0;
  c100_on = 1;
  while (kCountLimit == 0 || done < kCountLimit) {
    size_t want = kBuf;
    if (kCountLimit != 0 && kCountLimit - done < want) want = kCountLimit - done;
    size_t got = fread(buf, 1, want, f);
    if (!got) break;
    done += got;
    for (size_t i = 0; i < got; ++i) {
      if (c100_on && done - got + i == kSnapBytes) {
        if (wl) BumpEx(word, wl, 0);      // partial word at the mark
        c100_on = 0;
      }
      int c = (unsigned char)buf[i];
      if (c >= 'a' && c <= 'z') {
        if (nu > 1) { if (wl) Bump(word, wl); wl = nu = nl = 0; }
        ++nl;
        if (wl < 79) word[wl++] = (char)c;
      } else if (c >= 'A' && c <= 'Z') {
        if (nl > 0) { if (wl) Bump(word, wl); wl = nu = nl = 0; }
        ++nu;
        if (wl < 79) word[wl++] = (char)(c - 'A' + 'a');
      } else {
        if (wl) Bump(word, wl);
        wl = nu = nl = 0;
      }
    }
  }
  if (wl) Bump(word, wl);
  free(buf);
  fclose(f);
  c100_on = 0;
}

int CmpOff(uint32_t a, uint32_t b) { return strcmp(W(a), W(b)); }
bool OffLess(uint32_t a, uint32_t b) { return CmpOff(a, b) < 0; }
bool ScoreLess(uint32_t a, uint32_t b) {   // score descending, word ascending
  int64_t sa = (int64_t)(ent_cnt[a] - 2) * ((int64_t)strlen(W(ent_off[a])) - 3);
  int64_t sb = (int64_t)(ent_cnt[b] - 2) * ((int64_t)strlen(W(ent_off[b])) - 3);
  if (sa != sb) return sa > sb;
  return CmpOff(ent_off[a], ent_off[b]) < 0;
}

// compare by reversed word (suffix order)
bool RevLess(uint32_t a, uint32_t b) {
  const char* x = W(a); const char* y = W(b);
  size_t ia = strlen(x), ib = strlen(y);
  while (ia && ib) {
    unsigned char ca = (unsigned char)x[--ia], cb = (unsigned char)y[--ib];
    if (ca != cb) return ca < cb;
  }
  return strlen(x) < strlen(y);
}

// A single sorter (quicksort with median-of-3 + insertion sort) instead of
// several std::sort instances: all comparisons here are total orders on
// pairwise distinct words, so the result is unambiguous.
typedef bool (*LessFn)(uint32_t, uint32_t);

void Sort(uint32_t* a, uint32_t n, LessFn less) {
  struct { uint32_t lo, hi; } stk[64];
  int sp = 0;
  stk[sp].lo = 0; stk[sp].hi = n; ++sp;
  while (sp) {
    --sp;
    uint32_t lo = stk[sp].lo, hi = stk[sp].hi;
    while (hi - lo > 12) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (less(a[mid], a[lo])) { uint32_t t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
      if (less(a[hi - 1], a[mid])) {
        uint32_t t = a[hi - 1]; a[hi - 1] = a[mid]; a[mid] = t;
        if (less(a[mid], a[lo])) { t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
      }
      uint32_t piv = a[mid];
      uint32_t i = lo, j = hi - 1;
      for (;;) {
        while (less(a[i], piv)) ++i;
        while (less(piv, a[j])) --j;
        if (i >= j) break;
        uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
        ++i; --j;
      }
      if (j + 1 - lo < hi - (j + 1)) {
        stk[sp].lo = j + 1; stk[sp].hi = hi; ++sp;
        hi = j + 1;
      } else {
        stk[sp].lo = lo; stk[sp].hi = j + 1; ++sp;
        lo = j + 1;
      }
    }
    for (uint32_t i = lo + 1; i < hi; ++i) {
      uint32_t v = a[i], j = i;
      while (j > lo && less(v, a[j - 1])) { a[j] = a[j - 1]; --j; }
      a[j] = v;
    }
  }
}

uint64_t ReadVarint(const uint8_t* p, size_t& pos) {
  uint64_t x = 0; int sh = 0;
  for (;;) {
    uint8_t b = p[pos++];
    x |= (uint64_t)(b & 0x7F) << sh;
    if (!(b & 0x80)) return x;
    sh += 7;
  }
}

const int kNSeg = 9;
const int kTopK = 256;              // strongest PPMI features per word
const int kNormK = 8;               // fill normalization score*64/(fill+8)
const int kWin = 5;                 // co-occurrence window +-5 tokens
const uint32_t kCtxMinF = 20;       // frequency window of the context vocabulary
const uint32_t kCtxMaxF = 60000;
const uint32_t kOccCap = 2000;      // occurrence cap per word (deterministic)
const int kMaxTok = 1024;           // longest [A-Za-z] run in enwik9: 435

// log2(x) * 65536, integer table with 17 support points + linear interpolation
int32_t Log2Q16(uint64_t x) {
  if (!x) return 0;
  int e = 63 - __builtin_clzll(x);
  uint64_t m = (e >= 16) ? (x >> (e - 16)) : (x << (16 - e));
  static const int32_t T[17] = {0, 5732, 11136, 16251, 21109, 25737, 30155,
      34383, 38437, 42331, 46076, 49684, 53164, 56525, 59774, 62919, 65536};
  uint32_t f = (uint32_t)m - 65536u, i = f >> 12, r = f & 4095;
  return e * 65536 + T[i] + (int32_t)(((int64_t)(T[i + 1] - T[i]) * r) >> 12);
}
uint64_t ISqrt64(uint64_t n) {
  if (!n) return 0;
  uint64_t x = n, y = (x + 1) >> 1;
  while (y < x) { x = y; y = (x + n / x) >> 1; }
  return x;
}
int ILog2i(uint32_t x) { int r = 0; while (x > 1) { x >>= 1; ++r; } return r; }

// ascending quicksort on uint64 (in place, no second buffer: the pair list
// is 700 MB and runs before the predictor)
void SortU64(uint64_t* a, size_t n) {
  struct { size_t lo, hi; } stk[128];
  int sp = 0;
  stk[sp].lo = 0; stk[sp].hi = n; ++sp;
  while (sp) {
    --sp;
    size_t lo = stk[sp].lo, hi = stk[sp].hi;
    while (hi - lo > 16) {
      size_t mid = lo + (hi - lo) / 2;
      if (a[mid] < a[lo]) { uint64_t t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
      if (a[hi - 1] < a[mid]) {
        uint64_t t = a[hi - 1]; a[hi - 1] = a[mid]; a[mid] = t;
        if (a[mid] < a[lo]) { t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
      }
      uint64_t piv = a[mid];
      size_t i = lo, j = hi - 1;
      for (;;) {
        while (a[i] < piv) ++i;
        while (piv < a[j]) --j;
        if (i >= j) break;
        uint64_t t = a[i]; a[i] = a[j]; a[j] = t;
        ++i; --j;
      }
      if (j + 1 - lo < hi - (j + 1)) { stk[sp].lo = j + 1; stk[sp].hi = hi; ++sp; hi = j + 1; }
      else { stk[sp].lo = lo; stk[sp].hi = j + 1; ++sp; lo = j + 1; }
    }
    for (size_t i = lo + 1; i < hi; ++i) {
      uint64_t v = a[i]; size_t j = i;
      while (j > lo && v < a[j - 1]) { a[j] = a[j - 1]; --j; }
      a[j] = v;
    }
  }
}

// --- token table: all [A-Za-z]+ types (case folded) of the stream ----------
const uint32_t kTBits = 22, kTMask = (1u << kTBits) - 1;
char* t_ar; size_t t_arlen, t_arcap;
uint32_t* t_slot; uint32_t* t_off; uint32_t* t_cnt; uint32_t* t_cid;
uint32_t t_n;

uint32_t TArenaPut(const char* s, size_t n) {
  if (t_arlen + n + 1 > t_arcap) {
    t_arcap = t_arcap * 2 + n + 1;
    t_ar = (char*)realloc(t_ar, t_arcap);
  }
  uint32_t off = (uint32_t)t_arlen;
  memcpy(t_ar + t_arlen, s, n);
  t_arlen += n;
  t_ar[t_arlen++] = 0;
  return off;
}
// returns the entry index; add==0 -> 0xffffffff if unknown
uint32_t TFind(const char* s, size_t n, int add) {
  uint32_t h = Hash(s, n) & kTMask;
  for (;;) {
    uint32_t e = t_slot[h];
    if (!e) {
      if (!add) return 0xffffffffu;
      uint32_t off = TArenaPut(s, n);
      t_off[t_n] = off; t_cnt[t_n] = 0; t_cid[t_n] = 0xffffffffu;
      t_slot[h] = ++t_n;
      return t_n - 1;
    }
    const char* w = t_ar + t_off[e - 1];
    if (strlen(w) == n && !memcmp(w, s, n)) return e - 1;
    h = (h + 1) & kTMask;
  }
}
bool TOffLess(uint32_t a, uint32_t b) { return strcmp(t_ar + t_off[a], t_ar + t_off[b]) < 0; }

// --- word table of bands 3+4 (alphabetically indexed) ----------------------
const uint32_t kWBits = 18, kWMask = (1u << kWBits) - 1;
uint32_t* w_slot;                   // hash -> word index+1
uint32_t* w_off;                    // word index -> offset in arena

uint32_t WFind(const char* s, size_t n) {
  uint32_t h = Hash(s, n) & kWMask;
  for (;;) {
    uint32_t e = w_slot[h];
    if (!e) return 0xffffffffu;
    const char* w = W(w_off[e - 1]);
    if (strlen(w) == n && !memcmp(w, s, n)) return e - 1;
    h = (h + 1) & kWMask;
  }
}
void WInsert(uint32_t idx, uint32_t off) {
  const char* s = W(off);
  size_t n = strlen(s);
  uint32_t h = Hash(s, n) & kWMask;
  while (w_slot[h]) h = (h + 1) & kWMask;
  w_slot[h] = idx + 1;
}

// --- segment table: 128-bit keys -> 9 counters -----------------------------
const uint32_t kSBits = 21, kSMask = (1u << kSBits) - 1;
uint64_t* s_ka; uint64_t* s_kb; uint8_t* s_used; int32_t* s_cnt;

int32_t* SegSlot(uint64_t a, uint64_t b) {
  uint32_t h = (uint32_t)((a * 1000003u) ^ (b * 2654435761u)) & kSMask;
  for (;;) {
    if (!s_used[h]) { s_used[h] = 1; s_ka[h] = a; s_kb[h] = b; return s_cnt + (size_t)h * kNSeg; }
    if (s_ka[h] == a && s_kb[h] == b) return s_cnt + (size_t)h * kNSeg;
    h = (h + 1) & kSMask;
  }
}

// FNV-1a over s[from..]
uint64_t SufHash(const char* s, size_t len, int k) {
  size_t from = ((size_t)k >= len) ? 0 : len - (size_t)k;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = from; i < len; ++i) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
  return h;
}

// intermediate vote state (for sorting via Sort())
int64_t* v_score;
bool CluLess(uint32_t a, uint32_t b) {
  if (v_score[a] != v_score[b]) return v_score[a] > v_score[b];
  return a < b;
}
// processing order: whole stream descending, ties by word ascending
uint32_t* o_cnt; uint32_t* o_off;
bool OrdLess(uint32_t a, uint32_t b) {
  if (o_cnt[a] != o_cnt[b]) return o_cnt[a] > o_cnt[b];
  return strcmp(W(o_off[a]), W(o_off[b])) < 0;
}

// features per word (case/frequency/DF/first occurrence), from enwik9
int32_t *f_cb, *f_fb, *f_db, *f_pb;

// one pass: type frequencies + case/DF/first-occurrence statistics.
// token rule = maximal [A-Za-z] run (the same rule as the hint builder).
void ScanFeatures(const char* path, uint32_t n3) {
  int64_t *c0 = (int64_t*)calloc(n3, 8), *c1 = (int64_t*)calloc(n3, 8);
  int64_t *c2 = (int64_t*)calloc(n3, 8), *c3 = (int64_t*)calloc(n3, 8);
  int64_t *dfa = (int64_t*)calloc(n3, 8), *fst = (int64_t*)malloc(8ull * n3);
  int32_t* lastpage = (int32_t*)malloc(4ull * n3);
  for (uint32_t i = 0; i < n3; ++i) { fst[i] = -1; lastpage[i] = -1; }
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "dict_derive: %s missing\n", path); exit(1); }
  const size_t kBuf = 1 << 22;
  char* buf = (char*)malloc(kBuf);
  char cur[kMaxTok], low[kMaxTok];
  unsigned l = 0;                     // always < kMaxTok (counter is capped)
  int64_t pos = 0, tokstart = 0;
  int page = 0, pgs = 0;
  static const char* PG = "<page>";
  size_t got;
  while ((got = fread(buf, 1, kBuf, f)) > 0) {
    for (size_t i = 0; i < got; ++i, ++pos) {
      int c = (unsigned char)buf[i];
      if (c == PG[pgs]) { if (++pgs == 6) { ++page; pgs = 0; } }
      else pgs = (c == '<') ? 1 : 0;
      int alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
      if (alpha) {
        if (!l) tokstart = pos;
        if (l < kMaxTok - 1) cur[l++] = (char)c;   // longest run in enwik9: 435
      } else if (l) {
        unsigned n = l < kMaxTok ? l : kMaxTok - 1;
        for (unsigned j = 0; j < n; ++j)
          low[j] = (cur[j] >= 'A' && cur[j] <= 'Z') ? (char)(cur[j] + 32) : cur[j];
        low[n] = 0;
        uint32_t te = TFind(low, n, 1);
        ++t_cnt[te];
        uint32_t k = WFind(low, n);
        if (k != 0xffffffffu) {
          int anyup = 0, allup = 1;
          for (unsigned j = 0; j < n; ++j) {
            if (cur[j] >= 'A' && cur[j] <= 'Z') anyup = 1; else allup = 0;
          }
          int form;
          if (!anyup) form = 0;
          else if (allup && n > 1) form = 2;
          else {
            int rest = 1;
            for (unsigned j = 1; j < n; ++j) if (cur[j] >= 'A' && cur[j] <= 'Z') { rest = 0; break; }
            form = (cur[0] >= 'A' && cur[0] <= 'Z' && rest) ? 1 : 3;
          }
          if (form == 0) ++c0[k]; else if (form == 1) ++c1[k];
          else if (form == 2) ++c2[k]; else ++c3[k];
          if (lastpage[k] != page) { lastpage[k] = page; ++dfa[k]; }
          if (fst[k] < 0) fst[k] = tokstart;
        }
        l = 0;
      }
    }
  }
  if (l) {   // stream ends mid-token (not the case for enwik9)
    unsigned n = l < kMaxTok ? l : kMaxTok - 1;
    for (unsigned j = 0; j < n; ++j)
      low[j] = (cur[j] >= 'A' && cur[j] <= 'Z') ? (char)(cur[j] + 32) : cur[j];
    low[n] = 0;
    ++t_cnt[TFind(low, n, 1)];
  }
  free(buf);
  fclose(f);
  // feature buckets as in the hint builder
  static const int TH[8] = {10, 50, 150, 350, 600, 800, 950, 990};
  for (uint32_t k = 0; k < n3; ++k) {
    int64_t t = c0[k] + c1[k] + c2[k] + c3[k];
    int64_t capm = t ? (1000 * c1[k]) / t : 0;
    int q = 0;
    for (int j = 0; j < 8; ++j) if (capm >= TH[j]) q = j + 1;
    f_cb[k] = q;
    f_fb[k] = ILog2i((uint32_t)(t > 0 ? t : 1));
    f_db[k] = ILog2i((uint32_t)(dfa[k] > 0 ? dfa[k] : 1));
    f_pb[k] = fst[k] < 0 ? 0 : (int32_t)(fst[k] / 100000000LL > 9 ? 9 : fst[k] / 100000000LL);
  }
  free(c0); free(c1); free(c2); free(c3); free(dfa); free(fst); free(lastpage);
}

// second pass: co-occurrence pairs (the same counting as the hint builder).
// pairs == 0 -> count only.
size_t ScanPairs(const char* path, uint32_t n3, uint64_t* pairs, uint32_t* used) {
  memset(used, 0, 4ull * n3);
  uint32_t ring[64], ringw[64];
  for (int i = 0; i < 64; ++i) { ring[i] = 0xffffffffu; ringw[i] = 0xffffffffu; }
  int64_t p = 0;
  size_t np = 0;
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "dict_derive: %s missing\n", path); exit(1); }
  const size_t kBuf = 1 << 22;
  char* buf = (char*)malloc(kBuf);
  char cur[kMaxTok];
  unsigned l = 0;                     // always < kMaxTok
  size_t got;
  // lambda-free, so that no STL instance is created
#define HINT_EMIT()                                                        \
  do {                                                                       \
    int64_t cp = p - kWin - 1;                                               \
    if (cp >= 0) {                                                           \
      uint32_t wc = ringw[cp & 63];                                          \
      if (wc != 0xffffffffu && used[wc] < kOccCap) {                         \
        ++used[wc];                                                          \
        for (int d = -kWin; d <= kWin; ++d) {                                \
          if (!d) continue;                                                  \
          int64_t q = cp + d;                                                \
          if (q < 0 || q >= p) continue;                                     \
          uint32_t cc = ring[q & 63];                                        \
          if (cc == 0xffffffffu) continue;                                   \
          if (pairs) pairs[np] = ((uint64_t)wc << 32) | cc;                  \
          ++np;                                                              \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  } while (0)
#define HINT_PUSH()                                                        \
  do {                                                                       \
    unsigned n = l < kMaxTok ? l : kMaxTok - 1;                              \
    cur[n] = 0;                                                              \
    uint32_t iw = WFind(cur, n);                                             \
    uint32_t te = TFind(cur, n, 0);                                          \
    ringw[p & 63] = iw;                                                      \
    ring[p & 63] = (te != 0xffffffffu) ? t_cid[te] : 0xffffffffu;            \
    ++p;                                                                     \
    HINT_EMIT();                                                           \
  } while (0)
  while ((got = fread(buf, 1, kBuf, f)) > 0) {
    for (size_t i = 0; i < got; ++i) {
      int c = (unsigned char)buf[i];
      if (c >= 'A' && c <= 'Z') c += 32;
      if (c >= 'a' && c <= 'z') { if (l < kMaxTok - 1) cur[l++] = (char)c; }
      else if (l) { HINT_PUSH(); l = 0; }
    }
  }
  if (l) { HINT_PUSH(); l = 0; }
  for (int k = 0; k <= kWin; ++k) {
    ringw[p & 63] = 0xffffffffu; ring[p & 63] = 0xffffffffu; ++p;
    HINT_EMIT();
  }
#undef HINT_EMIT
#undef HINT_PUSH
  free(buf);
  fclose(f);
  return np;
}

// ---------------------------------------------------------------------------
// core: derive the run number (dict order) for every word of bands 3+4 from
// enwik9 + head/seg/clu.  sg3 is sorted alphabetically.
// ---------------------------------------------------------------------------
bool DeriveRuns(const char* enwik9_path, const uint32_t* sg3, uint32_t n3,
                const uint8_t* head, const uint8_t* seg, const uint8_t* clu,
                uint32_t* run_of, uint32_t* nruns_out) {
  // --- head: first-occurrence number per dict run -> inversion -------------
  size_t q = 0;
  uint32_t nrun = (uint32_t)ReadVarint(head, q);
  uint32_t* dict_of_first = (uint32_t*)malloc(4ull * nrun);
  for (uint32_t c = 0; c < nrun; ++c) {
    uint32_t fi = (uint32_t)ReadVarint(head, q);
    if (fi >= nrun) return false;
    dict_of_first[fi] = c;
  }

  // --- word table ----------------------------------------------------------
  w_slot = (uint32_t*)calloc((size_t)1 << kWBits, 4);
  w_off = (uint32_t*)malloc(4ull * n3);
  for (uint32_t i = 0; i < n3; ++i) { w_off[i] = sg3[i]; WInsert(i, sg3[i]); }

  // --- pass 1: type frequencies + word features ----------------------------
  t_arcap = 1 << 22; t_ar = (char*)malloc(t_arcap); t_arlen = 0;
  t_slot = (uint32_t*)calloc((size_t)1 << kTBits, 4);
  t_off = (uint32_t*)malloc(4ull << kTBits);
  t_cnt = (uint32_t*)malloc(4ull << kTBits);
  t_cid = (uint32_t*)malloc(4ull << kTBits);
  t_n = 0;
  f_cb = (int32_t*)calloc(n3, 4); f_fb = (int32_t*)calloc(n3, 4);
  f_db = (int32_t*)calloc(n3, 4); f_pb = (int32_t*)calloc(n3, 4);
  ScanFeatures(enwik9_path, n3);

  // --- context vocabulary: frequency window, IDs alphabetical --------------
  uint32_t* cv = (uint32_t*)malloc(4ull * t_n);
  uint32_t ncv = 0;
  for (uint32_t i = 0; i < t_n; ++i)
    if (t_cnt[i] >= kCtxMinF && t_cnt[i] <= kCtxMaxF) cv[ncv++] = i;
  Sort(cv, ncv, TOffLess);
  for (uint32_t i = 0; i < ncv; ++i) t_cid[cv[i]] = i;
  const uint32_t NCTX = ncv;
  free(cv);

  // --- passes 2+3: co-occurrence pairs -------------------------------------
  uint32_t* used = (uint32_t*)malloc(4ull * n3);
  size_t np = ScanPairs(enwik9_path, n3, 0, used);
  uint64_t* pairs = (uint64_t*)malloc(np * 8);
  if (!pairs) return false;
  ScanPairs(enwik9_path, n3, pairs, used);
  free(used);
  SortU64(pairs, np);

  // --- PPMI (fixed point Q16) ----------------------------------------------
  uint64_t* Nw = (uint64_t*)calloc(n3, 8);
  uint64_t* Nc = (uint64_t*)calloc(NCTX, 8);
  uint32_t* dfc = (uint32_t*)calloc(NCTX, 4);
  uint64_t T = 0;
  for (size_t s = 0; s < np;) {
    size_t e = s; while (e < np && pairs[e] == pairs[s]) ++e;
    uint32_t w = (uint32_t)(pairs[s] >> 32), c = (uint32_t)pairs[s];
    uint64_t k = e - s;
    Nw[w] += k; Nc[c] += k; T += k; ++dfc[c];
    s = e;
  }
  int32_t lT = Log2Q16(T);
  uint32_t* woff = (uint32_t*)calloc(n3 + 1, 4);
  for (size_t s = 0; s < np;) {
    size_t e = s; while (e < np && pairs[e] == pairs[s]) ++e;
    uint32_t w = (uint32_t)(pairs[s] >> 32), c = (uint32_t)pairs[s];
    if (dfc[c] >= 2) {
      int64_t v = (int64_t)Log2Q16(e - s) + lT - Log2Q16(Nw[w]) - Log2Q16(Nc[c]);
      if (v > 0) ++woff[w + 1];
    }
    s = e;
  }
  for (uint32_t i = 0; i < n3; ++i) woff[i + 1] += woff[i];
  uint64_t* wkey = (uint64_t*)malloc((size_t)woff[n3] * 8);   // (~pp)<<32 | ctx
  if (!wkey) return false;
  {
    uint32_t* cur = (uint32_t*)malloc(4ull * (n3 + 1));
    memcpy(cur, woff, 4ull * (n3 + 1));
    for (size_t s = 0; s < np;) {
      size_t e = s; while (e < np && pairs[e] == pairs[s]) ++e;
      uint32_t w = (uint32_t)(pairs[s] >> 32), c = (uint32_t)pairs[s];
      if (dfc[c] >= 2) {
        int64_t v = (int64_t)Log2Q16(e - s) + lT - Log2Q16(Nw[w]) - Log2Q16(Nc[c]);
        if (v > 0) wkey[cur[w]++] = ((uint64_t)(0xffffffffu - (uint32_t)v) << 32) | c;
      }
      s = e;
    }
    free(cur);
  }
  free(pairs); free(Nw); free(Nc); free(dfc);

  // --- per word TOPK strongest features, L2-normalized to Q12 --------------
  uint32_t* foff = (uint32_t*)calloc(n3 + 1, 4);
  uint32_t* ffe = (uint32_t*)malloc(4ull * n3 * kTopK);
  int32_t* fve = (int32_t*)malloc(4ull * n3 * kTopK);
  if (!ffe || !fve) return false;
  {
    uint32_t nf = 0;
    for (uint32_t i = 0; i < n3; ++i) {
      uint32_t a = woff[i], b = woff[i + 1];
      SortU64(wkey + a, b - a);
      uint32_t m = b - a; if (m > (uint32_t)kTopK) m = kTopK;
      uint64_t ss = 0;
      for (uint32_t k = 0; k < m; ++k) {
        uint32_t val = 0xffffffffu - (uint32_t)(wkey[a + k] >> 32);
        ffe[nf + k] = (uint32_t)wkey[a + k];
        fve[nf + k] = (int32_t)val;
        ss += (uint64_t)val * val;
      }
      uint64_t nr = ISqrt64(ss); if (!nr) nr = 1;
      for (uint32_t k = 0; k < m; ++k)
        fve[nf + k] = (int32_t)((((int64_t)fve[nf + k]) << 12) / (int64_t)nr);
      nf += m;
      foff[i + 1] = nf;
    }
  }
  free(wkey); free(woff);

  // --- inverted index ------------------------------------------------------
  uint32_t* ioff = (uint32_t*)calloc(NCTX + 1, 4);
  for (uint32_t i = 0; i < n3; ++i)
    for (uint32_t k = foff[i]; k < foff[i + 1]; ++k) ++ioff[ffe[k] + 1];
  for (uint32_t c = 0; c < NCTX; ++c) ioff[c + 1] += ioff[c];
  uint32_t* ivw = (uint32_t*)malloc(4ull * ioff[NCTX]);
  int32_t* ivv = (int32_t*)malloc(4ull * ioff[NCTX]);
  if (!ivw || !ivv) return false;
  {
    uint32_t* cur = (uint32_t*)malloc(4ull * (NCTX + 1));
    memcpy(cur, ioff, 4ull * (NCTX + 1));
    for (uint32_t i = 0; i < n3; ++i)
      for (uint32_t k = foff[i]; k < foff[i + 1]; ++k) {
        uint32_t z = cur[ffe[k]]++;
        ivw[z] = i; ivv[z] = fve[k];
      }
    free(cur);
  }

  // --- processing order (whole stream descending) --------------------------
  uint32_t* order = (uint32_t*)malloc(4ull * n3);
  o_cnt = (uint32_t*)malloc(4ull * n3);
  o_off = (uint32_t*)malloc(4ull * n3);
  for (uint32_t i = 0; i < n3; ++i) {
    order[i] = i; o_off[i] = sg3[i];
    o_cnt[i] = CountOf(W(sg3[i]), ent_cnt);
  }
  Sort(order, n3, OrdLess);

  // --- main loop: segment (online model) + cluster (vote) ------------------
  s_ka = (uint64_t*)malloc(8ull << kSBits);
  s_kb = (uint64_t*)malloc(8ull << kSBits);
  s_used = (uint8_t*)calloc((size_t)1 << kSBits, 1);
  s_cnt = (int32_t*)calloc((size_t)kNSeg << kSBits, 4);
  if (!s_ka || !s_kb || !s_used || !s_cnt) return false;
  int64_t* wsim = (int64_t*)calloc(n3, 8);
  v_score = (int64_t*)calloc(nrun, 8);
  int32_t* fill = (int32_t*)calloc(nrun, 4);
  uint32_t* touch = (uint32_t*)malloc(4ull * nrun);
  uint32_t* tw = (uint32_t*)malloc(4ull * n3);
  uint8_t* placed = (uint8_t*)calloc(n3, 1);
  int32_t* dseg = (int32_t*)malloc(4ull * n3);
  uint32_t* dclu = (uint32_t*)malloc(4ull * n3);
  size_t si = 0, ci = 0;
  uint32_t nopen = 0;
  uint64_t ka[12], kb[12];
  int wt[12];
  for (uint32_t oi = 0; oi < n3; ++oi) {
    uint32_t w = order[oi];
    const char* s = W(sg3[w]);
    size_t len = strlen(s);
    int nk = 0;
    for (int k = 1; k <= 5; ++k) { ka[nk] = (uint64_t)(0x10 + k); kb[nk] = SufHash(s, len, k); wt[nk] = 1 << (3 * k); ++nk; }
    ka[nk] = 0x20; kb[nk] = (uint64_t)f_cb[w]; wt[nk] = 1 << 2; ++nk;
    ka[nk] = (uint64_t)(0x30 + f_cb[w]); kb[nk] = SufHash(s, len, 2); wt[nk] = 1 << 5; ++nk;
    ka[nk] = (uint64_t)(0x40 + f_cb[w]); kb[nk] = SufHash(s, len, 3); wt[nk] = 1 << 5; ++nk;
    ka[nk] = 0x50; kb[nk] = (uint64_t)(f_cb[w] * 64 + f_fb[w]); wt[nk] = 1 << 3; ++nk;
    ka[nk] = 0x60; kb[nk] = (uint64_t)(f_cb[w] * 64 + f_pb[w]); wt[nk] = 1 << 3; ++nk;
    ka[nk] = 0x70; kb[nk] = (uint64_t)(f_fb[w] * 64 + f_db[w]); wt[nk] = 1 << 2; ++nk;
    int64_t sc[kNSeg];
    for (int j = 0; j < kNSeg; ++j) sc[j] = 0;
    for (int k = 0; k < nk; ++k) {
      int32_t* cc = SegSlot(ka[k], kb[k]);
      for (int j = 0; j < kNSeg; ++j) if (cc[j]) sc[j] += (int64_t)cc[j] * wt[k];
    }
    int ordS[kNSeg];
    for (int j = 0; j < kNSeg; ++j) ordS[j] = j;
    for (int a2 = 1; a2 < kNSeg; ++a2) {          // stable insertion sort
      int t3 = ordS[a2], b2 = a2 - 1;
      while (b2 >= 0 && sc[ordS[b2]] < sc[t3]) { ordS[b2 + 1] = ordS[b2]; --b2; }
      ordS[b2 + 1] = t3;
    }
    if (seg[si] >= kNSeg) return false;
    int ds = ordS[seg[si++]];
    dseg[w] = ds;
    for (int k = 0; k < nk; ++k) SegSlot(ka[k], kb[k])[ds]++;
    // vote
    uint32_t ntw = 0, nt = 0;
    for (uint32_t k = foff[w]; k < foff[w + 1]; ++k) {
      uint32_t fx = ffe[k]; int32_t av = fve[k];
      for (uint32_t j = ioff[fx]; j < ioff[fx + 1]; ++j) {
        uint32_t v = ivw[j];
        if (v == w || !placed[v]) continue;
        if (wsim[v] == 0) tw[ntw++] = v;
        wsim[v] += ((int64_t)av * ivv[j]) >> 12;
      }
    }
    for (uint32_t k = 0; k < ntw; ++k) {
      uint32_t v = tw[k];
      if (wsim[v] > 0 && dseg[v] == ds) {
        uint32_t c = dclu[v];
        if (v_score[c] == 0) touch[nt++] = c;
        v_score[c] += wsim[v];
      }
    }
    for (uint32_t k = 0; k < ntw; ++k) wsim[tw[k]] = 0;
    for (uint32_t k = 0; k < nt; ++k)
      v_score[touch[k]] = (v_score[touch[k]] * 64) / (fill[touch[k]] + kNormK);
    Sort(touch, nt, CluLess);
    uint32_t cl;
    uint8_t sym = clu[ci++];
    if (sym == 253) cl = nopen++;
    else if (sym == 254) cl = (uint32_t)ReadVarint(clu, ci);
    else { if (sym >= nt) return false; cl = touch[sym]; }
    for (uint32_t k = 0; k < nt; ++k) v_score[touch[k]] = 0;
    if (cl >= nrun) return false;
    dclu[w] = cl; placed[w] = 1; ++fill[cl];
  }
  for (uint32_t i = 0; i < n3; ++i) run_of[i] = dict_of_first[dclu[i]];
  *nruns_out = nrun;

  free(dict_of_first); free(w_slot); free(w_off);
  free(t_ar); free(t_slot); free(t_off); free(t_cnt); free(t_cid);
  free(f_cb); free(f_fb); free(f_db); free(f_pb);
  free(foff); free(ffe); free(fve); free(ioff); free(ivw); free(ivv);
  free(order); free(o_cnt); free(o_off);
  free(s_ka); free(s_kb); free(s_used); free(s_cnt);
  free(wsim); free(v_score); free(fill); free(touch); free(tw);
  free(placed); free(dseg); free(dclu);
  return true;
}

}  // namespace

bool BuildDictFromInput(const char* enwik9_path, const char* hints_path,
                        const char* out_path) {
  // --- load hints ----------------------------------------------------------
  const int kParts = 8;      // H1 H2 h3exc H4 H5 head seg clu
  FILE* hf = fopen(hints_path, "rb");
  if (!hf) return false;
  fseek(hf, 0, SEEK_END);
  long hl = ftell(hf);
  fseek(hf, 0, SEEK_SET);
  uint8_t* hints = (uint8_t*)malloc(hl);
  if (!hints || fread(hints, 1, hl, hf) != (size_t)hl) { fclose(hf); return false; }
  fclose(hf);
  const uint8_t* part[kParts]; size_t plen[kParts];
  { size_t pos = 0;
    for (int i = 0; i < kParts; ++i) {
      uint64_t ln = ReadVarint(hints, pos);
      part[i] = hints + pos; plen[i] = ln; pos += ln;
      if (pos > (size_t)hl) return false;
    } }
  const uint32_t kTotal = (uint32_t)plen[6] + kG1 + kG2;   // seg: one byte per band-3 word

  // --- steps 1+2: candidate set -------------------------------------------
  arena_cap = 1 << 22; arena = (char*)malloc(arena_cap); arena_len = 0;
  slot = (uint32_t*)calloc((size_t)1 << kHashBits, 4);
  ent_off = (uint32_t*)malloc(4u << kHashBits);
  ent_cnt = (uint32_t*)malloc(4u << kHashBits);
  ent_c100 = (uint32_t*)malloc(4u << kHashBits);
  n_ent = 0;
  CountWords(enwik9_path);
  uint32_t* idx = (uint32_t*)malloc(4ull * n_ent);
  uint32_t n_idx = 0;
  for (uint32_t i = 0; i < n_ent; ++i) if (ent_cnt[i] >= 2) idx[n_idx++] = i;
  Sort(idx, n_idx, ScoreLess);
  uint32_t nc = n_idx < (uint32_t)kCand ? n_idx : (uint32_t)kCand;
  uint32_t* cand = (uint32_t*)malloc(4ull * nc);        // offsets in arena
  for (uint32_t i = 0; i < nc; ++i) cand[i] = ent_off[idx[i]];
  Sort(cand, nc, OffLess);
  free(idx);

  // --- H1: deletion list ---------------------------------------------------
  uint8_t* del = (uint8_t*)calloc(nc, 1);
  { size_t q = 0;
    uint64_t nd = ReadVarint(part[0], q), acc = 0;
    for (uint64_t i = 0; i < nd; ++i) {
      acc += ReadVarint(part[0], q);
      if (acc >= nc) return false;
      del[acc] = 1;
    } }
  uint32_t* kept = (uint32_t*)malloc(4ull * nc);
  uint32_t nk = 0;
  for (uint32_t i = 0; i < nc; ++i) if (!del[i]) kept[nk++] = cand[i];
  free(del); free(cand);

  // --- H2: insertions (merge into the final sorted list) -------------------
  uint32_t* fin = (uint32_t*)malloc(4ull * kTotal);
  { size_t q = 0;
    uint64_t na = ReadVarint(part[1], q), acc = 0;
    uint32_t src = 0, dst = 0;
    char tmp[80];
    for (uint64_t i = 0; i < na; ++i) {
      acc += ReadVarint(part[1], q);
      while (dst < acc) {
        if (src >= nk) return false;
        fin[dst++] = kept[src++];
      }
      uint32_t lcp = part[1][q++];
      const char* prev = dst > 0 ? W(fin[dst - 1]) : "";
      if (lcp > strlen(prev) || lcp >= sizeof(tmp)) return false;
      memcpy(tmp, prev, lcp);
      size_t k = lcp;
      while (part[1][q]) { if (k + 1 >= sizeof(tmp)) return false; tmp[k++] = (char)part[1][q++]; }
      ++q;
      fin[dst++] = ArenaPut(tmp, k);
    }
    while (dst < kTotal) {
      if (src >= nk) return false;
      fin[dst++] = kept[src++];
    }
    if (src != nk) return false; }
  free(kept);

  uint8_t* band = (uint8_t*)malloc(kTotal);
  { uint32_t* ord = (uint32_t*)malloc(4ull * kTotal);
    o_cnt = (uint32_t*)malloc(4ull * kTotal);
    o_off = (uint32_t*)malloc(4ull * kTotal);
    for (uint32_t i = 0; i < kTotal; ++i) {
      ord[i] = i; o_off[i] = fin[i];
      o_cnt[i] = CountOf(W(fin[i]), ent_c100);
    }
    Sort(ord, kTotal, OrdLess);
    for (uint32_t i = 0; i < kTotal; ++i)
      band[ord[i]] = (uint8_t)(i < (uint32_t)kG1 ? 0 : (i < (uint32_t)(kG1 + kG2) ? 1 : 2));
    free(ord); free(o_cnt); free(o_off); }
  { size_t q = 0;
    uint64_t ne = ReadVarint(part[2], q), acc = 0;
    for (uint64_t i = 0; i < ne; ++i) {
      acc += ReadVarint(part[2], q);
      if (acc >= kTotal) return false;
      band[acc] = part[2][q++];
    } }

  // --- H3/H4/H5: groups and ranks -----------------------------------------
  uint32_t* sg1 = (uint32_t*)malloc(4ull * kG1);
  uint32_t* sg2 = (uint32_t*)malloc(4ull * kG2);
  uint32_t* sg3 = (uint32_t*)malloc(4ull * kTotal);
  uint32_t n1 = 0, n2 = 0, n3 = 0;
  for (uint32_t i = 0; i < kTotal; ++i) {
    uint8_t g = band[i];
    if (g == 0) { if (n1 == kG1) return false; sg1[n1++] = fin[i]; }
    else if (g == 1) { if (n2 == kG2) return false; sg2[n2++] = fin[i]; }
    else sg3[n3++] = fin[i];
  }
  free(band);
  if (n1 != kG1 || n2 != kG2 || plen[3] != kG1 || plen[4] != 2u * n2 ||
      plen[6] != n3) return false;
  uint32_t* g1 = (uint32_t*)malloc(4ull * kG1);
  uint32_t* g2 = (uint32_t*)malloc(4ull * kG2);
  for (uint32_t i = 0; i < kG1; ++i) g1[part[3][i]] = sg1[i];
  uint32_t* ord2 = (uint32_t*)malloc(4ull * n2);
  memcpy(ord2, sg2, 4ull * n2);
  Sort(ord2, n2, RevLess);
  for (uint32_t i = 0; i < n2; ++i) {
    uint32_t r = ((uint32_t)part[4][2 * i] << 8) | part[4][2 * i + 1];
    if (r >= kG2) return false;
    g2[r] = ord2[i];
  }

  uint32_t* rid32 = (uint32_t*)malloc(4ull * n3);
  uint32_t nruns = 0;
  if (!DeriveRuns(enwik9_path, sg3, n3, part[5], part[6], part[7], rid32, &nruns))
    return false;
  uint32_t* start = (uint32_t*)calloc(nruns + 1, 4);
  for (uint32_t i = 0; i < n3; ++i) ++start[rid32[i] + 1];
  for (uint32_t r = 0; r < nruns; ++r) start[r + 1] += start[r];
  uint32_t* g3 = (uint32_t*)malloc(4ull * n3);
  { uint32_t* fillp = (uint32_t*)malloc(4ull * nruns);
    memcpy(fillp, start, 4ull * nruns);
    for (uint32_t i = 0; i < n3; ++i) g3[fillp[rid32[i]]++] = sg3[i];
    free(fillp); }
  free(rid32);

  // --- output --------------------------------------------------------------
  FILE* of = fopen(out_path, "wb");
  if (!of) return false;
  char* out = (char*)malloc((size_t)kTotal * 64);
  size_t o = 0;
  for (uint32_t i = 0; i < kG1; ++i) { const char* w = W(g1[i]); size_t l = strlen(w); memcpy(out + o, w, l); o += l; out[o++] = '\n'; }
  for (uint32_t i = 0; i < kG2; ++i) { const char* w = W(g2[i]); size_t l = strlen(w); memcpy(out + o, w, l); o += l; out[o++] = '\n'; }
  for (uint32_t i = 0; i < n3; ++i) { const char* w = W(g3[i]); size_t l = strlen(w); memcpy(out + o, w, l); o += l; out[o++] = '\n'; }
  fwrite(out, 1, o, of);
  fclose(of);

  // Give everything back: compression continues straight into the model run,
  // and every MB held here would add to its peak RSS.  malloc_trim returns
  // the large mmap blocks (slot/ent_*) to the kernel.
  free(out); free(g3); free(start);
  free(g1); free(g2); free(ord2); free(sg1); free(sg2); free(sg3);
  free(fin); free(hints);
  free(arena); arena = NULL; arena_len = arena_cap = 0;
  free(slot); slot = NULL;
  free(ent_off); ent_off = NULL;
  free(ent_cnt); ent_cnt = NULL;
  free(ent_c100); ent_c100 = NULL;
  n_ent = 0;
  malloc_trim(0);
  return true;
}

}  // namespace dict_derive

#ifdef DICT_DERIVE_MAIN
#include <ctime>
int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: %s <enwik9> <hints.bin> <out.dic>\n", argv[0]); return 2; }
  clock_t t0 = clock();
  bool ok = dict_derive::BuildDictFromInput(argv[1], argv[2], argv[3]);
  fprintf(stderr, "BuildDictFromInput: %s, %.1f s CPU\n", ok ? "OK" : "FEHLER",
          (double)(clock() - t0) / CLOCKS_PER_SEC);
  return ok ? 0 : 1;
}
#endif
