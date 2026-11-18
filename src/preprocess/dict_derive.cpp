
#include "../meld.h"
#include "dict_derive.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include "hd_cm.h"

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

// growing char buffer of NUL-separated words; a word is its offset
struct Arena { char* p; size_t len, cap; };
Arena ar;                           // all words
uint32_t* slot;                     // hash -> entry index+1
uint32_t* ent_off;                  // entry -> offset in arena
uint32_t* ent_cnt;                  // entry -> frequency
uint32_t n_ent;

const size_t kSnapBytes = 100000000ULL;
uint32_t* ent_c100;
int c100_on;

uint32_t* ent_cap;
uint32_t* ent_df;
uint32_t* ent_lp;                   // last page that counted the word
uint32_t g_page;                    // current page number of the count
int g_cap;                          // token being counted is capitalized

inline const char* W(uint32_t off) { return ar.p + off; }

uint32_t Hash(const char* s, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) h = (h ^ (uint8_t)s[i]) * 16777619u;
  return h;
}

uint32_t ArenaPut(Arena* a, const char* s, size_t n) {
  if (a->len + n + 1 > a->cap) {
    a->cap = a->cap * 2 + n + 1;
    a->p = (char*)realloc(a->p, a->cap);
  }
  uint32_t off = (uint32_t)a->len;
  memcpy(a->p + a->len, s, n);
  a->len += n;
  a->p[a->len++] = 0;
  return off;
}

// full == 0: only the counter of the first 100 MB (the partial word at the
// 100 MB mark is counted there, as the hint builder counts it).
void BumpEx(const char* s, size_t n, int full) {
  uint32_t h = Hash(s, n) & kHashMask;
  for (;;) {
    uint32_t e = slot[h];
    if (!e) {
      uint32_t off = ArenaPut(&ar, s, n);
      ent_off[n_ent] = off;
      ent_cnt[n_ent] = 0;
      ent_c100[n_ent] = 0;
      ent_cap[n_ent] = 0; ent_df[n_ent] = 0; ent_lp[n_ent] = 0xffffffffu;
      e = slot[h] = ++n_ent;
    } else {
      const char* w = W(ent_off[e - 1]);
      if (strlen(w) != n || memcmp(w, s, n)) { h = (h + 1) & kHashMask; continue; }
    }
    if (full) {
      ++ent_cnt[e - 1];
      ent_cap[e - 1] += (uint32_t)g_cap;
      if (ent_lp[e - 1] != g_page) { ent_lp[e - 1] = g_page; ++ent_df[e - 1]; }
    }
    if (c100_on) ++ent_c100[e - 1];
    return;
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
  if (!f) { msg_stop(114); exit(1); }
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
      // page count for the document frequency; the token flushed by this
      // byte is capitalized if it had exactly one upper-case letter
      { static const char kPg[] = "<page>";
        static int pgs = 0;
        if (c == kPg[pgs]) { if (++pgs == 6) { ++g_page; pgs = 0; } }
        else pgs = (c == '<');
        g_cap = (nu == 1); }
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

const uint32_t* g_idx;
ENC_ONLY bool RankOffLess(uint32_t a, uint32_t b) { return CmpOff(ent_off[g_idx[a]], ent_off[g_idx[b]]) < 0; }
// words by count descending, then alphabetically
ENC_ONLY bool CntLess(uint32_t a, uint32_t b) {
  if (ent_cnt[a] != ent_cnt[b]) return ent_cnt[a] > ent_cnt[b];
  return CmpOff(ent_off[a], ent_off[b]) < 0;
}
// a word of at most three letters as a number below 27^3
inline uint32_t SKey(const char* w) {
  uint32_t k = 0;
  for (; *w; ++w) k = k * 27 + (uint32_t)(*w - 'a' + 1);
  return k;
}
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

// One sorter for all index lists, the libc qsort: all comparisons here are
// total orders on pairwise distinct words (or ids), so every sorting method
// gives the same list.
typedef bool (*LessFn)(uint32_t, uint32_t);
LessFn g_less;
int LessCmp(const void* a, const void* b) {
  const uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
  return g_less(x, y) ? -1 : g_less(y, x);
}
void Sort(uint32_t* a, uint32_t n, LessFn less) {
  g_less = less;
  qsort(a, n, sizeof(uint32_t), LessCmp);
}

}  // namespace

uint64_t ReadVarint(const uint8_t* p, size_t& pos) {
  uint64_t x = 0; int sh = 0;
  for (;;) {
    uint8_t b = p[pos++];
    x |= (uint64_t)(b & 0x7F) << sh;
    if (!(b & 0x80)) return x;
    sh += 7;
  }
}

namespace {

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

}  // namespace

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

namespace {

// --- token table: all [A-Za-z]+ types (case folded) of the stream ----------
const uint32_t kTBits = 22, kTMask = (1u << kTBits) - 1;
Arena tar;
uint32_t* t_slot; uint32_t* t_off; uint32_t* t_cnt; uint32_t* t_cid;
uint32_t t_n;

// returns the entry index; add==0 -> 0xffffffff if unknown
uint32_t TFind(const char* s, size_t n, int add) {
  uint32_t h = Hash(s, n) & kTMask;
  for (;;) {
    uint32_t e = t_slot[h];
    if (!e) {
      if (!add) return 0xffffffffu;
      uint32_t off = ArenaPut(&tar, s, n);
      t_off[t_n] = off; t_cnt[t_n] = 0; t_cid[t_n] = 0xffffffffu;
      t_slot[h] = ++t_n;
      return t_n - 1;
    }
    const char* w = tar.p + t_off[e - 1];
    if (strlen(w) == n && !memcmp(w, s, n)) return e - 1;
    h = (h + 1) & kTMask;
  }
}
bool TOffLess(uint32_t a, uint32_t b) { return strcmp(tar.p + t_off[a], tar.p + t_off[b]) < 0; }

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

}  // namespace

void SortByScore(uint32_t* a, uint32_t n, const int64_t* score) {
  v_score = (int64_t*)score;
  Sort(a, n, CluLess);
}

namespace {

// processing order: whole stream descending, ties by word ascending
uint32_t* o_cnt; uint32_t* o_off;
bool OrdLess(uint32_t a, uint32_t b) {
  if (o_cnt[a] != o_cnt[b]) return o_cnt[a] > o_cnt[b];
  return strcmp(W(o_off[a]), W(o_off[b])) < 0;
}

// features per word (case/frequency/DF/first occurrence), from enwik9
int32_t *f_cb, *f_fb, *f_db, *f_pb;

// the token cur[0, l) in lower case into low (0-terminated); returns its length
__attribute__((noinline)) unsigned Lower(const char* cur, unsigned l, char* low) {
  unsigned n = l < kMaxTok ? l : kMaxTok - 1;
  for (unsigned j = 0; j < n; ++j)
    low[j] = (cur[j] >= 'A' && cur[j] <= 'Z') ? (char)(cur[j] + 32) : cur[j];
  low[n] = 0;
  return n;
}

// one pass: type frequencies + case/DF/first-occurrence statistics.
// token rule = maximal [A-Za-z] run (the same rule as the hint builder).
void ScanFeatures(const char* path, uint32_t n3) {
  int64_t *c0 = (int64_t*)calloc(n3, 8), *c1 = (int64_t*)calloc(n3, 8);
  int64_t *c2 = (int64_t*)calloc(n3, 8), *c3 = (int64_t*)calloc(n3, 8);
  int64_t *dfa = (int64_t*)calloc(n3, 8), *fst = (int64_t*)malloc(8ull * n3);
  int32_t* lastpage = (int32_t*)malloc(4ull * n3);
  for (uint32_t i = 0; i < n3; ++i) { fst[i] = -1; lastpage[i] = -1; }
  FILE* f = fopen(path, "rb");
  if (!f) { msg_stop(114); exit(1); }
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
        unsigned n = Lower(cur, l, low);
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
    unsigned n = Lower(cur, l, low);
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

// state of the co-occurrence scan: ring of the last 64 tokens (dictionary
// word, context id), token count, pair count
struct PairRing {
  uint32_t ring[64], ringw[64];
  int64_t p;
  size_t np;
  uint64_t* pairs;
  uint32_t* used;
};

// the pairs of the token kWin + 1 places back with its window
__attribute__((noinline)) void PairEmit(PairRing* z) {
  int64_t cp = z->p - kWin - 1;
  if (cp >= 0) {
    uint32_t wc = z->ringw[cp & 63];
    if (wc != 0xffffffffu && z->used[wc] < kOccCap) {
      ++z->used[wc];
      for (int d = -kWin; d <= kWin; ++d) {
        if (!d) continue;
        int64_t q = cp + d;
        if (q < 0 || q >= z->p) continue;
        uint32_t cc = z->ring[q & 63];
        if (cc == 0xffffffffu) continue;
        if (z->pairs) z->pairs[z->np] = ((uint64_t)wc << 32) | cc;
        ++z->np;
      }
    }
  }
}

// one token cur[0, l): into the ring, then PairEmit
__attribute__((noinline)) void PairPush(PairRing* z, char* cur, unsigned l) {
  unsigned n = l < kMaxTok ? l : kMaxTok - 1;
  cur[n] = 0;
  uint32_t iw = WFind(cur, n);
  uint32_t te = TFind(cur, n, 0);
  z->ringw[z->p & 63] = iw;
  z->ring[z->p & 63] = (te != 0xffffffffu) ? t_cid[te] : 0xffffffffu;
  ++z->p;
  PairEmit(z);
}

// second pass: co-occurrence pairs (the same counting as the hint builder).
// pairs == 0 -> count only.
size_t ScanPairs(const char* path, uint32_t n3, uint64_t* pairs, uint32_t* used) {
  memset(used, 0, 4ull * n3);
  PairRing z;
  for (int i = 0; i < 64; ++i) { z.ring[i] = 0xffffffffu; z.ringw[i] = 0xffffffffu; }
  z.p = 0;
  z.np = 0;
  z.pairs = pairs;
  z.used = used;
  FILE* f = fopen(path, "rb");
  if (!f) { msg_stop(114); exit(1); }
  const size_t kBuf = 1 << 22;
  char* buf = (char*)malloc(kBuf);
  char cur[kMaxTok];
  unsigned l = 0;                     // always < kMaxTok
  size_t got;
  while ((got = fread(buf, 1, kBuf, f)) > 0) {
    for (size_t i = 0; i < got; ++i) {
      int c = (unsigned char)buf[i];
      if (c >= 'A' && c <= 'Z') c += 32;
      if (c >= 'a' && c <= 'z') { if (l < kMaxTok - 1) cur[l++] = (char)c; }
      else if (l) { PairPush(&z, cur, l); l = 0; }
    }
  }
  if (l) { PairPush(&z, cur, l); l = 0; }
  for (int k = 0; k <= kWin; ++k) {
    z.ringw[z.p & 63] = 0xffffffffu; z.ring[z.p & 63] = 0xffffffffu; ++z.p;
    PairEmit(&z);
  }
  free(buf);
  fclose(f);
  return z.np;
}

// ---------------------------------------------------------------------------
// core: derive the run number (dict order) for every word of bands 3+4 from
// enwik9 + head/seg/clu.  sg3 is sorted alphabetically.
// ---------------------------------------------------------------------------
bool DeriveRuns(const char* enwik9_path, const uint32_t* sg3, uint32_t n3,
                const uint8_t* head, const uint8_t* seg, const uint8_t* clu,
                uint32_t* run_of, uint32_t* nruns_out
                , hdcm::Cm* cm
                ) {
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
  tar.cap = 1 << 22; tar.p = (char*)malloc(tar.cap); tar.len = 0;
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
    int segsym;
    { using namespace hdcm;
      int nz = 0;
      for (int j = 0; j < kNSeg; ++j) nz += sc[j] != 0;
      const int64_t s1 = sc[ordS[0]];
      const int r8 = R8(s1, sc[ordS[1]]), top = ordS[0], sec = ordS[1];
      cm->lim = 60; cm->lr = 20;
      const int ys = 0;
      segsym = 0;
      while (segsym < kNSeg - 1) {
        Ctx(cm, 0, K(K(K(11, r8), top), segsym));
        Ctx(cm, 1, K(K(K(K(12, Lg((uint64_t)s1, 40) / 2), nz < 4 ? nz : 4), sec), segsym));
        Ctx(cm, 2, K(K(K(K(13, top), sec), r8 / 2), segsym));
        if (!Bit(cm, ys > segsym, 1 + segsym)) break;
        ++segsym;
      }
      cm->lim = 255; cm->lr = 6; }
    int ds = ordS[segsym];
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
    { using namespace hdcm;
      int ys = 0;
      const uint32_t ye = 0;
      const int64_t s1 = nt > 0 ? v_score[touch[0]] : 0;
      const int r8 = R8(s1, nt > 1 ? v_score[touch[1]] : 0);
      const int r83 = R8(s1, nt > 2 ? v_score[touch[2]] : 0);
      const int lnt = Lg(nt, 5), ls = Lg((uint64_t)s1, 40) / 3;
      Ctx(cm, 0, K(K(21, r8), lnt)); Ctx(cm, 1, K(K(22, ls), lnt)); Ctx(cm, 2, K(23, Lg(nopen, 12)));
      if (Bit(cm, ys == 253, 10)) cl = nopen++;
      else {
        Ctx(cm, 0, K(K(24, r8), lnt)); Ctx(cm, 1, K(25, ls));
        if (Bit(cm, ys == 254, 11)) cl = (uint32_t)Bits(cm, (int)ye, Lg(nopen, 20), 8);
        else {
          ys = Rank(cm, ys, 24, 8, K(K(26, r8), lnt), K(K(27, r83), ls), K(K(28, ds), segsym), 12);
          if (ys >= (int)nt) return false;
          cl = touch[ys];
        }
      } }
    for (uint32_t k = 0; k < nt; ++k) v_score[touch[k]] = 0;
    if (cl >= nrun) return false;
    dclu[w] = cl; placed[w] = 1; ++fill[cl];
  }
  for (uint32_t i = 0; i < n3; ++i) run_of[i] = dict_of_first[dclu[i]];
  *nruns_out = nrun;

  free(dict_of_first); free(w_slot); free(w_off);
  free(tar.p); free(t_slot); free(t_off); free(t_cnt); free(t_cid);
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
  const int kParts = 4;      // v2: H2 h3exc head + coded stream
  enum { PH2 = 0, PH3 = 1, PHEAD = 2, PCS = 3 };
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
  hdcm::Cm* cm = (hdcm::Cm*)malloc(sizeof(hdcm::Cm));
  hdcm::Init(cm, (uint8_t*)part[PCS], plen[PCS], 255, 6);

  // --- steps 1+2: candidate set -------------------------------------------
  ar.cap = 1 << 22; ar.p = (char*)malloc(ar.cap); ar.len = 0;
  slot = (uint32_t*)calloc((size_t)1 << kHashBits, 4);
  ent_off = (uint32_t*)malloc(4u << kHashBits);
  ent_cnt = (uint32_t*)malloc(4u << kHashBits);
  ent_c100 = (uint32_t*)malloc(4u << kHashBits);
  ent_cap = (uint32_t*)malloc(4u << kHashBits);
  ent_df = (uint32_t*)malloc(4u << kHashBits);
  ent_lp = (uint32_t*)malloc(4u << kHashBits);
  n_ent = 0;
  CountWords(enwik9_path);
  uint32_t* idx = (uint32_t*)malloc(4ull * n_ent);
  uint32_t n_idx = 0;
  for (uint32_t i = 0; i < n_ent; ++i) if (ent_cnt[i] >= 2) idx[n_idx++] = i;
  Sort(idx, n_idx, ScoreLess);
  uint32_t nc = n_idx < (uint32_t)kCand ? n_idx : (uint32_t)kCand;
  uint32_t* cand = (uint32_t*)malloc(4ull * nc);        // offsets in arena
  // alphabetical order of the score ranks: ridx[i] = rank of candidate i
  uint32_t* ridx = (uint32_t*)malloc(4ull * nc);
  for (uint32_t i = 0; i < nc; ++i) ridx[i] = i;
  g_idx = idx;
  Sort(ridx, nc, RankOffLess);
  for (uint32_t i = 0; i < nc; ++i) cand[i] = ent_off[idx[ridx[i]]];

  // --- H1: deletion list ---------------------------------------------------
  uint8_t* del = (uint8_t*)calloc(nc, 1);
  { using namespace hdcm;
    int prev = 0;
    for (uint32_t i = 0; i < nc; ++i) {
      const uint32_t e = idx[ridx[i]], r = ridx[i] + 1, cnt = ent_cnt[e];
      size_t len = strlen(W(cand[i]));
      if (len > 14) len = 14;
      const int lc = Lg(cnt, 24);
      Ctx(cm, 0, K(1, r / 2000));
      Ctx(cm, 1, K(K(2, lc), (uint32_t)len));
      Ctx(cm, 2, K(K(K(3, r / 8000), (uint32_t)len), prev));
      Ctx(cm, 3, K(K(4, Q(ent_c100[e], cnt, 8)), lc / 2));
      Ctx(cm, 4, K(K(5, Q(ent_df[e], cnt, 8)), lc / 2));
      Ctx(cm, 5, K(K(6, Q(ent_cap[e], cnt, 8)), (uint32_t)len));
      prev = del[i] = (uint8_t)Bit(cm, del[i], 0);
    } }
  free(ridx); free(idx);
  free(ent_cap); free(ent_df); free(ent_lp);
  uint32_t* kept = (uint32_t*)malloc(4ull * nc);
  uint32_t nk = 0;
  for (uint32_t i = 0; i < nc; ++i) if (!del[i]) kept[nk++] = cand[i];
  free(del); free(cand);
  const uint32_t kShort = 1024;
  uint32_t* sp = (uint32_t*)malloc(4ull * n_ent);
  uint32_t nsp = 0;
  for (uint32_t e = 0; e < n_ent; ++e)
    if (ent_cnt[e] >= 2 && strlen(W(ent_off[e])) <= 3) sp[nsp++] = e;
  Sort(sp, nsp, CntLess);
  if (nsp > kShort) nsp = kShort;
  uint32_t n_ins;
  { size_t q = 0; n_ins = (uint32_t)ReadVarint(part[PH2], q); }
  uint32_t kTotal = nk + n_ins;                            // kept + inserted words

  // --- H2: insertions (merge into the final sorted list) -------------------
  uint32_t* fin = (uint32_t*)malloc(4ull * (kTotal + nsp));
  { size_t q = 0;
    uint64_t na = ReadVarint(part[PH2], q), acc = 0;
    uint32_t src = 0, dst = 0;
    char tmp[80];
    for (uint64_t i = 0; i < na; ++i) {
      acc += ReadVarint(part[PH2], q);
      while (dst < acc) {
        if (src >= nk) return false;
        fin[dst++] = kept[src++];
      }
      uint32_t lcp = part[PH2][q++];
      const char* prev = dst > 0 ? W(fin[dst - 1]) : "";
      if (lcp > strlen(prev) || lcp >= sizeof(tmp)) return false;
      memcpy(tmp, prev, lcp);
      size_t k = lcp;
      while (part[PH2][q]) { if (k + 1 >= sizeof(tmp)) return false; tmp[k++] = (char)part[PH2][q++]; }
      ++q;
      fin[dst++] = ArenaPut(&ar, tmp, k);
    }
    while (dst < kTotal) {
      if (src >= nk) return false;
      fin[dst++] = kept[src++];
    }
    if (src != nk) return false; }
  free(kept);
  { using namespace hdcm;
    uint8_t* inl = (uint8_t*)calloc(27 * 27 * 27, 1);
    uint32_t nm = 0;
    for (uint32_t i = 0; i < nsp; ++i) {
      const char* w = W(ent_off[sp[i]]);
      Ctx(cm, 0, K(7, Lg(i + 1, 12)));
      Ctx(cm, 1, K(K(8, (uint32_t)strlen(w)), Lg(i + 1, 12) / 2));
      if (Bit(cm, inl[SKey(w)], 2)) {
        fin[kTotal + nm++] = ent_off[sp[i]];
      }
    }
    kTotal += nm;
    Sort(fin, kTotal, OffLess);
    free(inl); }
  free(sp);

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
    uint64_t ne = ReadVarint(part[PH3], q), acc = 0;
    for (uint64_t i = 0; i < ne; ++i) {
      acc += ReadVarint(part[PH3], q);
      if (acc >= kTotal) return false;
      band[acc] = part[PH3][q++];
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
  if (n1 != kG1 || n2 != kG2) return false;
  uint32_t* g1 = (uint32_t*)malloc(4ull * kG1);
  uint32_t* g2 = (uint32_t*)malloc(4ull * kG2);
  uint32_t* rk = (uint32_t*)malloc(4ull * kG2);
  hdcm::Perm(cm, rk, kG1);
  for (uint32_t i = 0; i < kG1; ++i) g1[rk[i]] = sg1[i];
  uint32_t* ord2 = (uint32_t*)malloc(4ull * n2);
  memcpy(ord2, sg2, 4ull * n2);
  Sort(ord2, n2, RevLess);
  hdcm::Perm(cm, rk, n2);
  for (uint32_t i = 0; i < n2; ++i) g2[rk[i]] = ord2[i];
  free(rk);

  uint32_t* rid32 = (uint32_t*)malloc(4ull * n3);
  uint32_t nruns = 0;
  if (!DeriveRuns(enwik9_path, sg3, n3, part[PHEAD], 0, 0, rid32, &nruns, cm))
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
  free(cm);

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
  free(ar.p); ar.p = NULL; ar.len = ar.cap = 0;
  free(slot); slot = NULL;
  free(ent_off); ent_off = NULL;
  free(ent_cnt); ent_cnt = NULL;
  free(ent_c100); ent_c100 = NULL;
  n_ent = 0;
  malloc_trim(0);
  return true;
}

}  // namespace dict_derive

// The coder of hd_cm.h (hint package and order residual, both read in the
// compression path only).  The squash table with its interpolation, stretch
// as its inverse, the binary arithmetic coder and the counter rate
// 1/(n + 1.5) follow lpaq1 by Matt Mahoney (GPL); entry 24 of the table is
// 4024 here, 4022 in lpaq1.
namespace hdcm {
namespace {
int16_t g_stretch[4096];
ENC_ONLY int Squash(int d) {       // 4096 / (1 + e^(-d/256)), interpolated
  static const int16_t T[33] = {1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546,
      2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4024, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
  if (d > 2047) d = 2047;
  if (d < -2047) d = -2047;
  int w = d & 127;
  d = (d >> 7) + 16;
  return (T[d] * (128 - w) + T[d + 1] * w + 64) >> 7;
}
}  // namespace

ENC_ONLY void Init(Cm* c, uint8_t* buf, size_t len, int lim, int lr) {
  int pi = 0;
  for (int x = -2047; x <= 2047; ++x) {        // stretch = inverse of squash
    int v = Squash(x);
    for (int i = pi; i <= v; ++i) g_stretch[i] = (int16_t)x;
    pi = v + 1;
  }
  for (int i = pi; i < 4096; ++i) g_stretch[i] = 2047;
  for (int i = 0; i < (1 << kTabBits); ++i) c->t[i] = 0x80000000u;
  for (int s = 0; s < 32; ++s)
    for (int i = 0; i <= kIn; ++i) c->w[s][i] = 19661;      // 0.3
  c->n = 0; c->lim = lim; c->lr = lr;
  c->x1 = 0; c->x2 = 0xffffffffu; c->x = 0;
  c->p = buf; c->end = buf + len;
  for (int i = 0; i < 4; ++i) c->x = (c->x << 8) | (c->p < c->end ? *c->p++ : 0);
}

// one binary decision with probability pr/4096 of a 1 (arithmetic coder)
ENC_ONLY int Code(Cm* c, int y, int pr) {
  const uint32_t xmid = c->x1 + (uint32_t)(((uint64_t)(c->x2 - c->x1) * (uint32_t)pr) >> 12);
  y = c->x <= xmid;
  y ? (c->x2 = xmid) : (c->x1 = xmid + 1);
  while (((c->x1 ^ c->x2) & 0xff000000u) == 0) {
    c->x1 <<= 8; c->x2 = (c->x2 << 8) | 255;
    c->x = (c->x << 8) | (c->p < c->end ? *c->p++ : 0);
  }
  return y;
}

ENC_ONLY void Perm(Cm* c, uint32_t* r, uint32_t n) {
  uint8_t* used = (uint8_t*)calloc(n, 1);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t v = 0;                     // free values below r[i] (writer)
    uint32_t lo = 0, hi = n - i;        // v uniform in [lo, hi): binary split
    while (hi - lo > 1) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (Code(c, v >= mid, (int)(((uint64_t)(hi - mid) << 12) / (hi - lo)))) lo = mid;
      else hi = mid;
    }
    uint32_t j = 0;
    for (;; ++j) if (!used[j] && !lo--) break;
    r[i] = j; used[j] = 1;
  }
  free(used);
}

ENC_ONLY int Bit(Cm* c, int y, int ws) {
  const int n = c->n;
  int64_t dot = 0;
  for (int i = 0; i < n; ++i) {
    c->st[i] = g_stretch[c->t[c->ix[i]] >> 20];
    dot += (int64_t)c->st[i] * c->w[ws][i];
  }
  c->st[n] = 256;
  dot += (int64_t)256 * c->w[ws][n];
  int pr = Squash((int)(dot >> 16));
  if (pr < 1) pr = 1;
  if (pr > 4095) pr = 4095;
  y = Code(c, y, pr);
  const int err = ((y << 12) - pr) * c->lr;
  for (int i = 0; i <= n; ++i) c->w[ws][i] += (c->st[i] * err) >> 14;
  for (int i = 0; i < n; ++i) {               // counters: rate 1/(k + 1.5)
    uint32_t& e = c->t[c->ix[i]];
    int p = (int)(e >> 16), k = (int)(e & 255);
    p += ((y ? 65535 : 0) - p) * 2 / (2 * k + 3);
    if (k < c->lim) ++k;
    e = ((uint32_t)p << 16) | (uint32_t)k;
  }
  c->n = 0;
  return y;
}

ENC_ONLY int Bits(Cm* c, int v, int nb, uint32_t base) {
  int r = 1;
  for (int b = nb - 1; b >= 0; --b) {
    Ctx(c, 0, K(base, (uint32_t)r));
    r = r * 2 + Bit(c, (v >> b) & 1, 31);
  }
  return r - (1 << nb);
}

ENC_ONLY int Rank(Cm* c, int r, int kmax, int nbesc, uint32_t a, uint32_t b, uint32_t d, int ws0) {
  for (int i = 0; i < kmax; ++i) {
    Ctx(c, 0, K(a, i)); Ctx(c, 1, K(b, i)); Ctx(c, 2, K(d, i));
    if (!Bit(c, r > i, ws0 + (i < 7 ? i : 7))) return i;
  }
  return kmax + Bits(c, r - kmax, nbesc, 7);
}

}  // namespace hdcm

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
