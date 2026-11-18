// Reversible text rules on the coder stream (behind the WRT) and their inverse.  Three stages replace text the
// page itself predicts (repeats of the title, link texts that repeat the link target, the article "a" or "an"
// before its follow word) by short codes; only the deviations stay text.  C++17, header only; no files, no
// getenv, no exceptions, memory from malloc/realloc only.  The encoder side is a port of a reference
// implementation in C that is not part of this package; the a/an exceptions are a generated list.  Order:
//   encoder  coder stream -> a/an -> title -> link        decoder  link^-1 -> title^-1 -> a/an^-1
// Bytes before `start` stay untouched in every stage.  No stage creates, removes or reorders the structural bytes
// LF 0x0A, '[' 0x5B, '|' 0x51, ']' 0x5D, and none touches the page anchor "  </page>\n".
//
// Copy rules (title, link): a span X with candidates C[0..nc) becomes M^i if X == C[i], M^(k+nc) if X == M^k
// (k >= 0, M = 0x01), else stays; the decoder reads M^j as C[j] (j < nc) or M^(j-nc).
//   title   (pages whose anchor ends at or behind start; the title is the line behind the anchor)
//     rule 1  interwiki line '[' code ':' X ']' LF, code a language code, C = [title]
//     rule 2  first bold in the body within 3000 bytes, "'''" X "'''", C = [title, title without " (...)"]
//     rule 3  sort key '[' "category" ':' X '|' K ']', the span is K, C = [title, title without " (...)"]
//   link    lone '[' T '|' D ']' (T not category/image/file/media/kategorie), the span is D,
//           C = [p, tog(p), base, tog(base), T, tog(T), section(T)], base = T behind ':', p = base without " (...)"
//           or cut at ", ", tog = leading 0x40 toggled
//   a/an    article code 0xC4/0xC6 at a site (separator before, one blank behind); the class R of the follow word
//           (rules: vowel rule with the prefix rules, letter names for all-caps words and single letters,
//           digits 8/11/18, follow word behind <tag> and behind "[[" url ' ', exact exception list) decides:
//           the encoder writes 0xC4 iff the article agrees with R, else 0xC6.
//
// Encoder: Enc() runs the stages as batch passes over whole buffers.
// Decoder: one online inverse per stage (LinkDec, TitelDec, AanDec): a decoder loop over a growing input window;
// where a decision needs bytes that are not there yet, the loop stops and later resumes at the same place.
// Dec() (whole buffer) and Sicht (the original view for the readers of the coder stream, fed byte by byte) use
// the same decoders.  The scanning predicates are shared by encoder and decoder: they read through Rd, which knows
// how far the input reaches; with fin set they answer exactly as on the complete stream.
#ifndef RULE_STAGES_H_
#define RULE_STAGES_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "a_an_list.h"

// Marks the encoder-only functions (Enc, AanEnc, TitelEnc, LinkEnc and their helpers EncCand, PageByteE); no decoder
// path (Dec, Sicht, LinkDec/TitelDec/AanDec) calls one.  The build may place them in an encoder-only section.
#ifndef RS_ENC_ONLY
#define RS_ENC_ONLY
#endif

namespace rule_stages {

constexpr uint8_t kMk = 0x01, kLf = 0x0A, kLb = 0x5B, kPi = 0x51, kRb = 0x5D, kCol = 0x4A, kEsc = 0x0C, kQ = 0x27;
constexpr uint64_t kAnker = 0x20204c2fdf984e0aULL;   // "  </page>\n" in WRT form (the last 8 bytes, LF last)
constexpr size_t kOffen = (size_t)-1;                // segment bound: up to the stream end

inline bool IsSig(uint8_t c) { return c == kLf || c == kLb || c == kPi || c == kRb; }
inline bool IsBody1(uint8_t c) { return c >= 0x80 || (c >= 0x61 && c <= 0x7A); }

// ---------------------------------------------------------------- buffers
struct Buf {
  uint8_t* p = nullptr;
  size_t n = 0, cap = 0;
  bool err = false;                                  // an allocation failed; the content is incomplete
  bool Platz(size_t k) {
    if (n + k <= cap) return true;
    const size_t c = (n + k) * 3 / 2 + 4096;
    uint8_t* q = (uint8_t*)realloc(p, c);
    if (!q) { err = true; return false; }
    p = q; cap = c;
    return true;
  }
  void Put(const uint8_t* s, size_t k) { if (k && Platz(k)) { memcpy(p + n, s, k); n += k; } }
  void Put1(uint8_t c) { if (Platz(1)) p[n++] = c; }
  void Free() { free(p); p = nullptr; n = cap = 0; }
};

// Input window of an online stage: bytes [b0, b0 + b.n) of its input stream.  S() is indexed with absolute
// positions (the base pointer is shifted by b0, computed through uintptr_t).
struct Win {
  Buf b;
  size_t b0 = 0;
  size_t End() const { return b0 + b.n; }
  uint8_t* S() const { return (uint8_t*)((uintptr_t)b.p - (uintptr_t)b0); }
  void Drop(size_t x) {                              // bytes below x are no longer read; moves only big blocks
    if (x <= b0) return;
    size_t d = x - b0;
    if (d > b.n) d = b.n;
    if (d < 65536 || 2 * d < b.n) return;
    memmove(b.p, b.p + d, b.n - d);
    b.n -= d; b0 += d;
  }
};

// ---------------------------------------------------------------- dictionary
// Word list in code order (word i has code i of the five code bands of src/preprocess/dictionary.cpp):
// word i = blob[off[i] .. off[i] + len[i]), lower case a-z.  The caller may point it at a list already in memory.
struct Dict {
  const uint8_t* blob = nullptr;
  const uint32_t* off = nullptr;
  const uint32_t* len = nullptr;
  int n = 0;
};
// Owned list parsed from a .dic image: every maximal run of a-z is one word, in file order.
struct DictSpeicher {
  uint8_t* blob = nullptr;
  uint32_t* off = nullptr;
  uint32_t* len = nullptr;
  int n = 0;
  bool FromBytes(const uint8_t* m, size_t mn) {
    Free();
    blob = (uint8_t*)malloc(mn + 1);
    off = (uint32_t*)malloc((mn / 2 + 64) * 4);
    len = (uint32_t*)malloc((mn / 2 + 64) * 4);
    if (!blob || !off || !len) { Free(); return false; }
    size_t bl = 0, cur = 0;
    bool in = false;
    for (size_t i = 0; i <= mn; i++) {
      if (i < mn && m[i] >= 'a' && m[i] <= 'z') {
        if (!in) { cur = bl; in = true; }
        blob[bl++] = m[i];
      } else if (in) {
        off[n] = (uint32_t)cur; len[n] = (uint32_t)(bl - cur); n++; in = false;
      }
    }
    return true;
  }
  void Free() { free(blob); free(off); free(len); blob = nullptr; off = len = nullptr; n = 0; }
  Dict D() const { Dict d; d.blob = blob; d.off = off; d.len = len; d.n = n; return d; }
};

// ---------------------------------------------------------------- reading
// s is indexed with absolute positions, bytes below na are present.  H(x) answers "x < n" of the complete stream:
// true below na; at or above na false, and without fin it also sets need (the caller drops the answer and waits
// for input).
struct Rd {
  const uint8_t* s;
  size_t na;
  bool fin;
  bool need;
  bool H(size_t x) {
    if (x < na) return true;
    if (!fin) need = true;
    return false;
  }
};

// Token length at i with the segment end e (kOffen: the stream end).  i is present.
inline size_t TokLen(Rd& r, size_t i, size_t e) {
  const uint8_t c = r.s[i];
  if (c == kEsc) return (i + 1 < e && r.H(i + 1)) ? 2 : 1;
  if (c <= 0xCF) return 1;
  if (!(i + 1 < e && r.H(i + 1))) return 1;
  if (r.s[i + 1] <= 0xCF) return 2;
  return (i + 2 < e && r.H(i + 2)) ? 3 : 2;
}

// Word index of the code token s[0..L), -1 if none.
inline int CodeIdx(const uint8_t* s, size_t L) {
  const int b0 = s[0];
  if (L == 1) return b0 >= 0x80 && b0 <= 0xCF ? b0 - 0x80 : -1;
  if (L == 2) return 80 + (b0 - 0xD0) * 80 + (s[1] - 0x80);
  const int b1 = s[1], b2 = s[2];
  if (b0 >= 0xF0 && b1 >= 0xF0) return 44880 + (b0 - 0xF0) * 1280 + (b1 - 0xF0) * 80 + (b2 - 0x80);
  if (b0 >= 0xF0) return 3920 + ((b0 - 0xF0) * 32 + (b1 - 0xD0)) * 80 + (b2 - 0x80);
  return 65360 + ((b0 - 0xD0) * 48 + (b1 - 0xD0)) * 80 + (b2 - 0x80);
}

// Lower-case text of the tokens [a, b) (b present): case flags 0x40/0x07/0x06 skipped (flags: 1/2/4), codes,
// letters, digits and '-' kept; -1 if another token occurs or the text exceeds cap.  out holds cap + 1 bytes.
inline int RunText(Rd& r, const Dict& D, size_t a, size_t b, char* out, int cap, int* flags) {
  const uint8_t* s = r.s;
  int k = 0;
  if (flags) *flags = 0;
  for (size_t i = a; i < b;) {
    const size_t L = TokLen(r, i, b);
    const uint8_t c = s[i];
    if (c == 0x40 || c == 0x07 || c == 0x06) {
      if (flags) *flags |= c == 0x40 ? 1 : c == 0x07 ? 2 : 4;
    } else if (c >= 0x80) {
      const int w = CodeIdx(s + i, L);
      if (w < 0 || w >= D.n) return -1;
      if (k + (int)D.len[w] > cap) return -1;
      memcpy(out + k, D.blob + D.off[w], D.len[w]);
      k += (int)D.len[w];
    } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-') {
      if (k >= cap) return -1;
      out[k++] = (char)c;
    } else {
      return -1;
    }
    i += L;
  }
  out[k] = 0;
  return k;
}

// ---------------------------------------------------------------- candidates
struct Cand { size_t n; uint8_t own[700]; };
struct CList { Cand c[8]; int n; };

inline bool AllMk(const uint8_t* x, size_t n) { for (size_t i = 0; i < n; i++) if (x[i] != kMk) return false; return true; }
inline bool Has3q(const uint8_t* x, size_t n) {
  for (size_t i = 0; i + 2 < n; i++) if (x[i] == kQ && x[i + 1] == kQ && x[i + 2] == kQ) return true;
  return false;
}
inline void ClAdd(CList* L, const uint8_t* x, size_t n) {
  if (!x || n == 0 || AllMk(x, n) || L->n >= 8 || n > sizeof L->c[0].own) return;
  for (int i = 0; i < L->n; i++) if (L->c[i].n == n && !memcmp(L->c[i].own, x, n)) return;
  Cand* c = &L->c[L->n++];
  memcpy(c->own, x, n);
  c->n = n;
}
// title without a trailing " (...)": last '(' behind ' ', no ')' in between, ends in ')'; 0 if none
inline size_t ParenCut(const uint8_t* x, size_t n) {
  if (n < 4 || x[n - 1] != 0x29) return 0;
  size_t p = n - 1;
  while (p > 0 && x[p - 1] != 0x28) { if (x[p - 1] == 0x29) return 0; p--; }
  if (p == 0) return 0;
  p--;
  if (p < 2 || x[p - 1] != 0x20) return 0;
  return p - 1;
}
inline size_t CommaCut(const uint8_t* x, size_t n) {
  for (size_t i = 1; i + 1 < n; i++) if (x[i] == 0x2C && x[i + 1] == 0x20) return i;
  return 0;
}
// first letter case toggled (leading 0x40 removed or added)
inline bool Tog(const uint8_t* x, size_t n, uint8_t* out, size_t* on) {
  if (n >= 2 && x[0] == 0x40 && IsBody1(x[1])) { memcpy(out, x + 1, n - 1); *on = n - 1; return true; }
  if (n >= 1 && IsBody1(x[0]) && n + 1 <= 690) { out[0] = 0x40; memcpy(out + 1, x, n); *on = n + 1; return true; }
  return false;
}
inline bool TitleOk(const uint8_t* t, size_t n) {
  if (n == 0 || n > 600) return false;
  for (size_t i = 0; i < n; i++) if (IsSig(t[i])) return false;
  return !Has3q(t, n);
}
inline void Cands1(const uint8_t* t, size_t n, CList* L) { L->n = 0; if (TitleOk(t, n)) ClAdd(L, t, n); }
inline void Cands23(const uint8_t* t, size_t n, CList* L) {
  L->n = 0;
  if (!TitleOk(t, n)) return;
  ClAdd(L, t, n);
  const size_t t0 = ParenCut(t, n);
  if (t0) ClAdd(L, t, t0);
}

// ---------------------------------------------------------------- title predicates
inline bool LangOk(const char* s, int n) {
  if (n == 6 && !memcmp(s, "simple", 6)) return true;
  int i = 0;
  while (i < n && s[i] >= 'a' && s[i] <= 'z') i++;
  if (i < 2 || i > 3) return false;
  while (i < n) {
    if (s[i] != '-') return false;
    i++;
    const int j = i;
    while (i < n && s[i] >= 'a' && s[i] <= 'z') i++;
    if (i == j) return false;
  }
  return true;
}
// Title rule 1 at '[' i (line start): '[' code ':' X ']' LF, within the line.  The ':' scan runs token-wise (a
// token holding an LF ends the line, so it fails), and the first structural byte behind ':' must be ']' directly
// before LF.  *c = position of ':' once the code is valid.
inline bool T1Check(Rd& r, const Dict& D, size_t i, size_t* c, size_t* rb) {
  const uint8_t* s = r.s;
  *c = 0;
  size_t q = i + 1;
  for (;;) {
    if (!r.H(q)) return false;
    const uint8_t b = s[q];
    if (b == kCol) break;
    if (IsSig(b) || b == kEsc || b == 0x40 || b == 0x07) return false;
    const size_t L = TokLen(r, q, kOffen);
    for (size_t t = q + 1; t < q + L; t++) if (s[t] == kLf) return false;
    q += L;
    if (q - i - 1 > 24) return false;
  }
  if (q - i - 1 < 2) return false;
  char tx[64];
  int fl;
  const int k = RunText(r, D, i + 1, q, tx, 40, &fl);
  if (k < 0 || fl || !LangOk(tx, k)) return false;
  *c = q;
  size_t z = q + 1;
  while (r.H(z) && !IsSig(s[z])) z++;
  if (!r.H(z) || s[z] != kRb || s[z - 1] == kEsc || z + 1 < i + 5) return false;
  if (!r.H(z + 1) || s[z + 1] != kLf) return false;
  *rb = z;
  return true;
}
// Title rule 3 at '[' i: '[' cat ':' X '|' K ']' with cat = "category" (the ':' within 64 bytes, token-wise).
inline bool T3Struct(Rd& r, const Dict& D, size_t i, size_t* pp, size_t* rb) {
  const uint8_t* s = r.s;
  size_t q = i + 1;
  for (;;) {
    if (!(q < i + 64 && r.H(q))) return false;
    if (s[q] == kCol) break;
    if (IsSig(s[q]) || s[q] == kEsc) return false;
    q += TokLen(r, q, kOffen);
  }
  char tx[32];
  int fl;
  const int k = RunText(r, D, i + 1, q, tx, 20, &fl);
  if (!(k == 8 && !memcmp(tx, "category", 8) && !(fl & 4))) return false;
  size_t p = q + 1;
  while (r.H(p) && !IsSig(s[p])) p++;
  if (!r.H(p) || s[p] != kPi || p == q + 1 || s[p - 1] == kEsc) return false;
  size_t x = p + 1;
  while (r.H(x) && !IsSig(s[x])) x++;
  if (!r.H(x) || s[x] != kRb || s[x - 1] == kEsc) return false;
  *pp = p; *rb = x;
  return true;
}
// Title rule 2: the closing "'''" behind the opening at i, before any structural byte; true with *cl if found and
// not behind an escape byte.
inline bool T2Scan(Rd& r, size_t i, size_t* cl) {
  const uint8_t* s = r.s;
  for (size_t x = i + 3;; x++) {
    if (!r.H(x) || IsSig(s[x])) return false;
    if (s[x] != kQ) continue;
    if (!r.H(x + 1)) return false;
    if (s[x + 1] != kQ) continue;
    if (!r.H(x + 2)) return false;
    if (s[x + 2] != kQ) continue;
    *cl = x;
    return s[x - 1] != kEsc;
  }
}

// ---------------------------------------------------------------- link predicates
inline bool LinkExcluded(Rd& r, const Dict& D, size_t t0, size_t n) {
  const uint8_t* T = r.s + t0;
  size_t q = 0;
  while (q < n && T[q] != kCol) q++;
  if (q == n) return false;
  char tx[32];
  int fl;
  const int k = RunText(r, D, t0, t0 + q, tx, 24, &fl);
  if (k < 0) return false;
  static const char kEx[] = "\x08" "category" "\x05" "image" "\x09" "kategorie" "\x04" "file" "\x05" "media";
  for (const char* e = kEx; *e; e += 1 + *e) if (k == *e && !memcmp(tx, e + 1, (size_t)k)) return true;
  return false;
}
// x and x with the first letter case toggled; tb is scratch (ClAdd copies)
inline void ClAddTog(CList* L, const uint8_t* x, size_t n, uint8_t* tb) {
  size_t on;
  ClAdd(L, x, n);
  if (Tog(x, n, tb, &on)) ClAdd(L, tb, on);
}
// C = [p, tog(p), base, tog(base), T, tog(T), section(T)]; p and base only without '#' in T
inline void LinkCands(const uint8_t* T, size_t n, CList* L, uint8_t* tb) {
  L->n = 0;
  size_t c0 = 0;
  while (c0 < n && T[c0] != kCol) c0++;
  const uint8_t* base = c0 < n ? T + c0 + 1 : T;
  const size_t bn = c0 < n ? n - c0 - 1 : n;
  size_t hs = 0;
  while (hs < n && T[hs] != 0x23) hs++;
  if (hs == n) {
    size_t pn = ParenCut(base, bn);
    if (!pn) pn = CommaCut(base, bn);
    if (!pn) pn = bn;
    ClAddTog(L, base, pn, tb);                       // pn == 0 only with bn == 0: adds nothing
    ClAddTog(L, base, bn, tb);
  }
  ClAddTog(L, T, n, tb);
  if (hs + 1 < n) ClAdd(L, T + hs + 1, n - hs - 1);
}
// lone '[' i: T up to the first structural byte '|' (p), D up to the next one ']' (q)
inline bool LinkStruct(Rd& r, size_t i, size_t* pp, size_t* qq) {
  const uint8_t* s = r.s;
  if (s[i] != kLb || !r.H(i + 1) || s[i + 1] == kLb) return false;
  if (i > 0 && s[i - 1] == kLb) return false;
  size_t p = i + 1;
  while (r.H(p) && !IsSig(s[p])) p++;
  if (!r.H(p) || s[p] != kPi || p == i + 1 || s[p - 1] == kEsc || p - i - 1 > 600) return false;
  size_t q = p + 1;
  while (r.H(q) && !IsSig(s[q])) q++;
  if (!r.H(q) || s[q] != kRb || s[q - 1] == kEsc) return false;
  *pp = p; *qq = q;
  return true;
}

// ---------------------------------------------------------------- a/an predicates and rules
// Follow word at j up to the segment end e (kOffen: stream end): text in w (lower case), case class *cc (0 l,
// 1 C, 2 U), *stop = byte behind the word (0x100: segment end); returns the length, -1 if no word.  A lone article
// code reads as "a" (the stage never changes any other byte, so both directions read the same word).
inline int WordAt(Rd& r, const Dict& D, size_t j, size_t e, char* w, int* cc, int* stop) {
  const uint8_t* s = r.s;
  *cc = 0;
  *stop = 0x101;
  if (!(j < e && r.H(j))) return -1;
  if (s[j] == 0x40) { *cc = 1; j++; } else if (s[j] == 0x07) { *cc = 2; j++; }
  if (!(j < e && r.H(j))) return -1;
  int k = 0;
  if (s[j] >= '0' && s[j] <= '9') {
    while (j < e && r.H(j) && s[j] >= '0' && s[j] <= '9' && k < 60) w[k++] = (char)s[j++];
    w[k] = 0;
    return k;
  }
  size_t q = j;
  int ntok = 0, art = 0;
  while (q < e && r.H(q) && IsBody1(s[q])) {
    const size_t L = TokLen(r, q, e);
    if (L == 1 && (s[q] == 0xC4 || s[q] == 0xC6)) art = 1;
    ntok++;
    q += L;
  }
  if (q == j) return -1;
  *stop = (q < e && r.H(q)) ? s[q] : 0x100;
  if (ntok == 1 && art) { w[0] = 'a'; w[1] = 0; return 1; }
  const int n = RunText(r, D, j, q, w, 120, nullptr);
  return n <= 0 ? -1 : n;
}
inline bool IsQuote(uint8_t c) { return c == 0x22 || c == 0x28 || c == kQ; }
// follow word from j: behind '"', '(', '\'' runs; behind '[' the first word of D ('[T|D'), D empty or all M ->
// first word of base(T), no '|' -> first word of T
inline int FollowFrom(Rd& r, const Dict& D, size_t j, char* w, int* cc, int* stop) {
  const uint8_t* s = r.s;
  while (r.H(j) && IsQuote(s[j])) j++;
  if (r.H(j) && s[j] == kLb) {
    const size_t a = j + 1;
    size_t p = a;
    while (r.H(p) && !IsSig(s[p]) && s[p] != kEsc) p++;
    if (r.H(p) && s[p] == kPi) {
      size_t q = p + 1;
      while (r.H(q) && !IsSig(s[q]) && s[q] != kEsc) q++;
      if (!(r.H(q) && AllMk(s + p + 1, q - p - 1))) {
        size_t d = p + 1;
        while (d < q && IsQuote(s[d])) d++;
        return WordAt(r, D, d, q, w, cc, stop);
      }
      size_t c0 = a;
      while (c0 < p && s[c0] != kCol) c0++;
      return WordAt(r, D, c0 < p ? c0 + 1 : a, p, w, cc, stop);
    }
    return WordAt(r, D, a, p, w, cc, stop);
  }
  return WordAt(r, D, j, kOffen, w, cc, stop);
}
// follow word of the article at i: quote runs and simple tags '<' name '>' (0x4C .. 0x4E)
// skipped, external link "[[" url ' ' text -> first word of the text
inline int Follow3(Rd& r, const Dict& D, size_t i, char* w, int* cc, int* stop) {
  const uint8_t* s = r.s;
  size_t j = i + 2;
  for (int it = 0; it < 8; it++) {
    const size_t j0 = j;
    while (r.H(j) && IsQuote(s[j])) j++;
    if (r.H(j) && s[j] == 0x4C) {
      size_t k = j + 1;
      while (r.H(k) && IsBody1(s[k])) k += TokLen(r, k, kOffen);
      if (k > j + 1 && r.H(k) && s[k] == 0x4E) j = k + 1;
    }
    if (j == j0) break;
  }
  if (r.H(j) && s[j] == kLb && r.H(j + 1) && s[j + 1] == kLb) {
    size_t k = j + 2;
    while (r.H(k) && s[k] != ' ' && !IsSig(s[k])) k++;
    if (r.H(k) && s[k] == ' ') {
      size_t e = k + 1;
      while (r.H(e) && !IsSig(s[e])) e++;
      return WordAt(r, D, k + 1, e, w, cc, stop);
    }
  }
  return FollowFrom(r, D, j, w, cc, stop);
}
// site at the token i (length 1): article code, one blank behind, a separator before (not a digit, letter or
// case mark, not _ ' - . &, not behind an escape byte), position behind start.  Reads only bytes the stage never
// changes, or article codes in places where 0xC4 and 0xC6 give the same answer, so input and output agree.
inline bool Site(Rd& r, size_t i, size_t start) {
  const uint8_t* s = r.s;
  if (!(s[i] == 0xC4 || s[i] == 0xC6)) return false;
  if (!r.H(i + 1) || s[i + 1] != 0x20 || !r.H(i + 2) || s[i + 2] == 0x20) return false;
  size_t a = i;
  if (a > 0 && s[a - 1] == 0x40) a--;
  if (a <= start || a < 2) return false;
  const uint8_t c = s[a - 1];
  if (c == 0x07 || c == 0x06 || c == 0x40 || IsBody1(c) || (c >= '0' && c <= '9') || c == '_' || c == kQ || c == '-' ||
      c == '.' || c == '&')
    return false;
  return s[a - 2] != kEsc;
}
inline bool Vor(const char* w, int n, const char* p) { const int k = (int)strlen(p); return n >= k && !memcmp(w, p, (size_t)k); }
// all-caps words read as a word (-> "a"): leading consonant cluster (y a vowel behind the first letter) is an
// onset of this table
inline bool OnsetOk(const char* w, int n) {
  char k[8];
  int i = 0;
  while (i < n && !strchr("aeiou", w[i]) && !(i && w[i] == 'y')) i++;
  if (i == n || i > 5) return false;
  k[0] = ' ';
  memcpy(k + 1, w, (size_t)i);
  k[i + 1] = ' ';
  k[i + 2] = 0;
  return strstr(" f fl fr h l m n r s sc sch sh sl sp st str sw ", k) != nullptr;
}
// 1 = "an"
inline int DefaultAn3(const char* w, int n, int cc, int stop) {
  if (n <= 0) return 0;
  if (w[0] >= '0' && w[0] <= '9') {
    if (w[0] == '8') return 1;
    return (n == 2 || n == 4 || n == 5 || n == 7) && (!memcmp(w, "11", 2) || !memcmp(w, "18", 2));
  }
  if (cc == 2) {
    if (strchr("fhlmnrsx", w[0]) && n >= 3 && OnsetOk(w, n)) return 0;
    return strchr("aefhilmnorsx", w[0]) != nullptr;
  }
  if (n == 1 && stop != kEsc) return strchr("aefhilmnorsx", w[0]) != nullptr;
  if (Vor(w, n, "eu") || Vor(w, n, "one") || Vor(w, n, "once")) return 0;
  if (Vor(w, n, "uni") && !(n > 3 && (w[3] == 'n' || w[3] == 'm' || w[3] == 'd'))) return 0;
  if (n >= 3 && w[0] == 'u' && strchr("bgkrstv", w[1]) && strchr("aeiou", w[2])) return 0;
  if (Vor(w, n, "hour") || Vor(w, n, "hono") || Vor(w, n, "heir")) return 1;
  return strchr("aeiou", w[0]) != nullptr;
}
// exact lookup in the exception list (case class and word)
inline bool InListe(const char* w, int n, int cc) {
  if (n > 15) return false;   // the table holds words of 1..15 bytes (class << 4 | length in one byte)
  const unsigned char k = (unsigned char)((unsigned)cc << 4 | (unsigned)n);
  for (const unsigned char* e = a_an_list::kListe; *e; e += 1 + (*e & 15))
    if (*e == k && !memcmp(e + 1, w, (size_t)n)) return true;
  return false;
}
inline int Rule3(const char* w, int n, int cc, int stop) {
  if (n <= 0) return 0;
  const int d = DefaultAn3(w, n, cc, stop);
  return InListe(w, n, cc) ? !d : d;
}

// ================================================================ encoder (batch; RS_ENC_ONLY)
RS_ENC_ONLY inline void EncCand(Buf* o, const uint8_t* x, size_t n, const CList* L) {
  for (int i = 0; i < L->n; i++)
    if (L->c[i].n == n && !memcmp(L->c[i].own, x, n)) { for (int k = 0; k < i; k++) o->Put1(kMk); return; }
  if (AllMk(x, n)) { for (size_t k = 0; k < n + (size_t)L->n; k++) o->Put1(kMk); return; }
  o->Put(x, n);
}
// a/an: length-preserving, output = input with the site bytes rewritten
RS_ENC_ONLY inline void AanEnc(const Dict& D, const uint8_t* s, size_t n, size_t start, Buf* o) {
  o->Put(s, n);
  if (o->err) return;
  Rd r{s, n, true, false};
  for (size_t i = start; i < n;) {
    const size_t L = TokLen(r, i, kOffen);
    if (L == 1 && Site(r, i, start)) {
      char w[128];
      int cc, stop;
      const int k = Follow3(r, D, i, w, &cc, &stop);
      const int an = s[i] == 0xC6;
      o->p[i] = an == Rule3(w, k, cc, stop) ? 0xC4 : 0xC6;
    }
    i += L;
  }
}
struct PageE { bool ok, in_title, in_body, bold; size_t ts, te, bs; };
RS_ENC_ONLY inline void PageByteE(PageE* P, const uint8_t* u, size_t p, size_t start) {
  if (u[p] != kLf) return;
  if (p >= 7 && !memcmp(u + p - 7, "\x20\x20\x4c\x2f\xdf\x98\x4e\x0a", 8)) {
    P->ok = p + 1 >= start; P->in_title = true; P->in_body = false; P->ts = p + 1;
    return;
  }
  if (P->in_title) { P->in_title = false; P->te = p; P->in_body = true; P->bs = p + 1; P->bold = false; }
}
RS_ENC_ONLY inline void TitelEnc(const Dict& D, const uint8_t* s, size_t n, size_t start, Buf* o, CList* L) {
  o->Put(s, start);
  Rd r{s, n, true, false};
  PageE P = {};
  for (size_t p = 0; p < start; p++) PageByteE(&P, s, p, start);
  size_t i = start;
  while (i < n) {
    const bool body = P.ok && P.in_body;
    size_t c, x, y;
    if (body && s[i] == kLb && i > 0 && s[i - 1] == kLf && T1Check(r, D, i, &c, &y)) {
      Cands1(s + P.ts, P.te - P.ts, L);
      o->Put(s + i, c + 1 - i); EncCand(o, s + c + 1, y - c - 1, L); o->Put1(kRb);
      i = y + 1;
      continue;
    }
    if (body && !P.bold && s[i] == kQ && i + 2 < n && s[i + 1] == kQ && s[i + 2] == kQ && i - P.bs < 3000) {
      P.bold = true;
      if (T2Scan(r, i, &x)) {
        Cands23(s + P.ts, P.te - P.ts, L);
        o->Put(s + i, 3); EncCand(o, s + i + 3, x - i - 3, L); o->Put(s + x, 3);
        i = x + 3;
        continue;
      }
    }
    if (body && s[i] == kLb && T3Struct(r, D, i, &x, &y)) {
      Cands23(s + P.ts, P.te - P.ts, L);
      o->Put(s + i, x + 1 - i); EncCand(o, s + x + 1, y - x - 1, L); o->Put1(kRb);
      i = y + 1;
      continue;
    }
    const size_t Lt = TokLen(r, i, kOffen);
    for (size_t k = 0; k < Lt; k++) { o->Put1(s[i + k]); PageByteE(&P, s, i + k, start); }
    i += Lt;
  }
}
RS_ENC_ONLY inline void LinkEnc(const Dict& D, const uint8_t* s, size_t n, size_t start, Buf* o, CList* L, uint8_t* tb) {
  o->Put(s, start);
  Rd r{s, n, true, false};
  size_t i = start;
  while (i < n) {
    size_t p, q;
    if (s[i] == kLb && LinkStruct(r, i, &p, &q) && !LinkExcluded(r, D, i + 1, p - i - 1)) {
      LinkCands(s + i + 1, p - i - 1, L, tb);
      o->Put(s + i, p + 1 - i); EncCand(o, s + p + 1, q - p - 1, L); o->Put1(kRb);
      i = q + 1;
      continue;
    }
    const size_t Lt = TokLen(r, i, kOffen);
    o->Put(s + i, Lt);
    i += Lt;
  }
}
// The three stages on in[0..n): out = link(titel2(aan(in))).  Bytes before start stay.  Peak besides in: two
// buffers of about n bytes (a/an and titel2 output, then titel2 output and out).
RS_ENC_ONLY inline bool Enc(const Dict& D, const uint8_t* in, size_t n, size_t start, Buf* out) {
  if (start > n) return false;
  Buf a, t;
  CList L;
  uint8_t tb[700];
  AanEnc(D, in, n, start, &a);
  if (!a.err) { t.Platz(n); TitelEnc(D, a.p, a.n, start, &t, &L); }
  a.Free();
  out->n = 0;
  if (!t.err) { out->Platz(t.n); LinkEnc(D, t.p, t.n, start, out, &L, tb); }
  const bool ok = !a.err && !t.err && !out->err;
  t.Free();
  return ok;
}

// ================================================================ decoder (online)
// Each stage: input window `in` (the previous stage appends to in.b), cursor i_ (a token start),
// emit floor em_ (input bytes below it are already emitted: raw bytes of a span whose rule is still open, but
// that every outcome copies unchanged), output to *out_.  Run(fin) processes what is there; fin: the input is
// complete.

// link^-1: D of a lone link is decided at its ']'; '[' T '|' are emitted before.
class LinkDec {
 public:
  Win in;
  void Init(const Dict* d, size_t start, Buf* out) { d_ = d; start_ = start; out_ = out; }
  void SetStart(size_t s) { start_ = s; }
  void Run(bool fin) {
    Rd r{in.S(), in.End(), fin, false};
    const uint8_t* s = r.s;
    const size_t na = r.na;
    if (i_ < start_) { const size_t b = na < start_ ? na : start_; Raw(s, i_, b); i_ = b; }
    while (i_ >= start_ && i_ < na) {
      const size_t i = i_;
      r.need = false;
      if (s[i] == kLb) {
        size_t p, q;
        const bool m = LinkStruct(r, i, &p, &q);
        if (r.need) {                                // open: '[' T '|' go out unchanged in every outcome
          size_t h = i + 1;
          while (h < na && !IsSig(s[h])) h++;
          h = h < na ? h + 1 : na;
          Raw(s, i, h);
          if (h > em_) em_ = h;
          break;
        }
        if (m && !LinkExcluded(r, *d_, i + 1, p - i - 1)) {
          LinkCands(s + i + 1, p - i - 1, &L_, tb_);
          Raw(s, i, p + 1); Cand(s, p + 1, q); Raw(s, q, q + 1);
          i_ = q + 1;
          continue;
        }
      }
      const size_t Lt = TokLen(r, i, kOffen);
      if (r.need) break;
      Raw(s, i, i + Lt);
      i_ = i + Lt;
    }
    in.Drop(i_ > 8 ? i_ - 8 : 0);
  }

 private:
  void Raw(const uint8_t* s, size_t a, size_t b) { if (a < em_) a = em_; if (b > a) out_->Put(s + a, b - a); }
  void Cand(const uint8_t* s, size_t a, size_t b) {
    const size_t k = b - a;
    if (!AllMk(s + a, k)) { Raw(s, a, b); return; }
    if (k < (size_t)L_.n) out_->Put(L_.c[k].own, L_.c[k].n);
    else for (size_t x = 0; x < k - L_.n; x++) out_->Put1(kMk);
  }
  const Dict* d_ = nullptr;
  size_t start_ = kOffen, i_ = 0, em_ = 0;
  Buf* out_ = nullptr;
  CList L_;
  uint8_t tb_[700];
};

// title^-1: page state, bold flag and title on the OUTPUT (the stream after a/an).
class TitelDec {
 public:
  Win in;
  size_t on = 0;                                     // bytes emitted
  size_t nsig = 0;                                   // structural bytes emitted
  long long lastsig = -1;                            // output position of the last one
  void Init(const Dict* d, size_t start, Buf* out) { d_ = d; start_ = start; out_ = out; }
  // may come late (no byte at or behind s processed yet): the running page is a rule page if its anchor LF is at
  // s - 1 or later, as decided with start known
  void SetStart(size_t s) { start_ = s; P_.ok = ank_ != kOffen && ank_ + 1 >= s; }
  void Run(bool fin) {
    Rd r{in.S(), in.End(), fin, false};
    const uint8_t* s = r.s;
    const size_t na = r.na;
    if (i_ < start_) {
      const size_t b = na < start_ ? na : start_;
      for (; i_ < b; i_++) { Out(s[i_]); if (s[i_] == kLf) Page(); }
    }
    while (i_ >= start_ && i_ < na) {
      const size_t i = i_;
      r.need = false;
      const size_t up = on - (em_ > i ? em_ - i : 0);      // output position of input byte i
      if (P_.ok && P_.in_body) {
        if (s[i] == kLb) {
          size_t c = 0, x, y;
          const bool t1 = i > 0 && s[i - 1] == kLf;
          if (t1) {
            const bool m = T1Check(r, *d_, i, &c, &y);
            if (r.need) { Halt(s, na, i, up, c); break; }
            if (m) {
              Cands1(tt_, tn_, &L_);
              Raw(s, i, c + 1); Cand(s, c + 1, y); Raw(s, y, y + 1);
              i_ = y + 1;
              continue;
            }
          }
          const bool m = T3Struct(r, *d_, i, &x, &y);
          if (r.need) { Halt(s, na, i, up, 0); break; }
          if (m) {
            Cands23(tt_, tn_, &L_);
            Raw(s, i, x + 1); Cand(s, x + 1, y); Raw(s, y, y + 1);
            i_ = y + 1;
            continue;
          }
        } else if (s[i] == kQ && !P_.bold) {
          const bool trig = r.H(i + 1) && s[i + 1] == kQ && r.H(i + 2) && s[i + 2] == kQ && up - P_.bs < 3000;
          if (r.need) break;
          if (trig) {
            size_t cl = 0;
            const bool ok = T2Scan(r, i, &cl);
            if (r.need) break;
            P_.bold = true;
            if (ok) {
              Cands23(tt_, tn_, &L_);
              Raw(s, i, i + 3); Cand(s, i + 3, cl); Raw(s, cl, cl + 3);
              i_ = cl + 3;
              continue;
            }
          }
        }
      }
      const size_t Lt = TokLen(r, i, kOffen);
      if (r.need) break;
      for (size_t x = i < em_ ? em_ : i; x < i + Lt; x++) { Out(s[x]); if (s[x] == kLf) Page(); }
      i_ = i + Lt;
    }
    in.Drop(i_ > 8 ? i_ - 8 : 0);
  }

 private:
  struct Pg { bool ok, in_title, in_body, bold; size_t bs; };
  void Out(uint8_t c) {
    out_->Put1(c);
    const size_t pos = on++;
    sh_ = sh_ << 8 | c;
    if (P_.in_title && c != kLf && tn_ < 601) tt_[tn_++] = c;
    if (IsSig(c)) { nsig++; lastsig = (long long)pos; }
  }
  // page tracking for the LF just emitted (only LFs of the token copy reach it)
  void Page() {
    const size_t p = on - 1;
    if (sh_ == kAnker) { ank_ = p; P_.ok = p + 1 >= start_; P_.in_title = true; P_.in_body = false; tn_ = 0; return; }
    if (P_.in_title) { P_.in_title = false; P_.in_body = true; P_.bs = p + 1; P_.bold = false; }
  }
  void Raw(const uint8_t* s, size_t a, size_t b) { for (a = a < em_ ? em_ : a; a < b; a++) Out(s[a]); }
  void Cand(const uint8_t* s, size_t a, size_t b) {
    const size_t k = b - a;
    if (!AllMk(s + a, k)) { Raw(s, a, b); return; }
    if (k < (size_t)L_.n) for (size_t x = 0; x < L_.c[k].n; x++) Out(L_.c[k].own[x]);
    else for (size_t x = 0; x < k - L_.n; x++) Out(kMk);
  }
  void Halt(const uint8_t* s, size_t na, size_t i, size_t up, size_t c) {
    size_t z = i + 1, h = na;
    while (z < na && !IsSig(s[z])) z++;
    if (z < na) h = s[z] == kLf ? z : z + 1;
    if (!P_.bold && up - P_.bs < 3000)
      for (size_t x = i + 1; x < h; x++) if (s[x] == kQ) { h = x; break; }
    if (c && h > c + 1 && AllMk(s + c + 1, (z < h ? z : h) - c - 1)) h = c + 1;
    Raw(s, i, h);
    if (h > em_) em_ = h;
  }
  const Dict* d_ = nullptr;
  size_t start_ = kOffen, i_ = 0, em_ = 0;
  Buf* out_ = nullptr;
  Pg P_ = {};
  uint64_t sh_ = 0;                                  // last 8 output bytes
  size_t ank_ = kOffen;                              // output position of the last anchor LF
  uint8_t tt_[601];                                  // title line of the page (output), capped at 601
  size_t tn_ = 0;
  CList L_;
};

// a/an^-1: length-preserving; a site waits until its follow word is complete, later sites are decided anyway
// (their words never depend on an earlier site's value).  Emits up to the first open site.
class AanDec {
 public:
  Win in;
  size_t done = 0;                                   // bytes emitted
  void Init(const Dict* d, size_t start, Buf* out) { d_ = d; start_ = start; out_ = out; }
  void SetStart(size_t s) { start_ = s; }
  size_t Offen() const { return np_; }               // sites waiting for their follow word
  void Run(bool fin) {
    Rd r{in.S(), in.End(), fin, false};
    size_t k = 0;
    for (size_t q = 0; q < np_; q++) if (!Decide(r, pend_[q])) pend_[k++] = pend_[q];
    np_ = k;
    if (i_ < start_) i_ = r.na < start_ ? r.na : start_;
    while (i_ >= start_ && i_ < r.na) {
      r.need = false;
      const size_t L = TokLen(r, i_, kOffen);
      if (r.need) break;
      if (L == 1) {
        const bool site = Site(r, i_, start_);
        if (r.need) break;
        if (site && !Decide(r, i_)) Push(i_);
      }
      i_ += L;
    }
    const size_t e = np_ ? pend_[0] : i_;
    if (e > done) { out_->Put(in.S() + done, e - done); done = e; }
    in.Drop(e > 8 ? e - 8 : 0);
  }
  void Free() { free(pend_); pend_ = nullptr; np_ = cp_ = 0; }

 private:
  bool Decide(Rd& r, size_t i) {
    r.need = false;
    char w[128];
    int cc, stop;
    const int k = Follow3(r, *d_, i, w, &cc, &stop);
    if (r.need) return false;
    const int R = Rule3(w, k, cc, stop);
    uint8_t* s = in.S();
    const int an = s[i] == 0xC4 ? R : !R;
    s[i] = an ? 0xC6 : 0xC4;
    return true;
  }
  void Push(size_t i) {
    if (np_ == cp_) {
      const size_t c = cp_ ? 2 * cp_ : 64;
      size_t* q = (size_t*)realloc(pend_, c * sizeof(size_t));
      if (!q) { out_->err = true; return; }
      pend_ = q; cp_ = c;
    }
    pend_[np_++] = i;
  }
  const Dict* d_ = nullptr;
  size_t start_ = kOffen, i_ = 0;
  Buf* out_ = nullptr;
  size_t* pend_ = nullptr;
  size_t np_ = 0, cp_ = 0;
};

// The decoder chain link^-1 -> titel2^-1 -> a/an^-1; the caller appends the coder stream to link.in.b.
struct Kette {
  LinkDec link;
  TitelDec titel;
  AanDec aan;
  void Init(const Dict* d, size_t start, Buf* out) {
    link.Init(d, start, &titel.in.b); titel.Init(d, start, &aan.in.b); aan.Init(d, start, out);
  }
  void SetStart(size_t s) { link.SetStart(s); titel.SetStart(s); aan.SetStart(s); }
  void Run(bool fin) { link.Run(fin); titel.Run(fin); aan.Run(fin); }
  bool Err() const { return link.in.b.err || titel.in.b.err || aan.in.b.err; }
  void Free() { link.in.b.Free(); titel.in.b.Free(); aan.in.b.Free(); aan.Free(); }
};

// Inverse of Enc over a whole buffer: out = original (the stream before the three stages).
inline bool Dec(const Dict& D, const uint8_t* in, size_t n, size_t start, Buf* out) {
  Kette k;
  k.Init(&D, start, out);
  out->n = 0;
  for (size_t a = 0; a < n;) {                       // 1 MiB pieces: the windows stay small
    const size_t m = n - a < ((size_t)1 << 20) ? n - a : ((size_t)1 << 20);
    k.link.in.b.Put(in + a, m);
    a += m;
    k.Run(false);
  }
  k.Run(true);
  const bool ok = !k.Err() && !out->err;
  k.Free();
  return ok;
}

class Sicht {
 public:
  Sicht() { k_.Init(&d_, kOffen, &ob_); ob_.Platz(4 * kHalt + 65536); }
  Sicht(const Sicht&) = delete;
  Sicht& operator=(const Sicht&) = delete;
  void Init(const Dict& d) { d_ = d; }                // the word arrays must stay alive
  void Start(size_t pos) { k_.SetStart(pos); }
  void Byte(uint8_t b) {
    frei_n_ = 0;
    jetzt_ = false;
    t_ = t_ << 8 | b;
    k_.link.in.b.Put1(b);
    if (!IsSig(b)) return;
    ++nsig_;
    Freigabe(false);
    letzt_ = k_.titel.nsig == nsig_ ? k_.titel.lastsig : -1;
    if (t_ == kAnker) {
      nach_ = true; tok_ = false; tit_.n = 0;
      ta_ = letzt_ >= 0 ? letzt_ + 1 : -1;
    } else if (nach_ && b == kLf) {
      nach_ = false; jetzt_ = true;
      tok_ = ta_ >= 0 && letzt_ >= ta_ && fertig_ > letzt_;
      tn_ = tok_ ? letzt_ - ta_ : 0;
    }
  }
  void Ende() { frei_n_ = 0; jetzt_ = false; Freigabe(true); }
  long long Letzt() const { return letzt_; }
  int Frei(const uint8_t** p) const { *p = O() + frei_a_; return (int)frei_n_; }
  int Titel(const uint8_t** p) const { *p = tit_.p; return tok_ ? (int)tn_ : -1; }
  bool TitelJetzt() const { return jetzt_; }
  const uint8_t* O() const { return (const uint8_t*)((uintptr_t)ob_.p - (uintptr_t)ob0_); }
  long long Fertig() const { return fertig_; }        // original bytes released so far
  long long Basis() const { return ob0_; }
  bool Err() const { return k_.Err() || ob_.err || tit_.err; }
  void Free() { k_.Free(); ob_.Free(); tit_.Free(); }

 private:
  static constexpr size_t kHalt = (size_t)1 << 17;   // O() keeps at least this much behind Fertig()
  void Freigabe(bool fin) {
    if (ob_.n > 4 * kHalt) {
      const size_t d = ob_.n - kHalt;
      memmove(ob_.p, ob_.p + d, kHalt);
      ob_.n = kHalt; ob0_ += (long long)d;
    }
    const size_t a = ob_.n;
    k_.Run(fin);
    frei_a_ = ob0_ + (long long)a;
    frei_n_ = ob_.n - a;
    fertig_ = ob0_ + (long long)ob_.n;
    if (nach_ && ta_ >= 0 && fertig_ > ta_) {        // title bytes (behind the anchor LF)
      const long long x = frei_a_ > ta_ ? frei_a_ : ta_;
      tit_.Put(O() + x, (size_t)(fertig_ - x));
    }
  }
  Dict d_;
  Kette k_;
  Buf ob_, tit_;
  long long ob0_ = 0, frei_a_ = 0, fertig_ = 0, letzt_ = -1, ta_ = -1, tn_ = 0;
  size_t frei_n_ = 0, nsig_ = 0;
  uint64_t t_ = 0;
  bool nach_ = false, tok_ = false, jetzt_ = false;
};

}  // namespace rule_stages

#endif
