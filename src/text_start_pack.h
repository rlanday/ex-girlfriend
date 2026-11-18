// Text start, file stage.  Behind the line feed that ends the whole metadata
// block (behind the census line) the stream carries up to three slots, each
// closed by 0x0B 0x0B, in front of the first page:
//   slot 1  dictionary article: the word list ./.dict; a later word is written
//           with the codes of the earlier ones, as words are in the text
//   slot 2  author article: one line per author name moved out of part B
//   slot 3  census pseudo article: the template texts of the census block
// The slots are coded as text by the full model, where it can learn the words,
// codes, names and templates the pages use; slot 1 is the dictionary itself,
// so the archive carries no separate dictionary block.
//
// Pack, the last stage in front of the coder, puts the present slots behind
// that line feed and moves every author name of part B into slot 2:
//   part B before   ts TAB name [TAB uid] LF      (name WRT-coded)
//   part B after    ts TAB <mark> [TAB uid] LF
//   slot 2          name LF   (one line per moved record, part-B order)
// Which records move follows the decoder rule of the metadata header sort on
// the coded bytes: a third field with content = name (enters the name table),
// an empty third field = address with the collision mark (stays), two fields =
// name if the table knows it, else address (stays).  The kind of a moved
// record: new author (the uid follows), the same name as the previous named
// record, or another repetition.  The reverse does not depend on that rule:
// Unpack replaces exactly the fields that are a mark.
// Unpack, the first stage behind the decoder, reads the slots behind the
// metadata block, writes slots 1 and 3 back to their hand-over files and puts
// the article lines back in place of the marks.
// Slot interface:
//   coder    the producing stage writes the slot content to `.vs1` / `.vs3`
//            before Pack runs; Pack appends the end mark 0x0B 0x0B and
//            removes the file.  A missing file is an empty slot.
//   contract bytes of the frozen 208-byte set, already in coder form (WRT),
//            no 0x0B 0x0B pair, not ending with 0x0B, no page anchor, slots 1
//            and 2 no link opener; Pack checks all of it and stops loudly.
//   decoder  Unpack writes the content back to `.vs1` / `.vs3` before every
//            other inverse stage runs; an empty slot leaves no file.
//   model    the metadata block reader holds the content from the end of the
//            slot until the line behind the text start.
#ifndef TEXT_START_PACK_H_
#define TEXT_START_PACK_H_
#include "meld.h"
#include "u8_append.h"

#include "metablock.h"

#include "ip_pack.h"     // LiesDatei / SchreibDatei
#include "vocab_frozen.h"     // the frozen byte set
#include "preprocess/dictionary.h"   // kWrtSubMin, the partial-word rule of the WRT stage
#include "enc_only.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

namespace text_start {

static const unsigned char kEnde = 0x0B;
static const unsigned char kNL = 0x0A;
static const unsigned char kTab = 0x09;
// `  </page>\n` post-WRT: the page anchor (author_block.h kAnker); no slot may
// carry it, or every page reader would count a page.
static const unsigned char kAnker[8] = { 0x20, 0x20, 0x4c, 0x2f, 0xdf, 0x98, 0x4e, 0x0a };

inline const char* SlotDatei(int bit) {
  return bit == 1 ? ".vs1" : (bit == 4 ? ".vs3" : "");
}
inline bool IstMarke(unsigned char c) {
  return c == (unsigned char)0x05 || c == (unsigned char)0x03 ||
         c == (unsigned char)0x03;
}

__attribute__((noinline)) static void Fail(const char* what) {
  fprintf(stderr, "\n%s\n", what);
  fflush(stderr);
  exit(1);
}

// The boundaries of a stream without text start, by the one boundary rule
// (with the census line exactly when the model side expects it):
// [*ab, *bis) holds the N part-B lines, *ende is the position behind the line
// feed that ends the whole metablock.  False if the block does not close.
inline bool Grenzen(const std::vector<unsigned char>& d, size_t* ab,
                    size_t* bis, size_t* ende) {
  metablock::Grenze gr;
  gr.Zensus();
  const long long n_a = (long long)243425;
  *ab = *bis = *ende = 0;
  for (size_t p = 0; p < d.size(); ++p) {
    const int r = gr.Byte(d[p]);
    if (r == metablock::Grenze::kBlockzeile) {
      const long long i = gr.ZeileIdx();
      if (i == n_a - 1) *ab = p + 1;
      if (i == 2 * n_a - 1) *bis = p + 1;
    }
    if (gr.Endet()) { *ende = p + 1; return *bis > *ab && *ab > 0; }
  }
  return false;
}

// The name table of the decoder rule, on coded bytes: open addressing over
// (offset, length) into the stream, FNV-1a.  45,778 distinct names on enwik9
// against 2^18 places; a half full table stops loudly.
class Namen {
 public:
  explicit Namen(const unsigned char* d) : d_(d), off_(kN, 0u), len_(kN, 0u) {}
  // true if known; with `rein` it enters an unknown one
  ENC_ONLY bool Finde(size_t o, unsigned int l, bool rein) {
    unsigned long long h = 1469598103934665603ull;
    for (unsigned int i = 0; i < l; ++i) { h ^= d_[o + i]; h *= 1099511628211ull; }
    size_t m = (size_t)(h & (kN - 1));
    while (len_[m]) {
      if (len_[m] == l + 1 && memcmp(d_ + off_[m], d_ + o, l) == 0) return true;
      m = (m + 1) & (kN - 1);
    }
    if (rein) {
      if (++n_ * 2 >= kN) __builtin_unreachable();
      off_[m] = (unsigned int)o;
      len_[m] = l + 1;               // 0 marks a free place
    }
    return false;
  }

 private:
  static const size_t kN = 1u << 18;
  const unsigned char* d_;
  std::vector<unsigned int> off_, len_;
  size_t n_ = 0;
};

// a slot content: frozen byte set, no end-mark pair, not ending with 0x0B,
// no page anchor; slots 1 and 2 carry no link opener (the link readers are
// cut off there, the census automaton reads slot 3 itself)
ENC_ONLY inline void SlotPruefe(const std::vector<unsigned char>& v, int bit) {
  unsigned char vor = 0;
  for (size_t k = 0; k < v.size(); ++k) {
    if (!vocab_frozen::InSet(v[k]))
      __builtin_unreachable();
    if (v[k] == kEnde && vor == kEnde)
      __builtin_unreachable();
    if (bit != 4 && v[k] == 0x5b && vor == 0x5b)
      __builtin_unreachable();
    vor = v[k];
  }
  if (!v.empty() && v.back() == kEnde)
    __builtin_unreachable();
  if (v.size() >= 8)
    for (size_t k = 0; k + 8 <= v.size(); ++k)
      if (v[k] == kAnker[0] && memcmp(&v[k], kAnker, 8) == 0)
        __builtin_unreachable();
}

// `behalten` (keep): leave the hand-over files in place (the -e path removes them)
ENC_ONLY inline bool Pack(const std::string& pfad, bool aktiv, bool behalten) {
#if AUX_BUILD
  (void)pfad; (void)aktiv; (void)behalten;
  return true;
#else
  if (!aktiv) return true;
  std::vector<unsigned char> d;
  if (!ip_pack::LiesDatei(pfad.c_str(), &d)) return false;
  std::vector<unsigned char> slot[3];
  for (int i = 0; i < 3; i += 2) {
    if (!((7) & (1 << i))) continue;
    const char* f = SlotDatei(1 << i);
    if (ip_pack::LiesDatei(f, &slot[i])) { if (!behalten) remove(f); }
    else slot[i].clear();
  }
  size_t ab = 0, bis = 0, ende = 0;
  if (!Grenzen(d, &ab, &bis, &ende))
    __builtin_unreachable();
  unsigned long long namen = 0, kl[3] = {0, 0, 0};
  for (size_t p = ab; p < bis; ++p)
    if (d[p] == 0x01 || d[p] == 0x03 || d[p] == 0x05)
      __builtin_unreachable();
  Namen tafel(d.data());
  FILE* f = fopen(pfad.c_str(), "wb");
  if (!f) return false;
  bool ok = true;
  ok = ok && fwrite(d.data(), 1, ab, f) == ab;
  // part B with the marks, written piece by piece in stream order
  size_t nb = 0;
  auto stueck = [&](const unsigned char* x, size_t n) {
    ok = ok && (!n || fwrite(x, 1, n, f) == n);
    nb += n;
  };
  size_t vor_o = 0; unsigned int vor_l = 0; bool vor_da = false;   // previous named record
  for (size_t s = ab; s < bis;) {
    size_t e = s;
    while (d[e] != kNL) ++e;                     // the LF of this line
    size_t t1 = s;
    while (t1 < e && d[t1] != kTab) ++t1;
    if (t1 == e) __builtin_unreachable();
    size_t t2 = t1 + 1;
    while (t2 < e && d[t2] != kTab) ++t2;
    const size_t f1 = t1 + 1;
    const unsigned int l1 = (unsigned int)(t2 - f1);
    bool name;
    int art = 2;                                   // 0 new, 1 same as before, 2 repetition
    if (t2 < e && t2 + 1 < e) { tafel.Finde(f1, l1, true); name = true; art = 0; }   // uid
    else if (t2 < e) name = false;                 // collision mark: address
    else name = tafel.Finde(f1, l1, false);        // known name, else address
    if (name && art != 0 && vor_da && vor_l == l1 &&
        memcmp(d.data() + vor_o, d.data() + f1, l1) == 0)
      art = 1;
    if (name) {
      ++namen; ++kl[art];
      vor_o = f1; vor_l = l1; vor_da = true;
      const unsigned char mk = art == 0 ? (unsigned char)0x05
          : (art == 1 ? (unsigned char)0x03 : (unsigned char)0x03);
      u8_append::Anh(slot[1], d.begin() + (long)f1, d.begin() + (long)t2);
      slot[1].push_back(kNL);
      stueck(d.data() + s, f1 - s);
      stueck(&mk, 1);
      stueck(d.data() + t2, e + 1 - t2);
    } else {
      stueck(d.data() + s, e + 1 - s);
    }
    s = e + 1;
  }
  for (int i = 0; i < 3; ++i) SlotPruefe(slot[i], 1 << i);
  static const unsigned char kMark2[2] = { kEnde, kEnde };
  unsigned long long ta = 0;
  ok = ok && (ende == bis || fwrite(d.data() + bis, 1, ende - bis, f) == ende - bis);
  for (int i = 0; i < 3; ++i) {
    if (!((7) & (1 << i))) continue;
    if (!slot[i].empty()) ok = ok && fwrite(slot[i].data(), 1, slot[i].size(), f) == slot[i].size();
    ok = ok && fwrite(kMark2, 1, 2, f) == 2;
    ta += slot[i].size() + 2;
  }
  ok = ok && (d.size() == ende || fwrite(d.data() + ende, 1, d.size() - ende, f) == d.size() - ende);
  ok = (fclose(f) == 0) && ok;
  (void)namen; (void)kl; (void)nb;
  return ok;
#endif
}

inline bool Unpack(const std::string& pfad, bool aktiv) {
#if AUX_BUILD
  (void)pfad; (void)aktiv;
  return true;
#else
  if (!aktiv) return true;
  std::vector<unsigned char> d;
  if (!ip_pack::LiesDatei(pfad.c_str(), &d)) return false;
  size_t ab = 0, bis = 0, ende = 0;
  if (!Grenzen(d, &ab, &bis, &ende))
    __builtin_unreachable();
  // the K slots behind the metablock
  size_t p = ende, a2 = 0, e2 = 0;     // slot 2 content [a2, e2)
  for (int i = 0; i < 3; ++i) {
    if (!((7) & (1 << i))) continue;
    const size_t a = p;
    unsigned char vor = 0;
    for (;; ++p) {
      if (p >= d.size()) __builtin_unreachable();
      if (d[p] == kEnde && vor == kEnde) break;
      vor = d[p];
    }
    const size_t e = p - 1;          // content [a, e)
    ++p;
    if (i == 1) { a2 = a; e2 = e; continue; }
    const char* f = SlotDatei(1 << i);
    if (e > a) {   // the slot file straight from the stream
      FILE* sf = fopen(f, "wb");
      if (!sf) return false;
      const bool sok = fwrite(d.data() + a, 1, e - a, sf) == e - a;
      fclose(sf);
      if (!sok) return false;
    } else {
      remove(f);
    }
  }
  FILE* f = fopen(pfad.c_str(), "wb");
  if (!f) return false;
  bool ok = true;
  ok = ok && fwrite(d.data(), 1, ab, f) == ab;
  // part B back, written piece by piece in stream order
  auto stueck = [&](size_t x, size_t y) { ok = ok && (y == x || fwrite(d.data() + x, 1, y - x, f) == y - x); };
  size_t z = a2;                     // next article line
  for (size_t s = ab; s < bis;) {
    size_t e = s;
    while (d[e] != kNL) ++e;
    size_t t1 = s;
    while (t1 < e && d[t1] != kTab) ++t1;
    if (t1 + 1 < e && IstMarke(d[t1 + 1]) &&
        (t1 + 2 == e || d[t1 + 2] == kTab)) {
      size_t ze = z;
      while (ze < e2 && d[ze] != kNL) ++ze;
      if (ze >= e2) __builtin_unreachable();
      stueck(s, t1 + 1);
      stueck(z, ze);
      stueck(t1 + 2, e + 1);
      z = ze + 1;
    } else {
      stueck(s, e + 1);
    }
    s = e + 1;
  }
  if (z != e2) __builtin_unreachable();
  ok = ok && (ende == bis || fwrite(d.data() + bis, 1, ende - bis, f) == ende - bis);
  ok = ok && (d.size() == p || fwrite(d.data() + p, 1, d.size() - p, f) == d.size() - p);
  ok = (fclose(f) == 0) && ok;
  return ok;
#endif
}

}  // namespace text_start

namespace fxcmv1 { void DictVorab(); }
namespace dict_codes {

constexpr unsigned long long kLaufAb = 80 + 3840;
// the fixed tables of the sort rule (runner.cpp dict_runs): letters of all
// words back to back and letters of one word; the coder stops on a
// dictionary beyond them (E551), so both sides never cut a word (enwik9:
// 465,947 letters, words up to 58 letters)
constexpr unsigned kLaufTxt = 1u << 19, kLaufWort = 128;
// both sides: the slot without its run marks (./.dict has no empty line)
__attribute__((noinline)) inline void Entmarken(std::vector<unsigned char>* v) {
  size_t m = 0;
  for (size_t k = 0; k < v->size(); ++k)
    if (!((*v)[k] == 0x0A && m && (*v)[m - 1] == 0x0A)) (*v)[m++] = (*v)[k];
  v->resize(m);
}
// the rank of the code word at b[0..m), its length in *k; -1 outside the plan
inline long Rang(const unsigned char* b, size_t m, size_t* k) {
  const unsigned b0 = b[0];
  if (b0 < 0x80) return -1;
  *k = 1;
  if (b0 <= 0xCF) return (long)(b0 - 0x80);
  if (m < 2 || b[1] < 0x80) return -1;
  const unsigned b1 = b[1];
  *k = 2;
  if (b1 <= 0xCF) return (long)(80 + (b0 - 0xD0) * 80 + (b1 - 0x80));
  if (m < 3 || b0 < 0xF0 || b[2] < 0x80 || b[2] > 0xCF) return -1;
  *k = 3;
  const unsigned b2 = b[2] - 0x80;
  if (b1 <= 0xEF) return (long)(3920 + ((b0 - 0xF0) * 32 + (b1 - 0xD0)) * 80 + b2);
  return (long)(44880 + ((b0 - 0xF0) * 16 + (b1 - 0xF0)) * 80 + b2);
}
// the decoder (both sides): the dictionary from the coded slot; a code may
// only name a word decoded before it.  False on any other byte, an empty
// word or a missing last line feed.  *codes: the number of codes read.
__attribute__((noinline)) inline bool Dekodieren(const unsigned char* b, size_t n,
                                                 std::vector<unsigned char>* out,
                                                 unsigned long* codes) {
  std::vector<unsigned> ab;            // start of word i in *out
  out->clear();
  out->reserve(n + (n >> 2));
  size_t s = 0;                        // start of the current word
  *codes = 0;
  for (size_t i = 0; i < n;) {
    const unsigned char c = b[i];
    if (c == 0x0A) {
      if (out->size() == s) return false;
      out->push_back(c);
      ab.push_back((unsigned)s);
      s = out->size();
      ++i;
    } else if (c >= 0x61 && c <= 0x7A) {
      out->push_back(c);
      ++i;
    } else {
      size_t k = 0;
      const long r = Rang(b + i, n - i, &k);
      if (r < 0 || (size_t)r >= ab.size()) return false;
      const size_t e = ((size_t)r + 1 < ab.size() ? ab[r + 1] : s) - 1;
      for (size_t q = ab[r]; q < e; ++q) { const unsigned char x = (*out)[q]; out->push_back(x); }
      ++*codes;
      i += k;
    }
  }
  return s == out->size() && s;
}
// the code word of rank r: preprocessor::CodeWord (dictionary.cpp)
// the word table of the coder: FNV-1a, linear probing (as FastWordMap)
struct Tafel {
  const unsigned char* t = nullptr;
  std::vector<unsigned> tab, off, len;
  unsigned mask = 0;
  ENC_ONLY unsigned Hash(const unsigned char* p, unsigned l) const {
    unsigned h = 2166136261u;
    for (unsigned q = 0; q < l; ++q) { h ^= p[q]; h *= 16777619u; }
    return h & mask;
  }
  ENC_ONLY long Finde(const unsigned char* p, unsigned l) const {
    for (unsigned h = Hash(p, l); tab[h]; h = (h + 1) & mask) {
      const unsigned j = tab[h] - 1;
      if (len[j] == l && !memcmp(t + off[j], p, l)) return (long)j;
    }
    return -1;
  }
};
// the coder: ./.dict (a-z lines) in the incremental form.  Word i is looked
// up only among the words 0..i-1 (they enter the table behind it, so a
// repeated word finds its first occurrence, as FastWordMap does).
ENC_ONLY inline void Kodieren(const std::vector<unsigned char>& d,
                              std::vector<unsigned char>* o,
                              unsigned long* words, unsigned long* codes) {
  const unsigned char* t = d.data();
  const size_t L = d.size();
  unsigned mask = 1;
  while (mask < L) mask <<= 1;         // > 2 x the number of words
  Tafel ta;
  ta.t = t;
  ta.tab.assign(mask, 0u);
  ta.mask = mask - 1;
  std::vector<unsigned>& off = ta.off;
  std::vector<unsigned>& len = ta.len;
  o->clear();
  o->reserve(L);
  unsigned char cw[3];
  *codes = 0;
  unsigned nmax = 0;                   // the longest word (run-mark tables, E551)
  size_t s = 0;
  for (size_t p = 0; p < L; ++p) {
    const unsigned char c = t[p];
    if (c >= 0x61 && c <= 0x7A) continue;
    if (c != 0x0A || p == s)
      __builtin_unreachable();
    const unsigned char* w = t + s;
    const unsigned n = (unsigned)(p - s), i = (unsigned)off.size();
    long r = -1;                       // the code in front of the letters
    unsigned a = 0;                    // letters [a, n)
    const unsigned m = preprocessor::kWrtSubMin;   // the stream rule: earlier words of at least 7 letters
    // the stream rule (Dictionary::EncodeSubstring): the longest earlier
    // prefix, no suffix code
    if (n > m)
      for (unsigned l = n - 1; l >= m && r < 0; --l) if ((r = ta.Finde(w, l)) >= 0) a = l;
    // the run mark: an empty line in front of every word from kLaufAb + 1 on
    // that is not greater than its predecessor (bytewise, a prefix is smaller)
    if (i > kLaufAb) {
      const unsigned n0 = len[i - 1];
      const int v0 = memcmp(w, t + off[i - 1], n < n0 ? n : n0);
      const bool mk = v0 < 0 || (v0 == 0 && n <= n0);
      if (mk) o->push_back(0x0A);
    }
    if (n > nmax) nmax = n;
    if (r >= 0) {
      u8_append::Anh(*o, cw, preprocessor::CodeWord((unsigned)r, cw));
      ++*codes;
    }
    u8_append::Anh(*o, w + a, w + n);
    o->push_back(0x0A);
    off.push_back((unsigned)s);
    len.push_back(n);
    unsigned h = ta.Hash(w, n);
    while (ta.tab[h]) h = (h + 1) & ta.mask;
    ta.tab[h] = i + 1;
    s = p + 1;
  }
  if (s != L || !L)
    __builtin_unreachable();
  if (L - off.size() > kLaufTxt || nmax > kLaufWort)
    __builtin_unreachable();
  *words = off.size();
}
}  // namespace dict_codes
namespace dict_article {
inline unsigned long long strom = 0;
inline bool wrt = false;
ENC_ONLY inline bool SlotDatei() {
#if AUX_BUILD
  return true;
#else
  std::vector<unsigned char> v;
  ip_pack::LiesDatei(".dict", &v);
  {
    std::vector<unsigned char> k;
    unsigned long w = 0, c = 0;
    dict_codes::Kodieren(v, &k, &w, &c);
    return ip_pack::SchreibDatei(".vs1", k);
  }
#endif
}
// both sides, before the model is built: an existing ./.dict waits as
// .dict.ta until the end of slot 1 (the coder always has one, the decoder of
// archive9 has none).  A stale .dict.ta of an earlier run goes first.
__attribute__((noinline)) inline void Verbergen() {
#if !AUX_BUILD
  remove(".dict.ta");
  rename(".dict", ".dict.ta");
  fxcmv1::DictVorab();   // fxcm: the code-word ranks in front of slot 1
#endif
}
// both sides, at the last byte of slot 1: ./.dict from the slot
__attribute__((noinline)) inline void Schreiben(const unsigned char* b, size_t n) {
  std::vector<unsigned char> v;
  unsigned long c = 0;
  std::vector<unsigned char> u; u8_append::Anh(u, b, b + n);   // without the run marks
  dict_codes::Entmarken(&u);
  b = u.data();
  n = u.size();
  if (!dict_codes::Dekodieren(b, n, &v, &c))
    __builtin_unreachable();
  ip_pack::SchreibDatei(".dict", v);
}
// the coder, behind the last byte: the hidden copy of ./.dict goes (the
// dictionary of slot 1 is the one the run started with)
ENC_ONLY inline void Pruefen() {
#if !AUX_BUILD
  remove(".dict.ta");
#endif
}

__attribute__((noinline)) inline bool DateiAusSlot() {
  std::vector<unsigned char> s, v;
  unsigned long c = 0;
  if (!ip_pack::LiesDatei(".vs1", &s)) return false;
  dict_codes::Entmarken(&s);   // without the run marks
  if (!dict_codes::Dekodieren(s.data(), s.size(), &v, &c))
    __builtin_unreachable();
  ip_pack::SchreibDatei(".dict", v);
  remove(".vs1");
  return true;
}
}  // namespace dict_article

#endif
