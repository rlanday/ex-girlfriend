// Census block, page level: the template form of the census pages.
//
// The machine-written census pages (title "<name>, <US state>", census
// paragraph "... people, ... households, and ... families residing in the
// ...") follow ONE automatic form per census type.  The page is cut into units
// (lines, sentences outside links); a unit is abstracted (numbers and list
// directions -> value slots, page name / state / title / postal code ->
// placeholders, the name part of a state link "[[X, State]]" -> text slot).
// Units of one sentence family (same skeleton: link targets, footnote form,
// tense and spacing removed) form a position of the automatic form; the
// abstracted units of a position are its variants (subforms).  Positions are
// ordered by their mean place in the pages of the type.  Only families of the
// automatic form count (value slots or placeholders, the section headings of
// the bot, on at least 2 % of the pages of the type); everything else stays
// text, also recurring sentences written by people.
//
// Per page the census block holds: per position a flag (absent, variant v, or
// a list of instances), the gap bits (is there text between two automatic
// units), and the values.  The page keeps the text of the deviations (each
// run closed by 0x01) and the text slots (names, each closed by 0x02).  The
// texts of the variants are written into the block once.  Every automatic unit
// is checked byte-wise by the encoder (expansion == original).
//
// Stages (runner.cpp):
//   -e   the census transform (census_transform.h) skips every page this form takes (Nimmt)
//        reorder() -> Zerlege(".main_reordered")    stubs + side file .censblock
//        RunCompression before the metablock codec -> Einfuegen(temp)   block behind the metablock
//   dec  RunDecompression after the metablock codec -> Entnehmen(temp)   block -> .censblock
//        phda9_resto() -> Setze(".main_decomp_restored")       pages restored
// The page loop sees the same page sequence at both points (stream order,
// behind reorder()), so a record names its page by the page index.
//
// The census line: ONE line in the frozen 208-byte alphabet of the coder stream
// (every byte is checked by the writer), behind the last line of the metadata
// block.  The variant texts are not in it; they travel in the pseudo article.
//   "zcb3"  for each type of kTypen: npos[w2] { nvar[w2] { nlab[w1] { label[w1] [year[w2]] }* }* }*
//   first page[w3] count[w3] { per page of that range: 0x80 + type, 0x88 = no record }
//   per type, in page order: { state[w1] title code[w1] flag[w1]*npos gaps[w1]*ceil((n+1)/6) { token '\t' }* 0x0B }*
//   '\n'
// label: field label of a value slot (year only for a row of the population
// table).  flag: 0 absent, 1..89 one instance of variant flag-1, 99 = count[w2]
// then variant[w1] per instance.  gaps: 6 bits per digit, bit j of digit g =
// gap 6g+j (gap i lies in front of automatic unit i, gap n behind the last),
// written as XOR with a context majority (LueckeXor).  token: a value as the
// number coder writes it (censblock_zahl.h).  [wN] = N base-100 digits
// 0x80..0xE3, most significant first.  A record names neither its type nor its
// page; both come from the per-page symbols.
//
// Further parts of the form:
//   * the model side injects the position index before every flag of a record
//     (column context of the variant flags).
//   * the variant texts travel as a pseudo article, slot 3 of the text start
//     behind the census line and the author article, coded like a page; the
//     transformer sees it as a document of its own behind the slot head.
//   * the census line belongs to the metadata block.  The restarts at the end
//     of the whole metadata block (PPMd with its warm-up, the mixer learning
//     rate staircase) stand behind the census line, and a second PPMd restart
//     stands in front of it, so the pages begin with a fresh PPMd.  PPMd codes
//     the census line with the order table of the metadata block (order 14).
//   * the title of a census page loses its state and its type word ("Kiowa
//     County, Colorado" -> "Kiowa"); the record carries the type word as a
//     code and the state anyway.  The transformer gets state and type word in
//     its metadata prompt row, and every reader of the title text sees the
//     restored title.
//   * in a state link "[[X County, S|X County]]" the label is the repeat of the
//     text slot X plus its suffix, no second text slot.
//   * county index: the county link of a place ("[[X County, S]]", the first one
//     of the page whose County page has a record) becomes a value slot, the rank
//     of the county among the County records of the state; the page restores
//     the name from the title of that County page.
//   * county lists: the lines of the place list of a County page that name its
//     members (the places whose record carries this county, in front of the
//     County page in the stream), sorted by name, become automatic units; the
//     link name is the next member (value slot: members skipped).
//   * neighbour counties: a list line of a County page that links another
//     County page with a record ("*[[X County, S|X County]]  (east)") names it
//     by its rank in the neighbour order (counties that named this one first,
//     then the counties of the state, then the others).
//   * county category: a further link of a place to its own county (category
//     or repeated county link) is a placeholder without value.
//   * a family whose units the chain of a page often leaves out gets a further
//     position between the neighbours of the left-out units (up to three dry
//     rounds); the chain takes each unit at one of the positions of its family.
//   * coordinates: the coordinate sentence of the stub stays; behind its anchor
//     "located at " the model side injects "$" + the DMS pair computed from the
//     record coordinate + the decimal pair " (<lat>, <lon>)" + "$" and the
//     anchor once more into cmix and the transformer (observed, not coded), so
//     the first digit keeps its context and the match models find the copy.
//   * the model side decodes the census line once at its end and keeps the
//     values per page.  On a census page the metadata prompt row of the page id
//     carries "<population> <total area km2>", and behind the alias rows a
//     block for the transformer only gives the census values of the page in
//     the Rambot wording (at most 512 tokens).

#ifndef CENSBLOCK_SEITE_H_
#define CENSBLOCK_SEITE_H_
#include "../meld.h"
#include "../enc_only.h"
#include "page_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <memory>
#include <array>
#include "censblock_zahl.h"   // the number coder
#include "censblock_sort.h"   // sorted pointers for the sort sites (qsort)
namespace ppmd_page { void dyn_tick(void); }   // models/ppmd_page.cpp: the PPMd pool clamp

// the encoder runs all passes twice: the written records need values of
// later records (the counties of records of earlier types)
inline int zwei_runde = 1;   // encoder: 0 = first round (collects), 1 = the written round
// the written round of the encoder
inline bool SkalaRunde() {
  return zwei_runde != 0;
}

namespace census_block {

// ---------------------------------------------------------------- basics --
// names of a field are separated by NUL; Nth walks to name i
__attribute__((noinline)) inline const char* Nth(const char* f, int i) {
  while (i-- > 0) f += strlen(f) + 1;
  return f;
}
constexpr int ZeichenZahl(const char* f, size_t n, char c) {
  int k = 0;
  for (size_t i = 0; i < n; ++i) k += f[i] == c;
  return k;
}
static constexpr char kStaatFeld[] =
  "Alabama\0Alaska\0Arizona\0Arkansas\0California\0Colorado\0"
  "Connecticut\0Delaware\0Florida\0Georgia\0Hawaii\0Idaho\0"
  "Illinois\0Indiana\0Iowa\0Kansas\0Kentucky\0Louisiana\0Maine\0"
  "Maryland\0Massachusetts\0Michigan\0Minnesota\0Mississippi\0"
  "Missouri\0Montana\0Nebraska\0Nevada\0New Hampshire\0New Jersey\0"
  "New Mexico\0New York\0North Carolina\0North Dakota\0Ohio\0"
  "Oklahoma\0Oregon\0Pennsylvania\0Rhode Island\0South Carolina\0"
  "South Dakota\0Tennessee\0Texas\0Utah\0Vermont\0Virginia\0"
  "Washington\0West Virginia\0Wisconsin\0Wyoming\0"
  "District of Columbia\0Puerto Rico";
// postal codes, same order, two letters each
static constexpr char kPostFeld[] =
  "ALAKAZARCACOCTDEFLGAHIIDILINIAKS"
  "KYLAMEMDMAMIMNMSMOMTNENVNHNJNMNY"
  "NCNDOHOKORPARISCSDTNTXUTVTVAWAWV"
  "WIWYDCPR";
static const int kStaaten = 52;
static_assert(ZeichenZahl(kStaatFeld, sizeof kStaatFeld, 0) == kStaaten && sizeof kPostFeld == 2 * kStaaten + 1,
              "census: state tables");
// the county lists of all states empty (the buffers of the lists stay, as with assign)
__attribute__((noinline)) inline void KreislisteLeer() {
  for (auto& v : zahl::kreisliste) v.clear();
  zahl::kreisliste.resize((size_t)kStaaten);
}
static const char kTypen[] = "kctvpsbx";
static const int kNTypen = 8;

// placeholders inside a variant (raw page text never holds these bytes)
enum : unsigned char {
  kZahl = 1,     // number, value from the record
  kText = 2,     // text slot, from the page
  kWdh = 3,      // repeat of the last text slot
  kName = 4,     // name part of the title
  kStaatP = 5,   // state of the title
  kTitel = 6,    // whole title
  kPostP = 8,    // postal code of the state
  kRicht = 14,   // list direction (north ... southwest), value from the record
  kRichtG = 15   // the same, capitalised in the text, lower case in the record
};
static const unsigned char kKreis = 16;   // county of the place: value slot F_KREIS, the name of that County page
static const unsigned char kKreisW = 19;   // the county of the record again (no value); pseudo article: 0x06 0x06
static const unsigned char kGlied = 17;   // place of a county list: value slot F_GLIED, the name of the next member
static const unsigned char kNachbar = 18;   // adjacent county: value slot F_NACHBAR, the title of that County page
inline bool IstWert(unsigned char c) {
  return c == kZahl || c == kRicht || c == kRichtG || c == kKreis || c == kGlied || c == kNachbar;
}
// A census map keyed by a long (page index, context key): the table of the
// keys and the values by dense index.
template <class V>
struct LMap {
  zahl::LTab i;
  std::vector<V> v;
  V* Find(long k) {
    const long x = i.Find(k);
    return x < 0 ? nullptr : &v[(size_t)x];
  }
  V& Ref(long k) {   // as operator[]: a new key gets a value-initialised V
    bool neu;
    const long x = i.Neu(k, &neu);
    if (neu) v.resize(v.size() + 1);
    return v[(size_t)x];
  }
  void Setze(long k, const V& w) {   // as operator[] = w
    bool neu;
    const long x = i.Neu(k, &neu);
    if (neu) v.push_back(w);
    else v[(size_t)x] = w;
  }
  void clear() {
    i.Clear();
    v.clear();
  }
  // iteration (diagnostic exports only): (key, value), in no particular order
  struct Paar { long first; const V& second; };
  struct It {
    const LMap* m;
    unsigned long j;
    void Weiter() { while (j < m->i.cap && !m->i.t[2 * j]) ++j; }
    bool operator!=(const It& o) const { return j != o.j; }
    void operator++() { ++j; Weiter(); }
    Paar operator*() const { return Paar{(long)(m->i.t[2 * j] - 1), m->v[(size_t)m->i.t[2 * j + 1]]}; }
  };
  It begin() const { It a{this, 0}; a.Weiter(); return a; }
  It end() const { return It{this, i.cap}; }
};
// type word code and state of the County pages with a record (whole title of a
// neighbour): two values per page
inline zahl::LTab kreisart_i;
inline std::vector<long> kreisart_w;
// (type word code, state) of County page k, or nullptr
inline const long* KreisArt(long k) {
  const long x = kreisart_i.Find(k);
  return x < 0 ? nullptr : &kreisart_w[(size_t)x * 2];
}
__attribute__((noinline)) inline void KreisArtSetze(long k, long tcode, long st) {
  bool neu;
  const long x = kreisart_i.Neu(k, &neu);
  if (neu) {
    kreisart_w.push_back(tcode);
    kreisart_w.push_back(st);
  } else {
    kreisart_w[(size_t)x * 2] = tcode;
    kreisart_w[(size_t)x * 2 + 1] = st;
  }
}
inline void KreisArtLeer() {
  kreisart_i.Clear();
  kreisart_w.clear();
}
// neighbour model: county page -> the County pages that named it as a neighbour, in record order
inline LMap<std::vector<long>> nachbar_ang;
// members of a county: (short title, page index), sorted by name; encoder: from
// the place records it wrote, decoder: from the records and the stub titles
inline LMap<std::vector<std::pair<std::string, long>>> kreisglieder;
// sorts the members by (name, page) like std::sort: sorted pointers, then moved
inline int GliedVgl(const void* x, const void* y) {
  const std::pair<std::string, long>& a = **(const std::pair<std::string, long>* const*)x;
  const std::pair<std::string, long>& b = **(const std::pair<std::string, long>* const*)y;
  const int c = a.first.compare(b.first);
  return c ? c : (a.second > b.second) - (a.second < b.second);
}
__attribute__((noinline)) inline void SortiereGlieder(std::vector<std::pair<std::string, long>>& g) {
  void** z = ZeigerSortiert(g.data(), g.size(), sizeof(g[0]), GliedVgl);
  std::vector<std::pair<std::string, long>> n;
  for (size_t i = 0; i < g.size(); ++i) n.push_back(std::move(*(std::pair<std::string, long>*)z[i]));
  free(z);
  g.swap(n);
}
inline const std::vector<std::pair<std::string, long>>* glied = nullptr;   // members of the running County page
inline long glied_ptr = 0;   // next member of the running County page
inline bool liste_akt = false;   // encoder: the unit being abstracted is a list line
// short titles of the County pages with a record, by page index (encoder: from
// the pre-pass, decoder: from the restored stub titles)
inline LMap<std::string> kreisname;
inline long kreis_akt = -1;   // county of the running page (encoder: first county link, decoder: F_KREIS of the record)
// stub bytes inside the page text
enum : unsigned char { kRun = 1, kEnde = 2 };
static const int kMinSeiten = 4;      // pages a family, a folded constant needs
// pages a variant needs (= kMinSeiten); fewer pages would let rare
// wordings of the bot sentences into the form instead of the stub
static const int kMinVar = 4;
static const int kMinLaenge = 4;      // shorter units stay text
static const int kMaxVar = 89;        // variants per position (flag 1..89)
static const char kMagic[4] = {'z', 'c', 'b', '3'};
// list directions (encoder only: .rodata.enc_only lies in .encoder_only)
__attribute__((section(".rodata.enc_only"))) static constexpr char kRichtungFeld[] =
  "northeast\0northwest\0southeast\0southwest\0north\0south\0east\0west";

inline bool ImAlphabet(unsigned char b) {
  static const unsigned char bm[32] = { 0xfe, 0x1e, 0x00, 0x00, 0xff, 0xff, 0xff, 0x03, 0x01, 0xfc, 0x0f, 0xf9, 0xfe, 0xff, 0xff, 0x07, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
  return (bm[b >> 3] >> (b & 7)) & 1;
}
inline bool Alnum(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
inline bool Ziffer(unsigned char c) { return c >= '0' && c <= '9'; }

inline void B100(std::string* o, unsigned long v, int w) {
  char t[8];
  for (int i = w - 1; i >= 0; --i) { t[i] = (char)(0x80 + v % 100); v /= 100; }
  o->append(t, (size_t)w);
}
// reads w base-100 digits at *p, advances; -1 on a foreign byte
inline long LiesB100(const unsigned char*& p, const unsigned char* e, int w) {
  long v = 0;
  for (int i = 0; i < w; ++i) {
    if (p >= e || *p < 0x80 || *p > 0xE3) return -1;
    v = v * 100 + (*p++ - 0x80);
  }
  return v;
}
// The pseudo article carries the variant texts, each closed by 0x0B, the whole
// closed by one more 0x0B.  Its placeholders have to stay below 0x08 (in the
// coder alphabet and untouched by the WRT): the direction slot becomes a
// number slot (both append the value), the postal code 0x07, the capitalised
// direction 0x07 0x07.
inline void PseudoText(const std::string& v, std::string* o) {
  for (unsigned char c : v) {
    if (c == kKreis) { o->push_back((char)kZahl); continue; }   // restored from its label F_KREIS
    if (c == kKreisW) { o->push_back((char)kTitel); o->push_back((char)kTitel); continue; }
    if (c == kGlied) { o->push_back((char)kZahl); continue; }   // restored from its label F_GLIED
    if (c == kNachbar) { o->push_back((char)kZahl); continue; }   // restored from its label F_NACHBAR
    if (c == kRicht) o->push_back((char)kZahl);
    else if (c == kRichtG) { o->push_back('\x07'); o->push_back('\x07'); }
    else if (c == kPostP) o->push_back('\x07');
    else o->push_back((char)c);
  }
  o->push_back('\x0b');
}
// splits the decoded pseudo article into the variant texts; false if broken
inline bool PseudoTexte(const std::string& p, std::vector<std::string>* tx) {
  tx->clear();
  std::string v;
  for (size_t i = 0; i < p.size(); ++i) {
    const unsigned char c = (unsigned char)p[i];
    if (c == 0x0B) {
      if (v.empty()) return i + 1 == p.size();   // the closing empty variant
      tx->push_back(v);
      v.clear();
    } else if (c == 0x07) {
      if (i + 1 < p.size() && p[i + 1] == '\x07') { v.push_back((char)kRichtG); ++i; }
      else v.push_back((char)kPostP);
    } else if (c == kTitel && i + 1 < p.size() && p[i + 1] == (char)kTitel) {
      v.push_back((char)kKreisW);
      ++i;
    } else {
      v.push_back((char)c);
    }
  }
  return false;
}

// type words at the end of the name part of a title (code = index, 0 = none);
// the longer of two words with the same end comes first
static constexpr char kTypwortFeld[] =
  "\0 City and Borough\0 Census Area\0 Charter Township\0 County\0 Parish\0 Borough\0"
  " Township\0 Municipality\0 (city)\0 (town)\0 (village)\0 (CDP)\0 (township)\0"
  " (borough)\0 (census-designated place)\0 (municipality)";
static const int kTypworte = 17;
static_assert(ZeichenZahl(kTypwortFeld, sizeof kTypwortFeld, 0) == kTypworte, "census: type words");

__attribute__((section(".rodata.enc_only"))) static constexpr char kPseudoWoerter[] =
  "city\013town\013village\013census-designated place\013township\013borough\013municipality\013"
  "census population households\013located at\013North\013South\013East\013West\013"
  "As of 2000, the population is \013As of the 2000 census, the \013 had a total population of \013, with \013"
  " households\013 families\013 housing units\013, \013 and \013.\013 The \013 has a total area of \013 km\013"
  " mi\013: \013 of it is land\013 of it is water\013% water\013 The population density is \013/km\013/mi\013"
  " is located at \013 (\013).\013)\013 The median income for a household is $\013, for a family $\013"
  ", and the per capita income is $\013North\013South\013East\013West\013county\013city\013town\013village\013"
  "CDP\013township\013borough\013community\013\013";
static const int kKontextWorte = 7;
static const int kRechWorte = 5;
static const int kK1Worte = 40;
static_assert(ZeichenZahl(kPseudoWoerter, sizeof kPseudoWoerter, 0x0B) == kKontextWorte + 1 + kRechWorte + kK1Worte + 1,
              "census: words of the pseudo article");
// "&amp;deg;" and "&amp;sup2;" of the page text in coder-stream form: the pseudo
// article codes its words without the entity stage of the preprocessor, so
// these two pieces are written here (as the stub text carries them)
static const char kGradStrom[] = "&&\x0c\xc2\x0c\xb0";
static const char kQuadratStrom[] = "&&\x0b";

enum K1Stueck {
  K1_POPK, K1_POPO1, K1_POPO2, K1_WITH, K1_HH, K1_FAM, K1_HU, K1_KOMMA, K1_AND, K1_PUNKT, K1_THE, K1_AREA, K1_KM,
  K1_MI, K1_DP, K1_LAND, K1_WATER, K1_PCT, K1_DENS, K1_KMD, K1_MID, K1_LOC, K1_KLA, K1_KLZ,
  K1_ZU, K1_INC1, K1_INC2, K1_INC3, K1_NORTH, K1_SOUTH, K1_EAST, K1_WEST, K1_TYP
};
static_assert(K1_TYP + 8 == kK1Worte, "census: K1 pieces");

// code of the type word that ends the name part n[0..nb), the base length in *b
inline int TitelTyp(const char* n, long nb, long* b) {
  const char* w = kTypwortFeld + 1;   // behind the empty word 0
  for (int i = 1; i < kTypworte; ++i) {
    const long l = (long)strlen(w);
    if (nb > l && memcmp(n + nb - l, w, (size_t)l) == 0) { *b = nb - l; return i; }
    w += l + 1;
  }
  *b = nb;
  return 0;
}

inline int TypIndex(char t) {
  for (int i = 0; i < kNTypen; ++i) if (kTypen[i] == t) return i;
  return -1;
}

// ------------------------------------------------------------- page parts --
struct Seite {
  const char* p = nullptr;  // the page
  long n = 0;
  long t0 = 0, t1 = 0;      // title [t0, t1)
  long x0 = 0, x1 = 0;      // text body [x0, x1)
  int staat = -1;
  long basis = 0;           // length of the name part of the title
};

// title "<name>, <state>" and the text body; false if this is no census candidate
inline bool Teile(const char* p, long n, Seite* s) {
  long a = page_find(p, n, 0, "<title>", 7);
  if (a < 0) return false;
  a += 7;
  long b = page_find(p, n, a, "</title>", 8);
  if (b < 0) return false;
  long k = -1;
  for (long i = b - 2; i >= a; --i) if (p[i] == ',' && p[i + 1] == ' ') { k = i; break; }
  if (k <= a) return false;
  int st = -1;
  const char* w = kStaatFeld;
  for (int i = 0; i < kStaaten; ++i) {
    const long l = (long)strlen(w);
    if (l == b - (k + 2) && memcmp(p + k + 2, w, (size_t)l) == 0) { st = i; break; }
    w += l + 1;
  }
  if (st < 0) return false;
  long x = page_find(p, n, b, "<text", 5);
  if (x < 0) return false;
  long g = page_find(p, n, x, ">", 1);
  if (g < 0 || p[g - 1] == '/') return false;
  long y = page_find(p, n, g + 1, "</text>", 7);
  if (y < 0) return false;
  s->p = p; s->n = n; s->t0 = a; s->t1 = b; s->x0 = g + 1; s->x1 = y;
  s->staat = st; s->basis = k - a;
  return true;
}

// The page predicate.  Evaluated by the census transform on the raw page and
// by Zerlege on the reordered page - both see the same title and text bytes.
inline bool Nimmt(const char* p, long n) {
  Seite s;
  if (!Teile(p, n, &s)) return false;
  const char* t = p + s.x0;
  const long m = s.x1 - s.x0;
  return page_find(t, m, 0, " people, ", 9) >= 0 &&
         page_find(t, m, 0, " households, and ", 17) >= 0 &&
         page_find(t, m, 0, " families residing in the ", 26) >= 0;
}

// type of the page from the census sentence: k county, c city, t town,
// v village, p CDP, s township, b borough, x other
inline char Typ(const Seite& s) {
  const char* t = s.p + s.x0;
  const long m = s.x1 - s.x0;
  long i = page_find(t, m, 0, " families residing in the ", 26);
  if (i < 0) return 'x';
  i += 26;
  static const char* const w[] = {"county", "city", "town", "village", "CDP", "township", "borough"};
  for (int j = 0; j < 7; ++j) {
    const long l = (long)strlen(w[j]);
    if (i + l < m && memcmp(t + i, w[j], (size_t)l) == 0 && t[i + l] == '.') return kTypen[j];
  }
  return 'x';
}

// ------------------------------------------------------------- expansion --
// variant + values + text slots + title -> unit text.  Used by the encoder as
// the byte-wise check and by the decoder to restore.  false if parts run out.
struct Teile2 {
  const char* basis; long nb;
  const char* staat; long ns;
  const char* titel; long nt;
  const char* post;
};
// one coordinate sentence: start/length of the six numbers, the decimal pair as
// |value| * 10^nk, and the span [a, e) of the sentence
struct Dms {
  size_t z[6][2];
  long long dez[2];
  int nk[2];
  size_t a, e;
  bool eng;     // " West(" without the space: the form with left-out numbers
  size_t klam;  // position of the "(" of the decimal pair
};
inline size_t DmsZiff(const std::string& o, size_t i) {
  size_t j = i;
  while (j < o.size() && o[j] >= '0' && o[j] <= '9') ++j;
  return j - i;
}
inline size_t DmsZiffRueck(const std::string& o, size_t e) {
  size_t j = e;
  while (j > 0 && o[j - 1] >= '0' && o[j - 1] <= '9') --j;
  return e - j;
}
inline size_t DmsNum(const std::string& o, size_t i) { return DmsZiff(o, i); }
inline size_t DmsNumRueck(const std::string& o, size_t e) { return DmsZiffRueck(o, e); }
inline bool DmsIst(const std::string& o, size_t i, const char* w) {
  const size_t n = strlen(w);
  return i + n <= o.size() && o.compare(i, n, w) == 0;
}
inline size_t DmsRueck(const std::string& o, size_t e, const char* const* w, int nw) {   // a word ending at e
  for (int k = 0; k < nw; ++k) {
    const size_t n = strlen(w[k]);
    if (e >= n && o.compare(e - n, n, w[k]) == 0) return n;
  }
  return 0;
}
inline size_t DmsVor(const std::string& o, size_t i, const char* const* w, int nw) {   // a word starting at i
  for (int k = 0; k < nw; ++k) if (DmsIst(o, i, w[k])) return strlen(w[k]);
  return 0;
}
static const char* const kDmsGrad[3] = {"&amp;deg;", "&deg;", "\xC2\xB0"};
static const char* const kDmsZoll[2] = {"&quot;", "\""};
// a decimal at i: [-]digits.digits; |value| * 10^nk
inline size_t DmsDez(const std::string& o, size_t i, long long* v, int* nk) {
  size_t j = i;
  if (j < o.size() && o[j] == '-') ++j;
  const size_t g = DmsZiff(o, j);
  if (!g || g > 3 || j + g >= o.size() || o[j + g] != '.') return 0;
  const size_t f = DmsZiff(o, j + g + 1);
  if (!f || f > 12) return 0;
  *v = strtol(o.substr(j, g).c_str(), nullptr, 10);
  for (size_t q = 0; q < f; ++q) *v = *v * 10 + (o[j + g + 1 + q] - '0');
  *nk = (int)f;
  return j + g + 1 + f - i;
}
// the sentence whose latitude half ends with the hemisphere word at h (" North, " / " South, ")
inline bool DmsBei(const std::string& o, size_t h, Dms* d) {
  static const char* const kNs[2] = {" North, ", " South, "};
  static const char* const kEw[4] = {" East (", " West (", " East(", " West("};
  // latitude, backwards from h: Q, seconds, "'", minutes, glue, degrees
  size_t e = h;
  size_t q = DmsRueck(o, e, kDmsZoll, 2);
  if (!q) return false;
  e -= q;
  size_t n = DmsNumRueck(o, e);
  d->z[2][0] = e - n; d->z[2][1] = n;
  e -= n;
  if (e == 0 || o[e - 1] != '\'') return false;
  --e;
  n = DmsNumRueck(o, e);
  d->z[1][0] = e - n; d->z[1][1] = n;
  e -= n;
  q = DmsRueck(o, e, kDmsGrad, 3);
  if (!q) return false;
  e -= q;
  n = DmsNumRueck(o, e);
  d->z[0][0] = e - n; d->z[0][1] = n;
  d->a = e - n;
  // longitude, forwards behind ", "
  size_t i = h + strlen(kNs[0]);
  n = DmsNum(o, i);
  d->z[3][0] = i; d->z[3][1] = n;
  i += n;
  q = DmsVor(o, i, kDmsGrad, 3);
  if (!q) return false;
  i += q;
  n = DmsNum(o, i);
  d->z[4][0] = i; d->z[4][1] = n;
  i += n;
  if (i >= o.size() || o[i] != '\'') return false;
  ++i;
  n = DmsNum(o, i);
  d->z[5][0] = i; d->z[5][1] = n;
  i += n;
  q = DmsVor(o, i, kDmsZoll, 2);
  if (!q) return false;
  i += q;
  q = 0;
  for (int k = 0; k < 4 && !q; ++k) {
    if (!DmsIst(o, i, kEw[k])) continue;
    q = strlen(kEw[k]);
    d->eng = k >= 2;
  }
  if (!q) return false;
  i += q;
  d->klam = i - 1;
  size_t m = DmsDez(o, i, &d->dez[0], &d->nk[0]);
  if (!m) return false;
  i += m;
  if (!DmsIst(o, i, ", ")) return false;
  i += 2;
  m = DmsDez(o, i, &d->dez[1], &d->nk[1]);
  if (!m) return false;
  i += m;
  if (!DmsIst(o, i, ")")) return false;
  d->e = i + 1;
  return true;
}
// the next sentence from position i, or false
inline bool DmsNaechste(const std::string& o, size_t i, Dms* d, size_t* weiter) {
  for (;;) {
    const size_t a = o.find(" North, ", i), b = o.find(" South, ", i);
    const size_t h = a < b ? a : b;
    if (h == std::string::npos) return false;
    *weiter = h + 1;
    if (DmsBei(o, h, d)) return true;
    i = h + 1;
  }
}
// degrees, minutes, seconds of |value| * 10^nk: rounded (r) or truncated seconds
inline void DmsAus(long long v, int nk, bool r, std::string out[3]) {
  long long p10 = 1;
  for (int k = 0; k < nk; ++k) p10 *= 10;
  if (v < 0) v = -v;
  const long long sek = r ? (v * 7200 + p10) / (2 * p10) : (v * 3600) / p10;
  out[0] = zahl::Dez((long)(sek / 3600));
  out[1] = zahl::Dez((long)(sek % 3600 / 60));
  out[2] = zahl::Dez((long)(sek % 60));
}
// the page: numbers of the left-out halves back (decoder)
inline void DmsZurueck(std::string* o) {
  size_t i = 0, w = 0;
  Dms d;
  std::string neu;
  size_t kopiert = 0;
  while (DmsNaechste(*o, i, &d, &w)) {
    i = d.e > w ? d.e : w;
    // only the form without the space in front of "(" has left-out numbers
    bool leer = false;
    for (int h = 0; h < 2; ++h) leer = leer || (d.z[3 * h][1] == 0 && d.z[3 * h + 1][1] == 0);
    if (!d.eng || !leer) continue;
    for (int h = 0; h < 2; ++h) {
      if (d.z[3 * h][1] != 0 || d.z[3 * h + 1][1] != 0) continue;   // degrees present: written out
      std::string x[3];
      DmsAus(d.dez[h], d.nk[h], d.z[3 * h + 2][1] == 0, x);
      for (int k = 0; k < 3; ++k) {
        if (k == 2 && d.z[3 * h + 2][1] != 0) continue;   // the kept seconds
        neu.append(*o, kopiert, d.z[3 * h + k][0] - kopiert);
        neu += x[k];
        kopiert = d.z[3 * h + k][0];
      }
    }
    neu.append(*o, kopiert, d.klam - kopiert);   // the space back in front of "("
    neu.push_back(' ');
    kopiert = d.klam;
  }
  if (kopiert == 0) return;
  neu.append(*o, kopiert, std::string::npos);
  o->swap(neu);
}
// value slot wi as a whole decimal number (kKreis, kNachbar); false = missing or no number
__attribute__((noinline)) inline bool WertZahl(const std::vector<std::string>& slot_vals, size_t wi, long* k) {
  if (wi >= slot_vals.size()) return false;
  char* e = nullptr;
  *k = strtol(slot_vals[wi].c_str(), &e, 10);
  return !slot_vals[wi].empty() && !*e;
}
inline bool Expandiere(const std::string& key, const std::vector<std::string>& slot_vals, size_t* wi,
                       const std::vector<std::string>& text, const Teile2& t, std::string* o) {
  size_t ti = 0;
  const std::string* letzt = nullptr;
  for (unsigned char c : key) {
    switch (c) {
      case kZahl:
      case kRicht:
        if (*wi >= slot_vals.size()) return false;
        o->append(slot_vals[(*wi)++]);
        break;
      case kRichtG:
        if (*wi >= slot_vals.size() || slot_vals[*wi].empty()) return false;
        o->push_back((char)(slot_vals[*wi][0] - 32));
        o->append(slot_vals[*wi].data() + 1, slot_vals[*wi].size() - 1);
        ++*wi;
        break;
      case kText:
        if (ti >= text.size()) return false;
        letzt = &text[ti++];
        o->append(*letzt);
        break;
      case kKreis: {
        long k;
        if (!WertZahl(slot_vals, *wi, &k)) return false;
        const std::string* nm = kreisname.Find(k);
        if (!nm) return false;
        ++*wi;
        letzt = nm;
        o->append(*nm);
        break;
      }
      case kGlied: {   // "@<index>" (encoder check) or the members skipped since the last one
        if (*wi >= slot_vals.size() || !glied) return false;
        const std::string& w = slot_vals[*wi];
        long j = glied_ptr;
        if (!w.empty()) {
          char* e = nullptr;
          const bool abs = w[0] == '@';
          const long x = strtol(w.c_str() + (abs ? 1 : 0), &e, 10);
          if (*e || x < 0) return false;
          j = abs ? x : glied_ptr + x;
        }
        if (j < 0 || j >= (long)glied->size()) return false;
        ++*wi;
        glied_ptr = j + 1;
        letzt = &(*glied)[(size_t)j].first;
        o->append(*letzt);
        break;
      }
      case kNachbar: {   // the whole title of a County page; the repeat is its name with type word
        long k;
        if (!WertZahl(slot_vals, *wi, &k)) return false;
        const std::string* nm = kreisname.Find(k);
        const long* ar = KreisArt(k);
        if (!nm || !ar) return false;
        ++*wi;
        static std::string nb_name;
        nb_name = *nm;
        nb_name += Nth(kTypwortFeld, (int)ar[0]);
        letzt = &nb_name;
        o->append(nb_name);
        o->append(", ");
        o->append(Nth(kStaatFeld, (int)ar[1]));
        break;
      }
      case kKreisW: {
        const std::string* nm = kreisname.Find(kreis_akt);
        if (!nm) return false;
        letzt = nm;
        o->append(*nm);
        break;
      }
      case kWdh:
        if (!letzt) return false;
        o->append(*letzt);
        break;
      case kName: o->append(t.basis, (size_t)t.nb); break;
      case kStaatP: o->append(t.staat, (size_t)t.ns); break;
      case kTitel: o->append(t.titel, (size_t)t.nt); break;
      case kPostP: o->append(t.post, 2); break;
      default: o->push_back((char)c); break;
    }
  }
  return ti == text.size();
}

inline int Textslots(const std::string& key) {
  int k = 0;
  for (unsigned char c : key) k += (c == kText);
  return k;
}
inline int Wertslots(const std::string& key) {
  int k = 0;
  for (unsigned char c : key) k += IstWert(c);
  return k;
}
inline Teile2 TitelTeile(const Seite& s) {
  const char* st = Nth(kStaatFeld, s.staat);
  const Teile2 t = {s.p + s.t0, s.basis, st, (long)strlen(st), s.p + s.t0, s.t1 - s.t0, kPostFeld + 2 * s.staat};
  return t;
}

// --------------------------------------------------- values of a record --
// A value slot of a record: its field label and, for a row of the population
// table, the year of the row (from the variant text).
struct Slot { unsigned char lab = 0; int jahr = 0; };

// year "[[YYYY]]" in front of value slot k of a variant (rows of the table), else 0
inline int PvJahr(const std::string& v, int k) {
  int s = -1;
  for (size_t i = 0; i < v.size(); ++i) {
    if (!IstWert((unsigned char)v[i]) || ++s != k) continue;
    for (size_t j = i; j >= 8; --j) {
      if (v[j - 1] == ']' && v[j - 2] == ']' && v[j - 7] == '[' && v[j - 8] == '[') {
        int y = 0;
        for (size_t q = j - 6; q < j - 2; ++q) {
          if (v[q] < '0' || v[q] > '9') return 0;
          y = y * 10 + (v[q] - '0');
        }
        return y;
      }
    }
    return 0;
  }
  return 0;
}

// the neighbour order of County page p (state st): the counties that named p,
// the last first, then the County records of st, then those of the other states;
// without p and without the ones the page named already
inline void NachbarFolge(long p, int st, const std::vector<long>& seen_list, std::vector<long>* c) {
  c->clear();
  auto drin = [&](long k) {
    if (k == p) return true;
    for (long x : seen_list) if (x == k) return true;
    for (long x : *c) if (x == k) return true;
    return false;
  };
  if (const std::vector<long>* an = nachbar_ang.Find(p))
    for (size_t i = an->size(); i-- > 0;) if (!drin((*an)[i])) c->push_back((*an)[i]);
  const size_t na = c->size();
  for (int r = -1; r < kStaaten; ++r) {   // r = -1: the state st itself, then the others
    const int q = r < 0 ? st : r;
    if (r >= 0 && q == st) continue;
    for (long k : zahl::kreisliste[(size_t)q]) {
      bool d = k == p;
      for (long x : seen_list) d = d || x == k;
      for (size_t i = 0; i < na && !d; ++i) d = (*c)[i] == k;
      if (!d) c->push_back(k);
    }
  }
}

inline bool NachbarRaenge(long p, int st, const std::vector<Slot>& slots, std::vector<std::string>* slot_values, bool enc) {
  std::vector<long> seen_list, c;
  for (size_t i = 0; i < slots.size() && i < slot_values->size(); ++i) {
    if (slots[i].lab != zahl::F_NACHBAR) continue;
    std::string& w = (*slot_values)[i];
    NachbarFolge(p, st, seen_list, &c);
    long k = -1;
    if (enc) {
      k = strtol(w.c_str(), nullptr, 10);
      long r = -1;
      for (size_t j = 0; j < c.size(); ++j) if (c[j] == k) { r = (long)j; break; }
      if (r < 0) w.insert(0, "#");   // "#" + w without the operator+ instance
      else w = r ? zahl::Dez(r) : std::string();
    } else {
      if (!w.empty() && w[0] == '#') {
        k = strtol(w.c_str() + 1, nullptr, 10);
      } else {
        const long r = w.empty() ? 0 : strtol(w.c_str(), nullptr, 10);
        if (r < 0 || r >= (long)c.size()) return false;
        k = c[(size_t)r];
      }
      w = zahl::Dez(k);
    }
    seen_list.push_back(k);
  }
  for (long k : seen_list) nachbar_ang.Ref(k).push_back(p);
  return true;
}
// per state: (page, type index, county) of the records (County records: their own page)
inline std::vector<std::vector<std::array<long, 3>>> anl4_tab;
// the candidates of a record of type ti on page pg: counties of the nearest records of earlier types
inline void Anl4Setzen(int ti, long pg, int st) {
  zahl::anl4_u = zahl::anl4_o = -1;
  if (st < 0 || (size_t)st >= anl4_tab.size()) return;
  long bu = -1, bo = -1;
  for (const auto& e : anl4_tab[(size_t)st]) {
    if (e[1] >= ti || e[2] < 0) continue;
    if (e[0] < pg && (bu < 0 || e[0] > bu)) { bu = e[0]; zahl::anl4_u = e[2]; }
    if (e[0] > pg && (bo < 0 || e[0] < bo)) { bo = e[0]; zahl::anl4_o = e[2]; }
  }
}
inline void Anl4Merken(int ti, long pg, int st, long kreis) {
  if (st < 0) return;
  if ((size_t)st >= anl4_tab.size()) anl4_tab.resize((size_t)st + 1);
  anl4_tab[(size_t)st].push_back({pg, (long)ti, kreis});
}
// context -> {zeros, ones}: two counts per context
inline zahl::LTab luecke_i;
inline std::vector<long> luecke_w;
inline void LueckeLeer() {
  luecke_i.Clear();
  luecke_w.clear();
}
// units: (position, variant) of the record in order; bit: the gap bits (units + 1),
// in place: true bits -> XOR bits (enc) or back (dec)
inline void LueckeXor(int typ, int npos, const std::vector<std::pair<int, int>>& u, std::vector<unsigned char>* bit,
                      bool enc) {
  for (size_t i = 0; i < bit->size(); ++i) {
    const int pp = i > 0 && i - 1 < u.size() ? u[i - 1].first : -1;
    const int pn = i < u.size() ? u[i].first : npos;
    const unsigned long long lu = (unsigned long long)(pn - pp - 1 > 4 ? 4 : (pn - pp - 1 < 0 ? 0 : pn - pp - 1));
    const unsigned long long a = i > 0 && i - 1 < u.size() ? (unsigned long long)u[i - 1].first << 7 | (unsigned long long)u[i - 1].second
                                                           : 0x1FFFFull;
    const unsigned long long b = i < u.size() ? (unsigned long long)u[i].first << 7 | (unsigned long long)u[i].second : 0x1FFFEull;
    const unsigned long long key = (unsigned long long)typ << 40 | a << 23 | b << 3 | lu;
    bool neu;
    const long ci = luecke_i.Neu((long)key, &neu);
    if (neu) {
      const long z0 = 0;
      luecke_w.push_back(z0);
      luecke_w.push_back(z0);
    }
    long* c = &luecke_w[(size_t)ci * 2];
    const unsigned char pred = c[1] > c[0] ? 1 : (c[0] > c[1] ? 0 : (lu >= 3 ? 1 : 0));
    const unsigned char wahr = enc ? (*bit)[i] : (unsigned char)((*bit)[i] ^ pred);
    (*bit)[i] = enc ? (unsigned char)(wahr ^ pred) : wahr;
    ++c[wahr ? 1 : 0];
  }
}
// Every place record carries its county: when no chosen unit has the county
// slot, one more value F_KREIS follows the values of the units (the county of
// the first county link of the page, or -1).  Encoder, decoder and the model
// side apply the same rule.
inline bool KreisZusatz(char typ, std::vector<Slot>* slots) {
  if (typ == 'k') return false;
  for (const Slot& q : *slots) if (q.lab == zahl::F_KREIS) return false;
  Slot sl;
  sl.lab = zahl::F_KREIS;
  slots->push_back(sl);
  return true;
}

__attribute__((noinline)) inline void LeereWerte(std::vector<std::string>* w, size_t n) {
  const std::string leer;
  w->clear();
  for (size_t i = 0; i < n; ++i) w->push_back(leer);
}
inline bool Zahlen(zahl::Typmodell& tm, bool cnty, int staat, const std::vector<Slot>& slots,
                   std::vector<std::string>* slot_values, std::vector<std::string>* tok, bool enc,
                   std::vector<unsigned char>* folge = nullptr, long vorschau = -1, std::string* est = nullptr
                   , zahl::SkalaEst* sk = nullptr
                   , std::vector<char>* bekannt = nullptr
                   ) {
  zahl::Rec R;
  std::vector<int> rolle(slots.size(), -1);   // field, -2 pv row, -1 free
  std::vector<std::pair<int, size_t>> pvi;
  for (size_t i = 0; i < slots.size(); ++i) {
    const int l = slots[i].lab;
    if (l == zahl::F_PV) { pvi.push_back(std::make_pair(slots[i].jahr, i)); rolle[i] = -2; }
    else if (l != zahl::F_X && l < zahl::kFelder && !R.da[l]) { R.da[l] = true; rolle[i] = l; }
  }
  // by year; .second is the slot index (ascending), so (year, index) is the stable order
  qsort(pvi.data(), pvi.size(), sizeof(pvi[0]), [](const void* x, const void* y) {
    const std::pair<int, size_t>& a = *(const std::pair<int, size_t>*)x;
    const std::pair<int, size_t>& b = *(const std::pair<int, size_t>*)y;
    if (a.first != b.first) return a.first < b.first ? -1 : 1;
    return (a.second > b.second) - (a.second < b.second);
  });
  for (size_t i = 0; i < slots.size(); ++i) if (enc && rolle[i] >= 0) R.s[rolle[i]] = (*slot_values)[i];
  const std::string leer;
  for (const auto& pr : pvi) { R.pvj.push_back(pr.first); R.pvs.push_back(enc ? (*slot_values)[pr.second] : leer); }
  zahl::Run L(tm, R, cnty, staat, enc, tok, 0);
  if (folge) L.Trocken(folge);
  if (vorschau >= 0) {   // preview: the estimate of token `vorschau` (model side)
    // the neighbours of the record before stay: the fields in front of the
    // coordinates read them, in the preview as in the coder
    const zahl::Nb8 nb0 = tm.nbi;
    // the rows of the population table share one model, which the coder
    // updates row by row: the preview does the same and puts it back
    const zahl::Cont pv0 = tm.M[zahl::F_PV];
    const bool pvok0 = tm.mok[zahl::F_PV];
    L.Skala(sk);
    L.Vorschau((size_t)vorschau, est);
    L.Alles();
    tm.nbi = nb0;
    tm.M[zahl::F_PV] = pv0;
    tm.mok[zahl::F_PV] = pvok0;
    if (bekannt) {   // the values known in front of token `vorschau` (XFI): the fields of the number coder, when
                     // it ran through in front of it, and the free tokens in front of it
      LeereWerte(slot_values, slots.size());
      bekannt->assign(slots.size(), 0);
      if (L.Gut()) {
        for (size_t i = 0; i < slots.size(); ++i) if (rolle[i] >= 0 && !L.Genutzt(rolle[i])) rolle[i] = -1;
        size_t ti = L.Gelesen();
        for (size_t i = 0; i < slots.size(); ++i) {
          if (rolle[i] >= 0) {
            (*slot_values)[i] = R.s[rolle[i]];
            (*bekannt)[i] = 1;
          } else if (rolle[i] == -1) {
            if (ti < (size_t)vorschau && ti < tok->size()) { (*slot_values)[i] = (*tok)[ti]; (*bekannt)[i] = 1; }
            ++ti;
          }
        }
        for (size_t k = 0; k < pvi.size(); ++k) { (*slot_values)[pvi[k].second] = R.pvs[k]; (*bekannt)[pvi[k].second] = 1; }
      }
    }
    return true;
  }
  L.Alles();
  if (!L.Gut()) return false;
  for (size_t i = 0; i < slots.size(); ++i) if (rolle[i] >= 0 && !L.Genutzt(rolle[i])) rolle[i] = -1;
  if (folge) {
    for (size_t i = 0; i < slots.size(); ++i) if (rolle[i] == -1) folge->push_back((unsigned char)zahl::F_X);
    return true;
  }
  size_t ti = enc ? 0 : L.Gelesen();
  if (!enc) LeereWerte(slot_values, slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    if (rolle[i] != -1) continue;
    if (enc) tok->push_back((*slot_values)[i]);
    else { if (ti >= tok->size()) return false; (*slot_values)[i] = (*tok)[ti++]; }
  }
  if (!enc) {
    if (ti != tok->size()) return false;
    for (size_t i = 0; i < slots.size(); ++i) if (rolle[i] >= 0) (*slot_values)[i] = R.s[rolle[i]];
    for (size_t k = 0; k < pvi.size(); ++k) (*slot_values)[pvi[k].second] = R.pvs[k];
  }
  return tok->size() == slots.size();
}

// ------------------------------------------------------------ the encoder --
// One unit -> variant key, values, text slots.  Rules (all checked by Expandiere):
//   [[X, S]] / [[X, S|Y]] with S a state: X == page name and S == page state
//     -> placeholders; else X (without a trailing " County" etc.) is a text
//     slot, S the placeholder if it is the page state, Y a repeat of X or a
//     second text slot.  Other links and templates are copied, page name,
//     state and postal code inside them become placeholders, digits stay
//     literal (years, footnote numbers) except decimals inside {{ }} (Mapit).
//   page title / name / state as whole words -> placeholders, the postal code
//   as a word or in front of "Map".
//   numbers outside links: -?digits(,ddd)*(.digits)? between non-alphanumerics.
//   in list lines ("*..."): direction words -> value slot.
struct Abstr {
  std::string key;
  std::vector<std::string> slot_vals;
  std::vector<std::string> text;
};

// the suffixes of county links: " County", " Parish", " Borough", " Census Area",
// " Township" (type word codes)
static const unsigned char kSuffix[] = {4, 5, 6, 2, 7};

ENC_ONLY inline long KreisVon(const unsigned char* x, long xn, int st, const Seite& s, const char* suf,
                              const std::string& slot);
ENC_ONLY inline bool KreisWieder(const unsigned char* x, long xn, int st, const std::string& slot);
ENC_ONLY inline long GliedVon(int st, const Seite& s, const char* suf, const std::string& slot);
ENC_ONLY inline long GliedName(const std::string& nm);
ENC_ONLY inline long NachbarVon(const unsigned char* x, long xn, int st);
// a value slot from a number, "@" in front for a member index; a copy through
// push_back(const&) (no rvalue insert instance of vector<string>)
ENC_ONLY __attribute__((noinline)) inline void WertDez(std::vector<std::string>* w, long x, bool at) {
  std::string t = zahl::Dez(x);
  if (at) t.insert(0, "@");
  w->push_back(t);
}
ENC_ONLY inline bool StaatLink(const unsigned char* u, long n, long i, const Seite& s, Abstr* a, long* weiter) {
  // u[i..] starts with "[["
  long j = i + 2, bar = -1;
  for (; j + 1 < n && j < i + 160; ++j) {
    const unsigned char c = u[j];
    if (c == '\n' || c == '[' || c == '{' || c == '}') return false;
    if (c == ']') break;
    if (c == '|') { if (bar >= 0) return false; bar = j; }
  }
  if (j + 1 >= n || u[j] != ']' || u[j + 1] != ']') return false;
  const long xe = bar >= 0 ? bar : j;   // X = u[i+2, xe)
  long k = -1;
  for (long q = xe - 2; q >= i + 2; --q) if (u[q] == ',' && u[q + 1] == ' ') { k = q; break; }
  if (k <= i + 2) return false;
  int st = -1;
  const char* sw = kStaatFeld;
  for (int q = 0; q < kStaaten; ++q) {
    const long l = (long)strlen(sw);
    if (l == xe - (k + 2) && memcmp(u + k + 2, sw, (size_t)l) == 0) { st = q; break; }
    sw += l + 1;
  }
  if (st < 0) return false;
  const unsigned char* x = u + i + 2;
  long xn = k - (i + 2);
  const unsigned char* nm = (const unsigned char*)(s.p + s.t0);
  std::string& key = a->key;
  {
    const long nbk = NachbarVon(x, xn, st);
    if (nbk >= 0) {   // an adjacent county: a value slot for "X County, S"
      key += "[[";
      key.push_back((char)kNachbar);
      WertDez(&a->slot_vals, nbk, false);
      if (bar >= 0) {
        key.push_back('|');
        const unsigned char* y = u + bar + 1;
        const long yn = j - (bar + 1);
        if (yn == xn && memcmp(y, x, (size_t)xn) == 0) {
          key.push_back((char)kWdh);
        } else {
          key.push_back((char)kText);
          const std::string tx((const char*)y, (size_t)yn);
          a->text.push_back(tx);
        }
      }
      key += "]]";
      *weiter = j + 2;
      return true;
    }
  }
  key += "[[";
  if (xn > 9 && memcmp(x, "Category:", 9) == 0) { key += "Category:"; x += 9; xn -= 9; }
  std::string slot;
  const bool name = (xn == s.basis && memcmp(x, nm, (size_t)xn) == 0 && st == s.staat);
  const char* suf = nullptr;
  if (name) {
    key.push_back((char)kName);
  } else {
    long sl = xn;
    for (unsigned char q : kSuffix) {
      const char* f = Nth(kTypwortFeld, q);
      const long fl = (long)strlen(f);
      if (xn > fl && memcmp(x + xn - fl, f, (size_t)fl) == 0) { sl = xn - fl; suf = f; break; }
    }
    slot.assign((const char*)x, (size_t)sl);
    const long kseite = KreisVon(x, xn, st, s, suf, slot);
    const long gj = kseite < 0 ? GliedVon(st, s, suf, slot) : -1;
    if (kseite < 0 && suf && st == s.staat && kreis_akt >= 0 && KreisWieder(x, xn, st, slot)) {
      key.push_back((char)kKreisW);   // the county of the record again
      key += suf;
    } else
    if (gj >= 0) {   // a member of the county list: a value slot, no text slot
      key.push_back((char)kGlied);
      WertDez(&a->slot_vals, gj, true);
    } else
    if (kseite >= 0) {   // the county of the place: a value slot, no text slot
      key.push_back((char)kKreis);
      if (suf) key += suf;
      WertDez(&a->slot_vals, kseite, false);
    } else
    {
      key.push_back((char)kText);
      if (suf) key += suf;
      a->text.push_back(slot);
    }
  }
  key += ", ";
  if (st == s.staat) key.push_back((char)kStaatP); else key += Nth(kStaatFeld, st);
  if (bar >= 0) {
    key.push_back('|');
    const unsigned char* y = u + bar + 1;
    const long yn = j - (bar + 1);
    if (name && yn == s.basis && memcmp(y, nm, (size_t)yn) == 0) {
      key.push_back((char)kName);
    } else if (!name && yn == (long)slot.size() && memcmp(y, slot.data(), (size_t)yn) == 0) {
      key.push_back((char)kWdh);
    } else if (!name && suf && yn == (long)(slot.size() + strlen(suf)) && memcmp(y, slot.data(), slot.size()) == 0 &&
               memcmp(y + slot.size(), suf, strlen(suf)) == 0) {
      // "[[X County, S|X County]]": the label repeats the slot with its suffix
      key.push_back((char)kWdh);
      key += suf;
    } else {
      key.push_back((char)kText);
      const std::string tx((const char*)y, (size_t)yn);
      a->text.push_back(tx);
    }
  }
  key += "]]";
  *weiter = j + 2;
  return true;
}

ENC_ONLY inline void Abstrahiere(const unsigned char* u, long n, const Seite& s, Abstr* a) {
  a->key.clear(); a->slot_vals.clear(); a->text.clear();
  const unsigned char* nm = (const unsigned char*)(s.p + s.t0);
  const long nb = s.basis, nt = s.t1 - s.t0;
  const char* st = Nth(kStaatFeld, s.staat);
  const long ns = (long)strlen(st);
  const char* po = kPostFeld + 2 * s.staat;
  const bool liste = (n > 0 && u[0] == '*');
  liste_akt = liste;
  if (liste && glied) {   // "*Name" / "* Name": a member without a link
    long i0 = 1;
    if (i0 < n && u[i0] == ' ') ++i0;
    long e1 = i0;
    while (e1 < n && u[e1] != '\n') ++e1;
    while (e1 > i0 && u[e1 - 1] == ' ') --e1;
    if (e1 > i0) {
      const long j = GliedName(std::string((const char*)u + i0, (size_t)(e1 - i0)));
      if (j >= 0) {
        a->key.assign((const char*)u, (size_t)i0);
        a->key.push_back((char)kGlied);
        a->key.append((const char*)u + e1, (size_t)(n - e1));
        WertDez(&a->slot_vals, j, true);
        return;
      }
    }
  }
  int link = 0, vorl = 0;   // depth of [[ ]] and {{ }}
  long i = 0;
  while (i < n) {
    const unsigned char c = u[i];
    const bool grenze = (i == 0 || !Alnum(u[i - 1]));
    if (c == '[' && i + 1 < n && u[i + 1] == '[' && link == 0) {
      long w = 0;
      if (StaatLink(u, n, i, s, a, &w)) { i = w; continue; }
    }
    if (c == '[' && i + 1 < n && u[i + 1] == '[') { ++link; a->key += "[["; i += 2; continue; }
    if (c == ']' && i + 1 < n && u[i + 1] == ']' && link > 0) { --link; a->key += "]]"; i += 2; continue; }
    if (c == '{' && i + 1 < n && u[i + 1] == '{') { ++vorl; a->key += "{{"; i += 2; continue; }
    if (c == '}' && i + 1 < n && u[i + 1] == '}' && vorl > 0) { --vorl; a->key += "}}"; i += 2; continue; }
    if (grenze && nt >= 2 && i + nt <= n && memcmp(u + i, nm, (size_t)nt) == 0 &&
        (i + nt == n || !Alnum(u[i + nt]))) {
      a->key.push_back((char)kTitel); i += nt; continue;
    }
    if (grenze && nb >= 2 && i + nb <= n && memcmp(u + i, nm, (size_t)nb) == 0 &&
        (i + nb == n || !Alnum(u[i + nb]))) {
      a->key.push_back((char)kName); i += nb; continue;
    }
    if (grenze && i + ns <= n && memcmp(u + i, st, (size_t)ns) == 0 && (i + ns == n || !Alnum(u[i + ns]))) {
      a->key.push_back((char)kStaatP); i += ns; continue;
    }
    if (grenze && i + 2 <= n && u[i] == (unsigned char)po[0] && u[i + 1] == (unsigned char)po[1] &&
        (i + 2 == n || !Alnum(u[i + 2]) || (i + 5 <= n && memcmp(u + i + 2, "Map", 3) == 0))) {
      a->key.push_back((char)kPostP); i += 2; continue;
    }
    if (liste && grenze && link == 0) {
      bool r = false;
      for (const char* w = kRichtungFeld; w < kRichtungFeld + sizeof kRichtungFeld; w += strlen(w) + 1) {
        const long l = (long)strlen(w);
        if (i + l > n || (i + l < n && Alnum(u[i + l]))) continue;
        const bool gross = (c == (unsigned char)(w[0] - 32));
        if ((c != (unsigned char)w[0] && !gross) || memcmp(u + i + 1, w + 1, (size_t)(l - 1)) != 0) continue;
        a->key.push_back((char)(gross ? kRichtG : kRicht));
        const std::string rw(w, (size_t)l);
        a->slot_vals.push_back(rw);
        i += l;
        r = true;
        break;
      }
      if (r) continue;
    }
    // number
    if (link == 0 && grenze && !(i > 0 && (u[i - 1] == '.' || u[i - 1] == '#'))) {
      long j = i;
      if (Ziffer(c) || (c == '-' && i + 1 < n && Ziffer(u[i + 1]))) {
        if (c == '-') ++j;
        while (j < n && Ziffer(u[j])) ++j;
        while (j + 3 < n && u[j] == ',' && Ziffer(u[j + 1]) && Ziffer(u[j + 2]) && Ziffer(u[j + 3]) &&
               (j + 4 == n || !Ziffer(u[j + 4]))) j += 4;
        bool dez = false;
        if (j + 1 < n && u[j] == '.' && Ziffer(u[j + 1])) {
          dez = true; ++j;
          while (j < n && Ziffer(u[j])) ++j;
        }
        const bool frei = (j == n || !Alnum(u[j]));
        const bool hoch = (i >= 4 && memcmp(u + i - 4, "&gt;", 4) == 0);
        if (frei && (vorl == 0 || dez) && !(hoch && !dez && j - i <= 2)) {
          a->key.push_back((char)kZahl);
          const std::string zw((const char*)u + i, (size_t)(j - i));
          a->slot_vals.push_back(zw);
          i = j;
          continue;
        }
      }
    }
    a->key.push_back((char)c);
    ++i;
  }
}

// units of a text body: ends after '\n', and after ". " plus spaces outside
// links when the next byte is not a lower-case letter
ENC_ONLY inline void Einheiten(const unsigned char* t, long m, std::vector<long>* ende) {
  ende->clear();
  int tiefe = 0;
  long anf = 0;   // start of the running unit
  for (long i = 0; i < m; ++i) {
    const unsigned char c = t[i];
    if (c == '[' && i + 1 < m && t[i + 1] == '[') { ++tiefe; ++i; continue; }
    if (c == ']' && i + 1 < m && t[i + 1] == ']') { if (tiefe) --tiefe; ++i; continue; }
    if (c == '\n') {
      tiefe = 0;
      ende->push_back(i + 1);
      anf = i + 1;
      continue;
    }
    if (c == '.' && tiefe == 0 && i + 1 < m && t[i + 1] == ' ') {
      long j = i + 1;
      while (j < m && t[j] == ' ') ++j;
      if (j < m && !(t[j] >= 'a' && t[j] <= 'z') && t[j] != '\n') { ende->push_back(j); anf = j; i = j - 1; }
    }
  }
  if (ende->empty() || ende->back() != m) ende->push_back(m);
}

// Sentence family of a variant: the skeleton without link targets, footnote
// references, tense, unit spelling and spacing.  Only used to group variants.
ENC_ONLY inline std::string Skelett(const std::string& k) {
  static const struct { char a[27], b[6]; } ers[] = {
    {" were ", " are "}, {" was ", " is "}, {" had ", " has "}, {" have ", " has "},
    {"km&amp;sup2;", "km2"}, {"mi&amp;sup2;", "mi2"}, {"km\xc2\xb2", "km2"}, {"mi\xc2\xb2", "mi2"},
    {"km&lt;sup&gt;2&lt;/sup&gt;", "km2"}, {"mi&lt;sup&gt;2&lt;/sup&gt;", "mi2"}};
  std::string o;
  const size_t n = k.size();
  size_t i = 0;
  while (i < n) {
    if (k.compare(i, 5, "{{GR|") == 0) {
      size_t e = k.find("}}", i);
      if (e != std::string::npos) { i = e + 2; continue; }
    }
    if (k.compare(i, 23, "[[Geographic references") == 0) {
      size_t e = k.find("]]", i);
      if (e != std::string::npos) { i = e + 2; continue; }
    }
    if (k.compare(i, 2, "[[") == 0) {
      size_t e = k.find("]]", i + 2);
      if (e != std::string::npos) {
        std::string in = k.substr(i + 2, e - i - 2);
        size_t bar = in.size();   // the last '|', npos when none
        while (bar-- > 0 && in[bar] != '|') {}
        bool ganz = false;
        ganz = in.find((char)kGlied) != std::string::npos;   // the member link keeps its target
        ganz = ganz || in.find((char)kNachbar) != std::string::npos;
        if (!ganz && bar != std::string::npos && in.compare(0, 9, "Category:") != 0 && in.compare(0, 6, "Image:") != 0)
          in = in.substr(bar + 1);
        o += "[["; o += in; o += "]]";
        i = e + 2;
        continue;
      }
    }
    bool hit = false;
    for (const auto& r : ers) {
      const size_t l = strlen(r.a);
      if (k.compare(i, l, r.a) == 0) { o += r.b; i += l; hit = true; break; }
    }
    if (hit) continue;
    if (k[i] == ' ' && !o.empty() && o.back() == ' ') { ++i; continue; }
    o.push_back(k[i++]);
  }
  size_t oe = o.size();
  while (oe && (o[oe - 1] == ' ' || o[oe - 1] == '\n' || o[oe - 1] == '\t')) --oe;
  o.erase(oe);
  return o;
}

// A family is part of the automatic form if its skeleton carries a value slot
// or a placeholder, or is one of the section headings of the bot; interwiki
// lines are never part of it.
ENC_ONLY inline bool Automatisch(const std::string& sk) {
  if (sk.size() >= 5 && sk.compare(0, 2, "[[") == 0) {
    size_t j = 2;
    while (j < sk.size() && sk[j] >= 'a' && sk[j] <= 'z') ++j;
    if (j > 2 && j <= 6 && j < sk.size() && sk[j] == ':') return false;
  }
  for (unsigned char c : sk)
    if (c == kZahl || c == kText || c == kName || c == kStaatP || c == kTitel || c == kPostP ||
        c == kRicht || c == kRichtG) return true;
  for (unsigned char c : sk) if (c == kKreis) return true;
  for (unsigned char c : sk) if (c == kGlied) return true;
  for (unsigned char c : sk) if (c == kNachbar) return true;
  // "[[X County, S|X County]]": the skeleton keeps the label, the repeat of the
  // slot with its suffix; the family stays automatic as with the slot before
  for (size_t i = 0; i + 1 < sk.size(); ++i) {
    if ((unsigned char)sk[i] != kWdh) continue;
    for (unsigned char q : kSuffix) {
      const char* f = Nth(kTypwortFeld, q);
      if (sk.compare(i + 1, strlen(f), f) == 0) return true;
    }
  }
  if (sk.size() >= 4 && sk[0] == '=' && sk[1] == '=') {
    static const char head[][15] = {"Geography", "Demographics", "External links", "Adjacent", "Cities",
                                    "Towns", "Villages", "Townships", "Communities", "Boroughs"};
    for (const auto& w : head) if (sk.find(w) != std::string::npos) return true;
  }
  return false;
}

// Field labels of the value slots of a variant (encoder; the labels travel in
// the block).  The sentence is recognised by its key words, the slots by their
// place in it and by the text right behind them; unrecognised slots get F_X and
// are written as they are.
// Etiketten: the text behind value slot k of v (up to the next slot, at most 40
// bytes) starts with w
ENC_ONLY inline bool NachBeginnt(const std::string& v, const std::vector<size_t>& pos, size_t k, const char* w) {
  const size_t a = pos[k] + 1, e = k + 1 < pos.size() ? pos[k + 1] : v.size();
  const size_t l = strlen(w);
  return (e - a < 40 ? e - a : 40) >= l && memcmp(v.data() + a, w, l) == 0;
}
// the text in front of value slot k (behind the slot before, at most 60 bytes) ends with w
ENC_ONLY inline bool VorEndet(const std::string& v, const std::vector<size_t>& pos, size_t k, const char* w) {
  const size_t b = k > 0 ? pos[k - 1] + 1 : 0;
  const size_t l = strlen(w);
  return (pos[k] - b > 60 ? 60 : pos[k] - b) >= l && memcmp(v.data() + pos[k] - l, w, l) == 0;
}
// w lies inside v[a, a + n) (its first occurrence from a on ends inside)
ENC_ONLY inline bool HatIn(const std::string& v, size_t a, size_t n, const char* w) {
  const size_t r = v.find(w, a);
  return r != std::string::npos && r + strlen(w) <= a + n;
}
ENC_ONLY inline void Etiketten(const std::string& v, std::vector<unsigned char>* lab) {
  using namespace zahl;
  bool sonder = v.find((char)kKreis) != std::string::npos;
  sonder = sonder || v.find((char)kGlied) != std::string::npos;
  sonder = sonder || v.find((char)kNachbar) != std::string::npos;
  if (sonder) {
    // the county slot is F_KREIS, a list member F_GLIED; the rules of the sentence see a letter there
    std::string r(v);
    for (char& c : r) if ((unsigned char)c == kKreis || (unsigned char)c > kKreis && IstWert((unsigned char)c)) c = 'K';
    std::vector<unsigned char> l2;
    Etiketten(r, &l2);
    lab->clear();
    size_t q = 0;
    for (unsigned char c : v) {
      if (!IstWert(c)) continue;
      if (c == kKreis) lab->push_back((unsigned char)F_KREIS);
      else if (c == kGlied) lab->push_back((unsigned char)F_GLIED);
      else if (c == kNachbar) lab->push_back((unsigned char)F_NACHBAR);
      else lab->push_back(q < l2.size() ? l2[q++] : (unsigned char)F_X);
    }
    return;
  }
  lab->clear();
  std::vector<size_t> pos;
  for (size_t i = 0; i < v.size(); ++i) if (IstWert((unsigned char)v[i])) pos.push_back(i);
  lab->assign(pos.size(), F_X);
  auto hat = [&](const char* w) { return v.find(w) != std::string::npos; };
  const size_t n = pos.size();
  for (size_t k = 0; k < n; ++k) if ((unsigned char)v[pos[k]] != kZahl) (*lab)[k] = F_X;   // directions
  // percent slots in order (the number is followed by '%')
  std::vector<size_t> pz;
  for (size_t k = 0; k < n; ++k) if ((unsigned char)v[pos[k]] == kZahl && NachBeginnt(v, pos, k, "%")) pz.push_back(k);
  auto zuteilen = [&](std::initializer_list<int> fs) {
    size_t i = 0;
    for (int f : fs) { if (i >= pz.size()) break; (*lab)[pz[i++]] = (unsigned char)f; }
  };
  if (hat(" people, ") && hat(" households, and ") && hat(" families residing")) {
    for (size_t k = 0; k < n; ++k) {
      if (NachBeginnt(v, pos, k, " people")) (*lab)[k] = F_POP;
      else if (NachBeginnt(v, pos, k, " households")) (*lab)[k] = F_HH;
      else if (NachBeginnt(v, pos, k, " families")) (*lab)[k] = F_FAM;
    }
  } else if (hat("housing units")) {
    for (size_t k = 0; k < n; ++k) {
      if (NachBeginnt(v, pos, k, " housing")) (*lab)[k] = F_HU;
      else if (NachBeginnt(v, pos, k, "/km") || NachBeginnt(v, pos, k, " persons/km")) (*lab)[k] = F_HUD_KM;
      else if (NachBeginnt(v, pos, k, "/mi") || NachBeginnt(v, pos, k, " persons/mi")) (*lab)[k] = F_HUD_MI;
    }
  } else if (hat("population density")) {
    for (size_t k = 0; k < n; ++k) {
      if (NachBeginnt(v, pos, k, "/km") || NachBeginnt(v, pos, k, " persons/km") || NachBeginnt(v, pos, k, " inhabitants per km")) (*lab)[k] = F_DENS_KM;
      else if (NachBeginnt(v, pos, k, "/mi") || NachBeginnt(v, pos, k, " persons/mi")) (*lab)[k] = F_DENS_MI;
    }
  } else if (hat("racial makeup")) {
    for (size_t k : pz) {
      // the text behind the slot up to the next slot, at most 70 bytes
      const size_t a = pos[k] + 1, la = std::min<size_t>(70, v.size() - a);
      const size_t e = k + 1 < n ? pos[k + 1] - a : la;
      const size_t lb = std::min(e, la);
      int f = F_X;
      if (HatIn(v, a, lb, "two or more")) f = F_R6;
      else if (HatIn(v, a, lb, "other races")) f = F_R5;
      else if (HatIn(v, a, lb, "Pacific Islander")) f = F_R4;
      else if (HatIn(v, a, lb, "Asian")) f = F_R3;
      else if (HatIn(v, a, lb, "Native American")) f = F_R2;
      else if (HatIn(v, a, lb, "African American") || HatIn(v, a, lb, "Black")) f = F_R1;
      else if (HatIn(v, a, lb, "White")) f = F_R0;
      (*lab)[k] = (unsigned char)f;
    }
  } else if (hat("of any race")) {
    if (!pz.empty()) (*lab)[pz[0]] = F_HISP;
  } else if (hat("households out of which")) {
    for (size_t k = 0; k < n; ++k) {
      if (NachBeginnt(v, pos, k, " households")) (*lab)[k] = F_HH2;
      else if (NachBeginnt(v, pos, k, "% have children") || NachBeginnt(v, pos, k, "% had children")) (*lab)[k] = F_HH_KID;
      else if (NachBeginnt(v, pos, k, "% are [[Marr") || NachBeginnt(v, pos, k, "% were [[Marr") || NachBeginnt(v, pos, k, "% are married") ||
               NachBeginnt(v, pos, k, "% were married")) (*lab)[k] = F_HH_MAR;
      else if (NachBeginnt(v, pos, k, "% have a ") || NachBeginnt(v, pos, k, "% had a ")) (*lab)[k] = F_HH_FEM;
      else if (NachBeginnt(v, pos, k, "% are non-") || NachBeginnt(v, pos, k, "% were non-")) (*lab)[k] = F_HH_NF;
    }
  } else if (hat("made up of individuals")) {
    for (size_t k = 0; k < n; ++k) {
      if (NachBeginnt(v, pos, k, "% of all households")) (*lab)[k] = F_HH_IND;
      else if (NachBeginnt(v, pos, k, "% have someone") || NachBeginnt(v, pos, k, "% had someone")) (*lab)[k] = F_HH_65;
    }
  } else if (hat("household size")) {
    for (size_t k = 0; k < n; ++k) {
      if (VorEndet(v, pos, k, "household size is ") || VorEndet(v, pos, k, "household size was ")) (*lab)[k] = F_HHS;
      else if (VorEndet(v, pos, k, "family size is ") || VorEndet(v, pos, k, "family size was ")) (*lab)[k] = F_FS;
    }
  } else if (hat("spread out with")) {
    zuteilen({F_A0, F_A1, F_A2, F_A3, F_A4});
  } else if (hat("median age")) {
    for (size_t k = 0; k < n; ++k) if (NachBeginnt(v, pos, k, " years")) { (*lab)[k] = F_MED; break; }
  } else if (hat("For every") && hat("females")) {
    const bool m18 = hat("age 18") || hat("females age");
    for (size_t k = 0; k < n; ++k) if (NachBeginnt(v, pos, k, " males")) (*lab)[k] = m18 ? F_M18 : F_M100;
  } else if (hat("median income for a household")) {
    int i = 0;
    for (size_t k = 0; k < n && i < 2; ++k) if (VorEndet(v, pos, k, "$")) (*lab)[k] = i++ == 0 ? F_INC_HH : F_INC_FAM;
  } else if (hat("median income of $") && hat("for females")) {
    int i = 0;
    for (size_t k = 0; k < n && i < 2; ++k) if (VorEndet(v, pos, k, "$")) (*lab)[k] = i++ == 0 ? F_INC_M : F_INC_F;
  } else if (hat("per capita income")) {
    for (size_t k = 0; k < n; ++k) if (VorEndet(v, pos, k, "$")) { (*lab)[k] = F_INC_PC; break; }
  } else if (hat("of families") && hat("poverty line")) {
    zuteilen({F_POV, F_POV_FAM});
  } else if (hat("Out of the total population")) {
    zuteilen({F_POV_U18, F_POV_65});
  } else if (hat("of it is land")) {
    int i = 0;
    static const int fs4[4] = {F_LAND_KM, F_LAND_MI, F_WAT_KM, F_WAT_MI};
    for (size_t k = 0; k < n; ++k) {
      if (NachBeginnt(v, pos, k, "% is water")) { (*lab)[k] = F_WAT_PCT2; continue; }
      if (i < 4) (*lab)[k] = (unsigned char)fs4[i++];
    }
  } else if (hat("total area of")) {
    int i = 0;
    for (size_t k = 0; k < n && i < 2; ++k) (*lab)[k] = i++ == 0 ? F_TOT_KM : F_TOT_MI;
  } else if (hat("The total area is") && hat("% water")) {
    if (!pz.empty()) (*lab)[pz[0]] = F_WAT_PCT;
  } else if (hat("is located at") && n == 8) {
    static const int fs8[8] = {F_LAT_D, F_LAT_M, F_LAT_S, F_LON_D, F_LON_M, F_LON_S, F_LAT, F_LON};
    for (size_t k = 0; k < 8; ++k) (*lab)[k] = (unsigned char)fs8[k];
  } else if ((hat("Mapit-US-cityscale") || hat("Geolinks-US-cityscale")) && n == 2) {
    (*lab)[0] = F_MLAT;
    (*lab)[1] = F_MLON;
  } else if (hat("total population of") || hat("the population is") || hat("the population was")) {
    if (n >= 1) (*lab)[0] = F_POP_I;
  } else {
    for (size_t k = 0; k < n; ++k) if (PvJahr(v, (int)k)) (*lab)[k] = F_PV;
  }
}

// A number slot whose value is the same in every unit of the variant is folded
// into the variant text (the "65" of "65 years of age or older").
// A key string -> dense index, open addressing (encoder only).  The keys live in
// the nodes of the owner (stable addresses), the table points to them.
struct STab {
  struct E { unsigned long h; const std::string* k; long i; };
  E* t = nullptr;
  unsigned long cap = 0, n = 0;
  STab() = default;
  STab(const STab&) = delete;
  STab& operator=(const STab&) = delete;
  ~STab() { free(t); }
  ENC_ONLY static unsigned long Hash(const char* p, size_t l) {
    unsigned long h = 0xcbf29ce484222325ull;
    for (size_t j = 0; j < l; ++j) h = (h ^ (unsigned char)p[j]) * 0x100000001b3ull;
    return h;
  }
  // the index of the key p[0..l), or -1
  ENC_ONLY long Find(const char* p, size_t l) const {
    if (!n) return -1;
    const unsigned long h = Hash(p, l);
    for (unsigned long j = h & (cap - 1);; j = (j + 1) & (cap - 1)) {
      const E& e = t[j];
      if (!e.k) return -1;
      if (e.h == h && e.k->size() == l && memcmp(e.k->data(), p, l) == 0) return e.i;
    }
  }
  // a new key (not in the table) with index i
  ENC_ONLY void Add(const std::string* k, long i) {
    if (2 * (n + 1) > cap) {
      const unsigned long c = cap ? 2 * cap : 64;
      E* a = (E*)calloc(c, sizeof(E));
      if (!a) abort();
      for (unsigned long j = 0; j < cap; ++j) {
        if (!t[j].k) continue;
        unsigned long q = t[j].h & (c - 1);
        while (a[q].k) q = (q + 1) & (c - 1);
        a[q] = t[j];
      }
      free(t);
      t = a;
      cap = c;
    }
    const unsigned long h = Hash(k->data(), k->size());
    unsigned long q = h & (cap - 1);
    while (t[q].k) q = (q + 1) & (cap - 1);
    t[q].h = h;
    t[q].k = k;
    t[q].i = i;
    ++n;
  }
};
struct Variante {
  std::string key;                  // the variant key (abstracted unit)
  int seiten = 0;
  long letzte = -1;
  long neinh = 0;                   // units
  std::vector<std::string> konst;   // value of the first unit per slot
  std::vector<long> ngleich;        // units with that value per slot
  std::vector<char> gleich;         // slot folded (value in >= 99 % of the units)
  std::vector<unsigned char> lab;   // field label per remaining value slot
  int fam = -1;                     // family index (per type, after pass 1)
  int nr = -1;                      // variant number inside the family
  std::string text;                 // the variant as written to the block
};
struct Familie {
  std::string skelett;
  long seiten = 0, letzte = -1;
  double pos = 0.0;                 // sum of relative positions
  long n = 0;
  int ord = -1;                     // index in the form of the type (-1: not automatic)
  bool doppel = false;              // further position of a family
  Familie* basis = nullptr;         // the family it repeats
  std::vector<Variante*> var;
};
struct Form {
  std::vector<Familie*> fam;              // the families (owned), in order of appearance
  STab fam_t;                             // skeleton -> index into fam
  std::vector<Familie*> pos;              // the automatic form, in order
  long seiten = 0;
  Form() = default;
  Form(const Form&) = delete;
  Form& operator=(const Form&) = delete;
  ~Form() { for (Familie* f : fam) delete f; }
};
// the family of a skeleton, created when absent (the operator[] of a map)
ENC_ONLY inline Familie* FamHol(Form& F, const std::string& sk) {
  const long i = F.fam_t.Find(sk.data(), sk.size());
  if (i >= 0) return F.fam[(size_t)i];
  Familie* f = new Familie();
  f->skelett = sk;
  F.fam_t.Add(&f->skelett, (long)F.fam.size());
  F.fam.push_back(f);
  return f;
}
// "<name>, <state>" of a County page -> its page index
struct KreisTitel { std::string t; long page_index; };

struct Kodierer {
  std::vector<Variante*> var[kNTypen];   // the variants (owned) per type, in order of appearance
  STab var_t[kNTypen];                   // key -> index into var
  Form form[kNTypen];
  zahl::Typmodell tm[kNTypen];
  std::string block[kNTypen];            // records per type (the head is written at the end)
  long page_index = -1;                       // page index
  long letzte[kNTypen] = {-1, -1, -1, -1, -1, -1, -1, -1};   // last record page per type
  std::vector<long> lauf_seiten[kNTypen];   // record pages per type (the run in front of the records)
  unsigned long long nrec = 0;
  int pass = 1;
  std::vector<std::vector<int>> dopp[kNTypen];   // position -> its further positions
  std::vector<double> dsum[kNTypen];             // position q -> [2q] left-out units, [2q + 1] sum of their place
  std::vector<KreisTitel*> kreis_titel;   // "X County, State" -> page index of the County page (owned)
  STab kreis_titel_t;                     // the title -> index into kreis_titel
  zahl::LTab kreis_ok;                    // County pages usable as F_KREIS
  int ti_akt = -1;                                     // type of the running page
  bool kreis_genutzt = false;                          // the page has its county slot already
  long kreis_seite = -1;                               // county of the running page (first county link)
  LMap<std::vector<std::pair<std::string, long>>> glieder1;   // pass 1: members as seen
  long glied_prov = 0;                                 // next member while the units of the page are abstracted
  long glied_dek = 0;                                  // next member as the decoder will see it (pass 2)
  std::string ob;                        // output page
  Abstr ab;
  std::vector<long> ende;
  std::string chk;
  Kodierer() = default;
  Kodierer(const Kodierer&) = delete;
  Kodierer& operator=(const Kodierer&) = delete;
  ~Kodierer() {
    for (int ti = 0; ti < kNTypen; ++ti) for (Variante* v : var[ti]) delete v;
    for (KreisTitel* k : kreis_titel) delete k;
  }
};
inline Kodierer* kd = nullptr;
// the variant of a key (type ti), created when absent (the operator[] of a map)
ENC_ONLY inline Variante& VarHol(Kodierer& K, int ti, const std::string& key) {
  const long i = K.var_t[ti].Find(key.data(), key.size());
  if (i >= 0) return *K.var[ti][(size_t)i];
  Variante* v = new Variante();
  v->key = key;
  K.var_t[ti].Add(&v->key, (long)K.var[ti].size());
  K.var[ti].push_back(v);
  return *v;
}
// the County page of the title "<x>, <state st>", or -1
ENC_ONLY inline long KreisTitelSeite(const unsigned char* x, long xn, int st) {
  Kodierer& K = *kd;
  std::string t((const char*)x, (size_t)xn);
  t += ", ";
  t += Nth(kStaatFeld, st);
  const long i = K.kreis_titel_t.Find(t.data(), t.size());
  return i < 0 ? -1 : K.kreis_titel[(size_t)i]->page_index;
}

ENC_ONLY inline bool AbstrGeprueft1(const unsigned char* u, long n, const Seite& s, Abstr* a);
// the County page of a county link of the running place page, or -1: the first
// county link of the page, same state, a County page with a record whose short
// title plus the suffix of the link is the link name
// a link to the county of the running place page again
ENC_ONLY inline bool KreisWieder(const unsigned char* x, long xn, int st, const std::string& slot) {
  Kodierer& K = *kd;
  if (K.ti_akt < 0 || kTypen[K.ti_akt] == 'k') return false;
  const long kp = KreisTitelSeite(x, xn, st);
  if (kp < 0 || kp != kreis_akt) return false;
  const std::string* nm = kreisname.Find(kp);
  return nm && *nm == slot;
}
ENC_ONLY inline long KreisVon(const unsigned char* x, long xn, int st, const Seite& s, const char* suf,
                              const std::string& slot) {
  Kodierer& K = *kd;
  if (!suf || st != s.staat || K.kreis_genutzt || K.ti_akt < 0 || kTypen[K.ti_akt] == 'k') return -1;
  const long kp = KreisTitelSeite(x, xn, st);
  if (kp < 0 || K.kreis_ok.Find(kp) < 0) return -1;
  const std::string* nm = kreisname.Find(kp);
  if (!nm || *nm != slot) return -1;
  K.kreis_genutzt = true;
  K.kreis_seite = kp;
  kreis_akt = kp;
  if (K.pass == 1) {   // pass 1: the member as seen (pass 2 takes the written records)
    long tb = s.basis;
    TitelTyp(s.p + s.t0, s.basis, &tb);
    K.glieder1.Ref(kp).push_back(std::make_pair(std::string(s.p + s.t0, (size_t)tb), K.page_index));
  }
  return kp;
}
// the County page (with a record, not the running page) that a list line of a
// County page links by its whole title "X County, S", or -1
ENC_ONLY inline long NachbarVon(const unsigned char* x, long xn, int st) {
  Kodierer& K = *kd;
  if (K.ti_akt < 0 || kTypen[K.ti_akt] != 'k' || !liste_akt) return -1;
  const long kp = KreisTitelSeite(x, xn, st);
  if (kp < 0 || kp == K.page_index || K.kreis_ok.Find(kp) < 0) return -1;
  const std::string* nm = kreisname.Find(kp);
  const long* ar = KreisArt(kp);
  if (!nm || !ar || ar[1] != st) return -1;
  std::string w(*nm);
  w += Nth(kTypwortFeld, (int)ar[0]);
  if (w.size() != (size_t)xn || memcmp(w.data(), x, (size_t)xn) != 0) return -1;
  return kp;
}
// a member named without a link ("*Name"), from the next member on, or -1
ENC_ONLY inline long GliedName(const std::string& nm) {
  Kodierer& K = *kd;
  if (!glied) return -1;
  for (long j = K.glied_prov; j < (long)glied->size(); ++j)
    if ((*glied)[(size_t)j].first == nm) { K.glied_prov = j + 1; return j; }
  return -1;
}
// the member of the running County page that a link without suffix names, from
// the next member on, or -1
ENC_ONLY inline long GliedVon(int st, const Seite& s, const char* suf, const std::string& slot) {
  Kodierer& K = *kd;
  if (suf || !glied || st != s.staat || !liste_akt) return -1;
  for (long j = K.glied_prov; j < (long)glied->size(); ++j) {
    if ((*glied)[(size_t)j].first == slot) { K.glied_prov = j + 1; return j; }
  }
  return -1;
}
ENC_ONLY inline bool AbstrGeprueft(const unsigned char* u, long n, const Seite& s, Abstr* a) {
  const long gp = kd->glied_prov;
  if (!AbstrGeprueft1(u, n, s, a)) { kd->glied_prov = gp; return false; }
  return true;
}
ENC_ONLY inline bool AbstrGeprueft1(const unsigned char* u, long n, const Seite& s, Abstr* a) {
  Abstrahiere(u, n, s, a);
  const Teile2 t = TitelTeile(s);
  kd->chk.clear();
  size_t wi = 0;
  if (!Expandiere(a->key, a->slot_vals, &wi, a->text, t, &kd->chk) || wi != a->slot_vals.size()) return false;
  if ((long)kd->chk.size() != n || memcmp(kd->chk.data(), u, (size_t)n) != 0) return false;
  for (const std::string& x : a->text) {
    for (unsigned char c : x) if (c < 0x09) return false;
  }
  return true;
}

// longest chain of candidates whose position index does not decrease
// (patience sorting); returns the chosen indices into `pos`, -1 = no candidate
ENC_ONLY inline void Kette(const std::vector<int>& pos, std::vector<char>* best) {
  std::vector<int> spitze, vor(pos.size(), -1), spitze_i;
  for (size_t i = 0; i < pos.size(); ++i) {
    if (pos[i] < 0) continue;
    // first chain end with a position strictly greater (non-decreasing chain)
    size_t lo = 0, hi = spitze.size();
    while (lo < hi) {
      const size_t mi = (lo + hi) / 2;
      if (spitze[mi] <= pos[i]) lo = mi + 1; else hi = mi;
    }
    if (lo > 0) vor[i] = spitze_i[lo - 1];
    if (lo == spitze.size()) { spitze.push_back(pos[i]); spitze_i.push_back((int)i); }
    else { spitze[lo] = pos[i]; spitze_i[lo] = (int)i; }
  }
  best->assign(pos.size(), 0);
  int j = spitze_i.empty() ? -1 : spitze_i.back();
  while (j >= 0) { (*best)[(size_t)j] = 1; j = vor[(size_t)j]; }
}

ENC_ONLY inline int IntAbsteigend(const void* x, const void* y) {
  const int a = *(const int*)x, b = *(const int*)y;
  return (a < b) - (a > b);
}
// longest non-decreasing chain where unit i may stand at p1[i] or p2[i] (-1:
// none); the alternatives of a unit go in in descending order, so a chain never
// takes one unit twice.  wahl: chosen, pw: its position.
ENC_ONLY inline void Kette2(const std::vector<int>& p1, const std::vector<std::vector<int>>& p2,
                            std::vector<char>* best, std::vector<int>* pw) {
  // the nodes (unit, value, predecessor) as three ints each in one vector<int>
  // (the instances of vector<int> exist; a node struct had its own)
  std::vector<int> kn;
  std::vector<int> spitze, spitze_k;
  std::vector<int> c;
  for (size_t i = 0; i < p1.size(); ++i) {
    c.clear();
    if (p1[i] >= 0) {   // the order before the sort does not matter; c is empty: a copy without operator=
      for (int x : p2[i]) c.push_back(x);
      c.push_back(p1[i]);
    }
    qsort(c.data(), c.size(), sizeof(int), IntAbsteigend);
    for (int w : c) {
      if (w < 0) continue;
      size_t lo = 0, hi = spitze.size();
      while (lo < hi) {
        const size_t mi = (lo + hi) / 2;
        if (spitze[mi] <= w) lo = mi + 1; else hi = mi;
      }
      const int unit = (int)i, vor = lo > 0 ? spitze_k[lo - 1] : -1;
      kn.push_back(unit);
      kn.push_back(w);
      kn.push_back(vor);
      const int kz = (int)(kn.size() / 3) - 1;
      if (lo == spitze.size()) { spitze.push_back(w); spitze_k.push_back(kz); }
      else { spitze[lo] = w; spitze_k[lo] = kz; }
    }
  }
  best->assign(p1.size(), 0);
  std::vector<int>(p1.size(), -1).swap(*pw);   // as assign(n, -1)
  int j = spitze_k.empty() ? -1 : spitze_k.back();
  while (j >= 0) {
    const int* k = &kn[3 * (size_t)j];
    (*best)[(size_t)k[0]] = 1;
    (*pw)[(size_t)k[0]] = k[1];
    j = k[2];
  }
}
ENC_ONLY inline int SeiteKod(const char* p, long n, long* outn, const char** ob) {
  Kodierer& K = *kd;
  ++K.page_index;
  if (!Nimmt(p, n)) return 0;
  Seite s;
  Teile(p, n, &s);
  const int ti = TypIndex(Typ(s));
  if (K.pass == 0) {   // pre-pass: the County pages (title -> page, short title)
    if (kTypen[ti] == 'k') {
      long tb = s.basis;
      TitelTyp(p + s.t0, s.basis, &tb);
      std::string t(p + s.t0, (size_t)s.basis);
      t += ", ";
      t += Nth(kStaatFeld, s.staat);
      const long ki = K.kreis_titel_t.Find(t.data(), t.size());
      if (ki >= 0) {   // the last page of a title wins
        K.kreis_titel[(size_t)ki]->page_index = K.page_index;
      } else {
        KreisTitel* kt = new KreisTitel{t, K.page_index};
        K.kreis_titel_t.Add(&kt->t, (long)K.kreis_titel.size());
        K.kreis_titel.push_back(kt);
      }
      kreisname.Setze(K.page_index, std::string(p + s.t0, (size_t)tb));
      bool neu;
      K.kreis_ok.Neu(K.page_index, &neu);
      KreisArtSetze(K.page_index, TitelTyp(p + s.t0, s.basis, &tb), s.staat);
    }
    return 0;
  }
  if (K.pass == 3 && kTypen[ti] != 'k') return 0;   // dry pass: only the County pages
  K.ti_akt = ti;
  K.kreis_genutzt = false;
  K.kreis_seite = -1;
  kreis_akt = -1;
  glied = nullptr;
  K.glied_prov = 0;
  K.glied_dek = 0;
  if (kTypen[ti] == 'k' && (K.pass == 1 || K.pass == 4 || (K.pass == 2 && K.kreis_ok.Find(K.page_index) >= 0))) {   // pass 3 has none
    std::vector<std::pair<std::string, long>>& g = K.pass != 2 ? K.glieder1.Ref(K.page_index) : kreisglieder.Ref(K.page_index);
    SortiereGlieder(g);
    glied = &g;
  }
  const unsigned char* t = (const unsigned char*)p + s.x0;
  const long m = s.x1 - s.x0;
  Einheiten(t, m, &K.ende);
  const long ne = (long)K.ende.size();
  if (K.pass == 1) {
    Form& F = K.form[ti];
    ++F.seiten;
    long a = 0, idx = 0;
    for (long e : K.ende) {
      if (e - a >= kMinLaenge && AbstrGeprueft(t + a, e - a, s, &K.ab)) {
        Variante& v = VarHol(K, ti, K.ab.key);
        if (v.letzte != K.page_index) { v.letzte = K.page_index; ++v.seiten; }
        if (v.neinh++ == 0) {
          for (const std::string& w : K.ab.slot_vals) v.konst.push_back(w);   // konst is empty: a copy
          std::vector<long>(K.ab.slot_vals.size(), 1L).swap(v.ngleich);       // as assign(n, 1)
        } else {
          for (size_t j = 0; j < v.ngleich.size(); ++j) v.ngleich[j] += K.ab.slot_vals[j] == v.konst[j];
        }
        Familie& f = *FamHol(F, Skelett(K.ab.key));
        if (f.letzte != K.page_index) { f.letzte = K.page_index; ++f.seiten; }
        f.pos += (double)idx / (double)ne;
        ++f.n;
      }
      a = e;
      ++idx;
    }
    return 0;
  }
  // pass 2: candidates -> chain -> record + stub
  std::vector<int> pos((size_t)ne, -1);
  std::vector<Variante*> pv((size_t)ne, nullptr);
  std::vector<Abstr> ab((size_t)ne);
  {
    long a = 0;
    for (long i = 0; i < ne; ++i) {
      const long e = K.ende[(size_t)i];
      if (e - a >= kMinLaenge && AbstrGeprueft(t + a, e - a, s, &ab[(size_t)i])) {
        const long vi = K.var_t[ti].Find(ab[(size_t)i].key.data(), ab[(size_t)i].key.size());
        Variante* vp = vi >= 0 ? K.var[ti][(size_t)vi] : nullptr;
        if (vp && vp->fam >= 0) {
          bool passt = true;
          for (size_t j = 0; j < vp->gleich.size(); ++j)
            passt = passt && (!vp->gleich[j] || ab[(size_t)i].slot_vals[j] == vp->konst[j]);
          if (passt) {
            pos[(size_t)i] = vp->fam;
            pv[(size_t)i] = vp;
          }
        }
      }
      a = e;
    }
  }
  std::vector<char> best;
  std::vector<int> pw;
  {
    std::vector<std::vector<int>> pos2(pos.size());
    for (size_t i = 0; i < pos.size(); ++i)
      if (pos[i] >= 0 && (size_t)pos[i] < K.dopp[ti].size())
        for (int x : K.dopp[ti][(size_t)pos[i]]) pos2[i].push_back(x);   // pos2[i] is empty: a copy
    Kette2(pos, pos2, &best, &pw);
  }
  if (K.pass == 4) {   // dry pass: the left-out units and the places between their chosen neighbours
    const Form& F0 = K.form[ti];
    // dsum[ti] is empty here or has this size already (cleared by BaueDoppel, F0 fixed in a pass)
    if (K.dsum[ti].size() < 2 * F0.pos.size()) std::vector<double>(2 * F0.pos.size(), 0.0).swap(K.dsum[ti]);
    auto mitte = [&](int q) { const Familie* f = F0.pos[(size_t)q]; return f->pos / (double)f->n; };
    for (long i = 0; i < ne; ++i) {
      if (pos[(size_t)i] < 0 || best[(size_t)i]) continue;
      double lo = 0.0, hi = 1.0;
      for (long j = i - 1; j >= 0; --j) if (best[(size_t)j]) { lo = mitte(pos[(size_t)j]); break; }
      for (long j = i + 1; j < ne; ++j) if (best[(size_t)j]) { hi = mitte(pos[(size_t)j]); break; }
      K.dsum[ti][2 * (size_t)pos[(size_t)i]] += 1.0;   // at the family's first position
      K.dsum[ti][2 * (size_t)pos[(size_t)i] + 1] += 0.5 * (lo + hi);
    }
    return 0;
  }
  long nauto = 0;
  for (char w : best) nauto += w;
  if (nauto == 0) return 0;
  if (K.pass == 3) {   // this County page gets a record
    if ((size_t)s.staat >= zahl::kreisliste.size()) zahl::kreisliste.resize((size_t)s.staat + 1);
    zahl::kreisliste[(size_t)s.staat].push_back(K.page_index);
    return 0;
  }
  const Form& F = K.form[ti];
  std::vector<std::vector<int>> inst(F.pos.size());   // per position: variant numbers
  std::string& o = K.ob;
  // the title without type word and state: "<name>" (the record keeps both)
  long tbase = s.basis;
  const int tcode = TitelTyp(p + s.t0, s.basis, &tbase);
  o.assign(p, (size_t)(s.t0 + tbase));
  o.append(p + s.t1, (size_t)(s.x0 - s.t1));
  std::vector<Slot> slots;
  std::vector<std::string> slot_values, tok;
  std::vector<unsigned char> bit;   // the gap bits (0/1), in the type LueckeXor works on (no char-vector instances)
  bool run_list = false;
  long a = 0;
  for (long i = 0; i < ne; ++i) {
    const long e = K.ende[(size_t)i];
    if (!best[(size_t)i]) {
      o.append((const char*)t + a, (size_t)(e - a));
      run_list = true;
    } else {
      if (run_list) o.push_back((char)kRun);
      bit.push_back(run_list);
      run_list = false;
      const Variante& v = *pv[(size_t)i];
      inst[(size_t)pw[(size_t)i]].push_back(v.nr);
      for (const std::string& x : ab[(size_t)i].text) {
        o += x;
        o.push_back((char)kEnde);
      }
      size_t k = 0;
      for (size_t j = 0; j < ab[(size_t)i].slot_vals.size(); ++j) {
        if (v.gleich[j]) continue;   // folded into the variant
        Slot sl;
        sl.lab = v.lab[k];
        sl.jahr = PvJahr(v.text, (int)k);
        ++k;
        slots.push_back(sl);
        slot_values.push_back(ab[(size_t)i].slot_vals[j]);
        if (!slot_values.back().empty() && slot_values.back()[0] == '@') {   // absolute member -> members skipped
          const long gj = strtol(slot_values.back().c_str() + 1, nullptr, 10);
          const long sk = gj - K.glied_dek;
          K.glied_dek = gj + 1;
          slot_values.back() = sk ? zahl::Dez(sk) : std::string();
        }
      }
    }
    a = e;
  }
  if (run_list) o.push_back((char)kRun);
  bit.push_back(run_list);
  o.append(p + s.x1, (size_t)(n - s.x1));
  // record
  std::string& b = K.block[ti];
  K.lauf_seiten[ti].push_back(K.page_index);
  B100(&b, (unsigned long)s.staat, 1);
  B100(&b, (unsigned long)tcode, 1);
  for (const std::vector<int>& in : inst) {
    if (in.empty()) { B100(&b, 0, 1); continue; }
    if (in.size() == 1 && in[0] < kMaxVar) { B100(&b, (unsigned long)(in[0] + 1), 1); continue; }
    B100(&b, 99, 1);
    B100(&b, (unsigned long)in.size(), 2);
    for (int v : in) B100(&b, (unsigned long)v, 1);
  }
  {
    std::vector<std::pair<int, int>> u;
    for (size_t j = 0; j < inst.size(); ++j) for (int v : inst[j]) u.push_back(std::make_pair((int)j, v));
    LueckeXor(ti, (int)F.pos.size(), u, &bit, true);   // in place (the bits are 0/1)
  }
  for (size_t g = 0; g < bit.size(); g += 6) {
    unsigned long v = 0;
    for (size_t j = 0; j < 6 && g + j < bit.size(); ++j) v |= (unsigned long)bit[g + j] << j;
    B100(&b, v, 1);
  }
  if (KreisZusatz(kTypen[ti], &slots)) {
    const std::string kw = zahl::Dez(K.kreis_seite);
    slot_values.push_back(kw);
  }
  if (zwei_runde == 0) {
    long kr = kTypen[ti] == 'k' ? K.page_index : -1;
    for (size_t i = 0; i < slots.size() && i < slot_values.size() && kTypen[ti] != 'k'; ++i)
      if (slots[i].lab == zahl::F_KREIS) kr = slot_values[i].empty() ? -1 : strtol(slot_values[i].c_str(), nullptr, 10);
    Anl4Merken(ti, K.page_index, s.staat, kr);
  }
  Anl4Setzen(ti, K.page_index, s.staat);
  if (kTypen[ti] == 'k') NachbarRaenge(K.page_index, s.staat, slots, &slot_values, true);
  if (kTypen[ti] != 'k') {   // a member of its county from now on
    for (size_t i = 0; i < slots.size(); ++i) {
      if (slots[i].lab != zahl::F_KREIS) continue;
      long tb = s.basis;
      TitelTyp(p + s.t0, s.basis, &tb);
      const long kp = strtol(slot_values[i].c_str(), nullptr, 10);
      if (kp >= 0) kreisglieder.Ref(kp).push_back(std::make_pair(std::string(p + s.t0, (size_t)tb), K.page_index));
      break;
    }
  }
#if PORT_CMX_MEAS
  zahl::g_cur_page = K.page_index;   // research V2 (censblock_zahl.h)
#endif
  if (!Zahlen(K.tm[ti], kTypen[ti] == 'k', s.staat, slots, &slot_values, &tok, true)) {
    __builtin_unreachable();
    return -1;
  }
  for (const std::string& x : tok) { b += x; b.push_back('\t'); }
  b.push_back('\x0b');
  K.letzte[ti] = K.page_index;
  ++K.nrec;
  *outn = (long)o.size();
  *ob = o.data();
  return 1;
}
ENC_ONLY inline int SeiteKodCb(const char* p, long n, long* outn, const char** ob) {
  return SeiteKod(p, n, outn, ob);
}

// Escape form of a variant byte outside the alphabet: 0xE4 + (b >> 6),
// 0x80 + (b & 63).  The current line form has no caller (the variant texts go
// into the pseudo article).
ENC_ONLY inline void Esc(std::string* o, unsigned char c) {
  if (c < 0x80 && c != 0x09 && c != 0x0A && c != 0x0B && c != 0x0C && ImAlphabet(c)) {
    o->push_back((char)c);
  } else {
    o->push_back((char)(0xE4 + (c >> 6)));
    o->push_back((char)(0x80 + (c & 63)));
  }
}

// After pass 1: the automatic form of every type.  Families on at least 2 % of
// the pages of the type (and kMinSeiten), skeleton of the automatic form,
// ordered by mean place; variants with kMinSeiten pages, the most frequent
// first, at most kMaxVar.  Ties are broken by the bytes (deterministic).
ENC_ONLY inline int VarVgl(const void* x, const void* y) {   // more pages first, then by the bytes
  const Variante& a = **(const Variante* const*)x;
  const Variante& b = **(const Variante* const*)y;
  if (a.seiten != b.seiten) return a.seiten > b.seiten ? -1 : 1;
  return a.key.compare(b.key);
}
ENC_ONLY inline int FamVgl(const void* x, const void* y) {   // by mean place, then by the skeleton
  const Familie& f = **(const Familie* const*)x;
  const Familie& g = **(const Familie* const*)y;
  const double a = f.pos / (double)f.n, b = g.pos / (double)g.n;
  if (a != b) return a < b ? -1 : 1;
  return f.skelett.compare(g.skelett);
}
ENC_ONLY inline void BaueFormen(Kodierer& K) {
  for (int ti = 0; ti < kNTypen; ++ti) {
    Form& F = K.form[ti];
    // variants -> families
    std::vector<Variante*> vs;   // the order of appearance does not matter: VarVgl is total
    for (Variante* v : K.var[ti]) if (v->seiten >= kMinVar) vs.push_back(v);
    qsort(vs.data(), vs.size(), sizeof(vs[0]), VarVgl);
    const long schwelle = std::max<long>(kMinSeiten, (F.seiten * 2 + 99) / 100);
    std::vector<Familie*> fs;   // FamVgl is total (the skeletons are unique)
    for (Familie* f : F.fam) if (f->seiten >= schwelle && Automatisch(f->skelett)) fs.push_back(f);
    qsort(fs.data(), fs.size(), sizeof(fs[0]), FamVgl);
    for (size_t i = 0; i < fs.size(); ++i) { fs[i]->ord = (int)i; F.pos.push_back(fs[i]); }
    for (Variante* kv : vs) {
      Familie& f = *FamHol(F, Skelett(kv->key));
      if (f.ord < 0) continue;
      if ((int)f.var.size() >= kMaxVar) {
        continue;
      }
      Variante& v = *kv;
      v.fam = f.ord;
      v.nr = (int)f.var.size();
      v.gleich.assign(v.ngleich.size(), 0);
      for (size_t q = 0; q < v.ngleich.size(); ++q)
        v.gleich[q] = v.ngleich[q] >= kMinSeiten && v.ngleich[q] * 100 >= v.neinh * 99;
      {   // a county slot is never folded: its value is a page index, not text
        size_t q = 0;
        for (unsigned char c : kv->key) {
          if (!IstWert(c)) continue;
          if ((c == kKreis || c > kKreis) && q < v.gleich.size()) v.gleich[q] = 0;
          ++q;
        }
      }
      size_t j = 0;
      for (unsigned char c : kv->key) {
        if (IstWert(c) && v.gleich[j++]) {
          if (c == kRichtG) { v.text.push_back((char)(v.konst[j - 1][0] - 32)); v.text += v.konst[j - 1].substr(1); }
          else v.text += v.konst[j - 1];
        } else {
          v.text.push_back((char)c);
        }
      }
      Etiketten(v.text, &v.lab);
      f.var.push_back(&v);
    }
  }
}

// after the dry pass: second positions for families with many left-out units
// (at least 20 and 1 % of the pages of the type), placed at the mean of the
// places between the chosen neighbours of those units; the order is rebuilt.
ENC_ONLY inline void BaueDoppel(Kodierer& K) {
  for (int ti = 0; ti < kNTypen; ++ti) {
    Form& F = K.form[ti];
    const long mind = std::max<long>(20, F.seiten / 100);
    std::vector<std::pair<Familie*, Familie*>> paare;
    for (size_t q = 0; q < F.pos.size() && q < K.dsum[ti].size() / 2; ++q) {
      const double nd = K.dsum[ti][2 * q];
      if (nd < (double)mind || F.pos[q]->doppel) continue;
      Familie* f = F.pos[q];
      std::string key = f->skelett + "\x7f";
      while (F.fam_t.Find(key.data(), key.size()) >= 0) key.push_back('\x7f');
      Familie& d = *FamHol(F, key);
      d.pos = K.dsum[ti][2 * q + 1];
      d.n = (long)nd;
      d.seiten = (long)nd;
      d.doppel = true;
      d.basis = f;
      for (Variante* v : f->var) d.var.push_back(v);   // d is new: a copy without vector::operator=
      paare.push_back(std::make_pair(f, &d));
    }
    K.dsum[ti].clear();   // the next round counts again
    if (!paare.empty()) {
      std::vector<Familie*>& alle = F.pos;   // rebuilt in place
      for (auto& pr : paare) alle.push_back(pr.second);
      qsort(alle.data(), alle.size(), sizeof(alle[0]), FamVgl);
      for (size_t i = 0; i < alle.size(); ++i) alle[i]->ord = (int)i;
      for (Familie* f : alle) if (!f->doppel) for (Variante* v : f->var) v->fam = f->ord;
    }
    std::vector<std::vector<int>>(F.pos.size()).swap(K.dopp[ti]);
    for (Familie* f : F.pos) if (f->doppel) K.dopp[ti][(size_t)f->basis->ord].push_back(f->ord);
  }
}
// the passes over the pages; schreiben = false: pass 2 writes nothing (the
// first round, which collects)
ENC_ONLY inline bool Paesse(Kodierer& K, const char* pfad, bool schreiben) {
  // 0 County titles, 1 forms, 3 dry run of the County pages (records), 2 write
  KreislisteLeer();
  kreisname.clear();
  kreisglieder.clear();
  KreisArtLeer();
  nachbar_ang.clear();
  // three dry passes: further positions per family
  static const int kPaesse[] = {0, 1, 4, 4, 4, 3, 2};
  int vorher = 0;
  for (int pass : kPaesse) {
    K.pass = pass;
    LueckeLeer();
    K.page_index = -1;
    if (pass == 2) {   // usable counties: the ones with a record
      K.kreis_ok.Clear();
      bool neu;
      for (int q = 0; q < kStaaten; ++q) for (long kp : zahl::kreisliste[(size_t)q]) K.kreis_ok.Neu(kp, &neu);
    }
    FILE* in = fopen(pfad, "rb");
    if (!in) { kd = nullptr; return false; }
    const std::string tmp = std::string(pfad, strlen(pfad)) + ".cb";
    FILE* out = fopen(pass == 2 && schreiben ? tmp.c_str() : "/dev/null", "wb");
    if (!out) { fclose(in); kd = nullptr; return false; }
    if (pass == 4 && vorher == 1) BaueFormen(K);   // the forms, then the dry rounds
    if (pass != 1 && vorher == 4) BaueDoppel(K);   // after every dry round
    const int ok = page_stream_each(in, out, SeiteKodCb);
    vorher = pass;
    fclose(in);
    if (fclose(out) != 0 || !ok) { kd = nullptr; return false; }
    if (pass == 2 && schreiben && rename(tmp.c_str(), pfad) != 0) { kd = nullptr; return false; }
  }
  return true;
}
// -e, behind reorder(): pages -> stubs, block -> side file
ENC_ONLY inline bool Zerlege(const char* pfad, const char* seitenpfad) {
  zahl::Run::ZweigeSetzen();   // the encoder branches of the number coder
  // a first round over all passes collects what the written records need from
  // later records (the centroids of the counties; the counties of the records
  // of earlier types)
  anl4_tab.clear();
  zwei_runde = 0;
  {
    std::unique_ptr<Kodierer> K0(new Kodierer());
    kd = K0.get();
    if (!Paesse(*K0, pfad, false)) return false;
  }
  zwei_runde = 1;
  std::unique_ptr<Kodierer> KP(new Kodierer());
  Kodierer& K = *KP;
  kd = &K;
  if (!Paesse(K, pfad, true)) return false;
  // head (the forms) + records + '\n'
  std::string blk(kMagic, 4);
  size_t nvar = 0;
  std::string pseudo;   // the texts, for the pseudo article
  for (int ti = 0; ti < kNTypen; ++ti) {
    const Form& F = K.form[ti];
    B100(&blk, (unsigned long)F.pos.size(), 2);
    for (const Familie* f : F.pos) {
      B100(&blk, (unsigned long)f->var.size(), 2);
      for (const Variante* v : f->var) {
        // text to the pseudo article; here: the labels and the years of the
        // rows of the population table
        PseudoText(v->text, &pseudo);
        if (v->lab.size() > 99) { kd = nullptr; return false; }
        B100(&blk, (unsigned long)v->lab.size(), 1);
        for (size_t k = 0; k < v->lab.size(); ++k) {
          B100(&blk, v->lab[k], 1);
          if (v->lab[k] == zahl::F_PV) B100(&blk, (unsigned long)PvJahr(v->text, (int)k), 2);
        }
        ++nvar;
      }
    }
  }
  // the word list of the stripped titles: states, then type words (the model
  // side takes their WRT form from the coded article)
  for (const char c : kStaatFeld) pseudo.push_back(c ? c : '\x0b');
  for (const char* c = kTypwortFeld + 1; c < kTypwortFeld + sizeof kTypwortFeld; ++c) pseudo.push_back(*c ? *c : '\x0b');
  pseudo.append(kPseudoWoerter, sizeof kPseudoWoerter - 1);
  {
    const std::string pp = std::string(seitenpfad, strlen(seitenpfad)) + "p";
    FILE* g = fopen(pp.c_str(), "wb");
    if (!g || fwrite(pseudo.data(), 1, pseudo.size(), g) != pseudo.size() || fclose(g) != 0) { kd = nullptr; return false; }
  }
  const size_t head = blk.size();
  {
    long r0 = -1, r1 = -1;
    for (int ti = 0; ti < kNTypen; ++ti)
      for (long pg : K.lauf_seiten[ti]) {
        if (r0 < 0 || pg < r0) r0 = pg;
        if (pg > r1) r1 = pg;
      }
    if (r0 < 0) r0 = r1 = 0;
    std::vector<unsigned char> sym((size_t)(r1 - r0 + 1), (unsigned char)(0x80 + kNTypen));
    for (int ti = 0; ti < kNTypen; ++ti) for (long pg : K.lauf_seiten[ti]) sym[(size_t)(pg - r0)] = (unsigned char)(0x80 + ti);
    if (K.nrec == 0) sym.clear();
    B100(&blk, (unsigned long)r0, 3);
    B100(&blk, (unsigned long)sym.size(), 3);
    blk.append((const char*)sym.data(), sym.size());
  }
  size_t basis[kNTypen];
  for (int ti = 0; ti < kNTypen; ++ti) { basis[ti] = blk.size(); blk += K.block[ti]; }
  blk.push_back('\n');
  for (size_t i = 0; i + 1 < blk.size(); ++i) {
    const unsigned char c = (unsigned char)blk[i];
    if (!ImAlphabet(c) || c == '\n') {
      __builtin_unreachable();
      kd = nullptr;
      return false;
    }
  }
  FILE* f = fopen(seitenpfad, "wb");
  if (!f || fwrite(blk.data(), 1, blk.size(), f) != blk.size() || fclose(f) != 0) { kd = nullptr; return false; }
  kd = nullptr;
  return true;
}

// ------------------------------------------------------------ the decoder --
struct Satz {
  long kreis = -1;   // county (page index of its County record), -1 none
  bool kreis_zusatz = false;   // the county is the extra value behind the values of the units
  int titel = 0;   // type word code of the title
  int typ = -1;
  int staat = -1;
  long page_index = 0;
  std::vector<std::pair<int, int>> inst;   // (position, variant) in order
  std::vector<unsigned char> bit;
  std::vector<std::string> slot_values;
};
struct VarD { std::string text; std::vector<unsigned char> lab; std::vector<int> jahr; };
struct Leser {
  // type -> position -> variants; the types as a fixed array (no vector<vector<vector<VarD>>> instances)
  std::vector<std::vector<VarD>> form[kNTypen];
  zahl::Typmodell tm[kNTypen];
  std::vector<Satz> satz;
  struct Ord { void** p = nullptr; ~Ord() { free(p); } } ord;   // the records by page (sorted pointers)
  size_t naechst = 0;
  long page_index = -1;
  bool fehler = false;
  std::string ob;
  std::string tg;                        // the page with its title restored
  std::vector<std::string> text;
};
inline Leser* ls = nullptr;

inline bool LiesSaetze(const unsigned char* p, const unsigned char* e, Leser* L);
inline bool LiesBlock(const std::string& b, Leser* L) {
  const unsigned char* p = (const unsigned char*)b.data();
  const unsigned char* e = p + b.size();
  // the decoded pseudo article (texts) in front of the census line
  const size_t pe = b.find("\x0b\x0b");
  if (pe == std::string::npos) return false;
  std::vector<std::string> texte;
  if (!PseudoTexte(b.substr(0, pe + 2), &texte)) return false;
  size_t tn = 0;
  p += pe + 2;
  if ((size_t)(e - p) < 8 || memcmp(p, kMagic, 4) != 0) return false;
  p += 4;
  for (int ti = 0; ti < kNTypen; ++ti) {
    const long np = LiesB100(p, e, 2);
    if (np < 0) return false;
    std::vector<std::vector<VarD>>((size_t)np).swap(L->form[ti]);   // empty before: as resize(np)
    for (long j = 0; j < np; ++j) {
      const long nv = LiesB100(p, e, 2);
      if (nv < 0) return false;
      for (long k = 0; k < nv; ++k) {
        VarD& vd = L->form[ti][(size_t)j].emplace_back();
        if (tn >= texte.size()) return false;
        vd.text = texte[tn++];
        const long nl = LiesB100(p, e, 1);
        if (nl != Wertslots(vd.text)) return false;
        for (long q = 0; q < nl; ++q) {
          const long l = LiesB100(p, e, 1);
          if (l < 0 || l >= zahl::kFelder) return false;
          vd.lab.push_back((unsigned char)l);
          if (l == zahl::F_PV && LiesB100(p, e, 2) < 0) return false;   // the year, as in the text
        }
        {   // the value slot of F_KREIS travels as a number slot: its placeholder back
          size_t q = 0;
          for (char& c : vd.text)
            if (IstWert((unsigned char)c)) {
              if (q < vd.lab.size() && vd.lab[q] == zahl::F_KREIS) c = (char)kKreis;
              if (q < vd.lab.size() && vd.lab[q] == zahl::F_GLIED) c = (char)kGlied;
              if (q < vd.lab.size() && vd.lab[q] == zahl::F_NACHBAR) c = (char)kNachbar;
              ++q;
            }
        }
      }
    }
  }
  if (tn + (size_t)kStaaten + (size_t)kTypworte - 1 + (size_t)kKontextWorte + 1 + (size_t)kRechWorte + (size_t)kK1Worte != texte.size())
    return false;   // + the word list
  return LiesSaetze(p, e, L);
}
// the records of the census line (behind its head), with the forms in L
inline bool LiesSaetze(const unsigned char* p, const unsigned char* e, Leser* L) {
  long seite_t[kNTypen] = {-1, -1, -1, -1, -1, -1, -1, -1};
  anl4_tab.clear();
  LueckeLeer();
  KreislisteLeer();
  KreisArtLeer();
  nachbar_ang.clear();
  std::vector<std::pair<int, long>> run_list;   // (type, page) in block order
  {
    const long r0 = LiesB100(p, e, 3), rn = LiesB100(p, e, 3);
    if (r0 < 0 || rn < 0 || rn > e - p) return false;
    std::vector<long> ls[kNTypen];
    for (long i = 0; i < rn; ++i) {
      const int v = (int)(unsigned char)*p++ - 0x80;
      if (v < 0 || v > kNTypen) return false;
      if (v < kNTypen) ls[v].push_back(r0 + i);
    }
    for (int ti = 0; ti < kNTypen; ++ti) for (long pg : ls[ti]) run_list.push_back(std::make_pair(ti, pg));
  }
  (void)seite_t;
  for (size_t lk = 0; lk < run_list.size(); ++lk) {
    // The model side reads the whole census line inside one stream byte
    // (WerteAusZeile, ~2 s, ~100 MB of records) while the PPMd pool clamp
    // polls only on model bytes: one tick per record lets it follow.  The pool
    // size does not change the model output.
    ppmd_page::dyn_tick();
    Satz s;
    s.typ = run_list[lk].first;
    s.page_index = run_list[lk].second;
    s.staat = (int)LiesB100(p, e, 1);
    s.titel = (int)LiesB100(p, e, 1);
    if (s.titel < 0 || s.titel >= kTypworte) return false;
    if (s.staat < 0 || s.staat >= kStaaten) return false;
    if (kTypen[s.typ] == 'k') zahl::kreisliste[(size_t)s.staat].push_back(s.page_index);
    if (kTypen[s.typ] == 'k') KreisArtSetze(s.page_index, s.titel, s.staat);
    const auto& F = L->form[(size_t)s.typ];
    int nw = 0;
    for (size_t j = 0; j < F.size(); ++j) {
      const long f = LiesB100(p, e, 1);
      if (f < 0) return false;
      if (f == 0) continue;
      if (f == 99) {
        const long c = LiesB100(p, e, 2);
        if (c < 0) return false;
        for (long q = 0; q < c; ++q) {
          const long v = LiesB100(p, e, 1);
          if (v < 0 || v >= (long)F[j].size()) return false;
          s.inst.push_back(std::make_pair((int)j, (int)v));
        }
      } else {
        if (f - 1 >= (long)F[j].size()) return false;
        s.inst.push_back(std::make_pair((int)j, (int)(f - 1)));
      }
    }
    std::vector<Slot> slots;
    for (const auto& iv : s.inst) {
      const VarD& vd = F[(size_t)iv.first][(size_t)iv.second];
      for (size_t k = 0; k < vd.lab.size(); ++k) {
        Slot sl;
        sl.lab = vd.lab[k];
        sl.jahr = k < vd.jahr.size() ? vd.jahr[k] : PvJahr(vd.text, (int)k);   // the model side has no texts
        slots.push_back(sl);
      }
    }
    s.kreis_zusatz = KreisZusatz(kTypen[s.typ], &slots);
    nw = (int)slots.size();
    const size_t nb = s.inst.size() + 1;
    for (size_t g = 0; g < nb; g += 6) {
      const long v = LiesB100(p, e, 1);
      if (v < 0 || v > 63) return false;
      for (size_t j = 0; j < 6 && g + j < nb; ++j) s.bit.push_back((unsigned char)((v >> j) & 1));
    }
    LueckeXor(s.typ, (int)F.size(), s.inst, &s.bit, false);
    std::vector<std::string> tok;
    for (int i = 0; i < nw; ++i) {
      const unsigned char* q = p;
      while (q < e && *q != '\t') ++q;
      if (q >= e) return false;
      const std::string tk((const char*)p, (size_t)(q - p));
      tok.push_back(tk);
      p = q + 1;
    }
    Anl4Setzen(s.typ, s.page_index, s.staat);
#if PORT_CMX_MEAS
    zahl::g_cur_page = s.page_index;   // research V2 (censblock_zahl.h)
#endif
    if (!Zahlen(L->tm[s.typ], kTypen[s.typ] == 'k', s.staat, slots, &s.slot_values, &tok, false)) return false;
    for (size_t q = 0; q < slots.size() && q < s.slot_values.size(); ++q)
      if (slots[q].lab == zahl::F_KREIS) { s.kreis = strtol(s.slot_values[q].c_str(), nullptr, 10); break; }
    Anl4Merken(s.typ, s.page_index, s.staat, kTypen[s.typ] == 'k' ? s.page_index : s.kreis);
    if (p >= e || *p != 0x0B) return false;
    ++p;
    L->satz.push_back(std::move(s));
  }
  // the neighbours of the County records (all County records known now): ranks -> page indices, in record order
  for (Satz& q : L->satz) {
    if (kTypen[q.typ] != 'k') continue;
    std::vector<Slot> sl;
    for (const auto& iv : q.inst)
      for (unsigned char l : L->form[(size_t)q.typ][(size_t)iv.first][(size_t)iv.second].lab) {
        Slot x;
        x.lab = l;
        sl.push_back(x);
      }
    if (!NachbarRaenge(q.page_index, q.staat, sl, &q.slot_values, false)) return false;
  }
  L->ord.p = ZeigerSortiert(L->satz.data(), L->satz.size(), sizeof(Satz), [](const void* x, const void* y) {
    const long a = (*(const Satz* const*)x)->page_index, b = (*(const Satz* const*)y)->page_index;
    return (a > b) - (a < b);   // pages are unique
  });
  return true;
}

// the values of the census records per page: label byte, value, 0x00, ...
inline LMap<std::string> census_werte;
// the census line is read on the model side: decode its records once and keep
// the values; the forms move into the reader (the automaton reads them no more)
inline bool WerteAusZeile(const std::string& saetze, std::vector<std::vector<VarD>>* form) {   // form[kNTypen]
  census_werte.clear();
  std::unique_ptr<Leser> LP(new Leser());
  for (int ti = 0; ti < kNTypen; ++ti) LP->form[ti].swap(form[ti]);
  const unsigned char* p = (const unsigned char*)saetze.data();
  if (!LiesSaetze(p, p + saetze.size(), LP.get())) return false;
  for (const Satz& q : LP->satz) {
    ppmd_page::dyn_tick();   // as in LiesSaetze: the clamp follows the value table
    std::vector<unsigned char> lab;
    for (const auto& iv : q.inst)
      for (unsigned char l : LP->form[(size_t)q.typ][(size_t)iv.first][(size_t)iv.second].lab) lab.push_back(l);
    if (q.kreis_zusatz) lab.push_back((unsigned char)zahl::F_KREIS);
    std::string w;
    w.push_back(kTypen[q.typ]);   // the census type first
    for (size_t i = 0; i < lab.size() && i < q.slot_values.size(); ++i) {
      w.push_back((char)(lab[i] + 1));
      w += q.slot_values[i];
      w.push_back('\0');
    }
    census_werte.Setze(q.page_index, w);   // the pages of the records are unique
  }
  return true;
}
// the first value of field lab on page seite, or nullptr
inline const char* CensWert(long page_index, int lab, size_t* n) {
  const std::string* wp = census_werte.Find(page_index);
  if (!wp) return nullptr;
  const std::string& w = *wp;
  for (size_t i = 1; i < w.size();) {
    const size_t e = w.find('\0', i + 1);
    if (e == std::string::npos) return nullptr;
    if ((unsigned char)w[i] == lab + 1) { *n = e - i - 1; return w.data() + i + 1; }
    i = e + 1;
  }
  return nullptr;
}
// the census type of page seite (kTypen), or 0
inline char CensTyp(long page_index) {
  const std::string* w = census_werte.Find(page_index);
  return !w || w->empty() ? 0 : (*w)[0];
}
inline int SeiteDek(const char* p, long n, long* outn, const char** ob) {
  Leser& L = *ls;
  ++L.page_index;
  if (L.naechst >= L.satz.size() || ((const Satz*)L.ord.p[L.naechst])->page_index != L.page_index) return 0;
  const Satz& S = *(const Satz*)L.ord.p[L.naechst++];
  kreis_akt = S.kreis;
  {
    static const std::vector<std::pair<std::string, long>> leer;
    glied = nullptr;
    glied_ptr = 0;
    if (kTypen[S.typ] == 'k') {
      glied = kreisglieder.Find(S.page_index);
      if (!glied) glied = &leer;
    }
  }
  // the title back: "<name>" + type word + ", " + state
  {
    const long a = page_find(p, n, 0, "<title>", 7);
    const long b = a < 0 ? -1 : page_find(p, n, a + 7, "</title>", 8);
    if (b < 0) { L.fehler = true; return -1; }
    std::string& g = L.tg;
    g.assign(p, (size_t)b);
    g += Nth(kTypwortFeld, S.titel);
    g += ", ";
    g += Nth(kStaatFeld, S.staat);
    g.append(p + b, (size_t)(n - b));
    p = g.data();
    n = (long)g.size();
  }
  Seite s;
  if (!Teile(p, n, &s) || s.staat != S.staat) { L.fehler = true; return -1; }
  const Teile2 t = TitelTeile(s);
  const auto& F = L.form[(size_t)S.typ];
  std::string& o = L.ob;
  o.assign(p, (size_t)s.x0);
  size_t wi = 0;
  long i = s.x0;
  for (size_t g = 0; g <= S.inst.size(); ++g) {
    if (S.bit[g]) {   // deviation text up to kRun
      long j = i;
      while (j < s.x1 && (unsigned char)p[j] != kRun) ++j;
      if (j >= s.x1) { L.fehler = true; return -1; }
      o.append(p + i, (size_t)(j - i));
      i = j + 1;
    }
    if (g == S.inst.size()) break;
    const std::string& key = F[(size_t)S.inst[g].first][(size_t)S.inst[g].second].text;
    const int nts = Textslots(key);
    L.text.clear();
    for (int k = 0; k < nts; ++k) {
      long j = i;
      while (j < s.x1 && (unsigned char)p[j] != kEnde) ++j;
      if (j >= s.x1) { L.fehler = true; return -1; }
      const std::string tx(p + i, (size_t)(j - i));
      L.text.push_back(tx);
      i = j + 1;
    }
    if (!Expandiere(key, S.slot_values, &wi, L.text, t, &o)) { L.fehler = true; return -1; }
  }
  if (i != s.x1 || wi + (S.kreis_zusatz ? 1 : 0) != S.slot_values.size()) { L.fehler = true; return -1; }
  o.append(p + s.x1, (size_t)(n - s.x1));
  *outn = (long)o.size();
  *ob = o.data();
  return 1;
}

// decoder, behind phda9_resto(): stubs -> pages
// pre-pass of Setze: the stub titles of the County pages with a record (their
// short titles are the county names of F_KREIS)
inline long kreis_zaehler = -1;
inline const zahl::LTab* kreis_seiten = nullptr;
inline int SeiteKreisname(const char* p, long n, long* outn, const char** ob) {
  (void)outn;
  (void)ob;
  ++kreis_zaehler;
  if (kreis_seiten->Find(kreis_zaehler) < 0) return 0;
  const long a = page_find(p, n, 0, "<title>", 7);
  const long b = a < 0 ? -1 : page_find(p, n, a + 7, "</title>", 8);
  if (b < 0) return -1;
  kreisname.Setze(kreis_zaehler, std::string(p + a + 7, (size_t)(b - a - 7)));
  return 0;
}
inline bool Setze(const char* pfad, const char* seitenpfad) {
  std::unique_ptr<Leser> LP(new Leser());
  Leser& L = *LP;
  {
    FILE* bf = fopen(seitenpfad, "rb");
    if (!bf) return false;
    std::string b;
    char tb[1 << 16];
    size_t g;
    while ((g = fread(tb, 1, sizeof tb, bf)) > 0) b.append(tb, g);
    fclose(bf);
    if (!LiesBlock(b, &L)) {
      msg_note(404);
      return false;
    }
  }
  {
    zahl::LTab ks;
    bool neu;
    for (const Satz& q : L.satz) ks.Neu(q.page_index, &neu);   // the titles of all record pages (members too)
    kreisname.clear();
    kreis_zaehler = -1;
    kreis_seiten = &ks;
    FILE* in0 = fopen(pfad, "rb");
    FILE* nul = fopen("/dev/null", "wb");
    const int ok0 = in0 && nul ? page_stream_each(in0, nul, SeiteKreisname) : 0;
    if (in0) fclose(in0);
    if (nul) fclose(nul);
    kreis_seiten = nullptr;
    if (!ok0) return false;
    kreisglieder.clear();
    for (const Satz& q : L.satz) {   // members: places carrying the county, in front of its page
      if (kTypen[q.typ] == 'k' || q.kreis < 0 || q.page_index >= q.kreis) continue;
      if (const std::string* nm = kreisname.Find(q.page_index))
        kreisglieder.Ref(q.kreis).push_back(std::make_pair(*nm, q.page_index));
    }
    for (auto& g : kreisglieder.v) SortiereGlieder(g);
  }
  ls = &L;
  FILE* in = fopen(pfad, "rb");
  const std::string tmp = std::string(pfad, strlen(pfad)) + ".cb";
  FILE* out = fopen(tmp.c_str(), "wb");
  if (!in || !out) { ls = nullptr; return false; }
  const int ok = page_stream_each(in, out, SeiteDek);
  fclose(in);
  const bool zu = fclose(out) == 0;
  ls = nullptr;
  if (!ok || !zu || L.fehler || L.naechst != L.satz.size()) {
    __builtin_unreachable()
;
    return false;
  }
  remove(seitenpfad);
  return rename(tmp.c_str(), pfad) == 0;
}

}  // namespace census_block
#endif
