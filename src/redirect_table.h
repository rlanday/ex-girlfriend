#ifndef REDIRECT_TABLE_H_
#define REDIRECT_TABLE_H_


#include <string.h>

#include "author_block.h"
static inline int redirect_block(long long page) {
  const long long p = page + (long long)0;
  return p >= 0 && p < (long long)71148;
}

// ---- plain text side (the page-form pass) -------------------------------
static const char kRcGross[] = "#REDIRECT";        // -> nothing
static const char kRcKlein[] = "#redirect";        // -> `r`
static const char kRcErst[]  = "#Redirect";        // -> `R`
static const char kRcSoft[]  = "{softredirect";    // -> `s` (phda9 wrote `{{` as `{`)

// Bytes a text may not begin with after `#REDIRECT` was dropped: the code
// bytes, the escape, and the line feed - an EMPTY text (the seven self-closing
// pages) begins with the line feed of the comment line, and the consumers must
// keep telling the two apart (0 such pages on enwik9, counted).
static inline int redirect_codebyte(unsigned char c) {
  return c == '#' || c == 'r' || c == 'R' || c == 's' ||
         c == 0x01 || c == 0x0A;
}

// Only a mark that ends as a WORD is coded: the WRT writes `redirect` as its
// dictionary code only when no letter or digit follows, and the consumers
// recognised the mark by exactly that code.  One front-block page carries
// `#redirecting[[...]]` (stream page 21.873): coded it would become a
// redirect the consumers never saw before.  It stays verbatim (it begins
// with `#`), so every consumer keeps its old answer.
static inline int redirect_word_end(const char* body, size_t bodyl) {
  if (bodyl == 9) return 1;
  const unsigned char c = (unsigned char)body[9];
  return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9'));
}

// ENCODER.  `body`/`bodyl` is the page text; returns the byte to write in
// front of it (0 = none) and advances `*skip` over the bytes the code
// replaces.  `zaehler[k]` counts the classes (0 gross, 1 klein, 2 erst,
// 3 soft, 4 verbatim `#`, 5 escape, 6 empty).
static inline unsigned char redirect_encode(const char* body, size_t bodyl,
                                           size_t* skip, long long* zaehler) {
  *skip = 0;
  if (bodyl == 0) { ++zaehler[6]; return 0; }
  if (bodyl > 9 && memcmp(body, kRcGross, 9) == 0
      && !redirect_codebyte((unsigned char)body[9])
      && redirect_word_end(body, bodyl)) {
    *skip = 9; ++zaehler[0]; return 0;
  }
  if (bodyl >= 9 && memcmp(body, kRcKlein, 9) == 0
      && redirect_word_end(body, bodyl)) {
    *skip = 9; ++zaehler[1]; return 'r';
  }
  if (bodyl >= 9 && memcmp(body, kRcErst, 9) == 0
      && redirect_word_end(body, bodyl)) {
    *skip = 9; ++zaehler[2]; return 'R';
  }
  if (bodyl >= 13 && memcmp(body, kRcSoft, 13) == 0) {
    *skip = 13; ++zaehler[3]; return 's';
  }
  if (body[0] == '#') { ++zaehler[4]; return 0; }
  ++zaehler[5];
  return (unsigned char)0x01;
}

// DECODER.  Looks at the first byte of a non-empty text only; returns the
// plain text to put in front (0 = none) and how many bytes of the text to
// drop (the code byte or the escape).
static inline const char* redirect_decode(const char* body, size_t bodyl,
                                         size_t* drop, long long* zaehler) {
  *drop = 0;
  if (bodyl == 0) { ++zaehler[6]; return 0; }
  switch ((unsigned char)body[0]) {
    case '#': ++zaehler[4]; return 0;
    case 'r': *drop = 1; ++zaehler[1]; return kRcKlein;
    case 'R': *drop = 1; ++zaehler[2]; return kRcErst;
    case 's': *drop = 1; ++zaehler[3]; return kRcSoft;
    case 0x01: *drop = 1; ++zaehler[5]; return 0;
    default: ++zaehler[0]; return kRcGross;
  }
}

#include <stdio.h>
#include <stdlib.h>

#include "alias_context_set.h"   // stamped mark table, link bytes
#include "metablock.h"

namespace redirect_table {

enum { kGross = 0, kKlein, kErst, kSoft, kKeine, kEsc };

static const unsigned char kAltBild[4][4] = {
    {0x23, 0x07, 0xdf, 0x94},    // `#REDIRECT`       -> dropped
    {0x23, 0xdf, 0x94, 0x00},    // `#redirect`       -> `r` 0x72
    {0x23, 0x40, 0xdf, 0x94},    // `#Redirect`       -> `R` 0x40 0x72
    {0x50, 0xf6, 0xe6, 0xab}};   // `{softredirect`   -> `s` 0xac (dict word 45)
static const int kAltLen[4] = {4, 3, 4, 4};
static const int kNeuLen[4] = {0, 1, 2, 1};
static const unsigned char kAnker[8] = {0x20, 0x20, 0x4c, 0x2f,
                                        0xdf, 0x98, 0x4e, 0x0a};  // page anchor
static const unsigned char kMarkTab[3]
                                   [4] =
    { { 0x23, 0x07, 0xdf, 0x94 }, { 0x23, 0x40, 0xdf, 0x94 }, { 0x23, 0xdf, 0x94, 0x00 } };
static const int kMarkLen[3] =
    { 4, 4, 3 };

struct Rec {
  unsigned char marke = 0, ziel_a = 0, ziel_n = 0;
  unsigned char d = 0;
};

class Tab {
 public:
  // The running page (bed relative, as the consumers count pages).
  const Rec* Seite(long long page) const {
    return page == seite_ ? &cur_ : (page == fertig_seite_ ? &fertig_rec_ : nullptr);
  }
  // EVENT: the byte just fed decided a page (end of its window or of the
  // page) - the byte on which the consumers used to evaluate their own
  // windows.  Ziel() is that page's link target, valid until the next byte.
  bool Fertig() const { return fertig_; }
  int Ziel(const unsigned char** p) const {
    if (!fertig_ || fertig_rec_.ziel_n == 0) return 0;
    *p = w_ + fertig_rec_.ziel_a;
    return fertig_rec_.ziel_n;
  }
  // The bytes a consumer that copies text gets for the byte just fed: the
  // stream as it was before the codes (0 bytes while a code is still open).
  int Alt(const unsigned char** p) const { *p = alt_; return alt_n_; }

  void Byte(unsigned char b) {
    alt_[0] = b; alt_n_ = 1;
    fertig_ = false;
    if (zustand_ == kVor) {                  // only the metablock needs it
      mb_.Byte(b);
      if (mb_.Endet()) NeuSeite(0);          // page 0 starts behind the block
      return;
    }
    if (zustand_ == kAus) return;
    for (int i = 0; i < 7; ++i) tail_[i] = tail_[i + 1];
    tail_[7] = b;
    if (zustand_ == kText) Text(b);
    else if (zustand_ == kTitel && b == 0x0a) zustand_ = kText;
    if (memcmp(tail_, kAnker, 8) == 0) {
      if (zustand_ == kText) Abschluss();
      NeuSeite(seite_ + 1);
    }
  }

  void Schluss() { if (zustand_ == kText) Abschluss(); }

  void Report(const char* side) const {
    fprintf(stderr, "%s: anchor=%lld marker=%llu target=%llu "
            "upper=%llu lower=%llu first=%llu soft=%llu none=%llu esc=%llu "
            "tabmarker=%llu\n",
            side, seite_, n_marke_, n_ziel_, n_kl_[kGross], n_kl_[kKlein],
            n_kl_[kErst], n_kl_[kSoft], n_kl_[kKeine], n_kl_[kEsc], n_tab_);
  }

 private:
  enum { kVor, kTitel, kText, kRest, kAus };
  metablock::Grenze mb_;
  int zustand_ = kVor;
  long long seite_ = -1, fertig_seite_ = -1;
  Rec cur_, fertig_rec_;
  bool fertig_ = false;
  unsigned char tail_[8] = {0};
  unsigned char w_[200];
  int wn_ = 0;
  int kl_ = -1;                 // class of the running page, -1 = undecided
  bool offen_ = false;          // `0x40` seen: `R` or not, the next byte says
  unsigned char alt_[8];
  int alt_n_ = 1;
  unsigned long long n_marke_ = 0, n_ziel_ = 0, n_kl_[6] = {0};
  unsigned long long n_tab_ = 0;

  void NeuSeite(long long s) {
    seite_ = s; wn_ = 0; kl_ = -1;
    offen_ = false;
    cur_ = Rec();
    zustand_ = ImBlock() ? kTitel : kAus;
  }
  bool ImBlock() const { return redirect_block(seite_) != 0; }
  int D() const { return (int)cur_.d; }
  void Setze(int k) {
    kl_ = k; ++n_kl_[k];
    if (k <= kErst) { cur_.marke = 1; cur_.d = (unsigned char)(kAltLen[k] - kNeuLen[k]); }
  }
  void Bild(int k, int extra, const unsigned char* e) {   // Alt() for a code
    alt_n_ = 0;
    for (int i = 0; i < kAltLen[k]; ++i) alt_[alt_n_++] = kAltBild[k][i];
    for (int i = 0; i < extra; ++i) alt_[alt_n_++] = e[i];
  }
  // The first bytes of a front-block text decide its class.
  void Klasse(unsigned char b) {
    if (offen_) {                                       // `0x40` before
      offen_ = false;
      if (b == 0x72) { Setze(kErst); Bild(kErst, 0, nullptr); }
      else { const unsigned char e[2] = {0x40, b}; Setze(kGross); Bild(kGross, 2, e); }
      return;
    }
    if (b == 0x23 || b == 0x0a) { kl_ = kKeine; ++n_kl_[kKeine]; return; }  // `#` / leer
    if (b == 0x01) { Setze(kEsc); alt_n_ = 0; return; }
    if (b == 0x72) { Setze(kKlein); Bild(kKlein, 0, nullptr); return; }
    if (b == 0xac) { Setze(kSoft); Bild(kSoft, 0, nullptr); return; }
    if (b == 0x40) { offen_ = true; alt_n_ = 0; return; }
    Setze(kGross); Bild(kGross, 1, &b);
  }
  void Text(unsigned char b) {
    if (wn_ < 200) w_[wn_++] = b;
    if (kl_ < 0 && ImBlock()) Klasse(b);
    else if (kl_ < 0) kl_ = kKeine;
    if (kl_ == kKeine && !cur_.marke && wn_ <= 4) {
      // the written mark: the stamped table, as the consumers read it
      for (int f = 0; f < 3; ++f)
        if (wn_ == kMarkLen[f] && memcmp(w_, kMarkTab[f], (size_t)wn_) == 0)
          { cur_.marke = 1; ++n_tab_; }
    }
    if (wn_ >= 200 - D()) Abschluss();
  }
  // Once per page, at the end of the window or of the page: the link target.
  void Abschluss() {
    zustand_ = kRest;
    ZielSuchen();
    fertig_ = true; fertig_seite_ = seite_; fertig_rec_ = cur_;
  }
  void ZielSuchen() {
    if (!cur_.marke) return;
    ++n_marke_;
    int ml = 0;
    if (kl_ >= 0 && kl_ <= kErst) ml = kNeuLen[kl_];
    else
    {
      for (int f = 0; f < 3; ++f)
        if (wn_ >= kMarkLen[f] && memcmp(w_, kMarkTab[f], (size_t)kMarkLen[f]) == 0) {
          ml = kMarkLen[f]; break;
        }
    }
    const int grenze = 12 - D() < wn_ ? 12 - D() : wn_;
    int lp = -1;
    for (int j = ml; j < grenze; ++j)
      if (w_[j] == 0x5b) { lp = j + (0 ? 2 : 1); break; }
    if (lp < 0) return;
    for (int j = lp; j < wn_; ++j)
      if (w_[j] == 0x5d) {
        if (j > lp) { cur_.ziel_a = (unsigned char)lp; cur_.ziel_n = (unsigned char)(j - lp); ++n_ziel_; }
        return;
      }
  }
};

inline Tab tab;

}

#endif
