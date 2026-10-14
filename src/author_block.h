#ifndef AUTHOR_BLOCK_H_
#define AUTHOR_BLOCK_H_

#include "metablock.h"
#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include "metablock.h"
#include "ts_progress.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace author_block {

static const unsigned char kAnker[8] =
    { 0x20, 0x20, 0x4c, 0x2f, 0xdf, 0x98, 0x4e, 0x0a };
// `  <page>\n` post-WRT - the page OPEN of the old form.  It is what ends the
// metablock there; it does not occur inside the block.
static const unsigned char kSeiteAuf[7] =
    { 0x20, 0x20, 0x4c, 0xdf, 0x98, 0x4e, 0x0a };
// `      <comment>` post-WRT - the deterministic injection point of the old
// form.  Full count 190.325, never inside article text.
static const unsigned char kKomAuf[10] =
    { 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x4c, 0xdf, 0x93, 0x4e };
// `<username>` / `<ip>` post-WRT - the author lines of the old form's block.
static const unsigned char kUser[3] = { 0xdf, 0x87, 0x4e };
static const unsigned char kIp[3]   = { 0x69, 0x70, 0x4e };

static const unsigned char kNL  = 0x0a;
static const unsigned char kTab = 0x09;

struct Stats {
  unsigned long long block_bytes = 0;   // metablock bytes absorbed
  unsigned long long saetze = 0;        // author records parsed
  unsigned long long seiten = 0;        // page boundaries seen
  unsigned long long stellen = 0;       // injection points hit
  unsigned long long bytes = 0;         // reinjected bytes
  unsigned long long autor_bytes = 0;   // of them AUTHOR bytes (no separator)
  unsigned long long ohne_satz = 0;     // injection point without a record
};

class Aut {
 public:
  void Observe(unsigned char b) {
    have_block_ = false;
#if AUX_BUILD
    (void)b; return;
#else
    const unsigned char prev = prev_;
    prev_ = b;

    if (phase_ == kBlock) { ObserveBlock(b, prev); return; }

    // --- page boundary: the anchor closes a page (both forms) --------------
    Push(b);
    if (Tail(8, kAnker, 8)) { ++seite_; ++st_.seiten; return; }

    if (prev == kNL && b == (unsigned char)3) Emit();
#endif
  }

  // The block for the byte that was just observed, or 0.
  int TakeBlock(const unsigned char** p) {
    if (!have_block_) return 0;
    have_block_ = false;
    *p = block_.data();
    return (int)block_.size();
  }

  void Finish() {}

  // Pages closed since the metablock, i.e. the index of the page being read.
  long long Seite() const { return seite_; }

  const unsigned char* Feld(long long page_index, int f, int* n) const {
    return metablock::mk.Feld(page_index, f, n);
  }
  // How many pages the block carries - the self-report counterpart of
  // `MetaTable::Pages()`.
  size_t Seiten() const {
    return metablock::mk.Seiten();
  }

  // Self-report of the author channel, left out of the package form.
  void Report(const char* side) const {
    (void)side;
  }

  const Stats& GetStats() const { return st_; }

 private:
  enum Phase { kBlock = 0, kBody = 1 };

  // ---- ring of the last 16 bytes, for the fixed marks --------------------
  void Push(unsigned char b) {
    for (int i = 0; i + 1 < (int)sizeof(tail_); ++i) tail_[i] = tail_[i + 1];
    tail_[sizeof(tail_) - 1] = b;
    ++gesehen_;
  }
  bool Tail(int n, const unsigned char* pat, int pl) const {
    if (gesehen_ < (unsigned long long)n) return false;
    return memcmp(tail_ + sizeof(tail_) - pl, pat, (size_t)pl) == 0;
  }

  // ---- phase 1: the metablock in front of the pages ----------------------
  void ObserveBlock(unsigned char b, unsigned char prev) {
    (void)b;
    (void)prev;
    if (metablock::mk.NachlaufEndet()) BlockFertig();
    return;
  }

  void BlockFertig() {
    ende_der_einheit_ = metablock::mk.Bis();
    block_fertig_bei_ = metablock::mk.Pos();
    st_.saetze = (unsigned long long)metablock::mk.Seiten();
    st_.block_bytes = metablock::mk.GetStats().block_bytes;
    if (st_.saetze == 0) Fail("no author record in the block");
    phase_ = kBody;
    seite_ = (long long)0;
    memset(tail_, 0, sizeof(tail_));
    gesehen_ = 0;
  }

  static void Fail(const char* what) {
    fprintf(stderr, "\nauthor: %s (metablock form not as expected)\n", what);
    fflush(stderr);
    exit(1);
  }

  // ---- the injection ------------------------------------------------------
  void Emit() {
    ++st_.stellen;
    long long k = seite_;
    const unsigned char* src = nullptr;
    unsigned int len = 0;
    if (k < 0 || (size_t)k >= metablock::mk.Seiten()) { ++st_.ohne_satz; return; }
    { int ln = 0; src = metablock::mk.AutorVon((size_t)k, &ln);
      len = (unsigned int)ln; }
    block_.clear();
    block_.reserve((size_t)len + 1);
    block_.insert(block_.end(), src, src + len);
    st_.autor_bytes += (unsigned long long)len;
    block_.push_back(kTab);      // the separator, see the head comment
    st_.bytes += block_.size();
    have_block_ = true;
  }

  Phase phase_ = kBlock;
  bool have_block_ = false;
  bool block_opened_ = false;
  unsigned long long block_fertig_bei_ = 0, ende_der_einheit_ = 0;
  unsigned char prev_ = 0;
  unsigned char tail_[16] = {0};
  unsigned long long gesehen_ = 0;
  long long seite_ = 0;
  std::vector<unsigned char> block_;    // the block to reinject
  Stats st_;
};

inline Aut aut;

}

#endif
