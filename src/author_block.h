#ifndef AUTHOR_BLOCK_H_
#define AUTHOR_BLOCK_H_
#include "meld.h"
#include "u8_append.h"

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

// The redirect front block is the one of the page form and of the comment
// index zone (redirect_block, src/redirect_table.h).  Declared as well, for the
// include order in which redirect_table.h includes this header first.
#include "redirect_table.h"
static inline int redirect_block(long long page);
namespace comment_tag {
// the two transformer bytes of the day of the last comment line
inline unsigned char g_tf[2] = {0, 0};
// Aut::Emit made a comment-line block; the next author prompt block
// (tf_author_prompt::Beginn) is that one
inline bool g_offen = false;
// transformer row inside that block: 0 and 1 are the day bytes, 2 = none
inline int g_j = 2;
}  // namespace comment_tag

namespace author_block {

static const unsigned char kAnker[8] =
    { 0x20, 0x20, 0x4c, 0x2f, 0xdf, 0x98, 0x4e, 0x0a };

// `  </page>\n` after the WRT is the page anchor above (243,425 times in the
// page area).  The four marks below belong to a stream form with page-open
// lines (`  <page>\n`, `      <comment>`, `<username>` / `<ip>`, all after the
// WRT); this build has no reader for them.
static const unsigned char kSeiteAuf[7] =
    { 0x20, 0x20, 0x4c, 0xdf, 0x98, 0x4e, 0x0a };
static const unsigned char kKomAuf[10] =
    { 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x4c, 0xdf, 0x93, 0x4e };
static const unsigned char kUser[3] = { 0xdf, 0x87, 0x4e };
static const unsigned char kIp[3]   = { 0x69, 0x70, 0x4e };

static const unsigned char kNL  = 0x0a;
static const unsigned char kTab = 0x09;

struct Stats {
  unsigned long long saetze = 0;        // author records parsed
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

    // --- page boundary: the anchor closes a page ---------------------------
    Push(b);
    if (Tail(8, kAnker, 8)) {
      ++seite_;
      return;
    }

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
    if (st_.saetze == 0) Fail("E41");
    phase_ = kBody;
    seite_ = (long long)0;
    memset(tail_, 0, sizeof(tail_));
    gesehen_ = 0;
  }

  [[noreturn]] __attribute__((cold, noinline)) static void Fail(const char* what) {
    fprintf(stderr, "\n%s\n", what);
    fflush(stderr);
    exit(1);
  }

  // ---- the injection ------------------------------------------------------
  void Emit() {
    long long k = seite_;
    const unsigned char* src = nullptr;
    unsigned int len = 0;
    if (k < 0 || (size_t)k >= metablock::mk.Seiten()) {
      return;
    }
    { int ln = 0; src = metablock::mk.AutorVon((size_t)k, &ln);
      len = (unsigned int)ln; }
    block_.clear();
    block_.reserve((size_t)len + 1);
    {
      const unsigned int tg = metablock::mk.TagnrVon((size_t)k);
      const int rd = redirect_block(k - (long long)0);
      block_.push_back((unsigned char)(rd ? 0x80 + tg / 100u : 0x80));
      block_.push_back((unsigned char)0x80);
      comment_tag::g_tf[0] = (unsigned char)(0x80 + tg / 100u);
      comment_tag::g_tf[1] = (unsigned char)(0x80 + tg % 100u);
      comment_tag::g_offen = true;
    }
    u8_append::Anh(block_, src, src + len);
    block_.push_back(kTab);      // the separator, see the head comment
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

}  // namespace author_block

#endif
