#ifndef TF1_Z3_H_
#define TF1_Z3_H_

#include "../transformer.h"

#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "tf1_engine.h"   // separator sequence (SeparatorIndices/-Length)

// Link context ("z3"): for each article, text around earlier links to its
// title is fed to the transformer as a prefix behind the title.  The training
// side is zctx/z3_extract.py.
namespace tf_z3 {

// ---- grammar (stamped, derived from the dictionary) ----------------------
const unsigned char* TitleEnd();
int TitleEndLen();
constexpr unsigned char kLinkOpen = 0x31;
constexpr unsigned char kParagraph = 0x08;
constexpr int kMaxTitleTokens = 64;   // as zctx/zgrammar.py MAX_TITLE_TOKENS
constexpr int kMinTitleTokens = 3;    // as zctx/zgrammar.py MIN_TITLE_TOKENS

// ---- parameters (as z3_extract.py DEFAULTS) ------------------------------
struct Params {
  int budget = 256;        // prefix tokens per article, separator INCLUDED
  int chunk = 64;          // token budget of ONE chunk, mention included
  int max_mentions = 8;    // never more than this many most recent mentions
  int line_scan = 512;     // bound of the paragraph search per side
  int open_len = 1;        // length of the link opener (mode="link")
};

// ---- window: one contiguous piece of the token stream --------------------
// The coder does not hold the stream in RAM.  It reads a window of about
// 1.1 kB from a file per mention; the chunk logic computes in absolute
// positions and reads through `At()`.  In tool mode the window is the whole
// token buffer.
struct Window {
  const unsigned char* p = nullptr;   // p[0] sits at absolute position base
  long long base = 0;
  long long len = 0;
  unsigned char At(long long q) const { return p[q - base]; }
  bool Covers(long long a, long long b) const {
    return a >= base && b <= base + len;
  }
};

// ---- the mention index ---------------------------------------------------
// Key = FNV-1a over the first kIndexKeyTokens = 4 tokens behind the link
// opener.  The index returns a superset of the mentions; full title equality
// is checked afterwards, so the index strategy may differ from the training
// side and still produce identical prefixes.
//
// 4 and not 3 tokens: with 3 the chains get long (longest 8,782 on a 20 MB
// sample, against a mean of 7.6 with 4).  A title has at least 3 tokens; for
// titles of exactly 3 the lookup fans out over all 256 possible fourth tokens,
// and the union of the chains is exactly the set of matching anchors.
//
// No bounded ring per key: every clipped form loses mentions; only the
// unclipped chain matches the training side exactly.
constexpr int kIndexKeyTokens = 4;

// Memory layout.  The three tables live in anonymous
// mappings of their own, not in std::vector or malloc:
//   * entries_/next_ grow by mremap (the pages move, nothing is copied).  A
//     vector doubles by allocate-copy-free, so old and new buffer were resident
//     at the same time: at 2^23 anchors (full enwik9: 10.7 M) that was +96 MiB,
//     and the munmap of the old buffer put the spike into ru_maxrss.
//   * head_ is unmapped BEFORE it is rebuilt one size up; it is derived from
//     entries_ and carries nothing of its own.
//   * load factor 2 (chains of about two links) halves head_ (-32 MiB at the
//     end of enwik9).  The bucket count cannot change a result: Lookup() walks
//     the whole chain, keeps the entries with the full 32-bit tag (same tag =
//     same bucket under every mask) and returns them in insertion order.
// Own mappings and not malloc+realloc: glibc may serve a large request from
// free heap space below the brk top, and a realloc there copies again.
class MentionIndex {
 public:
  MentionIndex() = default;
  ~MentionIndex();
  MentionIndex(const MentionIndex&) = delete;
  MentionIndex& operator=(const MentionIndex&) = delete;
  void Reserve(size_t n_anchors);
  // Record one mention position (position of the link opener).
  // `key4` is the key of the four tokens behind it.
  void Add(uint64_t key4, long long anchor);
  // All candidates with a matching key, ascending by position.  Never in
  // table order: the result must not depend on the table layout.
  void Lookup(uint64_t key, std::vector<long long>* out) const;
  size_t Size() const { return n_; }
  size_t Buckets() const { return buckets_; }
  // Approximate RAM demand in bytes (mapped size).
  size_t Bytes() const {
    return cap_ * (sizeof(Entry) + sizeof(int)) + buckets_ * sizeof(int);
  }

 private:
  // Compact and lossless: tag and position as 32 bits each.
  //   * The tag is only a filter; every candidate is followed by the full
  //     title comparison.  A 32-bit collision costs one extra discarded
  //     candidate, not a different result.
  //   * The coder stream is < 4 GiB, so the position fits in 32 bits; `Add()`
  //     checks it.
  // 12 B per anchor instead of 20 (plus head table).
  struct Entry { uint32_t tag; uint32_t pos; };
  static constexpr size_t kLoad = 2;   // entries per bucket before head_ grows
  Entry* entries_ = nullptr;
  int* next_ = nullptr;        // chain link
  int* head_ = nullptr;        // bucket -> first entry, -1 = empty
  size_t n_ = 0;               // entries in use
  size_t cap_ = 0;             // capacity of entries_ and next_
  size_t buckets_ = 0;         // size of head_, a power of two
  size_t mask_ = 0;
  void GrowEntries(size_t cap);
  void Rebuild(size_t buckets);
};

uint64_t KeyHash(const unsigned char* seq, int len);
uint64_t KeyOf(const unsigned char* seq);             // length >= 4
uint64_t KeyOfShort(const unsigned char* title3, unsigned char fourth);

// ---- the four stages -----------------------------------------------------
// Title length of the article starting at `start`; -1 = none usable.
// `win` must cover [start, min(start+64+5, next_boundary)).
int TitleLength(const Window& win, long long start, long long next_boundary);

// Paragraph rule: the chunk around one mention.  [*out_a, *out_b) is the chunk.
// `win` must cover [m0-line_scan, min(m1+line_scan, limit)).
void ChunkForMention(const Window& win, long long m0, long long m1,
    const Params& p, long long limit, long long* out_a, long long* out_b);

// Selection of an article's chunks (budget, overlap, order).
// `mentions` are ascending positions of the link opener.
// `get_chunk(m0, m1, limit, &a, &b)` yields the chunk per mention; the caller
// provides the window needed for it.  Return value: number of chunks taken,
// in selection order (most recent first);
// -1 = the causality guard fired.
template <class ChunkFn>
int SelectChunks(const std::vector<long long>& mentions, int title_len,
    long long article_start, const Params& p, ChunkFn get_chunk,
    long long* taken_a, long long* taken_b) {
  const int sep_len = transformer::SeparatorLength();
  const long long room = (long long)p.budget - sep_len;
  if (room <= 0) return 0;

  // earlier mentions: those that end before the article starts
  long long n_earlier = 0;
  for (size_t i = 0; i < mentions.size(); ++i) {
    if (mentions[i] + p.open_len + title_len <= article_start) ++n_earlier;
    else break;                                    // sorted ascending
  }
  if (n_earlier == 0) return 0;

  int n_taken = 0;
  long long used = 0;
  const int cap = p.max_mentions < 64 ? p.max_mentions : 64;
  for (int k = 0; k < cap; ++k) {                  // most recent first, at most cap
    const long long idx = n_earlier - 1 - k;
    if (idx < 0) break;
    const long long m0 = mentions[(size_t)idx];
    const long long m1 = m0 + p.open_len + title_len;
    long long a = 0, b = 0;
    get_chunk(m0, m1, article_start, &a, &b);
    if (b > article_start) return -1;              // causality guard
    bool overlaps = false;
    for (int t = 0; t < n_taken; ++t) {
      if (a < taken_b[t] && taken_a[t] < b) { overlaps = true; break; }
    }
    if (overlaps) continue;
    const long long cost = (b - a) + (n_taken ? 2 : 0);
    if (used + cost > room) break;
    taken_a[n_taken] = a;
    taken_b[n_taken] = b;
    ++n_taken;
    used += cost;
  }
  return n_taken;
}

// The prefix of one article over a window covering the whole stream (tool
// mode).  `out` is appended to, not cleared.
int BuildPrefix(const Window& win, const std::vector<long long>& mentions,
    int title_len, long long article_start, const Params& p,
    std::vector<unsigned char>* out);

// ---- counters ------------------------------------------------------------
struct Stats {
  unsigned long long articles = 0;
  unsigned long long usable_titles = 0;
  unsigned long long articles_with_mentions = 0;
  unsigned long long articles_with_prefix = 0;
  unsigned long long mentions_total = 0;
  unsigned long long prefix_tokens_total = 0;
  unsigned long long chunks_total = 0;
  unsigned long long anchors = 0;
};

// ---- the coder side ------------------------------------------------------
// `CoderZ3` runs the causal form during the run.  It sees every coded token
// (encoder and decoder alike), maintains the mention index, recognizes article
// boundary and title end, and delivers the prefix when the title has been
// closed.  The token stream lives in a file, not in RAM; a window of about
// 1.1 kB is read per mention, and RAM is spent only on the index.
class CoderZ3 {
 public:
  explicit CoderZ3(const Params& p);
  ~CoderZ3();

  // Take in one coded token.  `piece_end` is true when this token was the
  // last of an article separator sequence (the coder recognizes that anyway
  // for its own reset).
  void Observe(unsigned char token, bool piece_end);

  // If the last `Observe` made a prefix due, this delivers it once; after
  // that it is consumed.
  bool TakePrefix(const unsigned char** toks, int* n);
  bool HasPrefix() const { return have_prefix_; }

  // The supplied order-0 row (208 x float16) for the prefix positions.
  const uint16_t* Prior0() const { return prior0_.data(); }
  bool Prior0Given() const { return prior0_given_; }

  void AddFed(unsigned long long n) { fed_ += n; }
  void Report() const;

 private:
  Params p_;
  FILE* file_ = nullptr;              // the token stream on disk
  long long pos_ = 0;                 // number of tokens taken in
  long long written_ = 0;             // of which already in the file
  unsigned char wbuf_[1 << 16];
  int wused_ = 0;
  unsigned char tail_[16];            // the last tokens (anchor indexing)
  MentionIndex index_;
  long long article_start_ = -1;      // absolute position of the first title token
  bool in_title_ = false;
  bool have_prefix_ = false;
  std::vector<unsigned char> prefix_;
  std::vector<long long> cand_, ment_;
  std::vector<unsigned char> scratch_;   // window per mention
  std::vector<unsigned char> title_;
  std::vector<uint16_t> prior0_;
  bool prior0_given_ = false;
  FILE* dump_tok_ = nullptr;
  FILE* dump_idx_ = nullptr;
  unsigned long long fed_ = 0;
  Stats st_;

  void Append(unsigned char token);
  void Flush();
  void ReadRange(long long a, long long b, unsigned char* dst);
  void BuildPrefixNow(int title_len);
};

}

#endif
