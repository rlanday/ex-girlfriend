#ifndef TF1_Z3_H_
#define TF1_Z3_H_

#include "../transformer.h"

#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "tf1_engine.h"   // separator sequence (SeparatorIndices/-Length)
#include "tf1_prior0.h"   // the stamped prior0 table

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

// The mentions depend on the 32-bit fold of the key, (key >> 32) ^ key, not
// only on the key: a fan-out lookup (BuildPrefixNow, needle of 2 or 3 tokens)
// took an anchor once for every fan-out key with the same fold, and the needle
// comparison keeps all of these copies (377 over enwik9).  The index
// compares 24 bits of the fold; BuildPrefixNow checks the whole fold against
// the stream (FoldOf), so the copies stay what they were.
//
// Memory layout: one 64-bit word per anchor and a fixed head table, both in
// anonymous mappings of their own (not std::vector or malloc: glibc may serve
// a large request from free heap below the brk top, and a realloc there
// copies):
//   * word = position (bits 0-29) | link (bits 30-53) | fifth token (bits
//     54-61) | tag (bits 62-63).  The link is 1 + the index of the previous
//     entry of the same bucket, 0 ends the chain.  Bucket = low 22 bits of the
//     fold, tag = the next 2 bits.
//   * the fifth token behind the link opener (the key holds four) rejects a
//     candidate of a needle of five or more tokens without reading the stream:
//     over enwik9 1.65 G candidates had the key of their needle, 1.39 G of
//     them differed in the fifth token.  An anchor goes into the
//     index when its fifth token is there.
//   * entries_ grows by mremap (the pages move, nothing is copied; a vector
//     would hold old and new buffer at the same time, +96 MiB in ru_maxrss).
//   * head_ has a fixed 2^22 buckets (16 MiB): about two anchors per bucket
//     at the end of enwik9 (9.9 M).  Zero = empty, so the fresh mapping needs
//     no initialisation; pages nobody touched are not resident.
// 8 B per anchor plus 16 MiB, against 12 B plus 32 MiB at the end of enwik9.
class MentionIndex {
 public:
  MentionIndex() = default;
  ~MentionIndex();
  MentionIndex(const MentionIndex&) = delete;
  MentionIndex& operator=(const MentionIndex&) = delete;
  void Reserve(size_t n_anchors);
  // Record one mention position (position of the link opener).
  // `key4` is the key of the four tokens behind it, `t5` the fifth token.
  void Add(uint64_t key4, long long anchor, unsigned char t5 = 0);
  // The entries with 24 bits of the fold of `key` (and with fifth token `t5`
  // unless t5 < 0), ascending by position.  Never in table order: the result
  // must not depend on the table layout.
  void Lookup(uint64_t key, std::vector<long long>* out, int t5 = -1) const;
  size_t Size() const { return n_; }
  size_t Buckets() const { return kBuckets; }
  // Approximate RAM demand in bytes (mapped size).
  size_t Bytes() const {
    return cap_ * sizeof(uint64_t) + kBuckets * sizeof(uint32_t);
  }

 private:
  static constexpr int kBucketBits = 22;
  static constexpr size_t kBuckets = (size_t)1 << kBucketBits;
  static constexpr int kPosBits = 30;    // the coder stream has ~0.5 G tokens
  static constexpr int kLinkBits = 24;   // anchors: 9.9 M on enwik9
  static constexpr int kT5 = kPosBits + kLinkBits;   // fifth token, then the tag
  static_assert(kT5 + 8 + 2 == 64, "one index word: position, link, fifth token, tag");
  uint64_t* entries_ = nullptr;
  uint32_t* head_ = nullptr;   // bucket -> 1 + latest entry, 0 = empty
  size_t n_ = 0;               // entries in use
  size_t cap_ = 0;             // capacity of entries_
  void Grow(size_t cap);
};

uint64_t KeyHash(const unsigned char* seq, int len);
// the 32-bit fold of the key of four tokens
inline uint32_t FoldOf(uint64_t key) { return (uint32_t)(key >> 32) ^ (uint32_t)key; }
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

template <class ChunkFn>
int SelectChunks(const std::vector<long long>& mentions, int title_len,
    long long article_start, const Params& p, ChunkFn get_chunk,
    long long* taken_a, long long* taken_b
    , const unsigned char* kurz = nullptr
    ) {
  const int sep_len = transformer::SeparatorLength();
  const long long room = (long long)p.budget - sep_len;
  if (room <= 0) return 0;

  // earlier mentions: those that end before the article starts
  long long n_earlier = 0;
  for (size_t i = 0; i < mentions.size(); ++i) {
    if (mentions[i] + p.open_len + title_len - (kurz ? kurz[i] : 0)
        <= article_start) ++n_earlier;
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
    const long long m1 = m0 + p.open_len + title_len
        - (kurz ? kurz[(size_t)idx] : 0);
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
  unsigned long long anchors_skipped[4] = {0, 0, 0, 0};   // anchors not indexed, by class of OhneTitel()
  unsigned long long needles_lo = 0;
  unsigned long long q_seiten = 0;             // pages in the page table
  unsigned long long q_seiten_redirect = 0;    // of which redirect pages
  unsigned long long q_seiten_ohne_titel = 0;  // first line empty or > 64 tokens
  unsigned long long q_gruppen_redirect = 0;    // source groups that are a redirect page
  unsigned long long q_gruppen_ohne_titel = 0;  // source groups without a title
  unsigned long long q_gruppen = 0;            // source groups (pages per prefix)
  unsigned long long q_gruppen_titel = 0;      // groups that got a title
  unsigned long long q_gruppen_mehr = 0;       // groups with two or more chunks
  unsigned long long q_titel_token = 0;        // tokens added ("<title>: ")
  unsigned long long q_max_praefix = 0;        // longest prefix (tokens)
  unsigned long long q_ueber_budget = 0;       // prefixes longer than the budget
  unsigned long long q_hist[9] = {0};          // prefixes by number of source pages
  // the other value, computed alongside (self-report only)
  unsigned long long q_alt_chunks = 0, q_alt_token = 0, q_alt_titel_token = 0;
  unsigned long long q_alt_seiten_weniger = 0;   // prefixes with fewer chunks
  unsigned long long klein_articles = 0;   // pages with a lower-case mention
  unsigned long long klein_mentions = 0;
  unsigned long long k_leer = 0;           // chunks with nothing of their own
  unsigned long long k_erweitert = 0;      // of which got neighbour lines
  unsigned long long k_zeilen = 0;         // neighbour lines added
  unsigned long long k_redir = 0;          // redirect lines left out
  unsigned long long k_dub = 0;            // repeated chunks left out
  unsigned long long k_kopf = 0;           // chunks in the head of the stream left out
};

constexpr int kKapMarke = transformer::kByteToIndex[0x40];
static_assert(kKapMarke >= 0 && kKapMarke < 208,
    "the capital marker 0x40 is not in the frozen byte set");

template <class ChunkFn, class QuelleFn, class KostenFn>
int SelectChunksQ(const std::vector<long long>& mentions, int title_len,
    long long article_start, const Params& p, ChunkFn get_chunk,
    long long* taken_a, long long* taken_b, const unsigned char* kurz,
    QuelleFn quelle, KostenFn tkosten) {
  const int sep_len = transformer::SeparatorLength();
  const long long room = (long long)p.budget - sep_len;
  if (room <= 0) return 0;
  long long n_earlier = 0;
  for (size_t i = 0; i < mentions.size(); ++i) {
    if (mentions[i] + p.open_len + title_len - (kurz ? kurz[i] : 0)
        <= article_start) ++n_earlier;
    else break;                                    // sorted ascending
  }
  if (n_earlier == 0) return 0;
  int n_taken = 0;
  long long used = 0, q_letzt = -1;
  const int cap = p.max_mentions < 64 ? p.max_mentions : 64;
  for (int k = 0; k < cap; ++k) {
    const long long idx = n_earlier - 1 - k;
    if (idx < 0) break;
    const long long m0 = mentions[(size_t)idx];
    const long long m1 = m0 + p.open_len + title_len
        - (kurz ? kurz[(size_t)idx] : 0);
    long long a = 0, b = 0;
    get_chunk(m0, m1, article_start, &a, &b);
    if (b > article_start) return -1;              // causality guard
    bool overlaps = false;
    for (int t = 0; t < n_taken; ++t) {
      if (a < taken_b[t] && taken_a[t] < b) { overlaps = true; break; }
    }
    if (overlaps) continue;
    const long long qs = quelle(a);
    const long long cost = (b - a) + (n_taken ? 2 : 0)
        + (qs != q_letzt ? tkosten(qs) : 0);
    if (used + cost > room) break;
    taken_a[n_taken] = a;
    taken_b[n_taken] = b;
    ++n_taken;
    used += cost;
    q_letzt = qs;
  }
  return n_taken;
}

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

  // The supplied order-0 row (208 x float16) for the prefix positions.  In
  // the package form the stamped table is the only source: no copy.
  const uint16_t* Prior0() const { return transformer::Prior0Stamped(); }
  bool Prior0Given() const { return prior0_given_; }

  void AddFed(unsigned long long n) { fed_ += n; }
  void Report() const;
  void PageEnd(bool redirect);

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
  std::vector<unsigned char> kurz_;      // per mention: 1 = lower-case spelling
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
  std::vector<uint32_t> s_ab_;          // first token of the page
  std::vector<unsigned char> s_tl_;     // title length, bit 7 = redirect page
  long long s_lauf_ab_ = 0;             // first token of the running page
  int s_lauf_tl_ = -1;
  std::vector<uint32_t> s_zi_;
  long long s_lauf_zi_ = -1;
  // source page of every chunk taken; at most 64 chunks (taken_a/taken_b of
  // BuildPrefixNow).  A field, not a vector
  long long q_seite_[64];
  long long Quelle(long long a) const;  // page that holds position a
  int TitelKosten(long long s) const;   // tokens of "<title>: ", 0 = none
  void QuelleTitel(long long s);        // append "<title>: " to prefix_
  unsigned long long QuelleMehr(int n) const;   // groups of >= 2 chunks
  void QuelleZaehlen(int n, const long long* a, const long long* b,
      unsigned long long* gruppen, unsigned long long* titel_tok,
      unsigned long long* chunk_tok) const;
  int WaehleK(int title_len, long long* taken_a, long long* taken_b);
  void NachbarnK(const Window& w, long long s, long long* a, long long* b);
};

}  // namespace tf_z3

#endif
