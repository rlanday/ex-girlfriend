
#include "tf1_z3.h"
#include "../cold_size.h"

#include "tf1_prior0.h"   // the stamped prior0 table

#include <string.h>
#include <sys/mman.h>

#include <string>

namespace tf_z3 {

static const unsigned char kTitleEnd[1] =
    { 0x08 };

const unsigned char* TitleEnd() { return kTitleEnd; }
int TitleEndLen() { return 1; }

// ---------------------------------------------------------------------------
// FNV-1a, identical to _key_hash in zctx/z3_extract.py.
// ---------------------------------------------------------------------------
uint64_t KeyHash(const unsigned char* seq, int len) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < len; ++i) {
    h = (h ^ (uint64_t)seq[i]) * 1099511628211ULL;
  }
  return h;
}

uint64_t KeyOf(const unsigned char* seq) {
  return KeyHash(seq, kIndexKeyTokens);
}

uint64_t KeyOfShort(const unsigned char* title3, unsigned char fourth) {
  unsigned char buf[kIndexKeyTokens];
  for (int i = 0; i < kMinTitleTokens; ++i) buf[i] = title3[i];
  buf[kMinTitleTokens] = fourth;
  return KeyHash(buf, kIndexKeyTokens);
}

// ---------------------------------------------------------------------------
// MentionIndex
// ---------------------------------------------------------------------------
// Anonymous mapping of `bytes`, or grow one with mremap (MREMAP_MAYMOVE: the
// pages move, nothing is copied, so old and new size are never resident side
// by side).  No fallback: a failure stops the run.
static void* MapNeu(size_t bytes) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) transformer::Fail("link context: mmap of %zu B failed", bytes);
  return p;
}
static void* MapGross(void* p, size_t alt, size_t neu) {
  if (!p) return MapNeu(neu);
  void* q = mremap(p, alt, neu, MREMAP_MAYMOVE);
  if (q == MAP_FAILED) transformer::Fail("link context: mremap to %zu B failed", neu);
  return q;
}

MentionIndex::~MentionIndex() {
  if (entries_) munmap(entries_, cap_ * sizeof(Entry));
  if (next_) munmap(next_, cap_ * sizeof(int));
  if (head_) munmap(head_, buckets_ * sizeof(int));
}

COLD_SIZE_ATTR void MentionIndex::Reserve(size_t n_anchors) {
  if (n_anchors < 1024) n_anchors = 1024;
  if (n_anchors > cap_) GrowEntries(n_anchors);
  size_t buckets = 1024;
  while (buckets * kLoad < n_anchors) buckets <<= 1;
  if (buckets > buckets_) Rebuild(buckets);
}

void MentionIndex::GrowEntries(size_t cap) {
  entries_ = (Entry*)MapGross(entries_, cap_ * sizeof(Entry), cap * sizeof(Entry));
  next_ = (int*)MapGross(next_, cap_ * sizeof(int), cap * sizeof(int));
  cap_ = cap;
}

// The head table is derived from entries_: drop the old one first, then chain
// every entry again in insertion order (identical chains to adding them one by
// one).
void MentionIndex::Rebuild(size_t buckets) {
  if (head_) munmap(head_, buckets_ * sizeof(int));
  head_ = (int*)MapNeu(buckets * sizeof(int));
  buckets_ = buckets;
  mask_ = buckets - 1;
  memset(head_, 0xFF, buckets * sizeof(int));   // -1 = empty
  for (size_t i = 0; i < n_; ++i) {
    const size_t b = (size_t)entries_[i].tag & mask_;
    next_[i] = head_[b];
    head_[b] = (int)i;
  }
}

void MentionIndex::Add(uint64_t key4, long long anchor) {
  if (!head_) Reserve(1024);
  if (n_ >= buckets_ * kLoad) Rebuild(buckets_ * 2);
  if (n_ == cap_) GrowEntries(cap_ * 2);
  if (anchor < 0 || anchor > 0xFFFFFFFFLL) {
    transformer::Fail("link context: position %lld exceeds 32 bits", anchor);
  }
  Entry e;
  e.tag = (uint32_t)(key4 >> 32) ^ (uint32_t)key4;
  e.pos = (uint32_t)anchor;
  const size_t b = (size_t)e.tag & mask_;
  entries_[n_] = e;
  next_[n_] = head_[b];
  head_[b] = (int)n_;
  ++n_;
}

void MentionIndex::Lookup(uint64_t key, std::vector<long long>* out) const {
  out->clear();
  if (!head_) return;
  const uint32_t tag = (uint32_t)(key >> 32) ^ (uint32_t)key;
  const size_t b = (size_t)tag & mask_;
  for (int i = head_[b]; i >= 0; i = next_[i]) {
    if (entries_[i].tag == tag) out->push_back((long long)entries_[i].pos);
  }
  // The chain runs backwards through the insertion order, i.e. descending by
  // position.  Reverse to ascending: the order is defined by the position,
  // never by the table order.
  for (size_t i = 0, j = out->size(); i + 1 < j; ++i, --j) {
    const long long t = (*out)[i];
    (*out)[i] = (*out)[j - 1];
    (*out)[j - 1] = t;
  }
}

// ---------------------------------------------------------------------------
// Stage 1: title
// ---------------------------------------------------------------------------
int TitleLength(const Window& win, long long start, long long next_boundary) {
  const int k = TitleEndLen();
  long long stop = start + kMaxTitleTokens + k;
  if (stop > next_boundary) stop = next_boundary;
  if (stop > win.base + win.len) stop = win.base + win.len;
  if (stop - start < k) return -1;
  const unsigned char first = kTitleEnd[0];
  for (long long p = start; p <= stop - k; ++p) {
    if (win.At(p) != first) continue;
    bool eq = true;
    for (int i = 1; i < k && eq; ++i) eq = win.At(p + i) == kTitleEnd[i];
    if (!eq) continue;
    const long long len = p - start;
    if (len < kMinTitleTokens) return -1;
    return (int)len;
  }
  return -1;
}

static void ParagraphBounds(const Window& win, long long m0, long long m1,
    const Params& p, long long limit, long long* out_start, long long* out_end) {
  long long lo = m0 - p.line_scan;
  if (lo < 0) lo = 0;
  long long start = lo;
  for (long long q = m0 - 1; q >= lo; --q) {       // last paragraph left
    if (win.At(q) == kParagraph) { start = q + 1; break; }
  }
  long long hi = m1 + p.line_scan;
  const long long n = win.base + win.len;
  if (hi > n) hi = n;
  if (hi > limit) hi = limit;
  long long end = hi;
  for (long long q = m1; q < hi; ++q) {            // first paragraph right
    if (win.At(q) == kParagraph) { end = q; break; }
  }
  *out_start = start;
  *out_end = end;
}

void ChunkForMention(const Window& win, long long m0, long long m1,
    const Params& p, long long limit, long long* out_a, long long* out_b) {
  const long long span = m1 - m0;
  if (span >= p.chunk) {                           // the mention alone
    *out_a = m0;                                   // fills the budget
    *out_b = m0 + p.chunk;
    return;
  }
  long long start = 0, end = 0;
  ParagraphBounds(win, m0, m1, p, limit, &start, &end);
  const long long room_before = m0 - start;
  const long long room_after = end - m1;
  const long long need = p.chunk - span;
  const long long half = need / 2;                 // integer halving as in Python
  long long take_before, take_after;
  if (room_after <= half) {                        // mention at paragraph end
    take_after = room_after;
    take_before = room_before < need - take_after ? room_before
                                                  : need - take_after;
  } else if (room_before <= half) {                // at paragraph start
    take_before = room_before;
    take_after = room_after < need - take_before ? room_after
                                                 : need - take_before;
  } else {                                         // in the middle: symmetric
    take_before = half;
    take_after = need - half;
  }
  *out_a = m0 - take_before;
  *out_b = m1 + take_after;
}

// ---------------------------------------------------------------------------
// Stage 4: prefix (tool mode, the window covers the whole stream)
// ---------------------------------------------------------------------------
namespace {
struct WholeStreamChunk {
  const Window* win;
  const Params* p;
  void operator()(long long m0, long long m1, long long limit,
      long long* a, long long* b) const {
    ChunkForMention(*win, m0, m1, *p, limit, a, b);
  }
};
}  // namespace

int BuildPrefix(const Window& win, const std::vector<long long>& mentions,
    int title_len, long long article_start, const Params& p,
    std::vector<unsigned char>* out) {
  long long taken_a[64], taken_b[64];
  WholeStreamChunk fn{&win, &p};
  const int n_taken = SelectChunks(mentions, title_len, article_start, p, fn,
      taken_a, taken_b);
  if (n_taken <= 0) return n_taken;

  for (int k = n_taken - 1, w = 0; k >= 0; --k, ++w) {  // oldest first
    if (w) { out->push_back(kParagraph); out->push_back(kParagraph); }
    for (long long q = taken_a[k]; q < taken_b[k]; ++q) {
      out->push_back(win.At(q));
    }
  }
  const unsigned char* sep = transformer::SeparatorIndices();
  const int sep_len = transformer::SeparatorLength();
  for (int i = 0; i < sep_len; ++i) out->push_back(sep[i]);
  return n_taken;
}

// ===========================================================================
// CoderZ3: the coder side of the causal form
// ===========================================================================
COLD_SIZE_ATTR CoderZ3::CoderZ3(const Params& p) : p_(p) {
  file_ = tmpfile();
  if (!file_) transformer::Fail("link context: cannot create the stream file");
  memset(tail_, 0xFF, sizeof(tail_));
  prefix_.reserve(512);
  scratch_.reserve(4096);
  prior0_.assign(208, 0);
  // The prior0 row (208 x float16) is a stamped constant (the tf1_prior0_set
  // table chosen at build time).
  if (const uint16_t* st = transformer::Prior0Stamped()) {
    memcpy(prior0_.data(), st, sizeof(uint16_t) * 208);
    prior0_given_ = true;
  } else {
    // Uniform distribution if nothing is stamped.
    const uint16_t u = transformer::FloatToHalf(1.0f / (float)208);
    for (int i = 0; i < 208; ++i) prior0_[i] = u;
  }
  if (const char* q = transformer::Z3DumpPath()) {
    std::string a = std::string(q) + "z3-prefix-tokens.uint8";
    std::string b = std::string(q) + "z3-prefix-index.i64";
    dump_tok_ = fopen(a.c_str(), "wb");
    if (!dump_tok_) transformer::Fail("cannot create %s", a.c_str());
    dump_idx_ = fopen(b.c_str(), "wb");
    if (!dump_idx_) transformer::Fail("cannot create %s", b.c_str());
  }
}

COLD_SIZE_ATTR CoderZ3::~CoderZ3() {
  if (dump_tok_) fclose(dump_tok_);
  if (dump_idx_) fclose(dump_idx_);
  if (file_) fclose(file_);
}

void CoderZ3::Flush() {
  if (!wused_) return;
  // Reading and writing share one file pointer, so the write position is
  // sought before every write; otherwise the next append would land where the
  // last window was read and silently corrupt the stream.
  if (fseeko(file_, (off_t)written_, SEEK_SET) != 0) {
    transformer::Fail("link context: seek to %lld failed",
        written_);
  }
  if (fwrite(wbuf_, 1, (size_t)wused_, file_) != (size_t)wused_) {
    transformer::Fail("link context: short write");
  }
  written_ += wused_;
  wused_ = 0;
}

void CoderZ3::Append(unsigned char token) {
  if (wused_ == (int)sizeof(wbuf_)) Flush();
  wbuf_[wused_++] = token;
  ++pos_;
}

void CoderZ3::ReadRange(long long a, long long b, unsigned char* dst) {
  if (b <= a) return;
  Flush();
  if (fseeko(file_, (off_t)a, SEEK_SET) != 0) {
    transformer::Fail("link context: seek to %lld failed", a);
  }
  const size_t n = (size_t)(b - a);
  if (fread(dst, 1, n, file_) != n) {
    transformer::Fail("link context: short read [%lld,%lld)", a, b);
  }
}

COLD_SIZE_ATTR void CoderZ3::BuildPrefixNow(int title_len) {
  prefix_.clear();
  ++st_.usable_titles;

  // Fetch the title from the stream (it sits right before the current spot).
  title_.assign((size_t)title_len, 0);
  ReadRange(article_start_, article_start_ + title_len, title_.data());

  // Candidates: one key when L >= 4, otherwise fanned out over the fourth
  // token (the union is exactly the same set).
  ment_.clear();
  const int n_keys = (title_len >= kIndexKeyTokens) ? 1 : 256;
  unsigned char probe[kMaxTitleTokens];
  for (int kq = 0; kq < n_keys; ++kq) {
    const uint64_t key = (n_keys == 1)
        ? KeyOf(title_.data())
        : KeyOfShort(title_.data(), (unsigned char)kq);
    index_.Lookup(key, &cand_);
    for (size_t c = 0; c < cand_.size(); ++c) {
      const long long s = cand_[c] + 1;
      if (s + title_len > pos_) continue;
      ReadRange(s, s + title_len, probe);
      if (memcmp(probe, title_.data(), (size_t)title_len) == 0) {
        ment_.push_back(cand_[c]);
      }
    }
  }
  if (ment_.empty()) return;
  // Order defined by the position, not by the table.
  for (size_t x = 1; x < ment_.size(); ++x) {
    const long long v = ment_[x];
    size_t y = x;
    while (y > 0 && ment_[y - 1] > v) { ment_[y] = ment_[y - 1]; --y; }
    ment_[y] = v;
  }
  ++st_.articles_with_mentions;
  st_.mentions_total += (unsigned long long)ment_.size();

  // Read one window per mention and compute the chunk.
  long long taken_a[64], taken_b[64];
  std::vector<unsigned char>& buf = scratch_;
  CoderZ3* self = this;
  auto get_chunk = [self, &buf](long long m0, long long m1, long long limit,
      long long* a, long long* b) {
    long long lo = m0 - self->p_.line_scan;
    if (lo < 0) lo = 0;
    long long hi = m1 + self->p_.line_scan;
    if (hi > self->pos_) hi = self->pos_;
    if (hi > limit) hi = limit;
    if (hi < m1) hi = m1;
    buf.assign((size_t)(hi - lo), 0);
    self->ReadRange(lo, hi, buf.data());
    Window w;
    w.p = buf.data();
    w.base = lo;
    w.len = hi - lo;
    ChunkForMention(w, m0, m1, self->p_, limit, a, b);
  };
  const int n_taken = SelectChunks(ment_, title_len, article_start_, p_,
      get_chunk, taken_a, taken_b);
  if (n_taken < 0) {
    transformer::Fail("link context: chunk reaches past the article start %lld", article_start_);
  }
  if (n_taken == 0) return;

  for (int k = n_taken - 1, w = 0; k >= 0; --k, ++w) {   // oldest first
    if (w) { prefix_.push_back(kParagraph); prefix_.push_back(kParagraph); }
    const size_t at = prefix_.size();
    prefix_.resize(at + (size_t)(taken_b[k] - taken_a[k]));
    ReadRange(taken_a[k], taken_b[k], prefix_.data() + at);
  }
  const unsigned char* sep = transformer::SeparatorIndices();
  const int sep_len = transformer::SeparatorLength();
  for (int i = 0; i < sep_len; ++i) prefix_.push_back(sep[i]);

  ++st_.articles_with_prefix;
  st_.chunks_total += (unsigned long long)n_taken;
  st_.prefix_tokens_total += (unsigned long long)prefix_.size();
  have_prefix_ = true;

  if (dump_tok_) {
    fwrite(prefix_.data(), 1, prefix_.size(), dump_tok_);
    const long long rec[2] = { article_start_, (long long)prefix_.size() };
    fwrite(rec, sizeof(long long), 2, dump_idx_);
  }
}

void CoderZ3::Observe(unsigned char token, bool piece_end) {
  Append(token);
  for (int i = 0; i + 1 < (int)sizeof(tail_); ++i) tail_[i] = tail_[i + 1];
  tail_[sizeof(tail_) - 1] = token;

  // Index the anchor as soon as the four tokens behind it are there.
  const int back = kIndexKeyTokens + 1;                 // anchor + 4 tokens
  if (pos_ >= back) {
    const unsigned char* t = tail_ + sizeof(tail_) - back;
    if (t[0] == kLinkOpen) {
      ++st_.anchors;
      index_.Add(KeyHash(t + 1, kIndexKeyTokens), pos_ - back);
    }
  }

  if (piece_end) {                     // the next position is the title
    article_start_ = pos_;
    in_title_ = true;
    ++st_.articles;
    return;
  }
  if (!in_title_) return;

  const int k = TitleEndLen();
  const long long len_here = pos_ - article_start_;     // tokens since start
  if (len_here >= k) {
    const unsigned char* t = tail_ + sizeof(tail_) - k;
    if (memcmp(t, TitleEnd(), (size_t)k) == 0) {
      const int title_len = (int)(len_here - k);
      in_title_ = false;
      if (title_len >= kMinTitleTokens && title_len <= kMaxTitleTokens) {
        BuildPrefixNow(title_len);
      }
      return;
    }
  }
  if (len_here > kMaxTitleTokens + k) in_title_ = false;   // no title
}

bool CoderZ3::TakePrefix(const unsigned char** toks, int* n) {
  if (!have_prefix_) return false;
  have_prefix_ = false;
  *toks = prefix_.data();
  *n = (int)prefix_.size();
  return true;
}

void CoderZ3::Report() const {
}

}

