
#include "../meld.h"
#include "../census_title.h"
#include "tf1_z3.h"
#include "../cold_size.h"

#include "tf1_prior0.h"   // the stamped prior0 table

#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <string>

namespace tf_z3 {

static const unsigned char kTitleEnd[1] =
    { 0x08 };

const unsigned char* TitleEnd() { return kTitleEnd; }
int TitleEndLen() { return 1; }

constexpr unsigned char kTok(int b) { return (unsigned char)transformer::kByteToIndex[b]; }
static_assert(transformer::kByteToIndex[0x5D] >= 0 && transformer::kByteToIndex[0x23] >= 0 &&
    transformer::kByteToIndex[0x20] >= 0 && transformer::kByteToIndex[0x2A] >= 0 &&
    transformer::kByteToIndex[0x4A] >= 0 && transformer::kByteToIndex[0x4B] >= 0 &&
    transformer::kByteToIndex[0x03] >= 0, "a token of the Z3 selection rules is not in the frozen byte set");
constexpr unsigned char kZu = kTok(0x5D);      // ']'
constexpr unsigned char kRaute = kTok(0x23);   // '#'
constexpr unsigned char kSp = kTok(0x20);

static_assert(kEndeKlasse < 8, "the end row of the link context is a sentinel row");
constexpr int kEndeZeile = 208 + kEndeKlasse;
constexpr int kEndLen = 1;
// the tail [m1, b) of a chunk: an optional ']' and blanks
static bool NurSchluss(const Window& w, long long m1, long long b) {
  long long q = m1;
  if (q < b && w.At(q) == kZu) ++q;
  for (; q < b; ++q) if (w.At(q) != kSp) return false;
  return true;
}

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
  if (p == MAP_FAILED) msg_stop(226);
  return p;
}
static void* MapGross(void* p, size_t alt, size_t neu) {
  if (!p) return MapNeu(neu);
  void* q = mremap(p, alt, neu, MREMAP_MAYMOVE);
  if (q == MAP_FAILED) msg_stop(227);
  return q;
}

MentionIndex::~MentionIndex() {
  if (entries_) munmap(entries_, cap_ * sizeof(uint64_t));
  if (head_) munmap(head_, kBuckets * sizeof(uint32_t));
}

COLD_SIZE_ATTR void MentionIndex::Reserve(size_t n_anchors) {
  if (n_anchors > cap_) Grow(n_anchors);
}

void MentionIndex::Grow(size_t cap) {
  entries_ = (uint64_t*)MapGross(entries_, cap_ * sizeof(uint64_t), cap * sizeof(uint64_t));
  cap_ = cap;
}

void MentionIndex::Add(uint64_t key4, long long anchor, unsigned char t5) {
  if (!head_) head_ = (uint32_t*)MapNeu(kBuckets * sizeof(uint32_t));
  if (n_ == cap_) Grow(cap_ ? cap_ * 2 : (size_t)1 << 16);
  if (anchor < 0 || (anchor >> kPosBits) || ((n_ + 1) >> kLinkBits)) {
    __builtin_unreachable();
  }
  const uint32_t fold = FoldOf(key4);
  uint32_t& head = head_[fold & (kBuckets - 1)];
  entries_[n_] = (uint64_t)anchor | (uint64_t)head << kPosBits | (uint64_t)t5 << kT5 |
                 (uint64_t)(fold >> kBucketBits & 3) << (kT5 + 8);
  head = (uint32_t)++n_;
}

void MentionIndex::Lookup(uint64_t key, std::vector<long long>* out, int t5) const {
  out->clear();
  if (!head_) return;
  const uint32_t fold = FoldOf(key);
  // tag and fifth token in the position of the word, the fifth token masked
  // out when it is not compared
  const uint64_t mask = t5 < 0 ? 0x300 : 0x3FF;
  const uint64_t want = ((uint64_t)(fold >> kBucketBits & 3) << 8 | (uint64_t)(t5 & 255)) & mask;
  for (uint32_t i = head_[fold & (kBuckets - 1)]; i;) {
    const uint64_t e = entries_[i - 1];
    if ((e >> kT5 & mask) == want) {
      out->push_back((long long)(e & (((uint64_t)1 << kPosBits) - 1)));
    }
    i = (uint32_t)(e >> kPosBits) & (((uint32_t)1 << kLinkBits) - 1);
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

// ---------------------------------------------------------------------------
// Anchors no title can follow.  An anchor whose tokens
// rule out every title never becomes a mention and stays out of the mention
// index.  The rule reads only the tokens behind the link opener that the index
// entry reads anyway (key and fifth token), the same on the coder and on the
// decoder side.  Over enwik9 every class loses no mention (whole-stream
// replay); the self-report counts the needles that could reach an anchor
// outside the index (0 = nothing lost).
// ---------------------------------------------------------------------------
namespace {
constexpr unsigned char kTk(int b) { return (unsigned char)transformer::kByteToIndex[b]; }
constexpr unsigned char kDp = kTk(0x4A);   // ':' (the WRT stage swaps it to 0x4A)
constexpr unsigned char kSl = kTk(0x2F);   // '/'
constexpr unsigned char kKa = kTk(0x61);   // 'a'; 'a'..'z' are 26 tokens in a row
static_assert(transformer::kByteToIndex[0x61] >= 0 &&
    transformer::kByteToIndex[0x7A] - transformer::kByteToIndex[0x61] == 25,
    "the lower-case letters are not 26 tokens in a row");
// First tokens no title begins with: MediaWiki forbids # < > [ ] { } | and the
// line feed in a title and strips a leading blank, '_' and ':'.  The first '['
// of "[[" (in the coder stream mostly a single '[' of the text, hent9 swaps runs
// of one and two brackets) is one of them.  enwik9: 424,845 anchors behind
// '[', 39,930 behind the others.
constexpr uint64_t Bit(int b) { return (uint64_t)1 << kTk(b); }
constexpr uint64_t kKeinAnfang = Bit(0x0A) | Bit(0x20) | Bit(0x23) | Bit(0x4A) |
    Bit(0x4C) | Bit(0x4E) | Bit(0x50) | Bit(0x51) | Bit(0x52) | Bit(0x5B) | Bit(0x5D) |
    Bit(0x5F);
static_assert(kTk(0x0A) < 64 && kTk(0x20) < 64 && kTk(0x23) < 64 && kTk(0x4A) < 64 &&
    kTk(0x4C) < 64 && kTk(0x4E) < 64 && kTk(0x50) < 64 && kTk(0x51) < 64 &&
    kTk(0x52) < 64 && kTk(0x5B) < 64 && kTk(0x5D) < 64 && kTk(0x5F) < 64 &&
    kTk(0x5B) == kLinkOpen, "a first token of the anchor rule is not below 64");
// Interwiki links "[xy:" (mostly interlanguage links) with the 143 two-letter
// codes whose anchors never become a mention over enwik9 and that no needle
// can reach (no title "Xy" or "Xy:...").  Row = first letter, bit = second.
// Left out: io ru hu vi el lb ta mi ku li ga ka os ur yo co mg sn ne rm pa kr
// mo si (mentions of short titles such as "Io") and na pi ha re cf (a needle
// could reach them).  enwik9: 526,011 of 576,232 anchors "[xy:".
const uint32_t kSprache[26] = {
    0x3221023, 0x00471c1, 0x32e0091, 0x0200411, 0x01c6010, 0x1164321, 0x0322808,
    0x14a0100, 0x0100551, 0x0208001, 0x1647d00, 0x0306001, 0x10e2c80, 0x1202842, 0x0000204,
    0x00c4800, 0x0100000, 0x0406000, 0x0771cdd, 0x04e2d50, 0x2000440, 0x0004001, 0x000c001,
    0x0100080, 0x0000100, 0x01000c1};
// 0 = index the anchor (`t` = the tokens behind the link opener), else the class
// that keeps it out: 1 first token, 2 external link "[x://" (a URL scheme of
// one token, http: 385,852 anchors behind "[[http://"), 3 interwiki link "[xy:".
inline int OhneTitel(const unsigned char* t) {
  if (t[0] < 64 && (kKeinAnfang >> t[0] & 1)) return 1;
  if (t[1] == kDp && t[2] == kSl && t[3] == kSl) return 2;
  const unsigned a = (unsigned)(t[0] - kKa), b = (unsigned)(t[1] - kKa);
  if (a < 26 && b < 26 && t[2] == kDp && (kSprache[a] >> b & 1)) return 3;
  return 0;
}
// 1 when a needle of `len` >= 2 tokens could match an anchor that OhneTitel()
// keeps out of the index (self-report only)
inline int NadelOhneIndex(const unsigned char* n, int len) {
  if (n[0] < 64 && (kKeinAnfang >> n[0] & 1)) return 1;
  if (n[1] == kDp && (len < 3 || n[2] == kSl) && (len < 4 || n[3] == kSl)) return 1;
  const unsigned a = (unsigned)(n[0] - kKa), b = (unsigned)(n[1] - kKa);
  if (a < 26 && b < 26 && (len < 3 || n[2] == kDp) && (kSprache[a] >> b & 1)) return 1;
  return 0;
}
}  // namespace

// ===========================================================================
// CoderZ3: the coder side of the causal form
// ===========================================================================
COLD_SIZE_ATTR CoderZ3::CoderZ3(const Params& p) : p_(p) {
  file_ = tmpfile();
  if (!file_) __builtin_unreachable();
  memset(tail_, 0xFF, sizeof(tail_));
  prefix_.reserve(512);
  scratch_.reserve(4096);
  prior0_given_ = true;   // Prior0() is the stamped table
}

COLD_SIZE_ATTR CoderZ3::~CoderZ3() {
  if (file_) fclose(file_);
}

// The stream file is read and written with pread/pwrite at explicit offsets
// (no shared file position, no stdio buffer): one system call per window or
// candidate instead of a seek plus a 4 KiB buffer refill.
void CoderZ3::Flush() {
  if (!wused_) return;
  if (pwrite(fileno(file_), wbuf_, (size_t)wused_, (off_t)written_) != (ssize_t)wused_) {
    __builtin_unreachable();
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
  const size_t n = (size_t)(b - a);
  if (pread(fileno(file_), dst, n, (off_t)a) != (ssize_t)n) {
    __builtin_unreachable();
  }
}

COLD_SIZE_ATTR void CoderZ3::BuildPrefixNow(int title_len) {
  prefix_.clear();

  // Fetch the title from the stream (it sits right before the current spot).
  title_.assign((size_t)title_len, 0);
  ReadRange(article_start_, article_start_ + title_len, title_.data());
  for (unsigned char c : census_title::zusatz) title_.push_back((unsigned char)transformer::g_b2i[c]);
  if (!census_title::zusatz.empty()) ++census_title::n_z3;
  title_len = (int)title_.size();

  // Candidates: one key when L >= 4, otherwise fanned out over the fourth
  // token (the union is exactly the same set).
  ment_.clear();
  unsigned char probe[kMaxTitleTokens];
  const int n_form = (title_[0] == (unsigned char)kKapMarke) ? 2 : 1;
  unsigned long long n_klein = 0;
  for (int form = 0; form < n_form; ++form) {
    const unsigned char* nadel = title_.data() + form;
    const int nlen = title_len - form;             // >= 2
    // The index key is four tokens; a shorter needle is fanned out over the
    // tokens it lacks (V keys for three tokens, V * V for two).
    const int n_keys = nlen >= kIndexKeyTokens ? 1
        : (nlen == 3 ? 208 : 208 * 208);
    for (int kq = 0; kq < n_keys; ++kq) {
      unsigned char k4[kIndexKeyTokens];
      k4[0] = nadel[0];
      k4[1] = nadel[1];
      k4[2] = nlen > 2 ? nadel[2] : (unsigned char)(kq / 208);
      k4[3] = nlen > 3 ? nadel[3] : (unsigned char)(kq % 208);
      const uint64_t key = KeyHash(k4, kIndexKeyTokens);
      index_.Lookup(key, &cand_, nlen > kIndexKeyTokens ? nadel[kIndexKeyTokens] : -1);
      // a needle shorter than the key reads the four key tokens as well: the
      // anchor counts under this fan-out key only with the same 32-bit fold
      const int lesen = nlen < kIndexKeyTokens ? kIndexKeyTokens : nlen;
      for (size_t c = 0; c < cand_.size(); ++c) {
        const long long s = cand_[c] + 1;
        if (s + nlen > pos_) continue;
        // A mention need not end the link target: "[[Title (band)]]" is a
        // mention of "Title" too.  These prefix matches stay on purpose:
        // their chunks carry more links of the article text than the older
        // exact mentions that would take their places.
        ReadRange(s, s + lesen, probe);
        if (memcmp(probe, nadel, (size_t)nlen) == 0 && (nlen >= kIndexKeyTokens ||
            FoldOf(KeyHash(probe, kIndexKeyTokens)) == FoldOf(key))) {
          ment_.push_back(cand_[c] * 2 + (form == 0 ? 1 : 0));
          n_klein += (unsigned long long)form;
        }
      }
    }
  }
  if (ment_.empty()) return;
  if (n_klein) {
  }
  // Order defined by the position, not by the table.
  for (size_t x = 1; x < ment_.size(); ++x) {
    const long long v = ment_[x];
    size_t y = x;
    while (y > 0 && ment_[y - 1] > v) { ment_[y] = ment_[y - 1]; --y; }
    ment_[y] = v;
  }
  // Back to positions; the form goes into kurz_ (1 = one token shorter).
  kurz_.resize(ment_.size());
  for (size_t x = 0; x < ment_.size(); ++x) {
    kurz_[x] = (unsigned char)(1 - (int)(ment_[x] & 1));
    ment_[x] >>= 1;
  }

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
  const int n_taken = WaehleK(title_len, taken_a, taken_b);
  (void)get_chunk;
  if (n_taken < 0) {
    __builtin_unreachable();
  }
  if (n_taken == 0) return;
  for (int k = 0; k < n_taken; ++k) q_seite_[(size_t)k] = Quelle(taken_a[k]);

  for (int k = n_taken - 1, w = 0; k >= 0; --k, ++w) {   // oldest first
    if (w) { prefix_.push_back(kParagraph); prefix_.push_back(kParagraph); }
    if (k + 1 == n_taken || q_seite_[(size_t)k] != q_seite_[(size_t)k + 1])
      QuelleTitel(q_seite_[(size_t)k]);
    const size_t at = prefix_.size();
    prefix_.resize(at + (size_t)(taken_b[k] - taken_a[k]));
    ReadRange(taken_a[k], taken_b[k], prefix_.data() + at);
  }
  prefix_.push_back((unsigned char)kEndeZeile);

  have_prefix_ = true;

}

void CoderZ3::Observe(unsigned char token, bool piece_end) {
  Append(token);
  for (int i = 0; i + 1 < (int)sizeof(tail_); ++i) tail_[i] = tail_[i + 1];
  tail_[sizeof(tail_) - 1] = token;
  if (s_lauf_tl_ < 0) {
    const int kq = TitleEndLen();
    const long long lq = pos_ - s_lauf_ab_;
    if (lq >= kq && memcmp(tail_ + sizeof(tail_) - kq, TitleEnd(), (size_t)kq) == 0) {
      s_lauf_tl_ = (lq - kq <= kMaxTitleTokens) ? (int)(lq - kq) : 0;
    } else if (lq > kMaxTitleTokens + kq) {
      s_lauf_tl_ = 0;
    }
    // the census suffix of this page (set by the census automaton at the page
    // anchor, as for the title of the link context itself)
    if (s_lauf_tl_ > 0) s_lauf_zi_ = census_title::zusatz_i;
  }

  // Index the anchor as soon as the five tokens behind it are there (the key
  // and the fifth token), one token later than the key alone needs.  This
  // hides nothing from a lookup: the anchor it could hide stands four tokens
  // before the title end, i.e. in the title or in the separator in front of
  // it, and neither holds '['.
  // Anchors no title can follow stay out (OhneTitel above).
  const int back = kIndexKeyTokens + 2;                 // anchor + 5 tokens
  if (pos_ >= back) {
    const unsigned char* t = tail_ + sizeof(tail_) - back;
    if (t[0] == kLinkOpen) {
      const int ohne = OhneTitel(t + 1);
      if (!ohne) {
        index_.Add(KeyHash(t + 1, kIndexKeyTokens), pos_ - back, t[kIndexKeyTokens + 1]);
      } else {
      }
    }
  }

  if (piece_end) {                     // the next position is the title
    article_start_ = pos_;
    in_title_ = true;
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
      const int voll = title_len + (int)census_title::zusatz.size();   // the restored title
      if (voll >= kMinTitleTokens && voll <= kMaxTitleTokens) {
        BuildPrefixNow(title_len);
      }
      return;
    }
  }
  if (len_here > kMaxTitleTokens + k) in_title_ = false;   // no title
}

void CoderZ3::PageEnd(bool redirect) {
  if (s_lauf_ab_ > 0xFFFFFFFFLL) {
    __builtin_unreachable();
  }
  const int tl = s_lauf_tl_ > 0 ? s_lauf_tl_ : 0;
  if (s_ab_.empty()) { s_ab_.reserve(1 << 18); s_tl_.reserve(1 << 18); }
  s_ab_.push_back((uint32_t)s_lauf_ab_);
  s_tl_.push_back((unsigned char)(tl | (redirect ? 0x80 : 0)));
  s_zi_.push_back((uint32_t)(int32_t)(tl ? s_lauf_zi_ : -1));
  s_lauf_zi_ = -1;
  s_lauf_ab_ = pos_;
  s_lauf_tl_ = -1;
}

// The page that holds position a: the last finished page that starts at or
// before a.  A chunk lies before the current title, i.e. in a finished page.
long long CoderZ3::Quelle(long long a) const {
  if (a >= s_lauf_ab_) {
    __builtin_unreachable()
;
  }
  size_t lo = 0, hi = s_ab_.size();
  while (lo < hi) {                                // first start > a
    const size_t m = (lo + hi) / 2;
    if ((long long)s_ab_[m] <= a) lo = m + 1; else hi = m;
  }
  if (lo == 0) __builtin_unreachable();
  return (long long)lo - 1;
}

int CoderZ3::TitelKosten(long long s) const {
  const unsigned char t = s_tl_[(size_t)s];
  if (t & 0x80) return 0;                          // redirect page
  const int tl = t & 0x7f;
  const int32_t zi = (int32_t)s_zi_[(size_t)s];    // the census suffix of a stripped title
  if (tl && zi >= 0) return tl + (int)census_title::zusaetze[(size_t)zi].size() + 2;
  return tl ? tl + 2 : 0;                          // title + ": "
}

void CoderZ3::QuelleTitel(long long s) {
  const unsigned char t = s_tl_[(size_t)s];
  if (t & 0x80) {return; }
  const int tl = t & 0x7f;
  if (!tl) {return; }
  const size_t at = prefix_.size();
  prefix_.resize(at + (size_t)tl);
  ReadRange((long long)s_ab_[(size_t)s], (long long)s_ab_[(size_t)s] + tl,
            prefix_.data() + at);
  const int32_t zi = (int32_t)s_zi_[(size_t)s];
  if (zi >= 0) {
    const std::string& z = census_title::zusaetze[(size_t)zi];
    for (unsigned char ch : z) prefix_.push_back((unsigned char)transformer::g_b2i[ch]);
    st_.q_titel_token += (unsigned long long)z.size();
    ++census_title::n_z3q;
  }
  // ": " in coder-stream form: the WRT stage swaps 0x3A and 0x4A
  // (dictionary.cpp next_char), so the colon of the text is the byte 0x4A.
  static_assert(transformer::kByteToIndex[0x4A] >= 0 && transformer::kByteToIndex[0x20] >= 0,
      "colon or space is not in the frozen byte set");
  prefix_.push_back((unsigned char)transformer::kByteToIndex[0x4A]);
  prefix_.push_back((unsigned char)transformer::kByteToIndex[0x20]);
}

unsigned long long CoderZ3::QuelleMehr(int n) const {
  unsigned long long r = 0;
  for (int k = n - 1; k > 0; --k) {               // print order
    const bool neu = (k + 1 == n) || q_seite_[(size_t)k] != q_seite_[(size_t)k + 1];
    if (neu && q_seite_[(size_t)k] == q_seite_[(size_t)k - 1]) ++r;
  }
  return r;
}

// Groups (source pages), title tokens and chunk tokens of a selection given
// most recent first; a group is a run of chunks of one page.
void CoderZ3::QuelleZaehlen(int n, const long long* a, const long long* b,
    unsigned long long* gruppen, unsigned long long* titel_tok,
    unsigned long long* chunk_tok) const {
  long long q_vor = -1;
  for (int k = n - 1; k >= 0; --k) {                // print order, oldest first
    const long long s = Quelle(a[k]);
    *chunk_tok += (unsigned long long)(b[k] - a[k]);
    if (s != q_vor) { ++*gruppen; *titel_tok += (unsigned long long)TitelKosten(s); }
    q_vor = s;
  }
}

static const unsigned char kRedir[3][4] =
    { { 0x0e, 0x06, 0xaf, 0x64 }, { 0x0e, 0x25, 0xaf, 0x64 }, { 0x0e, 0xaf, 0x64, 0x00 } };
static const int kRedirLen[3] = { 4, 4, 3 };

// [0x03] spelling of the redirect mark, blanks, the mention, ']', blanks
static bool RedirZeile(const Window& w, long long a, long long b, long long m0, long long m1) {
  long long q = a;
  if (q < m0 && w.At(q) == kTok(0x03)) ++q;
  bool hit = false;
  for (int f = 0; f < 3 && !hit; ++f) {   // longest first
    if (q + kRedirLen[f] > m0) continue;
    hit = true;
    for (int i = 0; i < kRedirLen[f] && hit; ++i) hit = w.At(q + i) == kRedir[f][i];
    if (hit) q += kRedirLen[f];
  }
  if (!hit) return false;
  for (; q < m0; ++q) if (w.At(q) != kSp) return false;
  return NurSchluss(w, m1, b);
}

// list markers ('*', '#', ':', ';') and blanks, the mention, ']', blanks
static bool Leer(const Window& w, long long a, long long b, long long m0, long long m1) {
  for (long long q = a; q < m0; ++q) {
    const unsigned char t = w.At(q);
    if (t != kSp && t != kTok(0x2A) && t != kRaute && t != kTok(0x4A) && t != kTok(0x4B)) return false;
  }
  return NurSchluss(w, m1, b);
}

// Whole neighbour lines around the one-line chunk [a, b) inside the body of
// source page s (behind its title line, in front of its separator): one line
// before, one line after, ..., at most two per side, no empty line, the chunk
// no longer than p_.chunk.
COLD_SIZE_ATTR void CoderZ3::NachbarnK(const Window& w, long long s, long long* pa, long long* pb) {
  constexpr int kNachbar = 2;
  const int tl = s_tl_[(size_t)s] & 0x7f;
  long long lo = (long long)s_ab_[(size_t)s] + (tl ? tl + 1 : 0);
  long long hi = ((size_t)s + 1 < s_ab_.size() ? (long long)s_ab_[(size_t)s + 1] : s_lauf_ab_)
      - transformer::SeparatorLength();
  if (lo < w.base) lo = w.base;                    // never binding: chunk << line scan
  if (hi > w.base + w.len) hi = w.base + w.len;
  long long a = *pa, b = *pb;
  int nv = 0, nn = 0;
  bool vor = true, after_at = true;
  while (vor || after_at) {
    if (vor) {
      if (nv >= kNachbar || a - 1 <= lo) {
        vor = false;
      } else {
        long long q = a - 2;
        while (q >= lo && w.At(q) != kParagraph) --q;
        if (q + 1 >= a - 1 || b - (q + 1) > p_.chunk) vor = false;
        else { a = q + 1; ++nv; }
      }
    }
    if (after_at) {
      if (nn >= kNachbar || b + 1 >= hi) {
        after_at = false;
      } else {
        long long q = b + 1;
        while (q < hi && w.At(q) != kParagraph) ++q;
        if (q <= b + 1 || q - a > p_.chunk) after_at = false;
        else { b = q; ++nn; }
      }
    }
  }
  if (nv + nn) {
  }
  *pa = a;
  *pb = b;
}

COLD_SIZE_ATTR int CoderZ3::WaehleK(int title_len, long long* taken_a, long long* taken_b) {
  const long long room = (long long)p_.budget - kEndLen;
  if (room <= 0) return 0;
  long long n_earlier = 0;
  for (size_t i = 0; i < ment_.size(); ++i) {
    if (ment_[i] + p_.open_len + title_len - kurz_[i] <= article_start_) ++n_earlier;
    else break;                                    // sorted ascending
  }
  const int cap = p_.max_mentions < 64 ? p_.max_mentions : 64;
  unsigned char inhalt[64][64];   // the chunks taken (a chunk is <= p_.chunk = 64 tokens)
  long long page_index[64];
  std::vector<unsigned char>& buf = scratch_;
  int n_taken = 0, gesehen = 0;
  long long used = 0;
  for (long long idx = n_earlier - 1; idx >= 0 && gesehen < 64 && n_taken < cap; --idx) {
    ++gesehen;
    const long long m0 = ment_[(size_t)idx];
    const long long m1 = m0 + p_.open_len + title_len - kurz_[(size_t)idx];
    long long lo = m0 - p_.line_scan;
    if (lo < 0) lo = 0;
    long long hi = m1 + p_.line_scan;
    if (hi > pos_) hi = pos_;
    if (hi > article_start_) hi = article_start_;
    if (hi < m1) hi = m1;
    buf.assign((size_t)(hi - lo), 0);
    ReadRange(lo, hi, buf.data());
    Window w;
    w.p = buf.data();
    w.base = lo;
    w.len = hi - lo;
    long long a = 0, b = 0;
    ChunkForMention(w, m0, m1, p_, article_start_, &a, &b);
    if (b > article_start_) return -1;             // causality guard
    const long long s = Quelle(a);
    if (s == 0) {                                  // the head of the stream, no page
      continue;
    }
    if (Leer(w, a, b, m0, m1)) {
      NachbarnK(w, s, &a, &b);
    }
    bool overlaps = false;
    for (int t = 0; t < n_taken; ++t) {
      if (a < taken_b[t] && taken_a[t] < b) { overlaps = true; break; }
    }
    if (overlaps) continue;
    if ((s_tl_[(size_t)s] & 0x80) && RedirZeile(w, a, b, m0, m1)) {
      continue;
    }
    bool gleich = false;
    for (int t = 0; t < n_taken && !gleich; ++t) {
      gleich = page_index[t] == s && taken_b[t] - taken_a[t] == b - a &&
          memcmp(inhalt[t], w.p + (a - w.base), (size_t)(b - a)) == 0;
    }
    if (gleich) {
      continue;
    }
    const long long cost = (b - a) + (n_taken ? 2 : 0);
    if (used + cost > room) break;
    memcpy(inhalt[n_taken], w.p + (a - w.base), (size_t)(b - a));
    page_index[n_taken] = s;
    taken_a[n_taken] = a;
    taken_b[n_taken] = b;
    ++n_taken;
    used += cost;
  }
  return n_taken;
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

}  // namespace tf_z3

