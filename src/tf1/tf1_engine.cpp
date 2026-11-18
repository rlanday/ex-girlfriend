// Coder-side driver of the transformer.  Parts follow
// fx2-cmix-transformer/src/predictor.cpp (Vladimir Ivanov, GPL-3); see
// tf1_engine.h.

#include "../meld.h"
#include "tf1_engine.h"
#include "../census_title.h"

#include "../alias_title_subst.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>

#ifdef __F16C__
#include <immintrin.h>
#else

#error "needs F16C (build with -march=x86-64-v3 or newer)."
#endif

#include "tf1_load_oz.h"
#include "../redirect_table.h"
#include "../alias_table.h"
#include "../rule_view.h"
#include "tf1_z3.h"      // link context of the current article
#include "tf1_prompt.h"  // prompt rows (article number, synonyms, headings)
#include "tf1_gate.h"    // gate zones

#include "model.h"          // fp32 reference engine (Ivanov, cpp_infer/src)
#include "opt/model_opt.h"  // optimized engine  (Ivanov, cpp_infer/src/opt)
static_assert(300 >= 0 && 300 <= 10000,
              "is the floor in units of 1e-10 (0 = 1e-6)");
// Floor of the transformer row handed to its slot.  The row arrives as float16
// values, and about half of its entries are 0 there (output below 3e-8).  The
// inherited floor 1e-6 lifted all of them; the transformer is calibrated down
// to its float16 resolution, so the floor is 3e-8.  The fast-weight row keeps
// 1e-6 (together with the 1e-6 clamp of the mixer inputs a lower floor there
// cost rate).

namespace transformer {

int g_b2i[256];
// fills g_b2i before main (number of set bytes below i = the rank of a set byte);
// the empty asm keeps the loop from being evaluated into .data
__attribute__((constructor, cold, minsize, noinline)) static void FuelleB2i() {
  int k = 0;
  for (int i = 0; i < 256; ++i) {
    __asm__("" ::: "memory");
    g_b2i[i] = InSet(i) ? k++ : k;
  }
}

namespace {
const size_t kHalfBufferSize = 1 << 22;  // 4M halfs = 8 MB per buffer
}  // namespace
#define GATE_ARG(x, k) , (x)

LOAD_PATH_ATTR void Fail(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  putc('\n', stderr);
  vfprintf(stderr, fmt, args);
  putc('\n', stderr);
  va_end(args);
  exit(1);
}

// ---------------------------------------------------------------------------
// float16, verbatim from fx2-cmix-transformer/src/predictor.cpp (GPL-3,
// Vladimir Ivanov).  Note: subnormal halves decode to HALF their true value
// in the scalar path; the F16C bulk path decodes exactly, so only the last
// n%8 elements of a conversion are affected.  This quirk is part of the
// probability rounding contract; do not "fix" it.
// ---------------------------------------------------------------------------
uint16_t FloatToHalf(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 112;  // half-biased exponent
  uint32_t mant = x & 0x7FFFFF;
  if (exp >= 31) return sign | 0x7C00;  // overflow, infinity and NaN
  if (exp <= 0) {  // subnormal half (or zero)
    if (exp < -10) return sign;
    mant |= 0x800000;
    int shift = 14 - exp;
    uint16_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return sign | h;
  }
  uint16_t h = sign | (exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;  // round to nearest even
  return h;
}

float HalfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {  // subnormal half: normalize
      int e = 0;
      while (!(mant & 0x400)) {
        mant <<= 1;
        ++e;
      }
      mant &= 0x3FF;
      x = sign | ((uint32_t)(112 - e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7F800000 | (mant << 13);
  } else {
    x = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}

void FloatsToHalves(const float* src, uint16_t* dst, size_t n) {
  size_t i = 0;
#ifdef __F16C__
  for (; i + 8 <= n; i += 8) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
        _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
  }
#endif
  for (; i < n; ++i) dst[i] = FloatToHalf(src[i]);
}

void HalvesToFloats(const uint16_t* src, float* dst, size_t n) {
  size_t i = 0;
#ifdef __F16C__
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
  }
#endif
  for (; i < n; ++i) dst[i] = HalfToFloat(src[i]);
}

// One out-of-line copy each of the float16 rounding of a full row and of the
// row floor: inlined into Step() and FwRow(), every call site carried its own
// fully unrolled loop.  Same conversions and comparisons as above.
static_assert(208 % 8 == 0, "a row is a whole number of 8-lane groups");
__attribute__((noinline)) static void RowToHalves(const float* src, uint16_t* dst) {
#ifdef __F16C__
#pragma nounroll
  for (int i = 0; i < 208; i += 8)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
        _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#else
  FloatsToHalves(src, dst, 208);
#endif
}
__attribute__((noinline)) static void RowToFloats(const uint16_t* src, float* dst) {
#ifdef __F16C__
#pragma nounroll
  for (int i = 0; i < 208; i += 8)
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
#else
  HalvesToFloats(src, dst, 208);
#endif
}
// entries below lo (and NaN) become lo
__attribute__((noinline)) static void RowFloor(float* p, float lo) {
  for (int i = 0; i < 208; ++i)
    if (!(p[i] >= lo)) p[i] = lo;
}

// ---------------------------------------------------------------------------
// The fixed engine opens no float16 sink (the prior and
// probability dumps exist only in the dump builds), but predictor.cpp
// still names Write() and the destructor of its always empty unique_ptr.  Traps
// instead of the bodies drop the out-of-line FloatsToHalves; a tool that builds
// such a sink in this form anyway stops at the constructor.
LOAD_PATH_ATTR HalfFileWriter::HalfFileWriter(const std::string& path) {
  Fail("float16 sink %s: not in the fixed-engine build", path.c_str());
}
HalfFileWriter::~HalfFileWriter() { __builtin_trap(); }
void HalfFileWriter::Write(const float*, size_t) { __builtin_trap(); }
void HalfFileWriter::WriteHalves(const uint16_t*, size_t) { __builtin_trap(); }
void HalfFileWriter::Flush() { __builtin_trap(); }

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
namespace {
unsigned char g_sep[64];
int g_sep_len = 0;
bool g_sep_ready = false;

LOAD_PATH_ATTR void InitSeparator() {
  if (g_sep_ready) return;
  g_sep_ready = true;
  static const unsigned char kStamped[8] = { 0x0b, 0x0b, 0x28, 0x1a, 0xaf, 0x68, 0x2a, 0x08 };
  g_sep_len = 8;
  memcpy(g_sep, kStamped, g_sep_len);
  // The separator is a stamped constant; an empty stamp stops the program.
  if (g_sep_len == 0) {
    Fail("the article separator is not stamped (src/tf1/tf1_sep_set.h) and "
        "is unset.  Derive it from OUR dictionary with "
        "tf1_dump --sep and stamp it -- never copy the upstream constant, it "
        "belongs to a different dictionary.");
  }
}
}  // namespace

const unsigned char* SeparatorIndices() { InitSeparator(); return g_sep; }
int SeparatorLength() { InitSeparator(); return g_sep_len; }

static const unsigned long long kMaxArticleTokens = 131072;
// Rotary-table limit: a document, prompt block included, never exceeds it.
static const unsigned long long kPieceGrenze =
    kMaxArticleTokens - (unsigned long long)68;
static_assert(131072 == 131072,
    "the training data loader cuts long articles at 2^17 tokens.  Changing "
    "the coder's value alone makes the two sides drift apart from the first "
    "long article on; change both and re-export the rotary table.");

// The rows of the engine live in fixed arrays of Impl, and
// Impl carries no member initialiser: `new Impl()` value-initialises it (all
// zero), the constructor sets the few non-zero starts and points the rows at
// their arrays.  In the submission form Impl has then neither a constructor
// nor a destructor, and the O3 instances of vector::_M_fill_assign are gone.
// A row is reached through a pointer as before (a fixed array address let
// clang unroll Step() by 2 KB more); data(), [] and swap() keep the call sites.
template <class T>
struct Tf1Row {
  T* p;
  T* data() { return p; }
  const T* data() const { return p; }
  T& operator[](int i) { return p[i]; }
  const T& operator[](int i) const { return p[i]; }
  void swap(Tf1Row& o) {
    T* t = p;
    p = o.p;
    o.p = t;
  }
};

struct Tf1Engine::Impl {
  fx2::opt::TransformerOpt* opt;

  int byte_to_index[256];
  int vocab_bytes[208];

  float probs_v[208];   // the arrays of the rows below (no alignas: an over-aligned Impl would need the aligned operator new)
  float tf_out_v[208];
  uint16_t half_v[208];
  Tf1Row<float> probs;   // V, the row handed to the byte mixer (1.0 from the ctor)
  Tf1Row<float> tf_out;  // V, raw transformer output
  Tf1Row<uint16_t> half;
  uint16_t ppmd_v[2][208];   // the two rows below point here
  Tf1Row<uint16_t> ppmd_cur;
  Tf1Row<uint16_t> ppmd_prev;
  bool ppmd_prev_gueltig;   // "valid": true from the second step on
  bool ppmd_erster_schritt;   // true from the ctor
  unsigned char window[64];
  int sep_len;
  unsigned long long article_tokens;
  bool doc_begun;
  bool ta_doc_auf;
  unsigned long long tokens;
  double loss_sum;
  bool print_loss;   // set in the ctor
  HalfFileWriter* probs_writer;
  unsigned long long piece_cuts;
  unsigned long long pre_verworfen_zone;

  enum { SG_VOR = 0, SG_STROM = 1, SG_NACH = 2, SG_Z3 = 3, SG_SEITE = 4,
         SG_ALIAS = 5, SG_N = 6 };

  int pk_prev1;
  int pk_prev2;
  unsigned long long pk_fed;
  unsigned long long doc_rows;
  unsigned long long max_doc_rows;
  unsigned long long pk_inserted;   // of them: rows the coder inserts
  unsigned long long pk_documents;
  long long pk_stream_pos;   // -1 from the ctor

  // Link context (z3).  Only its index is kept in RAM.
  tf_z3::CoderZ3* z3;
  // Prompt rows.  They need no stream storage: what they read fits into small
  // running buffers (title, link window, redirect window); RAM is spent only
  // on the title/alias table.
  tf_prompt::CoderPrompt* prompt;
  tf_gate::CoderGate* gate;
  bool gate_masked;
  unsigned long long alias_rows;
  unsigned long long kopierende_pk;
  bool in_redir_run;
  // fast weights after ex-girlfriend 1.0 (Ryan Landay,
  // GPL-3.0-only).  The target of a row is the next STREAM token past all
  // inserted rows.  fw_have: the head's scratch holds the second-head
  // prediction of a read row that has not had its target yet (a later read
  // row supersedes it).  fw_live: the slot row the coder uses for the next
  // stream token came from the second head (not neutral/off/PPMd).  The
  // update happens at the start of the next stream Step(), with that token.
  bool fw_have;
  bool fw_live;
  bool probs_from_tf;  // `probs` was taken from tf_out (a read row)
  unsigned long long fw_n_upd, fw_n_live;   // key figures
  float fw_row_v[208];
  uint16_t fw_half_v[208];
  Tf1Row<float> fw_row;   // 1.0 from the ctor
  Tf1Row<uint16_t> fw_half;
  __attribute__((noinline)) void Feed(int tok, const uint16_t* prior, bool inserted, float* out
            , bool gelesen
           ) {
    if (opt) opt->set_gelesen(gelesen);
    fx2::PaketChannels ch;
    ch.prev1 = pk_prev1;
    ch.prev2 = pk_prev2;
    ch.match_byte = inserted ? 208 : m_hot;
    ch.match_class = inserted ? 0 : (m_class & 15);
    opt->step((uint8_t)tok, prior, out, ch);
    pk_prev2 = pk_prev1;
    pk_prev1 = tok;
    ++doc_rows;
    if (doc_rows > max_doc_rows) max_doc_rows = doc_rows;
    if (inserted) {}
    if (gelesen) fw_have = true;
  }

  void NewDocument() {
    pk_prev1 = 216;
    pk_prev2 = 216;
    doc_rows = 0;
  }

  int m_hot;
  int m_class;
  int m_raw;
  bool m_have;
  int m_prev_hot;              // for the hit rate (-1 from the ctor)
  unsigned long long m_positions;
  unsigned long long m_sentinel;
  unsigned long long m_missing;
  unsigned long long m_hits;
  unsigned long long m_hit_base;
  unsigned long long m_class_hist[16];
};

LOAD_PATH_ATTR Tf1Engine::Tf1Engine(const std::vector<bool>& vocab, const char* weights_path)
    : impl_(new Impl()) {
  (void)vocab;   // See CheckVocabSubset
  for (int i = 0; i < 256; ++i) impl_->byte_to_index[i] = -1;
  for (int i = 0; i < 256; ++i) {
    if (InSet(i)) {
      impl_->byte_to_index[i] = g_b2i[i];
      impl_->vocab_bytes[g_b2i[i]] = i;
    }
  }
  // New Impl() zero-filled everything (tf_out, half, the
  // PPMD rows, fw_half, counters); the rows and the non-zero starts:
  impl_->probs.p = impl_->probs_v;
  impl_->tf_out.p = impl_->tf_out_v;
  impl_->half.p = impl_->half_v;
  for (int i = 0; i < 208; ++i) impl_->probs_v[i] = 1.0f;  // uniform for the first byte
  impl_->fw_row.p = impl_->fw_row_v;
  impl_->fw_half.p = impl_->fw_half_v;
  for (int i = 0; i < 208; ++i) impl_->fw_row_v[i] = 1.0f;
  impl_->ppmd_cur.p = impl_->ppmd_v[0];
  impl_->ppmd_prev.p = impl_->ppmd_v[1];
  impl_->ppmd_erster_schritt = true;
  impl_->pk_prev1 = 216;
  impl_->pk_prev2 = 216;
  impl_->pk_stream_pos = -1;
  impl_->m_hot = 208;
  impl_->m_prev_hot = -1;
  impl_->sep_len = SeparatorLength();
  if (impl_->sep_len > (int)sizeof(impl_->window)) {
    Fail("article separator longer than the match window");
  }
  // 0xFF is not a token index, so the window cannot match before sep_len
  // real tokens have been seen.
  memset(impl_->window, 0xFF, sizeof(impl_->window));

  // The coder always runs the optimized int8 engine; the fp32 reference path
  // is not used.
  const bool use_ref = false;
  impl_->opt = new fx2::opt::TransformerOpt(weights_path,
      fx2::opt::AttnKind::KVI8);
  {
    tf_z3::Params zp;
    impl_->z3 = new tf_z3::CoderZ3(zp);
  }
  {
    tf_prompt::Params pp;
    impl_->prompt = new tf_prompt::CoderPrompt(
        pp, tf_prompt::ArmsFromEnv());
  }
  {
    tf_gate::Params gp;
    impl_->gate = new tf_gate::CoderGate(gp);
  }
  impl_->print_loss = false;      // the coder prints no loss report
}

LOAD_PATH_ATTR Tf1Engine::~Tf1Engine() {
  // Loss report (off unless the constructor enables it).
  // Close the last article before the document-length check below.
  if (impl_->prompt) impl_->prompt->Finish();
  ReportChannels();
  delete impl_->z3;
  delete impl_->gate;
  delete impl_->prompt;
  delete impl_->opt;
  delete impl_;
}

const float* Tf1Engine::Step(const std::valarray<float>& ppmd,
    unsigned int byte) {
  Impl& s = *impl_;
  if (timestamp_skip::Laeuft()) return s.probs.data();
  if (alias_prompt::Laeuft()) {
    if (comment_tag::g_j < 2) byte = comment_tag::g_tf[comment_tag::g_j++];
    const int atok = s.byte_to_index[byte];
    if (atok < 0) {
      Fail("E208 %02x", byte);
    }
    // The PPMd row of the token that follows this one.  Only the LAST byte of
    // the block is followed by a real stream token, so only its row carries a
    // stream target; every other alias row gets prior0.
    for (int i = 0; i < 208; ++i) s.probs[i] = ppmd[s.vocab_bytes[i]];
    RowToHalves(s.probs.data(), s.half.data());
    const uint16_t* aprior = s.half.data();
    const bool kende = tf_copy_end::g_nach > 0
        && alias_prompt::g_rest == tf_copy_end::g_nach;
    const bool last_row = alias_prompt::Verbrauche();
    if (!last_row) {
      aprior = s.prompt->Prior0();
    }
    // The end mark row takes over the prior of this row (the PPMd row if this
    // was the last byte of the block); this row now predicts the end mark.
    const uint16_t* kprior = aprior;
    if (kende) aprior = s.prompt->Prior0();
    s.m_have = false;
    if (s.article_tokens == 0 && !s.ta_doc_auf
        && !s.doc_begun
        ) {
      if (s.opt) s.opt->begin_article();
      s.ta_doc_auf = true;
    }
    s.Feed(atok, aprior, /*inserted=*/true, s.tf_out.data()
           GATE_ARG(last_row && !kende, Impl::SG_ALIAS));
    if (kende) {
      s.Feed(208 + 6, kprior, /*inserted=*/true,
             s.tf_out.data() GATE_ARG(last_row, Impl::SG_ALIAS));
    }
    RowToHalves(s.tf_out.data(), s.half.data());
    RowToFloats(s.half.data(), s.probs.data());
    s.probs_from_tf = true;
    RowFloor(s.probs.data(), ((float)(300) * 1e-10f));
    return s.probs.data();
  }

  // (1) no loss is computed; only ReportLoss() would read it, and the coder
  //     does not print that report (print_loss is false), so loss_sum and
  //     tokens stay 0.

  // (2) the PPMd prior, rounded to float16 exactly as the training pipeline
  //     writes it.
  for (int i = 0; i < 208; ++i) s.probs[i] = ppmd[s.vocab_bytes[i]];
  RowToHalves(s.probs.data(), s.half.data());
  s.ppmd_prev.swap(s.ppmd_cur);
  s.ppmd_prev_gueltig = !s.ppmd_erster_schritt;
  s.ppmd_erster_schritt = false;
  memcpy(s.ppmd_cur.data(), s.half.data(),
      sizeof(uint16_t) * (size_t)208);

  int token = s.byte_to_index[byte];
  if (token < 0) {
    Fail("E209 %02x", byte);
  }
  // The fast-weight target is this stream token, if it was coded
  // with a second-head row.
  if (s.fw_have && s.fw_live && s.opt) {
    s.opt->fw_update((uint8_t)token);
  }
  s.fw_have = false;
  s.fw_live = false;
  const bool ta_tok = metablock::mk.ImTextanfang();
  if (!ta_tok) rule_view::Byte((unsigned char)byte);
  redirect_table::tab.Byte((unsigned char)byte);
  if (ta_tok) alias_table::tab.Pause();
  else
  alias_table::tab.Byte((unsigned char)byte);
  // Gate: the flag for this token was fixed after the previous one; the
  // automaton never looks forward.  feed = false means the transformer does
  // not get the token as input at all: no step, KV cache and context stay
  // untouched, and its position does not advance.  The training stream does
  // not carry those tokens either.
  const bool gate_feed = s.gate ? s.gate->Feeds() : true;
  {
    if (!s.m_have) { s.m_hot = 208; s.m_class = 0; s.m_raw = 0;
}
    if (s.m_prev_hot >= 0) {
      if (s.m_prev_hot == token) {}
    }
    s.m_prev_hot = s.m_hot;
    if (s.m_hot == 208) {}
    s.m_have = false;
  }
  memmove(s.window, s.window + 1, s.sep_len - 1);
  s.window[s.sep_len - 1] = (unsigned char)token;
  ++s.article_tokens;

  const bool at_separator =
      memcmp(s.window, SeparatorIndices(), s.sep_len) == 0;
  bool redir_merge = false;
  if (at_separator && s.prompt && s.prompt->InRedirect()) {
    redir_merge = true;
    s.prompt->CountMerge(!s.in_redir_run);
    s.in_redir_run = true;
  } else if (at_separator) {
    s.in_redir_run = false;
  }
  const bool doc_break = at_separator && !redir_merge;
    const unsigned long long piece_zaehler = s.doc_rows;
  bool teilb_docbreak = false;
  if (metablock::mk.SchliesstMit((unsigned char)byte) ||
      metablock::mk.TaEndmarke((unsigned char)byte))
    teilb_docbreak = true;
  bool piece_faellig = piece_zaehler >= kPieceGrenze;
  const bool last_of_piece = doc_break || teilb_docbreak || piece_faellig;
  const bool piece_cut = last_of_piece && !doc_break && !teilb_docbreak;
  if (piece_cut) {}
  if (!ta_tok)
  if (rule_view::z3_orig) {
    const unsigned char* fr = nullptr;
    const int n = rule_view::Frei(&fr);
    for (int k = 0; k < n; ++k)
      s.z3->Observe((unsigned char)s.byte_to_index[fr[k]], k + 1 == n && doc_break);
  } else
  {
    const unsigned char* alt = nullptr;
    const int n = redirect_table::tab.Alt(&alt);
    for (int k = 0; k < n; ++k)
      s.z3->Observe((unsigned char)s.byte_to_index[alt[k]],
                    k + 1 == n && doc_break);
    if (redirect_table::tab.FromBytes() && byte == 0x0a) rule_view::z3_orig = true;
  }
  if (at_separator) s.z3->PageEnd(s.prompt->InRedirect());
  // Show the coded token to the gate automaton (same sequence on both sides).
  // After this call the automaton answers for the next token.
  if (s.gate) {
    s.gate->Observe((unsigned char)token);
    s.gate_masked = s.gate->Gated();
  }
  // Show the coded token to the prompt automaton (same sequence on both
  // sides).  As for z3, the narrow separator condition, not `last_of_piece`.
  s.prompt->Observe((unsigned char)token, doc_break);
  if (redir_merge) s.prompt->PageBoundary();
  if (piece_cut && !ta_tok) s.prompt->PieceCut();
  if (last_of_piece) {
    // The next token starts a fresh context; its distribution is PPMd's.
    RowToFloats(s.half.data(), s.probs.data());
    s.probs_from_tf = false;
    s.article_tokens = 0;
    s.NewDocument();
    if (s.opt) s.opt->begin_article();
    s.doc_begun = true;
  } else {
    if (s.article_tokens == 1) {
      const bool ta_schon = s.ta_doc_auf;
      s.ta_doc_auf = false;
      const bool seen_list = s.doc_begun;
      s.doc_begun = false;
      if (!seen_list)
      if (!ta_schon)
      {
      if (s.opt) s.opt->begin_article();
      }
    }
    // Pre block: fed before this token's model step.  That puts the rows that
    // precede the title (page metadata, or the article number in the form that
    // has it there) before the title.  These rows bypass the separator
    // detection.
    // vorblock_n (pre-block size) is needed by the guard below.
    int vorblock_n = 0;
    {
      const unsigned char* ptoks = nullptr;
      int pn = 0;
      if (s.prompt->TakePre(&ptoks, &pn)) {
        vorblock_n = pn;
        if (68 > 0
            && (unsigned long long)pn
               > (unsigned long long)68) {
          fprintf(stderr, "\nE210 %d %d\n",
              pn, (int)68);
          fflush(nullptr);
          exit(1);
        }
        bool feed_pre = true;
        feed_pre = gate_feed;
        if (feed_pre) {
          const uint16_t* p0 = s.prompt->Prior0();
          for (int i = 0; i < pn; ++i) {
            const uint16_t* pz = (i + 1 == pn && s.ppmd_prev_gueltig)
                ? s.ppmd_prev.data() : p0;
            s.Feed(ptoks[i], pz, /*inserted=*/true, s.tf_out.data()
                   GATE_ARG(false, Impl::SG_VOR));
          }
          s.prompt->AddFed((unsigned long long)pn);
        } else {
        }
      }
    }
    // The gating step.  No `step()`: the KV cache, the context position and
    // every internal state stay as they were.  `s.tf_out` keeps the row of the
    // last executed step, so after the zone the transformer predicts from a
    // context that never contained it.
    if (!gate_feed) {
      s.gate->CountStepSaved();
    } else
    {
      const uint16_t* pstrom = s.half.data();
      bool block_folgt = s.prompt->HasPost();
      block_folgt = block_folgt ||
          (!s.prompt->PriorNext() && s.z3->HasPrefix());
      if (block_folgt) pstrom = s.prompt->Prior0();
      bool strom_gelesen = true;
      strom_gelesen = !block_folgt;
      s.Feed(token, pstrom, /*inserted=*/false, s.tf_out.data()
             GATE_ARG(strom_gelesen, Impl::SG_STROM));
    }
    bool z3_folgt = false;
    z3_folgt = !s.prompt->PriorNext() && s.z3->HasPrefix();
    {
      const unsigned char* ptoks = nullptr;
      int pn = 0;
      if (s.prompt->TakePost(&ptoks, &pn)) {
        // A post block belongs behind the token of this step, a pre block
        // before the token of the next one, so if both occur in one cycle the
        // post block would have to go first; this code feeds them the other
        // way round.  The case cannot occur with this grammar (a heading
        // reminder needs `=` at the previous position, a sentinel row a digit;
        // the rows before the title sit on the first token, the prompt on the
        // title end, and no sentinel row is fed inside a title), so it gets a guard
        // instead of an assumption.
        if (vorblock_n > 0) {
          Fail("E211 %d %d", vorblock_n, pn);
        }
        const uint16_t* p0 = s.prompt->Prior0();
        for (int i = 0; i < pn; ++i) {
          const uint16_t* pz =
              (i + 1 == pn && !z3_folgt) ? s.half.data() : p0;
          s.Feed(ptoks[i], pz, /*inserted=*/true, s.tf_out.data()
                 GATE_ARG(i + 1 == pn && !z3_folgt, Impl::SG_NACH));
        }
        s.prompt->AddFed((unsigned long long)pn);
      }
    }
    // Title mask and sentinel gating: the distribution for the next token
    // comes from the PPMd prior instead of from the transformer.  `s.half`
    // carries that prior, the same row as in the `last_of_piece` branch.
    if (s.prompt->PriorNext()) {
      RowToFloats(s.half.data(), s.probs.data());
      s.probs_from_tf = false;
    } else {
    // Link-context prefix.  It ends on the separator sequence, the reset
    // trigger, so it is fed directly through `step()`: `window` and
    // `article_tokens` are not touched and no reset can fire.  The output of
    // the last prefix step is the distribution for the first body byte, hence
    // the block stands before the conversion into `probs`, and after the post
    // block above (training document order: article number, synonyms, link
    // context).
    {
      const unsigned char* ptoks = nullptr;
      int pn = 0;
      if (s.z3->TakePrefix(&ptoks, &pn)) {
        const uint16_t* p0 = s.z3->Prior0();
        for (int i = 0; i < pn; ++i) {
          const uint16_t* pz = (i + 1 == pn) ? s.half.data() : p0;
          s.Feed(ptoks[i], pz, /*inserted=*/true, s.tf_out.data()
                 GATE_ARG(i + 1 == pn, Impl::SG_Z3));
        }
        s.z3->AddFed((unsigned long long)pn);
      }
    }
    RowToHalves(s.tf_out.data(), s.half.data());
    RowToFloats(s.half.data(), s.probs.data());
    s.probs_from_tf = true;
    }
  }
  if (at_separator) {
    const unsigned char* vt = nullptr;
    int vn = 0, mv = 0, mb = 0;
    if (s.prompt->TakeSeitenVorblock(&vt, &vn, &mv, &mb)) {
      bool vfeed = true;
      if (s.gate) vfeed = s.gate->Feeds();
      if (vfeed) {
        const uint16_t* p0 = s.prompt->Prior0();
        for (int i = 0; i < vn; ++i) {
          const uint16_t* pz = (i + 1 == vn) ? s.ppmd_cur.data() : p0;
          s.Feed(vt[i], pz, /*inserted=*/true, s.tf_out.data()
                 GATE_ARG(i + 1 == vn, Impl::SG_SEITE));
        }
        s.prompt->AddFed((unsigned long long)vn);
        RowToHalves(s.tf_out.data(), s.half.data());
        RowToFloats(s.half.data(), s.probs.data());
        s.probs_from_tf = true;
      } else {
      }
    }
  }
  RowFloor(s.probs.data(), ((float)(300) * 1e-10f));
  return s.probs.data();
}

void Tf1Engine::SetMatch(int hot_byte, int raw_len_bits) {
  Impl& s = *impl_;
  s.m_hot = (hot_byte >= 0 && hot_byte < 256 && InSet(hot_byte))
      ? g_b2i[hot_byte] : 208;
  if (raw_len_bits < 0) raw_len_bits = 0;
  if (raw_len_bits > 255) raw_len_bits = 255;
  s.m_raw = raw_len_bits;
  s.m_class = MatchLengthClass(raw_len_bits);
  s.m_have = true;
}

int Tf1Engine::ZwangByte() const { return impl_->prompt ? impl_->prompt->ZwangByte() : -1; }

bool Tf1Engine::GateMasked() const { return impl_->gate_masked; }

// fast weights after ex-girlfriend 1.0 (Ryan Landay,
// GPL-3.0-only): see tf1_engine.h.
const float* Tf1Engine::FwRow() {
  Impl& s = *impl_;
  if (s.probs_from_tf && s.opt) {
    // the same float16 rounding and 1e-6 floor as the transformer row
    RowToHalves(s.opt->fw_probs(), s.fw_half.data());
    RowToFloats(s.fw_half.data(), s.fw_row.data());
    RowFloor(s.fw_row.data(), 1e-6f);
    s.fw_live = true;
  } else {
    memcpy(s.fw_row.data(), s.probs.data(), sizeof(float) * 208);
    s.fw_live = false;
  }
  return s.fw_row.data();
}

void Tf1Engine::FwNeutral() { impl_->fw_live = false; }

// Pure observation, see tf1_engine.h.
const float* Tf1Engine::LastHidden() const {
  return impl_->opt ? impl_->opt->last_hidden() : nullptr;
}
int Tf1Engine::HiddenSize() const {
  return impl_->opt ? fx2::opt::TransformerOpt::hidden_size() : 0;
}
const float* Tf1Engine::LastKey() const {
  return impl_->opt ? impl_->opt->last_key() : nullptr;
}

void Tf1Engine::ReportChannels() const {
  if (impl_->max_doc_rows > kMaxArticleTokens) {
    __builtin_unreachable()
;
    fflush(nullptr);
    exit(1);
  }
}

void Tf1Engine::ReportLoss() const {
  const Impl& s = *impl_;
  if (s.tokens == 0) return;
  fprintf(stderr, "\r%*s\r", 70, "");
  printf("E213 %.6f %llu\n",
      s.loss_sum / s.tokens, s.tokens);
  fflush(stdout);
}

}  // namespace transformer

