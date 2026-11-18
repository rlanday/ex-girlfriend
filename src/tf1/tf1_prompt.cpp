
#include "../meld.h"
#include "../u8_append.h"
#include "tf1_prompt.h"
#include "../census_title.h"
#include "../census_block.h"

#include "../cold_size.h"
#include "../alias_substring_dedup.h"

#include "tf1_prior0.h"   // the stamped prior0 table
#include "../rule_view.h"
#include "../computed_values.h"
static_assert(!1 || tf_z3::kEndeKlasse != 7,
    "the copy marker class is the end row of the link context.");
static_assert(!6 || tf_z3::kEndeKlasse != 6,
    "the end mark class is the end row of the link context; take another class.");
#include "../metablock.h"

#include <stdlib.h>
#include <string.h>

namespace tf_prompt {

// ---- grammar --------------------------------------------------------------
static const unsigned char kTitleEnd[1] =
    { 0x08 };
const unsigned char* TitleEnd() { return kTitleEnd; }
int TitleEndLen() { return 1; }
// These characters stand as BYTES in the stream (the WRT stage leaves them
// alone); their token index therefore follows directly from the frozen byte
// set and cannot drift apart from a hand-written constant.
static unsigned char ByteTok(int b) {
  if (!transformer::InSet(b)) {
    __builtin_unreachable();
  }
  return (unsigned char)transformer::g_b2i[b];
}
// The fixed bytes of the rows are in the frozen set and the ten digits have
// consecutive tokens (the set is ordered by byte value): checked here at
// compile time instead of at every row.  The line feed of a row given as
// bytes is the paragraph token.
constexpr unsigned char kTokHash = (unsigned char)transformer::kByteToIndex[0x23];
constexpr unsigned char kTokNull = (unsigned char)transformer::kByteToIndex[0x30];
constexpr bool ZiffernFolgen() {
  for (int d = 0; d < 10; ++d)
    if (!transformer::InSet(0x30 + d) || transformer::kByteToIndex[0x30 + d] != kTokNull + d) return false;
  return true;
}
static_assert(transformer::InSet(0x23) && ZiffernFolgen(),
    "prompt: '#' or a digit is not where the rows expect it in the 208-byte set");
static_assert(transformer::InSet(0x0A) && transformer::kByteToIndex[0x0A] == kParagraph,
    "prompt: the line feed is not the paragraph token");
unsigned char LinkClose() { return ByteTok(0x5D); }
unsigned char Hash() { return kTokHash; }
unsigned char Digit(int d) { return (unsigned char)(kTokNull + d); }

// Token index -> byte value.  The training side (zctx/zgate.py) computes on
// bytes (`b = bitmap[tokens]`), so the gate automaton needs the same table.
static const unsigned char* VocabBytes() {
  static unsigned char tab[208];
  static bool ready = false;
  if (!ready) {
    int k = 0;
    for (int b = 0; b < 256; ++b) {
      if (transformer::InSet(b)) tab[k++] = (unsigned char)b;
    }
    if (k != 208) {
      __builtin_unreachable()
;
    }
    ready = true;
  }
  return tab;
}

// "#<n>" as a token sequence, no new symbol (the training side writes the
// same rows).
static void NumberTokens(long long n, std::vector<unsigned char>* out) {
  out->push_back(Hash());
  unsigned char buf[24];
  int k = 0;
  if (n <= 0) {
    buf[k++] = 0;
  } else {
    long long v = n;
    while (v > 0) { buf[k++] = (unsigned char)(v % 10); v /= 10; }
  }
  for (int i = k - 1; i >= 0; --i) out->push_back(Digit(buf[i]));
}

// Despite the name nothing is read from the environment: the prompt form is
// fixed at build time.
COLD_SIZE_ATTR Arms ArmsFromEnv() { return FesteArme(); }

// ---- CoderPrompt ----------------------------------------------------------
COLD_SIZE_ATTR CoderPrompt::CoderPrompt(const Params& p, const Arms& a) : p_(p), a_(a) {
  memset(tail_, 0xFF, sizeof(tail_));
  prior0_given_ = true;
  VocabBytes();
  StartArticle();
  article_ = 0;
  article_start_ = 0;
// Self-report counters only
}

COLD_SIZE_ATTR CoderPrompt::~CoderPrompt() {
}

void CoderPrompt::EmitBlock(int slot, const unsigned char* p, int n) {
  (void)slot; (void)p; (void)n;
}

COLD_SIZE_ATTR void CoderPrompt::StartArticle() {
  in_title_ = true;
  title_len_ = 0;
  title_.clear();
  a2_count_ = 0;
  redir_now_ = false;
  no_prompt_ = false;
  rd_open_ = 0;
  rd_take_ = -1;
}

// A page starts at pos_ (a separator, or a page boundary inside a merged
// redirect run): page counter, article start and the resets of StartArticle().
COLD_SIZE_ATTR void CoderPrompt::NeueSeite() {
  ++page_;
  article_start_ = pos_;
  StartArticle();
}

// The pre block of this step (rows in front of the token) and its slices.
void CoderPrompt::LeerePre() {
  pre_.clear();
  have_pre_ = false;
  meta_pre_von_ = 0;
  meta_pre_bis_ = 0;
  rd_pre_von_ = 0;
  rd_pre_bis_ = 0;
}

COLD_SIZE_ATTR void CoderPrompt::EndArticle() {
  redir_last_ = redir_now_;
}

COLD_SIZE_ATTR void CoderPrompt::Finish() {
  // If the stream ends exactly on a separator sequence, the coder has opened
  // an empty trailing piece.  The training package does not have it, so it
  // gets no row.
  if (pos_ == article_start_ && article_ > 0) {
    return;
  }
  EndArticle();
}

// ---- the prompt behind `</title>` ----------------------------------------
// One prompt row in front of the title that the dump books as a `#rd` row
// (the META slice starts with it): `#rd`, the census context row, the block
// mark.  Given as bytes; its line feed is the paragraph token.
void CoderPrompt::RdZeile(const unsigned char* p, int n) {
  rd_pre_von_ = (int)pre_.size();
  for (int i = 0; i < n; ++i) pre_.push_back(ByteTok(p[i]));
  rd_pre_bis_ = (int)pre_.size();
  have_pre_ = true;
}

static void Tokens(const unsigned char* p, int n, std::vector<unsigned char>* out) {
  for (int i = 0; i < n; ++i) {
    const unsigned char c = p[i];
    if (!transformer::InSet(c)) continue;
    out->push_back((unsigned char)transformer::g_b2i[c]);
  }
}

void CoderPrompt::BuildMeta(long long page_index, std::vector<unsigned char>* out_bytes) {
  out_bytes->clear();
  std::vector<unsigned char>& b = *out_bytes;
  static const int kFeld[3] = {1,
                               1,
                               1};
  static const int kOrdnung[4] = {3, 0, 1, 2};
  const int n_ordnung = 4;
  unsigned char kbuf[24];
  int n_roh = 0;
  const unsigned char* roh = nullptr;
  bool eintrag = false;
  unsigned long long tok[4] = {0, 0, 0, 0};
  for (int oi = 0; oi < n_ordnung; ++oi) {
    const int fnr = kOrdnung[oi];
    if (fnr < 3 && !kFeld[fnr]) continue;
    if (fnr == 3) {
      long long v = page_index + (0);
      int nz = 0;
      do { kbuf[nz++] = (unsigned char)(0x30 + (int)(v % 10)); v /= 10; }
      while (v);
      for (int q = 0; q < nz / 2; ++q) {
        const unsigned char t = kbuf[q];
        kbuf[q] = kbuf[nz - 1 - q];
        kbuf[nz - 1 - q] = t;
      }
      roh = kbuf;
      n_roh = nz;
      eintrag = true;
    } else {
      roh = author_block::aut.Feld(page_index + (0), fnr,
          &n_roh);
      if (!roh) continue;                       // page not in the table
      eintrag = true;
      static std::string pw;
      if (fnr == 0) {
        size_t n1 = 0, n2 = 0;
        const char* w1 = census_block::CensWert(page_index + (0), census_block::zahl::F_POP, &n1);
        if (!w1)   // a hand-edited census sentence: the population of the intro sentence
          w1 = census_block::CensWert(page_index + (0), census_block::zahl::F_POP_I, &n1);
        const char* w2 = census_block::CensWert(page_index + (0), census_block::zahl::F_TOT_KM, &n2);
        if (w1 || w2) {
          pw.clear();
          if (w1) pw.append(w1, n1);
          if (w1 && w2) pw.push_back(' ');
          if (w2) pw.append(w2, n2);
          roh = (const unsigned char*)pw.data();
          n_roh = (int)pw.size();
        }
      }
    }
    if (n_roh == 0) {continue; }
    // Byte-set watch: every byte of the row must live in the frozen 208 set,
    // otherwise there is no token for it.  A byte outside it is DROPPED and
    // counted -- not a Fail: the set is frozen against the WRT stream, and a
    // single exotic contributor name must not kill a twenty-hour dump run.
    std::vector<unsigned char> z;
    Tokens(roh, n_roh, &z);
    if (z.empty()) {continue; }
    z.push_back(kParagraph);                    // the line-end token
    // The cap is on the BLOCK, not the row: a row that would break it is
    // dropped WHOLE, because a truncated author name is a different name.
    if ((int)b.size() + (int)z.size() > 64) {
      continue;
    }
    tok[fnr] = z.size();
    u8_append::Anh(b, z.begin(), z.end());
  }
  if (!eintrag) {}
}

COLD_SIZE_ATTR void CoderPrompt::BuildPrompt() {
  post_.clear();
  const int sep_len = transformer::SeparatorLength();
  const unsigned char* sep = transformer::SeparatorIndices();

  if (PROMPT_ARM(z1t)) {                                  // article number
    std::vector<unsigned char> b;
    NumberTokens(article_, &b);
    for (int i = 0; i < sep_len; ++i) b.push_back(sep[i]);
    EmitBlock(1, b.data(), (int)b.size());
    u8_append::Anh(post_, b.begin(), b.end());
  }

  meta_von_ = (int)post_.size();
  meta_bis_ = meta_von_;

  alias_table::Tab& tab = alias_table::tab;
  const unsigned char* tpo = nullptr;
  const int tlo = rule_view::Titel(&tpo);
  std::vector<unsigned char> tb; u8_append::Anh(tb, tpo, tpo + (tlo > 0 ? tlo : 0));
  for (unsigned char c : census_title::zusatz) tb.push_back(c);
  if (PROMPT_ARM(syn)) {                                  // synonyms
    // A title longer than kMaxTitleTokens is not in the redirect table of the
    // training side, so it gets no synonym block.
    std::vector<alias_table::Ref> q;
    if ((int)tb.size() + TitleEndLen() <= kMaxTitleTokens) {   // the full title
      const alias_table::Wahl w = {PROMPT_PAR(syn_hops), kMaxTitleTokens - TitleEndLen(), &doc_,
          false, &alias_substring_dedup::g_syn_ges, &alias_substring_dedup::g_syn};
      tab.Kandidaten(tb.data(), (int)tb.size(), w, &q);
    }
    std::vector<unsigned char> b;
    int summe = 0;
    const int take = (int)q.size() < PROMPT_PAR(syn_max) ? (int)q.size() : PROMPT_PAR(syn_max);
    alias_table::Syn().Neu(alias_table::SchluesselHash(tb.data(), (int)tb.size()));
    for (int i = 0; i < take; ++i) {
      const int n = q[(size_t)i].len;
      if (summe + n + TitleEndLen() > PROMPT_PAR(syn_tokens)) break;
      const unsigned char* d = tab.Data(q[(size_t)i]);
      for (int k = 0; k < n; ++k) b.push_back((unsigned char)transformer::g_b2i[d[k]]);
      u8_append::Anh(b, TitleEnd(), TitleEnd() + TitleEndLen());
      summe += n + TitleEndLen();
      alias_table::Syn().Merke(tab.K(q[(size_t)i]));
    }
    EmitBlock(2, b.data(), (int)b.size());
    if (!b.empty()) {}
    u8_append::Anh(post_, b.begin(), b.end());
  }
  if (PROMPT_ARM(alias)) {                                // aliases
    std::vector<alias_table::Ref> id_list;
    const alias_table::Wahl w = {PROMPT_PAR(alias_hops), 0, &doc_, true,
        &alias_substring_dedup::g_p1_ges, &alias_substring_dedup::g_p1};
    tab.Kandidaten(tb.data(), (int)tb.size(), w, &id_list);
    std::vector<unsigned char> b;
    for (size_t i = 0; i < id_list.size(); ++i) {
      const int n = id_list[i].len;
      if ((int)b.size() + 1 + n > PROMPT_PAR(alias_cap)) break;
      b.push_back(kParagraph);
      const unsigned char* d = tab.Data(id_list[i]);
      for (int k = 0; k < n; ++k) b.push_back((unsigned char)transformer::g_b2i[d[k]]);
    }
    EmitBlock(3, b.data(), (int)b.size());
    if (!b.empty()) {}
    u8_append::Anh(post_, b.begin(), b.end());
  }
  {
    std::string name, raw;
    for (int i = 0; i < title_len_ - TitleEndLen(); ++i) name.push_back((char)VocabBytes()[title_[(size_t)i]]);
    if (census_block::K1Text(page_ + (0), name, &raw)) {
      std::vector<unsigned char> b(1, kParagraph);
      Tokens((const unsigned char*)raw.data(), (int)raw.size(), &b);   // as META
      if ((int)b.size() > 512) b.resize((size_t)512);
      EmitBlock(3, b.data(), (int)b.size());
      u8_append::Anh(post_, b.begin(), b.end());
      ++census_title::n_k1;
    }
  }
  if (!post_.empty()) have_post_ = true;
}

void CoderPrompt::TitleClosed(int title_len) {
  // Check: `title_len_` indexes `title_` in the heading-reminder rows.  The
  // two bounds in Observe() are one apart on purpose; changing one without
  // the other stops here instead of reading past the end.
  if (title_len > (int)title_.size()) {
    __builtin_unreachable()
;
  }
  title_len_ = title_len;
  if (!no_prompt_) BuildPrompt();
}

void CoderPrompt::ScanRedirectPage(unsigned char token) {
  if (rd_take_ > 0) {
    --rd_take_;
    if (rd_take_ == 0) {
      rd_take_ = -1;
      const redirect_table::Rec* rr = redirect_table::tab.Seite(page_);
      if (rr && rr->marke) {
        redir_now_ = true;
      }
    }
    return;
  }
  if (rd_take_ == 0 || redir_now_) return;
  if (rd_open_ == 0 && token == kParagraph) {
    rd_open_ = 1;
    rd_take_ = kRedirMaxLen;
  }
}

void CoderPrompt::PageBoundary() {
  redir_last_ = redir_now_;
  NeueSeite();         // the resets of StartArticle(), then no prompt in the run
  no_prompt_ = true;
  prior_next_ = PROMPT_ARM(meta) && !1;
}

// ---- Observe --------------------------------------------------------------
void CoderPrompt::Observe(unsigned char token, bool piece_end) {
  if (metablock::mk.ImTextanfang()) {
    (void)piece_end;
    prior_next_ = false;
    have_post_ = false;
    LeerePre();
    ++st_ta_;
    return;
  }
  for (int i = 0; i + 1 < (int)sizeof(tail_); ++i) tail_[i] = tail_[i + 1];
  tail_[sizeof(tail_) - 1] = token;

  // Preamble of the stream head: no page, no title, no prompt rows.  The ring
  // and the tail above are updated, because they are the coded stream.  The
  // rule below is the tab rule of an earlier block form; it ends inside the
  // metadata block, where the gate withholds every row, so it has no effect
  // on the output.
  if (g1_preamble_) {
    bool ende = false;
    {
      static const unsigned char kTabTok = transformer::kByteToIndex[0x09];
      if (token == kTabTok) mf_tab_ = true;
      if (token == kParagraph) {
        if (mf_tab_) mf_auf_ = true;
        else if (mf_auf_) ende = true;
        mf_tab_ = false;
      }
    }
    ++pos_;
    prior_next_ = false;
    have_post_ = false;
    LeerePre();
    if (ende) {
      g1_preamble_ = false;
      article_start_ = pos_;
      in_title_ = false;
      title_len_ = 0;
      title_.clear();
      rd_open_ = 1;
      rd_take_ = kRedirMaxLen;
      redir_now_ = false;
    }
    return;
  }

  const unsigned char bv = VocabBytes()[token];

  while (doc_.size() <= alias_table::tab.Quellen_()) doc_.push_back((uint32_t)article_);
  ScanRedirectPage(token);

  have_post_ = false;
  LeerePre();
  const long long q = pos_;
  ++pos_;

  if (piece_end) {
    EndArticle();
    ++article_;
    NeueSeite();
    prior_next_ = PROMPT_ARM(meta) && !1;
    piece_rest_ = false;       // the next article is whole again
    return;
  }

  // --- article number before the title (number-first form only) ------------
  if (PROMPT_ARM(num) && q == article_start_) {
    std::vector<unsigned char> b;
    NumberTokens(article_, &b);
    EmitBlock(0, b.data(), (int)b.size());
    pre_.insert(pre_.begin(), b.begin(), b.end());
    have_pre_ = true;
  }

  // --- title ---------------------------------------------------------------
  if (in_title_) {
    const long long len_here = q + 1 - article_start_;
    // The store must cover every value `title_len_` can take.  The give-up
    // branch below fires at `len_here > kMaxTitleTokens + k`, and the
    // terminator is checked first, so a title can still close at
    // `len_here == kMaxTitleTokens + k + 1` (= 70).  A read past the end in
    // the heading-reminder rows could differ between encoder and decoder
    // (two processes, different allocation histories) and break decoding.
    if ((int)title_.size() < kMaxTitleTokens + TitleEndLen() + 1) {
      title_.push_back(token);
    }
    const int k = TitleEndLen();
    if (len_here >= k
        && memcmp(tail_ + sizeof(tail_) - k, TitleEnd(), (size_t)k) == 0) {
      in_title_ = false;
      if ((int)(len_here - k) >= kMinTitleTokens) TitleClosed((int)len_here);
    } else if (len_here > kMaxTitleTokens + k) {
      in_title_ = false;
    }
  }

  // Heading reminder (a2): at a `==` section heading the running heading
  // number (a2n) and the article title are fed again.
  const bool a2_hit = (token == kParagraph && q >= 2
      && tail_[sizeof(tail_) - 2] == kEq && tail_[sizeof(tail_) - 3] == kEq);
  if (a2_hit && title_len_ == 0 && !in_title_) {}
  if (title_len_ > 0 && a2_hit) {
    const long long site = q + 1;                // as the training side places it
    if (piece_rest_) {}
    if (PROMPT_ARM(a2) && !piece_rest_ && !no_prompt_
        && site >= article_start_ + title_len_) {
      ++a2_count_;
      std::vector<unsigned char> b;
      if (PROMPT_ARM(a2n)) {
        std::vector<unsigned char> nb;
        NumberTokens(a2_count_, &nb);
        EmitBlock(4, nb.data(), (int)nb.size());
        u8_append::Anh(b, nb.begin(), nb.end());
      }
      u8_append::Anh(b, title_.begin(), title_.begin() + title_len_);
      if (have_post_) {
        transformer::Fail("E225 %lld %d", q, (int)post_.size());
      }
      post_.swap(b);
      meta_von_ = 0;
      meta_bis_ = 0;
      have_post_ = true;
    }
  }

  {
    static long long rech_seite = -1;
    static bool rech_zensus = false;
    if (article_start_ != rech_seite) {
      rech_seite = article_start_;
      rech_zensus = census_title::Kontext(page_ + (0)) != nullptr;
    }
    bool rech_frei = !rech_zensus;
    // census form: the pseudo article and the census line stand between the
    // metablock and page 0, and this automaton sees them as page text; the
    // unit closes behind the census line (fed after this step, so the line
    // feed that closes it is still outside)
    rech_frei = rech_frei && metablock::mk.Fertig();
    const unsigned char* rb = nullptr;
    const int rn = computed_values::aut.Byte(bv, article_start_,
        rech_frei && !in_title_ && !no_prompt_ && !have_pre_ && !have_post_, &rb);
    if (rn) {
      post_.assign(1, (unsigned char)(208 + computed_values::kMarke));
      for (int i = 0; i < rn; ++i) post_.push_back(ByteTok(rb[i]));
      post_.push_back((unsigned char)(208 + 6));
      meta_von_ = 0;   // no META rows in this block (see the A2 branch)
      meta_bis_ = 0;
      have_post_ = true;
    }
  }

  // --- the distribution for the next token --------------------------------
  prior_next_ = false;
  if (PROMPT_ARM(titelmask) && in_title_ && !no_prompt_) {
    prior_next_ = true;
  }
  if (prior_next_) {}
}

void CoderPrompt::SeitenVorblock(long long q) {
  if (ImRedirectBlock() && q == article_start_) {
    static const unsigned char kRd[4] = {0x23, 0x72, 0x64, 0x0A};   // "#rd\n"
    RdZeile(kRd, 4);
  }
  if (q == article_start_) CensusZeile();
  if (!ImRedirectBlock() && q == article_start_) BlockZeile();

  if (PROMPT_ARM(meta) && q == article_start_) {
    std::vector<unsigned char> b;
    BuildMeta(page_, &b);
    if (!b.empty()) {
      EmitBlock(5, b.data(), (int)b.size());
      meta_pre_von_ = rd_pre_bis_ > 0 ? rd_pre_von_ : (int)pre_.size();
      u8_append::Anh(pre_, b.begin(), b.end());
      meta_pre_bis_ = (int)pre_.size();
      have_pre_ = true;
    }
  }

}

bool CoderPrompt::TakeSeitenVorblock(const unsigned char** toks, int* n,
    int* meta_von, int* meta_bis) {
  if (!PROMPT_ARM(meta) || pos_ != article_start_) return false;
  LeerePre();
  SeitenVorblock(pos_);
  if (!have_pre_) return false;
  have_pre_ = false;
  *toks = pre_.data();
  *n = (int)pre_.size();
  *meta_von = meta_pre_von_;
  *meta_bis = meta_pre_bis_;
  return true;
}

void CoderPrompt::CensusZeile() {
  const std::string* z = census_title::Kontext(page_ + (0));
  if (!z) return;
  RdZeile((const unsigned char*)z->data(), (int)z->size());
  ++census_title::n_prompt;
}

#define PROMPT_BLOCK_N 193
#define PROMPT_BLOCK_WORD_MAX 5
#define PROMPT_BLOCK_FROM { 71184, 72268, 72964, 73572, 74042, 75025, 75371, 75698, 76354, 76915, 77216, 77346, 77659, 77862, 78017, 78360, 78593, 79028, 79381, 79596, 79851, 80004, 80161, 81367, 81792, 82494, 83218, 83811, 84252, 85388, 85780, 86019, 87939, 88217, 88772, 89457, 89728, 89833, 90693, 91501, 91864, 92239, 93861, 94132, 94332, 94766, 95067, 95825, 96301, 97363, 99068, 99548, 100503, 101213, 102044, 102486, 105820, 106205, 106597, 106742, 108328, 108504, 108712, 109133, 109325, 109680, 110319, 110673, 111346, 112091, 112493, 113288, 114018, 114212, 116260, 116422, 116788, 117655, 118673, 119636, 120954, 121681, 122024, 122795, 124346, 125524, 125910, 126113, 126841, 127118, 127298, 128925, 129740, 129890, 130025, 130346, 130681, 131786, 132253, 132576, 133317, 133574, 133773, 133952, 134623, 135028, 135277, 135671, 135831, 136329, 137474, 139245, 140226, 140499, 140878, 141196, 141371, 141895, 142335, 142561, 142950, 143227, 147161, 147328, 147651, 147783, 148107, 148352, 148966, 149257, 149502, 150722, 151612, 151814, 152661, 153631, 153798, 154003, 154113, 154361, 155204, 155803, 156157, 157645, 157818, 158050, 160058, 161252, 161426, 161797, 161988, 162109, 163501, 164342, 165235, 166032, 166354, 166974, 167484, 167682, 168443, 168874, 169248, 169526, 171313, 171754, 171946, 172099, 172457, 173156, 174715, 175028, 176301, 176858, 179991, 180439, 180856, 180975, 181319, 181429, 181671, 181899, 182014, 182285, 182787, 183659, 183880, 184073, 184166, 235016, 235500, 242948, 243268 }
#define PROMPT_BLOCK_TO { 72130, 72405, 73148, 73874, 74150, 75154, 75516, 75979, 76617, 77059, 77327, 77642, 77799, 78016, 78180, 78466, 78868, 79235, 79595, 79849, 79988, 80107, 80430, 81505, 81892, 82643, 83595, 83943, 84351, 85578, 85889, 86250, 88040, 88424, 88996, 89570, 89832, 90042, 90799, 91619, 92023, 92526, 94131, 94331, 94543, 95066, 95608, 95980, 96427, 97525, 99429, 99670, 100607, 102043, 102395, 103071, 106033, 106341, 106741, 106844, 108443, 108605, 108855, 109267, 109468, 109800, 110422, 111345, 111501, 112410, 112828, 113414, 114165, 114528, 116371, 116749, 116897, 118251, 118959, 119778, 121120, 121862, 122200, 123015, 124536, 125717, 126084, 126218, 126947, 127280, 127457, 129113, 129857, 130024, 130189, 130465, 130786, 131907, 132405, 133070, 133487, 133772, 133876, 134125, 134962, 135159, 135523, 135823, 135994, 136462, 139244, 139371, 140418, 140642, 141115, 141308, 141527, 142177, 142503, 142663, 143097, 146643, 147297, 147520, 147782, 147906, 148209, 148458, 149178, 149410, 149795, 151004, 151813, 152029, 152798, 153797, 153981, 154112, 154249, 154487, 155517, 156069, 156335, 157752, 157917, 159293, 160159, 161400, 161752, 161982, 162108, 162330, 164175, 164465, 165349, 166318, 166495, 167331, 167681, 167957, 168593, 169247, 169456, 169676, 171645, 171945, 172080, 172228, 172562, 173371, 174828, 175305, 176438, 177212, 180100, 180542, 180956, 181118, 181428, 181668, 181846, 182013, 182281, 182473, 183061, 183681, 183947, 184078, 220684, 235141, 242947, 243267, 243424 }
#define PROMPT_BLOCK_WORD_LEN { 2, 3, 3, 3, 2, 4, 2, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 4, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 4, 3, 2, 3, 2, 2, 3, 2, 3, 4, 2, 3, 2, 3, 2, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 2, 4, 5, 4 }
#define PROMPT_BLOCK_WORD { {0xf2, 0xbb, 0x00, 0x00, 0x00}, {0xfb, 0xe7, 0xba, 0x00, 0x00}, {0x40, 0xe4, 0xc8, 0x00, 0x00}, {0xf5, 0xdc, 0xa8, 0x00, 0x00}, {0xfe, 0xa8, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0xe9, 0xc8, 0x00}, {0xef, 0x8a, 0x00, 0x00, 0x00}, {0xef, 0x8a, 0x00, 0x00, 0x00}, {0x72, 0x61, 0x70, 0x00, 0x00}, {0xf5, 0xe8, 0xb0, 0x00, 0x00}, {0xef, 0xc1, 0x00, 0x00, 0x00}, {0x40, 0xef, 0xc0, 0x00, 0x00}, {0xef, 0x81, 0x00, 0x00, 0x00}, {0x40, 0xe7, 0x9b, 0x00, 0x00}, {0xef, 0xc7, 0x00, 0x00, 0x00}, {0xe3, 0xac, 0x00, 0x00, 0x00}, {0xf6, 0xbf, 0x00, 0x00, 0x00}, {0xfa, 0x86, 0x00, 0x00, 0x00}, {0xfa, 0x86, 0x00, 0x00, 0x00}, {0xeb, 0xce, 0x00, 0x00, 0x00}, {0x5b, 0xf2, 0xb9, 0x00, 0x00}, {0xf2, 0xb8, 0x00, 0x00, 0x00}, {0xf7, 0xbe, 0x00, 0x00, 0x00}, {0xf4, 0xe8, 0xa4, 0x00, 0x00}, {0xf9, 0xa5, 0x00, 0x00, 0x00}, {0xe6, 0xc2, 0x00, 0x00, 0x00}, {0xe6, 0xa1, 0x00, 0x00, 0x00}, {0xea, 0xbc, 0x00, 0x00, 0x00}, {0xf8, 0x92, 0x00, 0x00, 0x00}, {0xe6, 0xc7, 0x00, 0x00, 0x00}, {0xe6, 0xb3, 0x00, 0x00, 0x00}, {0xe6, 0xc7, 0x00, 0x00, 0x00}, {0xf2, 0x8f, 0x00, 0x00, 0x00}, {0xe7, 0x80, 0x00, 0x00, 0x00}, {0xf9, 0xd5, 0xb5, 0x00, 0x00}, {0xf9, 0xbc, 0x00, 0x00, 0x00}, {0xf2, 0xcb, 0x00, 0x00, 0x00}, {0xde, 0xbd, 0x00, 0x00, 0x00}, {0xeb, 0x81, 0x00, 0x00, 0x00}, {0xf0, 0x8e, 0x00, 0x00, 0x00}, {0x5b, 0xf6, 0xad, 0x00, 0x00}, {0x40, 0xf8, 0x9b, 0x00, 0x00}, {0xe7, 0x86, 0x00, 0x00, 0x00}, {0xf5, 0xdd, 0x9d, 0x00, 0x00}, {0xf8, 0x9a, 0x00, 0x00, 0x00}, {0xee, 0xa7, 0x00, 0x00, 0x00}, {0xff, 0xa3, 0x00, 0x00, 0x00}, {0xe7, 0x84, 0x00, 0x00, 0x00}, {0xfb, 0xcf, 0x00, 0x00, 0x00}, {0xf6, 0xb7, 0x00, 0x00, 0x00}, {0xeb, 0x9a, 0x00, 0x00, 0x00}, {0xf5, 0xd7, 0x9b, 0x00, 0x00}, {0xea, 0xca, 0x00, 0x00, 0x00}, {0xde, 0xc1, 0x00, 0x00, 0x00}, {0xff, 0x88, 0x00, 0x00, 0x00}, {0xe7, 0xba, 0x00, 0x00, 0x00}, {0xea, 0xcd, 0x00, 0x00, 0x00}, {0x07, 0xdd, 0xcd, 0x00, 0x00}, {0xf2, 0xcb, 0x00, 0x00, 0x00}, {0x40, 0xf7, 0xba, 0x00, 0x00}, {0xf5, 0xdb, 0x8b, 0x00, 0x00}, {0x40, 0xe6, 0xaf, 0x00, 0x00}, {0x40, 0xfa, 0xdf, 0x82, 0x00}, {0xe4, 0xc9, 0x00, 0x00, 0x00}, {0xf7, 0xa8, 0x00, 0x00, 0x00}, {0xe7, 0x97, 0x00, 0x00, 0x00}, {0xd5, 0xb4, 0x00, 0x00, 0x00}, {0x07, 0x74, 0x76, 0x00, 0x00}, {0xf6, 0xbb, 0x00, 0x00, 0x00}, {0xef, 0x89, 0x00, 0x00, 0x00}, {0x40, 0xe5, 0x80, 0x00, 0x00}, {0x40, 0xf1, 0x8e, 0x00, 0x00}, {0x40, 0xeb, 0xcc, 0x00, 0x00}, {0x40, 0xeb, 0xcc, 0x00, 0x00}, {0x5b, 0xe7, 0x94, 0x00, 0x00}, {0x5b, 0xe7, 0x94, 0x00, 0x00}, {0xf6, 0xb9, 0x00, 0x00, 0x00}, {0xeb, 0xc9, 0x00, 0x00, 0x00}, {0x40, 0xf4, 0xa1, 0x00, 0x00}, {0xf5, 0xdd, 0xac, 0x00, 0x00}, {0xf5, 0xdd, 0x9f, 0x00, 0x00}, {0xeb, 0x86, 0x00, 0x00, 0x00}, {0xfe, 0xbb, 0x00, 0x00, 0x00}, {0xfd, 0xcc, 0x00, 0x00, 0x00}, {0xe2, 0x9b, 0x00, 0x00, 0x00}, {0xf0, 0xb7, 0x00, 0x00, 0x00}, {0xe7, 0xb3, 0x00, 0x00, 0x00}, {0x40, 0xe7, 0xac, 0x00, 0x00}, {0xf5, 0xdd, 0x8a, 0x00, 0x00}, {0xe6, 0xcf, 0x00, 0x00, 0x00}, {0xf8, 0xca, 0x00, 0x00, 0x00}, {0xef, 0x93, 0x00, 0x00, 0x00}, {0x40, 0xf4, 0x9b, 0x00, 0x00}, {0xf2, 0xb0, 0x00, 0x00, 0x00}, {0xf0, 0xe1, 0xcb, 0x00, 0x00}, {0xe2, 0x9b, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0xc0, 0x00, 0x00}, {0xff, 0x85, 0x00, 0x00, 0x00}, {0x40, 0xe2, 0xb2, 0x00, 0x00}, {0xff, 0x80, 0x00, 0x00, 0x00}, {0x40, 0xe2, 0xb2, 0x00, 0x00}, {0x40, 0xe6, 0x80, 0x00, 0x00}, {0x40, 0xf5, 0xa2, 0x00, 0x00}, {0xe6, 0x9b, 0x00, 0x00, 0x00}, {0xe9, 0xa9, 0x00, 0x00, 0x00}, {0xf4, 0xe5, 0xcb, 0x00, 0x00}, {0xe6, 0x99, 0x00, 0x00, 0x00}, {0x40, 0xf4, 0x98, 0x00, 0x00}, {0x40, 0xf5, 0xa4, 0x00, 0x00}, {0x40, 0xf4, 0xa4, 0x00, 0x00}, {0xf4, 0x82, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0xb0, 0x00, 0x00}, {0xde, 0xbf, 0x00, 0x00, 0x00}, {0xfb, 0xb5, 0x00, 0x00, 0x00}, {0xf2, 0xb3, 0x00, 0x00, 0x00}, {0xf7, 0x8a, 0x00, 0x00, 0x00}, {0xe1, 0xbb, 0x00, 0x00, 0x00}, {0x40, 0xf3, 0xcb, 0x00, 0x00}, {0xe7, 0xce, 0x00, 0x00, 0x00}, {0xf8, 0xe8, 0x9b, 0x00, 0x00}, {0xe2, 0x95, 0x00, 0x00, 0x00}, {0x40, 0xf9, 0x90, 0x00, 0x00}, {0x40, 0xf5, 0xbf, 0x00, 0x00}, {0x40, 0xe2, 0xa3, 0x00, 0x00}, {0x40, 0xea, 0x81, 0x00, 0x00}, {0x40, 0xf5, 0xe1, 0x9c, 0x00}, {0x40, 0xf3, 0xc1, 0x00, 0x00}, {0x40, 0xf7, 0xc3, 0x00, 0x00}, {0xe4, 0x8a, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0xa4, 0x00, 0x00}, {0x40, 0xf5, 0x9a, 0x00, 0x00}, {0x40, 0xf5, 0xbb, 0x00, 0x00}, {0xf9, 0xac, 0x00, 0x00, 0x00}, {0xec, 0xc8, 0x00, 0x00, 0x00}, {0x07, 0x6d, 0x70, 0x00, 0x00}, {0x40, 0xeb, 0xa5, 0x00, 0x00}, {0x40, 0xf3, 0xc8, 0x00, 0x00}, {0x40, 0xec, 0xa6, 0x00, 0x00}, {0x40, 0xea, 0x95, 0x00, 0x00}, {0xeb, 0xab, 0x00, 0x00, 0x00}, {0x40, 0xea, 0x8e, 0x00, 0x00}, {0xea, 0x92, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0x95, 0x00, 0x00}, {0xea, 0x97, 0x00, 0x00, 0x00}, {0xea, 0x97, 0x00, 0x00, 0x00}, {0xeb, 0xc3, 0x00, 0x00, 0x00}, {0x40, 0xf4, 0x9d, 0x00, 0x00}, {0x40, 0xfa, 0xdd, 0xc6, 0x00}, {0x40, 0xfc, 0xa3, 0x00, 0x00}, {0xfb, 0x8f, 0x00, 0x00, 0x00}, {0x40, 0xf4, 0xb2, 0x00, 0x00}, {0xd8, 0x8e, 0x00, 0x00, 0x00}, {0xfc, 0xc8, 0x00, 0x00, 0x00}, {0x40, 0xf7, 0xb3, 0x00, 0x00}, {0xf8, 0x88, 0x00, 0x00, 0x00}, {0xfb, 0xea, 0xc6, 0x00, 0x00}, {0x40, 0xf0, 0xe0, 0xa5, 0x00}, {0xfc, 0xa9, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0xb4, 0x00, 0x00}, {0xfc, 0xbb, 0x00, 0x00, 0x00}, {0x40, 0xf4, 0x95, 0x00, 0x00}, {0x5b, 0x64, 0x00, 0x00, 0x00}, {0x40, 0xf9, 0x9b, 0x00, 0x00}, {0xef, 0xbc, 0x00, 0x00, 0x00}, {0xd3, 0xbb, 0x00, 0x00, 0x00}, {0xfa, 0xd5, 0xc2, 0x00, 0x00}, {0xf3, 0xd6, 0xca, 0x00, 0x00}, {0x40, 0xf6, 0xc1, 0x00, 0x00}, {0xf6, 0xc2, 0x00, 0x00, 0x00}, {0x40, 0xfc, 0x99, 0x00, 0x00}, {0xf8, 0xe5, 0xc6, 0x00, 0x00}, {0x40, 0xf8, 0x8a, 0x00, 0x00}, {0xfe, 0xad, 0x00, 0x00, 0x00}, {0xef, 0xa1, 0x00, 0x00, 0x00}, {0xed, 0xbd, 0x00, 0x00, 0x00}, {0xf5, 0xdc, 0xc8, 0x00, 0x00}, {0xf6, 0xd1, 0xb3, 0x00, 0x00}, {0xfa, 0x96, 0x00, 0x00, 0x00}, {0xf6, 0xd2, 0xb4, 0x00, 0x00}, {0xfb, 0xa6, 0x00, 0x00, 0x00}, {0xde, 0xb1, 0x00, 0x00, 0x00}, {0xf2, 0x94, 0x00, 0x00, 0x00}, {0xde, 0xb0, 0x00, 0x00, 0x00}, {0xfb, 0xa7, 0x00, 0x00, 0x00}, {0xfb, 0xe7, 0xb3, 0x00, 0x00}, {0x5b, 0xf3, 0xce, 0x5d, 0x00}, {0x5b, 0xf3, 0xce, 0x5d, 0x00}, {0x5b, 0xf3, 0xce, 0x5d, 0x00}, {0x5b, 0xf3, 0xce, 0x5d, 0x00}, {0xee, 0xb1, 0x00, 0x00, 0x00}, {0x50, 0xdf, 0xa5, 0x52, 0x00}, {0x50, 0xf6, 0xdf, 0x98, 0x52}, {0x50, 0xf6, 0xdf, 0xaf, 0x00} }
// The binary carries the stamped table in a compact form, built here at
// compile time: the first start, then per entry the gap to the end of the
// previous range and the range length (16 bits each), the word length and the
// words back to back.  The stamped arrays are only read by the compiler.
namespace tf1_block {
constexpr int32_t kVon[PROMPT_BLOCK_N] = PROMPT_BLOCK_FROM;
constexpr int32_t kBis[PROMPT_BLOCK_N] = PROMPT_BLOCK_TO;
constexpr unsigned char kLen[PROMPT_BLOCK_N] = PROMPT_BLOCK_WORD_LEN;
constexpr unsigned char kWort[PROMPT_BLOCK_N][PROMPT_BLOCK_WORD_MAX] =
    PROMPT_BLOCK_WORD;
constexpr int Bytes() {
  int s = 0;
  for (int b = 0; b < PROMPT_BLOCK_N; ++b) s += kLen[b];
  return s;
}
constexpr bool Passt() {   // sorted, disjoint, every value fits its field
  for (int b = 0; b < PROMPT_BLOCK_N; ++b) {
    if (kBis[b] < kVon[b] || kBis[b] - kVon[b] > 0xFFFF) return false;
    if (b && (kVon[b] <= kBis[b - 1] || kVon[b] - kBis[b - 1] - 1 > 0xFFFF)) return false;
    if (kLen[b] < 1 || kLen[b] > PROMPT_BLOCK_WORD_MAX) return false;
  }
  return true;
}
static_assert(Passt(), "block table: ranges must be sorted and disjoint, gaps and lengths below 2^16");
static_assert(PROMPT_BLOCK_N <= 256, "block table larger than the counters");
struct Kompakt {
  uint16_t luecke[PROMPT_BLOCK_N];   // start - (end of the previous range + 1)
  uint16_t span_len[PROMPT_BLOCK_N];   // end - start
  unsigned char wlen[PROMPT_BLOCK_N];
  unsigned char wort[Bytes()];
};
constexpr Kompakt Baue() {
  Kompakt t{};
  int o = 0;
  for (int b = 0; b < PROMPT_BLOCK_N; ++b) {
    t.luecke[b] = (uint16_t)(b ? kVon[b] - kBis[b - 1] - 1 : 0);
    t.span_len[b] = (uint16_t)(kBis[b] - kVon[b]);
    t.wlen[b] = kLen[b];
    for (int i = 0; i < kLen[b]; ++i) t.wort[o++] = kWort[b][i];
  }
  return t;
}
constexpr Kompakt kTafel = Baue();
}  // namespace tf1_block

void CoderPrompt::BlockZeile() {
  const long long k = page_ + (0);
  if (census_title::Kontext(k)) return;
  // the ranges are sorted and disjoint: walk them in order
  long long von = tf1_block::kVon[0];
  int o = 0;
  for (int b = 0; b < PROMPT_BLOCK_N; ++b) {
    von += tf1_block::kTafel.luecke[b];
    if (k < von) return;
    const long long bis = von + tf1_block::kTafel.span_len[b];
    const int n = tf1_block::kTafel.wlen[b];
    if (k <= bis) {
      unsigned char z[PROMPT_BLOCK_WORD_MAX + 2];   // "#<word>\n"
      z[0] = 0x23;
      memcpy(z + 1, tf1_block::kTafel.wort + o, (size_t)n);
      z[n + 1] = 0x0A;
      RdZeile(z, n + 2);
      return;
    }
    von = bis + 1;
    o += n;
  }
}

int CoderPrompt::ZwangByte() const {
  static const int32_t kVon[4] = { 220685, 234593, 234595, 234596 };
  static const int32_t kBis[4] = { 234592, 234594, 234595, 235499 };
  static const unsigned char kLen[4] = { 3, 4, 7, 4 };
  static const unsigned char kByte[4][7] =
      { {0x40, 0xaf, 0x4a, 0x00, 0x00, 0x00, 0x00}, {0x40, 0xd6, 0xc5, 0x4a, 0x00, 0x00, 0x00}, {0x40, 0xe7, 0x9e, 0x40, 0xfe, 0xca, 0x4a}, {0x40, 0xf7, 0xa6, 0x4a, 0x00, 0x00, 0x00} };
  if (g1_preamble_ || !in_title_) return -1;   // as in ZwangJetzt
  const long long i = pos_ - article_start_;
  const long long k = page_ + (0);
  for (int b = 0; b < 4; ++b) {
    if (k < kVon[b] || k > kBis[b]) continue;
    return (i >= 0 && i < kLen[b]) ? kByte[b][i] : -1;
  }
  return -1;
}

void CoderPrompt::PieceCut() {
  piece_rest_ = true;
}

bool CoderPrompt::TakePre(const unsigned char** toks, int* n) {
  if (!have_pre_) return false;
  have_pre_ = false;
  *toks = pre_.data();
  *n = (int)pre_.size();
  return true;
}

bool CoderPrompt::TakePost(const unsigned char** toks, int* n) {
  if (!have_post_) return false;
  have_post_ = false;
  *toks = post_.data();
  *n = (int)post_.size();
  return true;
}

void CoderPrompt::Report() const {
}

}  // namespace tf_prompt

