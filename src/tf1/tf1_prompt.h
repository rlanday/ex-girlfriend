#ifndef TF1_PROMPT_H_
#define TF1_PROMPT_H_

#include "../transformer.h"

#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "tf1_engine.h"
#include "../redirect_table.h"
#include "../string_pool.h"
#include "../alias_table.h"
namespace tf_prompt {

// ---- grammar --------------------------------------------------------------
const unsigned char* TitleEnd();
int TitleEndLen();
constexpr unsigned char kParagraph = 0x08;   // "\n"
constexpr unsigned char kLinkOpen  = 0x31;   // "["
constexpr unsigned char kEq        = 0x29;      // "="
constexpr unsigned char kPipe      = 0x2d;    // "|"
constexpr unsigned char kColon     = 0x26;   // ":"
constexpr bool kLinkDouble         = 0 != 0;
constexpr int kMaxTitleTokens = 64;   // as zctx/zgrammar.py MAX_TITLE_TOKENS
constexpr int kMinTitleTokens = 3;    // as zctx/zgrammar.py MIN_TITLE_TOKENS
const unsigned char* RedirMark();

constexpr int kRedirMaxLen = 4;

int RedirMarkLen();
unsigned char LinkClose();            // "]"
unsigned char Hash();                 // "#"
unsigned char Digit(int d);           // "0".."9"

// digit-run length class 0..7 (bounds 1, 2, 3, 4, 6, 9, 15, as zctx/zgate.py)
int LengthClass(int run_length);

// ---- parameters (the values of the training package) ----------------------
struct Params {
  int syn_max = 4;         // at most this many synonyms
  int syn_tokens = 64;     // token budget of the synonym block
  int alias_cap = 64;      // token budget of the alias block
  int gate_m = 1;          // minimum digit-run length for a sentinel row
  bool gate_exclude = true;  // no sentinel rows in links and list/table lines
  int syn_hops = 2;        // one chain resolution
  int alias_hops = 12;     // up to twelve
  int link_ring = 1 << 16; // ring buffer for the link in progress
};

// ---- enabled prompt kinds (fixed at build time, see ArmsFromEnv) ----------
struct Arms {
  bool num = false, z1t = false, syn = false, alias = false;
  bool a2 = false, a2n = false, sent = false, titelmask = false;
  bool sent_prior = true;    // take the PPMd prior before a sentinel row
  bool meta = false;
};
Arms ArmsFromEnv();

class MetaTable {
 public:
  void Load(const char* path);
  size_t Pages() const { return off_.size(); }
  const unsigned char* Field(long long p, int f, int* n) const;

 private:
  struct Eintrag { uint32_t off[3]; uint32_t len[3]; };
  std::vector<unsigned char> bytes_;
  std::vector<Eintrag> off_;
};

// ---- counters ------------------------------------------------------------
struct Stats {
  unsigned long long articles = 0;
  unsigned long long titles = 0;
  unsigned long long num_tokens = 0;
  unsigned long long z1t_tokens = 0;
  unsigned long long syn_tokens = 0;
  unsigned long long syn_articles = 0;
  unsigned long long alias_tokens = 0;
  unsigned long long alias_articles = 0;
  unsigned long long a2_sites = 0;        // heading sites (unfiltered)
  unsigned long long a2_used = 0;         // of them fed
  unsigned long long a2_tokens = 0;
  unsigned long long a2n_tokens = 0;
  unsigned long long sent_sites = 0;
  unsigned long long sent_fed = 0;
  unsigned long long sent_class[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  unsigned long long gate_positions = 0;
  unsigned long long masked_title = 0;
  unsigned long long prior_positions = 0;
  unsigned long long links = 0;
  unsigned long long links_piped = 0;
  unsigned long long links_dropped = 0;   // candidate older than the ring
  unsigned long long redirects = 0;
  unsigned long long piece_cuts = 0;      // 2^17 cuts seen
  unsigned long long a2_piece_rest = 0;   // A2 sites dropped behind such a cut
  unsigned long long sent_piece_rest = 0;
  unsigned long long sent_in_titel = 0;
  unsigned long long a2_ohne_titel = 0;
  unsigned long long redirect_pages = 0;
  unsigned long long redir_merges = 0;
  unsigned long long redir_runs = 0;
  unsigned long long meta_seiten = 0;        // pages with >= 1 META row
  unsigned long long meta_zeilen = 0;        // META rows fed
  unsigned long long meta_tok_id = 0;        // tokens, line-end included
  unsigned long long meta_tok_autor = 0;
  unsigned long long meta_tok_datum = 0;
  unsigned long long meta_ohne_eintrag = 0;  // page index not in the table
  unsigned long long meta_feld_leer = 0;     // entry there, field empty
  unsigned long long meta_zeile_gekappt = 0; // row dropped, 64-token cap
  unsigned long long meta_byte_fremd = 0;    // byte outside the 208 set
  unsigned long long meta_sent = 0;          // sentinel rows fed in META rows
  unsigned long long meta_sent_id = 0;       // ... of them in the id row
  unsigned long long meta_sent_autor = 0;    // ... in the author row (names
                                             //     with digits, e.g. "ams80")
  unsigned long long meta_sent_datum = 0;    // ... in the YYYY / month row
  unsigned long long meta_vorn_seiten = 0;   // pages with a META block BEFORE
  unsigned long long meta_vorn_tokens = 0;   //   the title, and its tokens
  unsigned long long rd_seiten = 0;          // pages that got the `#rd` row
  unsigned long long rd_tokens = 0;
  unsigned long long meta_tok_k = 0;
  unsigned long long teilb_prompts = 0;      // prompt blocks in part B
  unsigned long long teilb_tokens = 0;       // their tokens (with line ends)
  unsigned long long teilb_marken = 0;
  unsigned long long teilb_rd = 0;
  unsigned long long teilb_stuecke = 0;      // piece cuts in part B
};

// ---- the prompt automaton ------------------------------------------------
class CoderPrompt {
 public:
  static const int kSlots = 6;        // num, z1t, syn, alias, a2n, meta

  CoderPrompt(const Params& p, const Arms& a);
  ~CoderPrompt();

  // Take in one coded token.  Must be called before the two Take calls and
  // before the model step of this token.  `piece_end` is true when this token
  // was the last of an article separator sequence.
  void Observe(unsigned char token, bool piece_end);

  void PieceCut();

  // Feeding before the model step of the current token (rows before the title
  // at the start of an article, sentinel row after a run of digits).
  bool TakePre(const unsigned char** toks, int* n);
  // Feeding after the model step (prompt behind the title, heading reminder
  // behind "==\n").  The output of the last fed step is the distribution for the
  // next coded token.
  bool TakePost(const unsigned char** toks, int* n);
  bool TakeSeitenVorblock(const unsigned char** toks, int* n, int* meta_von,
                          int* meta_bis);
  bool HasPost() const { return have_post_; }

  // Should the distribution for the next token come from the PPMd prior
  // instead of the transformer?  Two reasons, both causal: title mask and
  // sentinel gating.
  bool PriorNext() const { return prior_next_; }

  bool InRedirect() const { return redir_now_; }
  bool LastArticleRedirect() const { return redir_last_; }
  void PageBoundary();
  void CountMerge(bool neuer_lauf) {   // neuer_lauf: a new redirect run starts
    ++st_.redir_merges;
    if (neuer_lauf) ++st_.redir_runs;
  }

  int PostMetaFrom() const { return meta_von_; }
  int PostMetaTo() const { return meta_bis_; }
  int PreMetaFrom() const { return meta_pre_von_; }
  int PreMetaTo() const { return meta_pre_bis_; }
  bool ImRedirectBlock() const {
    return a_.meta && page_ + (0)
                          < 71148;
  }

  const uint16_t* Prior0() const { return prior0_.data(); }
  bool Prior0Given() const { return prior0_given_; }
  void AddFed(unsigned long long n) { fed_ += n; }
  void Finish();                 // close the last article
  void Report() const;
  const Stats& GetStats() const { return st_; }

 private:
  Params p_;
  Arms a_;
  long long pos_ = 0;                 // number of tokens taken in
  long long article_ = -1;            // running article index
  long long article_start_ = 0;
  long long page_ = 0;
  MetaTable meta_tab_;
  bool meta_tab_geladen_ = false;
  // Slice of `post_` that carries the META rows (see PostMetaFrom()).
  int meta_von_ = 0;
  int meta_bis_ = 0;
  int meta_pre_von_ = 0;
  int meta_pre_bis_ = 0;
  int rd_pre_von_ = 0;
  int rd_pre_bis_ = 0;

  bool in_title_ = false;
  bool g1_preamble_ = true;
  bool mf_tab_ = false;
  bool mf_auf_ = false;
  int title_len_ = 0;                 // T including the terminator, 0 = none
  std::vector<unsigned char> title_;  // the T tokens

  // --- feed buffers
  std::vector<unsigned char> pre_, post_;
  bool have_pre_ = false, have_post_ = false;
  // Does the pre block of this step carry a sentinel row?  Only used by the
  // guard in the heading-reminder branch.
  bool sent_pre_ = false;
  bool prior_next_ = false;
  unsigned long long fed_ = 0;

  // --- digit-run automaton (as zctx/zgate.py gate_mask, incremental)
  int digit_run_ = 0;                 // running digit sequence (for gate_m)
  int link_depth_ = 0;
  bool marker_line_ = false;
  int nonspace_since_nl_ = 0;
  void ZeilenMarke(unsigned char b);
  unsigned char prev_byte_ = 0xFF;
  int gate_run_ = 0;
  long long last_sent_pos_ = -1;      // for the last-position guard in EndArticle
  bool piece_rest_ = false;           // behind a 2^17 cut

  // --- heading reminder
  int a2_count_ = 0;                  // sites of this article (1-based)
  unsigned char tail_[24];            // the last tokens

  bool no_prompt_ = false;
  bool redir_now_ = false;
  bool redir_last_ = false;
  int rd_open_ = 0;                   // first line break of the page seen
  int rd_take_ = -1;
  std::vector<uint32_t> doc_;

  // --- training-data dumps (written only by the dump and tool builds)
  std::string dump_;
  struct Sink { FILE* f = nullptr; };
  Sink d_tok_[kSlots];
  FILE* d_off_[kSlots] = {nullptr};
  int off_run_[kSlots] = {0};
  FILE* d_title_ = nullptr;
  FILE* d_a2_idx_ = nullptr;
  FILE* d_a2_off_ = nullptr;
  int a2_run_ = 0;
  FILE* d_sent_idx_ = nullptr;
  FILE* d_sent_pos_ = nullptr;
  FILE* d_sent_cls_ = nullptr;
  int sent_run_ = 0;

  std::vector<uint16_t> prior0_;
  bool prior0_given_ = false;
  Stats st_;

  void StartArticle();
  void EndArticle();
  void TitleClosed(int title_len);
  void BuildPrompt();
  void ScanRedirectPage(unsigned char token);
  void EmitBlock(int slot, const unsigned char* p, int n);
  void BuildMeta(long long page_index, std::vector<unsigned char>* out_bytes);
  void SeitenVorblock(long long q);
};

// Tool mode: run the same class over a package.
int RunTool(int argc, char** argv);

}

#endif
