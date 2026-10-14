
#include "tf1_prompt.h"

#include "../cold_size.h"
#include "../alias_substring_dedup.h"

#include "tf1_prior0.h"   // the stamped prior0 table

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
    transformer::Fail("prompt: byte 0x%02x not in the 208-byte set", b);
  }
  return (unsigned char)transformer::kByteToIndex[b];
}
unsigned char LinkClose() { return ByteTok(0x5D); }
unsigned char Hash() { return ByteTok(0x23); }
unsigned char Digit(int d) { return ByteTok(0x30 + d); }

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
      transformer::Fail("prompt: byte set has %d values, expected %d", k,
          208);
    }
    ready = true;
  }
  return tab;
}

// digit-run length classes, as zctx/zgate.py LENGTH_CLASS_BOUNDS
static const int kBounds[7] = {1, 2, 3, 4, 6, 9, 15};
int LengthClass(int run_length) {
  for (int k = 0; k < 7; ++k) {
    if (run_length <= kBounds[k]) return k;
  }
  return 7;
}

const char* const kMetaTabelle = "meta_seiten.tsv";

void MetaTable::Load(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    transformer::Fail("%s is not readable in the run "
        "directory.  The table is the ONLY field "
        "source -- running on without it would silently measure the control "
        "arm under the name of the meta arm.", path);
  }
  // One pass, two buffers: the bytes go into an arena, the entry gets three
  // (offset, length) pairs into it.  No std::string per field -- the table has
  // a quarter million lines and is read once per run, but the coder is also
  // the thing a 20 h chain waits on.
  std::vector<unsigned char> line_buf;
  long long erwartet = 0;         // the next page index that must come
  bool kopf_geprueft = false;
  int c;
  for (;;) {
    c = fgetc(f);
    if (c != EOF && c != '\n') { line_buf.push_back((unsigned char)c); continue; }
    if (!line_buf.empty() && line_buf.back() == '\r') line_buf.pop_back();
    if (!line_buf.empty()) {
      // Split on tabs.  Exactly four fields; more or fewer is a form error,
      // and a form error here is worth a stop -- a shifted column would put
      // every author on the wrong page and still produce a plausible run.
      const unsigned char* b = line_buf.data();
      const int n = (int)line_buf.size();
      int von[4] = {0, 0, 0, 0};
      int len[4] = {0, 0, 0, 0};
      int k = 0;
      int start = 0;
      for (int i = 0; i <= n; ++i) {
        if (i == n || b[i] == '\t') {
          if (k < 4) { von[k] = start; len[k] = i - start; }
          ++k;
          start = i + 1;
        }
      }
      if (!kopf_geprueft) {
        kopf_geprueft = true;
        if (len[0] == 5 && memcmp(b, "seite", 5) == 0) { line_buf.clear();
          if (c == EOF) break; else continue; }
      }
      if (k != 4) {
        transformer::Fail("%s, line %lld has %d tab-separated "
            "fields, not 4 (seite/id/autor/jjjj-mm)", path,
            (long long)off_.size() + 1, k);
      }
      long long page_index = 0;
      for (int i = 0; i < len[0]; ++i) {
        const unsigned char d = b[von[0] + i];
        if (d < '0' || d > '9') {
          transformer::Fail("%s, page index is not a decimal "
              "number in line %lld", path, (long long)off_.size() + 1);
        }
        page_index = page_index * 10 + (d - '0');
      }
      if (page_index != erwartet) {
        transformer::Fail("%s is not gapless -- expected page "
            "%lld, found %lld.  The lookup is an array index, so a gap would "
            "shift every page behind it onto a foreign author.", path,
            erwartet, page_index);
      }
      ++erwartet;
      Eintrag e;
      for (int fnr = 0; fnr < 3; ++fnr) {
        e.off[fnr] = (uint32_t)bytes_.size();
        e.len[fnr] = (uint32_t)len[fnr + 1];
        bytes_.insert(bytes_.end(), b + von[fnr + 1],
                      b + von[fnr + 1] + len[fnr + 1]);
      }
      off_.push_back(e);
    }
    line_buf.clear();
    if (c == EOF) break;
  }
  fclose(f);
  if (off_.empty()) {
    transformer::Fail("%s carries no page line", path);
  }
  fprintf(stderr, "page table %s loaded -- %lld pages, "
      "%lld field bytes\n", path, (long long)off_.size(),
      (long long)bytes_.size());
}

const unsigned char* MetaTable::Field(long long p, int f, int* n) const {
  if (p < 0 || (size_t)p >= off_.size()) { *n = 0; return nullptr; }
  const Eintrag& e = off_[(size_t)p];
  *n = (int)e.len[f];
  return bytes_.data() + e.off[f];
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
COLD_SIZE_ATTR Arms ArmsFromEnv() {
  Arms a;
  a.syn = true;
  a.a2 = true;
  a.a2n = true;
  a.sent = true;
  a.meta = true;
  a.sent_prior = 0 != 0;
  return a;
}

// ---- training-data dumps (written only by the dump and tool builds) ------
COLD_SIZE_ATTR static FILE* OpenDump(const std::string& base, const char* name) {
  if (base.empty()) return nullptr;
  const std::string path = base + name;
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) transformer::Fail("prompt dump: cannot create %s", path.c_str());
  return f;
}
static void PutI32(FILE* f, int32_t v) { if (f) fwrite(&v, 4, 1, f); }

static const char* kSlotTok[CoderPrompt::kSlots] = {
    "prompt-num-tokens.uint8", "prompt-z1t-tokens.uint8",
    "prompt-syn-tokens.uint8", "prompt-alias-tokens.uint8",
    "prompt-a2n-tokens.uint8"
    , "prompt-meta-tokens.uint8"
};
static const char* kSlotOff[CoderPrompt::kSlots] = {
    "prompt-num-offsets.int32", "prompt-z1t-offsets.int32",
    "prompt-syn-offsets.int32", "prompt-alias-offsets.int32",
    "prompt-a2n-offsets.int32"
    , "prompt-meta-offsets.int32"
};

// ---- CoderPrompt ----------------------------------------------------------
COLD_SIZE_ATTR CoderPrompt::CoderPrompt(const Params& p, const Arms& a) : p_(p), a_(a) {
  memset(tail_, 0xFF, sizeof(tail_));
  prior0_.assign(208, 0);
  // The stamped prior0 table; uniform only if none is stamped.
  if (const uint16_t* st = transformer::Prior0Stamped()) {
    memcpy(prior0_.data(), st, sizeof(uint16_t) * 208);
    prior0_given_ = true;
  } else {
    const float u = 1.0f / (float)208;
    for (int i = 0; i < 208; ++i) prior0_[i] = transformer::FloatToHalf(u);
  }
  if (const char* q = transformer::PromptDumpPath()) dump_ = q;
  for (int s = 0; s < kSlots; ++s) {
    d_tok_[s].f = OpenDump(dump_, kSlotTok[s]);
    d_off_[s] = OpenDump(dump_, kSlotOff[s]);
    PutI32(d_off_[s], 0);
  }
  d_title_ = OpenDump(dump_, "prompt-title-lengths.int32");
  d_a2_idx_ = OpenDump(dump_, "prompt-a2-index.int32");
  PutI32(d_a2_idx_, 0);
  d_a2_off_ = OpenDump(dump_, "prompt-a2-offsets.int32");
  d_sent_idx_ = OpenDump(dump_, "prompt-sent-index.int32");
  PutI32(d_sent_idx_, 0);
  d_sent_pos_ = OpenDump(dump_, "prompt-sent-pos.int32");
  d_sent_cls_ = OpenDump(dump_, "prompt-sent-cls.uint8");
  VocabBytes();
  StartArticle();
  article_ = 0;
  article_start_ = 0;
  st_.articles = 1;
}

COLD_SIZE_ATTR CoderPrompt::~CoderPrompt() {
  for (int s = 0; s < kSlots; ++s) {
    if (d_tok_[s].f) fclose(d_tok_[s].f);
    if (d_off_[s]) fclose(d_off_[s]);
  }
  if (d_title_) fclose(d_title_);
  if (d_a2_idx_) fclose(d_a2_idx_);
  if (d_a2_off_) fclose(d_a2_off_);
  if (d_sent_idx_) fclose(d_sent_idx_);
  if (d_sent_pos_) fclose(d_sent_pos_);
  if (d_sent_cls_) fclose(d_sent_cls_);
}

void CoderPrompt::EmitBlock(int slot, const unsigned char* p, int n) {
  if (d_tok_[slot].f && n) fwrite(p, 1, (size_t)n, d_tok_[slot].f);
  off_run_[slot] += n;
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
  gate_run_ = 0;
  last_sent_pos_ = -1;
}

COLD_SIZE_ATTR void CoderPrompt::EndArticle() {
  redir_last_ = redir_now_;
  // The training side discards a digit run whose end falls on the last article
  // position.  In the stream that position always carries a separator token,
  // i.e. no digit; the guard checks it instead of assuming it.
  if (last_sent_pos_ >= 0 && last_sent_pos_ >= pos_ - 1) {
    transformer::Fail("prompt: sentinel %lld on the last article position (%lld)", last_sent_pos_,
        pos_ - 1);
  }
  for (int s = 0; s < kSlots; ++s) PutI32(d_off_[s], off_run_[s]);
  PutI32(d_title_, title_len_);
  PutI32(d_a2_idx_, a2_run_);
  PutI32(d_sent_idx_, sent_run_);
}

COLD_SIZE_ATTR void CoderPrompt::Finish() {
  // If the stream ends exactly on a separator sequence, the coder has opened
  // an empty trailing piece.  The training package does not have it, so it
  // gets no row.
  if (pos_ == article_start_ && article_ > 0) {
    --st_.articles;
    return;
  }
  EndArticle();
  for (int s = 0; s < kSlots; ++s) {
    if (d_tok_[s].f) fflush(d_tok_[s].f);
    if (d_off_[s]) fflush(d_off_[s]);
  }
  if (d_title_) fflush(d_title_);
  if (d_a2_idx_) fflush(d_a2_idx_);
  if (d_a2_off_) fflush(d_a2_off_);
  if (d_sent_idx_) fflush(d_sent_idx_);
  if (d_sent_pos_) fflush(d_sent_pos_);
  if (d_sent_cls_) fflush(d_sent_cls_);
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
    }
    if (n_roh == 0) { ++st_.meta_feld_leer; continue; }
    // Byte-set watch: every byte of the row must live in the frozen 208 set,
    // otherwise there is no token for it.  A byte outside it is DROPPED and
    // counted -- not a Fail: the set is frozen against the WRT stream, and a
    // single exotic contributor name must not kill a twenty-hour dump run.
    std::vector<unsigned char> z;
    z.reserve((size_t)n_roh + 1);
    for (int i = 0; i < n_roh; ++i) {
      if (!transformer::InSet(roh[i])) { ++st_.meta_byte_fremd; continue; }
      z.push_back((unsigned char)transformer::kByteToIndex[roh[i]]);
    }
    if (z.empty()) { ++st_.meta_feld_leer; continue; }
    z.push_back(kParagraph);
    {
      std::vector<unsigned char> m;
      m.reserve(z.size() + 4);
      int lauf = 0;
      for (size_t i = 0; i < z.size(); ++i) {
        const unsigned char bv = VocabBytes()[z[i]];
        const bool ziffer = (bv >= 0x30 && bv <= 0x39);
        if (!ziffer && lauf > 0) {
          m.push_back((unsigned char)(208 + LengthClass(lauf)));
          ++st_.meta_sent;
          if (fnr == 0) ++st_.meta_sent_id;
          else if (fnr == 1) ++st_.meta_sent_autor;
          else ++st_.meta_sent_datum;
          lauf = 0;
        }
        m.push_back(z[i]);
        if (ziffer) ++lauf;
      }
      if (lauf > 0) {
        m.push_back((unsigned char)(208 + LengthClass(lauf)));
        ++st_.meta_sent;
        if (fnr == 0) ++st_.meta_sent_id;      // end-of-line token,
        else if (fnr == 1) ++st_.meta_sent_autor;
        else ++st_.meta_sent_datum;
      }
      z.swap(m);
    }
    // The cap is on the BLOCK, not the row: a row that would break it is
    // dropped WHOLE, because a truncated author name is a different name.
    if ((int)b.size() + (int)z.size() > 64) {
      ++st_.meta_zeile_gekappt;
      continue;
    }
    tok[fnr] = z.size();
    b.insert(b.end(), z.begin(), z.end());
    ++st_.meta_zeilen;
  }
  if (!eintrag) ++st_.meta_ohne_eintrag;
  st_.meta_tok_id += tok[0];
  st_.meta_tok_autor += tok[1];
  st_.meta_tok_datum += tok[2];
  st_.meta_tok_k += tok[3];
}

COLD_SIZE_ATTR void CoderPrompt::BuildPrompt() {
  post_.clear();
  const int sep_len = transformer::SeparatorLength();
  const unsigned char* sep = transformer::SeparatorIndices();

  if (a_.z1t) {                                  // article number
    std::vector<unsigned char> b;
    NumberTokens(article_, &b);
    for (int i = 0; i < sep_len; ++i) b.push_back(sep[i]);
    EmitBlock(1, b.data(), (int)b.size());
    st_.z1t_tokens += b.size();
    post_.insert(post_.end(), b.begin(), b.end());
  }

  meta_von_ = (int)post_.size();
  meta_bis_ = meta_von_;

  alias_table::Tab& tab = alias_table::tab;
  std::vector<unsigned char> tb(title_.begin(),
      title_.begin() + (title_len_ - TitleEndLen()));
  for (size_t i = 0; i < tb.size(); ++i) tb[i] = VocabBytes()[tb[i]];
  if (a_.syn) {                                  // synonyms
    // A title longer than kMaxTitleTokens is not in the redirect table of the
    // training side, so it gets no synonym block.
    std::vector<alias_table::Ref> q;
    if (title_len_ <= kMaxTitleTokens) {
      const alias_table::Wahl w = {p_.syn_hops, kMaxTitleTokens - TitleEndLen(), &doc_,
          false, &alias_substring_dedup::g_syn_ges, &alias_substring_dedup::g_syn};
      tab.Kandidaten(tb.data(), (int)tb.size(), w, &q);
    }
    std::vector<unsigned char> b;
    int summe = 0;
    const int take = (int)q.size() < p_.syn_max ? (int)q.size() : p_.syn_max;
    alias_table::Syn().Neu(alias_table::SchluesselHash(tb.data(), (int)tb.size()));
    for (int i = 0; i < take; ++i) {
      const int n = q[(size_t)i].len;
      if (summe + n + TitleEndLen() > p_.syn_tokens) break;
      const unsigned char* d = tab.Data(q[(size_t)i]);
      for (int k = 0; k < n; ++k) b.push_back((unsigned char)transformer::kByteToIndex[d[k]]);
      b.insert(b.end(), TitleEnd(), TitleEnd() + TitleEndLen());
      summe += n + TitleEndLen();
      alias_table::Syn().Merke(tab.K(q[(size_t)i]));
    }
    EmitBlock(2, b.data(), (int)b.size());
    if (!b.empty()) ++st_.syn_articles;
    st_.syn_tokens += b.size();
    post_.insert(post_.end(), b.begin(), b.end());
  }
  if (a_.alias) {                                // aliases
    std::vector<alias_table::Ref> id_list;
    const alias_table::Wahl w = {p_.alias_hops, 0, &doc_, true,
        &alias_substring_dedup::g_p1_ges, &alias_substring_dedup::g_p1};
    tab.Kandidaten(tb.data(), (int)tb.size(), w, &id_list);
    std::vector<unsigned char> b;
    for (size_t i = 0; i < id_list.size(); ++i) {
      const int n = id_list[i].len;
      if ((int)b.size() + 1 + n > p_.alias_cap) break;
      b.push_back(kParagraph);
      const unsigned char* d = tab.Data(id_list[i]);
      for (int k = 0; k < n; ++k) b.push_back((unsigned char)transformer::kByteToIndex[d[k]]);
    }
    EmitBlock(3, b.data(), (int)b.size());
    if (!b.empty()) ++st_.alias_articles;
    st_.alias_tokens += b.size();
    post_.insert(post_.end(), b.begin(), b.end());
  }
  if (!post_.empty()) have_post_ = true;
}

void CoderPrompt::TitleClosed(int title_len) {
  // Check: `title_len_` indexes `title_` in the heading-reminder rows.  The
  // two bounds in Observe() are one apart on purpose; changing one without
  // the other stops here instead of reading past the end.
  if (title_len > (int)title_.size()) {
    transformer::Fail("prompt: title length %d exceeds the stored title (%d tokens)", title_len,
        (int)title_.size());
  }
  title_len_ = title_len;                        // including the terminator
  ++st_.titles;
  if (!no_prompt_) BuildPrompt();
}

void CoderPrompt::ZeilenMarke(unsigned char b) {
  if (b == 0x0A) {
    nonspace_since_nl_ = 0;
    marker_line_ = false;
  } else if (b != 0x20) {
    ++nonspace_since_nl_;
    if (nonspace_since_nl_ == 1
        && (b == 0x2A || b == 0x23 || b == 0x7C || b == 0x21 || b == 0x3B
            || b == 0x3A)) {
      marker_line_ = true;
    }
  }
}

void CoderPrompt::ScanRedirectPage(unsigned char token) {
  if (rd_take_ > 0) {
    --rd_take_;
    if (rd_take_ == 0) {
      rd_take_ = -1;
      const redirect_table::Rec* rr = redirect_table::tab.Seite(page_);
      if (rr && rr->marke) {
        redir_now_ = true;
        ++st_.redirect_pages;
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
  ++page_;
  ++st_.articles;
  article_start_ = pos_;
  in_title_ = true;
  title_len_ = 0;
  title_.clear();
  a2_count_ = 0;
  redir_now_ = false;
  rd_open_ = 0;
  rd_take_ = -1;
  gate_run_ = 0;
  last_sent_pos_ = -1;
  no_prompt_ = true;
  prior_next_ = a_.meta && !1;
}

// ---- Observe --------------------------------------------------------------
void CoderPrompt::Observe(unsigned char token, bool piece_end) {
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
    have_pre_ = false;
    have_post_ = false;
    sent_pre_ = false;
    pre_.clear();
    meta_pre_von_ = 0;
    meta_pre_bis_ = 0;
    rd_pre_von_ = 0;
    rd_pre_bis_ = 0;
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

  // --- digit-run automaton (as zctx/zgate.py gate_mask), state after this token
  const bool is_digit = (bv >= 0x30 && bv <= 0x39);
  if (is_digit) ++digit_run_; else digit_run_ = 0;
  if (prev_byte_ == 0x5B && bv == 0x5B) ++link_depth_;          // `[[`
  if (prev_byte_ == 0x5D && bv == 0x5D && link_depth_ > 0) --link_depth_;
  {
    const unsigned char* alt = nullptr;
    const int an = redirect_table::tab.Alt(&alt);
    for (int k = 0; k < an; ++k) ZeilenMarke(alt[k]);
  }
  prev_byte_ = bv;
  const bool excl = (link_depth_ > 0) || marker_line_;
  const bool gate = (digit_run_ >= p_.gate_m) && !(p_.gate_exclude && excl);
  if (gate) ++st_.gate_positions;

  while (doc_.size() <= alias_table::tab.Quellen_()) doc_.push_back((uint32_t)article_);
  ScanRedirectPage(token);

  have_pre_ = false;
  have_post_ = false;
  sent_pre_ = false;
  pre_.clear();
  meta_pre_von_ = 0;
  meta_pre_bis_ = 0;
  rd_pre_von_ = 0;
  rd_pre_bis_ = 0;
  const long long q = pos_;
  ++pos_;

  if (piece_end) {
    EndArticle();
    ++article_;
    ++page_;
    ++st_.articles;
    article_start_ = pos_;
    StartArticle();
    digit_run_ = 0;
    prior_next_ = a_.meta && !1;
    piece_rest_ = false;       // the next article is whole again
    return;
  }

  // --- sentinel row: a digit run has ended --------------------------------
  if (gate) {
    ++gate_run_;
  } else if (gate_run_ > 0) {
    const int cls = LengthClass(gate_run_);
    ++st_.sent_sites;
    ++st_.sent_class[cls];
    PutI32(d_sent_pos_, (int32_t)q);
    if (d_sent_cls_) {
      const unsigned char c = (unsigned char)cls;
      fwrite(&c, 1, 1, d_sent_cls_);
    }
    ++sent_run_;
    last_sent_pos_ = q;
    if (piece_rest_) ++st_.sent_piece_rest;
    if (in_title_) ++st_.sent_in_titel;
    const bool sent_titel_ok = !in_title_;
    if (a_.sent && 1 && !piece_rest_ && sent_titel_ok
        && q >= article_start_ + title_len_) {
      pre_.push_back((unsigned char)(208 + cls));
      have_pre_ = true;
      sent_pre_ = true;
      ++st_.sent_fed;
    }
    gate_run_ = 0;
  }

  // --- article number before the title (number-first form only) ------------
  if (a_.num && q == article_start_) {
    std::vector<unsigned char> b;
    NumberTokens(article_, &b);
    EmitBlock(0, b.data(), (int)b.size());
    st_.num_tokens += b.size();
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
  if (a2_hit && title_len_ == 0 && !in_title_) ++st_.a2_ohne_titel;
  if (title_len_ > 0 && a2_hit) {
    const long long site = q + 1;                // as the training side places it
    ++st_.a2_sites;
    if (piece_rest_) ++st_.a2_piece_rest;
    PutI32(d_a2_off_, (int32_t)site);
    ++a2_run_;
    if (a_.a2 && !piece_rest_ && !no_prompt_
        && site >= article_start_ + title_len_) {
      ++a2_count_;
      ++st_.a2_used;
      std::vector<unsigned char> b;
      if (a_.a2n) {
        std::vector<unsigned char> nb;
        NumberTokens(a2_count_, &nb);
        EmitBlock(4, nb.data(), (int)nb.size());
        st_.a2n_tokens += nb.size();
        b.insert(b.end(), nb.begin(), nb.end());
      }
      b.insert(b.end(), title_.begin(), title_.begin() + title_len_);
      st_.a2_tokens += b.size();
      // The heading reminder must be fed before a sentinel row.  The two
      // cannot meet in one step with this grammar: a2 needs `token ==
      // kParagraph` with two kEq before it, a sentinel row needs a digit at
      // the previous position, and `=` is no digit.  The engine feeds the pre
      // block first, so if they ever met the order would be reversed; a
      // changed dictionary or gate stops here instead.
      if (sent_pre_) {
        transformer::Fail("prompt: section title and sentinel meet at %lld", q);
      }
      if (have_post_) {
        transformer::Fail("prompt: section title at %lld overwrites a post-block (%d tokens)", q, (int)post_.size());
      }
      post_ = b;
      meta_von_ = 0;
      meta_bis_ = 0;
      have_post_ = true;
    }
  }

  // --- the distribution for the next token --------------------------------
  prior_next_ = false;
  if (a_.titelmask && in_title_ && !no_prompt_) {
    prior_next_ = true;
    ++st_.masked_title;
  }
  if (a_.sent && a_.sent_prior && gate) prior_next_ = true;
  if (prior_next_) ++st_.prior_positions;
}

void CoderPrompt::SeitenVorblock(long long q) {
  if (ImRedirectBlock() && q == article_start_) {
    unsigned char rd[4] = {Hash(), ByteTok(0x72), ByteTok(0x64), kParagraph};
    rd_pre_von_ = (int)pre_.size();
    pre_.insert(pre_.end(), rd, rd + 4);
    rd_pre_bis_ = (int)pre_.size();
    ++st_.rd_seiten;
    st_.rd_tokens += 4;
    have_pre_ = true;
  }

  if (a_.meta && q == article_start_) {
    std::vector<unsigned char> b;
    BuildMeta(page_, &b);
    if (!b.empty()) {
      ++st_.meta_seiten;
      ++st_.meta_vorn_seiten;
      st_.meta_vorn_tokens += b.size();
      EmitBlock(5, b.data(), (int)b.size());
      meta_pre_von_ = rd_pre_bis_ > 0 ? rd_pre_von_ : (int)pre_.size();
      pre_.insert(pre_.end(), b.begin(), b.end());
      meta_pre_bis_ = (int)pre_.size();
      have_pre_ = true;
    }
  }

}

bool CoderPrompt::TakeSeitenVorblock(const unsigned char** toks, int* n,
    int* meta_von, int* meta_bis) {
  if (!a_.meta || pos_ != article_start_) return false;
  pre_.clear();
  have_pre_ = false;
  meta_pre_von_ = 0;
  meta_pre_bis_ = 0;
  rd_pre_von_ = 0;
  rd_pre_bis_ = 0;
  SeitenVorblock(pos_);
  if (!have_pre_) return false;
  have_pre_ = false;
  *toks = pre_.data();
  *n = (int)pre_.size();
  *meta_von = meta_pre_von_;
  *meta_bis = meta_pre_bis_;
  return true;
}

void CoderPrompt::PieceCut() {
  piece_rest_ = true;
  gate_run_ = 0;
  ++st_.piece_cuts;
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

}

