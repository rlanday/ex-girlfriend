#ifndef ALIAS_CONTEXT_H_
#define ALIAS_CONTEXT_H_


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>
#include <vector>

#include "alias_context_set.h"
#include "string_pool.h"
#include "alias_table.h"
#include "alias_substring_dedup.h"
#include "alias_title_subst.h"
#include "redirect_table.h"
#undef ALIAS_OPEN_LEN
#undef ALIAS_OPEN_INIT
#define ALIAS_OPEN_LEN 8
#define ALIAS_OPEN_INIT { 0x20, 0x20, 0x4c, 0x2f, 0xdf, 0x98, 0x4e, 0x0a }
#undef ALIAS_CLOSE_LEN
#undef ALIAS_CLOSE_INIT
#define ALIAS_CLOSE_LEN 1
#define ALIAS_CLOSE_INIT { 0x0a }

namespace alias_context {

static const unsigned char kOpen[ALIAS_OPEN_LEN] =
    ALIAS_OPEN_INIT;
static const unsigned char kClose[ALIAS_CLOSE_LEN] =
    ALIAS_CLOSE_INIT;

constexpr unsigned char kNL = 0x0a;
constexpr unsigned char kLB = 0x5b;
constexpr unsigned char kRB = 0x5d;
constexpr unsigned char kHash = 0x23;
constexpr unsigned char kPipe = 0x51;
constexpr unsigned char kColon = 0x4a;
constexpr unsigned char kUnder = 0x5f;
constexpr unsigned char kBlank = 0x20;
constexpr bool kLinkDouble = 0 != 0;
constexpr int kMaxTitle = 64;   // as zctx/zgrammar.py MAX_TITLE_TOKENS
constexpr int kMinTitle = 3;
[[noreturn]] static inline void Fail(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  fflush(nullptr);
  exit(1);
}

using string_pool::StrPool;

struct Stats {
  unsigned long long articles = 0;      // title spans seen
  unsigned long long titles = 0;        // of them long enough for a prompt
  unsigned long long blocks = 0;        // articles with a non-empty block
  unsigned long long bytes = 0;         // reinjected bytes
  unsigned long long links = 0;
  unsigned long long links_piped = 0;
  unsigned long long links_dropped = 0;
  unsigned long long redirects = 0;
  unsigned long long from_redirect = 0; // entries of part (i)
  unsigned long long from_linktext = 0; // entries of part (ii)
};

class Aut {
 public:
  void Observe(unsigned char b) {
    (void)b;
    have_block_ = false;
    ++pos_;
    const unsigned char* t = nullptr;
    if (alias_table::tab.Anker()) {
      ++st_.articles;
    } else if (const int n = alias_table::tab.TitelZu(&t)) {
      title_.assign(t, t + n);
      TitleClosed(n);
    }
  }

  int PraefixLen() const { return praefix_; }
  int TakeBlock(const unsigned char** p) {
    if (!have_block_) return 0;
    have_block_ = false;
    *p = block_.data();
    return (int)block_.size();
  }

  void Finish() {
    st_.links = alias_table::tab.links;
    st_.links_piped = alias_table::tab.piped;
    st_.links_dropped = alias_table::tab.verworfen;
    st_.redirects = alias_table::tab.redirects;
  }

  // Read-only view of the block built for the title that just closed; not used
  // on the coding path.
  const std::vector<unsigned char>& LastBlock() const { return block_; }

  // Statistics of the alias channel on stderr, left out of the package form.
  void Report(const char* side) const {
    (void)side;

    const bool w_link = (st_.links > 1000 && st_.links_piped == 0);
    const double je_mb = pos_ > 0
        ? (double)st_.blocks * 1048576.0 / (double)pos_ : 0.0;
    const bool w_leer = (st_.articles > 100
                         && je_mb < (double)1);
    const int wacht = (w_link ? 1 : 0) + (w_leer ? 1 : 0);
  }

  const Stats& GetStats() const { return st_; }

 private:
  // --- state -------------------------------------------------------------
  long long pos_ = 0;
  std::vector<unsigned char> title_;

  // prev_block_: its reader was removed; the member is kept so
  // the object layout and therefore the binary stay identical.
  std::vector<unsigned char> block_, prev_block_;
  int praefix_ = 0;
  bool have_block_ = false;

  Stats st_;

  void BlockZiel(int title_len, std::vector<unsigned char>* bp, size_t* praefix) {
    std::vector<unsigned char>& b = *bp;
    (void)praefix;
    alias_table::Tab& tab = alias_table::tab;
    std::vector<alias_table::Ref> id_list;
    int nq = 0, nt = 0;
    const alias_table::Wahl w = {12, 0, nullptr, true,
        &alias_substring_dedup::g_p2_ges, &alias_substring_dedup::g_p2};
    tab.Kandidaten(title_.data(), title_len, w, &id_list, &nq, &nt);
    st_.from_redirect += (unsigned long long)nq;
    st_.from_linktext += (unsigned long long)nt;
    {
      alias_table::SynMenge& syn = alias_table::Syn();
      if (syn.Passt(alias_table::SchluesselHash(title_.data(), title_len))) {
        size_t w2 = 0;
        for (size_t i = 0; i < id_list.size(); ++i) {
          ++syn.n_ges;
          if (syn.Kennt(tab.K(id_list[i]))) { ++syn.n_p2; continue; }
          id_list[w2++] = id_list[i];
        }
        id_list.resize(w2);
      }
      syn.Verbraucht();
    }
    {                     // count title lines; the longest entry that
      ++alias_title_subst::g_seiten;   // contains the title is dropped below
      uint64_t th = 0;
      const std::vector<unsigned char>& tk = tab.TitelSchluessel(&th);
      size_t best = id_list.size();
      int best_len = -1;
      for (size_t i = 0; i < id_list.size(); ++i) {
        if (id_list[i].len <= best_len) continue;
        if (!tab.Teil(id_list[i], tk.data(), (int)tk.size(), th, false)) continue;
        best = i;
        best_len = id_list[i].len;
      }
      if (best < id_list.size()) {
        const int n = id_list[best].len;
        ++alias_title_subst::g_ersetzt;
        alias_title_subst::g_titel_bytes += (unsigned long long)title_len;
        alias_title_subst::g_ersatz_bytes += (unsigned long long)n;
        alias_title_subst::g_block_gespart += (unsigned long long)(n + 1);
        id_list.erase(id_list.begin() + (long)best);
      }
    }
    for (size_t i = 0; i < id_list.size(); ++i) {
      const int n = id_list[i].len;
      if ((int)b.size() + 1 + n > 64) break;
      b.push_back(kNL);
      b.insert(b.end(), tab.Data(id_list[i]), tab.Data(id_list[i]) + n);
    }
  }

  void TitleClosed(int title_len) {
    ++st_.titles;
    std::vector<unsigned char> b;
    size_t ersatz_praefix = 0;
    BlockZiel(title_len, &b, &ersatz_praefix);
    block_.swap(b);
    praefix_ = (int)ersatz_praefix;
    if (block_.size() > ersatz_praefix) {
      ++st_.blocks;
      st_.bytes += block_.size() - ersatz_praefix;
    }
    if (!block_.empty()) have_block_ = true;
  }

};

inline Aut aut;

}

#endif
