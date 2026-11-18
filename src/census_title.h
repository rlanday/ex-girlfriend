// The stripped titles of the census pages, for the readers of the title text.
// The census automaton fills them from the census line and follows the pages
// of the coded stream.  The alias table and the link context use the full
// title as their key (" County, Colorado" appended to "Kiowa"); the
// transformer sees the short title behind a context row with the census
// words, the type word and the state (tf1_prompt.cpp CensusZeile).
#ifndef CENSUS_TITLE_H_
#define CENSUS_TITLE_H_
// The injected census blocks reach the transformer as prompt blocks of the
// author class (context rows without a stream position, as in the training
// data).  The pseudo article is slot 3 of the text start, a transformer
// document of its own behind the slot head.
namespace census_title {
// the census automaton injects the head of the pseudo article (author, title);
// the prompt opens a title behind its tab
inline bool pseudo_kopf = false;
// the census automaton waits for a record type or for the line feed that ends
// the census line (set behind every observed byte)
inline bool typ_folgt = false;
inline unsigned long long n_pseudo = 0;
}  // namespace census_title
#include <string>
#include <vector>
#include <algorithm>

namespace census_title {

// the running page: WRT-coded " <type word>, <state>" (empty: no census page)
// and its index in `seiten` (-1: no census page)
inline std::string zusatz;
inline long long zusatz_i = -1;
// all census pages with a stripped title (sorted by page index, absolute), the
// line "<state><type word>" (no reader in this form) and the
// context row of the transformer, all WRT-coded
inline std::vector<long long> seiten;
inline std::vector<std::string> line_buf;
inline std::vector<std::string> kontext;
// the suffix of every stripped title (as `seiten`): readers that keep a page
// past its title (the source titles of the link context) restore it from here
inline std::vector<std::string> zusaetze;

inline const std::string* Kontext(long long page_index) {
  const auto it = std::lower_bound(seiten.begin(), seiten.end(), page_index);
  if (it == seiten.end() || *it != page_index) return nullptr;
  return &kontext[(size_t)(it - seiten.begin())];
}

inline const std::string* TitleLine(long long page_index) {
  const auto it = std::lower_bound(seiten.begin(), seiten.end(), page_index);
  if (it == seiten.end() || *it != page_index) return nullptr;
  return &line_buf[(size_t)(it - seiten.begin())];
}

// statistics of the readers: how many titles got the suffix
inline unsigned long long n_aliastab = 0, n_prompt = 0, n_z3 = 0, n_k1 = 0;
// source titles of the link context that got the suffix back
inline unsigned long long n_z3q = 0;

}  // namespace census_title
#endif
