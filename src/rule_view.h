// Reader view of the reversible text rules, for the page readers of the
// transformer.  The alias table, the prompt automaton and z3 look titles and
// link targets up by key; they read the ORIGINAL bytes, the stream as it was
// before the rule stages.  The view inverts the stages online, in the coder
// and in the decoder alike, from the bytes the readers are fed
// (Tf1Engine::Step, page text only): identity up to the end of the rule
// region's first anchor, then the inverse stages.  The
// structural bytes LF, '[', '|', ']' map one to one; on each of them the view
// releases the original bytes up to it.  An article whose follow word is not
// read yet holds the release back to the structural byte that completes it
// (always within its line); z3 reads the released bytes, so it sees the same
// bytes in the same order.
//   Byte(b)   every page byte, before every reader
//   Letzt()   original position of the structural byte fed last
//   Frei()    original bytes released by the byte just fed
//   Titel()   original title line of the running page
//   At(x)     original byte at position x (Letzt() - x < 2^17)
#ifndef RULE_VIEW_H_
#define RULE_VIEW_H_
#include "u8_append.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include "enc_only.h"
#ifndef RS_ENC_ONLY
#define RS_ENC_ONLY ENC_ONLY
#endif
#include "meld.h"
#include "rule_stages.h"

namespace rule_view {

inline rule_stages::Sicht sicht;
inline rule_stages::DictSpeicher dict;
inline long long anker = 0;        // page anchors fed so far (up to the rule region)
inline long long gefuettert = 0;   // bytes fed
inline uint64_t t8 = 0;            // the last eight bytes
inline bool z3_orig = false;       // z3 reads the view (set behind the redirect block, tf1_engine.cpp)

__attribute__((noinline)) inline void Beginn() {
  // ./.dict exists on both sides since the end of slot 1 (dict_article::Schreiben)
  std::vector<unsigned char> d;
  FILE* f = fopen(".dict", "rb");
  if (f) {
    unsigned char b[1 << 16];
    size_t k;
    while ((k = fread(b, 1, sizeof b, f)) > 0) u8_append::Anh(d, b, b + k);
    fclose(f);
  }
  if (d.empty() || !dict.FromBytes(d.data(), d.size())) {
    msg_stop(555);
    fflush(nullptr);
    exit(1);
  }
  sicht.Init(dict.D());
  sicht.Start((size_t)gefuettert);
}
inline void Byte(unsigned char b) {
  sicht.Byte(b);
  ++gefuettert;
  t8 = t8 << 8 | b;
  if (t8 == 0x20204c2fdf984e0aULL && anker < 71361 && ++anker == 71361) Beginn();
  if (sicht.Err()) {
    fprintf(stderr, "E556\n");
    fflush(nullptr);
    exit(1);
  }
}
inline long long Letzt() { return sicht.Letzt(); }
inline int Frei(const unsigned char** p) { return sicht.Frei(p); }
inline int Titel(const unsigned char** p) { return sicht.Titel(p); }
inline bool TitelJetzt() { return sicht.TitelJetzt(); }
inline unsigned char At(long long x) { return sicht.O()[x]; }

}  // namespace rule_view
#endif
