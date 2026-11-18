// ISBN-10 check characters out of the text.  The check character of a correct
// ISBN-10 follows from its nine digits, so the text stage in front of the WRT
// removes it and the inverse stage computes it again.
//   -e: Pack() on .ready4cmix right before the compression (the last text
//       stage; census line, metadata block codec, IP stage and text start
//       work on the coder stream behind the WRT).
//   archive9: Put() is a byte filter on the WRT inverse of the main stream
//       (preprocessor::Decode hands every byte to Put() instead of putc()).
// Preprocessor: a correct ISBN-10 check character is removed.  Where the
//   recognizer fires on the original (a genuine 9-digit "ISBN"), the mark
//   byte is inserted at P.  A wrong check character stays: the number keeps
//   ten characters, and the recognizer does not fire on it.
// Postprocessor: where the recognizer fires, it drops the mark at P if
//   there is one and inserts the computed check character otherwise.
// Recognizer (Step() of the recognizer state, the same on both sides): the word "isbn" in any case; up
//   to 3 of ' ' '-' ':' '=' TAB; nine digits, between them runs of at most 3
//   ' '/'-'.  If the nine digits carry a separator, P lies behind the one
//   ' '/'-' that follows the ninth digit (none: no site), otherwise directly
//   behind the ninth digit.  It fires at P iff the byte there ends the number:
//   no digit, no X/x, no ' '/'-' followed by a digit or X/x.  The separator in
//   front of the check character stays.  A separator behind an unseparated
//   body ("047026844-1") is no place of a check character: a removed
//   "0643090681" followed by '-' would read the same (on enwik9 a '-' would
//   allow 13 more removals but block 27, a ' ' would block 3,526).
// The coder side (Isbn2State) finds P in the coder stream by itself; the check
//   characters the stage left are ordinary tokens of the transformer.
// The mark is 0x0B (kMarke): it must end a number for the recognizer and pass
//   the WRT as one byte of the 208-value set.  0x0B differs from the most
//   frequent byte behind an ISBN-10 (line feed, 44 %) only in the last bit and
//   never follows an ISBN-10 in enwik9.
#ifndef ISBN_CHECK_H_
#define ISBN_CHECK_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace isbn_check {

static const unsigned int kMarke = 0x0B;
static_assert(kMarke < 0x80 && !(kMarke >= '0' && kMarke <= '9') &&
              kMarke != 'X' && kMarke != 'x' && kMarke != ' ' && kMarke != '-' &&
              kMarke != 0x06 && kMarke != 0x07 && kMarke != 0x0C && kMarke != 0x40 &&
              !((kMarke | 0x20u) >= 'a' && (kMarke | 0x20u) <= 'z'),
    "the marker must end a number and pass the WRT as one byte");

inline bool Ziffer(unsigned int c) { return c - '0' < 10u; }
inline bool Zehnt(unsigned int c) { return Ziffer(c) || (c | 0x20u) == 'x'; }
inline bool Trenner(unsigned int c) { return c == ' ' || c == '-'; }
inline bool Buchst(unsigned int c) { return (c | 0x20u) - 'a' < 26u; }

// ------------------------------------------------------------ text stage
// The buffer of Pack() ends in a zero byte: a zero is no digit, no
// separator and no letter, so no length guard is needed.
// Does the byte at p end the number?
inline bool Endet(const unsigned char* d, size_t p) {
  if (Zehnt(d[p])) return false;
  return !(Trenner(d[p]) && Zehnt(d[p + 1]));
}

struct IsbnTail {
  unsigned char an;    // set in the archive9 branch: only the main stream
  unsigned char kw;    // letters of the current word matched against "isbn", 5: another word
  unsigned char zu;    // 0 free, 1 prefix and digits, 2 behind the ninth digit of a
                       // separated body, 3 at P, 4 at P with a separator held back
  unsigned char nd, gap, sep, hold;
  int s;               // check sum, from the ninth digit on the check character
  void Step(unsigned int u);
  void Put(int c, FILE* o);
};
inline IsbnTail after_at = {0, 5, 0, 0, 0, 0, 0, 0};   // kw 5: no keyword at byte 0

// The recognizer on one byte of the text outside P (zu < 3).
__attribute__((noinline, minsize)) inline void IsbnTail::Step(unsigned int u) {
  if (Buchst(u)) {                       // a letter: the word grows
    kw = (kw < 4 && (u | 0x20u) == (unsigned char)"isbn"[kw]) ? kw + 1 : 5;
    zu = 0;
    return;
  }
  if (kw == 4) {                         // "isbn" as a word, u is the first byte behind it
    zu = 1;
    nd = gap = sep = 0;
    s = 0;
  }
  kw = 0;
  if (zu == 2) {
    zu = Trenner(u) ? 3 : 0;
  } else if (zu == 1) {
    if (Ziffer(u)) {
      s += ++nd * (int)(u - '0');
      gap = 0;
      if (nd == 9) {
        zu = sep ? 2 : 3;
        s %= 11;
        s = s == 10 ? 'X' : '0' + s;
      }
    } else if (Trenner(u) || (!nd && (u == ':' || u == '=' || u == '\t'))) {
      sep |= nd;
      if (++gap > 3) zu = 0;
    } else {
      zu = 0;
    }
  }
}

__attribute__((noinline, minsize)) inline void IsbnTail::Put(int c, FILE* o) {
  if (zu >= 3) {                         // c is the byte at P or the one behind it
    const int z = zu;
    zu = 0;
    if (z == 3 && c == (int)kMarke) return;           // genuine 9-digit number
    if (z == 3 && c >= 0 && Trenner((unsigned int)c)) {
      zu = 4;
      hold = (unsigned char)c;
      return;
    }
    if (c < 0 || !Zehnt((unsigned int)c)) putc(s, o);
    if (z == 4) putc(hold, o);
  }
  if (c < 0) return;
  putc(c, o);
  if (an) Step((unsigned int)c);
}

__attribute__((noinline, minsize, cold)) inline void Pack() {
  FILE* f = fopen(".ready4cmix", "rb");
  if (!f) exit(3);
  fseek(f, 0L, SEEK_END);
  const size_t n = (size_t)ftell(f);
  fseek(f, 0L, SEEK_SET);
  unsigned char* d = (unsigned char*)calloc(n + 1, 1);   // a zero byte behind
  if (!d || fread(d, 1, n, f) != n) exit(3);
  fclose(f);
  f = fopen(".ready4cmix", "wb");
  if (!f) exit(3);
  IsbnTail z = {1, 5, 0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    if (z.zu == 3) {                     // i is P
      z.zu = 0;
      if (d[i] == z.s && Endet(d, i + 1) && d[i + 1] != kMarke) {   // correct: out
        continue;
      }
      if (Endet(d, i)) {
        putc((int)kMarke, f);
      }
    }
    putc(d[i], f);
    z.Step(d[i]);
  }
  if (z.zu == 3) putc((int)kMarke, f);   // a site at the end of the text
  if (fclose(f)) exit(3);
  free(d);
}

}  // namespace isbn_check

#endif
