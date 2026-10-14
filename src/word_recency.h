
#ifndef WORD_RECENCY_H_
#define WORD_RECENCY_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Word recency: distance in model bytes since the current word prefix was
// last seen, bucketed into 3 bits for an APM context.
namespace wordrec {

enum { kRecBits = 18 ? 18 : 20,
       kRecSize = 1u << kRecBits };

inline unsigned int* RecAlloc() {
  return (unsigned int*)calloc(kRecSize, sizeof(unsigned int));
}
inline unsigned int* rec_last = RecAlloc();
inline unsigned int  rec_bucket = 0;   // 0 = no word open, 1 = never seen,
                                       // 2..7 = increasing distance
inline unsigned long long rec_n = 0, rec_new = 0;

// whash = word0 (rolling prefix hash), bpos = byte pos in the model stream.
// Called on EVERY byte inside a word, not only at the end of a word.
inline void PrefixSeen(unsigned int whash, unsigned int bpos) {
  if (whash == 0 || rec_last == 0) { rec_bucket = 0; return; }
  const unsigned int k = (whash * 2654435761u) >> (32 - kRecBits);
  const unsigned int last = rec_last[k];
  const unsigned int now = bpos + 1;            // 0 stays "never seen"
  if (last == 0) {
    rec_bucket = 1; ++rec_new;
  } else {
    const unsigned int d = now - last;
    rec_bucket = d <= 256u    ? 2u
               : d <= 1024u   ? 3u
               : d <= 4096u   ? 4u
               : d <= 16384u  ? 5u
               : d <= 262144u ? 6u : 7u;
  }
  rec_last[k] = now;
  ++rec_n;
}

inline unsigned int ApmExtra() {
  return rec_bucket & 7u;
}

}

#endif
