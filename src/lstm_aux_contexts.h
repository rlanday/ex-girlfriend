#ifndef LSTM_AUX_CONTEXTS_H_
#define LSTM_AUX_CONTEXTS_H_


#include <stdio.h>
#include <stdlib.h>

namespace lstm_aux_contexts {

constexpr unsigned int kBigN =
    12 ? (1u << (unsigned int)12) : 0u;
constexpr unsigned int kTriN =
    14 ? (1u << (unsigned int)14) : 0u;

// Number of blocks and the index of each one inside cur.col[].
constexpr unsigned int kNB =
    (12 ? 1u : 0u) + (14 ? 1u : 0u) +
    (1 ? 1u : 0u);
constexpr unsigned int kIBig = 0u;
constexpr unsigned int kITri = 12 ? 1u : 0u;
constexpr unsigned int kIMatch = kITri + (14 ? 1u : 0u);

// First column of each block inside the extra array, and its total width.
constexpr unsigned int kBigOff = 0u;
constexpr unsigned int kTriOff = kBigN;
constexpr unsigned int kMatchOff = kBigN + kTriN;
constexpr unsigned int kExtra = kBigN + kTriN;
constexpr unsigned int kBase = 208;

inline void SetGeom(unsigned int base) {
  if (base != kBase) {
    fprintf(stderr,
        "\nabort: the LSTM constructor reports a "
        "vocabulary width of %u, compiled in is %u.  That "
        "number is the size of the frozen byte set and follows it.\n",
        base, kBase);
    fflush(stderr);
    abort();
  }
}

// Total number of extra columns: BIG + TRI + the match block (vocab + 1).
constexpr unsigned int Extra() {
  return kExtra + kBase + 1u;
}

// Column of the thin match block.  `sym` is 0..vocab, with vocab reserved for
// "no usable match" - that is why the block is vocab + 1 wide.
inline unsigned int MatchCol(unsigned int sym) { return kMatchOff + sym; }

// Hash of the pair (s1, s2) into the BIG block.  Integer only.
inline unsigned int BigCol(int s1, int s2) {
  unsigned int h = (unsigned int)s1 * 2654435761u ^
                   ((unsigned int)s2 * 2246822519u);
  h ^= h >> 15;
  return kBigOff + (h & (kBigN - 1u));
}

// Hash of the triple (s1, s2, s3) into the TRI block.  Three odd multipliers
// plus two xor-shifts: each symbol reaches the low bits that the mask keeps,
// so a change in ANY of the three moves the bucket.
inline unsigned int TriCol(int s1, int s2, int s3) {
  unsigned int h = (unsigned int)s1 * 2654435761u;
  h ^= (unsigned int)s2 * 2246822519u;
  h ^= (unsigned int)s3 * 3266489917u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  return kTriOff + (h & (kTriN - 1u));
}

// The absolute column of every block for the step currently being computed.
// All layers use the same ones; Lstm sets them immediately before each
// forward/backward call.  Keeping the COLUMNS here (instead of the symbols)
// leaves the LstmLayer signatures untouched.  There is exactly one Lstm
// object.
struct Cur {
  unsigned int col[kNB];
};
inline Cur cur = {};

// Ring of the recently coded symbols.  Power of two; three entries are read,
// the size only has to exceed that.
constexpr unsigned int kRingN = 256u;

struct Ring {
  unsigned char buf[kRingN];
  unsigned long long cnt;
  void Reset() {
    for (unsigned int i = 0; i < kRingN; ++i) buf[i] = 0;
    cnt = 0;
  }
  void Push(unsigned int s) {
    buf[cnt & (unsigned long long)(kRingN - 1u)] = (unsigned char)s;
    ++cnt;
  }
  // buf[cnt-1-j], i.e. j+1 bytes before the byte that is about to be pushed.
  // Must be called BEFORE Push of the current byte.
  int Raw(unsigned int j) const {
    const unsigned long long d = (unsigned long long)j + 1ull;
    if (cnt < d) return 0;
    return (int)buf[(cnt - d) & (unsigned long long)(kRingN - 1u)];
  }
};

}

#endif
