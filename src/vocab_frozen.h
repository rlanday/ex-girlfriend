#ifndef VOCAB_FROZEN_H_
#define VOCAB_FROZEN_H_

#include <stdio.h>

// The frozen byte set of the coder stream: exactly 208 byte values, stored as
// a 32-byte bitmap (LSB first), and the map from byte value to its index in
// the set.
namespace vocab_frozen {

// Streams of the build helper use the full 256-value alphabet; the flag is
// also carried in the stream header.  Both sides set it before the predictor
// is built.
inline bool g_aux = false;

// The stamped tables: read-only data, never computed, every value known to
// the compiler.
constexpr unsigned char kBitmap[32] = { 0xfe, 0x1e, 0x00, 0x00, 0xff, 0xff, 0xff, 0x03, 0x01, 0xfc, 0x0f, 0xf9, 0xfe, 0xff, 0xff, 0x07, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
constexpr int kByteMap[256] = { 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 48, 48, 48, 48, 49, 49, 49, 50, 51, 52, 53, 54, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 80, 80, 80, 80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207 };

// --- compile-time checks (no code in the binary) --------------------------
constexpr int PopBits(unsigned char b) {
  return ((b >> 0) & 1) + ((b >> 1) & 1) + ((b >> 2) & 1) + ((b >> 3) & 1) +
         ((b >> 4) & 1) + ((b >> 5) & 1) + ((b >> 6) & 1) + ((b >> 7) & 1);
}
constexpr int PopAll(int i = 0) {
  return i == 32 ? 0 : PopBits(kBitmap[i]) + PopAll(i + 1);
}
static_assert(PopAll() == 208,
    "bitmap and frozen set size do not match.");
static_assert(kByteMap[255] + ((kBitmap[31] >> 7) & 1) == 208,
    "byte_map is not a prefix sum of this bitmap.");

constexpr bool InSet(int v) { return ((kBitmap[v >> 3] >> (v & 7)) & 1) != 0; }

inline void PutLen(FILE* out, unsigned int len) {
  for (int s = 28; s >= 0; s -= 7) putc(0x80 | ((len >> s) & 0x7F), out);
}
inline int GetLen(FILE* in) {
  unsigned int len = 0;
  for (int i = 0; i < 5; ++i) {
    int c = getc(in);
    len = (len << 7) | (unsigned int)(c & 0x7F);
  }
  return (int)len;
}

constexpr int kTypeDefault = 0x01;
constexpr int kWrtOff = 0x01;
constexpr int kWrtOn = 0x07;

inline void PutHeader(FILE* out, int type, unsigned int len) {
  putc(type ? type : kTypeDefault, out);
  PutLen(out, len);
}

static_assert(InSet(kTypeDefault) && InSet(kWrtOn) && InSet(kWrtOff),
    "the control bytes of the segment header are not in the "
    "frozen set.");

template <class VocabVec>
inline void FillVocab(VocabVec* vocab) {
  if (g_aux) { for (int i = 0; i < 256; ++i) (*vocab)[i] = true; return; }
  for (int i = 0; i < 256; ++i) (*vocab)[i] = InSet(i);
}

constexpr int kColClamp(int i) {
  return kByteMap[i] < 208 ? kByteMap[i] : 208 - 1;
}

}

#endif
