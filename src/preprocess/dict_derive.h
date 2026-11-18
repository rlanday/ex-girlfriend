#ifndef DICT_DERIVE_H
#define DICT_DERIVE_H

#include <stddef.h>
#include <stdint.h>

namespace dict_derive {

bool BuildDictFromInput(const char* enwik9_path, const char* hints_path,
                        const char* out_path);
// Helpers shared with the article order codec.
// a[0..n) ascending, in place.
void SortU64(uint64_t* a, size_t n);
// a[0..n) by score[a[i]] descending, then a[i] ascending.
void SortByScore(uint32_t* a, uint32_t n, const int64_t* score);
// LEB128 varint at p[pos], pos behind it.
uint64_t ReadVarint(const uint8_t* p, size_t& pos);
}  // namespace dict_derive

#endif  // DICT_DERIVE_H
