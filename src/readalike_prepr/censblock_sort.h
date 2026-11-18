// censblock_sort.h: the sort sites of the census form.
//
// std::sort / std::stable_sort / std::partial_sort instantiate a family of
// functions per element type and comparator; here every site calls qsort of
// the C library with a three-way comparison.  The result does not depend on
// the algorithm of the library: every comparison is a strict total order on
// distinguishable elements (0 only for equal elements, checked per site).
// Sites with elements that are not trivially copyable sort pointers to them.
#ifndef CENSBLOCK_SORT_H_
#define CENSBLOCK_SORT_H_
#include <stdlib.h>

namespace census_block {

typedef int (*SortVgl)(const void* x, const void* y);   // <0, 0, >0 as for qsort

// pointers to the n elements of sz bytes at a, sorted (vgl gets pointers to the
// pointers); the caller frees the array
inline void** ZeigerSortiert(void* a, size_t n, size_t sz, SortVgl vgl) {
  void** z = (void**)malloc((n ? n : 1) * sizeof(void*));
  if (!z) abort();
  for (size_t i = 0; i < n; ++i) z[i] = (char*)a + i * sz;
  qsort(z, n, sizeof(void*), vgl);
  return z;
}

}  // namespace census_block
#endif
