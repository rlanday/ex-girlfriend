
#ifndef ARENA_ALLOC_H_
#define ARENA_ALLOC_H_

#define ARENA_MALLOC(n) malloc(n)
#define ARENA_FREE(p)   free(p)

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <sys/mman.h>
#include <unistd.h>
#include "page_registry.h"

namespace arena {

// Bump allocator for model tables: 64 MiB chunks, 2 MiB aligned and marked
// for transparent huge pages.  Nothing is freed individually.
struct Bump {
  char* cur = nullptr;
  char* end = nullptr;
  unsigned long long vergeben = 0;    // bytes handed out, after rounding up
  unsigned long long angefordert = 0; // bytes requested, before rounding up
  unsigned long long bloecke = 0;     // chunks mapped
  unsigned long long block_bytes = 0;

  static const size_t kAlign = (size_t)64;
  static const size_t kChunk = ((size_t)64) << 20;

  // Maps a new chunk of at least `mindestens` (minimum) bytes; the unaligned
  // head and tail (`hinten`) of the mapping are unmapped again.
  void NeuerBlock(size_t mindestens) {
    size_t n = kChunk;
    if (n < mindestens) n = (mindestens + kChunk - 1) / kChunk * kChunk;
    // Over-allocate by 2 MiB and keep the aligned part, so that
    // MADV_HUGEPAGE can take effect at all.
    const size_t kPmd = 2UL << 20;
    n = (n + kPmd - 1) & ~(kPmd - 1);
    char* raw = (char*)mmap(nullptr, n + kPmd, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) { perror("arena mmap"); abort(); }
    char* base = (char*)(((size_t)raw + kPmd - 1) & ~(kPmd - 1));
    if (base > raw) munmap(raw, (size_t)(base - raw));
    size_t hinten = (size_t)(raw + n + kPmd - (base + n));
    if (hinten) munmap(base + n, hinten);
    madvise(base, n, MADV_HUGEPAGE);
    PAGE_REGISTRY_ADD(base, n);
    cur = base;
    end = base + n;
    ++bloecke;
    block_bytes += n;
  }

  // Hands out n bytes: 16-byte aligned below 256 bytes, else 64-byte aligned.
  // The length grows to the 16-byte grid only (as in glibc); the alignment is
  // applied to the pointer.
  void* Hole(size_t n) {
    angefordert += n;
    size_t a = kAlign;
    if (n < (size_t)256 && a > 16) a = 16;
    size_t m = (n + 15) & ~(size_t)15;
    if (m == 0) m = 16;
    char* p = (char*)(((size_t)cur + (a - 1)) & ~(a - 1));
    if (p + m > end) {
      NeuerBlock(m + a);
      p = (char*)(((size_t)cur + (a - 1)) & ~(a - 1));
    }
    vergeben += (size_t)(p + m - cur);
    cur = p + m;
    return p;
  }
};

inline Bump& G() { static Bump b; return b; }
inline void* Hole(size_t n) { return G().Hole(n); }

// Row: pointer + length, exactly the size of a std::valarray (16 B on x86-64).
// Trivially copyable on purpose: rows are never freed, so a shallow copy is
// correct and free.
template <class T>
struct Row {
  T* p_;
  size_t n_;

  Row() : p_(nullptr), n_(0) {}
  explicit Row(size_t n) : n_(n) {
    p_ = (T*)Hole(n * sizeof(T));
    if (n) memset((void*)p_, 0, n * sizeof(T));   // zero-filled like std::valarray
  }
  // shallow on purpose (these rows are never freed)
  Row(const Row&) = default;
  Row& operator=(const Row&) = default;
  Row(Row&&) = default;
  Row& operator=(Row&&) = default;

  // adopt existing memory (one block for two rows)
  Row(T* p, size_t n) : p_(p), n_(n) {}

  T& operator[](size_t i) { return p_[i]; }
  const T& operator[](size_t i) const { return p_[i]; }
  size_t size() const { return n_; }
  Row& operator=(const T& v) { for (size_t i = 0; i < n_; ++i) p_[i] = v; return *this; }
};
static_assert(sizeof(Row<float>) == 16, "row must be 16 B");

}

#endif
