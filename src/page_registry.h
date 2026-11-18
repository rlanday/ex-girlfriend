
#ifndef PAGE_REGISTRY_H_
#define PAGE_REGISTRY_H_


#include <stddef.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

namespace pageregistry {

struct Table {
  unsigned long a[256];
  unsigned long b[256];
  unsigned char h[256];
  // No member initialisers: the table is trivial, lies zeroed in .bss and
  // needs no guarded initialisation at its first use.
  unsigned int n;
  unsigned int lost;
};
inline Table& T() { static Table t; return t; }

// Start of the last writable PT_LOAD of the executable: .bss (the linker
// script places .encoder_only between .got.plt and .bss, so .bss starts its own
// segment).  The linker defines __bss_start; no /proc, no auxiliary vector.
extern "C" char __bss_start[];
inline unsigned long DataStart() { return (unsigned long)__bss_start; }

inline unsigned char& NaechsteHeiss() { static unsigned char f = 0; return f; }
// Counter of refused 2-MiB blocks; the message comes once, at the first failure.
inline unsigned long& WunschFehler() { static unsigned long n = 0; return n; }

inline void Wunsch(unsigned long a, unsigned long b) {
  const unsigned long kPmd = 2UL << 20;
  unsigned long s = (a + kPmd - 1) & ~(kPmd - 1), e = b & ~(kPmd - 1);
  if (e <= s) return;
  if (madvise((void*)s, e - s, MADV_HUGEPAGE) == 0) return;
  WunschFehler() += (e - s) >> 21;
  static bool gemeldet = false;
  if (!gemeldet) {
    gemeldet = true;
    fprintf(stderr, "\nmadvise(MADV_HUGEPAGE) refused for %lu 2-MiB blocks "
                    "(range %lx-%lx).  The run stays correct; the host page size applies "
                    "there, and MADV_COLLAPSE follows up.\n",
            (e - s) >> 21, s, e);
    fflush(stderr);
  }
}

// Out of line: it was inlined at about 20 allocation sites.
__attribute__((noinline)) inline void Add(const void* p, size_t n) {
  // ALWAYS take the mark first, even if the area is below MINMAP_MB and dropped right
  // away -- otherwise it would wander to the next, unrelated registration.
  const unsigned char heiss = NaechsteHeiss();
  NaechsteHeiss() = 0;
  if (!p || n < ((size_t)2 << 20)) return;
  unsigned long a = (unsigned long)p, b = a + (unsigned long)n;
  Table& t = T();
  if (t.n < 256) {
    t.a[t.n] = a; t.b[t.n] = b;
    t.h[t.n] = heiss;
    ++t.n;
  }
  else ++t.lost;
}

}  // namespace pageregistry

#define PAGE_REGISTRY_ADD(p, n) ::pageregistry::Add((const void*)(p), (size_t)(n))

#define HUGEPAGE_HOT_NEXT() (::pageregistry::NaechsteHeiss() = 1)

#endif
