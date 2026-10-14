
#ifndef PAGE_REGISTRY_H_
#define PAGE_REGISTRY_H_


#include <stddef.h>
#include <elf.h>
#include <link.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

namespace pageregistry {

struct Table {
  unsigned long a[256];
  unsigned long b[256];
  unsigned char h[256];
  unsigned int n = 0;
  unsigned int lost = 0;
};
inline Table& T() { static Table t; return t; }

// Start of the writable PT_LOAD of the executable (.data/.bss), from the
// auxiliary vector - no /proc needed.  0 if not found.
inline unsigned long DataStart() {
  const ElfW(Phdr)* ph = (const ElfW(Phdr)*)getauxval(AT_PHDR);
  unsigned long num = getauxval(AT_PHNUM);
  if (!ph) return 0;
  unsigned long bias = 0, lo = 0;
  for (unsigned long i = 0; i < num; ++i)       // PIE: bias from PT_PHDR
    if (ph[i].p_type == PT_PHDR) bias = (unsigned long)ph - ph[i].p_vaddr;
  for (unsigned long i = 0; i < num; ++i)
    if (ph[i].p_type == PT_LOAD && (ph[i].p_flags & PF_W))
      lo = bias + ph[i].p_vaddr;
  return lo;
}

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

inline void Add(const void* p, size_t n) {
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

}

#define PAGE_REGISTRY_ADD(p, n) ::pageregistry::Add((const void*)(p), (size_t)(n))

#define HUGEPAGE_HOT_NEXT() (::pageregistry::NaechsteHeiss() = 1)

#endif
