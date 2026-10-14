
#ifndef HUGE_PAGES_H_
#define HUGE_PAGES_H_

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include "page_registry.h"

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

// Asks the kernel to back densely used anonymous memory with 2 MiB pages
// (madvise MADV_COLLAPSE), once after start-up and then periodically (every
// 16 MiB of stream by default).  The targets are the table areas of the
// allocation registry plus the data segment and heap of the binary; no /proc.
// The output does not depend on it.
namespace hugepage {

const unsigned long kPmd = 2UL << 20;   // 2 MiB
const unsigned long kPage = 4096UL;

struct Merged {
  unsigned long a[256], b[256];
  unsigned char h[256];
  unsigned int n = 0, built_for = ~0u;
};
inline const Merged& M() {
  static Merged m;
  const pageregistry::Table& t = pageregistry::T();
  if (m.built_for != t.n) {           // areas are only added during init
    m.built_for = t.n; m.n = 0;
    for (unsigned int i = 0; i < t.n; ++i) {   // insertion sort by start
      unsigned int j = m.n++;
      while (j > 0 && m.a[j - 1] > t.a[i]) {
        m.a[j] = m.a[j - 1]; m.b[j] = m.b[j - 1];
        m.h[j] = m.h[j - 1];
        --j;
      }
      m.a[j] = t.a[i]; m.b[j] = t.b[i];
      m.h[j] = t.h[i];
    }
    unsigned int k = 0;
    for (unsigned int i = 0; i < m.n; ++i) {
      if (k > 0 && m.a[i] <= m.b[k - 1] + kPmd
          && m.h[i] == m.h[k - 1]
         ) {
        if (m.b[i] > m.b[k - 1]) m.b[k - 1] = m.b[i];
      } else {
        m.a[k] = m.a[i]; m.b[k] = m.b[i];
        m.h[k] = m.h[i];
        ++k;
      }
    }
    m.n = k;
  }
  return m;
}

// One pass over the registry.  tag only labels the optional log line.
inline void Collapse(const char* tag) {
  static const unsigned long data_lo = pageregistry::DataStart();
  const unsigned long heap_hi = (unsigned long)sbrk(0);
  const Merged& m = M();
  unsigned char vec[kPmd / kPage];
  for (unsigned int i = 0; i <= m.n; ++i) {
    unsigned long a, b;
    if (i < m.n) {
      a = m.a[i]; b = m.b[i];
    }
    else { if (!data_lo) break; a = data_lo; b = heap_hi; }
    unsigned long s = (a + kPmd - 1) & ~(kPmd - 1);
    unsigned long e = b & ~(kPmd - 1);
    for (unsigned long o = s; o + kPmd <= e; o += kPmd) {
      if (mincore((void*)o, kPmd, vec) == 0) {
        unsigned int present = 0;
        for (unsigned int j = 0; j < kPmd / kPage; ++j) present += (vec[j] & 1);
        if (present < (unsigned int)448) {
          continue;
        }
      }
      int rc = madvise((void*)o, kPmd, MADV_COLLAPSE);
      (void)rc;
    }
  }
  (void)tag;
  (void)heap_hi;
}

// Periodic call from the coding loop, only when the progress counter fires
// (about 10,000 times per run), so it costs nothing on the per-byte path.
// Collapses once per period (16 MiB of stream by default).
inline void Tick(unsigned long long pos) {
  {
    static unsigned long long naechste = 1ULL << 20;
    const unsigned long long grenze = (unsigned long long)8 << 20;
    if (naechste <= grenze && pos >= naechste) {
      // Move to the next whole MiB mark BEYOND pos: if several marks lie between two
      // progress steps, ONE pass is made, not several.
      naechste = ((pos >> 20) + 1) << 20;
      Collapse("anlauf");
    }
  }
  static unsigned long long next =
      (unsigned long long)16 << 20;
  if (pos >= next) {
    next = pos + ((unsigned long long)16 << 20);
    Collapse("tick");
  }
}

}

#define HUGEPAGE_COLLAPSE(tag) ::hugepage::Collapse(tag)
#define HUGEPAGE_TICK(pos) ::hugepage::Tick(pos)

#endif
