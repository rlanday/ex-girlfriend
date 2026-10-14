#ifndef HELPER_PROCESS_H_
#define HELPER_PROCESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <unistd.h>
#include <fcntl.h>

// Release steps around the helper processes of the self-extraction: the
// compressor frees its image buffer before the first helper; every helper
// start is preceded by malloc_trim(0); the decompressor copies the blocks
// out of its own image piece by piece and drops the page cache of the image
// once they are out.

namespace helper_process {

// Immediately before every system() call.  malloc_trim hands the free
// tops of the arenas back to the kernel; what is still in use stays.  The
// call sits in fx_self_d/fx_self_c themselves so that NO call site can miss
// it -- not even one added later.
inline void VorStart(const char* wo) {
  malloc_trim(0);
  (void)wo;
}

// Copy one block from the own, still open image into a file without
// holding the image in memory.  The 1 MiB buffer is deliberately on the HEAP
// (not in the BSS: a BSS buffer would be 1 MiB in EVERY build, also at
// default 0 -- and it would stay resident after the first touch) and is
// freed right afterwards.
// Returns 0 = fine, otherwise an error number (the caller reports and stops).
inline int BlockAusAbbild(FILE* f, long off, size_t len, const char* ziel) {
  const size_t kPuf = 1u << 20;
  unsigned char* buf = (unsigned char*)malloc(kPuf);
  FILE* o = fopen(ziel, "wb");
  int rc = 0;
  if (!buf || !o || fseek(f, off, SEEK_SET) != 0) rc = 1;
  while (rc == 0 && len) {
    size_t n = len < kPuf ? len : kPuf;
    if (fread(buf, 1, n, f) != n || fwrite(buf, 1, n, o) != n) { rc = 2; break; }
    len -= n;
  }
  if (o && fclose(o) != 0) rc = 3;
  free(buf);
  malloc_trim(0);
  return rc;
}

// The page cache of the own archive (~109 MB) is not needed once the
// blocks are cut out.  It does not count in the jury metric, but fully in
// the cgroup's memory.max -- and the jury runs a 10 GiB limit.  DONTNEED
// drops only CLEAN pages not used elsewhere; a failure is harmless and is
// therefore not reported.
inline void CacheZurueck(FILE* f) {
  if (f) {
    fflush(f);
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
  }
}

}

#endif
