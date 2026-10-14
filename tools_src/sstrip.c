/* sstrip.c - removes the section header table from an ELF64 file.
 *
 *   sstrip <in> <out>
 *
 * Written for this entry (2026), GPL-3.0-only.  The name follows Brian
 * Raiter's sstrip from the ELF kickers package; no code of that program is
 * used.
 *
 * It sets e_shoff, e_shentsize, e_shnum and e_shstrndx to 0 and cuts the file
 * behind the last byte that a program header covers.  The result is a valid,
 * executable ELF without any section view.  `llvm-objcopy --strip-sections`
 * gives the same file; the build uses it first and falls back to this
 * program only where that option is missing.
 *
 * Build:  make sstrip     (or: cc -O2 -o sstrip tools_src/sstrip.c)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rd(const unsigned char *p, int n) {
  unsigned long long v = 0;
  int i;
  for (i = n - 1; i >= 0; --i) v = (v << 8) | p[i];   /* little endian */
  return v;
}

static void wr(unsigned char *p, int n, unsigned long long v) {
  int i;
  for (i = 0; i < n; ++i) { p[i] = (unsigned char)(v & 0xff); v >>= 8; }
}

int main(int argc, char **argv) {
  FILE *f;
  unsigned char *b;
  long fsize;
  size_t got, end, i;
  unsigned long long e_phoff;
  unsigned e_phentsize, e_phnum;

  if (argc != 3) {
    fprintf(stderr, "usage: %s <in.elf> <out.elf>\n", argv[0]);
    return 2;
  }
  f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 2; }
  if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); fclose(f); return 2; }
  fsize = ftell(f);
  if (fsize < 64) { fprintf(stderr, "%s: too small for an ELF64\n", argv[1]); fclose(f); return 2; }
  rewind(f);
  b = (unsigned char *)malloc((size_t)fsize);
  if (!b) { fprintf(stderr, "out of memory for %ld bytes\n", fsize); fclose(f); return 2; }
  got = fread(b, 1, (size_t)fsize, f);
  fclose(f);
  if (got != (size_t)fsize) { fprintf(stderr, "%s: read incompletely\n", argv[1]); free(b); return 2; }

  if (memcmp(b, "\177ELF", 4) != 0 || b[4] != 2 /*ELFCLASS64*/ || b[5] != 1 /*ELFDATA2LSB*/) {
    fprintf(stderr, "%s: not a little-endian ELF64\n", argv[1]);
    free(b); return 2;
  }

  e_phoff     = rd(b + 0x20, 8);
  e_phentsize = (unsigned)rd(b + 0x36, 2);
  e_phnum     = (unsigned)rd(b + 0x38, 2);
  if (e_phnum == 0 || e_phentsize < 0x38 ||
      e_phoff + (unsigned long long)e_phnum * e_phentsize > (unsigned long long)fsize) {
    fprintf(stderr, "%s: unusable program header table\n", argv[1]);
    free(b); return 2;
  }

  /* Find the last byte covered by any program header; the table itself and the
     ELF header count too (in practice covered by PT_PHDR/PT_LOAD). */
  end = (size_t)(e_phoff + (unsigned long long)e_phnum * e_phentsize);
  if (end < 64) end = 64;
  for (i = 0; i < e_phnum; ++i) {
    const unsigned char *ph = b + e_phoff + i * (size_t)e_phentsize;
    unsigned long long off = rd(ph + 0x08, 8);
    unsigned long long fsz = rd(ph + 0x20, 8);
    unsigned long long e   = off + fsz;
    if (e > (unsigned long long)fsize) {
      fprintf(stderr, "%s: program header %u points past end of file\n", argv[1], (unsigned)i);
      free(b); return 2;
    }
    if (e > (unsigned long long)end) end = (size_t)e;
  }

  wr(b + 0x28, 8, 0);   /* e_shoff     */
  wr(b + 0x3A, 2, 0);   /* e_shentsize */
  wr(b + 0x3C, 2, 0);   /* e_shnum     */
  wr(b + 0x3E, 2, 0);   /* e_shstrndx  */

  f = fopen(argv[2], "wb");
  if (!f) { perror(argv[2]); free(b); return 2; }
  if (fwrite(b, 1, end, f) != end || fclose(f) != 0) {
    fprintf(stderr, "%s: write error\n", argv[2]);
    free(b); return 2;
  }
  free(b);
  printf("sstrip: %ld -> %lu\n", fsize, (unsigned long)end);
  return 0;
}
