/* stub.c - stage 2 of the self-extraction stub: x86-aware
 * context-mixing unpacker (cm_core.h, with the exact address transform of
 * x86t.h in place of a BCJ filter) with a TRUNCATABLE encoder block.
 *
 * Stage 1 (stub_boot.c) carries the 6-field header (u32, file offset 120) in plain
 * form, decodes THIS code to its link address (s2.lds) and calls
 * s2_main(sp, &header).  The cut of the self-extraction (self_extract.h)
 * takes the .decomp_bin prefix and rewrites two fields of that header.
 *
 * The inner ELF is linked so that all encoder-only code sits in a section
 * of its own at the END OF THE FILE (linker script).  Up to offset bcj_split
 * (= Nd) the image is a complete DECOMPRESSOR.  The coder is a sequential
 * arithmetic coder, so a PREFIX of length K is enough for the first Nd
 * output bytes.
 *
 *   Compressor package: [stub][full stream]      decode_len = image_len = Nf
 *   .decomp_bin:        [stub][stream prefix K]  decode_len = bcj_split = Nd
 *
 * The rest up to image_len is padded with ZEROS, so the ELF program header
 * of the inner binary stays untouched; the encoder area is never executed on
 * the -d path.  The address transform works per executable PT_LOAD and
 * leaves a range that starts at or behind decode_len alone, so the prefix
 * does not depend on the truncated part.
 *
 * The real path of the package travels along in the memfd image
 * ([path][u32 length][16-B mark]), otherwise the self-extraction does not
 * find the appended blocks.
 *
 * The image is started from an anonymous memory file only (memfd_create +
 * execveat, Linux 3.19 or later); if that fails, the stub exits with 107.
 *
 * The probe macro (test build only, never shipped): instead of executing the
 * image, write it to stdout and exit; the build script of the packer uses
 * this to check the real stub code path against the input image.
 *
 * No libc: direct syscalls.
 */
#include <stdint.h>
#include <stddef.h>

static long sys6(long n, long a, long b, long c, long d, long e, long f) {
  long ret;
  register long r10 __asm__("r10") = d;
  register long r8  __asm__("r8")  = e;
  register long r9  __asm__("r9")  = f;
  __asm__ volatile("syscall" : "=a"(ret)
    : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
    : "rcx", "r11", "memory");
  return ret;
}
#define SYS_read 0
#define SYS_write 1
#define SYS_close 3
#define SYS_lseek 8
#define SYS_mmap 9
#define SYS_exit_group 231
#define SYS_openat 257
#define SYS_memfd_create 319
#define SYS_execveat 322
#define SYS_readlinkat 267
#define AT_FDCWD (-100)
#define AT_EMPTY_PATH 0x1000
#define O_RDONLY 0

static long s_open(const char *p, long fl, long mo) { return sys6(SYS_openat, AT_FDCWD, (long)p, fl, mo, 0, 0); }
static long s_read(long fd, void *b, long n) { return sys6(SYS_read, fd, (long)b, n, 0, 0, 0); }
static long s_write(long fd, const void *b, long n) { return sys6(SYS_write, fd, (long)b, n, 0, 0, 0); }
static long s_lseek(long fd, long o, long w) { return sys6(SYS_lseek, fd, o, w, 0, 0, 0); }
static void s_exit(long c) { sys6(SYS_exit_group, c, 0, 0, 0, 0, 0); for (;;) {} }

void *memset(void *d, int c, size_t n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)c; return d; }
void *memcpy(void *d, const void *s, size_t n) { unsigned char *a = d; const unsigned char *b = s; while (n--) *a++ = *b++; return d; }

/* The big model tables come from anonymous mmap: a failure is visible
 * (clean exit 111) instead of a fault on first touch. */
static void *cm_mmap(size_t n) {
  long r = sys6(SYS_mmap, 0, (long)n, 3 /* RW */, 0x22 /* PRIVATE|ANON */, -1, 0);
  if (r < 0 && r > -4096) return 0;
  return (void *)r;
}
#define CM_ALLOC(n) cm_mmap(n)
static unsigned char in_buf[1u << 21];
static unsigned char out_buf[1u << 21];
/* flat stage-2 image (s2.lds): the warm-up data of the x86-CM (cm_core.h) */
extern const unsigned char s2_base[], s2_textend[], s2_filend[];
#define CM_WARM_P out_buf
#define CM_WARM_N ((size_t)(s2_filend - s2_base))
#define CM_WARM_T ((long)(s2_textend - s2_base))
#include "cm_core.h"
#include "x86t.h"

/* Header: six little-endian u32 at file offset 120, lives in stage 1.
 * Written by the packer
 * (pack_tool.c, "pack") and rewritten by the -e path (selfextract_comp) for
 * .decomp_bin: comp_len <- dec_clen, decode_len <- bcj_split (a historical
 * field name: the cut point Nd). */
typedef struct {
  uint32_t payload_off;   /* file offset of the stream                    */
  uint32_t comp_len;      /* available compressed bytes                   */
  uint32_t decode_len;    /* output bytes to decode (Nd or Nf)            */
  uint32_t image_len;     /* final image size (Nf), rest = zeros          */
  uint32_t bcj_split;     /* Nd: boundary decomp part / encoder part      */
  uint32_t dec_clen;      /* K: stream length for the first Nd bytes      */
} cut_hdr_t;

static char selfpath[] = "/proc/self/exe";
static char memname[] = "cmix";
static char empty[] = "";
static const char fx_mark[16] = {'B','5','C','2','S','E','L','F',
                                 'P','A','T','H','_','V','1','\0'};
static char pathbuf[4096];
static unsigned char tailbuf[4096 + 20];
static uint64_t taillen;

/* write all of buf, 0 = ok */
static int s_writeall(long fd, const unsigned char *buf, uint64_t n) {
  while (n) {
    long w = s_write(fd, buf, (long)n);
    if (w <= 0) return 1;
    buf += w; n -= (uint64_t)w;
  }
  return 0;
}

__attribute__((section(".text.s2entry")))
void s2_main(long *sp, volatile cut_hdr_t *hp) {
  long argc = sp[0];
  char **argv = (char **)(sp + 1);
  char **envp = (char **)(sp + 1 + argc + 1);
  uint64_t off = hp->payload_off, clen = hp->comp_len;
  uint64_t dlen = hp->decode_len, ilen = hp->image_len;
  uint64_t got = 0;
  long fd, mfd, n;
  int j;

  if (clen > sizeof(in_buf) || ilen > sizeof(out_buf) || dlen > ilen) s_exit(101);
  fd = s_open(selfpath, O_RDONLY, 0);
  if (fd < 0) s_exit(102);
  if (s_lseek(fd, (long)off, 0) < 0) s_exit(103);
  while (got < clen) {
    long r = s_read(fd, in_buf + got, (long)(clen - got));
    if (r <= 0) s_exit(104);
    got += (uint64_t)r;
  }
  sys6(SYS_close, fd, 0, 0, 0, 0, 0);

  /* Warm-up data: a copy of this unpacker's flat image, with the
   * address transform of the image; cm_decompress learns from it first and
   * then overwrites out_buf. */
  memcpy(out_buf, s2_base, CM_WARM_N);
  x86t_range(out_buf, 0, CM_WARM_T, CM_WARM_VA, (int)CM2_xt >> 1, +1);
  if (!cm_decompress(in_buf, (size_t)clen, out_buf, (size_t)dlen)) s_exit(111);
  /* Undo the exact x86 address transform over the executable PT_LOADs of
   * the decoded image (ranges read from its own ELF header).  A range that
   * starts at or behind dlen (the cut-off encoder block) stays untouched. */
  cm_elf_ranges(out_buf, (size_t)dlen);
  for (j = 0; j < cm_ncr; ++j)
    if (cm_cr_lo[j] < (long)dlen)
      x86t_range(out_buf, cm_cr_lo[j], cm_cr_hi[j] < (long)dlen ? cm_cr_hi[j] : (long)dlen,
                 cm_cr_va[j], (int)CM2_xt >> 1, -1);
#ifndef CUT_FILL
#define CUT_FILL 0x00
#endif
  memset(out_buf + dlen, CUT_FILL, (size_t)(ilen - dlen));

#ifdef CUT_PROBE
  (void)argv; (void)envp; (void)mfd; (void)n;
  s_exit(s_writeall(1, out_buf, ilen) ? 120 : 0);
#else
  n = sys6(SYS_readlinkat, AT_FDCWD, (long)selfpath, (long)pathbuf,
           (long)sizeof(pathbuf), 0, 0);
  if (n > 0 && (uint64_t)n < sizeof(pathbuf)) {
    uint32_t pl = (uint32_t)n;
    memcpy(tailbuf, pathbuf, (size_t)n);
    memcpy(tailbuf + n, &pl, 4);
    memcpy(tailbuf + n + 4, fx_mark, 16);
    taillen = (uint64_t)n + 20;
  }
  mfd = sys6(SYS_memfd_create, (long)memname, 0, 0, 0, 0, 0);
  if (mfd >= 0) {
    if (!s_writeall(mfd, out_buf, ilen) && !s_writeall(mfd, tailbuf, taillen))
      sys6(SYS_execveat, mfd, (long)empty, (long)argv, (long)envp, AT_EMPTY_PATH, 0);
    sys6(SYS_close, mfd, 0, 0, 0, 0, 0);
  }
  s_exit(107);
#endif
}
