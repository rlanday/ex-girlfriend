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
#define SYS_execve 59
#define SYS_execveat 322
#define SYS_unlinkat 263
#define SYS_readlinkat 267
#define AT_FDCWD (-100)
#define AT_EMPTY_PATH 0x1000
#define O_RDONLY 0
#define O_RDWR 2
#define O_CREAT 0100
#define O_TRUNC 01000

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
#include "cm_core.h"
#include "x86t.h"

/* Header: 16-B mark + six little-endian u64, lives in stage 1.  Written by
 * the packer (pack_tool.c, "pack") and rewritten by the -e path
 * (selfextract_comp) for .decomp_bin: comp_len <- dec_clen,
 * decode_len <- bcj_split (a historical field name: the cut point Nd). */
typedef struct {
  char     magic[16];
  uint64_t payload_off;   /* file offset of the stream                    */
  uint64_t comp_len;      /* available compressed bytes                   */
  uint64_t decode_len;    /* output bytes to decode (Nd or Nf)            */
  uint64_t image_len;     /* final image size (Nf), rest = zeros          */
  uint64_t bcj_split;     /* Nd: boundary decomp part / encoder part      */
  uint64_t dec_clen;      /* K: stream length for the first Nd bytes      */
} cut_hdr_t;

static unsigned char in_buf[1u << 21];
static unsigned char out_buf[1u << 21];
static char selfpath[] = "/proc/self/exe";
static char memname[] = "cmix";
static char empty[] = "";
static char cwdtmpl[] = "./.b5c_run";
static char tmpl[] = "/tmp/.b5cstub";
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

/* Fallback: write the original ELF to <path>, CLOSE the fd (no ETXTBSY),
 * then start it via execve. Return = failure -> next path. */
static void try_path_exec(const char *path, uint64_t olen, char **argv, char **envp) {
  long fd = s_open(path, O_RDWR | O_CREAT | O_TRUNC, 0700);
  if (fd < 0) return;
  if (s_writeall(fd, out_buf, olen)) { sys6(SYS_close, fd, 0, 0, 0, 0, 0); return; }
  s_writeall(fd, tailbuf, taillen);
  sys6(SYS_close, fd, 0, 0, 0, 0, 0);
  sys6(SYS_execve, (long)path, (long)argv, (long)envp, 0, 0, 0);
  sys6(SYS_unlinkat, AT_FDCWD, (long)path, 0, 0, 0, 0);
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
#ifdef PACK_NO_MEMFD
  mfd = -1;
#else
  mfd = sys6(SYS_memfd_create, (long)memname, 0, 0, 0, 0, 0);
#endif
  if (mfd >= 0) {
    if (!s_writeall(mfd, out_buf, ilen) && !s_writeall(mfd, tailbuf, taillen))
      sys6(SYS_execveat, mfd, (long)empty, (long)argv, (long)envp, AT_EMPTY_PATH, 0);
    sys6(SYS_close, mfd, 0, 0, 0, 0, 0);
  }
  try_path_exec(cwdtmpl, ilen, argv, envp);
  try_path_exec(tmpl, ilen, argv, envp);
  s_exit(107);
#endif
}
