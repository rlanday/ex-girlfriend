#include <stdint.h>
#include <stddef.h>
#include "boot_cm.h"

__attribute__((used, aligned(16)))
static volatile struct {
  char     magic[16];
  uint64_t payload_off, comp_len, decode_len, image_len, bcj_split, dec_clen;   /* bcj_split: historical name of the cut point Nd */
} cut_hdr = { {'B','5','J','P','A','Y','L','O','A','D','_','V','1','_','_'},
              0, 0, 0, 0, 0, 0 };

extern const unsigned char s2blob[];
__asm__(".section .rodata.s2blob,\"a\"\n.globl s2blob\ns2blob:\n.incbin \"" S2CM "\"\n.previous\n");

void s1_main(long *sp) {
  register long r10 __asm__("r10") = 0x32;           /* MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED */
  register long r8 __asm__("r8") = -1;
  register long r9 __asm__("r9") = 0;
  long ret;
  __asm__ volatile("syscall" : "=a"(ret)
    : "a"(9L), "D"((long)S2BASE), "S"((long)S2MEM), "d"(7L), "r"(r10), "r"(r8), "r"(r9)
    : "rcx", "r11", "memory");
  if (ret != (long)S2BASE) {
    __asm__ volatile("syscall" :: "a"(231L), "D"(110L));
    for (;;) {}
  }
  boot_decompress(s2blob, (unsigned char *)S2BASE, S2LEN);
  ((void (*)(long *, volatile void *))S2BASE)(sp, &cut_hdr);
}

__asm__(
  ".globl _start\n"
  "_start:\n"
  "  xor %ebp,%ebp\n"
  "  mov %rsp,%rdi\n"
  "  and $-16,%rsp\n"
  "  call s1_main\n"
  "  mov $231,%eax\n"
  "  xor %edi,%edi\n"
  "  syscall\n"
);
