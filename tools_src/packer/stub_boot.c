/* stub_boot.c - stage 1 of the two-stage packer stub.
 *
 * Carries the 24-byte header (six u32 at file offset 120, right behind the
 * ELF and program header; the packer and the cut of the self-extraction
 * read and rewrite it there), the model list (boot_models.h) and the stage-2
 * image (the x86-CM unpacker of stub.c, linked at S2BASE), compressed
 * with the count-mixing bootstrap coder of boot_cm.h.  Stage 1 maps S2MEM
 * bytes at S2BASE plus a BOOT_TSZ-byte hash table behind them, decodes stage 2
 * there and calls it with the stack pointer and the header address.  No libc;
 * direct syscalls.
 *
 * Layout (s1.lds, one PT_LOAD with ELF and program header, no PT_GNU_STACK):
 *   [ELF header][program header][header][model table][code][stage-2 stream]
 *
 * The decoder is boot_decompress() of boot_cm.h written in assembly: stage 1
 * is stored uncompressed, so every byte of it counts twice in S.  The host
 * tool encodes with boot_compress(); the probe build of the packer build script
 * runs this code on the real stream.  Model list: BOOT_NM entries, each a byte
 * mask over the last 8 bytes and a weight; one hash-table slot per context
 * (two u8 counters) at S2BASE + S2MEM.  Scalars: BOOT_TB bits, BOOT_EPS, cap.
 * S2BASE/S2MEM/S2LEN/S2CM are set by the packer build script. */
#include <stdint.h>
#include "boot_models.h"

_Static_assert(BOOT_SHIFT == 64 - BOOT_TB, "BOOT_SHIFT must be 64-BOOT_TB");
/* the slot addresses lie at -128(%rbp,%rbx,8) (8-bit displacement) below tbl at
 * -56(%rbp), so at most 9 models fit */
_Static_assert(BOOT_NM <= 9, "slots[BOOT_NM] would overwrite tbl at -56(%rbp)");

#define STR_(x) #x
#define STR(x) STR_(x)

__attribute__((used, section(".cuthdr")))
volatile struct __attribute__((packed)) {
  uint32_t payload_off, comp_len, decode_len, image_len, bcj_split, dec_clen;   /* bcj_split: historical name of the cut point Nd */
} cut_hdr = { 0, 0, 0, 0, 0, 0 };

/* model list in plain form, right behind the header (counts in S, 2*BOOT_NM B);
 * same bytes as boot_mask[]/boot_wt[] via the shared macros. */
__attribute__((used, section(".cutmodels")))
const unsigned char cut_models[2 * BOOT_NM] = { BOOT_MASK_LIST, BOOT_WT_LIST };

__asm__(".section .rodata.s2blob,\"a\"\n.globl s2blob\ns2blob:\n.incbin \"" S2CM "\"\n.previous\n");

/* frame (rbp = original sp): -8 x1 -16 x2 -24 x -32 hist -40 outp -48 nleft
 * -56 tbl ; slots[BOOT_NM] at -128(%rbp,%rbx,8).  Constants C1/C2/CS via movabs. */
__asm__(
  ".text\n"
  ".globl _start\n"
  "_start:\n"
  "  mov %rsp,%rbp\n"                   /* original sp (passed to stage 2) */
  "  sub $256,%rsp\n"
  /* mmap(S2BASE, S2MEM + table, RWX, PRIVATE|ANON|FIXED, -1, 0) */
  "  push $9\n  pop %rax\n"
  "  mov $" STR(S2BASE) ",%edi\n"
  "  mov $(" STR(S2MEM) "+" STR(BOOT_TSZ) "),%esi\n"
  "  push $7\n  pop %rdx\n"
  "  push $0x32\n  pop %r10\n"
  "  or $-1,%r8\n"
  "  xor %r9d,%r9d\n"
  "  syscall\n"
  "  cmp %rdi,%rax\n"
  "  je 1f\n"
  "  push $110\n  pop %rdi\n  mov $231,%eax\n  syscall\n"
  "1:\n"
  /* state */
  "  movq $0,-32(%rbp)\n"               /* hist = 0 */
  "  movl $0,-8(%rbp)\n"                /* x1 */
  "  movl $0xffffffff,-12(%rbp)\n"      /* x2 */
  "  mov %rdi,-40(%rbp)\n"              /* outp = S2BASE */
  "  movq $" STR(S2LEN) ",-48(%rbp)\n"  /* nleft */
  "  mov $(" STR(S2BASE) "+" STR(S2MEM) "),%eax\n  mov %rax,-56(%rbp)\n"  /* tbl */
  /* x = first 4 stream bytes big-endian; rsi = input pointer */
  "  mov $s2blob,%esi\n"
  "  xor %eax,%eax\n  mov $4,%ecx\n"
  "2: shl $8,%eax\n  movzbl (%rsi),%r10d\n  inc %rsi\n  or %r10d,%eax\n  dec %ecx\n  jnz 2b\n"
  "  mov %eax,-24(%rbp)\n"
  ".Lbyte:\n"
  "  mov $1,%r12d\n"                    /* c0 */
  "  mov $8,%r13d\n"                    /* bit counter */
  ".Lbit:\n"
  "  mov $" STR(BOOT_EPS) ",%r14\n"     /* s0 */
  "  mov $" STR(BOOT_EPS) ",%r15\n"     /* s1 */
  "  xor %ebx,%ebx\n"                   /* m */
  "  xor %r11,%r11\n"                   /* salt */
  ".Lmodel:\n"
  "  mov %r12,%rax\n"                   /* h = c0 */
  "  mov -32(%rbp),%r10\n"              /* hv = hist */
  "  movzbl cut_models(%rbx),%ecx\n"    /* mask[m] */
  "  test %cl,%cl\n  jz .Lhdone\n"
  ".Lhl:\n"
  "  shr $1,%cl\n  jnc .Lhs\n"
  "  movabs $0x100000001B3,%rdi\n  imul %rdi,%rax\n"
  "  movzbl %r10b,%edi\n  add %rdi,%rax\n"
  ".Lhs:\n"
  "  shr $8,%r10\n"
  "  test %cl,%cl\n  jnz .Lhl\n"
  ".Lhdone:\n"
  "  add %r11,%rax\n"                   /* + salt */
  "  movabs $0x2545F4914F6CDD1D,%rdi\n  imul %rdi,%rax\n"
  "  movabs $0x9E3779B97F4A7C15,%rdi\n  add %rdi,%r11\n"  /* salt += CS */
  "  shr $" STR(BOOT_SHIFT) ",%rax\n"
  "  mov -56(%rbp),%rcx\n  lea (%rcx,%rax,2),%rcx\n"      /* slot addr */
  "  mov %rcx,-128(%rbp,%rbx,8)\n"
  "  movzbl (%rcx),%r8d\n  movzbl 1(%rcx),%r9d\n"         /* n0,n1 */
  "  test %r8d,%r8d\n  jnz 1f\n  shl $1,%r9d\n  jmp 2f\n"
  "1: test %r9d,%r9d\n  jnz 2f\n  shl $1,%r8d\n"
  "2:\n"
  "  movzbl cut_models+" STR(BOOT_NM) "(%rbx),%ecx\n"     /* wt[m] */
  "  shl %cl,%r8\n  shl %cl,%r9\n"
  "  add %r8,%r14\n  add %r9,%r15\n"
  "  inc %rbx\n  cmp $" STR(BOOT_NM) ",%rbx\n  jl .Lmodel\n"
  /* p = (s1<<12)/(s0+s1), clamp 1..4095 */
  "  mov %r15,%rax\n  shl $12,%rax\n  xor %edx,%edx\n"
  "  mov %r14,%rcx\n  add %r15,%rcx\n  div %rcx\n"
  "  mov $1,%ecx\n  cmp %rcx,%rax\n  cmovb %rcx,%rax\n"
  "  mov $4095,%ecx\n  cmp %rcx,%rax\n  cmova %rcx,%rax\n"
  /* xm = x1 + ((x2-x1)*p >> 12) */
  "  mov -12(%rbp),%r8d\n  sub -8(%rbp),%r8d\n"
  "  imul %rax,%r8\n  shr $12,%r8\n  add -8(%rbp),%r8d\n"
  "  lea 1(%r8),%ecx\n"
  "  mov -24(%rbp),%edx\n"
  "  cmp %ecx,%edx\n"                   /* CF = x < xm+1 = bit */
  "  setb %r9b\n  movzbl %r9b,%r9d\n"
  "  mov -8(%rbp),%eax\n  mov -12(%rbp),%edi\n"
  "  test %r9d,%r9d\n  jz 3f\n  mov %r8d,%edi\n  jmp 4f\n"
  "3: mov %ecx,%eax\n"
  "4: mov %eax,-8(%rbp)\n  mov %edi,-12(%rbp)\n"
  ".Lnorm:\n"
  "  mov -8(%rbp),%eax\n  xor -12(%rbp),%eax\n  shr $24,%eax\n  jnz .Lnd\n"
  "  shll $8,-8(%rbp)\n"
  "  mov -12(%rbp),%eax\n  shl $8,%eax\n  or $255,%eax\n  mov %eax,-12(%rbp)\n"
  "  mov -24(%rbp),%eax\n  shl $8,%eax\n  movzbl (%rsi),%r10d\n  inc %rsi\n  or %r10d,%eax\n  mov %eax,-24(%rbp)\n"
  "  jmp .Lnorm\n"
  ".Lnd:\n"
  "  xor %ebx,%ebx\n"
  ".Lupd:\n"
  "  mov -128(%rbp,%rbx,8),%rcx\n"
  "  movzbl (%rcx,%r9),%eax\n  cmp $" STR(BOOT_CAP) ",%al\n  jae 5f\n  inc %al\n  mov %al,(%rcx,%r9)\n"
  "5:\n"
  "  mov %r9d,%edx\n  xor $1,%edx\n"
  "  movzbl (%rcx,%rdx),%eax\n  cmp $1,%al\n  jbe 6f\n  shr $1,%al\n  inc %al\n  mov %al,(%rcx,%rdx)\n"
  "6:\n"
  "  inc %rbx\n  cmp $" STR(BOOT_NM) ",%rbx\n  jl .Lupd\n"
  "  add %r12d,%r12d\n  or %r9d,%r12d\n"     /* c0 = c0<<1 | bit */
  "  dec %r13d\n  jnz .Lbit\n"
  "  mov -40(%rbp),%rcx\n  mov %r12b,(%rcx)\n  inc %rcx\n  mov %rcx,-40(%rbp)\n"
  "  mov -32(%rbp),%rax\n  shl $8,%rax\n  movzbl %r12b,%edi\n  or %rdi,%rax\n  mov %rax,-32(%rbp)\n"
  "  decq -48(%rbp)\n  jnz .Lbyte\n"
  /* stage 2: s2_main(original_sp, header); does not return */
  "  mov %rbp,%rdi\n"
  "  mov $cut_hdr,%esi\n"
  "  call " STR(S2BASE) "\n"
);
