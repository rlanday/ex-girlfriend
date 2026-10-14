// ALU peak throughput microbenchmarks (L1-resident, unrolled, independent).
// Every kernel is a hand-written asm loop; reported unit = instructions of
// interest per CORE cycle.
#include "bench_common.h"

// ---- pure ALU kernels: N independent instances of one instruction per iter ----
#define K_HEAD                                            \
  uint64_t n = iters;                                     \
  uint64_t t0 = rdtsc_ser();
#define K_TAIL(clob...)                                   \
  : "+r"(n)::clob, "cc", "memory");                       \
  return rdtsc_ser() - t0;

#define INT_INIT                                          \
  "vpcmpeqd %%ymm12,%%ymm12,%%ymm12\n\t"                  \
  "vpcmpeqd %%ymm13,%%ymm13,%%ymm13\n\t"                  \
  "vpsrlw $14,%%ymm13,%%ymm13\n\t"  /* ymm13 = 0x0003 words */

static uint64_t k_maddubsw(uint64_t iters) {  // 10 independent vpmaddubsw
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm9\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm12","xmm13")
}
static uint64_t k_maddwd(uint64_t iters) {
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpmaddwd %%ymm13,%%ymm12,%%ymm9\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm12","xmm13")
}
static uint64_t k_paddw(uint64_t iters) {  // 12 independent vpaddw
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm9\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_pand(uint64_t iters) {
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm9\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpand %%ymm13,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_psrlw(uint64_t iters) {  // 10 independent vpsrlw $4
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpsrlw $4,%%ymm12,%%ymm0\n\t"
      "vpsrlw $4,%%ymm12,%%ymm1\n\t"
      "vpsrlw $4,%%ymm12,%%ymm2\n\t"
      "vpsrlw $4,%%ymm12,%%ymm3\n\t"
      "vpsrlw $4,%%ymm12,%%ymm4\n\t"
      "vpsrlw $4,%%ymm12,%%ymm5\n\t"
      "vpsrlw $4,%%ymm12,%%ymm6\n\t"
      "vpsrlw $4,%%ymm12,%%ymm7\n\t"
      "vpsrlw $4,%%ymm12,%%ymm8\n\t"
      "vpsrlw $4,%%ymm12,%%ymm9\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm12","xmm13")
}
// mixes: madd:add ratios. 12 instr/iter each.
static uint64_t k_mix2M2A(uint64_t iters) {
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm9\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_mix1M1A(uint64_t iters) {
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm9\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_mix2M1A(uint64_t iters) {
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm9\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpaddw %%ymm13,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_mix1M1S(uint64_t iters) {  // vpmaddubsw : vpsrlw 1:1 (pipe sharing probe)
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpsrlw $4,%%ymm12,%%ymm1\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpsrlw $4,%%ymm12,%%ymm3\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpsrlw $4,%%ymm12,%%ymm5\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpsrlw $4,%%ymm12,%%ymm7\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpsrlw $4,%%ymm12,%%ymm9\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpsrlw $4,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_mix1M1SH(uint64_t iters) {  // vpmaddubsw : vpshufb 1:1 (alt unpack probe)
  K_HEAD
  asm volatile(
      INT_INIT
      "1:\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm0\n\t"
      "vpshufb %%ymm13,%%ymm12,%%ymm1\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm2\n\t"
      "vpshufb %%ymm13,%%ymm12,%%ymm3\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm4\n\t"
      "vpshufb %%ymm13,%%ymm12,%%ymm5\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm6\n\t"
      "vpshufb %%ymm13,%%ymm12,%%ymm7\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm8\n\t"
      "vpshufb %%ymm13,%%ymm12,%%ymm9\n\t"
      "vpmaddubsw %%ymm13,%%ymm12,%%ymm10\n\t"
      "vpshufb %%ymm13,%%ymm12,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
  K_TAIL("xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13")
}
static uint64_t k_fma(uint64_t iters) {  // 12 independent FMA accumulators
  static const float tiny = 1e-30f, onef = 1.0f;
  K_HEAD
  asm volatile(
      "vbroadcastss %1,%%ymm14\n\t"
      "vbroadcastss %2,%%ymm15\n\t"
      "vxorps %%xmm0,%%xmm0,%%xmm0\n\t" "vxorps %%xmm1,%%xmm1,%%xmm1\n\t"
      "vxorps %%xmm2,%%xmm2,%%xmm2\n\t" "vxorps %%xmm3,%%xmm3,%%xmm3\n\t"
      "vxorps %%xmm4,%%xmm4,%%xmm4\n\t" "vxorps %%xmm5,%%xmm5,%%xmm5\n\t"
      "vxorps %%xmm6,%%xmm6,%%xmm6\n\t" "vxorps %%xmm7,%%xmm7,%%xmm7\n\t"
      "vxorps %%xmm8,%%xmm8,%%xmm8\n\t" "vxorps %%xmm9,%%xmm9,%%xmm9\n\t"
      "vxorps %%xmm10,%%xmm10,%%xmm10\n\t" "vxorps %%xmm11,%%xmm11,%%xmm11\n\t"
      "1:\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm0\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm1\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm2\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm3\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm4\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm5\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm6\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm7\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm8\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm9\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm10\n\t"
      "vfmadd231ps %%ymm14,%%ymm15,%%ymm11\n\t"
      "dec %0\n\tjnz 1b\n\t"
      : "+r"(n)
      : "m"(tiny), "m"(onef)
      : "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9",
        "xmm10","xmm11","xmm14","xmm15","cc","memory");
  return rdtsc_ser() - t0;
}
static uint64_t k_divps(uint64_t iters) {  // 8 independent vdivps
  static const float a = 1.234567f, b = 0.987654f;
  K_HEAD
  asm volatile(
      "vbroadcastss %1,%%ymm14\n\t"
      "vbroadcastss %2,%%ymm15\n\t"
      "1:\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm0\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm1\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm2\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm3\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm4\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm5\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm6\n\t"
      "vdivps %%ymm14,%%ymm15,%%ymm7\n\t"
      "dec %0\n\tjnz 1b\n\t"
      : "+r"(n)
      : "m"(a), "m"(b)
      : "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm14","xmm15",
        "cc","memory");
  return rdtsc_ser() - t0;
}
// 16x 32B aligned loads per iter over a 16 KB buffer (wrap via AND mask)
static uint64_t k_loads(uint64_t iters, const void* buf) {
  uint64_t n = iters;
  uint64_t t0 = rdtsc_ser();
  asm volatile(
      "xor %%ecx,%%ecx\n\t"
      "1:\n\t"
      "vmovdqa 0(%1,%%rcx),%%ymm0\n\t"
      "vmovdqa 32(%1,%%rcx),%%ymm1\n\t"
      "vmovdqa 64(%1,%%rcx),%%ymm2\n\t"
      "vmovdqa 96(%1,%%rcx),%%ymm3\n\t"
      "vmovdqa 128(%1,%%rcx),%%ymm4\n\t"
      "vmovdqa 160(%1,%%rcx),%%ymm5\n\t"
      "vmovdqa 192(%1,%%rcx),%%ymm6\n\t"
      "vmovdqa 224(%1,%%rcx),%%ymm7\n\t"
      "vmovdqa 256(%1,%%rcx),%%ymm0\n\t"
      "vmovdqa 288(%1,%%rcx),%%ymm1\n\t"
      "vmovdqa 320(%1,%%rcx),%%ymm2\n\t"
      "vmovdqa 352(%1,%%rcx),%%ymm3\n\t"
      "vmovdqa 384(%1,%%rcx),%%ymm4\n\t"
      "vmovdqa 416(%1,%%rcx),%%ymm5\n\t"
      "vmovdqa 448(%1,%%rcx),%%ymm6\n\t"
      "vmovdqa 480(%1,%%rcx),%%ymm7\n\t"
      "add $512,%%rcx\n\t"
      "and $16383,%%rcx\n\t"
      "dec %0\n\tjnz 1b\n\t"
      : "+r"(n)
      : "r"(buf)
      : "rcx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","cc","memory");
  return rdtsc_ser() - t0;
}
// combined: 4 groups of (2 loads + 2 vpmaddubsw + 2 vpaddw) per iter (256 B)
static uint64_t k_combo(uint64_t iters, const void* buf) {
  uint64_t n = iters;
  uint64_t t0 = rdtsc_ser();
  asm volatile(
      "vpcmpeqd %%ymm12,%%ymm12,%%ymm12\n\t"
      "vpxor %%xmm4,%%xmm4,%%xmm4\n\t" "vpxor %%xmm5,%%xmm5,%%xmm5\n\t"
      "vpxor %%xmm6,%%xmm6,%%xmm6\n\t" "vpxor %%xmm7,%%xmm7,%%xmm7\n\t"
      "vpxor %%xmm8,%%xmm8,%%xmm8\n\t" "vpxor %%xmm9,%%xmm9,%%xmm9\n\t"
      "vpxor %%xmm10,%%xmm10,%%xmm10\n\t" "vpxor %%xmm11,%%xmm11,%%xmm11\n\t"
      "xor %%ecx,%%ecx\n\t"
      "1:\n\t"
      // g0
      "vmovdqa 0(%1,%%rcx),%%ymm0\n\t"
      "vmovdqa 32(%1,%%rcx),%%ymm1\n\t"
      "vpmaddubsw %%ymm12,%%ymm0,%%ymm2\n\t"
      "vpmaddubsw %%ymm12,%%ymm1,%%ymm3\n\t"
      "vpaddw %%ymm2,%%ymm4,%%ymm4\n\t"
      "vpaddw %%ymm3,%%ymm5,%%ymm5\n\t"
      // g1
      "vmovdqa 64(%1,%%rcx),%%ymm0\n\t"
      "vmovdqa 96(%1,%%rcx),%%ymm1\n\t"
      "vpmaddubsw %%ymm12,%%ymm0,%%ymm2\n\t"
      "vpmaddubsw %%ymm12,%%ymm1,%%ymm3\n\t"
      "vpaddw %%ymm2,%%ymm6,%%ymm6\n\t"
      "vpaddw %%ymm3,%%ymm7,%%ymm7\n\t"
      // g2
      "vmovdqa 128(%1,%%rcx),%%ymm0\n\t"
      "vmovdqa 160(%1,%%rcx),%%ymm1\n\t"
      "vpmaddubsw %%ymm12,%%ymm0,%%ymm2\n\t"
      "vpmaddubsw %%ymm12,%%ymm1,%%ymm3\n\t"
      "vpaddw %%ymm2,%%ymm8,%%ymm8\n\t"
      "vpaddw %%ymm3,%%ymm9,%%ymm9\n\t"
      // g3
      "vmovdqa 192(%1,%%rcx),%%ymm0\n\t"
      "vmovdqa 224(%1,%%rcx),%%ymm1\n\t"
      "vpmaddubsw %%ymm12,%%ymm0,%%ymm2\n\t"
      "vpmaddubsw %%ymm12,%%ymm1,%%ymm3\n\t"
      "vpaddw %%ymm2,%%ymm10,%%ymm10\n\t"
      "vpaddw %%ymm3,%%ymm11,%%ymm11\n\t"
      "add $256,%%rcx\n\t"
      "and $16383,%%rcx\n\t"
      "dec %0\n\tjnz 1b\n\t"
      : "+r"(n)
      : "r"(buf)
      : "rcx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8",
        "xmm9","xmm10","xmm11","xmm12","cc","memory");
  return rdtsc_ser() - t0;
}

static void report(const char* name, double instr_per_iter, uint64_t iters,
                   uint64_t (*fn)(uint64_t)) {
  Stats s = repeat_stat([&] { return (double)fn(iters); });
  double cyc_med = ticks_to_cycles(s.med);
  double ipc = instr_per_iter * iters / cyc_med;
  double ipc_hi = instr_per_iter * iters / ticks_to_cycles(s.mn);
  double ipc_lo = instr_per_iter * iters / ticks_to_cycles(s.mx);
  printf("RESULT %-14s %8.4f /cycle  (best %.4f worst %.4f)  cyc/instr=%.4f\n",
         name, ipc, ipc_hi, ipc_lo, 1.0 / ipc);
  fflush(stdout);
}
template <typename F>
static void report_buf(const char* name, double instr_per_iter, uint64_t iters, F fn) {
  Stats s = repeat_stat([&] { return (double)fn(iters); });
  double ipc = instr_per_iter * iters / ticks_to_cycles(s.med);
  double ipc_hi = instr_per_iter * iters / ticks_to_cycles(s.mn);
  double ipc_lo = instr_per_iter * iters / ticks_to_cycles(s.mx);
  printf("RESULT %-14s %8.4f /cycle  (best %.4f worst %.4f)  cyc/instr=%.4f\n",
         name, ipc, ipc_hi, ipc_lo, 1.0 / ipc);
  fflush(stdout);
}

int main() {
  pin_from_env();
  Calib c = calibrate();
  print_calib("alu_start", c);

  const uint64_t I = 40 * 1000 * 1000;
  report("vpmaddubsw", 10, I, k_maddubsw);
  report("vpmaddwd", 10, I, k_maddwd);
  report("vpaddw", 12, I, k_paddw);
  report("vpand", 12, I, k_pand);
  report("vpsrlw", 10, I, k_psrlw);
  report("mix2M2A", 12, I, k_mix2M2A);    // /cycle here = total instr mix IPC
  report("mix1M1A", 12, I, k_mix1M1A);
  report("mix2M1A", 12, I, k_mix2M1A);
  report("mix1M1S", 12, I, k_mix1M1S);
  report("mix1M1SH", 12, I, k_mix1M1SH);
  report("fma231ps", 12, I, k_fma);
  report("vdivps", 8, I / 8, k_divps);

  void* buf = alloc_buf(16384);
  report_buf("loads32B", 16, I / 2, [&](uint64_t it) { return k_loads(it, buf); });
  // combo: per iter 4 groups; report groups/cycle (target 1.0 = 2L+2M+2A per cycle)
  report_buf("combo6", 4, I / 2, [&](uint64_t it) { return k_combo(it, buf); });
  printf("# combo6: groups of (2 loads + 2 vpmaddubsw + 2 vpaddw) per cycle; 1.0 = sustained\n");

  Calib c2 = calibrate(true, 0.1);
  print_calib("alu_end", c2);
  return 0;
}
