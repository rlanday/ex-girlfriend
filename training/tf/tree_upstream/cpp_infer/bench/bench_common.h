// Common timing/calibration infrastructure for machine microbenchmarks.
// All "cycles" reported are CORE cycles: rdtsc ticks scaled by the measured
// core_hz/tsc_hz ratio (core freq measured with a dependent 1-cycle-latency
// vpaddd chain under AVX2-warm conditions).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>

static inline uint64_t rdtsc_ser() {
  unsigned lo, hi;
  asm volatile("lfence\n\trdtsc" : "=a"(lo), "=d"(hi)::"memory");
  return ((uint64_t)hi << 32) | lo;
}
static inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// ---------------- TSC frequency vs CLOCK_MONOTONIC_RAW ----------------
static double measure_tsc_hz(double seconds = 0.25) {
  uint64_t n0 = now_ns();
  uint64_t t0 = rdtsc_ser();
  uint64_t target = n0 + (uint64_t)(seconds * 1e9);
  uint64_t n1;
  do { n1 = now_ns(); } while (n1 < target);
  uint64_t t1 = rdtsc_ser();
  return (double)(t1 - t0) * 1e9 / (double)(n1 - n0);
}

// ------------- core frequency via dependent 1-cycle-op chains -------------
// 10 dependent vpaddd (latency 1 on Zen 2) per iteration -> 10 core cycles/iter.
static uint64_t chain_vpaddd_ticks(uint64_t iters) {
  uint64_t n = iters;
  uint64_t t0 = rdtsc_ser();
  asm volatile(
      "vpxor %%xmm0,%%xmm0,%%xmm0\n\t"
      "vpcmpeqd %%ymm1,%%ymm1,%%ymm1\n\t"
      "1:\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "vpaddd %%ymm1,%%ymm0,%%ymm0\n\t"
      "dec %0\n\t"
      "jnz 1b\n\t"
      : "+r"(n)::"xmm0", "xmm1", "cc", "memory");
  return rdtsc_ser() - t0;
}
// scalar cross-check: 10 dependent 64-bit adds
static uint64_t chain_scalar_ticks(uint64_t iters) {
  uint64_t n = iters, a = 0, one = 1;
  uint64_t t0 = rdtsc_ser();
  asm volatile(
      "1:\n\t"
      "add %2,%0\n\t" "add %2,%0\n\t" "add %2,%0\n\t" "add %2,%0\n\t"
      "add %2,%0\n\t" "add %2,%0\n\t" "add %2,%0\n\t" "add %2,%0\n\t"
      "add %2,%0\n\t" "add %2,%0\n\t"
      "dec %1\n\t"
      "jnz 1b\n\t"
      : "+r"(a), "+r"(n) : "r"(one) : "cc", "memory");
  uint64_t t1 = rdtsc_ser();
  asm volatile("" ::"r"(a));  // keep a live
  return t1 - t0;
}

// FMA warmup loop to reach steady-state AVX2 frequency
static void avx2_warmup(double seconds) {
  uint64_t end = now_ns() + (uint64_t)(seconds * 1e9);
  float onef = 1.0f, tiny = 1e-30f;
  do {
    uint64_t n = 2000000;
    asm volatile(
        "vbroadcastss %1,%%ymm14\n\t"
        "vbroadcastss %2,%%ymm15\n\t"
        "vxorps %%xmm0,%%xmm0,%%xmm0\n\t" "vxorps %%xmm1,%%xmm1,%%xmm1\n\t"
        "vxorps %%xmm2,%%xmm2,%%xmm2\n\t" "vxorps %%xmm3,%%xmm3,%%xmm3\n\t"
        "vxorps %%xmm4,%%xmm4,%%xmm4\n\t" "vxorps %%xmm5,%%xmm5,%%xmm5\n\t"
        "vxorps %%xmm6,%%xmm6,%%xmm6\n\t" "vxorps %%xmm7,%%xmm7,%%xmm7\n\t"
        "vxorps %%xmm8,%%xmm8,%%xmm8\n\t" "vxorps %%xmm9,%%xmm9,%%xmm9\n\t"
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
        "dec %0\n\t"
        "jnz 1b\n\t"
        : "+r"(n)
        : "m"(tiny), "m"(onef)
        : "xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9",
          "xmm14","xmm15","cc","memory");
  } while (now_ns() < end);
}

struct Calib {
  double tsc_hz;
  double core_hz;         // via vpaddd chain
  double core_hz_scalar;  // via scalar add chain (cross-check)
  double ratio() const { return core_hz / tsc_hz; }
};
static double g_tsc_hz = 0;
static Calib g_cal;

static Calib calibrate(bool warm = true, double warm_s = 0.4) {
  Calib c{};
  if (g_tsc_hz == 0) g_tsc_hz = measure_tsc_hz(0.25);
  c.tsc_hz = g_tsc_hz;
  if (warm) avx2_warmup(warm_s);
  const uint64_t iters = 20 * 1000 * 1000;  // 200M dependent ops
  double best_v = 1e18, best_s = 1e18;
  for (int r = 0; r < 5; r++) {
    uint64_t tv = chain_vpaddd_ticks(iters);
    uint64_t ts = chain_scalar_ticks(iters);
    best_v = std::min(best_v, (double)tv);
    best_s = std::min(best_s, (double)ts);
  }
  c.core_hz = c.tsc_hz * (double)(iters * 10) / best_v;
  c.core_hz_scalar = c.tsc_hz * (double)(iters * 10) / best_s;
  g_cal = c;
  return c;
}
static void print_calib(const char* tag, const Calib& c) {
  printf("CALIB %s tsc_hz=%.4e core_hz=%.4e (scalar-chain check %.4e) ratio=%.4f\n",
         tag, c.tsc_hz, c.core_hz, c.core_hz_scalar, c.ratio());
  fflush(stdout);
}
static inline double ticks_to_cycles(double ticks) { return ticks * g_cal.ratio(); }

// ---------------- stats over repetitions ----------------
struct Stats { double med, mn, mx; };
template <typename F>
static Stats repeat_stat(F f, int reps = 5, int warmup = 1) {
  std::vector<double> v;
  for (int i = 0; i < warmup; i++) f();
  for (int i = 0; i < reps; i++) v.push_back(f());
  std::sort(v.begin(), v.end());
  Stats s;
  s.med = v[v.size() / 2];
  s.mn = v.front();
  s.mx = v.back();
  return s;
}

// ---------------- allocation ----------------
static void* alloc_buf(size_t bytes) {
  size_t asz = (bytes + (2 << 20) - 1) & ~((size_t)(2 << 20) - 1);
  void* p = mmap(nullptr, asz, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); exit(1); }
  if (bytes >= (2u << 20)) madvise(p, asz, MADV_HUGEPAGE);
  memset(p, 1, asz);  // touch
  return p;
}

static void pin_from_env() {
  const char* e = getenv("BENCH_CPU");
  if (!e) return;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(atoi(e), &s);
  if (sched_setaffinity(0, sizeof(s), &s) != 0) perror("sched_setaffinity");
}
