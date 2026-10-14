// Cache/memory sequential bandwidth: read (raw sink and vpmaddubsw-style sink),
// read with software prefetcht0 at several distances, and 50/50 read+write.
#include "bench_common.h"

static volatile uint32_t g_sink;

// one full sequential pass, raw: per 32B load -> vpor into rotating accs
__attribute__((noinline)) static void pass_read_raw(const uint8_t* p, size_t bytes) {
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0,
          a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  for (size_t off = 0; off < bytes; off += 256) {
    const __m256i* q = (const __m256i*)(p + off);
    a0 = _mm256_or_si256(a0, _mm256_load_si256(q + 0));
    a1 = _mm256_or_si256(a1, _mm256_load_si256(q + 1));
    a2 = _mm256_or_si256(a2, _mm256_load_si256(q + 2));
    a3 = _mm256_or_si256(a3, _mm256_load_si256(q + 3));
    a4 = _mm256_or_si256(a4, _mm256_load_si256(q + 4));
    a5 = _mm256_or_si256(a5, _mm256_load_si256(q + 5));
    a6 = _mm256_or_si256(a6, _mm256_load_si256(q + 6));
    a7 = _mm256_or_si256(a7, _mm256_load_si256(q + 7));
  }
  __m256i s = _mm256_or_si256(_mm256_or_si256(a0, a1), _mm256_or_si256(a2, a3));
  s = _mm256_or_si256(s, _mm256_or_si256(_mm256_or_si256(a4, a5), _mm256_or_si256(a6, a7)));
  g_sink = _mm256_extract_epi32(s, 0);
}
// madd sink: per 32B: vpmaddubsw(const_u8, mem_s8) then vpaddw
__attribute__((noinline)) static void pass_read_madd(const uint8_t* p, size_t bytes) {
  const __m256i k = _mm256_set1_epi8(3);
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0,
          a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  for (size_t off = 0; off < bytes; off += 256) {
    const __m256i* q = (const __m256i*)(p + off);
    a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 0)));
    a1 = _mm256_add_epi16(a1, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 1)));
    a2 = _mm256_add_epi16(a2, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 2)));
    a3 = _mm256_add_epi16(a3, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 3)));
    a4 = _mm256_add_epi16(a4, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 4)));
    a5 = _mm256_add_epi16(a5, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 5)));
    a6 = _mm256_add_epi16(a6, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 6)));
    a7 = _mm256_add_epi16(a7, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 7)));
  }
  __m256i s = _mm256_add_epi16(_mm256_add_epi16(a0, a1), _mm256_add_epi16(a2, a3));
  s = _mm256_add_epi16(s, _mm256_add_epi16(_mm256_add_epi16(a4, a5), _mm256_add_epi16(a6, a7)));
  g_sink = _mm256_extract_epi32(s, 0);
}
// madd sink + prefetcht0 at distance dist (one per 64B line)
template <int DIST>
__attribute__((noinline)) static void pass_read_madd_pf(const uint8_t* p, size_t bytes) {
  const __m256i k = _mm256_set1_epi8(3);
  __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0,
          a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  for (size_t off = 0; off < bytes; off += 256) {
    const __m256i* q = (const __m256i*)(p + off);
    _mm_prefetch((const char*)p + off + DIST, _MM_HINT_T0);
    a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 0)));
    a1 = _mm256_add_epi16(a1, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 1)));
    _mm_prefetch((const char*)p + off + DIST + 64, _MM_HINT_T0);
    a2 = _mm256_add_epi16(a2, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 2)));
    a3 = _mm256_add_epi16(a3, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 3)));
    _mm_prefetch((const char*)p + off + DIST + 128, _MM_HINT_T0);
    a4 = _mm256_add_epi16(a4, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 4)));
    a5 = _mm256_add_epi16(a5, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 5)));
    _mm_prefetch((const char*)p + off + DIST + 192, _MM_HINT_T0);
    a6 = _mm256_add_epi16(a6, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 6)));
    a7 = _mm256_add_epi16(a7, _mm256_maddubs_epi16(k, _mm256_load_si256(q + 7)));
  }
  __m256i s = _mm256_add_epi16(_mm256_add_epi16(a0, a1), _mm256_add_epi16(a2, a3));
  s = _mm256_add_epi16(s, _mm256_add_epi16(_mm256_add_epi16(a4, a5), _mm256_add_epi16(a6, a7)));
  g_sink = _mm256_extract_epi32(s, 0);
}
// read+write 50/50: per 64B line: load 2x32B, modify, store back (KDA-state-like)
__attribute__((noinline)) static void pass_rw(uint8_t* p, size_t bytes) {
  const __m256i one = _mm256_set1_epi8(1);
  for (size_t off = 0; off < bytes; off += 256) {
    __m256i* q = (__m256i*)(p + off);
    __m256i v0 = _mm256_add_epi8(_mm256_load_si256(q + 0), one);
    __m256i v1 = _mm256_add_epi8(_mm256_load_si256(q + 1), one);
    _mm256_store_si256(q + 0, v0);
    _mm256_store_si256(q + 1, v1);
    __m256i v2 = _mm256_add_epi8(_mm256_load_si256(q + 2), one);
    __m256i v3 = _mm256_add_epi8(_mm256_load_si256(q + 3), one);
    _mm256_store_si256(q + 2, v2);
    _mm256_store_si256(q + 3, v3);
    __m256i v4 = _mm256_add_epi8(_mm256_load_si256(q + 4), one);
    __m256i v5 = _mm256_add_epi8(_mm256_load_si256(q + 5), one);
    _mm256_store_si256(q + 4, v4);
    _mm256_store_si256(q + 5, v5);
    __m256i v6 = _mm256_add_epi8(_mm256_load_si256(q + 6), one);
    __m256i v7 = _mm256_add_epi8(_mm256_load_si256(q + 7), one);
    _mm256_store_si256(q + 6, v6);
    _mm256_store_si256(q + 7, v7);
  }
}

// generic timing: run passes for ~target_s, return bytes/core-cycle
template <typename F>
static double time_bw(F onepass, size_t bytes_per_pass, double target_s = 0.05) {
  onepass();  // warm
  // calibrate pass count
  uint64_t t0 = rdtsc_ser();
  onepass();
  uint64_t one = rdtsc_ser() - t0;
  if (one < 1000) one = 1000;
  uint64_t P = (uint64_t)(target_s * g_cal.tsc_hz / (double)one);
  if (P < 2) P = 2;
  t0 = rdtsc_ser();
  for (uint64_t i = 0; i < P; i++) onepass();
  uint64_t ticks = rdtsc_ser() - t0;
  double cyc = ticks_to_cycles((double)ticks);
  return (double)bytes_per_pass * P / cyc;
}

struct Row { double raw, madd, best_pf; int best_dist; };

int main() {
  pin_from_env();
  Calib c = calibrate();
  print_calib("membw_start", c);
  const size_t sizes[] = {16u << 10, 128u << 10, 384u << 10, 1u << 20,
                          2u << 20, 4u << 20, 8u << 20, 12u << 20,
                          32u << 20, 256u << 20};
  printf("# sequential read bandwidth; B/cyc = bytes per CORE cycle; GB/s at measured core clock\n");
  printf("%-8s %10s %10s %10s %8s %12s %12s\n", "size", "raw B/c", "madd B/c",
         "bestPF B/c", "pf dist", "raw GB/s", "bestPF GB/s");
  for (size_t sz : sizes) {
    uint8_t* buf = (uint8_t*)alloc_buf(sz);
    Row r{};
    r.raw = repeat_stat([&] { return time_bw([&] { pass_read_raw(buf, sz); }, sz); }).med;
    r.madd = repeat_stat([&] { return time_bw([&] { pass_read_madd(buf, sz); }, sz); }).med;
    double pf256 = repeat_stat([&] { return time_bw([&] { pass_read_madd_pf<256>(buf, sz); }, sz); }).med;
    double pf512 = repeat_stat([&] { return time_bw([&] { pass_read_madd_pf<512>(buf, sz); }, sz); }).med;
    double pf1k = repeat_stat([&] { return time_bw([&] { pass_read_madd_pf<1024>(buf, sz); }, sz); }).med;
    double pf2k = repeat_stat([&] { return time_bw([&] { pass_read_madd_pf<2048>(buf, sz); }, sz); }).med;
    r.best_pf = pf256; r.best_dist = 256;
    if (pf512 > r.best_pf) { r.best_pf = pf512; r.best_dist = 512; }
    if (pf1k > r.best_pf) { r.best_pf = pf1k; r.best_dist = 1024; }
    if (pf2k > r.best_pf) { r.best_pf = pf2k; r.best_dist = 2048; }
    printf("RESULT %-8zu %10.3f %10.3f %10.3f %8d %12.2f %12.2f   (pf256=%.2f pf512=%.2f pf1k=%.2f pf2k=%.2f)\n",
           sz >> 10, r.raw, r.madd, r.best_pf, r.best_dist,
           r.raw * g_cal.core_hz / 1e9, r.best_pf * g_cal.core_hz / 1e9,
           pf256, pf512, pf1k, pf2k);
    fflush(stdout);
    munmap(buf, (sz + (2 << 20) - 1) & ~((size_t)(2 << 20) - 1));
  }

  Calib c2 = calibrate(true, 0.1);
  print_calib("membw_mid", c2);

  printf("# read+write 50/50 (load 64B, modify, store 64B per line); B/c counts BOTH directions\n");
  const size_t rw_sizes[] = {128u << 10, 384u << 10, 1u << 20, 2u << 20, 4u << 20, 8u << 20};
  for (size_t sz : rw_sizes) {
    uint8_t* buf = (uint8_t*)alloc_buf(sz);
    double bpc = repeat_stat([&] { return time_bw([&] { pass_rw(buf, sz); }, 2 * sz); }).med;
    printf("RESULT RW %-8zu %10.3f B/c  %10.2f GB/s (r+w)\n", sz >> 10, bpc,
           bpc * g_cal.core_hz / 1e9);
    fflush(stdout);
    munmap(buf, (sz + (2 << 20) - 1) & ~((size_t)(2 << 20) - 1));
  }
  Calib c3 = calibrate(true, 0.1);
  print_calib("membw_end", c3);
  return 0;
}
