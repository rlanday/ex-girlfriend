// Misc: vector exp throughput, scalar libm reference, QK int8 dot variants
// (vpmovsxbw+vpmaddwd vs vpmaddubsw), strided vs sequential KV-like access,
// rdtsc overhead.
#include "bench_common.h"

static volatile float g_fsink;
static volatile uint32_t g_isink;

// ---------------- exp256_ps: cephes-style polynomial, ~1-2 ulp ----------------
static inline __m256 exp256_ps(__m256 x) {
  const __m256 hi = _mm256_set1_ps(88.3762626647950f);
  const __m256 lo = _mm256_set1_ps(-87.3365478515625f);
  const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
  const __m256 C1 = _mm256_set1_ps(0.693359375f);
  const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);
  x = _mm256_min_ps(x, hi);
  x = _mm256_max_ps(x, lo);
  __m256 fx = _mm256_fmadd_ps(x, LOG2EF, half);
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, C1, x);
  x = _mm256_fnmadd_ps(fx, C2, x);
  __m256 z = _mm256_mul_ps(x, x);
  __m256 y = p0;
  y = _mm256_fmadd_ps(y, x, p1);
  y = _mm256_fmadd_ps(y, x, p2);
  y = _mm256_fmadd_ps(y, x, p3);
  y = _mm256_fmadd_ps(y, x, p4);
  y = _mm256_fmadd_ps(y, x, p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);
  __m256i n = _mm256_cvtps_epi32(fx);
  n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

static void exp_accuracy() {
  double maxrel = 0;
  for (int i = 0; i < 200000; i++) {
    float x = -30.0f + 60.0f * (float)i / 200000.0f;
    float v[8] __attribute__((aligned(32)));
    _mm256_store_ps(v, exp256_ps(_mm256_set1_ps(x)));
    double ref = exp((double)x);
    double rel = fabs((double)v[0] - ref) / ref;
    maxrel = std::max(maxrel, rel);
  }
  printf("# exp256_ps max rel err vs exp(double) on [-30,30]: %.3g (%.2f ulp-ish)\n",
         maxrel, maxrel / 1.19e-7);
}

__attribute__((noinline)) static void exp_pass(const float* in, float* outbuf, int n) {
  for (int i = 0; i < n; i += 32) {
    _mm256_store_ps(outbuf + i + 0, exp256_ps(_mm256_load_ps(in + i + 0)));
    _mm256_store_ps(outbuf + i + 8, exp256_ps(_mm256_load_ps(in + i + 8)));
    _mm256_store_ps(outbuf + i + 16, exp256_ps(_mm256_load_ps(in + i + 16)));
    _mm256_store_ps(outbuf + i + 24, exp256_ps(_mm256_load_ps(in + i + 24)));
  }
}
template <float F(float)>
__attribute__((noinline)) static void libm_pass(const float* in, float* outbuf, int n) {
  for (int i = 0; i < n; i++) outbuf[i] = F(in[i]);
}

// ---------------- QK dot variants ----------------
// k rows: 64 int8 (one attention head pos). Score = float(dot) * scale.
// variant 1: k int8 -> vpmovsxbw -> int16, vpmaddwd with q int16, vpaddd.
__attribute__((noinline)) static void qk_pmovsx(const int8_t* krows, int nrows,
                                                const int16_t* q16, float scale,
                                                float* scores) {
  const __m256i q0 = _mm256_load_si256((const __m256i*)(q16 + 0));
  const __m256i q1 = _mm256_load_si256((const __m256i*)(q16 + 16));
  const __m256i q2 = _mm256_load_si256((const __m256i*)(q16 + 32));
  const __m256i q3 = _mm256_load_si256((const __m256i*)(q16 + 48));
  for (int r = 0; r < nrows; r++) {
    const int8_t* k = krows + (size_t)r * 64;
    __m256i a = _mm256_madd_epi16(q0, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(k + 0))));
    a = _mm256_add_epi32(a, _mm256_madd_epi16(q1, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(k + 16)))));
    a = _mm256_add_epi32(a, _mm256_madd_epi16(q2, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(k + 32)))));
    a = _mm256_add_epi32(a, _mm256_madd_epi16(q3, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(k + 48)))));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    scores[r] = (float)_mm_cvtsi128_si32(s) * scale;
  }
}
// variant 2: q biased u8 (q+128), vpmaddubsw(q_u, k_s) + per-row corr 128*sum(k)
__attribute__((noinline)) static void qk_maddubs(const int8_t* krows, int nrows,
                                                 const uint8_t* qb, const int32_t* corr,
                                                 float scale, float* scores) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i q0 = _mm256_load_si256((const __m256i*)(qb + 0));
  const __m256i q1 = _mm256_load_si256((const __m256i*)(qb + 32));
  for (int r = 0; r < nrows; r++) {
    const __m256i* k = (const __m256i*)(krows + (size_t)r * 64);
    __m256i m = _mm256_add_epi16(_mm256_maddubs_epi16(q0, _mm256_load_si256(k)),
                                 _mm256_maddubs_epi16(q1, _mm256_load_si256(k + 1)));
    __m256i d = _mm256_madd_epi16(m, ones);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    scores[r] = (float)(_mm_cvtsi128_si32(s) - corr[r]) * scale;
  }
}

// ---------------- strided vs sequential 192B-row reads (KV-cache-like) ------
__attribute__((noinline)) static void rows_seq(const float* buf, int nrows, float* sink) {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
  for (int r = 0; r < nrows; r++) {
    const float* p = buf + (size_t)r * 48;
    a0 = _mm256_add_ps(a0, _mm256_load_ps(p + 0));
    a1 = _mm256_add_ps(a1, _mm256_load_ps(p + 8));
    a2 = _mm256_add_ps(a2, _mm256_load_ps(p + 16));
    a3 = _mm256_add_ps(a3, _mm256_load_ps(p + 24));
    a4 = _mm256_add_ps(a4, _mm256_load_ps(p + 32));
    a5 = _mm256_add_ps(a5, _mm256_load_ps(p + 40));
  }
  __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
  s = _mm256_add_ps(s, _mm256_add_ps(a4, a5));
  _mm256_store_ps(sink, s);
}
__attribute__((noinline)) static void rows_rand(const float* buf, const uint16_t* perm,
                                                int nrows, float* sink) {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
  for (int r = 0; r < nrows; r++) {
    const float* p = buf + (size_t)perm[r] * 48;
    a0 = _mm256_add_ps(a0, _mm256_load_ps(p + 0));
    a1 = _mm256_add_ps(a1, _mm256_load_ps(p + 8));
    a2 = _mm256_add_ps(a2, _mm256_load_ps(p + 16));
    a3 = _mm256_add_ps(a3, _mm256_load_ps(p + 24));
    a4 = _mm256_add_ps(a4, _mm256_load_ps(p + 32));
    a5 = _mm256_add_ps(a5, _mm256_load_ps(p + 40));
  }
  __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
  s = _mm256_add_ps(s, _mm256_add_ps(a4, a5));
  _mm256_store_ps(sink, s);
}

template <typename F>
static double time_rate(F onepass, double units_per_pass, double target_s = 0.05) {
  onepass();
  uint64_t t0 = rdtsc_ser();
  onepass();
  uint64_t one = rdtsc_ser() - t0;
  if (one < 1000) one = 1000;
  uint64_t P = (uint64_t)(target_s * g_cal.tsc_hz / (double)one);
  if (P < 2) P = 2;
  t0 = rdtsc_ser();
  for (uint64_t i = 0; i < P; i++) onepass();
  double cyc = ticks_to_cycles((double)(rdtsc_ser() - t0));
  return units_per_pass * P / cyc;  // units per core cycle
}

int main() {
  pin_from_env();
  Calib c = calibrate();
  print_calib("misc_start", c);
  exp_accuracy();

  // exp throughput
  const int NE = 4096;
  float* ein = (float*)alloc_buf(NE * 4);
  float* eout = (float*)alloc_buf(NE * 4);
  for (int i = 0; i < NE; i++) ein[i] = -21.0f + 21.0f * (float)i / NE;  // softmax-like args
  {
    Stats s = repeat_stat([&] { return time_rate([&] { exp_pass(ein, eout, NE); }, NE); });
    g_fsink = eout[7];
    printf("RESULT exp256_ps      %7.3f elem/cyc (best %.3f worst %.3f)  = %.2f cyc/elem\n",
           s.med, s.mx, s.mn, 1.0 / s.med);
    Stats se = repeat_stat([&] { return time_rate([&] { libm_pass<expf>(ein, eout, NE); }, NE); });
    printf("RESULT libm_expf      %7.4f elem/cyc  = %.1f cyc/elem\n", se.med, 1.0 / se.med);
    Stats st = repeat_stat([&] { return time_rate([&] { libm_pass<tanhf>(ein, eout, NE); }, NE); });
    printf("RESULT libm_tanhf     %7.4f elem/cyc  = %.1f cyc/elem\n", st.med, 1.0 / st.med);
    Stats ss = repeat_stat([&] { return time_rate([&] { libm_pass<sinf>(ein, eout, NE); }, NE); });
    printf("RESULT libm_sinf      %7.4f elem/cyc  = %.1f cyc/elem\n", ss.med, 1.0 / ss.med);
    fflush(stdout);
  }

  // QK variants, k streaming from 256KB and 1MB footprints
  {
    int8_t q8[64];
    int16_t q16[64] __attribute__((aligned(64)));
    uint8_t qb[64] __attribute__((aligned(64)));
    uint32_t st = 99;
    for (int i = 0; i < 64; i++) {
      st = st * 1664525u + 1013904223u;
      q8[i] = (int8_t)(st >> 24);
      q16[i] = q8[i];
      qb[i] = (uint8_t)(q8[i] + 128);
    }
    for (size_t foot : {256u << 10, 1u << 20}) {
      int nrows = (int)(foot / 64);
      int8_t* krows = (int8_t*)alloc_buf(foot);
      int32_t* corr = (int32_t*)alloc_buf(nrows * 4);
      for (size_t i = 0; i < foot; i++) { st = st * 1664525u + 1013904223u; krows[i] = (int8_t)(st >> 24); }
      for (int r = 0; r < nrows; r++) {
        int s = 0;
        for (int i = 0; i < 64; i++) s += krows[r * 64 + i];
        corr[r] = 128 * s;
      }
      float* scores = (float*)alloc_buf(nrows * 4);
      // correctness of pmovsx variant vs scalar reference
      qk_pmovsx(krows, 4, q16, 0.01f, scores);
      for (int r = 0; r < 4; r++) {
        int d = 0;
        for (int i = 0; i < 64; i++) d += (int)q8[i] * krows[r * 64 + i];
        float ref = (float)d * 0.01f;
        if (scores[r] != ref) { printf("QK pmovsx MISMATCH r%d %f vs %f\n", r, scores[r], ref); return 1; }
      }
      // NOTE: a vpmaddubsw-based QK ((q+128) u8 x k s8) is INVALID for
      // full-range int8: pair products reach 255*127*2 = 64770 > 32767 and
      // vpmaddubsw saturates. It is only safe when one operand is small
      // (e.g. |w|<=7 in the weight matmuls). We still measure its raw speed
      // below for reference, but it cannot be used for QK.
      double macs = 64.0 * nrows;
      Stats s1 = repeat_stat([&] { return time_rate([&] { qk_pmovsx(krows, nrows, q16, 0.01f, scores); }, macs); });
      Stats s2 = repeat_stat([&] { return time_rate([&] { qk_maddubs(krows, nrows, qb, corr, 0.01f, scores); }, macs); });
      g_fsink = scores[0];
      printf("RESULT qk_pmovsx_%-4zuK %7.3f MAC/cyc (best %.3f worst %.3f) = %.2f cyc/row64\n",
             foot >> 10, s1.med, s1.mx, s1.mn, 64.0 / s1.med);
      printf("RESULT qk_maddubs_%-3zuK %7.3f MAC/cyc (best %.3f worst %.3f) = %.2f cyc/row64\n",
             foot >> 10, s2.med, s2.mx, s2.mn, 64.0 / s2.med);
      fflush(stdout);
    }
  }

  // strided vs sequential 192B rows in ~1.18MB (KV-cache-like)
  {
    const int nrows = 6144;  // 6144 * 192B = 1.18 MB
    float* buf = (float*)alloc_buf((size_t)nrows * 192);
    uint16_t* perm = (uint16_t*)alloc_buf(nrows * 2);
    for (int i = 0; i < nrows; i++) perm[i] = (uint16_t)i;
    uint32_t st = 12345;
    for (int i = nrows - 1; i > 0; i--) {
      st = st * 1664525u + 1013904223u;
      int j = (int)((st >> 8) % (uint32_t)(i + 1));
      std::swap(perm[i], perm[j]);
    }
    float sink[8] __attribute__((aligned(32)));
    double bytes = 192.0 * nrows;
    Stats sq = repeat_stat([&] { return time_rate([&] { rows_seq(buf, nrows, sink); }, bytes); });
    Stats sr = repeat_stat([&] { return time_rate([&] { rows_rand(buf, perm, nrows, sink); }, bytes); });
    g_fsink = sink[0];
    printf("RESULT kvrows_seq     %7.3f B/cyc (%.2f cyc/192B-row)  %.2f GB/s\n",
           sq.med, 192.0 / sq.med, sq.med * g_cal.core_hz / 1e9);
    printf("RESULT kvrows_rand    %7.3f B/cyc (%.2f cyc/192B-row)  %.2f GB/s  penalty x%.2f\n",
           sr.med, 192.0 / sr.med, sr.med * g_cal.core_hz / 1e9, sq.med / sr.med);
    fflush(stdout);
  }

  // rdtsc overhead
  {
    const int N = 4096;
    std::vector<uint64_t> t(N);
    for (int rep = 0; rep < 3; rep++)
      for (int i = 0; i < N; i++) t[i] = rdtsc_ser();
    std::vector<double> d;
    for (int i = 1; i < N; i++) d.push_back((double)(t[i] - t[i - 1]));
    std::sort(d.begin(), d.end());
    double med = d[d.size() / 2];
    printf("RESULT rdtsc_lfence   %.1f tsc-ticks = %.1f core cycles (median back-to-back)\n",
           med, ticks_to_cycles(med));
    // rdtscp; lfence
    auto rdtscp_ser = []() -> uint64_t {
      unsigned lo, hi, aux;
      asm volatile("rdtscp\n\tlfence" : "=a"(lo), "=d"(hi), "=c"(aux)::"memory");
      return ((uint64_t)hi << 32) | lo;
    };
    for (int rep = 0; rep < 3; rep++)
      for (int i = 0; i < N; i++) t[i] = rdtscp_ser();
    d.clear();
    for (int i = 1; i < N; i++) d.push_back((double)(t[i] - t[i - 1]));
    std::sort(d.begin(), d.end());
    med = d[d.size() / 2];
    printf("RESULT rdtscp_lfence  %.1f tsc-ticks = %.1f core cycles (median back-to-back)\n",
           med, ticks_to_cycles(med));
  }
  Calib c2 = calibrate(true, 0.1);
  print_calib("misc_end", c2);
  return 0;
}
