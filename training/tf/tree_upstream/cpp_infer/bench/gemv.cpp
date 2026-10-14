// The realistic int8-activation x int4-weight GEMV inner-loop bound.
// Two storage variants:
//  (a) UNPACKED: weights stored as int8 in [-7,7]. Activations biased unsigned
//      (qa+128). Inner op per 32 weights: vpmaddubsw(act_reg_u8, w_mem_s8)
//      (memory operand folds the load) + vpaddw. Per-row correction 128*sum(qw)
//      streams with the weights (32B meta per 4-row group).
//  (b) PACKED int4: two weights/byte, stored as unsigned nibbles qw+7 in
//      [0,14]. Activations signed int8. Per 32B load: vpand / vpsrlw+vpand
//      unpack into two u4-in-u8 regs, then 2x vpmaddubsw(w_u, act_s) + vpaddw.
//      Correction is global: 7*sum(qa) (one scalar per matmul) -> cheaper than
//      unpacked's per-row correction. Per-group meta: 4 fp32 row scales.
// Unit "matrix" = 192 rows x IN cols (IN = 192 or 768), rows processed in
// groups of 4, weights laid out in exact consumption order (pure sequential
// stream, hardware-prefetch friendly). Arena = N matrices cycled -> footprint.
#include "bench_common.h"

static uint32_t rng_state = 0x12345678;
static inline int rnd(int lo, int hi) {  // inclusive
  rng_state = rng_state * 1664525u + 1013904223u;
  return lo + (int)((rng_state >> 8) % (uint32_t)(hi - lo + 1));
}

// ---------------- layout builders ----------------
// unpacked: mat = G groups; group = 32B meta (4x i32 corr, 4x f32 scale) +
//           CH chunks x 4 rows x 32B.   CH = IN/32
static size_t unpacked_matbytes(int IN) { return 48 * (32 + (size_t)(IN / 32) * 4 * 32); }
static size_t packed_matbytes(int IN) { return 48 * (32 + (size_t)(IN / 64) * 4 * 32); }

static void fill_mat(uint8_t* up, uint8_t* pk, int IN, int8_t* qw /*192*IN*/,
                     float* scales /*192*/) {
  for (int i = 0; i < 192 * IN; i++) qw[i] = (int8_t)rnd(-7, 7);
  for (int r = 0; r < 192; r++) scales[r] = 0.001f * rnd(1, 1000);
  const int CH = IN / 32, PR = IN / 64;
  for (int g = 0; g < 48; g++) {
    if (up) {
      uint8_t* gp = up + g * (32 + (size_t)CH * 128);
      int32_t* corr = (int32_t*)gp;
      float* scl = (float*)(gp + 16);
      for (int r = 0; r < 4; r++) {
        int s = 0;
        for (int i = 0; i < IN; i++) s += qw[(4 * g + r) * IN + i];
        corr[r] = 128 * s;
        scl[r] = scales[4 * g + r];
      }
      uint8_t* w = gp + 32;
      for (int c = 0; c < CH; c++)
        for (int r = 0; r < 4; r++)
          for (int j = 0; j < 32; j++)
            *w++ = (uint8_t)qw[(4 * g + r) * IN + 32 * c + j];
    }
    if (pk) {
      uint8_t* gp = pk + g * (32 + (size_t)PR * 128);
      float* scl = (float*)gp;
      for (int r = 0; r < 4; r++) scl[r] = scales[4 * g + r];
      uint8_t* w = gp + 32;
      for (int p = 0; p < PR; p++)
        for (int r = 0; r < 4; r++)
          for (int j = 0; j < 32; j++) {
            int lo = qw[(4 * g + r) * IN + 64 * p + j] + 7;
            int hi = qw[(4 * g + r) * IN + 64 * p + 32 + j] + 7;
            *w++ = (uint8_t)((hi << 4) | lo);
          }
    }
  }
}

// ---------------- kernels ----------------
// IN=192 unpacked: act held in 6 ymm registers
__attribute__((noinline)) static void gemv192_unpacked(
    const uint8_t* arena, int nmats, const uint8_t* act_u8, float* out) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i A0 = _mm256_load_si256((const __m256i*)(act_u8 + 0));
  const __m256i A1 = _mm256_load_si256((const __m256i*)(act_u8 + 32));
  const __m256i A2 = _mm256_load_si256((const __m256i*)(act_u8 + 64));
  const __m256i A3 = _mm256_load_si256((const __m256i*)(act_u8 + 96));
  const __m256i A4 = _mm256_load_si256((const __m256i*)(act_u8 + 128));
  const __m256i A5 = _mm256_load_si256((const __m256i*)(act_u8 + 160));
  const size_t MB = unpacked_matbytes(192);
  for (int m = 0; m < nmats; m++) {
    const uint8_t* mp = arena + m * MB;
    for (int g = 0; g < 48; g++) {
      const uint8_t* gp = mp + g * 800;
      __m128i corr = _mm_load_si128((const __m128i*)gp);
      __m128 scl = _mm_load_ps((const float*)(gp + 16));
      const __m256i* w = (const __m256i*)(gp + 32);
      __m256i s0 = _mm256_maddubs_epi16(A0, w[0]);
      __m256i s1 = _mm256_maddubs_epi16(A0, w[1]);
      __m256i s2 = _mm256_maddubs_epi16(A0, w[2]);
      __m256i s3 = _mm256_maddubs_epi16(A0, w[3]);
#define STEP_U(c, A)                                              \
  s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(A, w[(c)*4+0])); \
  s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(A, w[(c)*4+1])); \
  s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(A, w[(c)*4+2])); \
  s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(A, w[(c)*4+3]));
      STEP_U(1, A1) STEP_U(2, A2) STEP_U(3, A3) STEP_U(4, A4) STEP_U(5, A5)
#undef STEP_U
      __m256i d0 = _mm256_madd_epi16(s0, ones);
      __m256i d1 = _mm256_madd_epi16(s1, ones);
      __m256i d2 = _mm256_madd_epi16(s2, ones);
      __m256i d3 = _mm256_madd_epi16(s3, ones);
      __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1),
                                    _mm256_hadd_epi32(d2, d3));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                                _mm256_extracti128_si256(h, 1));
      s = _mm_sub_epi32(s, corr);
      _mm_store_ps(out + g * 4, _mm_mul_ps(_mm_cvtepi32_ps(s), scl));
    }
  }
}

// IN=192 packed: act (signed) held in 6 ymm registers
__attribute__((noinline)) static void gemv192_packed(
    const uint8_t* arena, int nmats, const int8_t* act_s8, int sum_act, float* out) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m128i corr = _mm_set1_epi32(7 * sum_act);
  const __m256i A0 = _mm256_load_si256((const __m256i*)(act_s8 + 0));
  const __m256i A1 = _mm256_load_si256((const __m256i*)(act_s8 + 32));
  const __m256i A2 = _mm256_load_si256((const __m256i*)(act_s8 + 64));
  const __m256i A3 = _mm256_load_si256((const __m256i*)(act_s8 + 96));
  const __m256i A4 = _mm256_load_si256((const __m256i*)(act_s8 + 128));
  const __m256i A5 = _mm256_load_si256((const __m256i*)(act_s8 + 160));
  const size_t MB = packed_matbytes(192);
  for (int m = 0; m < nmats; m++) {
    const uint8_t* mp = arena + m * MB;
    for (int g = 0; g < 48; g++) {
      const uint8_t* gp = mp + g * 416;
      __m128 scl = _mm_load_ps((const float*)gp);
      const __m256i* w = (const __m256i*)(gp + 32);
      __m256i s0, s1, s2, s3;
#define STEP_P0(r, Alo, Ahi)                                       \
  {                                                                \
    __m256i wv = _mm256_load_si256(w + r);                         \
    __m256i lo = _mm256_and_si256(wv, m0F);                        \
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F);  \
    s##r = _mm256_add_epi16(_mm256_maddubs_epi16(lo, Alo),         \
                            _mm256_maddubs_epi16(hi, Ahi));        \
  }
#define STEP_P(p, r, Alo, Ahi)                                     \
  {                                                                \
    __m256i wv = _mm256_load_si256(w + (p)*4 + r);                 \
    __m256i lo = _mm256_and_si256(wv, m0F);                        \
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F);  \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(lo, Alo));  \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(hi, Ahi));  \
  }
      STEP_P0(0, A0, A1) STEP_P0(1, A0, A1) STEP_P0(2, A0, A1) STEP_P0(3, A0, A1)
      STEP_P(1, 0, A2, A3) STEP_P(1, 1, A2, A3) STEP_P(1, 2, A2, A3) STEP_P(1, 3, A2, A3)
      STEP_P(2, 0, A4, A5) STEP_P(2, 1, A4, A5) STEP_P(2, 2, A4, A5) STEP_P(2, 3, A4, A5)
#undef STEP_P0
#undef STEP_P
      __m256i d0 = _mm256_madd_epi16(s0, ones);
      __m256i d1 = _mm256_madd_epi16(s1, ones);
      __m256i d2 = _mm256_madd_epi16(s2, ones);
      __m256i d3 = _mm256_madd_epi16(s3, ones);
      __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1),
                                    _mm256_hadd_epi32(d2, d3));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                                _mm256_extracti128_si256(h, 1));
      s = _mm_sub_epi32(s, corr);
      _mm_store_ps(out + g * 4, _mm_mul_ps(_mm_cvtepi32_ps(s), scl));
    }
  }
}

// v2 kernels: act loaded from L1 per chunk (fewer live registers, better
// scheduling) instead of pinned in 6 ymm. PF variant adds prefetcht0 +2KB.
template <bool PF>
__attribute__((noinline)) static void gemv192_unpacked_v2(
    const uint8_t* arena, int nmats, const uint8_t* act_u8, float* out) {
  const __m256i ones = _mm256_set1_epi16(1);
  const size_t MB = unpacked_matbytes(192);
  for (int m = 0; m < nmats; m++) {
    const uint8_t* mp = arena + m * MB;
    for (int g = 0; g < 48; g++) {
      const uint8_t* gp = mp + g * 800;
      __m128i corr = _mm_load_si128((const __m128i*)gp);
      __m128 scl = _mm_load_ps((const float*)(gp + 16));
      const __m256i* w = (const __m256i*)(gp + 32);
      __m256i av = _mm256_load_si256((const __m256i*)act_u8);
      __m256i s0 = _mm256_maddubs_epi16(av, w[0]);
      __m256i s1 = _mm256_maddubs_epi16(av, w[1]);
      __m256i s2 = _mm256_maddubs_epi16(av, w[2]);
      __m256i s3 = _mm256_maddubs_epi16(av, w[3]);
      for (int c = 1; c < 6; c++) {
        if (PF && (c & 1)) {
          _mm_prefetch((const char*)gp + c * 128 + 2048, _MM_HINT_T0);
          _mm_prefetch((const char*)gp + c * 128 + 2112, _MM_HINT_T0);
          _mm_prefetch((const char*)gp + c * 128 + 2176, _MM_HINT_T0);
        }
        av = _mm256_load_si256((const __m256i*)(act_u8 + 32 * c));
        s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(av, w[c * 4 + 0]));
        s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(av, w[c * 4 + 1]));
        s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(av, w[c * 4 + 2]));
        s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(av, w[c * 4 + 3]));
      }
      __m256i h = _mm256_hadd_epi32(
          _mm256_hadd_epi32(_mm256_madd_epi16(s0, ones), _mm256_madd_epi16(s1, ones)),
          _mm256_hadd_epi32(_mm256_madd_epi16(s2, ones), _mm256_madd_epi16(s3, ones)));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                                _mm256_extracti128_si256(h, 1));
      s = _mm_sub_epi32(s, corr);
      _mm_store_ps(out + g * 4, _mm_mul_ps(_mm_cvtepi32_ps(s), scl));
    }
  }
}

template <bool PF>
__attribute__((noinline)) static void gemv192_packed_v2(
    const uint8_t* arena, int nmats, const int8_t* act_s8, int sum_act, float* out) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m128i corr = _mm_set1_epi32(7 * sum_act);
  const size_t MB = packed_matbytes(192);
  for (int m = 0; m < nmats; m++) {
    const uint8_t* mp = arena + m * MB;
    for (int g = 0; g < 48; g++) {
      const uint8_t* gp = mp + g * 416;
      __m128 scl = _mm_load_ps((const float*)gp);
      const __m256i* w = (const __m256i*)(gp + 32);
      __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
      for (int p = 0; p < 3; p++) {
        if (PF && p == 0) _mm_prefetch((const char*)gp + 1024, _MM_HINT_T0);
        __m256i a0 = _mm256_load_si256((const __m256i*)(act_s8 + 64 * p));
        __m256i a1 = _mm256_load_si256((const __m256i*)(act_s8 + 64 * p + 32));
#define PSTEP2(r)                                                    \
  {                                                                  \
    __m256i wv = _mm256_load_si256(w + p * 4 + r);                   \
    __m256i lo = _mm256_and_si256(wv, m0F);                          \
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F);    \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(lo, a0));     \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(hi, a1));     \
  }
        PSTEP2(0) PSTEP2(1) PSTEP2(2) PSTEP2(3)
#undef PSTEP2
      }
      __m256i h = _mm256_hadd_epi32(
          _mm256_hadd_epi32(_mm256_madd_epi16(s0, ones), _mm256_madd_epi16(s1, ones)),
          _mm256_hadd_epi32(_mm256_madd_epi16(s2, ones), _mm256_madd_epi16(s3, ones)));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                                _mm256_extracti128_si256(h, 1));
      s = _mm_sub_epi32(s, corr);
      _mm_store_ps(out + g * 4, _mm_mul_ps(_mm_cvtepi32_ps(s), scl));
    }
  }
}

// IN=768 unpacked: act (768B, biased u8) loaded from L1 per chunk; int16 accs
// widened to int32 every 8 chunks
template <bool PF>
__attribute__((noinline)) static void gemv768_unpacked(
    const uint8_t* arena, int nmats, const uint8_t* act_u8, float* out) {
  const __m256i ones = _mm256_set1_epi16(1);
  const size_t MB = unpacked_matbytes(768);
  for (int m = 0; m < nmats; m++) {
    const uint8_t* mp = arena + m * MB;
    for (int g = 0; g < 48; g++) {
      const uint8_t* gp = mp + g * 3104;
      __m128i corr = _mm_load_si128((const __m128i*)gp);
      __m128 scl = _mm_load_ps((const float*)(gp + 16));
      const __m256i* w = (const __m256i*)(gp + 32);
      __m256i d0 = _mm256_setzero_si256(), d1 = d0, d2 = d0, d3 = d0;
      for (int b = 0; b < 3; b++) {
        const __m256i* wb = w + b * 32;
        const uint8_t* ab = act_u8 + b * 256;
        __m256i av = _mm256_load_si256((const __m256i*)ab);
        __m256i s0 = _mm256_maddubs_epi16(av, wb[0]);
        __m256i s1 = _mm256_maddubs_epi16(av, wb[1]);
        __m256i s2 = _mm256_maddubs_epi16(av, wb[2]);
        __m256i s3 = _mm256_maddubs_epi16(av, wb[3]);
        for (int c = 1; c < 8; c++) {
          if (PF) {
            _mm_prefetch((const char*)wb + c * 128 + 2048, _MM_HINT_T0);
            _mm_prefetch((const char*)wb + c * 128 + 2112, _MM_HINT_T0);
          }
          av = _mm256_load_si256((const __m256i*)(ab + 32 * c));
          s0 = _mm256_add_epi16(s0, _mm256_maddubs_epi16(av, wb[c * 4 + 0]));
          s1 = _mm256_add_epi16(s1, _mm256_maddubs_epi16(av, wb[c * 4 + 1]));
          s2 = _mm256_add_epi16(s2, _mm256_maddubs_epi16(av, wb[c * 4 + 2]));
          s3 = _mm256_add_epi16(s3, _mm256_maddubs_epi16(av, wb[c * 4 + 3]));
        }
        d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(s0, ones));
        d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(s1, ones));
        d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(s2, ones));
        d3 = _mm256_add_epi32(d3, _mm256_madd_epi16(s3, ones));
      }
      __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1),
                                    _mm256_hadd_epi32(d2, d3));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                                _mm256_extracti128_si256(h, 1));
      s = _mm_sub_epi32(s, corr);
      _mm_store_ps(out + g * 4, _mm_mul_ps(_mm_cvtepi32_ps(s), scl));
    }
  }
}

// IN=768 packed
__attribute__((noinline)) static void gemv768_packed(
    const uint8_t* arena, int nmats, const int8_t* act_s8, int sum_act, float* out) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i m0F = _mm256_set1_epi8(0x0F);
  const __m128i corr = _mm_set1_epi32(7 * sum_act);
  const size_t MB = packed_matbytes(768);
  for (int m = 0; m < nmats; m++) {
    const uint8_t* mp = arena + m * MB;
    for (int g = 0; g < 48; g++) {
      const uint8_t* gp = mp + g * 1568;
      __m128 scl = _mm_load_ps((const float*)gp);
      const __m256i* w = (const __m256i*)(gp + 32);
      __m256i d0 = _mm256_setzero_si256(), d1 = d0, d2 = d0, d3 = d0;
      for (int b = 0; b < 3; b++) {
        const __m256i* wb = w + b * 16;
        const int8_t* ab = act_s8 + b * 256;
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (int p = 0; p < 4; p++) {
          __m256i a0 = _mm256_load_si256((const __m256i*)(ab + 64 * p));
          __m256i a1 = _mm256_load_si256((const __m256i*)(ab + 64 * p + 32));
#define PSTEP(r)                                                     \
  {                                                                  \
    __m256i wv = _mm256_load_si256(wb + p * 4 + r);                  \
    __m256i lo = _mm256_and_si256(wv, m0F);                          \
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(wv, 4), m0F);    \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(lo, a0));     \
    s##r = _mm256_add_epi16(s##r, _mm256_maddubs_epi16(hi, a1));     \
  }
          PSTEP(0) PSTEP(1) PSTEP(2) PSTEP(3)
#undef PSTEP
        }
        d0 = _mm256_add_epi32(d0, _mm256_madd_epi16(s0, ones));
        d1 = _mm256_add_epi32(d1, _mm256_madd_epi16(s1, ones));
        d2 = _mm256_add_epi32(d2, _mm256_madd_epi16(s2, ones));
        d3 = _mm256_add_epi32(d3, _mm256_madd_epi16(s3, ones));
      }
      __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1),
                                    _mm256_hadd_epi32(d2, d3));
      __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h),
                                _mm256_extracti128_si256(h, 1));
      s = _mm_sub_epi32(s, corr);
      _mm_store_ps(out + g * 4, _mm_mul_ps(_mm_cvtepi32_ps(s), scl));
    }
  }
}

// ---------------- driver ----------------
static int8_t g_qa[768];
static uint8_t g_act_u8[768] __attribute__((aligned(64)));
static int8_t g_act_s8[768] __attribute__((aligned(64)));
static int g_sum_act192, g_sum_act768;
static float g_out[192] __attribute__((aligned(64)));
static volatile float g_fsink;

static void selfcheck(int IN, const uint8_t* up_arena, const uint8_t* pk_arena,
                      const int8_t* qw, const float* scales) {
  // scalar reference on mat 0
  float ref[192];
  for (int r = 0; r < 192; r++) {
    int d = 0;
    for (int i = 0; i < IN; i++) d += (int)g_qa[i] * qw[r * IN + i];
    ref[r] = (float)d * scales[r];
  }
  float out[192] __attribute__((aligned(64)));
  int sa = IN == 192 ? g_sum_act192 : g_sum_act768;
  auto chk = [&](const char* nm, void (*run)(float*)) {
    float o[192] __attribute__((aligned(64)));
    run(o);
    int bad = 0;
    for (int r = 0; r < 192; r++) if (o[r] != ref[r]) bad++;
    printf("# selfcheck IN=%d %s: %s\n", IN, nm, bad ? "FAIL" : "ok");
    if (bad) exit(1);
  };
  (void)out;
  static const uint8_t* s_up; static const uint8_t* s_pk; static int s_sa;
  s_up = up_arena; s_pk = pk_arena; s_sa = sa;
  if (IN == 192) {
    chk("unpacked", [](float* o) { gemv192_unpacked(s_up, 1, g_act_u8, o); });
    chk("packed", [](float* o) { gemv192_packed(s_pk, 1, g_act_s8, s_sa, o); });
    chk("unpacked_v2", [](float* o) { gemv192_unpacked_v2<false>(s_up, 1, g_act_u8, o); });
    chk("unpacked_v2pf", [](float* o) { gemv192_unpacked_v2<true>(s_up, 1, g_act_u8, o); });
    chk("packed_v2", [](float* o) { gemv192_packed_v2<false>(s_pk, 1, g_act_s8, s_sa, o); });
    chk("packed_v2pf", [](float* o) { gemv192_packed_v2<true>(s_pk, 1, g_act_s8, s_sa, o); });
  } else {
    chk("unpacked", [](float* o) { gemv768_unpacked<false>(s_up, 1, g_act_u8, o); });
    chk("unpacked_pf", [](float* o) { gemv768_unpacked<true>(s_up, 1, g_act_u8, o); });
    chk("packed", [](float* o) { gemv768_packed(s_pk, 1, g_act_s8, s_sa, o); });
  }
}

template <typename F>
static void bench_one(const char* name, F run_all_mats, size_t arena_bytes,
                      double macs_per_pass) {
  auto one = [&]() -> double {  // returns MACs/cycle of a timed run
    run_all_mats();  // warm
    uint64_t t0 = rdtsc_ser();
    run_all_mats();
    uint64_t one_t = rdtsc_ser() - t0;
    uint64_t P = (uint64_t)(0.06 * g_cal.tsc_hz / (double)one_t);
    if (P < 2) P = 2;
    t0 = rdtsc_ser();
    for (uint64_t i = 0; i < P; i++) run_all_mats();
    double cyc = ticks_to_cycles((double)(rdtsc_ser() - t0));
    g_fsink = g_out[0];
    return macs_per_pass * P / cyc;
  };
  Stats s = repeat_stat(one);
  double bpm = (double)arena_bytes / macs_per_pass;  // bytes per MAC
  printf("RESULT %-16s foot=%8.2fKB  %7.3f MAC/cyc (best %.3f worst %.3f)  eff %7.2f GB/s  %6.2f B/cyc\n",
         name, arena_bytes / 1024.0, s.med, s.mx, s.mn,
         s.med * bpm * g_cal.core_hz / 1e9, s.med * bpm);
  fflush(stdout);
}

int main() {
  pin_from_env();
  Calib c = calibrate();
  print_calib("gemv_start", c);

  for (int i = 0; i < 768; i++) {
    g_qa[i] = (int8_t)rnd(-128, 127);
    g_act_u8[i] = (uint8_t)(g_qa[i] + 128);
    g_act_s8[i] = g_qa[i];
  }
  g_sum_act192 = 0; g_sum_act768 = 0;
  for (int i = 0; i < 192; i++) g_sum_act192 += g_qa[i];
  for (int i = 0; i < 768; i++) g_sum_act768 += g_qa[i];

  std::vector<int8_t> qw0;
  std::vector<float> scl0;
  // footprint targets in bytes
  const size_t targets[] = {0 /*single mat*/, 256u << 10, 3u << 20, 6u << 20, 48u << 20};
  const char* tnames[] = {"1mat", "256K", "3M", "6M", "48M"};

  for (int IN : {192, 768}) {
    size_t umb = unpacked_matbytes(IN), pmb = packed_matbytes(IN);
    qw0.assign((size_t)192 * IN, 0);
    scl0.assign(192, 0.f);
    double macs_per_mat = 192.0 * IN;
    for (int ti = 0; ti < 5; ti++) {
      size_t tgt = targets[ti];
      if (IN == 768 && ti == 0) continue;  // single 768 mat isn't L1-resident anyway
      int nu = tgt ? std::max<size_t>(1, (tgt + umb / 2) / umb) : 1;
      int np = tgt ? std::max<size_t>(1, (tgt + pmb / 2) / pmb) : 1;
      uint8_t* ua = (uint8_t*)alloc_buf(nu * umb);
      uint8_t* pa = (uint8_t*)alloc_buf(np * pmb);
      rng_state = 0xC0FFEE + IN + ti;
      bool first_target = (ti == (IN == 768 ? 1 : 0));
      for (int m = 0; m < std::max(nu, np); m++) {
        // same weights land in both arenas while m < min(nu,np)
        fill_mat(m < nu ? ua + m * umb : nullptr, m < np ? pa + m * pmb : nullptr,
                 IN, qw0.data(), scl0.data());
        if (m == 0 && first_target)
          selfcheck(IN, ua, pa, qw0.data(), scl0.data());  // qw0/scl0 still hold mat 0
      }
      char nm[64];
      double um = macs_per_mat * nu, pm = macs_per_mat * np;
      size_t ub = nu * umb, pb = np * pmb;
      if (IN == 192) {
        snprintf(nm, sizeof nm, "u8_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv192_unpacked(ua, nu, g_act_u8, g_out); }, ub, um);
        snprintf(nm, sizeof nm, "u8v2_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv192_unpacked_v2<false>(ua, nu, g_act_u8, g_out); }, ub, um);
        snprintf(nm, sizeof nm, "u8v2pf_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv192_unpacked_v2<true>(ua, nu, g_act_u8, g_out); }, ub, um);
        snprintf(nm, sizeof nm, "p4_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv192_packed(pa, np, g_act_s8, g_sum_act192, g_out); }, pb, pm);
        snprintf(nm, sizeof nm, "p4v2_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv192_packed_v2<false>(pa, np, g_act_s8, g_sum_act192, g_out); }, pb, pm);
        snprintf(nm, sizeof nm, "p4v2pf_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv192_packed_v2<true>(pa, np, g_act_s8, g_sum_act192, g_out); }, pb, pm);
      } else {
        snprintf(nm, sizeof nm, "u8_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv768_unpacked<false>(ua, nu, g_act_u8, g_out); }, ub, um);
        snprintf(nm, sizeof nm, "u8pf_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv768_unpacked<true>(ua, nu, g_act_u8, g_out); }, ub, um);
        snprintf(nm, sizeof nm, "p4_%d_%s", IN, tnames[ti]);
        bench_one(nm, [&] { gemv768_packed(pa, np, g_act_s8, g_sum_act768, g_out); }, pb, pm);
      }
      munmap(ua, (nu * umb + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1));
      munmap(pa, (np * pmb + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1));
    }
    Calib cm = calibrate(true, 0.1);
    print_calib(IN == 192 ? "gemv_mid" : "gemv_end", cm);
  }
  return 0;
}
