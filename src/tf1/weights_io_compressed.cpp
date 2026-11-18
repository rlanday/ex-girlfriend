// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/weights_io_compressed.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0.
// weights_io_compressed: decoder for the losslessly compressed weights files
// written by pysrc/weights_compress.py (must stay in exact sync with it).
//
// format v1, magic FX2TFWC1: same container as FX2TFW01 but every DT_I8
// payload (quantized weights, 15 possible values in [-7, 7]) is range-coded
// with a uniform 1/15 model, i.e. log2(15) = 3.907 bits per weight; all other
// payloads and the metadata are raw.
//
// format v2, magic FX2TFWC2: one range-coded stream with adaptive binary
// models (LZMA-style 11-bit probabilities); rope.sin/rope.cos are not stored
// but recomputed with a bit-exact host port of CUDA libdevice
// __nv_sinf/__nv_cosf; DT_I8 uses an adaptive 15-symbol tree, DT_BF16 hi/lo
// byte models, DT_F32/DT_I32 per-byte-plane models, names an order-2
// character model and metadata an order-1 byte model.
//
// format v3, magic FX2TFWC3 (added for this entry): the same container, range
// coder and bit trees as v2.  The only difference is the update rule of the
// node probabilities on the three stationary trees (int4 weights, names,
// metadata), which count instead of decaying (see CountNode below).  bf16,
// f32/i32 byte planes and the raw fallback keep the v2 shift-5 rule, because
// those planes are not stationary and counting would cost bytes there.  About
// 21 KB smaller per container on a 4M-parameter model, same decode time.
//
// format v4, magic FX2TFWC4 (added for this entry):
// v3 plus three changes, everything else as in v3.
//  (a) ENC_PLANE4: the f32 tensors of a bf16 export are rounded to bf16, so
//      the two low byte planes (byte 0 and 1 of every value) are zero almost
//      everywhere; v4 codes those two planes with counting trees (limit 1023)
//      instead of shift-5, whose adaptation floor costs about 0.18 bit per
//      zero byte.  About 1.5 KB on the 8M container.
//  (b) ENC_INT4: one counting flag bit (limit 1023) in front of every row
//      (row = index of dimension 0; a tensor with ndim < 2 is one row).  Flag
//      1 means the whole row is zero and is not coded.  The writer sets it for
//      every all-zero row; the rows of collapsed attention channels are zeroed
//      by the export (zero_collapsed_rows in weights_compress_v3.py), about
//      5.5 KB on the 8M container.
//  (c) name and metadata from a template.  The canonical name replaces the
//      layer number of "blocks.<n>." by '#'.  Flag 1 (counting node): the
//      canonical name is the one that followed the previous tensor's
//      canonical name last time; for a block name a second flag says "same
//      layer as the previous block tensor" (else the next layer).  Flag 0:
//      the name is coded as in v3.  A third flag says "dtype, shape and
//      encoding as at the last occurrence of this canonical name"; flag 0
//      codes them as in v3.  About 3.3 KB on the 8M container.
// Bit-identical tensors to the v3 container of the same tensor set.
//
// format v5, magic FX2TFWC5: v4 plus three encodings (same tensors, only the
// coder models differ; written by convert_container_v5.py):
//  ENC_INT4W  int4 values as in ENC_INT4 of v4 (zero-row flag), but through
//             one of 24 counting trees (limit 4095) chosen by a causal width
//             context: the bucket of (row mean |q| so far) x (column mean |q|
//             in the rows above) / (tensor mean |q| so far), integer
//             arithmetic, same formula as _width_bucket in the writer.
//  ENC_F32HI  f32 tensors whose low 16 bits are zero (the bf16-rounded raw
//             tensors): the high 16 bits with hi/lo shift-5 trees of their own.
//  ENC_BF16C  row scales: hi/lo shift-5 trees of their own per canonical name.
// About 5.5 KB on the 8M container.
//
// v5 with the K extension (release configuration): the reader also knows
//  ENC_F32HC  f32 tensors as ENC_F32HI, but the high byte through counting
//             trees (limit 255) per canonical name and column (index within
//             the row, row = index of dimension 0; columns 4.. share one
//             tree); the low byte through the shared ENC_F32HI low trees.
// Written by weights_compress_v3.py and, with option k, by the NP/NP2 writers
// (tf1_np_codec.cpp, np2_container.py); the magic stays FX2TFWC5, a v5 reader
// without this encoding stops at the encoding byte.  About 1.9 KB on the 8M container.
//
// format v6, magic FX2TFWC6: v5 plus one encoding for weights trained with a
// row alphabet (no container of this release uses it, and its writer is not included):
//  ENC_INT4Q  int4 values as ENC_INT4W, plus per coded row its tightest
//             alphabet (15, 7 or 3 levels; two counting nodes in the context of
//             the previous row's alphabet); 7- and 3-level rows through width
//             trees of 3 and 2 bits; the width statistics count |q| in 15-level
//             units (Q_UNITS in weights_compress_v3.py).  A 1-level row is all
//             zero and goes through the zero-row flag of v4.
//
// format NP, magic FX2TFWN5/FX2TFWN6: v5/v6 plus a carrier of free bits (the
// writer can code the f32 tensors as ENC_F32HC instead of ENC_F32HI).  The
// order of the hidden units of a layer is free: the engine computes them with
// integer dots and per-unit quantisation, so any order gives bit-identical
// results.  The rows of a key tensor are compared bytewise; the order in which
// they are stored is a permutation of the sorted order, written in the
// factorial number system of Laisant (1888), known as the Lehmer code: digit k
// of a group of n rows is the number of later rows that compare smaller,
// radix n - k.  The digits of all groups form one mixed-radix number, read in
// file order (Horner: x = x * radix + digit).  Its K low bytes are the start
// of the second coder stream, so the container is K bytes shorter: the free
// choice of the order carries data, the "bits back" argument of Hinton and
// van Camp (1993).  A build that reads only its stamped container reads it as
// format NP; a build that compares the magic reads v2 to v6 as well.  Layout:
//   magic, u32 n_tensors, u32 nA, u32 lenA, u32 K,
//   stream A (lenA bytes, its zero cache byte not stored): tensors 0..nA-1,
//     the tensors up to the last one a group touches,
//   stream B without its zero cache byte and its first K bytes: tensors nA..
// Stream B continues all models of A except those the order chosen in A can
// reach (the int4 trees and the per-name trees of the row scales), which start
// fresh; so its bytes do not depend on that order.  The encoding
// byte of a key tensor carries 0x40 (one group over all rows) or 0x20 (groups
// of 64 rows).  The groups: the MLP units of the layers without fast weights,
// the 64 units of the forget and output gate bottlenecks of the KDA layers
// and the value dims of each head of the plain attention layers.
//
// format NP2, magic FX2TFWP5/FX2TFWP6: format NP plus groups that move f32
// rows, the key dims (per head) and the value dims (per layer, one order for
// all heads) of the KDA layers; the engine sums over these dims elementwise or
// per lane, so another order changes results only at the rounding level.
// Three differences to NP: stream B starts the f32 models fresh as well (the
// groups move rows of the short convolutions, dt_bias and the gated output
// norm); key rows are compared over their bytes (an f32 key tensor has four
// per element); key flag 0x60 is one group over the first 64 rows of the key
// tensor (the value dims of a KDA layer: the order of block 0 is that of all
// three head blocks).  A container of format NP reads as before.  A build
// that reads only its stamped container reads it as NP2.

#include "../meld.h"
#include "weights_io.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace fx2 {

namespace {

[[noreturn]] void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "\n");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
  std::exit(1);
}

size_t dtype_size(uint8_t dtype) {
  switch (dtype) {
    case DT_I8:
      return 1;
    case DT_BF16:
      return 2;
    case DT_F32:
      return 4;
    case DT_I32:
      return 4;
    default:
      msg_stop(234);
  }
}

struct Reader {
  const uint8_t* p;
  size_t left;
  const char* path;

  void need(size_t n) {
    if (left < n) msg_stop(235);
  }
  void read(void* dst, size_t n) {
    need(n);
    std::memcpy(dst, p, n);
    p += n;
    left -= n;
  }
  uint32_t u32() {
    uint32_t v;
    read(&v, 4);
    return v;
  }
  uint8_t u8() {
    uint8_t v;
    read(&v, 1);
    return v;
  }
};

// byte-wise range decoder matching pysrc/weights_compress.py RangeDecoder
// (LZMA-style: 32-bit range, renormalization below 2^24)
struct RangeDecoder {
  const uint8_t* p;
  const uint8_t* end;
  uint32_t range = 0xFFFFFFFFu;
  uint32_t code = 0;

  RangeDecoder(const uint8_t* data, size_t len) : p(data), end(data + len) {
    p++;  // the first byte is the encoder's initial zero cache
    for (int i = 0; i < 4; i++) code = (code << 8) | byte();
  }
  uint8_t byte() { return p < end ? *p++ : 0; }
  void normalize() {
    while (range < (1u << 24)) {
      code = (code << 8) | byte();
      range <<= 8;
    }
  }
  uint32_t decode_uniform(uint32_t tot) {
    uint32_t r = range / tot;
    uint32_t s = code / r;
    if (s > tot - 1) s = tot - 1;
    code -= r * s;
    range = r;
    normalize();
    return s;
  }
};

void decode_i8_uniform15(const uint8_t* stream, size_t stream_len, int8_t* out,
                         size_t count) {
  RangeDecoder rd(stream, stream_len);
  for (size_t i = 0; i < count; i++)
    out[i] = int8_t(int(rd.decode_uniform(15)) - 7);
}

// ---------------------------------------------------------------------------
// format v2
// ---------------------------------------------------------------------------

// payload encodings (pysrc/weights_compress.py)
enum : uint8_t {
  ENC_RAW = 0,
  ENC_INT4 = 1,
  ENC_BF16 = 2,
  ENC_PLANE4 = 3,
  ENC_ROPE_SIN = 4,
  ENC_ROPE_COS = 5,
  ENC_INT4W = 6,  // v5: int4 with the width context
  ENC_F32HI = 7,  // v5: f32 with zero low 16 bits, as bf16
  ENC_BF16C = 8,  // v5: row scale, trees per canonical name
  ENC_F32HC = 10,  // v5 + K: f32 high byte per canonical name and column
};

// --- bit-exact host port of CUDA libdevice __nv_sinf/__nv_cosf --------------
// transcribed from the __nv_sinf/__nv_cosf LLVM IR of CUDA 13.0's
// libdevice.10.bc; verified bit-identical to the rope tables computed by
// torch.sin/cos on CUDA over all 8388608 table entries (incl. 25457
// Payne-Hanek slowpath arguments).  cos(a) is sin's body with quadrant + 1.
// Algorithm, constants and coefficients are NVIDIA's (CUDA libdevice); the
// transcription is Vladimir Ivanov's, taken over with his engine.

inline float fbits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

const uint32_t kI2OverPi[6] = {0x3C439041u, 0xDB629599u, 0xF534DDC0u,
                               0xFC2757D1u, 0x4E441529u, 0xA2F9836Eu};

// __internal_trig_reduction_slowpath: Payne-Hanek for |a| >= 105615
float trig_slowpath(float a, int* quadrant) {
  uint32_t ia;
  std::memcpy(&ia, &a, 4);
  uint32_t sign = ia & 0x80000000u;
  int32_t e = (int32_t)((ia >> 23) & 0xffu) - 128;
  ia = (ia << 8) | 0x80000000u;

  uint32_t result[7];
  uint32_t hi = 0;
  for (int k = 0; k < 6; k++) {
    uint64_t p = (uint64_t)kI2OverPi[k] * ia + hi;
    result[k] = (uint32_t)p;
    hi = (uint32_t)(p >> 32);
  }
  result[6] = hi;

  int idx = 4 - ((uint32_t)e >> 5);  // e >= 16 on this path
  int sh = e & 31;
  uint32_t rhi = result[idx + 2], rlo = result[idx + 1];
  if (sh) {
    rhi = (result[idx + 2] << sh) + (result[idx + 1] >> (32 - sh));
    rlo = (result[idx + 1] << sh) + (result[idx] >> (32 - sh));
  }
  uint32_t q = rhi >> 30;
  uint32_t nhi = (rhi << 2) + (rlo >> 30);
  uint32_t nlo = rlo << 2;
  uint32_t top = nhi >> 31;
  q += top;
  int32_t qi = (int32_t)q;
  if (sign) qi = -qi;
  uint32_t s2 = sign;
  if (top) {
    nhi = ~nhi;
    nlo = ~nlo;
    s2 = sign ^ 0x80000000u;
  }
  *quadrant = qi;
  int64_t prod = (int64_t)(((uint64_t)nhi << 32) | nlo);
  double dscale;
  uint64_t dbits = 0x3BF921FB54442D19ull;  // pi/2 * 2^-64
  std::memcpy(&dscale, &dbits, 8);
  float r = (float)((double)prod * dscale);
  if (s2) r = -r;
  return r;
}

// __nv_sinf(a) for cos_bias 0, __nv_cosf(a) for cos_bias 1
float sincosf_cuda(float a, int cos_bias) {
  int i = (int)lrintf(a * fbits(0x3F22F983u));  // __float2int_rn(a * 2/pi)
  float j = (float)i;
  float t = fmaf(j, fbits(0xBFC90FDAu), a);
  t = fmaf(j, fbits(0xB3A22168u), t);
  t = fmaf(j, fbits(0xA7C234C5u), t);
  if (fabsf(a) >= 105615.0f) {
    if (std::isinf(a)) {
      t = a * 0.0f;
      i = 0;
    } else {
      t = trig_slowpath(a, &i);
    }
  }
  i += cos_bias;
  float x2 = t * t;
  float base = (i & 1) ? 1.0f : t;
  float p = fmaf(x2, base, 0.0f);
  float c = (i & 1) ? fmaf(fbits(0x37CBAC00u), x2, fbits(0xBAB607EDu))
                    : fbits(0xB94D4153u);
  c = fmaf(c, x2, (i & 1) ? fbits(0x3D2AAABBu) : fbits(0x3C0885E4u));
  c = fmaf(c, x2, (i & 1) ? fbits(0xBEFFFFFFu) : fbits(0xBE2AAAA8u));
  float z = fmaf(c, p, base);
  if (i & 2) z = fmaf(z, -1.0f, 0.0f);
  return z;
}

// --- adaptive binary range decoder (LZMA-style, 11-bit probs, shift-5) ------
struct CountNode {
  uint16_t n0 = 1, n1 = 1;
};
constexpr bool g_np2 = true;
enum : uint32_t {
  CNT_LIMIT_INT4 = 1023,
  CNT_LIMIT_NAME = 255,
  CNT_LIMIT_META = 255,
  CNT_LIMIT_PLANE01 = 1023,  // v4: byte planes 0 and 1 of ENC_PLANE4
  CNT_LIMIT_ZROW = 1023,     // v4: the zero-row flag of ENC_INT4
  CNT_LIMIT_TMPL = 255,      // v4: the three template flags of name/metadata
  CNT_LIMIT_INT4W = 4095,    // v5: the 24 width-context int4 trees
  CNT_LIMIT_F32HC = 255,     // v5 + K: the high-byte trees of ENC_F32HC
};

struct BinDecoder {
  const uint8_t* p;
  const uint8_t* end;
  uint32_t range = 0xFFFFFFFFu;
  uint32_t code = 0;

  BinDecoder(const uint8_t* data, size_t len) : p(data), end(data + len) {
    p++;  // the first byte is the encoder's initial zero cache
    for (int i = 0; i < 4; i++) code = (code << 8) | byte();
  }
  uint8_t byte() { return p < end ? *p++ : 0; }
  int decode_bit(uint16_t* prob) {
    uint32_t bound = (range >> 11) * *prob;
    int bit;
    if (code < bound) {
      range = bound;
      *prob = uint16_t(*prob + ((2048 - *prob) >> 5));
      bit = 0;
    } else {
      code -= bound;
      range -= bound;
      *prob = uint16_t(*prob - (*prob >> 5));
      bit = 1;
    }
    while (range < (1u << 24)) {
      code = (code << 8) | byte();
      range <<= 8;
    }
    return bit;
  }
  // probs: (1 << nbits) entries, indices 1.. used
  uint32_t decode_tree(uint16_t* probs, int nbits) {
    uint32_t node = 1;
    for (int k = 0; k < nbits; k++) node = (node << 1) | decode_bit(&probs[node]);
    return node - (1u << nbits);
  }

  int decode_cbit(CountNode* n, uint32_t limit) {
    uint32_t p = 2048u * n->n0 / (uint32_t(n->n0) + n->n1);
    if (p < 1) p = 1;
    else if (p > 2047) p = 2047;
    uint32_t bound = (range >> 11) * p;
    int bit;
    if (code < bound) {
      range = bound;
      bit = 0;
    } else {
      code -= bound;
      range -= bound;
      bit = 1;
    }
    while (range < (1u << 24)) {
      code = (code << 8) | byte();
      range <<= 8;
    }
    if (bit) n->n1++; else n->n0++;
    if (uint32_t(n->n0) + n->n1 > limit) {
      n->n0 = uint16_t(n->n0 >> 1 ? n->n0 >> 1 : 1);
      n->n1 = uint16_t(n->n1 >> 1 ? n->n1 >> 1 : 1);
    }
    return bit;
  }
  // nodes: (1 << nbits) entries, indices 1.. used
  uint32_t decode_ctree(CountNode* nodes, int nbits, uint32_t limit) {
    uint32_t node = 1;
    for (int k = 0; k < nbits; k++)
      node = (node << 1) | uint32_t(decode_cbit(&nodes[node], limit));
    return node - (1u << nbits);
  }
};

constexpr int kNamenN = 76;

// fresh tree nodes: shift-5 probabilities 1024, counting nodes {1, 1}
void Frisch16(uint16_t* p, size_t n) {
  for (size_t i = 0; i < n; i++) p[i] = 1024;
}
void FrischCN(CountNode* p, size_t n) {
  for (size_t i = 0; i < n; i++) p[i] = CountNode();
}

// the models of load_v2 that this container reaches
struct ModelleId {
  uint8_t meta_prev;
  CountNode zrow, tname, tblock, tmeta;
  CountNode cmeta[256 * 256];   // order-1 metadata trees
  CountNode cplane[2 * 256];    // byte planes 0 and 1 of ENC_PLANE4
  uint16_t plane[4 * 256];      // byte planes 2 and 3 of ENC_PLANE4
  uint16_t bf16_hi[256];        // ENC_BF16
  uint16_t bf16_lo[256 * 256];
  CountNode cint4w[24 * 16];    // ENC_INT4W, 24 width contexts
  uint16_t f32hl[257 * 256];    // lo trees of ENC_F32HC
  CountNode* cname[65536];      // order-2 name trees, per context on first use
  uint16_t* bf16c[kNamenN];     // ENC_BF16C trees per canonical name, on first use
  CountNode* f32hc[kNamenN];    // ENC_F32HC hi trees per canonical name, on first use
};

// format NP, end of stream A (np_fresh_values of load_v2): the int4 trees,
// the zero-row flag, the per-name trees of the row scales and, for NP2, the
// f32 trees start fresh
__attribute__((noinline, cold)) void NpFrisch(ModelleId* m) {
  FrischCN(m->cint4w, 24 * 16);
  m->zrow = CountNode();
  Frisch16(m->f32hl, 257 * 256);
  for (int c = 0; c < kNamenN; c++) {
    if (m->bf16c[c]) Frisch16(m->bf16c[c], 257 * 256);
    if (m->f32hc[c]) FrischCN(m->f32hc[c], 5 * 256);
  }
}

size_t Elementbytes(uint8_t dtype) { return dtype == DT_I8 ? 1 : (dtype == DT_BF16 ? 2 : 4); }

// format NP2: Lehmer digits of a key tensor folded into the carried number x
// (np_digits of load_v2)
__attribute__((noinline, cold)) void NpZiffern(const WTensor& t, uint8_t key, uint32_t* x, size_t nx) {
  const size_t R = key == 0x60 ? 64 : t.shape[0],
               C = t.numel * Elementbytes(t.dtype) / t.shape[0], g = key & 0x20 ? 64 : R;
  const uint8_t* q = t.data;
  for (size_t b0 = 0; b0 < R; b0 += g)
    for (size_t k = 0; k < g; k++) {
      uint64_t c = 0;
      for (size_t j = k + 1; j < g; j++)
        c += std::memcmp(q + (b0 + j) * C, q + (b0 + k) * C, C) < 0;
      for (size_t i = 0; i < nx; i++) {
        c += uint64_t(x[i]) * (g - k);
        x[i] = uint32_t(c);
        c >>= 32;
      }
    }
}

WeightsFile LoadId(Reader& r, const char* path) {
  const uint32_t n_tensors = r.u32();
  const uint32_t nA = r.u32(), lenA = r.u32(), K = r.u32();
  // stream A starts after the header; its zero cache byte is not stored
  BinDecoder dec(r.p - 1, size_t(lenA) + 1);
  const size_t nx = K / 4 + 1;
  uint32_t* x = static_cast<uint32_t*>(std::calloc(nx, sizeof(uint32_t)));   // the carried number
  uint8_t* bufB = nullptr;   // stream B with its carried head
  ModelleId* m = static_cast<ModelleId*>(std::calloc(1, sizeof(ModelleId)));
  m->zrow = m->tname = m->tblock = m->tmeta = CountNode();
  FrischCN(m->cmeta, 256 * 256);
  FrischCN(m->cplane, 2 * 256);
  Frisch16(m->plane, 4 * 256);
  Frisch16(m->bf16_hi, 256);
  Frisch16(m->bf16_lo, 256 * 256);
  FrischCN(m->cint4w, 24 * 16);
  Frisch16(m->f32hl, 257 * 256);

  auto get_meta = [&]() -> uint8_t {
    uint8_t b = uint8_t(dec.decode_ctree(&m->cmeta[size_t(m->meta_prev) * 256], 8, CNT_LIMIT_META));
    m->meta_prev = b;
    return b;
  };

  WeightsFile wf;
  wf.t = static_cast<WTensor*>(std::calloc(size_t(kNamenN) * (16 + 1), sizeof(WTensor)));
  // template state per canonical name (v4 (c) of load_v2); the names as bytes
  struct Tmpl {
    const char* cn;
    uint8_t cnlen;
    uint8_t isblk;   // "blocks.#." name
    uint8_t hlen;
    uint8_t hdr[35];  // dtype, ndim (<= 8), 4 bytes per dimension, encoding
    int succ;
  };
  Tmpl tmpl[kNamenN];
  char pool[kNamenN * 256];
  size_t n_cn = 0, belegt = 0;
  unsigned long long fnv = 14695981039346656037ULL;   // over the canonical names, E268
  int t_prev = -1, t_prev_blk = -1;
  for (uint32_t i = 0; i < n_tensors; i++) {
    if (i == nA) {
      // end of stream A: the K low bytes of the carried number are the head of
      // stream B, in front of it its zero cache byte (skipped unread)
      const size_t nb = size_t(1) + K + (r.left - lenA);
      bufB = static_cast<uint8_t*>(std::malloc(nb));
      std::memcpy(bufB + 1, x, K);
      std::memcpy(bufB + 1 + K, r.p + lenA, r.left - lenA);
      dec = BinDecoder(bufB, nb);
      NpFrisch(m);
    }
    size_t ci;
    int blk = -1;
    if (dec.decode_cbit(&m->tname, CNT_LIMIT_TMPL)) {
      ci = size_t(tmpl[t_prev].succ);
      if (tmpl[ci].isblk)
        blk = dec.decode_cbit(&m->tblock, CNT_LIMIT_TMPL) ? t_prev_blk : t_prev_blk + 1;
    } else {
      char nm[256];
      size_t len = get_meta();
      uint32_t c2 = 0, c1 = 0;
      for (size_t k = 0; k < len; k++) {
        CountNode*& tree = m->cname[(c2 << 8) | c1];
        if (!tree) {
          tree = static_cast<CountNode*>(std::malloc(256 * sizeof(CountNode)));
          FrischCN(tree, 256);
        }
        const uint32_t ch = dec.decode_ctree(tree, 8, CNT_LIMIT_NAME);
        nm[k] = char(ch);
        c2 = c1;
        c1 = ch;
      }
      // canonical name in place: "blocks.<n>." -> "blocks.#.", blk = n
      // (canon_name of load_v2)
      if (len > 7 && std::memcmp(nm, "blocks.", 7) == 0) {
        size_t j = 7;
        int n = 0;
        while (j < len && j < 16 && nm[j] >= '0' && nm[j] <= '9') n = n * 10 + (nm[j++] - '0');
        const size_t k = j - 7;
        if (!(k < 1 || j >= len || nm[j] != '.' || (k > 1 && nm[7] == '0'))) {
          nm[7] = '#';
          std::memmove(nm + 8, nm + j, len - j);
          len = len - k + 1;
          blk = n;
        }
      }
      for (ci = 0; ci < n_cn; ci++)
        if (tmpl[ci].cnlen == len && std::memcmp(tmpl[ci].cn, nm, len) == 0) break;
      if (ci == n_cn) {
        if (n_cn == size_t(kNamenN))
          msg_stop(268);
        char* d = pool + belegt;
        std::memcpy(d, nm, len);
        belegt += len;
        for (size_t k = 0; k < len; k++) fnv = (fnv ^ static_cast<unsigned char>(nm[k])) * 1099511628211ULL;
        fnv *= 1099511628211ULL;   // the terminating zero
        tmpl[ci].cn = d;
        tmpl[ci].cnlen = uint8_t(len);
        tmpl[ci].isblk = blk >= 0;
        tmpl[ci].hlen = 0;
        tmpl[ci].succ = -1;
        n_cn++;
      }
    }
    // metadata: dtype, ndim, 4 bytes per dimension, encoding
    uint8_t hdr[35];
    size_t hlen;
    if (dec.decode_cbit(&m->tmeta, CNT_LIMIT_TMPL)) {
      hlen = tmpl[ci].hlen;
      std::memcpy(hdr, tmpl[ci].hdr, hlen);
    } else {
      hdr[0] = get_meta();
      hdr[1] = get_meta();
      hlen = 3 + 4 * size_t(hdr[1]);
      for (size_t k = 2; k < hlen; k++) hdr[k] = get_meta();
    }
    if (t_prev >= 0) tmpl[t_prev].succ = int(ci);
    tmpl[ci].hlen = uint8_t(hlen);
    std::memcpy(tmpl[ci].hdr, hdr, hlen);
    t_prev = int(ci);
    if (blk >= 0) t_prev_blk = blk;

    WTensor& t = wf.T(int(ci), blk);
    t.dtype = hdr[0];
    t.ndim = hdr[1];
    size_t numel = 1;
    for (uint32_t d = 0; d < t.ndim; d++) {
      uint32_t v = 0;
      for (int k = 0; k < 4; k++) v |= uint32_t(hdr[2 + 4 * d + k]) << (8 * k);
      t.shape[d] = v;
      numel *= v;
    }
    t.numel = numel;
    const size_t bytes = numel * Elementbytes(t.dtype);
    t.data = static_cast<uint8_t*>(std::calloc(bytes, 1));
    const uint8_t npkey = uint8_t(hdr[hlen - 1] & 0x60);  // key tensor of a group
    const uint8_t encoding = uint8_t(hdr[hlen - 1] & 0x1F);
    const size_t rows = t.ndim > 1 ? t.shape[0] : 1;
    const size_t cols = rows ? numel / rows : 0;

    switch (encoding) {
      case ENC_INT4W: {
        // one zero-row flag per row, then the values through the tree of the
        // width context (T, N over the coded rows, cs[k] column sums of |q|,
        // n coded rows so far); a zero row stays as calloc left it
        int8_t* out = reinterpret_cast<int8_t*>(t.data);
        uint64_t* cs = static_cast<uint64_t*>(std::calloc(cols, sizeof(uint64_t)));
        uint64_t n = 0, T = 0, N = 0;
        for (size_t r0 = 0; r0 < rows; r0++) {
          int8_t* row = out + r0 * cols;
          if (dec.decode_cbit(&m->zrow, CNT_LIMIT_ZROW)) continue;
          uint64_t rs = 0;
          for (size_t k = 0; k < cols; k++) {
            uint64_t M = (T * 65536 + 64 * 190054) / (N + 64);
            if (M < 1) M = 1;
            uint64_t rr = (rs * 65536 + 64 * M) / (k + 64);
            uint64_t cc = (cs[k] * 65536 + 64 * M) / (n + 64);
            uint64_t b = (8 * rr * cc) / (M * 65536);
            b = b < 8 ? 0 : (b > 8 + 23 ? 23 : b - 8);
            int v = int(dec.decode_ctree(&m->cint4w[b * 16], 4, CNT_LIMIT_INT4W)) - 7;
            row[k] = int8_t(v);
            uint64_t a = uint64_t(v < 0 ? -v : v);
            rs += a;
            cs[k] += a;
            T += a;
            N++;
          }
          n++;
        }
        std::free(cs);
        break;
      }
      case ENC_BF16C:
      case ENC_F32HC: {
        // bf16 values through a hi tree [0, 256) and lo trees [256 (1 + hi), ...);
        // ENC_F32HC: the hi byte through counting trees per canonical name and
        // column (index within the row, columns from 4 on share one tree), the
        // lo byte through the shared trees f32hl
        const bool f32 = encoding == ENC_F32HC;
        uint16_t* s = m->f32hl;
        CountNode* hc = nullptr;
        if (f32) {
          CountNode*& c = m->f32hc[ci];
          if (!c) {
            c = static_cast<CountNode*>(std::malloc(5 * 256 * sizeof(CountNode)));
            FrischCN(c, 5 * 256);
          }
          hc = c;
        } else {
          uint16_t*& b = m->bf16c[ci];
          if (!b) {
            b = static_cast<uint16_t*>(std::malloc(257 * 256 * sizeof(uint16_t)));
            Frisch16(b, 257 * 256);
          }
          s = b;
        }
        for (size_t k = 0, j = 0; k < numel; k++) {
          uint32_t hi = hc ? dec.decode_ctree(hc + (j < 4 ? j : 4) * 256, 8, CNT_LIMIT_F32HC)
                           : dec.decode_tree(s, 8);
          if (++j == cols) j = 0;
          uint32_t v = (hi << 8) | dec.decode_tree(&s[256 + hi * 256], 8);
          if (f32)
            reinterpret_cast<uint32_t*>(t.data)[k] = v << 16;
          else
            reinterpret_cast<uint16_t*>(t.data)[k] = uint16_t(v);
        }
        break;
      }
      case ENC_BF16: {
        uint16_t* out = reinterpret_cast<uint16_t*>(t.data);
        for (size_t k = 0; k < numel; k++) {
          uint32_t hi = dec.decode_tree(m->bf16_hi, 8);
          uint32_t lo = dec.decode_tree(&m->bf16_lo[hi * 256], 8);
          out[k] = uint16_t((hi << 8) | lo);
        }
        break;
      }
      case ENC_PLANE4: {
        for (size_t k = 0; k < bytes; k++)
          t.data[k] = uint8_t((k & 3) < 2
                                  ? dec.decode_ctree(&m->cplane[(k & 3) * 256], 8, CNT_LIMIT_PLANE01)
                                  : dec.decode_tree(&m->plane[(k & 3) * 256], 8));
        break;
      }
      case ENC_ROPE_SIN:
      case ENC_ROPE_COS: {
        const float* invf = wf.T(TID("rope.inv_freq")).f32();
        float* out = reinterpret_cast<float*>(t.data);
        const int cos_bias = encoding == ENC_ROPE_COS ? 1 : 0;
        for (size_t pos = 0; pos < t.shape[0]; pos++)
          for (size_t col = 0; col < t.shape[1]; col++)
            out[pos * t.shape[1] + col] = sincosf_cuda(float(pos) * invf[col], cos_bias);
        break;
      }
      default:
        __builtin_unreachable();   // the stamped container has no other encoding
    }
    if (npkey) NpZiffern(t, npkey, x, nx);
  }
  if (n_cn != size_t(kNamenN) || fnv != std::integral_constant<unsigned long long, namen::Fnv()>::value)
    msg_stop(268);
  for (size_t c = 0; c < 65536; c++) std::free(m->cname[c]);
  for (int c = 0; c < kNamenN; c++) {
    std::free(m->bf16c[c]);
    std::free(m->f32hc[c]);
  }
  std::free(m);
  std::free(bufB);
  std::free(x);
  return wf;
}

}  // namespace

WeightsFile::~WeightsFile() {
  if (!t) return;
  for (size_t i = 0; i < size_t(kNamenN) * (16 + 1); i++) std::free(t[i].data);
  std::free(t);
}

WeightsFile WeightsFile::load_compressed(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) msg_stop(206);
  std::fseek(f, 0, SEEK_END);
  const size_t size = size_t(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  uint8_t* buf = static_cast<uint8_t*>(std::malloc(size));
  const size_t n = std::fread(buf, 1, size, f);
  std::fclose(f);
  // The fingerprint of the container over the buffer the
  // reader decodes (it was a second read of the file in the engine)
  unsigned long long h = 14695981039346656037ULL;
  for (size_t i = 0; i < n; i++) h = (h ^ buf[i]) * 1099511628211ULL;
  if (h != 0xaec368ac6a568632ULL)
    msg_stop(207)
;
  Reader r{buf + 8, n - 8, path};   // behind the magic (fixed by the fingerprint)
  WeightsFile wf = LoadId(r, path);
  std::free(buf);
  return wf;
}

}  // namespace fx2
