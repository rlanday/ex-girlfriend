// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/weights_io_compressed.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md.
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

#include "weights_io.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace fx2 {

namespace {

[[noreturn]] void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "weights_io_compressed: ");
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
      die("unknown dtype code %u", unsigned(dtype));
  }
}

struct Reader {
  const uint8_t* p;
  size_t left;
  const char* path;

  void need(size_t n) {
    if (left < n) die("%s: truncated file (need %zu more bytes)", path, n);
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

// count limits per tree family (as CNT_LIMIT_* in weights_compress_v3.py)
enum : uint32_t {
  CNT_LIMIT_INT4 = 1023,
  CNT_LIMIT_NAME = 255,
  CNT_LIMIT_META = 255,
  CNT_LIMIT_PLANE01 = 1023,  // v4: byte planes 0 and 1 of ENC_PLANE4
  CNT_LIMIT_ZROW = 1023,     // v4: the zero-row flag of ENC_INT4
  CNT_LIMIT_TMPL = 255,      // v4: the three template flags of name/metadata
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

// the adaptive model set of the v2/v3 stream (pysrc/weights_compress.py
// _Models; _ModelsV3 in training/tf/4_export/weights_compress_v3.py).  raw,
// bf16 and plane are shift-5 trees in BOTH formats; meta, name and int4
// count in v3.  Only the set the format actually uses is allocated, so v3
// does not pay for v2's 33.5 MB name table; its name trees are built per
// context on first use (a real container touches 208 of the 65536 contexts).
struct Models {
  uint8_t meta_prev = 0;
  std::vector<uint16_t> raw;      // order-0 byte tree
  std::vector<uint16_t> bf16_hi;  // byte tree
  std::vector<uint16_t> bf16_lo;  // hi byte -> byte tree
  std::vector<uint16_t> plane;    // 4 byte trees (byte position mod 4)
  std::vector<uint16_t> meta;     // v2 order-1: prev byte -> byte tree
  std::vector<uint16_t> name;     // v2 order-2: (prev2, prev1) -> byte tree
  std::vector<uint16_t> int4;     // v2 4-bit tree, symbols 0..14
  std::vector<CountNode> cmeta;   // v3 order-1 counting trees
  std::vector<CountNode> cint4;   // v3 4-bit counting tree
  std::vector<std::vector<CountNode> > cname;  // v3 order-2, per context
  std::vector<CountNode> cplane;  // v4: counting trees of byte planes 0 and 1
  CountNode zrow;                 // v4: zero-row flag of ENC_INT4
  CountNode tname, tblock, tmeta;  // v4: template flags of name/metadata

  explicit Models(bool v3)
      : raw(256, 1024),
        bf16_hi(256, 1024),
        bf16_lo(256 * 256, 1024),
        plane(4 * 256, 1024) {
    if (v3) {
      cmeta.resize(256 * 256);
      cint4.resize(16);
      cname.resize(65536);
    } else {
      meta.assign(256 * 256, 1024);
      name.assign(size_t(65536) * 256, 1024);
      int4.assign(16, 1024);
    }
  }

  CountNode* cname_tree(uint32_t ctx) {
    std::vector<CountNode>& t = cname[ctx];
    if (t.empty()) t.resize(256);
    return t.data();
  }
};

int canon_name(const std::string& name, std::string* canon) {
  *canon = name;
  if (name.compare(0, 7, "blocks.") != 0) return -1;
  size_t j = 7;
  int n = 0;
  while (j < name.size() && j < 16 && name[j] >= '0' && name[j] <= '9')
    n = n * 10 + (name[j++] - '0');
  size_t k = j - 7;
  if (k < 1 || j >= name.size() || name[j] != '.' || (k > 1 && name[7] == '0'))
    return -1;
  canon->replace(7, k, 1, '#');
  return n;
}

// v4 is v3 plus counting trees on byte planes 0 and 1 of ENC_PLANE4, the
// zero-row flag of ENC_INT4 and the name/metadata template.
WeightsFile load_v2(Reader& r, const char* path, bool v3, bool v4) {
  uint32_t n_tensors = r.u32();
  BinDecoder dec(r.p, r.left);
  Models m(v3);
  if (v4) m.cplane.resize(2 * 256);

  auto get_meta = [&]() -> uint8_t {
    size_t ctx = size_t(m.meta_prev) * 256;
    uint8_t b = uint8_t(v3 ? dec.decode_ctree(&m.cmeta[ctx], 8, CNT_LIMIT_META)
                           : dec.decode_tree(&m.meta[ctx], 8));
    m.meta_prev = b;
    return b;
  };

  WeightsFile wf;
  wf.tensors.reserve(n_tensors);
  // v4 (c): template state, one entry per canonical name in order of
  // appearance: the successor it had last time (index, -1 = none) and its
  // metadata bytes at the last occurrence (hlen 0 = none yet).  Sized once,
  // so the decoder needs no grow path.
  struct TmplEntry {
    std::string cn;
    int succ;
    uint8_t hlen;
    uint8_t hdr[35];  // dtype, ndim (<= 8), 4 bytes per dimension, encoding
  };
  std::vector<TmplEntry> tmpl(n_tensors);
  size_t n_cn = 0;
  int t_prev = -1, t_prev_blk = -1;
  for (uint32_t i = 0; i < n_tensors; i++) {
    std::string name, cn;
    int blk = -1;
    if (v4 && dec.decode_cbit(&m.tname, CNT_LIMIT_TMPL)) {
      if (t_prev < 0 || tmpl[t_prev].succ < 0)
        die("%s: tensor %u: bad v4 template", path, i);
      name = cn = tmpl[tmpl[t_prev].succ].cn;
      if (cn.compare(0, 9, "blocks.#.") == 0) {
        blk = dec.decode_cbit(&m.tblock, CNT_LIMIT_TMPL) ? t_prev_blk
                                                        : t_prev_blk + 1;
        char num[16];
        int len = std::snprintf(num, sizeof num, "%d", blk);
        name.replace(7, 1, num, size_t(len));
      }
    } else {
      uint32_t name_len = get_meta();
      name.assign(name_len, '\0');
      uint32_t c2 = 0, c1 = 0;
      for (uint32_t k = 0; k < name_len; k++) {
        uint32_t ctx = (c2 << 8) | c1;
        uint32_t ch = v3 ? dec.decode_ctree(m.cname_tree(ctx), 8, CNT_LIMIT_NAME)
                         : dec.decode_tree(&m.name[size_t(ctx) * 256], 8);
        name[k] = char(ch);
        c2 = c1;
        c1 = ch;
      }
      blk = canon_name(name, &cn);
    }
    // metadata: dtype, ndim, 4 bytes per dimension, encoding
    uint8_t hdr[35];
    size_t hlen;
    size_t ci = 0;
    if (v4) {
      while (ci < n_cn && tmpl[ci].cn != cn) ci++;
      if (ci == n_cn) {
        tmpl[n_cn].cn = cn;
        tmpl[n_cn++].succ = -1;
      }
    }
    if (v4 && dec.decode_cbit(&m.tmeta, CNT_LIMIT_TMPL)) {
      hlen = tmpl[ci].hlen;
      if (!hlen) die("%s: tensor %u: bad v4 template", path, i);
      std::memcpy(hdr, tmpl[ci].hdr, hlen);
    } else {
      hdr[0] = get_meta();
      hdr[1] = get_meta();
      if (hdr[1] > 8)
        die("%s: %s: absurd ndim %u", path, name.c_str(), unsigned(hdr[1]));
      hlen = 3 + 4 * size_t(hdr[1]);
      for (size_t k = 2; k < hlen; k++) hdr[k] = get_meta();
    }
    if (v4) {
      if (t_prev >= 0) tmpl[t_prev].succ = int(ci);
      tmpl[ci].hlen = uint8_t(hlen);
      std::memcpy(tmpl[ci].hdr, hdr, hlen);
      t_prev = int(ci);
      if (blk >= 0) t_prev_blk = blk;
    }

    WTensor t;
    t.dtype = hdr[0];
    dtype_size(t.dtype);  // validates the code
    uint32_t ndim = hdr[1];
    t.shape.resize(ndim);
    size_t numel = 1;
    for (uint32_t d = 0; d < ndim; d++) {
      uint32_t v = 0;
      for (int k = 0; k < 4; k++) v |= uint32_t(hdr[2 + 4 * d + k]) << (8 * k);
      t.shape[d] = v;
      numel *= v;
    }
    t.numel = numel;
    size_t bytes = numel * dtype_size(t.dtype);
    t.data.resize(bytes);
    uint8_t encoding = hdr[hlen - 1];

    switch (encoding) {
      case ENC_INT4: {
        if (t.dtype != DT_I8) die("%s: %s: ENC_INT4 on dtype %u", path,
                                  name.c_str(), unsigned(t.dtype));
        int8_t* out = reinterpret_cast<int8_t*>(t.data.data());
        if (v4) {
          // one flag per row; flag 1 = the row is all zero and not coded
          size_t rows = ndim > 1 ? t.shape[0] : 1;
          size_t cols = rows ? numel / rows : 0;
          for (size_t r0 = 0; r0 < rows; r0++) {
            int8_t* row = out + r0 * cols;
            if (dec.decode_cbit(&m.zrow, CNT_LIMIT_ZROW)) {
              std::memset(row, 0, cols);
              continue;
            }
            for (size_t k = 0; k < cols; k++)
              row[k] = int8_t(
                  int(dec.decode_ctree(m.cint4.data(), 4, CNT_LIMIT_INT4)) - 7);
          }
          break;
        }
        if (v3) {
          for (size_t k = 0; k < numel; k++)
            out[k] = int8_t(
                int(dec.decode_ctree(m.cint4.data(), 4, CNT_LIMIT_INT4)) - 7);
        } else {
          for (size_t k = 0; k < numel; k++)
            out[k] = int8_t(int(dec.decode_tree(m.int4.data(), 4)) - 7);
        }
        break;
      }
      case ENC_BF16: {
        if (t.dtype != DT_BF16) die("%s: %s: ENC_BF16 on dtype %u", path,
                                    name.c_str(), unsigned(t.dtype));
        uint16_t* out = reinterpret_cast<uint16_t*>(t.data.data());
        for (size_t k = 0; k < numel; k++) {
          uint32_t hi = dec.decode_tree(m.bf16_hi.data(), 8);
          uint32_t lo = dec.decode_tree(&m.bf16_lo[hi * 256], 8);
          out[k] = uint16_t((hi << 8) | lo);
        }
        break;
      }
      case ENC_PLANE4: {
        if (dtype_size(t.dtype) != 4)
          die("%s: %s: ENC_PLANE4 on dtype %u", path, name.c_str(),
              unsigned(t.dtype));
        for (size_t k = 0; k < bytes; k++)
          t.data[k] = uint8_t(
              v4 && (k & 3) < 2
                  ? dec.decode_ctree(&m.cplane[(k & 3) * 256], 8,
                                     CNT_LIMIT_PLANE01)
                  : dec.decode_tree(&m.plane[(k & 3) * 256], 8));
        break;
      }
      case ENC_ROPE_SIN:
      case ENC_ROPE_COS: {
        if (t.dtype != DT_F32 || ndim != 2)
          die("%s: %s: bad rope tensor", path, name.c_str());
        if (!wf.has("rope.inv_freq"))
          die("%s: %s: rope.inv_freq not decoded yet", path, name.c_str());
        const WTensor& inv = wf.get("rope.inv_freq");
        if (inv.dtype != DT_F32 || inv.numel != t.shape[1])
          die("%s: %s: rope.inv_freq mismatch", path, name.c_str());
        const float* invf = inv.f32();
        float* out = reinterpret_cast<float*>(t.data.data());
        int cos_bias = encoding == ENC_ROPE_COS ? 1 : 0;
        for (size_t pos = 0; pos < t.shape[0]; pos++)
          for (size_t col = 0; col < t.shape[1]; col++)
            out[pos * t.shape[1] + col] =
                sincosf_cuda(float(pos) * invf[col], cos_bias);
        break;
      }
      case ENC_RAW: {
        for (size_t k = 0; k < bytes; k++)
          t.data[k] = uint8_t(dec.decode_tree(m.raw.data(), 8));
        break;
      }
      default:
        die("%s: %s: unknown encoding %u", path, name.c_str(),
            unsigned(encoding));
    }

    if (!wf.tensors.emplace(name, std::move(t)).second)
      die("%s: duplicate tensor %s", path, name.c_str());
  }
  return wf;
}

}  // namespace

WeightsFile WeightsFile::load_compressed(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) die("cannot open %s", path);
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 12) die("%s: too small", path);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size())
    die("%s: short read", path);
  std::fclose(f);

  Reader r{buf.data(), buf.size(), path};
  char magic[8];
  r.read(magic, 8);
  if (std::memcmp(magic, "FX2TFWC4", 8) == 0)
    return load_v2(r, path, true, true);
  if (std::memcmp(magic, "FX2TFWC3", 8) == 0)
    return load_v2(r, path, true, false);
  if (std::memcmp(magic, "FX2TFWC2", 8) == 0)
    return load_v2(r, path, false, false);
  if (std::memcmp(magic, "FX2TFWC1", 8) != 0) die("%s: bad magic", path);

  uint32_t n_tensors = r.u32();
  WeightsFile wf;
  wf.tensors.reserve(n_tensors);
  for (uint32_t i = 0; i < n_tensors; i++) {
    uint32_t name_len = r.u32();
    if (name_len > 4096) die("%s: absurd name length %u", path, name_len);
    std::string name(name_len, '\0');
    r.read(&name[0], name_len);

    WTensor t;
    t.dtype = r.u8();
    dtype_size(t.dtype);  // validates the code
    uint32_t ndim = r.u32();
    if (ndim > 8) die("%s: %s: absurd ndim %u", path, name.c_str(), ndim);
    t.shape.resize(ndim);
    size_t numel = 1;
    for (uint32_t d = 0; d < ndim; d++) {
      t.shape[d] = r.u32();
      numel *= t.shape[d];
    }
    t.numel = numel;
    size_t bytes = numel * dtype_size(t.dtype);
    t.data.resize(bytes);
    if (t.dtype == DT_I8) {
      uint32_t stream_len = r.u32();
      r.need(stream_len);
      decode_i8_uniform15(r.p, stream_len,
                          reinterpret_cast<int8_t*>(t.data.data()), numel);
      r.p += stream_len;
      r.left -= stream_len;
    } else {
      r.read(t.data.data(), bytes);
    }

    if (!wf.tensors.emplace(name, std::move(t)).second)
      die("%s: duplicate tensor %s", path, name.c_str());
  }
  if (r.left != 0)
    die("%s: %zu trailing bytes after last tensor", path, r.left);
  return wf;
}

}  // namespace fx2
