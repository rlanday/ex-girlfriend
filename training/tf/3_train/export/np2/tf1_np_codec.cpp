// tf1_np_codec: writer of the transformer weight container in
// format NP (magic FX2TFWN5/FX2TFWN6) and NP2 (FX2TFWP5/FX2TFWP6;
// reader and layout: the head comment of
// src/tf1/weights_io_compressed.cpp).  A build-time tool of the container,
// like the converters convert_container_v4.py/v5.py/v5k.py of the training
// package; nothing of it is in the compressor.
//
// The order of the hidden units of the exact groups (MLP units of the layers
// without fast weights, the 64 units of the KDA gate bottlenecks, the value
// dims of each head of the plain attention layers) is free: the engine
// computes them with integer dots and per-unit quantisation.  The writer
// chooses that order so that its digits in the factorial number system
// (Laisant 1888, the Lehmer code) spell the first K bytes of the second coder
// stream, which are then not stored (the "bits back" argument of Hinton and
// van Camp 1993).
//
// Origin: the range encoder and the bit trees below are a C++ transcription of
// the encoder of pysrc/weights_compress.py of fx2-cmix-transformer by Vladimir
// Ivanov (GPL-3.0; its range coder follows the LZMA range coder of Igor
// Pavlov), with the models of formats v3 to v5 of this entry (training package
// tf/4_export/weights_compress_v3.py) and v6 (a converter of the development
// tree).  The reader side (models, trees, rope table, canonical names) is
// the reader of this tree, included below.
//
// With one K/V head (MQA, cmix-neif-2.0 form) the value dims of
// a plain attention layer are ONE group of 64 whose order also moves the three
// 64-column blocks of output_projection and the columns of the layer's
// value-embedding table.  Option np2 writes format NP2 (magic
// FX2TFWP5/FX2TFWP6) with the KDA key and value dims as further groups;
// option dead zeroes the parameters of dead KDA key dims (see main()).
//
// Build (from src/cmix-neifv9; all format switches of the reader on):
//   g++ -O2 -std=c++17 -DNEIF_TF1_WC4=1 -DNEIF_TF1_WC5=1 -DNEIF_TF1_WC5K=1 \
//       -DNEIF_TF1_QMIX=1 -DNEIF_TF1_NP2=1 tools_src/tf1_np_codec.cpp \
//       src/tf1/weights_io.cpp -o tf1_np_codec
//
// Usage:
//   tf1_np_codec np <in.c5|in.c6> <out> [k|nok] [fw] [np2] [dead|deadmark]
//       in: a v5 or v6 container (FX2TFWC5/FX2TFWC6, with or without
//       ENC_F32HC).  Steps, each one checked (as in convert_container_v5.py):
//         1. decode in with the reader of this tree;
//         2. identity probe: the plain v5/v6 writer of this tool reproduces
//            in bit for bit (with or without ENC_F32HC, whichever matches);
//         3. write the NP container (k: the bf16-rounded f32 tensors as
//            ENC_F32HC, read only by a build with NEIF_TF1_WC5K=1; nok: as
//            ENC_F32HI; default: as in the input);
//         4. read the NP container back with the reader of this tree: every
//            tensor equals the permuted one, the permutation is not trivial.
//       fw: also the MLP units of the four fast-weight layers.  The fast-weight
//       head reads the unit index (top-k ties, float sums in index order), so
//       the archive then follows the stored order: only for weights without a
//       reference archive.
//       np2: format NP2, also the key dims of each KDA head (key
//       query_convolution, with the q/k projection rows, the k convolution,
//       the forget-gate down rows, their row scales and dt_bias) and the value
//       dims of each KDA layer (one order for the three heads, key block 0 of
//       value_projection, with the v convolution, the output-gate down rows,
//       the output projection columns and the gated output norm).  The engine
//       sums over these dims elementwise or per lane: another order changes
//       its results at the rounding level, the archive follows the order;
//       read only by a build with NEIF_TF1_NP2=1.
//       dead: a KDA key dim whose q, k and forget-gate down rows are all zero
//       never enters the state (k = 0) nor the output (q = 0); its k and q
//       convolution rows and dt_bias are set to zero; with np2 the q
//       convolution is the key of its group, so the further dead dims of a
//       head get distinct marker rows (zero_dead_key_dims()); deadmark: the
//       markers also without np2 (engine probe).  Engine results stay
//       bit-identical; the container is no longer lossless against in.
//   tf1_np_codec ident <container>
//       step 1 and 2 only
//   tf1_np_codec plain <container> <out> [nok]
//       the decoded tensors of any container this reader reads (also NP and
//       NP2) as a plain v5/v6 container (ENC_F32HC; nok: ENC_F32HI), checked
//       by reading it back
//   tf1_np_codec dump <container> <out.raw>
//       the decoded tensors, raw in file order (FX2TFW01 layout)
#include "../src/tf1/weights_io_compressed.cpp"

#if !NEIF_TF1_WC5 || !NEIF_TF1_WC5K || !NEIF_TF1_QMIX || !NEIF_TF1_NP2 || NEIF_BL_TFLOAD
#error "tf1_np_codec: build with NEIF_TF1_WC4/WC5/WC5K/QMIX/NP2=1 and without NEIF_BL_TFLOAD (see the head comment)"
#endif

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace fx2 {
namespace {

// ------------------------------------------------------------------ tensors
struct Ent {
  std::string name;
  uint8_t dt;
  std::vector<uint32_t> shape;
  std::vector<uint8_t> data;
  size_t numel() const {
    size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  size_t rows() const { return shape.size() > 1 ? shape[0] : 1; }
  size_t cols() const { return rows() ? numel() / rows() : 0; }
};

// the tensors of a container, decoded by the reader of this tree, in file order
std::vector<Ent> load_ents(const char* path) {
  WeightsFile wf = WeightsFile::load_compressed(path);
  std::vector<Ent> out;
  for (const auto& e : wf.tensors.v) {
    Ent x;
    x.name = e.first;
    x.dt = e.second.dtype;
    x.shape.assign(e.second.shape.begin(), e.second.shape.end());
    x.data = e.second.data;
    out.push_back(std::move(x));
  }
  return out;
}

// ------------------------------------------------------------------ encoder
// the range encoder of weights_compress_v3.py (LZMA style, 11-bit nodes)
struct BinEncoder {
  uint64_t low = 0;
  uint32_t range = 0xFFFFFFFFu;
  uint8_t cache = 0;
  uint64_t cache_size = 1;
  std::vector<uint8_t> out;

  void shift_low() {
    if (low < 0xFF000000ull || low > 0xFFFFFFFFull) {
      uint8_t carry = uint8_t(low >> 32);
      out.push_back(uint8_t(cache + carry));
      for (; cache_size > 1; cache_size--) out.push_back(uint8_t(0xFF + carry));
      cache = uint8_t((low >> 24) & 0xFF);
      cache_size = 0;
    }
    cache_size++;
    low = (low << 8) & 0xFFFFFFFFull;
  }
  void norm() {
    while (range < (1u << 24)) {
      range <<= 8;
      shift_low();
    }
  }
  void encode_bit(uint16_t* prob, int bit) {
    uint32_t p = *prob;
    uint32_t bound = (range >> 11) * p;
    if (!bit) {
      range = bound;
      *prob = uint16_t(p + ((2048 - p) >> 5));
    } else {
      low += bound;
      range -= bound;
      *prob = uint16_t(p - (p >> 5));
    }
    norm();
  }
  void encode_tree(uint16_t* probs, int nbits, uint32_t sym) {
    uint32_t node = 1;
    for (int k = nbits - 1; k >= 0; k--) {
      int bit = (sym >> k) & 1;
      encode_bit(&probs[node], bit);
      node = (node << 1) | bit;
    }
  }
  void encode_cbit(CountNode* n, int bit, uint32_t limit) {
    uint32_t p = 2048u * n->n0 / (uint32_t(n->n0) + n->n1);
    if (p < 1) p = 1;
    else if (p > 2047) p = 2047;
    uint32_t bound = (range >> 11) * p;
    if (!bit) {
      range = bound;
    } else {
      low += bound;
      range -= bound;
    }
    norm();
    if (bit) n->n1++; else n->n0++;
    if (uint32_t(n->n0) + n->n1 > limit) {
      n->n0 = uint16_t(n->n0 >> 1 ? n->n0 >> 1 : 1);
      n->n1 = uint16_t(n->n1 >> 1 ? n->n1 >> 1 : 1);
    }
  }
  void encode_ctree(CountNode* nodes, int nbits, uint32_t sym, uint32_t limit) {
    uint32_t node = 1;
    for (int k = nbits - 1; k >= 0; k--) {
      int bit = (sym >> k) & 1;
      encode_cbit(&nodes[node], bit, limit);
      node = (node << 1) | bit;
    }
  }
  std::vector<uint8_t> finish() {
    for (int i = 0; i < 5; i++) shift_low();
    return out;
  }
};

bool ends_with(const std::string& s, const char* t) {
  size_t n = std::strlen(t);
  return s.size() >= n && s.compare(s.size() - n, n, t) == 0;
}

// encoding of a tensor: _pick_encoding + _pick_encoding_v5 of the python
// writer; ver 6 adds ENC_INT4Q (convert_container_v6.py), k ENC_F32HC
uint8_t pick_encoding(const Ent& e, const std::map<std::string, const Ent*>& seen, int ver,
                      bool k) {
  uint8_t enc = ENC_RAW;
  if (e.dt == DT_I8) {
    bool ok = true;
    for (uint8_t b : e.data) {
      int v = int8_t(b);
      if (v < -7 || v > 7) ok = false;
    }
    enc = ok ? ENC_INT4 : ENC_RAW;
  } else if (e.dt == DT_BF16) {
    enc = ENC_BF16;
  } else if (e.dt == DT_F32) {
    enc = ENC_PLANE4;
    if ((e.name == "rope.sin" || e.name == "rope.cos") && e.shape.size() == 2) {
      auto it = seen.find("rope.inv_freq");
      if (it != seen.end() && it->second->numel() == e.shape[1]) {
        const float* inv = reinterpret_cast<const float*>(it->second->data.data());
        int cb = e.name == "rope.cos" ? 1 : 0;
        bool eq = true;
        const float* d = reinterpret_cast<const float*>(e.data.data());
        for (size_t pos = 0; pos < e.shape[0] && eq; pos++)
          for (size_t c = 0; c < e.shape[1]; c++) {
            float v = sincosf_cuda(float(pos) * inv[c], cb);
            if (std::memcmp(&v, &d[pos * e.shape[1] + c], 4)) {
              eq = false;
              break;
            }
          }
        if (eq) enc = cb ? ENC_ROPE_COS : ENC_ROPE_SIN;
      }
    }
  } else if (e.dt == DT_I32) {
    enc = ENC_PLANE4;
  }
  if (enc == ENC_INT4) {
    if (ver >= 6) {
      size_t coarse = 0, R = e.rows(), C = e.cols();
      for (size_t r = 0; r < R; r++) {
        int m = 0;
        for (size_t c = 0; c < C; c++) m = std::max(m, std::abs(int(int8_t(e.data[r * C + c]))));
        if (m > 0 && m <= 3) coarse++;
      }
      if (coarse >= 16) return ENC_INT4Q;
    }
    return ENC_INT4W;
  }
  if (enc == ENC_PLANE4 && e.dt == DT_F32 && e.numel()) {
    bool low0 = true;
    const uint32_t* u = reinterpret_cast<const uint32_t*>(e.data.data());
    for (size_t i = 0; i < e.numel(); i++)
      if (u[i] & 0xFFFF) {
        low0 = false;
        break;
      }
    if (low0) return k ? ENC_F32HC : ENC_F32HI;
  }
  if (enc == ENC_BF16 && ends_with(e.name, ".weight.scale")) return ENC_BF16C;
  return enc;
}

// the value models of one stream (NP: stream A and stream B have one each)
struct ValModels {
  Models m{true};
  ValModels() {
    m.cplane.resize(2 * 256);
    m.cint4w.resize(24 * 16);
    m.f32hl.assign(257 * 256, 1024);
    m.cint4q7.resize(24 * 8);
    m.cint4q3.resize(24 * 4);
  }
  std::map<std::string, std::vector<uint16_t>> bf16c;  // per canonical name
  std::map<std::string, std::vector<CountNode>> f32hc;  // per canonical name, 5 hi trees
};

void encode_values(BinEncoder& E, ValModels& vm, const Ent& e, uint8_t enc,
                   const std::string& cn) {
  Models& m = vm.m;
  switch (enc) {
    case ENC_INT4W:
    case ENC_INT4Q: {
      static const uint8_t kU7[5] = {0, 2, 5, 7, 9};
      static const uint8_t kU3[3] = {0, 7, 14};
      size_t R = e.rows(), C = e.cols();
      std::vector<uint64_t> cs(C, 0);
      uint64_t n = 0, T = 0, N = 0;
      int prev = 0;
      for (size_t r = 0; r < R; r++) {
        const int8_t* row = reinterpret_cast<const int8_t*>(&e.data[r * C]);
        bool zero = true;
        for (size_t c = 0; c < C; c++)
          if (row[c]) {
            zero = false;
            break;
          }
        E.encode_cbit(&m.zrow, zero, CNT_LIMIT_ZROW);
        if (zero) continue;
        int alpha = 0;
        if (enc == ENC_INT4Q) {
          int mx = 0;
          for (size_t c = 0; c < C; c++) mx = std::max(mx, std::abs(int(row[c])));
          alpha = mx > 3 ? 0 : (mx > 1 ? 1 : 2);
          E.encode_cbit(&m.qa0[prev], alpha != 0, CNT_LIMIT_QALPHA);
          if (alpha) E.encode_cbit(&m.qa1[prev], alpha - 1, CNT_LIMIT_QALPHA);
          prev = alpha;
        }
        uint64_t rs = 0;
        for (size_t k = 0; k < C; k++) {
          uint64_t M = (T * 65536 + 64 * 190054) / (N + 64);
          if (M < 1) M = 1;
          uint64_t rr = (rs * 65536 + 64 * M) / (k + 64);
          uint64_t cc = (cs[k] * 65536 + 64 * M) / (n + 64);
          uint64_t b = (8 * rr * cc) / (M * 65536);
          b = b < 8 ? 0 : (b > 8 + 23 ? 23 : b - 8);
          int v = row[k];
          if (alpha == 0) E.encode_ctree(&m.cint4w[b * 16], 4, v + 7, CNT_LIMIT_INT4W);
          else if (alpha == 1) E.encode_ctree(&m.cint4q7[b * 8], 3, v + 3, CNT_LIMIT_INT4W);
          else E.encode_ctree(&m.cint4q3[b * 4], 2, v + 1, CNT_LIMIT_INT4W);
          uint64_t a = uint64_t(v < 0 ? -v : v);
          if (alpha == 1) a = kU7[a];
          else if (alpha == 2) a = kU3[a];
          rs += a;
          cs[k] += a;
          T += a;
          N++;
        }
        n++;
      }
      break;
    }
    case ENC_F32HI:
    case ENC_F32HC:
    case ENC_BF16C: {
      bool f32 = enc != ENC_BF16C;
      std::vector<uint16_t>* s = &m.f32hl;
      if (!f32) {
        auto& t = vm.bf16c[cn];
        if (t.empty()) t.assign(257 * 256, 1024);
        s = &t;
      }
      // ENC_F32HC: the hi byte through counting trees per canonical name and
      // column (columns from 4 on share one tree), the lo byte as ENC_F32HI
      CountNode* hc = nullptr;
      if (enc == ENC_F32HC) {
        auto& c = vm.f32hc[cn];
        if (c.empty()) c.resize(5 * 256);
        hc = c.data();
      }
      const size_t cols = e.cols();
      for (size_t i = 0, j = 0; i < e.numel(); i++) {
        uint32_t v;
        if (f32) {
          uint32_t u;
          std::memcpy(&u, &e.data[4 * i], 4);
          v = u >> 16;
        } else {
          uint16_t u;
          std::memcpy(&u, &e.data[2 * i], 2);
          v = u;
        }
        uint32_t hi = v >> 8;
        if (hc)
          E.encode_ctree(hc + (j < 4 ? j : 4) * 256, 8, hi, CNT_LIMIT_F32HC);
        else
          E.encode_tree(s->data(), 8, hi);
        if (++j == cols) j = 0;
        E.encode_tree(&(*s)[256 + hi * 256], 8, v & 0xFF);
      }
      break;
    }
    case ENC_BF16: {
      for (size_t i = 0; i < e.numel(); i++) {
        uint16_t u;
        std::memcpy(&u, &e.data[2 * i], 2);
        uint32_t hi = u >> 8;
        E.encode_tree(m.bf16_hi.data(), 8, hi);
        E.encode_tree(&m.bf16_lo[hi * 256], 8, u & 0xFF);
      }
      break;
    }
    case ENC_PLANE4: {
      for (size_t i = 0; i < e.data.size(); i++) {
        if ((i & 3) < 2) E.encode_ctree(&m.cplane[(i & 3) * 256], 8, e.data[i], CNT_LIMIT_PLANE01);
        else E.encode_tree(&m.plane[(i & 3) * 256], 8, e.data[i]);
      }
      break;
    }
    case ENC_ROPE_SIN:
    case ENC_ROPE_COS:
      break;  // recomputed by the reader
    case ENC_RAW:
      for (uint8_t b : e.data) E.encode_tree(m.raw.data(), 8, b);
      break;
    default:
      die("encoder: encoding %u", unsigned(enc));
  }
}

// name and metadata through the v4 template
struct HeadCoder {
  Models m{true};
  std::map<std::string, std::string> succ;
  std::map<std::string, std::vector<uint8_t>> meta;
  std::string prev;
  bool have_prev = false;
  int prev_blk = -1;

  void put_meta(BinEncoder& E, uint8_t b) {
    E.encode_ctree(&m.cmeta[size_t(m.meta_prev) * 256], 8, b, CNT_LIMIT_META);
    m.meta_prev = b;
  }
  void put_name(BinEncoder& E, const std::string& name) {
    put_meta(E, uint8_t(name.size()));
    uint32_t c2 = 0, c1 = 0;
    for (unsigned char ch : name) {
      E.encode_ctree(m.cname_tree((c2 << 8) | c1), 8, ch, CNT_LIMIT_NAME);
      c2 = c1;
      c1 = ch;
    }
  }
  // returns the canonical name
  std::string code(BinEncoder& E, const Ent& e, uint8_t enc_byte) {
    std::vector<uint8_t> mt;
    mt.push_back(e.dt);
    mt.push_back(uint8_t(e.shape.size()));
    for (auto d : e.shape)
      for (int k = 0; k < 4; k++) mt.push_back(uint8_t(d >> (8 * k)));
    mt.push_back(enc_byte);
    std::string cn;
    int blk = canon_name(e.name, &cn);
    bool hit;
    std::string pred;
    bool has_pred = false;
    if (have_prev) {
      auto it = succ.find(prev);
      if (it != succ.end()) {
        pred = it->second;
        has_pred = true;
      }
    }
    if (blk >= 0)
      hit = has_pred && pred == cn && (blk == prev_blk || blk == prev_blk + 1);
    else
      hit = has_pred && pred == cn && cn.compare(0, 9, "blocks.#.") != 0;
    E.encode_cbit(&m.tname, hit, CNT_LIMIT_TMPL);
    if (hit) {
      if (blk >= 0) E.encode_cbit(&m.tblock, blk == prev_blk, CNT_LIMIT_TMPL);
    } else {
      put_name(E, e.name);
    }
    auto mi = meta.find(cn);
    bool same = mi != meta.end() && mi->second == mt;
    E.encode_cbit(&m.tmeta, same, CNT_LIMIT_TMPL);
    if (!same)
      for (uint8_t b : mt) put_meta(E, b);
    if (have_prev) succ[prev] = cn;
    meta[cn] = mt;
    prev = cn;
    have_prev = true;
    if (blk >= 0) prev_blk = blk;
    return cn;
  }
};

std::vector<uint8_t> u32le(uint32_t v) {
  return {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
}

std::vector<uint8_t> encodings(const std::vector<Ent>& es, int ver, bool k) {
  std::vector<uint8_t> enc(es.size());
  std::map<std::string, const Ent*> seen;
  for (size_t i = 0; i < es.size(); i++) {
    enc[i] = pick_encoding(es[i], seen, ver, k);
    if (es[i].dt == DT_F32) seen[es[i].name] = &es[i];
  }
  return enc;
}

// the plain v5/v6 container (one stream)
std::vector<uint8_t> encode_plain(const std::vector<Ent>& es, int ver, bool k) {
  BinEncoder E;
  HeadCoder H;
  ValModels V;
  std::vector<uint8_t> enc = encodings(es, ver, k);
  for (size_t i = 0; i < es.size(); i++) {
    std::string cn = H.code(E, es[i], enc[i]);
    encode_values(E, V, es[i], enc[i], cn);
  }
  const char* magic = ver >= 6 ? "FX2TFWC6" : "FX2TFWC5";
  std::vector<uint8_t> out(magic, magic + 8);
  auto n = u32le(uint32_t(es.size()));
  out.insert(out.end(), n.begin(), n.end());
  auto s = E.finish();
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// ------------------------------------------------------------------ groups
// a group: the elements are the rows [r0, r0+n) of the key tensor; moving
// element k moves the row o+k of each (tensor, o) in rows_at, the element o+k
// of elem_at (row scales, 1-d tensors) and the column o+k of cols_at.  flag:
// the key flag of the encoding byte (0x40 all rows, 0x20 blocks of 64 rows,
// 0x60 the first 64 rows only, format NP2).
struct Group {
  std::string name;
  int key;
  uint8_t flag;
  int r0, n;
  std::vector<std::pair<int, int>> rows_at, elem_at, cols_at;
};

int find_t(const std::vector<Ent>& es, const std::string& n) {
  for (size_t i = 0; i < es.size(); i++)
    if (es[i].name == n) return int(i);
  die("tensor %s missing", n.c_str());
}

int find_opt(const std::vector<Ent>& es, const std::string& n) {
  for (size_t i = 0; i < es.size(); i++)
    if (es[i].name == n) return int(i);
  return -1;
}

// a group over rows [r0, r0+n) of key that moves rows, scales and columns at
// the same offset r0
Group group_at(const std::string& name, int key, int r0, int n, std::vector<int> rows,
               std::vector<int> elem, std::vector<int> cols, int key_rows) {
  Group G;
  G.name = name;
  G.key = key;
  G.flag = n == key_rows ? 0x40 : 0x20;
  G.r0 = r0;
  G.n = n;
  for (int t : rows) G.rows_at.push_back({t, r0});
  for (int t : elem) G.elem_at.push_back({t, r0});
  for (int t : cols) G.cols_at.push_back({t, r0});
  return G;
}

// the groups: exact ones (format NP); fw: also the MLP units of the last four
// (fast-weight) layers; kda: also the key and value dims of the KDA layers
// (format NP2)
std::vector<Group> np_groups(const std::vector<Ent>& es, bool with_fw, bool kda) {
  const Ent& ck = es[find_t(es, "config.kimi")];
  int NL = int(ck.numel());
  const int32_t* kimi = reinterpret_cast<const int32_t*>(ck.data.data());
  std::vector<Group> gs;
  int vi_count = 0;  // index of the plain attention layer (value_embeddings.<i>)
  for (int l = 0; l < NL; l++) {
    std::string b = "blocks." + std::to_string(l) + ".";
    std::string a = b + "attention.";
    if (kimi[l]) {
      for (const char* g : {"forget_gate_projection", "output_gate_projection"}) {
        std::string p = a + g;
        int key = find_t(es, p + ".up.weight.q");
        int n = int(es[key].rows());
        gs.push_back(group_at(p, key, 0, n, {key}, {find_t(es, p + ".up.weight.scale")},
                              {find_t(es, p + ".down.weight.q")}, n));
      }
      if (kda) {
        // key dims of head h: rows 64h.. of the q/k convolutions and
        // projections and of the forget-gate down projection, their scales
        // and dt_bias (the per-dim decay of the state rows)
        for (int h = 0; h < 3; h++) {
          std::vector<int> rows, elem;
          for (const char* t : {"query_convolution.weight", "key_convolution.weight",
                                "query_projection.weight.q", "key_projection.weight.q",
                                "forget_gate_projection.down.weight.q"})
            rows.push_back(find_t(es, a + t));
          for (const char* t : {"query_projection.weight.scale", "key_projection.weight.scale",
                                "forget_gate_projection.down.weight.scale", "dt_bias"})
            elem.push_back(find_t(es, a + t));
          int key = find_t(es, a + "query_convolution.weight");
          gs.push_back(group_at(a + "key_dims.h" + std::to_string(h), key, 64 * h, 64, rows, elem,
                                {}, int(es[key].rows())));
        }
        // value dims: one order for the three heads (the gated output norm
        // weight is shared by the heads); key block 0 of value_projection
        Group G;
        G.name = a + "value_dims";
        G.key = find_t(es, a + "value_projection.weight.q");
        G.flag = 0x60;
        G.r0 = 0;
        G.n = 64;
        for (int h = 0; h < 3; h++) {
          for (const char* t : {"value_convolution.weight", "value_projection.weight.q",
                                "output_gate_projection.down.weight.q"})
            G.rows_at.push_back({find_t(es, a + t), 64 * h});
          for (const char* t : {"value_projection.weight.scale",
                                "output_gate_projection.down.weight.scale"})
            G.elem_at.push_back({find_t(es, a + t), 64 * h});
          G.cols_at.push_back({find_t(es, a + "output_projection.weight.q"), 64 * h});
        }
        G.elem_at.push_back({find_t(es, a + "output_fused_norm_gate.weight"), 0});
        gs.push_back(G);
      }
    } else if (es[find_t(es, a + "value_projection.weight.q")].rows() == 64) {
      // one K/V head (MQA): its 64 value dims, moved in V, in the three head
      // blocks of the output projection and in the columns of the
      // value-embedding table of this layer
      int key = find_t(es, a + "value_projection.weight.q");
      Group G = group_at(a + "v_dims.mqa", key, 0, 64, {key},
                         {find_t(es, a + "value_projection.weight.scale")}, {}, 64);
      const int op = find_t(es, a + "output_projection.weight.q");
      G.cols_at = {{op, 0}, {op, 64}, {op, 128}};
      int ve = find_opt(es, "value_embeddings." + std::to_string(vi_count) + ".inner.weight.q");
      if (ve >= 0) G.cols_at.push_back({ve, 0});
      gs.push_back(G);
      vi_count++;
    } else {
      for (int h = 0; h < 3; h++) {
        int key = find_t(es, a + "value_projection.weight.q");
        gs.push_back(group_at(a + "v_dims.h" + std::to_string(h), key, 64 * h, 64, {key},
                              {find_t(es, a + "value_projection.weight.scale")},
                              {find_t(es, a + "output_projection.weight.q")}, int(es[key].rows())));
      }
    }
    if (l < NL - 4 || with_fw) {
      int key = find_t(es, b + "mlp.up.weight.q");
      int n = int(es[key].rows());
      gs.push_back(group_at(b + "mlp", key, 0, n, {key}, {find_t(es, b + "mlp.up.weight.scale")},
                            {find_t(es, b + "mlp.down.weight.q")}, n));
    }
  }
  // the order in which the reader meets the key tensors
  std::stable_sort(gs.begin(), gs.end(), [](const Group& x, const Group& y) {
    return x.key != y.key ? x.key < y.key : x.r0 < y.r0;
  });
  return gs;
}

// new element k = old element perm[k]
void apply_perm(std::vector<Ent>& es, const Group& G, const std::vector<int>& perm) {
  for (auto ra : G.rows_at) {
    Ent& e = es[ra.first];
    size_t C = e.cols() * dtype_size(e.dt);
    if (e.shape.size() < 2) die("group %s: rows of the 1-d tensor %s", G.name.c_str(), e.name.c_str());
    std::vector<uint8_t> old(e.data.begin() + size_t(ra.second) * C,
                             e.data.begin() + size_t(ra.second + G.n) * C);
    for (int k = 0; k < G.n; k++)
      std::memcpy(&e.data[size_t(ra.second + k) * C], &old[size_t(perm[k]) * C], C);
  }
  for (auto ea : G.elem_at) {
    Ent& e = es[ea.first];
    size_t s = dtype_size(e.dt);
    std::vector<uint8_t> old(e.data.begin() + size_t(ea.second) * s,
                             e.data.begin() + size_t(ea.second + G.n) * s);
    for (int k = 0; k < G.n; k++)
      std::memcpy(&e.data[size_t(ea.second + k) * s], &old[size_t(perm[k]) * s], s);
  }
  for (auto ca : G.cols_at) {
    Ent& e = es[ca.first];
    size_t R = e.rows(), C = e.cols(), s = dtype_size(e.dt);
    if (size_t(ca.second + G.n) > C) die("group %s: column block outside %s", G.name.c_str(), e.name.c_str());
    for (size_t r = 0; r < R; r++) {
      uint8_t* row = &e.data[r * C * s];
      std::vector<uint8_t> old(row + size_t(ca.second) * s, row + size_t(ca.second + G.n) * s);
      for (int k = 0; k < G.n; k++)
        std::memcpy(row + size_t(ca.second + k) * s, &old[size_t(perm[k]) * s], s);
    }
  }
}

// key rows are compared over their bytes (format NP2: f32 key tensors)
size_t key_bytes(const std::vector<Ent>& es, const Group& G) {
  const Ent& e = es[G.key];
  return e.cols() * dtype_size(e.dt);
}

const uint8_t* key_row(const std::vector<Ent>& es, const Group& G, int k) {
  return &es[G.key].data[size_t(G.r0 + k) * key_bytes(es, G)];
}

// option dead: zero the parameters of the dead KDA key dims (q, k and
// forget-gate down rows all zero).  With distinct (np2: the q convolution is
// the key of the key-dim groups) only the first dead dim of a head gets a
// zero q convolution row; the j-th further one (j = 0, 1, ...) gets the
// marker row with element j mod 4 = the f32 of high half 1 + j / 4 (a
// denormal) and zeros elsewhere: any finite value works there, since its
// input channel is zero.  Returns the number of dead dims; *qmark the marker
// rows.
int zero_dead_key_dims(std::vector<Ent>& es, bool distinct, int* qmark) {
  const Ent& ck = es[find_t(es, "config.kimi")];
  const int32_t* kimi = reinterpret_cast<const int32_t*>(ck.data.data());
  int dead = 0;
  *qmark = 0;
  for (int l = 0; l < int(ck.numel()); l++) {
    if (!kimi[l]) continue;
    std::string a = "blocks." + std::to_string(l) + ".attention.";
    const Ent& q = es[find_t(es, a + "query_projection.weight.q")];
    const Ent& k = es[find_t(es, a + "key_projection.weight.q")];
    const Ent& fd = es[find_t(es, a + "forget_gate_projection.down.weight.q")];
    Ent& qc = es[find_t(es, a + "query_convolution.weight")];
    Ent& kc = es[find_t(es, a + "key_convolution.weight")];
    Ent& dt = es[find_t(es, a + "dt_bias")];
    const size_t cb = qc.cols() * dtype_size(qc.dt), kb = kc.cols() * dtype_size(kc.dt);
    const size_t ds = dtype_size(dt.dt);
    if (qc.dt != DT_F32 || qc.cols() != 4) die("%squery_convolution.weight: f32 [*, 4] expected", a.c_str());
    int extra[3] = {-1, -1, -1};  // per head: further dead dims so far (-1: none dead yet)
    for (size_t r = 0; r < q.rows(); r++) {
      bool z = true;
      for (const Ent* e : {&q, &k, &fd})
        for (size_t c = 0; c < e->cols() && z; c++) z = e->data[r * e->cols() + c] == 0;
      if (!z) continue;
      dead++;
      std::memset(&kc.data[r * kb], 0, kb);
      std::memset(&dt.data[r * ds], 0, ds);
      std::memset(&qc.data[r * cb], 0, cb);
      int& j = extra[r / 64];
      if (distinct && j >= 0) {
        const uint32_t v = uint32_t(1 + j / 4) << 16;
        std::memcpy(&qc.data[r * cb + 4 * size_t(j % 4)], &v, 4);
        (*qmark)++;
      }
      j++;
    }
  }
  return dead;
}

// ------------------------------------------------------------------ bignum
struct Big {
  std::vector<uint32_t> l;  // little-endian limbs
  void mul_add(uint32_t m, uint32_t a) {
    uint64_t c = a;
    for (auto& x : l) {
      uint64_t t = uint64_t(x) * m + c;
      x = uint32_t(t);
      c = t >> 32;
    }
    if (c) l.push_back(uint32_t(c));
  }
  uint32_t divmod(uint32_t d) {  // this /= d, returns the remainder
    uint64_t r = 0;
    for (size_t i = l.size(); i-- > 0;) {
      uint64_t t = (r << 32) | l[i];
      l[i] = uint32_t(t / d);
      r = t % d;
    }
    while (!l.empty() && !l.back()) l.pop_back();
    return uint32_t(r);
  }
  size_t bitlen() const {
    if (l.empty()) return 0;
    size_t b = 32 * (l.size() - 1);
    uint32_t t = l.back();
    while (t) {
      b++;
      t >>= 1;
    }
    return b;
  }
  bool zero() const { return l.empty(); }
};

// whole bytes the groups can carry: floor((bitlen(prod n!) - 1) / 8)
size_t capacity_bytes(const std::vector<Group>& gs) {
  Big P;
  P.l = {1};
  for (const auto& G : gs)
    for (int i = 2; i <= G.n; i++) P.mul_add(uint32_t(i), 0);
  return (P.bitlen() - 1) / 8;
}

// the NP container:
//   magic | u32 n | u32 nA | u32 lenA | u32 K | A[1..] | B[1+K..]
// Stream A = the tensors up to the last one a group touches (file order);
// stream B = the rest.  B continues all models of A except the int4 trees and
// the per-name trees of the row scales (those start fresh), so its bytes do
// not depend on the order of the rows in A.  B[1..K] (B[0] is the coder's zero
// cache byte) are carried by the orders: per group, digit k of n elements has
// radix n - k and is the rank of the stored element among the remaining ones
// in bytewise order of the key row; the reader folds the digits in file order.
// Format NP2 (np2): stream B also starts the f32 models fresh, magic FX2TFWP5/6.
std::vector<uint8_t> encode_np(std::vector<Ent> es, int ver, bool k, const std::vector<Group>& gs,
                               bool np2, size_t* K_out, std::vector<Ent>* permuted_out) {
  std::vector<bool> inA(es.size(), false);
  std::vector<uint8_t> keyflag(es.size(), 0);
  for (const auto& G : gs) {
    for (auto p : G.rows_at) inA[p.first] = true;
    for (auto p : G.elem_at) inA[p.first] = true;
    for (auto p : G.cols_at) inA[p.first] = true;
    inA[G.key] = true;
    if (G.flag != 0x40 && G.flag != 0x20 && !(np2 && G.flag == 0x60))
      die("group %s: key flag 0x%02x", G.name.c_str(), unsigned(G.flag));
    // format NP leaves the f32 models of stream B running: no group may move f32 values
    for (const auto* v : {&G.rows_at, &G.elem_at, &G.cols_at})
      for (auto p : *v)
        if (!np2 && es[p.first].dt == DT_F32)
          die("group %s moves the f32 tensor %s: format NP2 only", G.name.c_str(),
              es[p.first].name.c_str());
    if (keyflag[G.key] && keyflag[G.key] != G.flag)
      die("tensor %s: two key flags", es[G.key].name.c_str());
    keyflag[G.key] = G.flag;
  }
  // encodings of the unpermuted tensors (an order does not change them)
  std::vector<uint8_t> enc = encodings(es, ver, k);
  std::vector<int> ia, ib;
  size_t last = 0;
  for (size_t i = 0; i < es.size(); i++)
    if (inA[i]) last = i;
  for (size_t i = 0; i < es.size(); i++) (i <= last ? ia : ib).push_back(int(i));
  // stream B: the header models after A (values do not enter them) and the
  // value models after A, int4 trees and per-name scale trees fresh
  HeadCoder HB;
  ValModels VB;
  {
    BinEncoder scratch;
    for (int i : ia) {
      std::string cn = HB.code(scratch, es[i], uint8_t(enc[i] | keyflag[i]));
      encode_values(scratch, VB, es[i], enc[i], cn);
    }
    Models& m = VB.m;
    std::fill(m.cint4.begin(), m.cint4.end(), CountNode());
    std::fill(m.cint4w.begin(), m.cint4w.end(), CountNode());
    std::fill(m.cint4q7.begin(), m.cint4q7.end(), CountNode());
    std::fill(m.cint4q3.begin(), m.cint4q3.end(), CountNode());
    m.zrow = CountNode();
    for (int c = 0; c < 3; c++) m.qa0[c] = m.qa1[c] = CountNode();
    VB.bf16c.clear();
    if (np2) {  // NP2: the groups move f32 rows, the f32 models start fresh too
      m.f32hl.assign(257 * 256, 1024);
      VB.f32hc.clear();
    }
  }
  BinEncoder EB;
  for (int i : ib) {
    std::string cn = HB.code(EB, es[i], enc[i]);
    encode_values(EB, VB, es[i], enc[i], cn);
  }
  std::vector<uint8_t> B = EB.finish();
  size_t K = capacity_bytes(gs);
  if (B.size() < K + 2) die("stream B too short for K");
  if (B[0] != 0) die("stream B does not start with the zero cache byte");
  // the carried number; the reader folds the digits in file order (Horner
  // x = x * radix + digit), so they come out of X from the last one
  Big X;
  for (size_t j = K; j-- > 0;) X.mul_add(256, B[1 + j]);
  std::vector<std::vector<uint32_t>> digs(gs.size());
  for (size_t g = 0; g < gs.size(); g++) digs[g].resize(gs[g].n);
  for (size_t g = gs.size(); g-- > 0;)
    for (int j = gs[g].n; j-- > 0;) digs[g][j] = X.divmod(uint32_t(gs[g].n - j));
  if (!X.zero()) die("carried number does not fit");
  for (size_t gi = 0; gi < gs.size(); gi++) {
    const Group& G = gs[gi];
    const size_t kb = key_bytes(es, G);
    std::vector<int> idx(G.n);
    for (int j = 0; j < G.n; j++) idx[j] = j;
    std::sort(idx.begin(), idx.end(), [&](int x, int y) {
      return std::memcmp(key_row(es, G, x), key_row(es, G, y), kb) < 0;
    });
    for (int j = 1; j < G.n; j++)
      if (!std::memcmp(key_row(es, G, idx[j - 1]), key_row(es, G, idx[j]), kb))
        die("group %s: two equal key rows, the order is not readable", G.name.c_str());
    std::vector<int> perm(G.n);
    for (int j = 0; j < G.n; j++) {
      uint32_t d = digs[gi][j];
      perm[j] = idx[d];
      idx.erase(idx.begin() + d);
    }
    apply_perm(es, G, perm);
  }
  // stream A, with the permuted tensors
  BinEncoder EA;
  HeadCoder H;
  ValModels VA;
  for (int i : ia) {
    std::string cn = H.code(EA, es[i], uint8_t(enc[i] | keyflag[i]));
    encode_values(EA, VA, es[i], enc[i], cn);
  }
  std::vector<uint8_t> A = EA.finish();
  if (A[0] != 0) die("stream A does not start with the zero cache byte");
  std::vector<uint8_t> out(8);
  std::memcpy(out.data(), np2 ? (ver >= 6 ? "FX2TFWP6" : "FX2TFWP5") : (ver >= 6 ? "FX2TFWN6" : "FX2TFWN5"), 8);
  auto put = [&](uint32_t v) {
    auto b = u32le(v);
    out.insert(out.end(), b.begin(), b.end());
  };
  put(uint32_t(es.size()));
  put(uint32_t(ia.size()));
  put(uint32_t(A.size() - 1));
  put(uint32_t(K));
  out.insert(out.end(), A.begin() + 1, A.end());
  out.insert(out.end(), B.begin() + 1 + K, B.end());
  *K_out = K;
  *permuted_out = es;
  return out;
}

}  // namespace
}  // namespace fx2

using namespace fx2;

static std::vector<uint8_t> slurp(const char* p) {
  FILE* f = std::fopen(p, "rb");
  if (!f) die("cannot open %s", p);
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> b(size_t(sz > 0 ? sz : 0));
  if (std::fread(b.data(), 1, b.size(), f) != b.size()) die("short read %s", p);
  std::fclose(f);
  return b;
}

static void spit(const char* p, const std::vector<uint8_t>& b) {
  FILE* f = std::fopen(p, "wb");
  if (!f) die("cannot write %s", p);
  if (std::fwrite(b.data(), 1, b.size(), f) != b.size()) die("short write %s", p);
  std::fclose(f);
}

// identity probe: version and ENC_F32HC of the input, from the plain writer
static void identity(const char* in, const std::vector<Ent>& es, int* ver, bool* k) {
  std::vector<uint8_t> orig = slurp(in);
  if (orig.size() < 8) die("%s: too small", in);
  if (!std::memcmp(orig.data(), "FX2TFWC5", 8)) *ver = 5;
  else if (!std::memcmp(orig.data(), "FX2TFWC6", 8)) *ver = 6;
  else die("%s: not a v5 or v6 container (magic %.8s)", in, reinterpret_cast<const char*>(orig.data()));
  for (int kk = 0; kk < 2; kk++)
    if (encode_plain(es, *ver, kk == 1) == orig) {
      *k = kk == 1;
      std::printf("identity probe (v%d writer%s): BIT IDENTICAL, %zu B\n", *ver,
                  *k ? ", ENC_F32HC" : "", orig.size());
      return;
    }
  die("STOP: the v%d writer of this tool does NOT reproduce %s bit for bit "
      "(with and without ENC_F32HC); nothing was written", *ver, in);
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "dump" && argc == 4) {
    std::vector<Ent> es = load_ents(argv[2]);
    FILE* f = std::fopen(argv[3], "wb");
    if (!f) die("cannot write %s", argv[3]);
    std::fwrite("FX2TFW01", 1, 8, f);
    uint32_t n = uint32_t(es.size());
    std::fwrite(&n, 4, 1, f);
    for (const auto& e : es) {
      uint32_t nl = uint32_t(e.name.size()), nd = uint32_t(e.shape.size());
      std::fwrite(&nl, 4, 1, f);
      std::fwrite(e.name.data(), 1, nl, f);
      std::fwrite(&e.dt, 1, 1, f);
      std::fwrite(&nd, 4, 1, f);
      std::fwrite(e.shape.data(), 4, nd, f);
      std::fwrite(e.data.data(), 1, e.data.size(), f);
    }
    std::fclose(f);
    std::printf("%u tensors\n", n);
    return 0;
  }
  if (mode == "ident" && argc == 3) {
    std::vector<Ent> es = load_ents(argv[2]);
    int ver;
    bool k;
    identity(argv[2], es, &ver, &k);
    return 0;
  }
  if (mode == "np" && argc >= 4) {
    const char* in = argv[2];
    const char* out = argv[3];
    int kopt = -1;  // -1: as the input
    bool fw = false, np2 = false, dead = false, mark = false;
    for (int a = 4; a < argc; a++) {
      std::string o = argv[a];
      if (o == "k") kopt = 1;
      else if (o == "nok") kopt = 0;
      else if (o == "fw") fw = true;
      else if (o == "np2") np2 = true;
      else if (o == "dead") dead = true;
      else if (o == "deadmark") dead = mark = true;  // markers also without np2 (engine probe)
      else die("unknown option %s", o.c_str());
    }
    FILE* ex = std::fopen(out, "rb");
    if (ex) {
      std::fclose(ex);
      die("refusing to overwrite %s", out);
    }
    std::vector<Ent> es = load_ents(in);
    int ver;
    bool kin;
    identity(in, es, &ver, &kin);
    bool k = kopt < 0 ? kin : kopt == 1;
    std::vector<uint8_t> plain_in = encode_plain(es, ver, k);
    int ndead = 0, qmark = 0;
    if (dead) {
      ndead = zero_dead_key_dims(es, np2 || mark, &qmark);
      std::printf("dead KDA key dims: %d zeroed (k conv, dt_bias, q conv; %d q conv marker rows)\n", ndead,
                  qmark);
    }
    std::vector<Group> gs = np_groups(es, fw, np2);
    size_t K;
    std::vector<Ent> perm;
    std::vector<uint8_t> np = encode_np(es, ver, k, gs, np2, &K, &perm);
    std::vector<uint8_t> plain = encode_plain(es, ver, k);
    spit(out, np);
    // read back with the reader of this tree
    std::vector<Ent> back = load_ents(out);
    bool same = back.size() == perm.size();
    for (size_t i = 0; same && i < back.size(); i++)
      same = back[i].name == perm[i].name && back[i].dt == perm[i].dt &&
             back[i].shape == perm[i].shape && back[i].data == perm[i].data;
    bool moved = false;
    for (size_t i = 0; i < es.size(); i++) moved |= es[i].data != perm[i].data;
    std::printf("read back %s against the permuted tensors: %s (%zu tensors), permutation %s\n",
                np2 ? "NP2" : "NP", same ? "BIT IDENTICAL" : "DIFFERENT", back.size(),
                moved ? "non-trivial" : "TRIVIAL");
    if (!same || !moved) {
      std::remove(out);
      die("STOP: output removed");
    }
    size_t so = slurp(in).size();
    std::printf("groups %zu%s%s, carried K = %zu B, plain v%d%s %zu B%s, %s %zu B (%+ld B against "
                "plain, form cost + K - plain %+ld B)\n",
                gs.size(), fw ? " (with the fast-weight layers)" : "", np2 ? " (with the KDA dims)" : "",
                K, ver, k ? "k" : "", plain.size(), dead ? " (after dead)" : "", np2 ? "NP2" : "NP",
                np.size(), long(np.size()) - long(plain.size()), long(np.size() + K) - long(plain.size()));
    if (dead)
      std::printf("dead: plain v%d%s %zu B before, %zu B after (%+ld B)\n", ver, k ? "k" : "",
                  plain_in.size(), plain.size(), long(plain.size()) - long(plain_in.size()));
    std::printf("output %s: %zu B, %+ld B against the input (%+ld B in S, the container "
                "ships twice)\n",
                out, np.size(), long(np.size()) - long(so), 2 * (long(np.size()) - long(so)));
    std::printf("the compressor must be built with NEIF_TF1_WC5=1%s%s to read this container.\n",
                k ? " and NEIF_TF1_WC5K=1" : "", np2 ? " and NEIF_TF1_NP2=1" : "");
    if (fw || np2)
      std::printf("WARNING %s: the engine results follow the stored order (%s); no bit-identical "
                  "archive against the input.\n", fw && np2 ? "fw np2" : (fw ? "fw" : "np2"),
                  fw ? "the fast-weight head reads the unit index" : "KDA sums at rounding level");
    return 0;
  }
  if (mode == "plain" && (argc == 4 || argc == 5)) {
    // the decoded tensors of any readable container as a plain v5/v6 one
    // (for readers without format NP); default ENC_F32HC, nok: ENC_F32HI
    FILE* ex = std::fopen(argv[3], "rb");
    if (ex) {
      std::fclose(ex);
      die("refusing to overwrite %s", argv[3]);
    }
    std::vector<uint8_t> head = slurp(argv[2]);
    if (head.size() < 8) die("%s: too small", argv[2]);
    const int ver = head[7] == '6' ? 6 : 5;
    std::vector<Ent> es = load_ents(argv[2]);
    bool k = true;
    if (argc == 5) {
      if (!std::strcmp(argv[4], "nok")) k = false;
      else if (std::strcmp(argv[4], "k")) die("unknown option %s", argv[4]);
    }
    std::vector<uint8_t> p = encode_plain(es, ver, k);
    spit(argv[3], p);
    std::vector<Ent> back = load_ents(argv[3]);
    bool same = back.size() == es.size();
    for (size_t i = 0; same && i < back.size(); i++)
      same = back[i].name == es[i].name && back[i].data == es[i].data;
    if (!same) {
      std::remove(argv[3]);
      die("STOP: plain read back differs, output removed");
    }
    std::printf("plain v%d%s %s: %zu B, read back BIT IDENTICAL (%zu tensors)\n", ver, k ? "k" : "",
                argv[3], p.size(), es.size());
    return 0;
  }
  std::fprintf(stderr, "usage: see the head comment of tools_src/tf1_np_codec.cpp\n");
  return 2;
}
