// Change in this file: mlp_width() and the mlp.up / mlp.down sites can take
// the layer's neuron count from the tensor form instead of the constant DMLP
// (layers with removed neurons); additive only.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/arena_build.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0.
#include "../../meld.h"
#include "arena_build.h"

#include <sys/mman.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../weights_io.h"
#include "glue.h"
#include "../../page_registry.h"
#include "attn.h"   // attention window (ATTN_WIN)

namespace fx2 {
namespace opt {

namespace {

[[noreturn]] void die(const char* msg) {
  std::fprintf(stderr, "%s\n", msg);
  std::exit(1);
}

constexpr size_t round64(size_t x) { return (x + 63) & ~size_t(63); }

}  // namespace

void* MapNull(size_t n) {
  void* m = ::mmap(nullptr, n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) msg_stop(187);
  return m;
}

void HugeBuf::alloc(size_t n) {
  bytes = (n + (size_t(2) << 20) - 1) & ~((size_t(2) << 20) - 1);
  void* m = MapNull(bytes);
  ::madvise(m, bytes, MADV_HUGEPAGE);
  std::memset(m, 0, bytes);  // touch every page
  p = static_cast<uint8_t*>(m);
  PAGE_REGISTRY_ADD(p, bytes);
}

void HugeBuf::trim(size_t keep) {
  const size_t k = (keep + 4095) & ~size_t(4095);
  if (p && k < bytes) ::madvise(p + k, bytes - k, MADV_DONTNEED);
}

HugeBuf::~HugeBuf() {
  if (p) ::munmap(p, bytes);
}

// The buffers the vectors held (operator delete of a null
// pointer does nothing)
OptModel::~OptModel() {
  ::operator delete(rope_sin);
  ::operator delete(rope_cos);
  for (int i = 0; i < NVAN; i++) ::operator delete(van[i].ve);
  ::operator delete(unembed_f32);
}

namespace {
// N floats from operator new, as vector<float>(n) took them;
// every caller writes all of them
float* FloatPuffer(size_t n) { return static_cast<float*>(::operator new(sizeof(float) * n)); }
}  // namespace
static_assert(TID_OPT("skip_connection_weights.value") < 0 &&
                  TID_OPT("blocks.#.residual_stream_coefficient.value") < 0 &&
                  TID_OPT("blocks.#.token_embedding_coefficient.value") < 0,
              "AttnRes build, but the container has U-Net skips / residual / token coefficients (E264)");
static_assert(TID_OPT("embedding.last_table.weight.q") < 0 || TID_OPT("embedding.last_projection.weight.q") >= 0,
              "container has last_table but no last_projection (E192)");
static_assert(TID_OPT("embedding.match_table.weight.q") < 0 || TID_OPT("embedding.match_projection.weight.q") >= 0,
              "container has match_table but no match_projection (E193)");

void OptModel::load(const char* weights_path) {
  WeightsFile wf = WeightsFile::load_compressed(weights_path);

  {  // config sanity: the line against the container
    const int32_t* ci = wf.T(TID("config.ints")).i32();
    const int32_t want[11] = {V, D, NL, DH, NH, DMLP, ATTN_WIN,
                              DH, NH, 4, 10000};
    for (int i = 0; i < 11; i++)
      if (ci[i] != want[i]) msg_stop(184);
    const int32_t* ck = wf.T(TID("config.kimi")).i32();
    for (int i = 0; i < NL; i++)
      if ((ck[i] != 0) != KIMI_L[i]) msg_stop(185);
  }

  constexpr bool want_last = TID_OPT("embedding.last_table.weight.q") >= 0;
  constexpr bool want_match = TID_OPT("embedding.match_table.weight.q") >= 0;
  paket.present = want_last || want_match;
  paket.emb_rows = V;
  const int emb_rows = paket.present ? 216 : V;
  pool.alloc(size_t(paket.present ? 20 : 16) << 20);
  size_t off = 0;
  auto take = [&](size_t n) -> uint8_t* {
    off = round64(off);
    uint8_t* q = pool.p + off;
    off += n;
    return q;
  };

  // dense arena site in stream order; iq/is/ia: weight.q, weight.scale,
  // quantize_activation.scale of one site
  auto qsite = [&](int iq, int is, int ia, int blk, int d_out, int d_in) -> QSite {
    QSite s;
    s.s_act = bf16_to_f32(wf.T(ia, blk).bf16_bits()[0]);
    const uint16_t* ws = wf.T(is, blk).bf16_bits();
    float fold[DMLP > V ? DMLP : V];
    for (int o = 0; o < d_out; o++) fold[o] = s.s_act * bf16_to_f32(ws[o]);
    uint8_t* dst = take(qdense_bytes(d_out, d_in));
    s.m = qdense_build(dst, wf.T(iq, blk).i8(), fold, d_out, d_in, true);
    return s;
  };
  // int4 sparse-column arena site (d_out 192): [fold | cols]
  auto qsp4 = [&](int iq, int is, int ia, int blk, int d_in, float* s_act_out) -> QSparse4 {
    const float s_act = bf16_to_f32(wf.T(ia, blk).bf16_bits()[0]);
    *s_act_out = s_act;
    const uint16_t* ws = wf.T(is, blk).bf16_bits();
    float fold[D];
    for (int o = 0; o < D; o++) fold[o] = s_act * bf16_to_f32(ws[o]);
    float* foldp = reinterpret_cast<float*>(take(sizeof(float) * D));
    uint8_t* cols = take(qsparse4_bytes(d_in));
    return qsparse4_build(cols, foldp, wf.T(iq, blk).i8(), fold, D, d_in);
  };
#define TF_QSITE(p, blk, d_out, d_in) \
  qsite(TID(p ".weight.q"), TID(p ".weight.scale"), TID(p ".quantize_activation.scale"), blk, d_out, d_in)
#define TF_QSP4(p, blk, d_in, s_act_out) \
  qsp4(TID(p ".weight.q"), TID(p ".weight.scale"), TID(p ".quantize_activation.scale"), blk, d_in, s_act_out)
  auto f32block = [&](const float* src, size_t n) -> float* {
    float* p = reinterpret_cast<float*>(take(sizeof(float) * n));
    std::memcpy(p, src, sizeof(float) * n);
    return p;
  };

  // ---- stream, in per-token consumption order ----
  prior = TF_QSP4("prior_embedding", -1, V, &prior_s_act);
  // AttnRes replace mode (after cmix-neif-beehive): one depth-mix query per sub-layer read, the final read
  for (int l = 0; l < NL; l++) {
    std::memcpy(ar_q_mix[l], wf.T(TID("blocks.#.q_mix.w"), l).f32(), sizeof(float) * D);
    std::memcpy(ar_q_mlp[l], wf.T(TID("blocks.#.q_mlp.w"), l).f32(), sizeof(float) * D);
  }
  std::memcpy(ar_q_final, wf.T(TID("blocks.#.q_final.w"), NL - 1).f32(), sizeof(float) * D);
  static_assert(NVAN == 4, "four value embeddings");
  static constexpr int kVeQ[4] = {TID("value_embeddings.0.inner.weight.q"), TID("value_embeddings.1.inner.weight.q"),
                                  TID("value_embeddings.2.inner.weight.q"), TID("value_embeddings.3.inner.weight.q")};
  static constexpr int kVeS[4] = {TID("value_embeddings.0.inner.weight.scale"), TID("value_embeddings.1.inner.weight.scale"),
                                  TID("value_embeddings.2.inner.weight.scale"), TID("value_embeddings.3.inner.weight.scale")};

  int ki = 0, vi = 0;
  for (int l = 0; l < NL; l++) {
    if (KIMI_L[l]) {
      KimiArenas& L = kimi[ki];
      layer2kimi[l] = ki;
      L.qp = TF_QSITE("blocks.#.attention.query_projection", l, D, D);
      L.kp = TF_QSITE("blocks.#.attention.key_projection", l, D, D);
      L.vp = TF_QSITE("blocks.#.attention.value_projection", l, D, D);
      L.fg_up = TF_QSITE("blocks.#.attention.forget_gate_projection.up", l, DH, D);
      L.fg_down = TF_QSITE("blocks.#.attention.forget_gate_projection.down", l, D, DH);
      L.og_up = TF_QSITE("blocks.#.attention.output_gate_projection.up", l, DH, D);
      L.og_down = TF_QSITE("blocks.#.attention.output_gate_projection.down", l, D, DH);
      L.beta_w = f32block(wf.T(TID("blocks.#.attention.beta_projection.weight"), l).f32(), size_t(NH) * D);
      static constexpr int kConv[3] = {TID("blocks.#.attention.query_convolution.weight"),
                                       TID("blocks.#.attention.key_convolution.weight"),
                                       TID("blocks.#.attention.value_convolution.weight")};
      for (int c = 0; c < 3; c++) {
        const float* cw = wf.T(kConv[c], l).f32();
        float* wt = reinterpret_cast<float*>(take(sizeof(float) * 4 * D));
        for (int j = 0; j < 4; j++)  // tap-major repack, same as naive
          for (int ch = 0; ch < D; ch++) wt[j * D + ch] = cw[ch * 4 + j];
        L.kw.conv_w[c] = wt;
      }
      L.kw.dt_bias = f32block(wf.T(TID("blocks.#.attention.dt_bias"), l).f32(), D);
      L.kw.gn_w = f32block(wf.T(TID("blocks.#.attention.output_fused_norm_gate.weight"), l).f32(), DH);
      const float* al = wf.T(TID("blocks.#.attention.log_baseline_decay_rate"), l).f32();
      for (int h = 0; h < NH; h++) L.kw.a_neg[h] = -std::exp(al[h]);
      L.op = TF_QSITE("blocks.#.attention.output_projection", l, D, D);
      ki++;
    } else {
      VanArenas& L = van[vi];
      layer2van[l] = vi;
      L.qp = TF_QSITE("blocks.#.attention.query_projection", l, D, D);
      constexpr int KVH = 1;   // query head h reads K/V head h % KVH
      L.kp = TF_QSITE("blocks.#.attention.key_projection", l, KVH * DH, D);
      L.vp = TF_QSITE("blocks.#.attention.value_projection", l, KVH * DH, D);
      L.op = TF_QSITE("blocks.#.attention.output_projection", l, D, D);
      const uint16_t* qs = wf.T(TID("blocks.#.attention.quantize_queries.scale"), l).bf16_bits();
      const uint16_t* ks = wf.T(TID("blocks.#.attention.quantize_keys.scale"), l).bf16_bits();
      const uint16_t* vs = wf.T(TID("blocks.#.attention.quantize_values.scale"), l).bf16_bits();
      for (int h = 0; h < NH; h++) {
        L.sq[h] = bf16_to_f32(qs[h]);
        L.sk[h] = bf16_to_f32(ks[h % KVH]);
        L.sv[h] = bf16_to_f32(vs[h % KVH]);
        L.coef[h] = 0.125f * L.sq[h] * L.sk[h];
      }
      {
        // value embedding of window layer vi (after cmix-neif-beehive): two mixing
        // coefficients and a V x (KVH*64) table, row = int8 value * bf16 row scale
        constexpr int W = 1 * DH;
        const float* vc = wf.T(TID("blocks.#.attention.value_embedding_coefficients.value"), l).f32();
        L.ve_c0 = vc[0];
        L.ve_c1 = vc[1];
        const int8_t* q = wf.T(kVeQ[vi]).i8();
        const uint16_t* sc = wf.T(kVeS[vi]).bf16_bits();
        L.ve = FloatPuffer(size_t(V) * W);   // the loop writes every value
        for (int r = 0; r < V; r++) {
          const float f = bf16_to_f32(sc[r]);
          for (int i = 0; i < W; i++)
            L.ve[size_t(r) * W + i] = static_cast<float>(q[size_t(r) * W + i]) * f;
        }
      }
      vi++;
    }
    const int d_mlp = static_cast<int>(wf.T(TID("blocks.#.mlp.up.weight.q"), l).shape[0]);
    mlp[l].up = TF_QSITE("blocks.#.mlp.up", l, d_mlp, D);
    mlp[l].down = TF_QSP4("blocks.#.mlp.down", l, d_mlp, &mlp[l].down_s_act);
  }

  unembed = TF_QSITE("unembedding", -1, V, D);
  std::memcpy(plm_w, wf.T(TID("logit_mixing_head.weight")).f32(), sizeof(float) * D);
  {
    const int8_t* uq = wf.T(TID("unembedding.weight.q")).i8();
    const uint16_t* us = wf.T(TID("unembedding.weight.scale")).bf16_bits();
    unembed_f32 = FloatPuffer(size_t(V) * D);   // the loop writes every value
    for (int o = 0; o < V; o++) {
      const float sc = bf16_to_f32(us[o]);
      for (int i = 0; i < D; i++)
        unembed_f32[size_t(o) * D + i] = float(uq[size_t(o) * D + i]) * sc;
    }
  }
  stream_bytes = round64(off);

  // ---- non-stream data: tail slack gap, then the embedding table ----
  off += QMAT_TAIL_SLACK;
  {
    const int8_t* eq = wf.T(TID("embedding.weight.q")).i8();
    const uint16_t* es = wf.T(TID("embedding.weight.scale")).bf16_bits();
    float rs[V + 8];
    for (int c = 0; c < emb_rows; c++) rs[c] = bf16_to_f32(es[c]);
    float* table = reinterpret_cast<float*>(take(sizeof(float) * emb_rows * D));
    if (paket.present) {
      // rows without norm: the embedding norm runs per step, after the two
      // projection terms, as in the training model
      for (int c = 0; c < emb_rows; c++) {
        float* row = table + static_cast<size_t>(c) * D;
        const int8_t* src = eq + static_cast<size_t>(c) * D;
        for (int i = 0; i < D; i++) row[i] = static_cast<float>(src[i]) * rs[c];
      }
      paket.emb_rows = emb_rows;
      paket.emb_raw = table;
    }
    tok_table = table;
  }

  if (paket.present) {
    // dequantized channel tables (row = int4 value * per-row bf16 scale)
    auto rows = [&](int iq, int is, int n_rows, int d) -> float* {
      const int8_t* q = wf.T(iq).i8();
      const uint16_t* s = wf.T(is).bf16_bits();
      float* p = reinterpret_cast<float*>(take(sizeof(float) * size_t(n_rows) * d));
      for (int r = 0; r < n_rows; r++) {
        const float sc = bf16_to_f32(s[r]);
        for (int i = 0; i < d; i++) p[size_t(r) * d + i] = static_cast<float>(q[size_t(r) * d + i]) * sc;
      }
      return p;
    };
    // a projection site: int4 weights kept raw, the row scale folded with the
    // activation scale exactly as qsite() does it
    auto proj = [&](int iq, int is, int ia, int d_in, const int8_t** q_out, const float** fold_out,
                    float* s_act_out) {
      const float s_act = bf16_to_f32(wf.T(ia).bf16_bits()[0]);
      int8_t* q = reinterpret_cast<int8_t*>(take(size_t(D) * d_in));
      std::memcpy(q, wf.T(iq).i8(), size_t(D) * d_in);
      float* fold = reinterpret_cast<float*>(take(sizeof(float) * D));
      const uint16_t* ws = wf.T(is).bf16_bits();
      for (int o = 0; o < D; o++) fold[o] = s_act * bf16_to_f32(ws[o]);
      *q_out = q;
      *fold_out = fold;
      *s_act_out = s_act;
    };
    if (want_last) {
      paket.last_rows = rows(TID("embedding.last_table.weight.q"), TID("embedding.last_table.weight.scale"),
                             PaketArenas::X * (emb_rows + 1), PaketArenas::DSLOT);
      proj(TID("embedding.last_projection.weight.q"), TID("embedding.last_projection.weight.scale"),
           TID("embedding.last_projection.quantize_activation.scale"), PaketArenas::DIN_LAST, &paket.last_q,
           &paket.last_fold, &paket.last_s_act);
      paket.has_last = true;
    }
    if (want_match) {
      paket.match_rows = rows(TID("embedding.match_table.weight.q"), TID("embedding.match_table.weight.scale"),
                              PaketArenas::NMATCH + PaketArenas::NLEN, PaketArenas::DSLOT);
      proj(TID("embedding.match_projection.weight.q"), TID("embedding.match_projection.weight.scale"),
           TID("embedding.match_projection.quantize_activation.scale"), PaketArenas::DIN_MATCH, &paket.match_q,
           &paket.match_fold, &paket.match_s_act);
      paket.has_match = true;
    }
  }
#undef TF_QSITE
#undef TF_QSP4

  {
    std::memcpy(inv_freq, wf.T(TID("rope.inv_freq")).f32(), sizeof(inv_freq));
    // own copies as the vectors made them (one operator new of 16 MiB each,
    // sin first), so the heap sees the same requests as before
    rope_sin = FloatPuffer(size_t(ROPE_LEN) * 32);
    std::memcpy(rope_sin, wf.T(TID("rope.sin")).f32(), sizeof(float) * size_t(ROPE_LEN) * 32);
    rope_cos = FloatPuffer(size_t(ROPE_LEN) * 32);
    std::memcpy(rope_cos, wf.T(TID("rope.cos")).f32(), sizeof(float) * size_t(ROPE_LEN) * 32);
    PAGE_REGISTRY_ADD(rope_sin, size_t(ROPE_LEN) * 32 * sizeof(float));
    PAGE_REGISTRY_ADD(rope_cos, size_t(ROPE_LEN) * 32 * sizeof(float));
  }
  pool.trim(off + QMAT_TAIL_SLACK);
}

}  // namespace opt
}  // namespace fx2
