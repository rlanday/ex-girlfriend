// Change in this file: mlp_width() and the mlp.up / mlp.down sites can take
// the layer's neuron count from the tensor form instead of the constant DMLP
// (layers with removed neurons); additive only.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/arena_build.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md.
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
  std::fprintf(stderr, "arena_build: %s\n", msg);
  std::exit(1);
}

constexpr size_t round64(size_t x) { return (x + 63) & ~size_t(63); }

}  // namespace

void HugeBuf::alloc(size_t n) {
  bytes = (n + (size_t(2) << 20) - 1) & ~((size_t(2) << 20) - 1);
  void* m = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) die("mmap failed");
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

void OptModel::load(const char* weights_path) {
  // accept both the raw FX2TFW01 file and the losslessly compressed
  // FX2TFWC1 to FX2TFWC4 files (bit-identical tensors either way)
  char magic[8] = {0};
  if (FILE* f = std::fopen(weights_path, "rb")) {
    if (std::fread(magic, 1, 8, f) != 8) magic[0] = 0;
    std::fclose(f);
  }
  WeightsFile wf = std::memcmp(magic, "FX2TFWC", 7) == 0
                       ? WeightsFile::load_compressed(weights_path)
                       : WeightsFile::load(weights_path);

  {  // config sanity (same checks as the naive loader)
    const WTensor& ci = wf.get("config.ints", DT_I32, {11});
    const int32_t want[11] = {V, D, NL, DH, NH, DMLP, ATTN_WIN,
                              DH, NH, 4, 10000};
    for (int i = 0; i < 11; i++) {
      if (ci.i32()[i] == want[i]) continue;
      die("config.ints mismatch");
    }
    const WTensor& ck = wf.get("config.kimi", DT_I32, {NL});
    for (int i = 0; i < NL; i++)
      if ((ck.i32()[i] != 0) != KIMI_L[i]) die("config.kimi mismatch");
  }

  const bool want_last = wf.has("embedding.last_table.weight.q");
  const bool want_match = wf.has("embedding.match_table.weight.q");
  paket.present = want_last || want_match;
  const int emb_rows = paket.present ? 216 : V;
  // the extra tables are ~220 KB; give them their own room instead of eating
  // into the tail slack of the weight stream
  pool.alloc(size_t(paket.present ? 20 : 16) << 20);
  size_t off = 0;
  auto take = [&](size_t n) -> uint8_t* {
    off = round64(off);
    uint8_t* q = pool.p + off;
    off += n;
    if (off + QMAT_TAIL_SLACK > pool.bytes) die("weight pool overflow");
    return q;
  };

  // dense arena site in stream order
  auto qsite = [&](const std::string& prefix, int d_out, int d_in,
                   bool biased) -> QSite {
    const WTensor& wq =
        wf.get(prefix + ".weight.q", DT_I8,
               {static_cast<uint32_t>(d_out), static_cast<uint32_t>(d_in)});
    const WTensor& ws = wf.get(prefix + ".weight.scale", DT_BF16,
                               {static_cast<uint32_t>(d_out)});
    const WTensor& sa =
        wf.get(prefix + ".quantize_activation.scale", DT_BF16, {1});
    QSite s;
    s.s_act = bf16_to_f32(sa.bf16_bits()[0]);
    if (!(s.s_act > 0.0f)) die("activation scale not positive");
    std::vector<float> fold(d_out);
    for (int o = 0; o < d_out; o++)
      fold[o] = s.s_act * bf16_to_f32(ws.bf16_bits()[o]);
    uint8_t* dst = take(qdense_bytes(d_out, d_in));
    s.m = qdense_build(dst, wq.i8(), fold.data(), d_out, d_in, biased);
    return s;
  };

  // int4 sparse-column arena site (d_out must be 192): [fold | cols]
  auto qsp4 = [&](const std::string& prefix, int d_in,
                  float* s_act_out) -> QSparse4 {
    const WTensor& wq =
        wf.get(prefix + ".weight.q", DT_I8,
               {static_cast<uint32_t>(D), static_cast<uint32_t>(d_in)});
    const WTensor& ws = wf.get(prefix + ".weight.scale", DT_BF16,
                               {static_cast<uint32_t>(D)});
    const WTensor& sa =
        wf.get(prefix + ".quantize_activation.scale", DT_BF16, {1});
    float s_act = bf16_to_f32(sa.bf16_bits()[0]);
    if (!(s_act > 0.0f)) die("activation scale not positive");
    *s_act_out = s_act;
    std::vector<float> fold(D);
    for (int o = 0; o < D; o++)
      fold[o] = s_act * bf16_to_f32(ws.bf16_bits()[o]);
    float* foldp = reinterpret_cast<float*>(take(sizeof(float) * D));
    uint8_t* cols = take(qsparse4_bytes(d_in));
    return qsparse4_build(cols, foldp, wq.i8(), fold.data(), D, d_in);
  };

  auto mlp_width = [&](const std::string& b) -> int {
    const WTensor& up = wf.get(b + "mlp.up.weight.q");
    if (up.dtype != DT_I8 || up.shape.size() != 2 ||
        up.shape[1] != static_cast<uint32_t>(D))
      die("mlp.up.weight.q: expected an int8 [d_mlp x D] matrix");
    const int d = static_cast<int>(up.shape[0]);
    if (d < 1 || d > DMLP) die("mlp width outside 1 .. config.ints[5]");
    const WTensor& dn = wf.get(b + "mlp.down.weight.q");
    if (dn.dtype != DT_I8 || dn.shape.size() != 2 ||
        dn.shape[0] != static_cast<uint32_t>(D) ||
        dn.shape[1] != static_cast<uint32_t>(d))
      die("mlp.down.weight.q does not match the mlp.up width");
    return d;
  };

  auto f32block = [&](const float* src, size_t n) -> float* {
    float* p = reinterpret_cast<float*>(take(sizeof(float) * n));
    std::memcpy(p, src, sizeof(float) * n);
    return p;
  };

  // ---- stream, in per-token consumption order ----
  prior = qsp4("prior_embedding", V, &prior_s_act);

  {
    const WTensor& sw = wf.get("skip_connection_weights.value", DT_F32, {NSKIP});
    for (int i = 0; i < NSKIP; i++) skip_w[i] = sw.f32()[i];
  }

  int ki = 0, vi = 0;
  for (int l = 0; l < NL; l++) {
    std::string b = "blocks." + std::to_string(l) + ".";
    rsc[l] =
        wf.get(b + "residual_stream_coefficient.value", DT_F32, {1}).f32()[0];
    tec[l] =
        wf.get(b + "token_embedding_coefficient.value", DT_F32, {1}).f32()[0];
    std::string a = b + "attention.";

    if (KIMI_L[l]) {
      KimiArenas& L = kimi[ki];
      layer2kimi[l] = ki;
      L.qp = qsite(a + "query_projection", D, D, true);
      L.kp = qsite(a + "key_projection", D, D, true);
      L.vp = qsite(a + "value_projection", D, D, true);
      L.fg_up = qsite(a + "forget_gate_projection.up", DH, D, true);
      L.fg_down = qsite(a + "forget_gate_projection.down", D, DH, true);
      L.og_up = qsite(a + "output_gate_projection.up", DH, D, true);
      L.og_down = qsite(a + "output_gate_projection.down", D, DH, true);

      {
        const WTensor& bw =
            wf.get(a + "beta_projection.weight", DT_F32, {NH, D});
        L.beta_w = f32block(bw.f32(), size_t(NH) * D);
      }
      const char* convs[3] = {"query_convolution.weight",
                              "key_convolution.weight",
                              "value_convolution.weight"};
      for (int c = 0; c < 3; c++) {
        const WTensor& cw = wf.get(a + convs[c], DT_F32, {D, 4});
        float* wt = reinterpret_cast<float*>(take(sizeof(float) * 4 * D));
        for (int j = 0; j < 4; j++)  // tap-major repack, same as naive
          for (int ch = 0; ch < D; ch++) wt[j * D + ch] = cw.f32()[ch * 4 + j];
        L.kw.conv_w[c] = wt;
      }
      {
        const WTensor& dt = wf.get(a + "dt_bias", DT_F32, {D});
        L.kw.dt_bias = f32block(dt.f32(), D);
      }
      {
        const WTensor& gw =
            wf.get(a + "output_fused_norm_gate.weight", DT_F32, {DH});
        L.kw.gn_w = f32block(gw.f32(), DH);
      }
      {
        const WTensor& al = wf.get(a + "log_baseline_decay_rate", DT_F32, {NH});
        for (int h = 0; h < NH; h++) L.kw.a_neg[h] = -std::exp(al.f32()[h]);
      }
      L.op = qsite(a + "output_projection", D, D, true);
      ki++;
    } else {
      VanArenas& L = van[vi];
      layer2van[l] = vi;
      L.qp = qsite(a + "query_projection", D, D, true);
      L.kp = qsite(a + "key_projection", D, D, true);
      L.vp = qsite(a + "value_projection", D, D, true);
      L.op = qsite(a + "output_projection", D, D, true);
      const WTensor& qs = wf.get(a + "quantize_queries.scale", DT_BF16, {NH});
      const WTensor& ks = wf.get(a + "quantize_keys.scale", DT_BF16, {NH});
      const WTensor& vs = wf.get(a + "quantize_values.scale", DT_BF16, {NH});
      for (int h = 0; h < NH; h++) {
        L.sq[h] = bf16_to_f32(qs.bf16_bits()[h]);
        L.sk[h] = bf16_to_f32(ks.bf16_bits()[h]);
        L.sv[h] = bf16_to_f32(vs.bf16_bits()[h]);
        L.coef[h] = 0.125f * L.sq[h] * L.sk[h];
      }
      vi++;
    }

    const int d_mlp = mlp_width(b);
    mlp[l].up = qsite(b + "mlp.up", d_mlp, D, true);
    mlp[l].down = qsp4(b + "mlp.down", d_mlp, &mlp[l].down_s_act);
  }
  if (ki != OptModel::NKIMI || vi != OptModel::NVAN)
    die("layer pattern mismatch");

  unembed = qsite("unembedding", V, D, true);
#if PORT_FW
  {  // FW: float copy of the unembedding for the fast-weight backward (Z arena_build.cpp:376-385)
    const WTensor& uq = wf.get("unembedding.weight.q", DT_I8,
                               {static_cast<uint32_t>(V), static_cast<uint32_t>(D)});
    const WTensor& us = wf.get("unembedding.weight.scale", DT_BF16,
                               {static_cast<uint32_t>(V)});
    unembed_f32.assign(size_t(V) * D, 0.0f);
    for (int o = 0; o < V; o++) {
      const float sc = bf16_to_f32(us.bf16_bits()[o]);
      for (int i = 0; i < D; i++)
        unembed_f32[size_t(o) * D + i] = float(uq.i8()[size_t(o) * D + i]) * sc;
    }
  }
#endif
  stream_bytes = round64(off);

  // ---- non-stream data: tail slack gap, then the normed embedding table ----
  off += QMAT_TAIL_SLACK;
  {
    const WTensor& eq = wf.get("embedding.weight.q", DT_I8,
                               {static_cast<uint32_t>(emb_rows),
                                static_cast<uint32_t>(D)});
    const WTensor& es = wf.get("embedding.weight.scale", DT_BF16,
                               {static_cast<uint32_t>(emb_rows)});
    std::vector<float> rs(emb_rows);
    for (int c = 0; c < emb_rows; c++) rs[c] = bf16_to_f32(es.bf16_bits()[c]);
    float* table =
        reinterpret_cast<float*>(take(sizeof(float) * emb_rows * D));
    if (paket.present) {
      // Rows without norm: the embedding norm runs per step, after the two
      // projection terms, as in the training model.
      for (int c = 0; c < emb_rows; c++) {
        float* row = table + static_cast<size_t>(c) * D;
        const int8_t* src = eq.i8() + static_cast<size_t>(c) * D;
        for (int i = 0; i < D; i++) row[i] = static_cast<float>(src[i]) * rs[c];
      }
      paket.emb_rows = emb_rows;
      paket.emb_raw = table;
    } else
    {
      build_normed_embedding_table(eq.i8(), rs.data(), table);
    }
    tok_table = table;
  }

  if (paket.present) {
    // dequantized channel tables (row = int4 value * per-row bf16 scale, the
    // same fake-quantized row CastedEmbedding.forward() hands out)
    auto rows = [&](const std::string& prefix, int n_rows, int d) -> float* {
      const WTensor& q = wf.get(prefix + ".weight.q", DT_I8,
                                {static_cast<uint32_t>(n_rows),
                                 static_cast<uint32_t>(d)});
      const WTensor& s = wf.get(prefix + ".weight.scale", DT_BF16,
                                {static_cast<uint32_t>(n_rows)});
      float* p = reinterpret_cast<float*>(
          take(sizeof(float) * size_t(n_rows) * d));
      for (int r = 0; r < n_rows; r++) {
        const float sc = bf16_to_f32(s.bf16_bits()[r]);
        for (int i = 0; i < d; i++)
          p[size_t(r) * d + i] =
              static_cast<float>(q.i8()[size_t(r) * d + i]) * sc;
      }
      return p;
    };
    // a projection site: int4 weights kept raw, the row scale folded with the
    // activation scale exactly as qsite() does it for the big matmuls
    auto proj = [&](const std::string& prefix, int d_in, const int8_t** q_out,
                    const float** fold_out, float* s_act_out) {
      const WTensor& wq = wf.get(prefix + ".weight.q", DT_I8,
                                 {static_cast<uint32_t>(D),
                                  static_cast<uint32_t>(d_in)});
      const WTensor& ws = wf.get(prefix + ".weight.scale", DT_BF16,
                                 {static_cast<uint32_t>(D)});
      const WTensor& sa =
          wf.get(prefix + ".quantize_activation.scale", DT_BF16, {1});
      const float s_act = bf16_to_f32(sa.bf16_bits()[0]);
      if (!(s_act > 0.0f)) die("activation scale not positive");
      int8_t* q = reinterpret_cast<int8_t*>(take(size_t(D) * d_in));
      std::memcpy(q, wq.i8(), size_t(D) * d_in);
      float* fold = reinterpret_cast<float*>(take(sizeof(float) * D));
      for (int o = 0; o < D; o++)
        fold[o] = s_act * bf16_to_f32(ws.bf16_bits()[o]);
      *q_out = q;
      *fold_out = fold;
      *s_act_out = s_act;
    };
    if (want_last) {
      if (!wf.has("embedding.last_projection.weight.q"))
        die("container has last_table but no last_projection");
      paket.last_rows = rows("embedding.last_table",
                             PaketArenas::X *
                                 (emb_rows + 1),
                             PaketArenas::DSLOT);
      proj("embedding.last_projection", PaketArenas::DIN_LAST, &paket.last_q,
           &paket.last_fold, &paket.last_s_act);
      paket.has_last = true;
    }
    if (want_match) {
      if (!wf.has("embedding.match_projection.weight.q"))
        die("container has match_table but no match_projection");
      paket.match_rows = rows("embedding.match_table",
                              PaketArenas::NMATCH + PaketArenas::NLEN,
                              PaketArenas::DSLOT);
      proj("embedding.match_projection", PaketArenas::DIN_MATCH,
           &paket.match_q, &paket.match_fold, &paket.match_s_act);
      paket.has_match = true;
    }
  }

  {
    const WTensor& fi = wf.get("rope.inv_freq", DT_F32, {32});
    std::memcpy(inv_freq, fi.f32(), sizeof(inv_freq));
    const WTensor& si = wf.get("rope.sin", DT_F32,
                               {static_cast<uint32_t>(ROPE_LEN), 32});
    const WTensor& co = wf.get("rope.cos", DT_F32,
                               {static_cast<uint32_t>(ROPE_LEN), 32});
    rope_sin.assign(si.f32(), si.f32() + size_t(ROPE_LEN) * 32);
    rope_cos.assign(co.f32(), co.f32() + size_t(ROPE_LEN) * 32);
    PAGE_REGISTRY_ADD(rope_sin.data(), rope_sin.size() * sizeof(float));
    PAGE_REGISTRY_ADD(rope_cos.data(), rope_cos.size() * sizeof(float));
  }
  pool.trim(off + QMAT_TAIL_SLACK);
}

}  // namespace opt
}  // namespace fx2
