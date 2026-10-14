// Change in this file: the side-channel tensors (PaketArenas; Paket =
// package), the layer pattern for 16 layers, and HugeBuf::trim(), which
// gives back the unused tail of a pool.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/opt/arena_build.h.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md.
// arena_build.h - integration loader: reads weights.bin (SPEC section 4 via
// weights_io) and repacks every quantized matmul into the qmat.h arenas laid
// out CONTIGUOUSLY IN EXACT PER-TOKEN CONSUMPTION ORDER, so the whole weight
// stream is one big sequential read per token (hardware prefetch friendly;
// the kernels' prefetcht0 +2KB runs across arena boundaries into the next
// matmul's stream, which is exactly what we want).
//
// Stream order (must equal the call order in model_opt.cpp):
//   [prior fold|cols int4]                                  (sparse, raw u8)
//   per layer l = 0..11:
//     kimi:  [qp][kp][vp][fg_up][fg_down][og_up][og_down]
//            [beta_w f32][conv_q f32][conv_k f32][conv_v f32][dt_bias][gn_w]
//            [out_proj]
//     van:   [qp][kp][vp][out_proj]
//     both:  [mlp.up][mlp.down fold|cols int4]
//   [unembed]
// The fp32 KDA weights sit inside the stream at their consumption point.
// The pool has QMAT_TAIL_SLACK mapped past the last arena (prefetch overrun),
// then the normed token-embedding table (random-access, not part of the
// stream). RoPE tables (32 MB, 256 B/token) live in separate heap vectors.
#pragma once

#include "../../port_cfg.h"
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kda.h"
#include "qmat.h"
#include "../../cold_size.h"
namespace tf_kimi_pattern {
struct Pat {
  bool v[16];
  constexpr Pat() : v() {
    for (int i = 0; i < 16; i++) v[i] = (i % 4) != 3;
  }
};
inline constexpr Pat kPat{};
}

namespace fx2 {
namespace opt {

// mmap'd, hugepage-advised (MADV_HUGEPAGE), zero-initialized buffer
struct HugeBuf {
  uint8_t* p = nullptr;
  size_t bytes = 0;
  void alloc(size_t n);
  void trim(size_t keep);
  ~HugeBuf();
  HugeBuf() = default;
  HugeBuf(const HugeBuf&) = delete;
  HugeBuf& operator=(const HugeBuf&) = delete;
};

// one dense matmul site: arena descriptor + this site's activation scale
struct QSite {
  QDense m;
  float s_act = 0.0f;
};

struct KimiArenas {
  QSite qp, kp, vp;        // 192x192, biased-u8 input
  QSite fg_up, og_up;      // 64x192, biased (consumed via quant_bias epilogue)
  QSite fg_down, og_down;  // 192x64, biased
  QSite op;                // 192x192, biased
  const float* beta_w = nullptr;  // [3][192] fp32 (in the stream)
  KdaWeights kw;                  // conv/dt_bias/gn_w point into the stream
};

struct VanArenas {
  QSite qp, kp, vp, op;  // 192x192, biased
  float sq[3] = {}, sk[3] = {}, sv[3] = {};
  float coef[3] = {};  // 0.125f * sq[h] * sk[h] (same fp order as naive)
};

struct MlpArenas {
  QSite up;                  // 768x192, biased; relu2q epilogue s_next below
  QSparse4 down;             // int4 columns, raw-u8 input (relu^2 output)
  float down_s_act = 0.0f;   // = mlp.down activation scale (relu2q s_next)
};

// Tensors of the side channels (preceding tokens, match byte and length
// class); present only in a container that carries them.
// Both projections are small quantized sites (192 x 32), kept as raw int8 with
// a folded row scale because the arena kernels need d_in >= 64.
struct PaketArenas {
  static constexpr int X = 2;
  static constexpr int NSENT = 8;
  static constexpr int DSLOT = 16;
  static constexpr int NMATCH = 209;
  static constexpr int NLEN = 16;       // 16 length classes
  static constexpr int DIN_LAST = X * DSLOT;             // 32
  static constexpr int DIN_MATCH = 2 * DSLOT;            // 32

  bool present = false;
  int emb_rows = 208;         // V (plain) or V + NSENT (with side channels)
  // Embedding rows without norm (q * row scale).  The norm runs per step after
  // the two projection terms have been added, as in the training model.
  const float* emb_raw = nullptr;

  bool has_last = false;
  const float* last_rows = nullptr;   // X * emb_rows rows of DSLOT, dequantized
  const int8_t* last_q = nullptr;     // D x DIN_LAST, int4 range
  const float* last_fold = nullptr;   // D, = s_act * row scale
  float last_s_act = 0.0f;

  bool has_match = false;
  const float* match_rows = nullptr;  // (NMATCH + NLEN) rows of DSLOT
  const int8_t* match_q = nullptr;    // D x DIN_MATCH
  const float* match_fold = nullptr;  // D
  float match_s_act = 0.0f;
};

struct OptModel {
  COLD_SIZE_ATTR OptModel() = default;
  static constexpr int V = 208, D = 192, NL = 16,
                       DH = 64, NH = 3,
                       DMLP = 768, ROPE_LEN = 131072;
  static constexpr int NKIMI = 12, NVAN = 4,
                       NSKIP = 8;
  static constexpr const bool* KIMI_L = ::tf_kimi_pattern::kPat.v;

  HugeBuf pool;          // weight stream + tok table
  size_t stream_bytes = 0;  // bytes of the per-token weight stream

  QSparse4 prior;        // 192x205, raw-u8 quantized prior
  float prior_s_act = 0.0f;
  KimiArenas kimi[NKIMI];
  VanArenas van[NVAN];
  MlpArenas mlp[NL];
  QSite unembed;         // 205x192 (rows_padded 208), biased

  int layer2kimi[NL] = {};
  int layer2van[NL] = {};
  float rsc[NL] = {}, tec[NL] = {}, skip_w[NSKIP] = {};
  const float* tok_table = nullptr;  // 205x192 normed embedding rows
                                     // (with side channels: rows without norm,
                                     // the per-step row comes from
                                     // paket.emb_raw)
  std::vector<float> rope_sin, rope_cos;  // 131072 x 32 each
  float inv_freq[32] = {};
  PaketArenas paket;
#if PORT_FW
  // FW (Lane C port of Z h3_fw_native_patch_20260924): float copy of the
  // unembedding, int8 weight x bf16 row scale, V x D row-major; read by the
  // fast-weight backward in model_opt.cpp only.
  std::vector<float> unembed_f32;
#endif

  void load(const char* weights_path);
};

}  // namespace opt
}  // namespace fx2
