// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/weights_io.h.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0.
// weights_io: loader for cpp_infer/data/weights.bin (SPEC.md section 4)
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "tf1_config.h"

namespace fx2 {

enum WDtype : uint8_t { DT_I8 = 0, DT_BF16 = 1, DT_F32 = 2, DT_I32 = 3 };

// one decoded tensor; the data buffer belongs to WeightsFile
struct WTensor {
  uint8_t dtype;
  uint8_t ndim;
  uint32_t shape[8];
  size_t numel;
  uint8_t* data;

  const int8_t* i8() const { return reinterpret_cast<const int8_t*>(data); }
  const uint16_t* bf16_bits() const { return reinterpret_cast<const uint16_t*>(data); }
  const float* f32() const { return reinterpret_cast<const float*>(data); }
  const int32_t* i32() const { return reinterpret_cast<const int32_t*>(data); }
};

struct WeightsFile {
  WTensor* t = nullptr;

  static WeightsFile load_compressed(const char* path);

  const WTensor& T(int id, int blk = -1) const { return t[id * (16 + 1) + blk + 1]; }
  WTensor& T(int id, int blk = -1) { return t[id * (16 + 1) + blk + 1]; }
  // the data buffer of a tensor passes to the caller (freed by it)
  uint8_t* Nimm(int id, int blk = -1) {
    uint8_t* p = T(id, blk).data;
    T(id, blk).data = nullptr;
    return p;
  }

  WeightsFile() = default;
  WeightsFile(WeightsFile&& o) : t(o.t) { o.t = nullptr; }
  WeightsFile(const WeightsFile&) = delete;
  WeightsFile& operator=(const WeightsFile&) = delete;
  ~WeightsFile();
};

// fp32 value of raw bfloat16 bits (bits << 16 reinterpreted as float)
inline float bf16_to_f32(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

}  // namespace fx2

#include <type_traits>
static_assert(0xaec368ac6a568632ULL == 0xaec368ac6a568632ULL,
              "the stamped name list comes from another container than the stamped weights");
namespace fx2 {
namespace namen {
// The stamped canonical names in order of first appearance; read by the
// compiler only (TID below), never in the image.
static constexpr const char* kNamen[] = {"embedding.weight.q", "embedding.weight.scale", "embedding.last_table.weight.q", "embedding.last_table.weight.scale", "embedding.last_projection.weight.q", "embedding.last_projection.weight.scale", "embedding.last_projection.quantize_activation.scale", "embedding.match_table.weight.q", "embedding.match_table.weight.scale", "embedding.match_projection.weight.q", "embedding.match_projection.weight.scale", "embedding.match_projection.quantize_activation.scale", "value_embeddings.0.inner.weight.q", "value_embeddings.0.inner.weight.scale", "value_embeddings.1.inner.weight.q", "value_embeddings.1.inner.weight.scale", "value_embeddings.2.inner.weight.q", "value_embeddings.2.inner.weight.scale", "value_embeddings.3.inner.weight.q", "value_embeddings.3.inner.weight.scale", "prior_embedding.weight.q", "prior_embedding.weight.scale", "prior_embedding.quantize_activation.scale", "blocks.#.attention.log_baseline_decay_rate", "blocks.#.attention.dt_bias", "blocks.#.attention.query_projection.weight.q", "blocks.#.attention.query_projection.weight.scale", "blocks.#.attention.query_projection.quantize_activation.scale", "blocks.#.attention.query_convolution.weight", "blocks.#.attention.key_projection.weight.q", "blocks.#.attention.key_projection.weight.scale", "blocks.#.attention.key_projection.quantize_activation.scale", "blocks.#.attention.key_convolution.weight", "blocks.#.attention.value_projection.weight.q", "blocks.#.attention.value_projection.weight.scale", "blocks.#.attention.value_projection.quantize_activation.scale", "blocks.#.attention.value_convolution.weight", "blocks.#.attention.forget_gate_projection.up.weight.q", "blocks.#.attention.forget_gate_projection.up.weight.scale", "blocks.#.attention.forget_gate_projection.up.quantize_activation.scale", "blocks.#.attention.forget_gate_projection.down.weight.q", "blocks.#.attention.forget_gate_projection.down.weight.scale", "blocks.#.attention.forget_gate_projection.down.quantize_activation.scale", "blocks.#.attention.beta_projection.weight", "blocks.#.attention.output_gate_projection.up.weight.q", "blocks.#.attention.output_gate_projection.up.weight.scale", "blocks.#.attention.output_gate_projection.up.quantize_activation.scale", "blocks.#.attention.output_gate_projection.down.weight.q", "blocks.#.attention.output_gate_projection.down.weight.scale", "blocks.#.attention.output_gate_projection.down.quantize_activation.scale", "blocks.#.attention.output_fused_norm_gate.weight", "blocks.#.attention.output_projection.weight.q", "blocks.#.attention.output_projection.weight.scale", "blocks.#.attention.output_projection.quantize_activation.scale", "blocks.#.mlp.up.weight.q", "blocks.#.mlp.up.weight.scale", "blocks.#.mlp.up.quantize_activation.scale", "blocks.#.mlp.down.weight.q", "blocks.#.mlp.down.weight.scale", "blocks.#.mlp.down.quantize_activation.scale", "blocks.#.q_mix.w", "blocks.#.q_mlp.w", "blocks.#.attention.quantize_queries.scale", "blocks.#.attention.quantize_keys.scale", "blocks.#.attention.quantize_values.scale", "blocks.#.attention.value_embedding_coefficients.value", "blocks.#.q_final.w", "unembedding.weight.q", "unembedding.weight.scale", "unembedding.quantize_activation.scale", "logit_mixing_head.weight", "rope.inv_freq", "rope.sin", "rope.cos", "config.ints", "config.kimi"};
static constexpr int kN = static_cast<int>(sizeof(kNamen) / sizeof(kNamen[0]));
static_assert(kN == 76, "the stamped name list and its count disagree");
constexpr bool Gleich(const char* a, const char* b) {
  while (*a && *a == *b) ++a, ++b;
  return *a == *b;
}
int NichtInDerListe();   // not constexpr: a name that is not listed stops the compiler
constexpr int Index(const char* s) {
  for (int i = 0; i < kN; ++i)
    if (Gleich(kNamen[i], s)) return i;
  return -1;
}
constexpr int IndexPflicht(const char* s) { return Index(s) >= 0 ? Index(s) : NichtInDerListe(); }
// FNV-1a-64 over the names, each with its terminating zero (the decoder folds
// the same bytes in at the first appearance of a canonical name)
constexpr unsigned long long Fnv() {
  unsigned long long h = 14695981039346656037ULL;
  for (int i = 0; i < kN; ++i) {
    const char* s = kNamen[i];
    do {
      h ^= static_cast<unsigned char>(*s);
      h *= 1099511628211ULL;
    } while (*s++);
  }
  return h;
}
}  // namespace namen
}  // namespace fx2
// index of a canonical name ("blocks.#." for a layer tensor), a compile-time constant
#define TID(s) (std::integral_constant<int, ::fx2::namen::IndexPflicht(s)>::value)
// the same, -1 if the container has no such tensor
#define TID_OPT(s) (std::integral_constant<int, ::fx2::namen::Index(s)>::value)
