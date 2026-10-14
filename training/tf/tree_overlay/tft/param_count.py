#!/usr/bin/env python3
"""Analytic parameter count for the fx2-cmix-transformer geometry.

Pure Python: no torch, no GPU, no import of pysrc.  Counts every Parameter
that pysrc/model.py constructs for a given TransformerConfig, so a candidate
geometry can be sized (and checked against Ivanov's 6M reference) on any
machine, including this project's Windows box.

Reference check: the shipped 6m geometry (V=205, D=192, L=12, d_mlp=768,
kimi=[T,T,T,F]*3) must come out near 5.9e6 total parameters, which is the
"6 million parameters" of writeup.md.

Usage:
    python scripts/param_count.py                 # table of all known geometries
    python scripts/param_count.py --json <key>    # one geometry as json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from math import log2


@dataclass(frozen=True)
class Geometry:
    """the subset of TransformerConfig that determines the parameter count"""

    name: str
    vocabulary_size: int
    d_model: int
    n_layers: int
    d_head: int
    n_query_heads: int
    n_key_value_heads: int
    d_mlp: int
    kimi_linear: list
    kimi_linear_d_head: int
    kimi_linear_n_heads: int
    kimi_linear_convolution_size: int
    window_size: int
    rope_base: int = 10_000
    prior_embedding: bool = True
    token_embedding_connections: bool = True
    prior_embedding_connections: bool = False
    skip_connections: bool = True
    quantized: bool = True  # count the per-output-row quantization scales
    note: str = ""

    def __post_init__(self) -> None:
        # the asserts of TransformerConfig.__post_init__ that matter here
        assert self.n_query_heads * self.d_head == self.d_model, (
            self.name + ": n_query_heads * d_head != d_model"
        )
        assert self.n_query_heads % self.n_key_value_heads == 0
        assert self.kimi_linear_n_heads * self.kimi_linear_d_head == self.d_model, (
            self.name + ": kimi_linear_n_heads * kimi_linear_d_head != d_model"
        )
        assert len(self.kimi_linear) == self.n_layers, (
            self.name + ": len(kimi_linear) != n_layers "
            "(holds only for the identity weight sharing pattern used here)"
        )
        # Every cell count is a multiple of 16 (the 16-divisibility
        # rule; the transformer widths are vectorized the same way)
        assert self.d_model % 16 == 0, self.name + ": d_model not a multiple of 16"
        assert self.d_mlp % 16 == 0, self.name + ": d_mlp not a multiple of 16"


@dataclass
class Counts:
    parts: dict = field(default_factory=dict)

    def add(self, name: str, n: int) -> None:
        self.parts[name] = self.parts.get(name, 0) + n

    @property
    def total(self) -> int:
        return sum(self.parts.values())


def count(g: Geometry) -> Counts:
    c = Counts()
    D, V = g.d_model, g.vocabulary_size
    n_kimi = sum(1 for k in g.kimi_linear if k)
    n_attn = g.n_layers - n_kimi
    kdim = g.kimi_linear_n_heads * g.kimi_linear_d_head  # == d_model
    rank = g.kimi_linear_d_head  # LowRankCastedLinear rank inside KimiLinearAttention

    # --- embeddings: CastedEmbedding weight (V, D), CastedLinear weight (out, in)
    c.add("embedding", V * D)
    if g.prior_embedding:
        c.add("prior_embedding", D * V)
    c.add("unembedding", V * D)

    # --- per-block scalars (CastedConstant)
    if g.token_embedding_connections or g.prior_embedding_connections:
        c.add("block.residual_stream_coefficient", g.n_layers)
    if g.token_embedding_connections:
        c.add("block.token_embedding_coefficient", g.n_layers)
    if g.prior_embedding_connections:
        c.add("block.prior_embedding_coefficient", g.n_layers)
    if g.skip_connections:
        c.add("skip_connection_weights", g.n_layers // 2)

    # --- MLP: up (d_mlp, D) + down (D, d_mlp)
    c.add("mlp", g.n_layers * 2 * D * g.d_mlp)

    # --- full attention: q, k, v, out
    per_attn = (
        g.n_query_heads * g.d_head * D
        + g.n_key_value_heads * g.d_head * D
        + g.n_key_value_heads * g.d_head * D
        + D * g.n_query_heads * g.d_head
    )
    c.add("attention", n_attn * per_attn)

    # --- kimi linear attention
    per_kimi = (
        kdim * D  # query_projection
        + kdim * D  # key_projection
        + kdim * D  # value_projection
        + D * kdim  # output_projection
        + 3 * kdim * g.kimi_linear_convolution_size  # q/k/v convolutions
        + (D * rank + rank * kdim)  # forget_gate_projection (low rank)
        + (D * rank + rank * kdim)  # output_gate_projection (low rank)
        + g.kimi_linear_n_heads * D  # beta_projection
        + g.kimi_linear_n_heads  # log_baseline_decay_rate
        + kdim  # dt_bias
        + g.kimi_linear_d_head  # output_fused_norm_gate.weight (fla)
    )
    c.add("kimi_linear", n_kimi * per_kimi)

    if g.quantized:
        # Quantize.scale has one entry per output row (block_size = size[-1])
        scales = 0
        scales += V  # embedding (V, D)
        if g.prior_embedding:
            scales += D  # prior_embedding (D, V)
        scales += V  # unembedding (V, D)
        scales += g.n_layers * (g.d_mlp + D)  # mlp up + down
        scales += n_attn * (
            g.n_query_heads * g.d_head + 2 * g.n_key_value_heads * g.d_head + D
        )
        scales += n_kimi * (
            4 * kdim  # q/k/v/out projections
            + 3 * kdim  # convolutions (weight (d, kernel) -> d rows)
            + 2 * (rank + kdim)  # the two low rank pairs
        )
        c.add("weight_quantization_scales", scales)
        # activation scales: one Quantize per quantized matmul input plus the
        # per-head qkv quantizers; tiny, counted for completeness
        act = g.n_layers * 2
        act += n_attn * (4 + 3)
        act += n_kimi * 8
        act += 2
        c.add("activation_scales_approx", act)

    return c


def weights_bin_bytes(c: Counts) -> int:
    """rough size of the tfwc2 container: int4 weights at log2(15) bits, the
    rest (bf16 scales, fp32 tensors) at 16 bits"""
    int4_parts = (
        "embedding",
        "prior_embedding",
        "unembedding",
        "mlp",
        "attention",
        "kimi_linear",
    )
    n_int4 = sum(v for k, v in c.parts.items() if k in int4_parts)
    n_fp = c.total - n_int4
    return int((n_int4 * log2(15) + n_fp * 16) / 8)


REFERENCE_6M = Geometry(
    name="6m",
    vocabulary_size=205,
    d_model=192,
    n_layers=12,
    d_head=64,
    n_query_heads=3,
    n_key_value_heads=3,
    d_mlp=768,
    kimi_linear=[True, True, True, False] * 3,
    kimi_linear_d_head=64,
    kimi_linear_n_heads=3,
    kimi_linear_convolution_size=4,
    window_size=1024,
    note="Ivanov, shipped (training_recipes/train_6m.py)",
)

GEOMETRIES = {}


def _add(g: Geometry) -> Geometry:
    GEOMETRIES[g.name] = g
    return g


_add(REFERENCE_6M)

_add(
    Geometry(
        name="6m208",
        vocabulary_size=208,
        d_model=192,
        n_layers=12,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 3,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1024,
        note="his geometry on our vocabulary; the target once the PoC has a sign",
    )
)

# --- HIS OWN SMALLER GEOMETRIES, recovered from the .git history -------------
# (commit 93edd6b, training_recipes/train_4m.py, wandb name "4m")
# WARNING from the history search: at HEAD the files train_4m.py / train_4m_v2.py
# carry the SIX million configuration: commit 6425829 overwrote them and
# e267262 renamed them to train_6m*.py with similarity index 100 %.  Only the
# middle revisions hold the real small configs.
_add(
    Geometry(
        name="iv4m",
        vocabulary_size=205,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=2048,
        note="Ivanov 93edd6b 'train_4m.py' verbatim (his window there is 2048)",
    )
)

# (commit 93edd6b, training_recipes/train_4m_v2.py, wandb name "3.7m")
_add(
    Geometry(
        name="iv37m",
        vocabulary_size=205,
        d_model=128,
        n_layers=16,
        d_head=64,
        n_query_heads=2,
        n_key_value_heads=2,
        d_mlp=512,
        kimi_linear=[True, True, True, False] * 4,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=2,
        kimi_linear_convolution_size=4,
        window_size=2048,
        note="Ivanov 93edd6b 'train_4m_v2.py' verbatim (deep and narrow)",
    )
)

# --- our PoC: his 4m geometry on our vocabulary, at his FINAL window ----------
_add(
    Geometry(
        name="poc4m",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1024,
        note="primary PoC geometry: his 4m, vocabulary 208, window 1024 (his final choice)",
    )
)

_add(
    Geometry(
        name="poc4m_w1280",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1280,
        note="poc4m with window 1280; only window_size differs, no tensor "
             "shape depends on it, one checkpoint serves both windows.",
    )
)

_add(
    Geometry(
        name="poc4m_w1280",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1280,
        note="poc4m with window 1280; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w1536",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1536,
        note="poc4m with window 1536; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w1280",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1280,
        note="poc4m with window 1280; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w1536",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1536,
        note="poc4m with window 1536; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w2048",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=2048,
        note="poc4m with window 2048; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w3072",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=3072,
        note="poc4m with window 3072; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w4096",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=4096,
        note="poc4m with window 4096; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)
_add(
    Geometry(
        name="poc4m_w5120",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=5120,
        note="poc4m with window 5120 (only window_size differs, no tensor shape)",
    )
)

_add(
    Geometry(
        name="poc4m_w1280",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1280,
        note="poc4m with window 1280; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w1536",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1536,
        note="poc4m with window 1536; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w2048",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=2048,
        note="poc4m with window 2048; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w3072",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=3072,
        note="poc4m with window 3072; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w4096",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=4096,
        note="poc4m with window 4096; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w6144",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=6144,
        note="poc4m with window 6144; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

_add(
    Geometry(
        name="poc4m_w8192",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=8192,
        note="poc4m with window 8192; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)

# --- the depth ladder for the size curve ------------------------------------
# poc2m / poc4m / 6m208 / poc8m are 4 / 8 / 12 / 16 layers of the SAME block:
# d_model 192, d_head 64, 3 heads, d_mlp 768, kimi pattern TTTF (75 %),
# window 1024, vocabulary 208.  Only the depth changes, which is exactly how
# Ivanov's own 6m -> 8m -> 10m sweep varies; a point that changes width AND
# depth AND d_mlp at once (his iv37m) cannot be attributed and is therefore
# NOT part of this ladder.
_add(
    Geometry(
        name="poc8m",
        vocabulary_size=208,
        d_model=192,
        n_layers=16,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 4,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1024,
        note="curve point: his 8m depth (16 layers) on our vocabulary",
    )
)

# --- smaller fallbacks if a night turns out to be too short -------------------
_add(
    Geometry(
        name="poc2m",
        vocabulary_size=208,
        d_model=192,
        n_layers=4,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False],
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1024,
        note="fallback: one repeat of his kimi pattern, every width unchanged",
    )
)

_add(
    Geometry(
        name="poc37m",
        vocabulary_size=208,
        d_model=128,
        n_layers=16,
        d_head=64,
        n_query_heads=2,
        n_key_value_heads=2,
        d_mlp=512,
        kimi_linear=[True, True, True, False] * 4,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=2,
        kimi_linear_convolution_size=4,
        window_size=1024,
        note="his 3.7m on our vocabulary; changes every C++ constexpr, fallback only",
    )
)

# --- window variants of the training chain of the shipped weights ------------
# Copies of 6m208 and poc8m in which only window_size differs; no tensor shape
# depends on it.  poc8m_w2048 is the geometry of the shipped weights.  During
# training these rows were appended to this file when a GPU host was set up;
# here they are in the registry directly, in the same order.
_add(
    Geometry(
        name="6m208_w1536",
        vocabulary_size=208,
        d_model=192,
        n_layers=12,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 3,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1536,
        note="his geometry on our vocabulary; the target once the PoC has a sign",
    )
)

_add(
    Geometry(
        name="poc8m_w1024",
        vocabulary_size=208,
        d_model=192,
        n_layers=16,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 4,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1024,
        note="curve point: his 8m depth (16 layers) on our vocabulary",
    )
)

_add(
    Geometry(
        name="poc8m_w1536",
        vocabulary_size=208,
        d_model=192,
        n_layers=16,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 4,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=1536,
        note="curve point: his 8m depth (16 layers) on our vocabulary",
    )
)

_add(
    Geometry(
        name="poc8m_w2048",
        vocabulary_size=208,
        d_model=192,
        n_layers=16,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 4,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=2048,
        note="curve point: his 8m depth (16 layers) on our vocabulary",
    )
)


def report(g: Geometry) -> dict:
    c = count(g)
    return {
        "name": g.name,
        "vocabulary_size": g.vocabulary_size,
        "d_model": g.d_model,
        "n_layers": g.n_layers,
        "d_mlp": g.d_mlp,
        "kimi_share": str(sum(1 for k in g.kimi_linear if k)) + "/" + str(g.n_layers),
        "window_size": g.window_size,
        "parameters_total": c.total,
        "parameters_million": round(c.total / 1e6, 3),
        "weights_bin_bytes_approx": weights_bin_bytes(c),
        "parts": c.parts,
        "note": g.note,
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--json":
        print(json.dumps(report(GEOMETRIES[sys.argv[2]]), indent=2))
        return

    ref = count(REFERENCE_6M).total
    rows = [(key, report(g)) for key, g in GEOMETRIES.items()]
    width = max(len(k) for k, _ in rows)
    head = (
        "key".ljust(width)
        + "  " + "params".rjust(11)
        + "  " + "M".rjust(7)
        + "  " + "x6m".rjust(6)
        + "  " + "V".rjust(4) + " " + "D".rjust(4) + " " + "L".rjust(3)
        + " " + "dmlp".rjust(5) + " " + "kimi".rjust(6)
        + "  " + "~weights".rjust(10)
    )
    print(head)
    for key, r in rows:
        print(
            key.ljust(width)
            + "  " + format(r["parameters_total"], ",").rjust(11)
            + "  " + format(r["parameters_million"], ".3f").rjust(7)
            + "  " + format(r["parameters_total"] / ref, ".3f").rjust(6)
            + "  " + str(r["vocabulary_size"]).rjust(4)
            + " " + str(r["d_model"]).rjust(4)
            + " " + str(r["n_layers"]).rjust(3)
            + " " + str(r["d_mlp"]).rjust(5)
            + " " + r["kimi_share"].rjust(6)
            + "  " + format(r["weights_bin_bytes_approx"], ",").rjust(10)
        )
    print()
    print("reference check: 6m must land near 5.9e6 ('6 million parameters', writeup.md)")
    print("6m total = " + format(ref, ",") + " = " + format(ref / 1e6, ".3f") + " million")
    print()
    for key, r in rows:
        if key == "poc4m":
            print("poc4m breakdown:")
            for part, n in r["parts"].items():
                print("  " + part.ljust(34) + format(n, ",").rjust(12))


if __name__ == "__main__":
    main()
