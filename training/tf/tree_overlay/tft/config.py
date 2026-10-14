"""TransformerConfig factory for the transformer proof of concept.

Derived from Vladimir Ivanov's training_recipes/{train_6m,quantize,cooldown}.py
and pysrc/export_weights.py (fx2-cmix-transformer, GPL-3).  Every field keeps
his value; what this module adds is

  * the vocabulary size as a PARAMETER (his recipes hard code 205, ours is 208),
  * the geometry as a named choice instead of six copies of the same literal,
  * the three training phases as one function instead of three near-identical
    600-line recipes.

The geometries live in tft/param_count.py so that they can be sized without
torch; this module only turns one of them into a TransformerConfig.

Every width is a multiple of 16 (enforced in param_count.Geometry).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch

from pysrc.model import MatmulQuantization, TransformerConfig
from pysrc.quantization import Quantization

from tft.param_count import GEOMETRIES, Geometry

Phase = Literal["bf16", "qat", "fp32"]

# Our coder stream with its frozen 208-value byte set
# (top-level README.md, section 4.6).
# His stream has 205 distinct bytes, ours has 208; the only number that has to
# change between his recipes and ours.
VOCABULARY_SIZE = int(os.environ.get("TFT_VOCAB", "208"))

# His loader splits articles at 2**17 tokens and his rope table has exactly
# that many positions; predictor.cpp:kMaxArticleTokens repeats the constant.
# Do not change one without the other two.
MAX_ARTICLE_TOKENS = 1 << 17

# His per-GPU micro batch on the 8x5090 machine.  Keeping it fixed and moving
# the gradient accumulation instead is his own recommendation for running on
# fewer GPUs ("Divide the batch size and multiply the number of gradient
# accumulation steps by N_GPUS / 8. This will reproduce the behavior exactly.").
# The micro batch is 2^18.  Every document then fits, and the shortening
# rule of the loader (drop A2, then the sentinels, then the prompt) no longer
# applies; the coder could not reproduce it causally (the total length is not
# known yet when the prompt is fed), and on pkgvoll_v2 it affected 14 of
# 243,426 articles.  The optimizer batch stays at his 2^20: the gradient
# accumulation is computed from both numbers instead of from a fixed
# reference GPU count (with 2^18 and 1 GPU that is 4 steps).
MICRO_BATCH_TOKENS = int(os.environ.get("TFT_MICRO_BATCH_TOKENS", 1 << 18))
OPTIMIZER_BATCH_TOKENS = int(os.environ.get("TFT_OPTIMIZER_BATCH_TOKENS", 1 << 20))
REFERENCE_WORLD_SIZE = 8

# The position limit (rotary table) is not the same number as the piece limit.
# The piece limit 2^17 is held in three places (here, `pysrc/data.py`,
# `predictor.cpp:kMaxArticleTokens`) and counts coded tokens in the coder; the
# number of positions of a document is coded plus inserted rows and lies above
# it.  Enlarging the micro batch without separating the two creates documents
# whose positions exceed the rotary table.  Default: the piece limit.
POSITION_LIMIT = int(os.environ.get("TFT_POSITION_LIMIT", MAX_ARTICLE_TOKENS))


def gradient_accumulation_steps(world_size: int) -> int:
    """keep the optimizer batch at his 2**20 tokens on any GPU count"""
    # computed from micro batch and optimizer batch, so that both can be set
    # independently
    gesamt = MICRO_BATCH_TOKENS * world_size
    assert OPTIMIZER_BATCH_TOKENS % gesamt == 0, (
        "optimizer batch %d is not a multiple of micro batch %d x %d GPUs"
        % (OPTIMIZER_BATCH_TOKENS, MICRO_BATCH_TOKENS, world_size)
    )
    return OPTIMIZER_BATCH_TOKENS // gesamt


@dataclass(frozen=True)
class PhaseSchedule:
    """the four numbers his recipes vary between phases"""

    epochs: int
    warmup_epochs: int
    learning_rate: float
    warmup_learning_rate_multiplier: float
    final_learning_rate_multiplier: float
    initialize_scales_steps: int | None


# His shipped schedules, for reference and as the shape our PoC shortens.
# NOTE writeup.md says 320 / 256 / 16 epochs; the shipped quantize.py says 96
# (2026-07-24 tree) or 128 (2026-08-21 tree).  The published recipes are
# therefore not exactly what produced models/6m-q4-fp32.tch.
IVANOV_SCHEDULES: dict[str, PhaseSchedule] = {
    "bf16": PhaseSchedule(
        epochs=256 + 64,
        warmup_epochs=64,
        learning_rate=5e-4,
        warmup_learning_rate_multiplier=8,
        final_learning_rate_multiplier=0.025,
        initialize_scales_steps=None,
    ),
    "qat": PhaseSchedule(
        epochs=96,
        warmup_epochs=32,
        learning_rate=2.5e-6,
        warmup_learning_rate_multiplier=25,
        final_learning_rate_multiplier=0.0,
        initialize_scales_steps=256,
    ),
    "fp32": PhaseSchedule(
        epochs=16,
        warmup_epochs=0,
        learning_rate=1e-7,
        warmup_learning_rate_multiplier=1,
        final_learning_rate_multiplier=1e-8,
        initialize_scales_steps=None,
    ),
}


# Warmup FRACTION of each phase.  Not his long-run numbers (64/320 = 0.20) but
# his own SHORT-run numbers, because a short schedule is what we are running:
#   bf16: commit 93edd6b, training_recipes/train_4m.py, "less epochs":
#         warmup_epochs=16, epochs=24  ->  0.667
#   qat:  the 2026-08-21 tree, training_recipes/quantize.py: 128 / 128  ->  1.0
#         (commit f1c95f3: "changed config ... to what we actually used")
#   fp32: his cooldown.py has no warmup at all  ->  0.0
WARMUP_FRACTION = {"bf16": 16 / 24, "qat": 1.0, "fp32": 0.0}

# Epoch defaults for a ~12 h night on 1x RTX 5090 with the poc4m geometry on a
# 200M token window, keeping his 320:128:16 ratio between the phases.
# Derived from his published 8x5090 / 26 h; a plan, not a measurement
# (`python -m tft.calibrate` replaces it with measured numbers).
DEFAULT_EPOCHS = {"bf16": 55, "qat": 22, "fp32": 3}


def poc_schedule(phase: Phase) -> PhaseSchedule:
    """his schedule with the epoch counts taken from the environment

    The epoch counts and the warmup fraction are the only things the PoC
    changes.  Learning rates, warmup shape and scale initialisation stay his,
    because a shortened schedule with changed learning rates would confound
    "the recipe does not carry" with "the recipe was not followed".

    Override per run, e.g.
        TFT_EPOCHS_BF16=48 TFT_EPOCHS_QAT=8 TFT_EPOCHS_FP32=2
    """
    base = IVANOV_SCHEDULES[phase]
    env = {"bf16": "TFT_EPOCHS_BF16", "qat": "TFT_EPOCHS_QAT", "fp32": "TFT_EPOCHS_FP32"}
    epochs = int(os.environ.get(env[phase], str(DEFAULT_EPOCHS[phase])))
    assert epochs >= 1
    fraction = float(os.environ.get("TFT_WARMUP_FRACTION", WARMUP_FRACTION[phase]))
    warmup = min(epochs, round(fraction * epochs))
    scale_steps = base.initialize_scales_steps
    if scale_steps is not None:
        scale_steps = int(os.environ.get("TFT_SCALE_STEPS", str(scale_steps)))
    return PhaseSchedule(
        epochs=epochs,
        warmup_epochs=warmup,
        learning_rate=base.learning_rate,
        warmup_learning_rate_multiplier=base.warmup_learning_rate_multiplier,
        final_learning_rate_multiplier=base.final_learning_rate_multiplier,
        initialize_scales_steps=scale_steps,
    )


def make_quantizations() -> tuple[Quantization, Quantization, MatmulQuantization]:
    """his int4 weights (15 buckets, -7..7) and int8 activations, verbatim"""
    weight_quantization = Quantization(
        buckets=15,
        block_size=None,
        scale_dtype=torch.bfloat16,
        arithmetic_dtype=torch.float32,
        scale_init="scaled_max",
        epsilon=None,
    )
    activation_quantization = Quantization(
        buckets=256,
        block_size=None,
        scale_dtype=torch.bfloat16,
        arithmetic_dtype=torch.float32,
        scale_init="scaled_mean",
        epsilon=None,
    )
    matmul_quantization = MatmulQuantization(
        weight=weight_quantization, activation=activation_quantization
    )
    return weight_quantization, activation_quantization, matmul_quantization


def geometry(name: str | None = None) -> Geometry:
    """the PoC geometry; default poc4m = HIS OWN 4m (commit 93edd6b) on our
    vocabulary, at his final window 1024 instead of the 2048 that config used
    (he rejected 2048 for the 6m because it makes CPU inference too slow; the
    PoC has no reason to inherit a window he discarded).  Override with
    TFT_GEOMETRY=poc2m|poc37m|6m208."""
    return GEOMETRIES[name or os.environ.get("TFT_GEOMETRY", "poc4m")]


def make_config(
    phase: Phase,
    geometry_name: str | None = None,
    vocabulary_size: int | None = None,
    variable_sequence_length: bool = True,
) -> TransformerConfig:
    """one TransformerConfig for one training phase

    phase "bf16": his train_6m.py:   no quantization, bf16 activations,
                                     flash_attention_2, torch.compile safe
    phase "qat":  his quantize.py:   int4 weights / int8 activations, bf16
                                     activations, gradient checkpointing
    phase "fp32": his cooldown.py:   same quantization, fp32 activations,
                                     flex_attention (the export reference)
    """
    g = geometry(geometry_name)
    V = vocabulary_size if vocabulary_size is not None else VOCABULARY_SIZE
    assert V == g.vocabulary_size, (
        "geometry " + g.name + " is sized for vocabulary " + str(g.vocabulary_size)
        + " but " + str(V) + " was requested; add a geometry instead of overriding"
    )

    quantized = phase in ("qat", "fp32")
    weight_q, activation_q, matmul_q = make_quantizations()

    return TransformerConfig(
        vocabulary_size=V,
        prior_embedding=True,
        prior_logit_mixing=False,
        max_variable_sequence_length=POSITION_LIMIT
        if variable_sequence_length
        else None,
        n_layers=g.n_layers,
        attention_weight_sharing_pattern=list(range(g.n_layers)),
        mlp_weight_sharing_pattern=list(range(g.n_layers)),
        d_model=g.d_model,
        kimi_linear=list(g.kimi_linear),
        window_size=g.window_size,
        window_size_multipliers=None,
        n_query_heads=g.n_query_heads,
        n_key_value_heads=g.n_key_value_heads,
        d_head=g.d_head,
        d_mla=None,
        rope_base=g.rope_base,
        half_truncate_rope=False,
        query_key_norm=True,
        # NOTE: pysrc/export_weights.py:make_reference_config() omits this field
        # and therefore raises TypeError before it exports anything.
        query_key_norm_gain=False,
        kimi_linear_d_head=g.kimi_linear_d_head,
        kimi_linear_n_heads=g.kimi_linear_n_heads,
        kimi_linear_convolution_size=g.kimi_linear_convolution_size,
        value_embedding_pattern=None,
        prior_value_embedding=False,
        attention_scale=None,
        d_mlp=g.d_mlp,
        activation_function="relu_squared",
        embedding_norm=True,
        prior_embedding_norm=True,
        skip_connections=True,
        token_embedding_connections=True,
        prior_embedding_connections=False,
        dropout=0.0,
        logit_softcap=15.0,
        prior_logprob_cap=-12.0,
        prior_embedding_on_logprobs=False,
        gradient_checkpointing=phase != "bf16",
        embedding_init="normal",
        prior_embedding_init="normal",
        up_init="kaiming_uniform",
        down_init="kaiming_uniform",
        unembedding_init="kaiming_uniform",
        convolution_init="kaiming_uniform",
        logit_mixing_head_init="zero",
        activation_dtype=torch.float32 if phase == "fp32" else torch.bfloat16,
        attention_dtype=torch.float32 if phase == "fp32" else torch.bfloat16,
        logit_dtype=torch.float32,
        weight_dtype=torch.float32,
        mlp_quantization=matmul_q if quantized else None,
        attention_full_rank_quantization=matmul_q if quantized else None,
        attention_low_rank_quantization=matmul_q if quantized else None,
        qkv_quantization=activation_q if quantized else None,
        kimi_linear_full_rank_quantization=matmul_q if quantized else None,
        kimi_linear_low_rank_quantization=matmul_q if quantized else None,
        kimi_linear_beta_projection_quantization=None,
        kimi_linear_convolution_quantization=None,
        embedding_quantization=weight_q if quantized else None,
        prior_embedding_quantization=matmul_q if quantized else None,
        unembedding_quantization=matmul_q if quantized else None,
        logit_mixing_head_quantization=None,
        adamw_quantization_scales=True,
        attention_implementation="flex_attention"
        if phase == "fp32"
        else "flash_attention_2",
    )


def data_paths() -> dict:
    """the four files of the frozen training-data interface

    His make_training_data.sh writes the first three; the vocabulary bitmap is
    ours.  Override the directory with TFT_DATA_DIR.
    """
    d = os.environ.get("TFT_DATA_DIR", "data")
    return {
        "tokens_filename": d + "/ppmd-bytes.uint8",
        "priors_filename": d + "/ppmd-probs.float16",
        "article_boundaries_filename": d + "/article-boundaries.int32",
        "vocab_bitmap_filename": d + "/vocab-bitmap.u8",
    }


def model_path(phase: Phase, geometry_name: str | None = None) -> str:
    g = geometry(geometry_name)
    suffix = {"bf16": "", "qat": "-q4", "fp32": "-q4-fp32"}[phase]
    # Runs with another ZCTX_ARM write next to, not over, the reference
    # checkpoints.
    arm = os.environ.get("ZCTX_ARM", "none")
    tag = "" if arm == "none" else "-" + arm
    return "models/" + g.name + tag + suffix + ".tch"
