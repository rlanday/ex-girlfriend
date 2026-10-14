#!/usr/bin/env python3
"""z1: article ordinal and global stream position as a model input

Both numbers are free of charge in S: the decoder counts article separators and
decoded tokens as it goes, so it knows them before it needs the prediction.
What costs is only the projection weight.

Two decisions that matter for the later C++ port, both made here and nowhere
else:

 1. THE PHASE IS AN INTEGER.  The encoding is the usual sine/cosine pair per
    dyadic level, but the argument is not computed in floating point.  With a
    range of 2**M and a table of 2**PHASE_BITS entries, the phase index of
    level k is a WINDOW OF BITS of the value:

        shift = M - PHASE_BITS - k
        index = (v >> shift) & (T - 1)          for shift >= 0
        index = (v << -shift) & (T - 1)          for shift <  0

    so encoder and decoder agree bit for bit on every machine and every libm.
    A `sin()` in the coder would not: the jury builds from source, and two
    libm versions that differ in the last ulp produce two different archives.

 2. THE FEATURE RIDES IN `input_tokens`.  His `train()` passes exactly five
    tensors from the loader to the model, so an extra tensor would mean
    copying his training loop.  Instead the loader packs

        packed = token | article_ordinal << 8 | global_position << 28

    into the int64 `input_tokens` and `zmodel.ZEmbedding` unpacks it in the one
    place that reads it (poc4m has `value_embedding_pattern=None`, so the token
    embedding is the ONLY consumer; checked, not assumed).

Bucket embedding variant (documented, not built): replace the sine table by a
learned table of `n_articles / 1000` rows, indexed by ordinal // 1000, and drop
the projection.  Costs ~40-50 KB x 2 instead of ~12 KB x 2, needs no
transcendental at all and is therefore the safer form for the coder.  Take it
only if the screening says the signal is there and the sine form turns out to
be the bottleneck.

    python -m zctx.zfeat selftest
"""

from __future__ import annotations

import sys

import numpy as np

# --- geometry of the encoding -------------------------------------------------
PHASE_BITS = 8                    # sine table has 2**PHASE_BITS entries
PHASE_TABLE_SIZE = 1 << PHASE_BITS

ARTICLE_BITS = 20                 # 2**20 = 1048576 >= 243427 articles of enwik9
ARTICLE_LEVELS = 16               # finest wavelength 2**20 / 2**16 = 16 articles

POSITION_BITS = 30                # 2**30 = 1073741824 >= 601.5 M tokens
POSITION_LEVELS = 16              # finest wavelength 2**30 / 2**16 = 32768 tokens

N_FEATURES = 2 * (ARTICLE_LEVELS + POSITION_LEVELS)     # 64

# --- packing into input_tokens -------------------------------------------------
TOKEN_BITS = 8
ARTICLE_SHIFT = TOKEN_BITS                     # 8
POSITION_SHIFT = ARTICLE_SHIFT + ARTICLE_BITS  # 28
TOKEN_MASK = (1 << TOKEN_BITS) - 1
ARTICLE_MASK = (1 << ARTICLE_BITS) - 1
POSITION_MASK = (1 << POSITION_BITS) - 1
# 8 + 20 + 30 = 58 bits, comfortably inside a signed int64


def phase_table() -> np.ndarray:
    """(T, 2) float32: column 0 is sin, column 1 is cos of 2*pi*i/T

    This table is the only floating point in the encoding.  In the coder it is
    a constant of the weights container (T = 256 -> 2 KB fp32, 1 KB fp16), not
    something recomputed, so both directions see identical numbers.
    """
    i = np.arange(PHASE_TABLE_SIZE, dtype=np.float64)
    angle = 2.0 * np.pi * i / PHASE_TABLE_SIZE
    return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def phase_indices(values: np.ndarray, value_bits: int, levels: int) -> np.ndarray:
    """(..., levels) int64 phase indices: pure integer bit windows"""
    v = np.asarray(values, dtype=np.int64)
    out = np.empty(v.shape + (levels,), dtype=np.int64)
    for k in range(levels):
        shift = value_bits - PHASE_BITS - k
        if shift >= 0:
            out[..., k] = (v >> shift) & (PHASE_TABLE_SIZE - 1)
        else:
            out[..., k] = (v << (-shift)) & (PHASE_TABLE_SIZE - 1)
    return out


def features(article_ordinal: np.ndarray, global_position: np.ndarray,
             table: np.ndarray | None = None) -> np.ndarray:
    """(..., N_FEATURES) float32, layout [art sin/cos per level | pos sin/cos]"""
    if table is None:
        table = phase_table()
    a = table[phase_indices(article_ordinal, ARTICLE_BITS, ARTICLE_LEVELS)]
    p = table[phase_indices(global_position, POSITION_BITS, POSITION_LEVELS)]
    shape = np.asarray(article_ordinal).shape
    return np.concatenate(
        [a.reshape(shape + (2 * ARTICLE_LEVELS,)), p.reshape(shape + (2 * POSITION_LEVELS,))],
        axis=-1,
    )


def pack(tokens: np.ndarray, article_ordinal: np.ndarray, global_position: np.ndarray) -> np.ndarray:
    """int64 carrier: token in the low 8 bits, ordinal next, position on top"""
    t = np.asarray(tokens, dtype=np.int64)
    a = np.asarray(article_ordinal, dtype=np.int64)
    p = np.asarray(global_position, dtype=np.int64)
    assert (t >= 0).all() and (t <= TOKEN_MASK).all()
    assert (a >= 0).all() and (a <= ARTICLE_MASK).all(), (
        "article ordinal does not fit in %d bits; raise ARTICLE_BITS" % ARTICLE_BITS
    )
    assert (p >= 0).all() and (p <= POSITION_MASK).all(), (
        "stream position does not fit in %d bits; raise POSITION_BITS" % POSITION_BITS
    )
    return t | (a << ARTICLE_SHIFT) | (p << POSITION_SHIFT)


def unpack(packed: np.ndarray):
    q = np.asarray(packed, dtype=np.int64)
    return (
        q & TOKEN_MASK,
        (q >> ARTICLE_SHIFT) & ARTICLE_MASK,
        (q >> POSITION_SHIFT) & POSITION_MASK,
    )


def projection_parameters(d_model: int) -> int:
    """weights of the z1 injection: the whole price of arm z1 in S/2"""
    return N_FEATURES * d_model


def selftest() -> int:
    rng = np.random.default_rng(0)
    table = phase_table()
    assert table.shape == (PHASE_TABLE_SIZE, 2)
    assert abs(float(table[0, 0])) < 1e-7 and abs(float(table[0, 1]) - 1.0) < 1e-7

    tok = rng.integers(0, 208, size=10000)
    art = rng.integers(0, 243427, size=10000)
    pos = rng.integers(0, 601_500_000, size=10000)
    packed = pack(tok, art, pos)
    t2, a2, p2 = unpack(packed)
    assert (t2 == tok).all() and (a2 == art).all() and (p2 == pos).all()
    assert packed.dtype == np.int64 and (packed >= 0).all()

    f = features(art, pos, table)
    assert f.shape == (10000, N_FEATURES) and f.dtype == np.float32
    assert np.isfinite(f).all() and float(np.abs(f).max()) <= 1.0

    # the level 0 phase of the article ordinal is its top PHASE_BITS bits
    idx = phase_indices(art, ARTICLE_BITS, ARTICLE_LEVELS)
    assert (idx[:, 0] == (art >> (ARTICLE_BITS - PHASE_BITS))).all()
    # neighbouring ordinals must not collide on every level at once
    d = features(np.array([1000, 1001]), np.array([0, 0]), table)
    assert not np.allclose(d[0], d[1]), "the encoding cannot tell two articles apart"

    print("phase table          %d entries" % PHASE_TABLE_SIZE)
    print("features             %d (article %d levels, position %d levels)"
          % (N_FEATURES, ARTICLE_LEVELS, POSITION_LEVELS))
    print("packing              token %d bits, ordinal %d, position %d = %d bits"
          % (TOKEN_BITS, ARTICLE_BITS, POSITION_BITS, TOKEN_BITS + ARTICLE_BITS + POSITION_BITS))
    print("projection @192      %d parameters" % projection_parameters(192))
    print("\nOK: zfeat self test")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if (len(sys.argv) > 1 and sys.argv[1] == "selftest") else (print(__doc__) or 2))
