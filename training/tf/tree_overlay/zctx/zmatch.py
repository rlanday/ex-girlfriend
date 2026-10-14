#!/usr/bin/env python3
"""arm m1 "matchk": the MATCH model of the compressor as a prior channel of the transformer

Derived from Vladimir Ivanov's pysrc/model.py (fx2-cmix-transformer, GPL-3):
`ZMatchEmbedding` subclasses his `CastedEmbedding`.

Not the mixer OUTPUT (that would be fed back, because the mixer learns from
the transformer's predictions) but its INPUTS: the source models learn from
the byte stream and from nothing else.  Among them the MATCH model is the
strongest candidate, because it is the only one that points at repetitions
arbitrarily far back: its reach is the 60 MB history ring, the attention
window of the transformer is 1024 tokens.

The channel consists of two numbers per position, both from
`tf1_dump --match`:

  match-bytes.uint8   the token the MATCH family predicts for the NEXT
                      position ("hot byte"), or 208 if the byte is not in the
                      208-value set
  match-len.uint8     the RAW length of the confirmed match in BITS (0..255)
                      of the model the byte comes from

Built like z4/zlast, for the same reasons: both numbers travel in the int64
`input_tokens` (the trainer hands the model exactly five tensors; a sixth
would mean copying its training loop).  The embedding below is the ONLY
consumer of `input_tokens` (poc4m has `value_embedding_pattern=None`).

    bit  0.. 7   the token itself
    bit  8..16   the match byte        (9 bits: 0..208, 208 is the mark)
    bit 17..20   the length class      (4 bits: 0..15)

COSTS 0 BYTES.  The decoder has both numbers anyway: it runs the same match
models to predict its next bit.  The only cost is the projection that turns
them into a vector.

CAUSALITY.  The dump writes entry i AFTER byte i has entered the history and
BEFORE byte i+1 is read, the same row convention as ppmd-probs.  This is
checked from outside as well: a dump over N and one over N+K agree byte for
byte in the first N entries.  A look ahead would break that.

WHY THE LENGTH IS QUANTIZED AND THE BYTE IS NOT.  The byte is an identity
(208 equivalent values, no order), the length a quantity with an order whose
information saturates: between 15 and 16 bytes of match there is almost
nothing, between 0 and 1 a lot.  Sixteen classes of one byte each represent
that; the RAW length stays in the file, so that another class division needs
no new dump.

CONTROL ARM (not run, budget): `m1c`, the same byte but the length class
shuffled by a fixed permutation.  It would separate "trust structure" from
"mere capacity".  The first screening asks a different question (does the
channel carry anything at all?), and the null check (projection initialised
to ZERO => step 0 identical to the reference) already rules out the initial
draw as an explanation.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from pysrc.model import CastedEmbedding, CastedLinear, InitializeScales, TransformerConfig

TOKEN_BITS = 8
TOKEN_MASK = (1 << TOKEN_BITS) - 1

MATCH_SHIFT = 8
MATCH_BITS = 9                       # 0..208 needs 9 bits
MATCH_MASK = (1 << MATCH_BITS) - 1

LEN_SHIFT = MATCH_SHIFT + MATCH_BITS  # 17
LEN_BITS = 4
LEN_MASK = (1 << LEN_BITS) - 1
# 8 + 9 + 4 = 21 bits, well inside the int64 carrier

N_MATCH_ROWS = 209                   # 208 token indices + the mark 208
N_LEN_CLASSES = 16

FILES = {
    "match_bytes": "match-bytes.uint8",
    "match_len": "match-len.uint8",
}

# Bits per class.  8 bits = 1 byte of match; class 15 collects everything from
# 120 bits (15 bytes) on and is the largest, with 26.7 % of the positions; its
# hit rate is 97.4 % (preliminary analysis on 2 M positions).
BITS_PER_CLASS = 8


def length_class(raw: np.ndarray) -> np.ndarray:
    """raw bit length -> class 0..15"""
    return np.minimum(np.asarray(raw, dtype=np.int64) // BITS_PER_CLASS,
                      N_LEN_CLASSES - 1)


def pack(tokens: np.ndarray, match_bytes: np.ndarray,
         match_len_raw: np.ndarray) -> np.ndarray:
    """int64 carrier: token at the bottom, match byte above, length class on top"""
    t = np.asarray(tokens, dtype=np.int64)
    mb = np.asarray(match_bytes, dtype=np.int64)
    cl = length_class(match_len_raw)
    assert len(mb) == len(t) and len(cl) == len(t)
    assert (t >= 0).all() and (t <= TOKEN_MASK).all()
    assert (mb >= 0).all() and (mb < N_MATCH_ROWS).all()
    return t | (mb << MATCH_SHIFT) | (cl << LEN_SHIFT)


def unpack(packed: np.ndarray):
    """(token, match_byte, length_class); the inverse of pack()"""
    q = np.asarray(packed, dtype=np.int64)
    return (q & TOKEN_MASK,
            (q >> MATCH_SHIFT) & MATCH_MASK,
            (q >> LEN_SHIFT) & LEN_MASK)


def load_channel(data_dir: str, mmap: bool = True):
    """the two channel files; their length MUST be the token count"""
    paths = [os.path.join(data_dir, FILES["match_bytes"]),
             os.path.join(data_dir, FILES["match_len"])]
    for p in paths:
        assert os.path.exists(p), (
            "The match channel is missing from the package: %s; without it the arm "
            "silently measures the reference.  (tf1_dump --match)" % p)
    mode = "r" if mmap else None
    if mmap:
        return (np.memmap(paths[0], dtype=np.uint8, mode=mode),
                np.memmap(paths[1], dtype=np.uint8, mode=mode))
    return (np.fromfile(paths[0], dtype=np.uint8),
            np.fromfile(paths[1], dtype=np.uint8))


def verify_channel(data_dir: str) -> dict:
    """checks the channel on the PACKAGE instead of assuming it

    Three things that can go wrong silently and then make a paid GPU hour
    measure the wrong thing:
      1. a length other than the token file's  -> offset by whole articles
      2. a match byte outside 0..208           -> device-side assert on the GPU
      3. the channel is constant               -> the arm is in fact the reference
    """
    tokens = np.fromfile(os.path.join(data_dir, "ppmd-bytes.uint8"), dtype=np.uint8)
    mb, ml = load_channel(data_dir)
    assert len(mb) == len(tokens), (
        "match-bytes has %d entries, the package %d tokens" % (len(mb), len(tokens)))
    assert len(ml) == len(tokens), (
        "match-len has %d entries, the package %d tokens" % (len(ml), len(tokens)))
    probe = np.asarray(mb[: 20_000_000])
    lprobe = np.asarray(ml[: 20_000_000])
    assert int(probe.max()) < N_MATCH_ROWS, "match byte outside 0..208"
    assert int(probe.min()) >= 0
    assert len(np.unique(probe)) > 1, "the match channel is constant"
    cls = length_class(lprobe)
    hits = int((probe[:-1] == tokens[1 : len(probe)]).sum())
    return {
        "n_tokens": len(tokens),
        "match_byte_max": int(probe.max()),
        "sentinel_share": float((probe == 208).mean()),
        "hit_rate_probe": hits / max(1, len(probe) - 1),
        "len_class_min": int(cls.min()),
        "len_class_max": int(cls.max()),
        "share_class_15": float((cls == 15).mean()),
    }


class ZMatchEmbedding(CastedEmbedding):
    """his token embedding plus an additive term from the MATCH channel

    Built like ZLastEmbedding (zmodel.py), for the same three reasons: the
    parameter name `embedding.weight` is kept (a subclass, not a wrapper), it
    is the only consumer of `input_tokens`, and `make_optimizers` finds the
    projection through `model.embedding.parameters()` in the AdamW group.

    Two slots, one shared table with an offset (one gather instead of two, one
    tensor in the checkpoint instead of two):

        rows   0..208   match byte (208 = no usable match)
        rows 209..224   length class 0..15

    The output projection is initialised to ZERO: with the same seed the twin
    is numerically identical to the reference at step 0, so a measured
    difference cannot come from a different initial draw.  The gradient of a
    zero weight in an additive branch is not zero, so it still learns.
    """

    def __init__(
        self,
        vocabulary_size: int,
        d_out: int,
        init,
        quantization,
        cfg: TransformerConfig,
        d_slot: int = 16,
        projection_quantization=None,
    ) -> None:
        super().__init__(vocabulary_size, d_out, init, quantization, cfg)
        self.d_slot = int(d_slot)
        self.slot_table = CastedEmbedding(
            N_MATCH_ROWS + N_LEN_CLASSES, self.d_slot, init, quantization, cfg
        )
        self.match_projection = CastedLinear(
            2 * self.d_slot, d_out, "zero", projection_quantization, cfg
        )

    def match_features(self, x_packed: Tensor, initialize_scales) -> Tensor:
        """(..., 2 * d_slot): match byte and length class, each through its rows"""
        mb = (x_packed >> MATCH_SHIFT) & MATCH_MASK
        cl = ((x_packed >> LEN_SHIFT) & LEN_MASK) + N_MATCH_ROWS
        ids = torch.stack([mb, cl], dim=-1)                  # (..., 2) int64
        rows = self.slot_table(ids, initialize_scales=initialize_scales)
        return rows.reshape(rows.shape[:-2] + (2 * self.d_slot,))

    def forward(self, x: Tensor, initialize_scales: InitializeScales) -> Tensor:
        tokens = x & TOKEN_MASK
        embedded = super().forward(tokens, initialize_scales=initialize_scales)
        features = self.match_features(x, initialize_scales).to(self.cfg.activation_dtype)
        return embedded + self.match_projection(features, initialize_scales=initialize_scales)


def install_match(raw_model, cfg: TransformerConfig, d_slot: int = 16) -> int:
    """installs the m1 embedding and keeps the trained weight"""
    old = raw_model.embedding
    device = old.weight.device
    new = ZMatchEmbedding(
        cfg.vocabulary_size,
        cfg.d_model,
        cfg.embedding_init,
        cfg.embedding_quantization,
        cfg,
        d_slot=d_slot,
        projection_quantization=cfg.prior_embedding_quantization,
    ).to(device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected, "unexpected keys: %s" % (unexpected,)
    raw_model.embedding = new
    return int(new.slot_table.weight.numel()) + sum(
        p.numel() for p in new.match_projection.parameters()
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "verify":
        for k, v in verify_channel(sys.argv[2]).items():
            print("%-20s %s" % (k, v))
        print("\nOK: the match channel matches the package")
    else:
        # self test of the packing: round trip over all edge values
        rng = np.random.default_rng(20260902)
        t = rng.integers(0, 208, 10000)
        mb = rng.integers(0, 209, 10000)
        ml = rng.integers(0, 256, 10000)
        q = pack(t, mb, ml)
        t2, mb2, cl2 = unpack(q)
        assert (t2 == t).all() and (mb2 == mb).all()
        assert (cl2 == length_class(ml)).all()
        assert (q >= 0).all()
        print("zmatch selftest OK  (max packed value %d)" % int(q.max()))
