#!/usr/bin/env python3
"""the model side of arm z1: one extra additive term in the residual stream

Derived from Vladimir Ivanov's pysrc/model.py (fx2-cmix-transformer, GPL-3):
`ZEmbedding` subclasses HIS `CastedEmbedding` and HIS `CastedLinear` is the
projection.  Nothing in `pysrc/` is patched, in line with the rule that
tft/ lies next to his tree and does not change it.

Why the embedding and not the model:

  * `train()` hands the model exactly five tensors, so an extra input would
    mean copying his training loop.  `zloader` packs ordinal and position into
    the int64 `input_tokens` instead (zfeat) and this is the single place that
    unpacks them; poc4m has `value_embedding_pattern=None`, so
    `self.embedding` is the ONLY consumer of `input_tokens`.
  * the parameter name `embedding.weight` is PRESERVED (subclass, not wrapper),
    so a checkpoint of the twin still loads into the plain model and the export
    sees one added tensor rather than a renamed tree.
  * `make_optimizers` starts from `model.embedding.parameters()`, so the
    projection lands in AdamW with the other embedding-shaped weights and his
    `len(muon) + len(adamw) == len(parameters)` assertion still holds.

Zero initialisation of the projection is deliberate: at step 0 the twin is
numerically IDENTICAL to poc4m, so a measured difference cannot come from a
different starting point.  The gradient of a zero weight in an additive branch
is not zero, so it still learns.

Cost of arm z1 in S: 64 x d_model weights (12288 at d_model=192, ~6 KB at int4
with scales, x2 = ~12 KB) plus the 256 entry phase table (<= 2 KB fp32, x2).
Not free: only the two NUMBERS cost nothing, because the decoder counts them
itself; the projection that turns them into a vector is weight like any other.

KNOWN LIMIT (screening only): the export chain asserts fixed tensor counts
(poc4m = 75/80/59).  With z1 installed the counts change and
`tft/export_weights_poc.py:expected_counts()` has to learn about the two extra
tensors.  For a screening run that only reads the loss this does not matter;
before anything is exported it does.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pysrc.model import CastedEmbedding, CastedLinear, InitializeScales, TransformerConfig

from zctx import zfeat


class ZEmbedding(CastedEmbedding):
    """his token embedding plus the z1 context term, unpacked from the input"""

    def __init__(
        self,
        vocabulary_size: int,
        d_out: int,
        init,
        quantization,
        cfg: TransformerConfig,
        projection_quantization=None,
    ) -> None:
        super().__init__(vocabulary_size, d_out, init, quantization, cfg)
        table = torch.from_numpy(zfeat.phase_table())          # (T, 2) float32
        self.register_buffer("phase_table", table, persistent=True)
        self.context_projection = CastedLinear(
            zfeat.N_FEATURES, d_out, "zero", projection_quantization, cfg
        )
        self._article_shifts = [
            zfeat.ARTICLE_BITS - zfeat.PHASE_BITS - k for k in range(zfeat.ARTICLE_LEVELS)
        ]
        self._position_shifts = [
            zfeat.POSITION_BITS - zfeat.PHASE_BITS - k for k in range(zfeat.POSITION_LEVELS)
        ]

    def _phase(self, value: Tensor, shifts: list) -> Tensor:
        mask = zfeat.PHASE_TABLE_SIZE - 1
        parts = [
            (value >> s) & mask if s >= 0 else (value << (-s)) & mask for s in shifts
        ]
        return torch.stack(parts, dim=-1)                       # (..., levels) int64

    def context_features(self, ordinal: Tensor, position: Tensor) -> Tensor:
        """(..., N_FEATURES): the exact torch mirror of zfeat.features()"""
        a = self.phase_table[self._phase(ordinal, self._article_shifts)]
        p = self.phase_table[self._phase(position, self._position_shifts)]
        head = a.shape[:-2]
        return torch.cat(
            [a.reshape(head + (2 * zfeat.ARTICLE_LEVELS,)),
             p.reshape(head + (2 * zfeat.POSITION_LEVELS,))],
            dim=-1,
        )

    def forward(self, x: Tensor, initialize_scales: InitializeScales) -> Tensor:
        tokens = x & zfeat.TOKEN_MASK
        ordinal = (x >> zfeat.ARTICLE_SHIFT) & zfeat.ARTICLE_MASK
        position = (x >> zfeat.POSITION_SHIFT) & zfeat.POSITION_MASK
        embedded = super().forward(tokens, initialize_scales=initialize_scales)
        features = self.context_features(ordinal, position).to(self.cfg.activation_dtype)
        return embedded + self.context_projection(features, initialize_scales=initialize_scales)


def install_z1(raw_model, cfg: TransformerConfig) -> int:
    """swap in the z1 embedding, keeping the trained weight; returns new params

    Call directly after `Transformer(cfg)` and BEFORE `load_state_dict`, so a
    checkpoint written by a z1 run loads its projection back.
    """
    old = raw_model.embedding
    device = old.weight.device
    new = ZEmbedding(
        cfg.vocabulary_size,
        cfg.d_model,
        cfg.embedding_init,
        cfg.embedding_quantization,
        cfg,
        projection_quantization=cfg.prior_embedding_quantization,
    ).to(device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected, "unexpected keys when re-using the embedding: %s" % (unexpected,)
    raw_model.embedding = new
    return sum(p.numel() for p in new.context_projection.parameters())


# --------------------------------------------------------------------------
# arm z4 "lastbytes"
# --------------------------------------------------------------------------

from zctx import zlast


class ZLastEmbedding(CastedEmbedding):
    """his token embedding plus a direct n-gram term from the last X tokens

    Same construction as ZEmbedding and for the same three reasons (parameter
    name preserved, single consumer of `input_tokens`, optimizer grouping).
    The difference is what is embedded: not a phase of two counters, but the
    X preceding token identities, each through its OWN small table, so the
    model can tell "the previous token was a space" from "the token before
    that was a space".  The tables are laid out as ONE table of X * V rows
    with a per slot offset: one gather instead of X, and one tensor instead
    of X in the checkpoint.

    Zero initialised output projection, exactly as in z1: identical to the
    reference at step 0.
    """

    def __init__(
        self,
        vocabulary_size: int,
        d_out: int,
        init,
        quantization,
        cfg: TransformerConfig,
        x: int = 4,
        d_slot: int = 16,
        projection_quantization=None,
    ) -> None:
        super().__init__(vocabulary_size, d_out, init, quantization, cfg)
        self.x = int(x)
        self.d_slot = int(d_slot)
        self.slot_table = CastedEmbedding(
            self.x * vocabulary_size, self.d_slot, init, quantization, cfg
        )
        self.last_projection = CastedLinear(
            self.x * self.d_slot, d_out, "zero", projection_quantization, cfg
        )
        offsets = torch.arange(self.x, dtype=torch.int64) * vocabulary_size
        self.register_buffer("slot_offsets", offsets, persistent=False)

    def last_features(self, x_packed: Tensor, initialize_scales) -> Tensor:
        """(..., x * d_slot): the X predecessors, each through its own rows"""
        mask = zlast.TOKEN_MASK
        parts = [
            ((x_packed >> (zlast.LAST_SHIFT + zlast.TOKEN_BITS * k)) & mask)
            + self.slot_offsets[k]
            for k in range(self.x)
        ]
        ids = torch.stack(parts, dim=-1)                      # (..., x) int64
        rows = self.slot_table(ids, initialize_scales=initialize_scales)
        return rows.reshape(rows.shape[:-2] + (self.x * self.d_slot,))

    def forward(self, x: Tensor, initialize_scales: InitializeScales) -> Tensor:
        tokens = x & zlast.TOKEN_MASK
        embedded = super().forward(tokens, initialize_scales=initialize_scales)
        features = self.last_features(x, initialize_scales).to(self.cfg.activation_dtype)
        return embedded + self.last_projection(features, initialize_scales=initialize_scales)


def install_z4(raw_model, cfg: TransformerConfig, x: int = 4, d_slot: int = 16) -> int:
    """swap in the z4 embedding, keeping the trained weight; returns new params"""
    old = raw_model.embedding
    device = old.weight.device
    new = ZLastEmbedding(
        cfg.vocabulary_size,
        cfg.d_model,
        cfg.embedding_init,
        cfg.embedding_quantization,
        cfg,
        x=x,
        d_slot=d_slot,
        projection_quantization=cfg.prior_embedding_quantization,
    ).to(device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected, "unexpected keys when re-using the embedding: %s" % (unexpected,)
    raw_model.embedding = new
    return sum(p.numel() for p in new.slot_table.parameters()) + sum(
        p.numel() for p in new.last_projection.parameters()
    )


# --------------------------------------------------------------------------
# digit gating: sentinel rows for arm g1
# --------------------------------------------------------------------------

def install_gate(raw_model, cfg: TransformerConfig, n_extra: int) -> int:
    """widens the token embedding by n_extra sentinel rows

    Arm g1 feeds ONE input token per skipped digit block, carrying the length
    class of the block.  Its indices lie ABOVE the vocabulary: it is a pure
    model input, never in the stream, and so costs no new symbol; what is
    paid for is n_extra x d_model weights (with 8 classes and d_model 192:
    1,536 values, about 0.8 KB at int4 with scales, x2 in the self-extractor).

    The OUTPUT LAYER stays at V: the sentinel row predicts nothing and is
    masked in the loss.  The first V rows are taken from the reference model,
    the new ones keep their random initialisation (they are meant to learn);
    the rest of the model is identical to the reference at step 0.
    """
    old = raw_model.embedding
    device = old.weight.device
    new = CastedEmbedding(
        cfg.vocabulary_size + n_extra,
        cfg.d_model,
        cfg.embedding_init,
        cfg.embedding_quantization,
        cfg,
    ).to(device)
    with torch.no_grad():
        new.weight[: cfg.vocabulary_size].copy_(old.weight)
    assert new.weight.shape[0] == cfg.vocabulary_size + n_extra
    raw_model.embedding = new
    return n_extra * cfg.d_model
