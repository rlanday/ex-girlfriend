#!/usr/bin/env python3
"""The model side of the combined context package (arm "pk")

Derived from Vladimir Ivanov's pysrc/model.py (fx2-cmix-transformer, GPL-3):
`ZPaketEmbedding` subclasses his `CastedEmbedding` and uses his `CastedLinear`.

One embedding instead of three separate ones: the three arms measured before
each brought their own subclass of `CastedEmbedding` (zmodel.ZLastEmbedding,
zmatch.ZMatchEmbedding, zmodel.install_gate), and they exclude each other,
because each replaces `raw_model.embedding` as a whole.  `ZPaketEmbedding`
combines the three terms in one class:

    output = token row
           + last_projection(  X predecessor rows )          (predecessor channel)
           + match_projection( match byte, length class )    (match channel)

and widens the token table by the sentinel rows.  The construction rules of
the three models are kept: a subclass, not a wrapper (the parameter name
`embedding.weight` survives), the only consumer of `input_tokens`, and both
output projections initialised to ZERO, so that at step 0 the twin is
numerically identical to the reference and a measured difference cannot come
from a different initial draw.

Cost in parameters (d_model 192, d_slot 16, X=2, V=208, 8 sentinel classes):

    sentinel rows           8 x 192                    =  1,536
    predecessor table       2 x 216 x 16               =  6,912
    predecessor projection  32 x 192                   =  6,144
    match table             (209 + 16) x 16            =  3,600
    match projection        32 x 192                   =  6,144
    ------------------------------------------------------------
                                                         24,336

For comparison, the separate models: match alone 9,744, X=2 alone 12,800,
sentinel alone 1,536, together 24,080.  The extra 256 are the predecessor
table widened by the sentinel rows (2 x 8 x 16).
"""

from __future__ import annotations

import torch
from torch import Tensor

from pysrc.model import CastedEmbedding, CastedLinear, InitializeScales, TransformerConfig

from zctx import zpaket


class ZPaketEmbedding(CastedEmbedding):
    """token embedding + predecessor term + match term, sentinel rows included"""

    def __init__(
        self,
        vocabulary_size: int,
        d_out: int,
        init,
        quantization,
        cfg: TransformerConfig,
        x: int = zpaket.DEFAULT_X,
        n_sentinel: int = zpaket.N_SENTINEL,
        d_slot: int = 16,
        use_last: bool = True,
        use_match: bool = True,
        projection_quantization=None,
    ) -> None:
        # the token table carries V + n_sentinel rows; the output layer stays
        # at V, because a sentinel row is never predicted (it is masked in the
        # loss, as in zgate/install_gate)
        super().__init__(vocabulary_size + n_sentinel, d_out, init, quantization, cfg)
        self.v_plus = vocabulary_size + n_sentinel
        # The predecessor table carries one more row per slot when the edge
        # mark has a row of its own (X0ROW).  The token table stays at v_plus;
        # the mark is never a token.
        self.v_last = zpaket.last_rows(vocabulary_size)
        self.x = int(x)
        self.d_slot = int(d_slot)
        self.use_last = bool(use_last)
        self.use_match = bool(use_match)
        s = zpaket.shifts(self.x)
        self.shift_last = s["last"]
        self.shift_match = s["match"]
        self.shift_len = s["len"]

        if self.use_last:
            self.last_table = CastedEmbedding(
                self.x * self.v_last, self.d_slot, init, quantization, cfg
            )
            self.last_projection = CastedLinear(
                self.x * self.d_slot, d_out, "zero", projection_quantization, cfg
            )
            self.register_buffer(
                "last_offsets",
                torch.arange(self.x, dtype=torch.int64) * self.v_last,
                persistent=False,
            )
        if self.use_match:
            self.match_table = CastedEmbedding(
                zpaket.N_MATCH_ROWS + zpaket.N_LEN_CLASSES, self.d_slot,
                init, quantization, cfg,
            )
            self.match_projection = CastedLinear(
                2 * self.d_slot, d_out, "zero", projection_quantization, cfg
            )

    def last_features(self, x_packed: Tensor, initialize_scales) -> Tensor:
        parts = [
            ((x_packed >> (self.shift_last + zpaket.TOKEN_BITS * k)) & zpaket.TOKEN_MASK)
            + self.last_offsets[k]
            for k in range(self.x)
        ]
        ids = torch.stack(parts, dim=-1)
        rows = self.last_table(ids, initialize_scales=initialize_scales)
        return rows.reshape(rows.shape[:-2] + (self.x * self.d_slot,))

    def match_features(self, x_packed: Tensor, initialize_scales) -> Tensor:
        mb = (x_packed >> self.shift_match) & zpaket.MATCH_MASK
        cl = ((x_packed >> self.shift_len) & zpaket.LEN_MASK) + zpaket.N_MATCH_ROWS
        ids = torch.stack([mb, cl], dim=-1)
        rows = self.match_table(ids, initialize_scales=initialize_scales)
        return rows.reshape(rows.shape[:-2] + (2 * self.d_slot,))

    def forward(self, x: Tensor, initialize_scales: InitializeScales) -> Tensor:
        tokens = x & zpaket.TOKEN_MASK
        out = super().forward(tokens, initialize_scales=initialize_scales)
        if self.use_last:
            f = self.last_features(x, initialize_scales).to(self.cfg.activation_dtype)
            out = out + self.last_projection(f, initialize_scales=initialize_scales)
        if self.use_match:
            f = self.match_features(x, initialize_scales).to(self.cfg.activation_dtype)
            out = out + self.match_projection(f, initialize_scales=initialize_scales)
        return out


def install_paket(raw_model, cfg: TransformerConfig, x: int = zpaket.DEFAULT_X,
                  n_sentinel: int = zpaket.N_SENTINEL, d_slot: int = 16,
                  use_last: bool = True, use_match: bool = True) -> int:
    """install the package embedding, keeping the trained token rows

    Call it directly after `Transformer(cfg)` and BEFORE `load_state_dict`, so
    that a checkpoint of this arm finds its extra tensors.
    """
    old = raw_model.embedding
    device = old.weight.device
    new = ZPaketEmbedding(
        cfg.vocabulary_size, cfg.d_model, cfg.embedding_init,
        cfg.embedding_quantization, cfg,
        x=x, n_sentinel=n_sentinel, d_slot=d_slot,
        use_last=use_last, use_match=use_match,
        projection_quantization=cfg.prior_embedding_quantization,
    ).to(device)
    with torch.no_grad():
        new.weight[: cfg.vocabulary_size].copy_(old.weight)
    assert new.weight.shape[0] == cfg.vocabulary_size + n_sentinel
    raw_model.embedding = new
    zusatz = n_sentinel * cfg.d_model
    if use_last:
        zusatz += int(new.last_table.weight.numel())
        zusatz += sum(p.numel() for p in new.last_projection.parameters())
    if use_match:
        zusatz += int(new.match_table.weight.numel())
        zusatz += sum(p.numel() for p in new.match_projection.parameters())
    return int(zusatz)
