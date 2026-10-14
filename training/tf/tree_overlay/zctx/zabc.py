#!/usr/bin/env python3
"""Two further arms on the z1 packing: z2 and z1c

Derived from Vladimir Ivanov's pysrc/model.py (fx2-cmix-transformer, GPL-3):
`ZClusterEmbedding` subclasses his `CastedEmbedding`.

Both use the existing z1 packing (zfeat: token | ordinal << 8 | position <<
28), so the loader stays unchanged and both arms are siblings of z1, measured
against the same reference.

  z2  stream position alone.  z1 fed article number and stream position
      together and made the loss worse (+0.000633, 6.1 sigma), which rules
      out the combination, not the position part.  z2 zeroes the article half
      of the 64 features and keeps the 32 position features; the weights on
      the zeroed inputs start at zero and get zero gradient, so the arm equals
      a projection with 32 inputs (cost in S 32 x d_model).

  z1c cluster number instead of the raw article number.  The training stream
      is in the base article order of this entry (training/data/README.md
      section 3; its .new_article_order has md5 2eb1a3a2, the residual file
      of that order).  That order derives from the article order of fx2-cmix,
      built by
      clustering (Byron Knoll's pipeline: embedding, 1-D t-SNE, k-means;
      clusters in axis order, articles inside a cluster by ascending id, then
      reversed with `tac`) and inherited through cmix-lex in the revision of
      Kaido Orav's fxcm_v26 archive.  The cluster partition
      follows without any extra artefact: inside a cluster the order values
      fall, a rise marks a cluster boundary.  `mkcluster.py` stamps the table
      ordinal -> cluster number (1,692 clusters, median 70 articles).
      Causal: the decoder counts the article separators, so it knows the
      ordinal, and with it the cluster, before the prediction; only the table
      is paid for, as a weight.  The form is an embedding (variant "(c)
      cluster embedding" in zfeat.py), not the sine phase: a learned table
      suits a few hundred rows and needs no transcendental function in the
      coder.

Both output projections start at zero, so with the same seed the twin is
numerically identical to the reference at step 0.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import embedding

from pysrc.model import CastedEmbedding, CastedLinear, InitializeScales, TransformerConfig

from zctx import zfeat
from zctx.zmodel import ZEmbedding

# mkcluster.py stamps the cluster table from the base article order of this entry
# (training/data/README.md section 3): 1,692 clusters over the regular part
# (median 70 articles); the five class blocks at the end of the stream share
# one collective number.  The 200 M training prefix (53,651 articles) reaches
# cluster 808.
CLUSTER_TABELLE = os.environ.get("ZCTX_CLUSTER_TABLE", "/workspace/data/cluster_of_article.int32")
CLUSTER_ROWS = int(os.environ.get("ZCTX_CLUSTER_ROWS", "1792"))


def _cluster_lookup() -> torch.Tensor:
    """Ordinal -> cluster number, from the stamped table; the file is required.

    There is deliberately no fallback formula: a silent substitute would look
    like a valid measurement and measure the wrong thing.
    """
    import numpy as np

    tab = np.fromfile(CLUSTER_TABELLE, dtype=np.int32)
    assert tab.size > 0, "cluster table missing or empty: " + CLUSTER_TABELLE
    return torch.from_numpy(tab.astype("int64"))


class ZPosEmbedding(ZEmbedding):
    """z2: like z1, but the article half of the features is switched off"""

    def context_features(self, ordinal: Tensor, position: Tensor) -> Tensor:
        feats = super().context_features(ordinal, position)
        n_article = 2 * zfeat.ARTICLE_LEVELS
        # the article half is forced to zero rather than dropped, so that the
        # tensor shape, and with it the comparison with z1, stays the same
        mask = torch.zeros_like(feats)
        mask[..., n_article:] = 1.0
        return feats * mask


class ZClusterEmbedding(CastedEmbedding):
    """z1c: token embedding plus one learned row per article cluster"""

    def __init__(
        self,
        vocabulary_size: int,
        d_out: int,
        init,
        quantization,
        cfg: TransformerConfig,
        rows: int = CLUSTER_ROWS,
        d_slot: int = 32,
        projection_quantization=None,
    ) -> None:
        super().__init__(vocabulary_size, d_out, init, quantization, cfg)
        self.rows = int(rows)
        lookup = _cluster_lookup()
        assert int(lookup.max()) < self.rows, (
            "cluster table has %d clusters, the table is built for %d rows"
            % (int(lookup.max()) + 1, self.rows)
        )
        self.register_buffer("cluster_of_article", lookup, persistent=False)
        self.d_slot = int(d_slot)
        # a plain parameter table instead of a second CastedEmbedding
        # submodule: an extra module layer is costly in the compiled path.
        self.cluster_weight = Parameter(
            torch.zeros((self.rows, self.d_slot), dtype=cfg.weight_dtype)
        )
        with torch.no_grad():
            self.cluster_weight.normal_(mean=0.0, std=0.02)
        self.cluster_projection = CastedLinear(
            self.d_slot, d_out, "zero", projection_quantization, cfg
        )

    def forward(self, x: Tensor, initialize_scales: InitializeScales) -> Tensor:
        tokens = x & zfeat.TOKEN_MASK
        ordinal = (x >> zfeat.ARTICLE_SHIFT) & zfeat.ARTICLE_MASK
        embedded = super().forward(tokens, initialize_scales=initialize_scales)
        idx = torch.clamp(ordinal, max=self.cluster_of_article.numel() - 1)
        ids = self.cluster_of_article[idx]
        rows = embedding(ids, self.cluster_weight.to(self.cfg.activation_dtype))
        return embedded + self.cluster_projection(
            rows.to(self.cfg.activation_dtype), initialize_scales=initialize_scales
        )


def install_z2(raw_model, cfg: TransformerConfig) -> int:
    old = raw_model.embedding
    device = old.weight.device
    new = ZPosEmbedding(
        cfg.vocabulary_size, cfg.d_model, cfg.embedding_init,
        cfg.embedding_quantization, cfg,
        projection_quantization=cfg.prior_embedding_quantization,
    ).to(device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected, "unexpected keys: %s" % (unexpected,)
    raw_model.embedding = new
    # counted as trained: only the 32 position columns get a gradient
    used = 2 * zfeat.POSITION_LEVELS * cfg.d_model
    return used


def install_z1c(raw_model, cfg: TransformerConfig,
                rows: int = CLUSTER_ROWS, d_slot: int = 32) -> int:
    old = raw_model.embedding
    device = old.weight.device
    new = ZClusterEmbedding(
        cfg.vocabulary_size, cfg.d_model, cfg.embedding_init,
        cfg.embedding_quantization, cfg,
        rows=rows, d_slot=d_slot,
        projection_quantization=cfg.prior_embedding_quantization,
    ).to(device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected, "unexpected keys: %s" % (unexpected,)
    raw_model.embedding = new
    return int(new.cluster_weight.numel()) + sum(
        p.numel() for p in new.cluster_projection.parameters()
    )
