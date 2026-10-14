#!/usr/bin/env python3
"""the loader side of both arms: Z1 packing and the Z3 prefix

Derived from Vladimir Ivanov's pysrc/data.py (fx2-cmix-transformer, GPL-3).
`ZDataLoader` subclasses HIS `CollatedArticlesDataLoader` and overrides exactly
two methods; `pysrc/` is not patched.

With `arm="none"` the loader is his, byte for byte in behaviour; that is the
control side of the twin, so both arms and the reference run the SAME code.

arm "z1"  packs the article ordinal and the global stream position into the
          int64 `input_tokens` (zfeat).  Nothing else changes: same documents,
          same number of supervised positions, same batching.

arm "z3"  puts the extracted link-context prefix in front of each article,
          inside the SAME document (one `cumulative_sequence_lengths` segment),
          and excludes it from the loss:

              document = prefix ... article separator ... article tokens
              loss_mask = False over the prefix, True over the article
              priors    = z3-prior0 over the prefix, the real rows after it

          The number of SUPERVISED positions per article is unchanged, so the
          loss is comparable to poc4m position for position; what grows is the
          number of positions the model has to run over (measured +3.1 % on
          the 200 M package), and that is the price of the arm.

          The prefix priors are the shipped order 0 table, NOT the original
          prior rows of the copied tokens.  The coder cannot replay a past PPMD
          state (it would have to cache 208 float16 per token, 83 GB), so
          training on rows the coder could never deliver would measure a
          feature we cannot build.  See SPEC_Z3.md section 7.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import tensor, zeros

from pysrc.data import CollatedArticlesDataLoader, MicroBatch

from zctx import z3_extract, zfeat, zgate, zlast, zmatch

ARMS = ("none", "z1", "z3", "z1z3", "z4", "z4c", "g0", "g1", "m1")

# Arm z4 "lastbytes": X comes from the environment, so that the scale 2/4/8
# runs without a code change.  z4c is the same arm with fixed shuffled slots,
# the control arm.
Z4_X = int(os.environ.get("ZCTX_LAST_X", "4"))

# Digit gating: the digit zones are cut out of the model run by the CAUSAL
# automaton rule.  "g0" drops them, "g1" inserts one sentinel row with the
# length class per skipped block.  M and the link/table exclusion come from
# the environment, so that the calibrated rule runs without a code change.
GATE_M = int(os.environ.get("ZGATE_M", "1"))
GATE_EXCLUDE = os.environ.get("ZGATE_EXCLUDE", "1") != "0"


class Z3Stream:
    """the extra stream produced by `python -m zctx.z3_extract build`"""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.tokens = np.fromfile(
            os.path.join(directory, z3_extract.FILES["prefix_tokens"]), dtype=np.uint8
        )
        self.offsets = np.fromfile(
            os.path.join(directory, z3_extract.FILES["prefix_offsets"]), dtype=np.int32
        ).astype(np.int64)
        self.prior0 = np.fromfile(
            os.path.join(directory, z3_extract.FILES["prior0"]), dtype=np.float16
        )
        with open(os.path.join(directory, z3_extract.FILES["manifest"])) as f:
            self.manifest = json.load(f)
        assert self.offsets[0] == 0
        assert int(self.offsets[-1]) == len(self.tokens)
        self.lengths = np.diff(self.offsets)

    @property
    def n_articles(self) -> int:
        return len(self.offsets) - 1

    def prefix(self, article: int) -> np.ndarray:
        return self.tokens[self.offsets[article] : self.offsets[article + 1]]


class ZDataLoader(CollatedArticlesDataLoader):
    """his loader with the Z1 packing and the Z3 prefix bolted on

    Deliberately NOT a dataclass: the parent is a frozen slots dataclass, and a
    plain subclass gets a __dict__, so the two extra attributes can be set with
    object.__setattr__ BEFORE the parent's __post_init__ runs (it already calls
    the overridden `_iter_micro_batch_articles`).
    """

    def __init__(self, *, arm: str = "none", z3_dir: str | None = None, **kwargs) -> None:
        assert arm in ARMS, "arm must be one of %s" % (ARMS,)
        z3 = Z3Stream(z3_dir) if ("z3" in arm and z3_dir) else None
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "z3", z3)
        object.__setattr__(self, "_prior0", None)
        object.__setattr__(self, "_match_bytes", None)
        object.__setattr__(self, "_match_len", None)
        super().__init__(**kwargs)

        # Arm m1: the two channel files lie next to the token file; they are
        # mapped, not read (2 x 200 MB per rank).  If the channel is missing,
        # load_channel stops, because an arm that silently runs as the
        # reference is the most expensive kind of error.
        if arm == "m1":
            _dir = os.path.dirname(self.tokens_filename)
            _mb, _ml = zmatch.load_channel(_dir, mmap=self.mmap)
            assert len(_mb) == len(self._tokens), (
                "match-bytes has %d entries, the package %d tokens"
                % (len(_mb), len(self._tokens)))
            assert len(_ml) == len(self._tokens), (
                "match-len has %d entries, the package %d tokens"
                % (len(_ml), len(self._tokens)))
            object.__setattr__(self, "_match_bytes", _mb)
            object.__setattr__(self, "_match_len", _ml)

        if z3 is not None:
            assert z3.n_articles == self.n_articles, (
                "the Z3 stream has %d articles, the loader has %d; the loader "
                "merges and splits articles, so a package whose longest article "
                "exceeds max_article_tokens cannot be used with Z3 as written"
                % (z3.n_articles, self.n_articles)
            )
            object.__setattr__(
                self,
                "_prior0",
                torch.from_numpy(np.ascontiguousarray(z3.prior0)).view(1, -1),
            )
        if arm != "none":
            extra = "" if z3 is None else ", z3=" + z3.directory
            if arm in ("z4", "z4c"):
                extra += ", X=%d, slots %s" % (
                    Z4_X,
                    zlast.control_permutation(Z4_X) if arm == "z4c" else list(range(Z4_X)),
                )
            if arm == "m1":
                extra += ", match channel: %d byte rows + %d length classes" % (
                    zmatch.N_MATCH_ROWS, zmatch.N_LEN_CLASSES)
            print("zctx: arm=%s%s" % (arm, extra), flush=True)

    # ---- gating (arms g0/g1) -------------------------------------------------

    def _gate_prep(self):
        """the gate mask of the WHOLE stream and the row count per article

        Lazy, because the parent's `__post_init__` sets `_tokens` and then at
        once iterates over `_iter_micro_batch_articles`.  The mask is computed
        on the whole stream, exactly as the decoder runs, while the trainer
        only orders the articles differently.
        """
        cached = getattr(self, "_gate", None)
        if cached is not None:
            return cached
        bitmap = np.fromfile(
            os.path.join(os.path.dirname(self.tokens_filename), "vocab-bitmap.u8"),
            dtype=np.uint8,
        )
        values = np.array(
            [b * 8 + k for b in range(32) for k in range(8) if bitmap[b] >> k & 1],
            dtype=np.uint8,
        )
        assert len(values) == self._vocabulary_size, (
            "the bitmap lists %d byte values, the prior rows have %d columns"
            % (len(values), self._vocabulary_size)
        )
        gate = zgate.gate_mask(self._tokens, values, GATE_M, exclude=GATE_EXCLUDE)

        starts = self._article_starts
        rows = np.empty(len(starts) - 1, dtype=np.int64)
        sent = self.arm == "g1"
        for a in range(len(rows)):
            s, e = int(starts[a]), int(starts[a + 1])
            body = e - s - 1
            if body <= 0:
                rows[a] = 0
                continue
            g = gate[s : s + body]
            n_keep = int((~g).sum())
            if sent:
                n_keep += sum(1 for x, y in zgate.runs_of(g) if y < body)
            rows[a] = n_keep
        object.__setattr__(self, "_gate", gate)
        object.__setattr__(self, "_gate_rows_per_article", rows)
        if self.rank == 0:
            total_pos = int(self._article_starts[-1] - (len(starts) - 1))
            print(
                "zgate: M=%d exclusion=%s: %d of %d positions gated (%.3f %%), "
                "model rows %d (%+.3f %%)"
                % (GATE_M, GATE_EXCLUDE, int(gate.sum()), total_pos,
                   100.0 * int(gate.sum()) / total_pos, int(rows.sum()),
                   100.0 * (int(rows.sum()) - total_pos) / total_pos),
                flush=True,
            )
        return gate

    def _gate_length(self, article: int) -> int:
        self._gate_prep()
        return int(self._gate_rows_per_article[article])

    # ---- the two overridden methods -----------------------------------------

    def _prefix_length(self, article: int) -> int:
        return 0 if self.z3 is None else int(self.z3.lengths[article])

    def _iter_micro_batch_articles(self, articles, drop_last: bool):
        if self.arm in ("g0", "g1"):
            micro_batch: list = []
            micro_batch_length: int = 0
            for article in articles:
                length = self._gate_length(article)
                if length <= 0:
                    continue
                if micro_batch_length + length > self.micro_batch_tokens:
                    yield micro_batch
                    micro_batch = []
                    micro_batch_length = 0
                micro_batch.append(article)
                micro_batch_length += length
            if micro_batch and not drop_last:
                yield micro_batch
            return

        micro_batch: list = []
        micro_batch_length: int = 0
        for article in articles:
            # his rule: an article of length L contributes L-1 supervised
            # positions, and one of length 1 contributes none.  A prefix must
            # NOT resurrect such an article: it would carry a prefix and no
            # supervised token, and _make_micro_batch asserts against it.
            body: int = int(
                self._article_starts[article + 1] - self._article_starts[article] - 1
            )
            if body <= 0:
                continue
            length: int = body + self._prefix_length(article)
            if micro_batch_length + length > self.micro_batch_tokens:
                yield micro_batch
                micro_batch = []
                micro_batch_length = 0
            micro_batch.append(article)
            micro_batch_length += length
        if micro_batch and not drop_last:
            yield micro_batch

    def _make_gated_micro_batch(self, articles: list) -> MicroBatch:
        """the gated micro-batch (arms g0/g1)

        The rows come from `zgate.gate_rows`, the SAME function the reference
        form `zgate.build_document` uses, so training stream and run stream
        cannot drift apart.  For every row, target = tokens[src+1] and
        prior = priors[src]; the sentinel row of a block carries src = the last
        gated position, that is the prior row of exactly the token the
        transformer did not predict.
        """
        self._gate_prep()
        n_positions: int = self.micro_batch_tokens
        pin: bool = self.device.type == "cuda"
        v: int = self._vocabulary_size
        sentinel: bool = self.arm == "g1"

        input_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        output_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        loss_mask = zeros((1, n_positions), dtype=torch.bool, pin_memory=pin)
        priors = zeros((1, n_positions, v), dtype=torch.float16, pin_memory=pin)

        boundaries: list = [0]
        position: int = 0
        for article in articles:
            start: int = int(self._article_starts[article])
            end: int = int(self._article_starts[article + 1])
            body: int = end - start - 1
            assert body > 0
            src, sent, cls = zgate.gate_rows(self._gate[start : start + body], sentinel)
            length: int = len(src)
            assert length > 0
            assert position + length <= n_positions

            tokens = self._tokens[start:end].astype(np.int64)
            inp = np.where(sent, v + cls, tokens[src])
            assert inp.max() < v + zgate.N_LENGTH_CLASSES
            rows = np.array(self._priors[start : start + body])[src]

            input_tokens[0, position : position + length] = torch.from_numpy(inp)
            output_tokens[0, position : position + length] = torch.from_numpy(
                tokens[src + 1]
            )
            loss_mask[0, position : position + length] = torch.from_numpy(~sent)
            priors[0, position : position + length] = torch.from_numpy(rows)

            position += length
            boundaries.append(position)

        if position < n_positions:
            boundaries.append(n_positions)

        return MicroBatch(
            input_tokens=input_tokens.to(self.device, non_blocking=pin),
            output_tokens=output_tokens.to(self.device, non_blocking=pin),
            loss_mask=loss_mask.to(self.device, non_blocking=pin),
            priors=priors.to(self.device, non_blocking=pin),
            cumulative_sequence_lengths=tensor(
                [boundaries], dtype=torch.int32
            ).to(self.device),
        )

    def _make_micro_batch(self, articles: list) -> MicroBatch:
        if self.arm == "none":
            return super()._make_micro_batch(articles)
        if self.arm in ("g0", "g1"):
            return self._make_gated_micro_batch(articles)

        n_positions: int = self.micro_batch_tokens
        pin: bool = self.device.type == "cuda"

        input_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        output_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        loss_mask = zeros((1, n_positions), dtype=torch.bool, pin_memory=pin)
        priors = zeros(
            (1, n_positions, self._vocabulary_size), dtype=torch.float16, pin_memory=pin
        )

        boundaries: list = [0]
        position: int = 0
        for article in articles:
            start: int = int(self._article_starts[article])
            end: int = int(self._article_starts[article + 1])
            body: int = end - start - 1
            assert body > 0
            prefix = (
                self.z3.prefix(article) if self.z3 is not None else np.zeros(0, np.uint8)
            )
            q: int = len(prefix)
            length: int = q + body
            assert position + length <= n_positions

            # the document is prefix ++ article; input is all but its last
            # token, output all but its first; written as one slice pair so
            # the empty prefix case is not a special case
            document = np.concatenate(
                [prefix, np.asarray(self._tokens[start:end])]
            ).astype(np.int64)
            document_in = document[:-1]
            document_out = document[1:]
            assert len(document_in) == length and len(document_out) == length

            if self.arm == "m1":
                # The channel is per position, not per document: entry j
                # belongs to stream position start+j and predicts start+j+1,
                # the same alignment as `self._priors[start : end - 1]`.
                # There is no z3 prefix for m1 (q == 0), so the slice covers
                # document_in exactly.
                assert q == 0, "m1 is not combined with a Z3 prefix"
                document_in = zmatch.pack(
                    document_in,
                    np.asarray(self._match_bytes[start : end - 1]),
                    np.asarray(self._match_len[start : end - 1]),
                )

            if self.arm in ("z4", "z4c"):
                # the prefix counts as part of the document: the decoder would
                # have seen it at this point too
                document_in = zlast.pack(document_in, Z4_X, permute=self.arm == "z4c")

            if "z1" in self.arm:
                ordinal = np.full(length, article, dtype=np.int64)
                stream = np.empty(length, dtype=np.int64)
                stream[:q] = start          # the prefix carries the article's own
                stream[q:] = np.arange(start, end - 1, dtype=np.int64)
                document_in = zfeat.pack(document_in, ordinal, stream)

            input_tokens[0, position : position + length] = torch.from_numpy(document_in)
            output_tokens[0, position : position + length] = torch.from_numpy(document_out)
            loss_mask[0, position + q : position + length] = True
            if q:
                priors[0, position : position + q] = self._prior0
            priors[0, position + q : position + length] = torch.from_numpy(
                np.array(self._priors[start : end - 1])
            )

            position += length
            boundaries.append(position)

        if position < n_positions:
            boundaries.append(n_positions)

        cumulative_sequence_lengths = tensor([boundaries], dtype=torch.int32)

        return MicroBatch(
            input_tokens=input_tokens.to(self.device, non_blocking=pin),
            output_tokens=output_tokens.to(self.device, non_blocking=pin),
            loss_mask=loss_mask.to(self.device, non_blocking=pin),
            priors=priors.to(self.device, non_blocking=pin),
            cumulative_sequence_lengths=cumulative_sequence_lengths.to(self.device),
        )
