#!/usr/bin/env python3
r"""The loader that takes its rows from the coder's row dump

Derived from Vladimir Ivanov's pysrc/data.py (fx2-cmix-transformer, GPL-3).

The difference to `zploader.py` in one sentence: there Python rebuilds the
document from pk tables (title, z1t/syn/alias/z3, A2, sentinel, gate, X2,
match); here it reads the finished rows that the coder itself fed.  The
training form is therefore not a second implementation of the same rules but a
copy of the only one that counts (the coder is the specification).

Record format and flag bits: dump format v2 (`training/README.md` section 3).
It is read by `zctx/dumpv2.py`, the same module that the package build and the
acceptance test use.

Target rule (see `zctx/dumpv2.py`):

    The target of row j is the next stream row of the same document, past all
    inserted rows; `flags` bit4 says whether there is one.
    scored <=> flags bit1 == 0 and the row has a stream target.

The second half is the correction of the stage-1 form (`PK_SENT_ONCE`): the
token behind the sentinel row is scored at the digit row, and the sentinel row
beside it is fed as context only.  The dump mode gets this for free instead of
rebuilding it, so `ziel_tok` is never >= V and the emergency stop of the old
loader is gone.

What the dump does not carry, and where it comes from:
  * the target of the last fed row of each document (the last token of a
    document is coded but never fed).  It is in the table
    `doc-last-ziel.int32`, stamped at package build time (one int32 per
    document, -1 = no target); it is the only value that does not come
    directly from the dump.
  * the priors.  They stay where they are: `priors[ziel-1]` from
    `ppmd-probs.float16`, indexed absolutely, and `prior0` for every row with
    `ziel < 0`, which the coder states itself in `flags` bit4 and which is
    cross-checked here.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import tensor, zeros

from pysrc.data import CollatedArticlesDataLoader, MicroBatch

from zctx import dumpv2, zpaket


def packe(tok, prev1, prev2, mb, cl, x: int = 2) -> np.ndarray:
    """int64 carrier per row

    The predecessors come from the dump: the coder shifts them over the fed
    sequence, and that sequence is here row for row.  Shifting them again in
    Python would be the reconstruction this mode abolishes, and at document
    boundaries and at the edge mark it would be the first place where both
    sides drift apart.
    """
    assert x == 2, "the dump carries exactly two predecessor columns"
    s = zpaket.shifts(x)
    out = np.asarray(tok, dtype=np.int64).copy()
    m = np.asarray(mb, dtype=np.int64)
    c = np.asarray(cl, dtype=np.int64)
    assert (m < zpaket.N_MATCH_ROWS).all(), "match byte outside 0..208"
    assert (c < zpaket.N_LEN_CLASSES).all(), "length class outside 0..15"
    out |= np.asarray(prev1, dtype=np.int64) << s["last"]
    out |= np.asarray(prev2, dtype=np.int64) << (s["last"] + zpaket.TOKEN_BITS)
    out |= m << s["match"]
    out |= c << s["len"]
    return out


class PaketDumpDataLoader(CollatedArticlesDataLoader):
    """Ivanov's loader, fed from the coder's row dump

    Not a dataclass, for the same reason as `PaketDataLoader`: the tables must
    exist before `super().__init__`, because the parent constructor already
    iterates over the micro-batches and asks for the lengths.

    In this mode `article_boundaries_filename` carries the document boundaries
    in stream coordinates (the coder has already made the 2^17 piece cut and,
    where redirect runs are merged, the merge).  So `n_articles` of the parent
    equals the number of documents, and the epoch permutations run over the
    right set.
    """

    def __init__(self, *, dump_dir: str, **kwargs) -> None:
        with open(os.path.join(dump_dir, "dump-meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        daten, kopf, _ = dumpv2.lade(os.path.join(dump_dir, "zeilen.dump"),
                                     mmap=kwargs.get("mmap", True))
        doc_rows = np.fromfile(os.path.join(dump_dir, "doc-rows.int64"),
                               dtype=np.int64)
        doc_ziel = np.fromfile(os.path.join(dump_dir, "doc-last-ziel.int32"),
                               dtype=np.int32).astype(np.int64)
        assert doc_rows[0] == 0 and int(doc_rows[-1]) == len(daten), (
            "doc-rows does not match the dump: %d/%d against %d rows"
            % (doc_rows[0], doc_rows[-1], len(daten)))
        assert len(doc_ziel) == len(doc_rows) - 1, "doc-last-ziel has the wrong length"
        assert (np.diff(doc_rows) > 0).all(), "empty document in the dump"

        object.__setattr__(self, "_dz", daten)
        object.__setattr__(self, "_doc_rows", doc_rows)
        object.__setattr__(self, "_doc_ziel", doc_ziel)
        object.__setattr__(self, "_dmeta", meta)
        object.__setattr__(self, "pk_x", int(meta.get("x", 2)))
        object.__setattr__(self, "_dump_rest", None)

        super().__init__(**kwargs)

        v = self._vocabulary_size
        zpaket.set_vocabulary(v)
        nsent = int(meta.get("nsent", zpaket.N_SENTINEL))
        x0row = bool(meta.get("x0row", False))
        marke = (v + nsent) if x0row else 0
        object.__setattr__(self, "_marke", marke)
        object.__setattr__(self, "_nsent", nsent)

        assert self.n_articles == len(doc_rows) - 1, (
            "after the parent constructor article-boundaries holds %d documents, "
            "the dump %d; the loader has CUT AGAIN.  `max_article_tokens` "
            "must be >= the largest document span in dump mode (the coder "
            "has already made the 2^17 cut, and the span can exceed 2^17 "
            "when the cut falls on a gated position)"
            % (self.n_articles, len(doc_rows) - 1))
        assert v == int(meta["v"]), "V of the package %d against dump %d" % (v, meta["v"])
        assert x0row == (os.environ.get("PK_LAST_X0ROW", "0") != "0"), (
            "PK_LAST_X0ROW is %s, the dump was stamped with x0row=%s; "
            "then the model reads a different row at the document start than in "
            "training" % (os.environ.get("PK_LAST_X0ROW", "0"), x0row))

        # ---- guards against a silent break of the training form -----------
        p_max = int(max(np.asarray(daten["p1"]).max(),
                        np.asarray(daten["p2"]).max()))
        if x0row:
            assert p_max == marke, (
                "X0ROW=1 needs the edge mark %d, the dump carries at most %d; "
                "it comes from a coder build without the X0ROW row form"
                % (marke, p_max))
        else:
            assert p_max < v + nsent, (
                "X0ROW=0, but the dump carries the predecessor value %d >= V+NSENT"
                % p_max)
        probe = daten[: min(len(daten), 1 << 22)]
        assert int(np.asarray(probe["res"]).max()) == 0, (
            "reserved != 0; the dump format has changed")

        # prior0: the replacement vector of every row without a stream target.
        # It lies beside the package, and its sha is in the manifest of the
        # exported weights.
        p0_weg = os.path.join(dump_dir, "z3-prior0.float16")
        if not os.path.exists(p0_weg):
            p0_weg = os.path.join(os.path.dirname(self.tokens_filename),
                                  "z3-prior0.float16")
        p0 = np.fromfile(p0_weg, dtype=np.float16)
        assert len(p0) == v, "prior0 has %d columns, the package %d" % (len(p0), v)
        object.__setattr__(self, "_prior0_np", np.ascontiguousarray(p0))
        object.__setattr__(self, "_prior0_weg", p0_weg)

        # ---- the position limit, checked ----------------------------------
        # The flash-attn kernel gets `max_variable_sequence_length =
        # POSITION_LIMIT` (tft/config.py).  A document with more positions
        # (coded plus inserted rows) runs out of the kernel's block plan:
        # plausible garbage forward, NaN backward.  The coder cuts in front of
        # the prompt block (at most 131,033 rows measured), so the limit holds;
        # it is checked here rather than assumed.
        from tft import config as _cfg
        _lmax = int(np.diff(doc_rows).max())
        assert _lmax <= _cfg.POSITION_LIMIT, (
            "longest document %d rows (positions, not coded tokens) above "
            "the position limit %d; the kernel then returns NaN in the "
            "backward pass.  Either raise TFT_POSITION_LIMIT (and the coder's "
            "piece cut with it) or cut the package again."
            % (_lmax, _cfg.POSITION_LIMIT))

        if self.rank == 0:
            self._selbstauskunft()
            print("dump loader: position limit %d, longest document %d rows, "
                  "margin %d"
                  % (_cfg.POSITION_LIMIT, _lmax, _cfg.POSITION_LIMIT - _lmax),
                  flush=True)

    # ---- self-report ------------------------------------------------------

    def _selbstauskunft(self) -> None:
        d = self._dz
        fl = np.asarray(d["fl"], dtype=np.int64)
        ein = (fl & dumpv2.EINGEFUEGT) != 0
        bew = (fl & dumpv2.MASKE) == 0
        sent = np.asarray(d["tok"]) >= self._vocabulary_size
        print("dump loader: %d documents | %d rows (%d stream, %d inserted, "
              "%d sentinel, %d redirect page) | scored %d (%.4f %%)"
              % (self.n_articles, len(d), int((~ein).sum()), int(ein.sum()),
                 int(sent.sum()), int(((fl & dumpv2.REDIRSEITE) != 0).sum()),
                 int(bew.sum()), 100.0 * int(bew.sum()) / len(d)), flush=True)
        print("dump loader: packing %d carrier bits, X=%d, edge mark %d, "
              "window %s, redirmerge %s, arms %s"
              % (zpaket.shifts(self.pk_x)["bits"], self.pk_x, self._marke,
                 self._dmeta.get("fenster"), self._dmeta.get("redirmerge"),
                 ",".join(self._dmeta.get("arme", []))), flush=True)
        laengen = np.diff(self._doc_rows)
        print("dump loader: longest document %d rows, micro-batch %d "
              "(truncation rule %s), mind the position limit"
              % (int(laengen.max()), self.micro_batch_tokens,
                 "not needed" if int(laengen.max()) <= self.micro_batch_tokens
                 else "APPLIES: micro-batch too small"), flush=True)

    # ---- the overridden methods -------------------------------------------

    def _iter_micro_batch_articles(self, articles, drop_last: bool):
        rows = self._doc_rows
        micro_batch: list = []
        micro_batch_length: int = 0
        for a in articles:
            length = int(rows[a + 1] - rows[a])
            if length <= 0:
                continue
            assert length <= self.micro_batch_tokens, (
                "document %d has %d rows and fits in no micro-batch (%d)"
                % (a, length, self.micro_batch_tokens))
            if micro_batch_length + length > self.micro_batch_tokens:
                yield micro_batch
                micro_batch = []
                micro_batch_length = 0
            micro_batch.append(a)
            micro_batch_length += length
        if micro_batch and not drop_last:
            yield micro_batch

    def zeilen(self, a: int):
        """(eingabe, ziel, ziel_tok, bewerte, src, prior) of one document

        Separate from `_make_micro_batch`, so that the acceptance test can hold
        exactly these rows against the loader dump without building a
        micro-batch.  (eingabe = input, ziel = target, bewerte = scored.)
        """
        v = self._vocabulary_size
        r0 = int(self._doc_rows[a])
        r1 = int(self._doc_rows[a + 1])
        f = dumpv2.felder(self._dz[r0:r1])
        L = r1 - r0
        dr = np.array([0, L], dtype=np.int64)
        ziel, ziel_tok, bewerte, _ = dumpv2.zielspalten(
            f, dr, v, doc_ziel=np.array([self._doc_ziel[a]]),
            strom_tok=self._tokens)
        assert (ziel_tok < v).all(), (
            "document %d: a sentinel row became a target" % a)
        # The prior follows `flags` bit4, not the target: at 8 rows of the full
        # stream (A2 blocks directly behind each other) the coder takes the
        # PPMd row of the next stream token instead of prior0.
        prior = dumpv2.priorziel(f, ziel)
        eingabe = packe(f["tok"], f["p1"], f["p2"], f["mb"], f["mc"], self.pk_x)
        return eingabe, ziel, ziel_tok, bewerte, f["src"], prior

    def _make_micro_batch(self, articles: list) -> MicroBatch:
        n_positions: int = self.micro_batch_tokens
        pin: bool = self.device.type == "cuda"
        v: int = self._vocabulary_size

        input_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        output_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        loss_mask = zeros((1, n_positions), dtype=torch.bool, pin_memory=pin)
        priors = zeros((1, n_positions, v), dtype=torch.float16, pin_memory=pin)

        boundaries: list = [0]
        position: int = 0
        for a in articles:
            eingabe, ziel, ziel_tok, bewerte, src, prior = self.zeilen(a)
            length = len(eingabe)
            assert length > 0
            assert position + length <= n_positions

            input_tokens[0, position:position + length] = torch.from_numpy(
                np.ascontiguousarray(eingabe))
            output_tokens[0, position:position + length] = torch.from_numpy(
                np.ascontiguousarray(ziel_tok))
            loss_mask[0, position:position + length] = torch.from_numpy(
                np.ascontiguousarray(bewerte))

            ps = np.where(prior >= 0, prior - 1, -1)
            gueltig = ps >= 0
            block = np.asarray(self._priors[np.where(gueltig, ps, 0)])
            if not gueltig.all():
                block = np.array(block, copy=True)
                block[~gueltig] = self._prior0_np
            priors[0, position:position + length] = torch.from_numpy(
                np.ascontiguousarray(block))

            self._vielleicht_dumpen(a, eingabe, ziel, ziel_tok, bewerte, src)
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
                [boundaries], dtype=torch.int32).to(self.device),
        )

    # ---- row dump in the format of the old loader (for the acceptance test)

    def _vielleicht_dumpen(self, a, eingabe, ziel, ziel_tok, bewerte, src):
        weg = os.environ.get("PK_ZEILENDUMP")
        if not weg or self._dump_rest == 0:
            return
        rest = self._dump_rest
        if rest is None:
            rest = int(os.environ.get("PK_ZEILENDUMP_N", "200000"))
            with open(weg, "w") as fh:
                fh.write("artikel\tzeile\tsrc_ein\ttok_ein\tsrc_ziel\ttok_ziel"
                         "\tmaske\tprev1\tprev2\tmatch_b\tmatch_cls\tprior\n")
        nimm = min(rest, len(eingabe))
        t, p, mb, cl = zpaket.unpack(np.asarray(eingabe[:nimm]), self.pk_x)
        with open(weg, "a") as fh:
            for i in range(nimm):
                fh.write("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n"
                         % (a, i, int(src[i]), int(t[i]), int(ziel[i]),
                            int(ziel_tok[i]), int(bewerte[i]), int(p[0][i]),
                            int(p[1][i]), int(mb[i]), int(cl[i]),
                            "prior0" if ziel[i] < 0 else str(int(ziel[i]) - 1)))
        object.__setattr__(self, "_dump_rest", rest - nimm)
