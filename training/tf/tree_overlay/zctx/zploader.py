#!/usr/bin/env python3
r"""The loader of the combined arm "pk"

Derived from Vladimir Ivanov's pysrc/data.py (fx2-cmix-transformer, GPL-3).
`PaketDataLoader` subclasses his `CollatedArticlesDataLoader` and overrides
the same two methods as `ZDataLoader`.

A module of its own beside `zloader.py` rather than a further arm there: the
arms of that module (z1/z3/z4/g0/g1/m1) are measured and their results are
final, and changing them would be a risk without benefit.

Nine switches, each of which can be turned off on its own; with ALL of them
off this loader is element by element the original loader (the null test):

    PK_TITLEMASK   do not score the title (it comes from the title list)
    PK_Z1T         article number as text in the prompt
    PK_SYN         redirect synonyms in the prompt
    PK_Z3          incoming link context in the prompt
    PK_A2          title re-injection at the section boundaries
    PK_A2N         running heading number before the re-injected title
    PK_SENT        sentinel length class behind every digit run
    PK_LAST        X=2 predecessor channel in the packing
    PK_MATCH       match channel in the packing

The document of an article:

    title . </title> | #<n> . synonyms . link context | body
    \_____ masked __________________________________/  \_ scored _/

The prefix stands BEHIND the title, because the coder needs the title to look
up the incoming link contexts.

The whole comparison rests on a counting argument.  A row is scored exactly
when the PREDICTED token is a body token of the original stream.  So the
scored set is independent of any insertion and identical to that of the masked
reference arm mref: a one-factor comparison without conversion.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import tensor, zeros

from pysrc.data import CollatedArticlesDataLoader, MicroBatch

from zctx import zgate, zpaket


def _an(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


class PrefixStream:
    """a prefix package in the z3 file format (tokens + offsets per article)"""

    def __init__(self, directory: str, name: str) -> None:
        self.directory = directory
        self.name = name
        self.tokens = np.fromfile(
            os.path.join(directory, "z3-prefix-tokens.uint8"), dtype=np.uint8
        )
        self.offsets = np.fromfile(
            os.path.join(directory, "z3-prefix-offsets.int32"), dtype=np.int32
        ).astype(np.int64)
        assert self.offsets[0] == 0, "%s: offsets do not start at 0" % name
        assert int(self.offsets[-1]) == len(self.tokens), (
            "%s: offsets (%d) and token file (%d) do not match"
            % (name, int(self.offsets[-1]), len(self.tokens))
        )
        self.lengths = np.diff(self.offsets)

    @property
    def n_articles(self) -> int:
        return len(self.offsets) - 1

    def prefix(self, article: int) -> np.ndarray:
        return self.tokens[self.offsets[article] : self.offsets[article + 1]]


class PaketDataLoader(CollatedArticlesDataLoader):
    """Ivanov's loader with the complete context package

    Not a dataclass, for the same reason as ZDataLoader: the tables must be
    set BEFORE `super().__init__`, because the parent constructor already
    iterates over the micro-batches and asks for their lengths.
    """

    def __init__(self, *, pk_dir: str, **kwargs) -> None:
        schalter = {
            "titelmaske": _an("PK_TITLEMASK"),
            "z1t": _an("PK_Z1T"),
            "syn": _an("PK_SYN"),
            "z3": _an("PK_Z3"),
            "a2": _an("PK_A2"),
            "a2n": _an("PK_A2N"),
            "sent": _an("PK_SENT"),
            "last": _an("PK_LAST"),
            "match": _an("PK_MATCH"),
        }
        x = int(os.environ.get("PK_LAST_X", str(zpaket.DEFAULT_X)))
        object.__setattr__(self, "pk", schalter)
        object.__setattr__(self, "pk_dir", pk_dir)
        object.__setattr__(self, "pk_x", x)

        # --- tables BEFORE the parent constructor -------------------------
        titel = None
        if schalter["titelmaske"] or schalter["a2"]:
            titel = np.fromfile(
                os.path.join(pk_dir, "title-lengths.int32"), dtype=np.int32
            ).astype(np.int64)
        a2i = a2o = None
        if schalter["a2"]:
            a2i = np.fromfile(
                os.path.join(pk_dir, "reinject-index.int32"), dtype=np.int32
            ).astype(np.int64)
            a2o = np.fromfile(
                os.path.join(pk_dir, "reinject-offsets.int32"), dtype=np.int32
            ).astype(np.int64)
            assert a2i[0] == 0 and int(a2i[-1]) == len(a2o), "A2: index inconsistent"
        sent_i = sent_p = sent_c = None
        if schalter["sent"]:
            wege = [os.path.join(pk_dir, f) for f in
                    ("sent-index.int32", "sent-pos.int32", "sent-cls.uint8")]
            if all(os.path.exists(w) for w in wege):
                sent_i = np.fromfile(wege[0], dtype=np.int32).astype(np.int64)
                sent_p = np.fromfile(wege[1], dtype=np.int32).astype(np.int64)
                sent_c = np.fromfile(wege[2], dtype=np.uint8)
                assert int(sent_i[-1]) == len(sent_p) == len(sent_c), (
                    "sentinel table inconsistent"
                )
        stroeme = []
        for schluessel, verzeichnis in (("z1t", "z1t"), ("syn", "syn"), ("z3", "z3")):
            if schalter[schluessel]:
                stroeme.append(PrefixStream(os.path.join(pk_dir, verzeichnis), schluessel))
        object.__setattr__(self, "_titel", titel)
        object.__setattr__(self, "_a2_idx", a2i)
        object.__setattr__(self, "_a2_off", a2o)
        object.__setattr__(self, "_prefixe", stroeme)
        object.__setattr__(self, "_sent_i", sent_i)
        object.__setattr__(self, "_sent_p", sent_p)
        object.__setattr__(self, "_sent_c", sent_c)
        # Zone mask of the coder tool `tf1_gate --tool`: one byte per stream
        # position, 0 = kept, 1/2/3 = KOPFREST/CMTIDX/HDRBLOCK (page head,
        # comment index, metadata block).  It must be set BEFORE the parent
        # constructor, because that already iterates over the micro-batches.
        _zone = None
        _zp = os.path.join(os.path.dirname(kwargs["tokens_filename"]),
                           "gate-mask.uint8")
        if os.environ.get("PK_ZONE", "1") != "0" and os.path.exists(_zp):
            _zone = np.fromfile(_zp, dtype=np.uint8) > 0
        object.__setattr__(self, "_zone", _zone)
        object.__setattr__(self, "_kept_cum", None)
        object.__setattr__(self, "_plan_cache", None)
        object.__setattr__(self, "_gate", None)
        object.__setattr__(self, "_sent_cache", None)
        object.__setattr__(self, "_match_bytes", None)
        object.__setattr__(self, "_match_len", None)
        object.__setattr__(self, "_prior0", None)
        object.__setattr__(self, "_prior0_np", None)
        object.__setattr__(self, "_tok_of_byte", None)

        super().__init__(**kwargs)

        v = self._vocabulary_size
        zpaket.set_vocabulary(v)
        if self._zone is not None:
            assert len(self._zone) == len(self._tokens), (
                "gate-mask.uint8 has %d entries, the package %d tokens"
                % (len(self._zone), len(self._tokens))
            )
        datadir = os.path.dirname(self.tokens_filename)
        bitmap = np.fromfile(os.path.join(datadir, "vocab-bitmap.u8"), dtype=np.uint8)
        object.__setattr__(self, "_tok_of_byte", zpaket.byte_to_token(bitmap))
        if schalter["a2n"]:
            zpaket.number_tokens(1234567890, self._tok_of_byte)   # stops if a byte is missing

        for s in stroeme:
            assert s.n_articles == self.n_articles, (
                "prefix package %s has %d articles, the loader %d; merge/split active?"
                % (s.name, s.n_articles, self.n_articles)
            )
        if titel is not None:
            assert len(titel) == self.n_articles, (
                "title-lengths has %d entries, the loader %d articles"
                % (len(titel), self.n_articles)
            )
        if a2i is not None:
            assert len(a2i) == self.n_articles + 1, "A2: index has the wrong length"

        if stroeme or schalter["a2"] or schalter["sent"]:
            p0 = np.fromfile(os.path.join(pk_dir, "z3-prior0.float16"), dtype=np.float16)
            assert len(p0) == v, "prior0 has %d columns, the package %d" % (len(p0), v)
            object.__setattr__(
                self, "_prior0", torch.from_numpy(np.ascontiguousarray(p0)).view(1, -1)
            )
            object.__setattr__(self, "_prior0_np", np.ascontiguousarray(p0))

        if schalter["match"]:
            for datei, feld in (("match-bytes.uint8", "_match_bytes"),
                                ("match-len.uint8", "_match_len")):
                pfad = os.path.join(datadir, datei)
                assert os.path.exists(pfad), (
                    "The match channel is missing from the package: %s; without it the arm "
                    "silently measures the reference." % pfad
                )
                arr = (np.memmap(pfad, dtype=np.uint8, mode="r") if self.mmap
                       else np.fromfile(pfad, dtype=np.uint8))
                assert len(arr) == len(self._tokens), (
                    "%s has %d entries, the package %d tokens"
                    % (datei, len(arr), len(self._tokens))
                )
                object.__setattr__(self, feld, arr)
            try:
                zpaket.verify_zmatch()
            except ImportError as fehler:
                # fla/triton are missing on the CPU machine; the cross-check
                # of the duplicated constants then runs on the GPU pod
                print("context package: zmatch cross-check skipped (%s)" % fehler,
                      flush=True)

        if self.rank == 0:
            self._selbstauskunft()

    # ---- self-report (Selbstauskunft) -------------------------------------

    def _selbstauskunft(self) -> None:
        plan = self._plan()
        n_pos = int(self._article_starts[-1] - self.n_articles)
        zusatz = int(plan["rows"].sum()) - n_pos
        print("context package: options %s" % json.dumps(self.pk, sort_keys=True), flush=True)
        print(
            "context package: %d articles | prompt %d tokens (z1t/syn/z3 %s) | "
            "A2 %d sites, %d tokens | sentinel %d rows | title masked %d tokens"
            % (
                self.n_articles,
                int(plan["q"].sum()),
                "/".join("%d" % int(s.lengths.sum()) for s in self._prefixe) or "-",
                int(plan["a2_sites"].sum()),
                int(plan["a2"].sum()),
                int(plan["sent"].sum()),
                int(plan["maske"].sum()),
            ),
            flush=True,
        )
        print(
            "context package: model rows %d against %d original positions (%+.3f %%), "
            "scored %d (%.4f %% of the original positions)"
            % (
                int(plan["rows"].sum()), n_pos, 100.0 * zusatz / n_pos,
                int(plan["bewertet"].sum()),
                100.0 * int(plan["bewertet"].sum()) / n_pos,
            ),
            flush=True,
        )
        print(
            "context package: packing %d carrier bits, X=%d"
            % (zpaket.shifts(self.pk_x)["bits"], self.pk_x),
            flush=True,
        )

    # ---- gate mask and sentinel sites -------------------------------------

    def _gate_prep(self):
        cached = getattr(self, "_gate", None)
        if cached is not None:
            return cached
        datadir = os.path.dirname(self.tokens_filename)
        bitmap = np.fromfile(os.path.join(datadir, "vocab-bitmap.u8"), dtype=np.uint8)
        werte = np.array(
            [b * 8 + k for b in range(32) for k in range(8) if bitmap[b] >> k & 1],
            dtype=np.uint8,
        )
        assert len(werte) == self._vocabulary_size
        gate = zgate.gate_mask(
            self._tokens, werte,
            int(os.environ.get("ZGATE_M", "1")),
            exclude=os.environ.get("ZGATE_EXCLUDE", "1") != "0",
        )
        object.__setattr__(self, "_gate", gate)
        return gate

    def _sentinels(self, article: int):
        """(positions, classes) of this article

        Preferably from the PRE-STAMPED table (pk/sent-*).  The live
        computation below stays as a fallback and as a cross-check; it gives
        the same result but is expensive: zgate.exclusion_mask allocates about
        a dozen int64 intermediate arrays over 200 M tokens (about 10 to
        15 GiB peak) and triggered the OOM killer on a pod with 123 GiB before
        the first step ran.  The sites are a deterministic function of the
        stream and therefore belong in the package.

        Filtering happens HERE, not in the table: the title mask is a switch,
        and a table stamped for ONE switch setting would be a silent trap.
        """
        if self._sent_i is not None:
            start = int(self._article_starts[article])
            T = self._titel_len(article)
            a = int(self._sent_i[article])
            b = int(self._sent_i[article + 1])
            pos = self._sent_p[a:b]
            cls = self._sent_c[a:b]
            if T:
                nimm = pos >= start + T
                pos, cls = pos[nimm], cls[nimm]
            return pos, cls
        cache = getattr(self, "_sent_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(self, "_sent_cache", cache)
        hit = cache.get(article)
        if hit is None:
            start = int(self._article_starts[article])
            end = int(self._article_starts[article + 1])
            body = end - start - 1
            T = self._titel_len(article)
            g = self._gate_prep()[start : start + body]
            roh = [(p, c) for p, c in zpaket.sentinel_sites(g, start, body)
                   if p >= start + T]
            hit = (np.asarray([p for p, _ in roh], dtype=np.int64),
                   np.asarray([c for _, c in roh], dtype=np.int8))
            cache[article] = hit
        return hit

    # ---- the PLAN: the only source for length and content -----------------

    def _titel_len(self, article: int) -> int:
        if self._titel is None:
            return 0
        return int(self._titel[article])

    def _a2_sites(self, article: int) -> np.ndarray:
        if self._a2_idx is None:
            return np.zeros(0, dtype=np.int64)
        a = int(self._a2_idx[article])
        b = int(self._a2_idx[article + 1])
        return self._a2_off[a:b]

    def _plan(self) -> dict:
        """per article: prompt, A2 rows, sentinel rows, rows, scored rows

        This table is the ONLY source for these numbers.  Length computation
        and document construction both ask it: if the two drift apart, only
        the assert in the micro-batch catches it, in the middle of a paid GPU
        hour.
        """
        cached = getattr(self, "_plan_cache", None)
        if cached is not None:
            return cached
        n = self.n_articles
        q = np.zeros(n, dtype=np.int64)
        a2 = np.zeros(n, dtype=np.int64)
        a2_sites = np.zeros(n, dtype=np.int64)
        sent = np.zeros(n, dtype=np.int64)
        rows = np.zeros(n, dtype=np.int64)
        maske = np.zeros(n, dtype=np.int64)
        bewertet = np.zeros(n, dtype=np.int64)
        aus = np.zeros(n, dtype=np.int8)          # bit mask: 1 A2 off, 2 sentinels off, 4 prompt off
        starts = self._article_starts
        grenze = self.micro_batch_tokens
        kept = None
        if self._zone is not None:
            kept = self._kept_cum
            if kept is None:
                kept = np.concatenate(
                    [[0], np.cumsum(~self._zone, dtype=np.int64)])
                object.__setattr__(self, "_kept_cum", kept)
        for a in range(n):
            start = int(starts[a])
            end = int(starts[a + 1])
            T = self._titel_len(a)
            if kept is None:
                body = end - start - 1
                t_kept = T
            else:
                body = int(kept[end] - kept[start]) - 1
                t_kept = int(kept[start + T] - kept[start]) if T else 0
            if body <= 0:
                continue
            qa = int(sum(int(s.lengths[a]) for s in self._prefixe))
            stellen = self._a2_sites(a) if self.pk["a2"] else np.zeros(0, np.int64)
            stellen = stellen[stellen >= start + T]
            extra_a2 = 0
            if T > 0 and len(stellen):
                extra_a2 = int(sum(
                    T + (len("#%d" % k) if self.pk["a2n"] else 0)
                    for k in range(1, len(stellen) + 1)
                ))
            extra_s = len(self._sentinels(a)[0]) if self.pk["sent"] else 0
            flag = 0
            if body + qa + extra_a2 + extra_s > grenze:
                extra_a2 = 0
                flag |= 1
                if body + qa + extra_s > grenze:
                    extra_s = 0
                    flag |= 2
                    if body + qa > grenze:
                        qa = 0
                        flag |= 4
            q[a] = qa
            a2[a] = extra_a2
            a2_sites[a] = len(stellen) if extra_a2 else 0
            sent[a] = extra_s
            rows[a] = body + qa + extra_a2 + extra_s
            maske[a] = (t_kept - 1) if (self.pk["titelmaske"] and T > 0) else 0
            # (the first position of the article is not in `body` anyway:
            #  body = end - start - 1)
            bewertet[a] = body - maske[a]
            aus[a] = flag
        plan = {"q": q, "a2": a2, "a2_sites": a2_sites, "sent": sent, "rows": rows,
                "maske": maske, "bewertet": bewertet, "aus": aus}
        object.__setattr__(self, "_plan_cache", plan)
        if self.rank == 0 and int((aus != 0).sum()):
            print("context package: %d articles with a shortened package (micro-batch limit)"
                  % int((aus != 0).sum()), flush=True)
        return plan

    # ---- the two overridden methods ---------------------------------------

    def _iter_micro_batch_articles(self, articles, drop_last: bool):
        plan = self._plan()
        micro_batch: list = []
        micro_batch_length: int = 0
        for article in articles:
            length = int(plan["rows"][article])
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

    def _dokument(self, article: int):
        """(tok, src) of an article, strictly following the plan"""
        plan = self._plan()
        start = int(self._article_starts[article])
        end = int(self._article_starts[article + 1])
        T = self._titel_len(article)
        flag = int(plan["aus"][article])

        prompt = np.zeros(0, dtype=np.int64)
        if not flag & 4 and self._prefixe:
            prompt = np.concatenate(
                [np.asarray(s.prefix(article), dtype=np.int64) for s in self._prefixe]
            )
        stellen = np.zeros(0, dtype=np.int64)
        if self.pk["a2"] and not flag & 1 and T > 0:
            stellen = self._a2_sites(article)
            stellen = stellen[stellen >= start + T]
        sentinels = (np.zeros(0, np.int64), np.zeros(0, np.int8))
        if self.pk["sent"] and not flag & 2:
            sentinels = self._sentinels(article)

        tok, src = zpaket.build_document(
            self._tokens, start, end, T, prompt, stellen,
            self.pk["a2n"], sentinels, self._tok_of_byte, self._zone,
        )
        assert len(tok) - 1 == int(plan["rows"][article]), (
            "plan and document construction disagree: %d vs. %d"
            % (len(tok) - 1, int(plan["rows"][article]))
        )
        return tok, src, start, T

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
        for article in articles:
            tok, src, start, T = self._dokument(article)
            length: int = len(tok) - 1
            assert length > 0
            assert position + length <= n_positions

            ziel = src[1:]                       # stream index of the prediction target
            eingabe = tok[:-1]
            src_zeile = src[:-1]                 # stream index of the INPUT of the row
            # Only BODY TOKENS of the original stream are scored.  The
            # `max(.., 1)` is the seam to the reference: Ivanov's loader never
            # predicts the FIRST token of an article (it is the first input).
            # With a prompt in front the transformer could predict it, but then
            # the scored set would be larger than that of the reference and the
            # comparison broken.  The gain of this one position per article is
            # therefore given away on purpose; in the coder it comes on top.
            schwelle = start + max(T if self.pk["titelmaske"] else 0, 1)

            # --- input: token + predecessors + match channel ----------------
            if self.pk["last"] or self.pk["match"]:
                if self.pk["match"]:
                    end = int(self._article_starts[article + 1])
                    mb_art = np.asarray(self._match_bytes[start:end], dtype=np.int64)
                    ml_art = np.asarray(self._match_len[start:end], dtype=np.int64)
                    lokal = np.where(src_zeile >= 0, src_zeile - start, 0)
                    mb = np.where(src_zeile >= 0, mb_art[lokal], 208)
                    cl = np.where(src_zeile >= 0,
                                  zpaket.match_length_class(ml_art[lokal]), 0)
                else:
                    mb = np.full(length, 208, dtype=np.int64)
                    cl = np.zeros(length, dtype=np.int64)
                x = self.pk_x if self.pk["last"] else 0
                eingabe = zpaket.pack(eingabe, mb, cl, x)

            input_tokens[0, position : position + length] = torch.from_numpy(
                np.ascontiguousarray(eingabe)
            )
            # The TARGET must be a valid index of the output layer (< V).  A
            # sentinel row carries V+class and is meant as INPUT, but it is
            # necessarily also the target of the row before it.  That row is
            # masked, so its target value is meaningless; passed on raw it
            # still triggers a device-side assert in the first step (seen on
            # the pod).  It is therefore set to 0, and the mask beside it
            # proves that no scored place changes.
            ziel_tok = np.asarray(tok[1:], dtype=np.int64)
            ausser = ziel_tok >= v
            unterdrueckt = np.zeros(length, dtype=bool)
            if ausser.any():
                if os.environ.get("PK_SENT_SCORE", "1") != "0":
                    # Without this branch the row `digit -> SENT` carries a
                    # meaningless target and is masked.  Here it gets the
                    # target X (the token BEHIND the sentinel row) and is
                    # SCORED, so that it matches the coder row by row.
                    j = np.flatnonzero(ausser)
                    j = j[j + 2 <= len(tok) - 1]
                    ziel = np.array(ziel, copy=True)
                    ziel_tok[j] = np.asarray(tok[j + 2], dtype=np.int64)
                    ziel[j] = src[j + 2]
                    # The SENT row j+1 predicts the SAME X as the digit row j
                    # just scored.  If both stay on, X is scored TWICE; on a
                    # 1 MB stream excerpt that gave 533,289 scored rows for
                    # 532,532 distinct target positions, the difference being
                    # the number of sentinel rows.  The scored set would then
                    # no longer be that of the reference arm mref (532,530),
                    # and the one-factor comparison would break; the coder,
                    # too, codes X exactly once.  So X is scored at the DIGIT
                    # row and the SENT row beside it is switched off.
                    # PK_SENT_ONCE=0 restores the double-scoring form.
                    if os.environ.get("PK_SENT_ONCE", "1") != "0":
                        jn = j + 1
                        unterdrueckt[jn[jn < length]] = True
                    rest = np.flatnonzero(ziel_tok >= v)
                    if len(rest):
                        assert not (ziel >= schwelle)[rest].any(), (
                            "a SCORED row points to a sentinel row"
                        )
                        ziel_tok[rest] = 0
                else:
                    assert not (ziel >= schwelle)[ausser].any(), (
                        "a SCORED row points to a sentinel row"
                    )
                    ziel_tok = np.where(ausser, 0, ziel_tok)
            output_tokens[0, position : position + length] = torch.from_numpy(
                np.ascontiguousarray(ziel_tok)
            )
            bewerte = (ziel >= schwelle) & ~unterdrueckt
            loss_mask[0, position : position + length] = torch.from_numpy(bewerte)

            # --- prior rows: row ziel-1 predicts ziel -----------------------
            # ONE gather per article instead of one copy group per contiguous
            # piece: the group form cost 219 ms per micro-batch on a small test
            # package against 20 ms for the original, because it builds an
            # intermediate tensor per group.  With about 900 insertion sites
            # per micro-batch that is the difference between computing and
            # waiting.
            ps = np.where(ziel >= 0, ziel - 1, -1)
            gueltig = ps >= 0
            block = np.asarray(self._priors[np.where(gueltig, ps, 0)])
            if not gueltig.all():
                block = np.array(block, copy=True)
                block[~gueltig] = self._prior0_np
            priors[0, position : position + length] = torch.from_numpy(
                np.ascontiguousarray(block)
            )

            _dmp = os.environ.get("PK_ZEILENDUMP")
            if _dmp and getattr(self, "_dump_rest", None) != 0:
                rest = getattr(self, "_dump_rest", None)
                if rest is None:
                    rest = int(os.environ.get("PK_ZEILENDUMP_N", "200000"))
                    with open(_dmp, "w") as f:
                        f.write("artikel\tzeile\tsrc_ein\ttok_ein\tsrc_ziel"
                                "\ttok_ziel\tmaske\tprev1\tprev2\tmatch_b"
                                "\tmatch_cls\tprior\n")
                nimm = min(rest, length)
                if self.pk["last"] or self.pk["match"]:
                    _t, _p, _mb, _cl = zpaket.unpack(
                        np.asarray(eingabe[:nimm]), self.pk_x)
                else:
                    _t = np.asarray(eingabe[:nimm], dtype=np.int64)
                    _p = [np.zeros(nimm, np.int64), np.zeros(nimm, np.int64)]
                    _mb = np.full(nimm, 208, np.int64)
                    _cl = np.zeros(nimm, np.int64)
                _m = bewerte[:nimm]
                with open(_dmp, "a") as f:
                    for i in range(nimm):
                        f.write("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d"
                                "\t%d\t%d\t%s\n"
                                % (article, i, int(src_zeile[i]), int(_t[i]),
                                   int(ziel[i]), int(ziel_tok[i]),
                                   int(_m[i]), int(_p[0][i]),
                                   int(_p[1][i] if len(_p) > 1 else 0),
                                   int(_mb[i]), int(_cl[i]),
                                   "prior0" if ziel[i] < 0 else str(int(ziel[i]) - 1)))
                object.__setattr__(self, "_dump_rest", rest - nimm)

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
