#!/usr/bin/env python3
r"""Reader of the dump record format v2

One module for all tools: package builder, acceptance test and loader read the
same code, so that they cannot drift apart.

The contract is the record format written by the dump mode that
`training/tf/1_dump/trainmodes.patch` adds to a copy of the entry's source
(`cmix_dump`, `cmix_priors`; `training/tf/1_dump/README.md` section 2
describes it); this module only says what the reader makes of it.

    Record 0 is a header record: `src == -2`, then 'P' 'K' 'D', format
    version, record size, number of used flag bits.
    Data records: int32 src (-1 = inserted) | u8 tok prev1 prev2 match_b
                  match_cls flags | u16 reserved

    flags: bit0 inserted · bit1 MASKED · bit2 DOCUMENT START ·
           bit3 redirect page · bit4 prior0 row (no stream target) ·
           bit5 reinjected alias or author byte · bit6 author byte ·
           bit7 metadata prompt row

Target rule:

    The target of row j is the next stream row (src >= 0) of the same
    document, past all inserted rows; `flags` bit4 says whether the row has
    a stream target at all (bit4 = the row gets prior0 instead of a PPMd row).
    The last row of a document targets the document's closing token (coded,
    never fed); it is not in the dump and comes from `doc-last-ziel.int32`
    (package build) or from the zone mask.

    It replaces the older rule "row j+1, or j+2 behind a sentinel row".  In
    the stream form of the shipped weights the metadata and redirect prompt
    rows form a block in front of the title of each page, so at the end of a
    page row j+1 is a prompt row (src = -1).  The older rule then gave the
    last stream row of the previous page no target and a prompt token as
    target token, while the coder does not set bit4 there.  On 1 MB and
    20 MB excerpts this affected 231 to 72,912 rows (2 in the older stream
    form), all of them masked.  On the older form the two rules differ only
    in these rows; sentinel rows are skipped because they are inserted rows
    (no sentinel row with src >= 0 in any dump checked).

German names: Satz = record, Kopf = header, Zeile = row, Ziel = target,
eingefuegt = inserted, bewerte = scored, Dokument = document.
"""

from __future__ import annotations

import os

import numpy as np

REC = 12
SATZ = np.dtype([
    ("src", "<i4"), ("tok", "u1"), ("p1", "u1"), ("p2", "u1"),
    ("mb", "u1"), ("mc", "u1"), ("fl", "u1"), ("res", "<u2"),
])
assert SATZ.itemsize == REC

EINGEFUEGT, MASKE, DOKANFANG, REDIRSEITE, PRIOR0 = 1, 2, 4, 8, 16
VERSION = 2


def kopf_pruefen(roh: np.ndarray) -> dict:
    """read and check the header record; returns {} if there is none"""
    if len(roh) == 0 or int(roh["src"][0]) != -2:
        return {}
    r = roh[0]
    magie = bytes([int(r["tok"]), int(r["p1"]), int(r["p2"])])
    assert magie == b"PKD", "header record without PKD mark: %r" % magie
    kopf = {"version": int(r["mb"]), "satz": int(r["mc"]), "flagbits": int(r["fl"])}
    assert kopf["satz"] == REC, "record size %d, this reader handles 12" % kopf["satz"]
    assert kopf["version"] == VERSION, (
        "dump format version %d, this reader knows version %d"
        % (kopf["version"], VERSION))
    return kopf


def lade(pfad: str, *, skip: int = 0, shift: int = 0, mmap: bool = True):
    """(daten, kopf, n_kopfsaetze)

    `skip`/`shift` correct for a stream excerpt: the coder writes a 6-byte
    segment header before an excerpt, i.e. six extra rows, and its stream
    positions are 6 higher than those of the training package.  On the full
    stream both are 0.
    """
    groesse = os.path.getsize(pfad)
    assert groesse % REC == 0, (
        "%s is %d B long, not a multiple of %d; truncated dump?"
        % (pfad, groesse, REC))
    roh = np.memmap(pfad, dtype=SATZ, mode="r") if mmap else np.fromfile(pfad, SATZ)
    kopf = kopf_pruefen(roh)
    n_kopf = 1 if kopf else 0
    daten = roh[n_kopf + skip:]
    return daten, kopf, n_kopf


def felder(daten: np.ndarray, shift: int = 0) -> dict:
    """the columns as narrow fields

    The dtypes matter: the full stream has 597,639,386 rows, and eight int64
    columns would take 38 GB, more than the build machine had.  `src` fits
    into int32 (stream position < 2^31, -1/-2 as marks), the token columns
    into int16, the length class into int8, the flags into bool; together
    ~11 GB.  Callers that compute with them cast themselves, as `packe()` and
    the target rule do.
    """
    src = np.array(daten["src"], dtype=np.int32)   # copy: the memmap is read-only
    if shift:
        np.subtract(src, np.int32(shift), out=src, where=src >= 0)
    fl = np.asarray(daten["fl"])
    return dict(
        src=src,
        tok=np.asarray(daten["tok"], dtype=np.int16),
        p1=np.asarray(daten["p1"], dtype=np.int16),
        p2=np.asarray(daten["p2"], dtype=np.int16),
        mb=np.asarray(daten["mb"], dtype=np.int16),
        mc=np.asarray(daten["mc"], dtype=np.int8),
        fl=fl,
        ein=(fl & EINGEFUEGT) != 0,
        maske=(fl & MASKE) == 0,          # scored == not masked
        anfang=(fl & DOKANFANG) != 0,
        redir=(fl & REDIRSEITE) != 0,
        prior0=(fl & PRIOR0) != 0,
    )


def dokumentstarts(f: dict, marke: int, laut=True) -> np.ndarray:
    """row indices of the document starts, from the best available source"""
    n = len(f["src"])
    if f["anfang"].any():
        idx = np.flatnonzero(f["anfang"])
        quelle = "flags bit2"
    else:
        idx = np.flatnonzero((f["p1"] == marke) & (f["p2"] == marke))
        quelle = ("X2 edge mark %d" % marke) + (
            " (stage-1 form, not provably unique)" if marke == 0 else "")
    if len(idx) == 0 or idx[0] != 0:
        # Row 0 is always a document start; on a stream excerpt the start of
        # the first document falls under `skip` (stream header).
        idx = np.concatenate([[0], idx])
    if laut:
        print("dump: %d documents (source: %s)" % (len(idx), quelle), flush=True)
    return idx, quelle


def dokumentschluss(f: dict, doc_rows: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """stream position of the last, never fed token of each document (-1 = none)

    That is the next kept stream position behind the last fed row, **unless it
    already belongs to the next document**.  The second case is real and not
    rare: the 2^17 piece cut counts coded tokens (`s.article_tokens`, gated
    ones included), so it can fall on a gated position.  Then the document has
    no kept token behind its last fed row, and that row gets no target.
    Measured on a 1 MB excerpt with merged redirect runs: 2 of 9 documents.

    Side finding for the coder: in exactly these two cases the coder leaves the
    last row UNMASKED (`flags` bit1 = 0) and without bit4, although no token of
    the document follows it; the loader has to mute it here.
    """
    n = len(f["src"])
    kum = np.concatenate([[0], np.cumsum(kept, dtype=np.int64)])
    idx_kept = np.flatnonzero(kept)
    lz = doc_rows[1:] - 1
    ls = f["src"][lz].astype(np.int64)
    ohne = ls < 0                       # body entirely inside a zone
    k = kum[np.where(ohne, 0, ls).astype(np.int64) + 1]
    gut = (k < len(idx_kept)) & ~ohne
    dz = np.where(gut, idx_kept[np.where(gut, k, 0)], -1)
    # Does the token found already belong to the next document?  One pass
    # instead of 172,449 slices: `nxt` holds the next stream position from
    # every row, computed like `naechste_idx` in priorziel().
    pos = np.where(f["src"] >= 0, np.arange(n, dtype=np.int64), n)
    nxt = np.minimum.accumulate(pos[::-1])[::-1]
    a0 = doc_rows[:-1]
    j = nxt[a0]
    erste = np.where(j < doc_rows[1:],
                     f["src"][np.minimum(j, n - 1)].astype(np.int64), -1)
    naechster = np.concatenate([erste[1:], [n + len(kept)]])
    dz = np.where((dz >= 0) & (dz >= naechster), -1, dz)
    return dz.astype(np.int64), erste


def naechste_stromzeile(src: np.ndarray) -> np.ndarray:
    """index of the next row k > j with `src[k] >= 0` (len(src) if none)

    A suffix minimum over the positions of the stream rows, the computation
    that `priorziel()` and `dokumentschluss()` use as well.  The caller checks
    the document boundary.
    """
    n = len(src)
    pos = np.where(src >= 0, np.arange(n, dtype=np.int64), n)
    suffmin = np.minimum.accumulate(pos[::-1])[::-1]
    naechste = np.full(n, n, dtype=np.int64)
    if n > 1:
        naechste[:-1] = suffmin[1:]
    return naechste


def zielspalten(f: dict, doc_rows: np.ndarray, v: int, doc_ziel=None,
                strom_tok=None):
    """(ziel, ziel_tok, bewerte, sent) for all rows

    `doc_ziel` (one value per document) and `strom_tok` supply the closing
    token of each document; without them the last row of each document stays
    unscored.
    """
    n = len(f["src"])
    tok, src = f["tok"], f["src"]
    sent = tok >= v
    lz = doc_rows[1:] - 1
    letzte = np.zeros(n, dtype=bool)
    letzte[lz] = True
    assert not sent[lz].any(), "a document ends on a sentinel row"
    # next stream row behind j, past all inserted rows (one suffix minimum)
    naechste = naechste_stromzeile(src)
    # it must lie in the same document
    dok_ende = np.repeat(doc_rows[1:], np.diff(doc_rows))
    hat = (naechste < dok_ende) & ~f["prior0"] & ~letzte
    sicher = np.minimum(naechste, n - 1)
    ziel = np.where(hat, src[sicher].astype(np.int64), np.int64(-1))
    ziel_tok = np.where(hat, tok[sicher].astype(np.int64), np.int64(0))
    if doc_ziel is not None:
        dz = np.asarray(doc_ziel, dtype=np.int64)
        ziel[lz] = dz
        if strom_tok is not None:
            ziel_tok[lz] = np.where(dz >= 0, strom_tok[np.where(dz >= 0, dz, 0)], 0)
    # A row without a stream target is never scored; `output_tokens` would
    # hold a 0 or a prompt token there.  Besides the last row of a document
    # this occurs once per full stream, at the start of the metadata block.
    bewerte = f["maske"] & (ziel >= 0)
    return ziel, ziel_tok, bewerte, sent


def priorziel(f: dict, ziel: np.ndarray) -> np.ndarray:
    """stream position whose PPMd row this row gets as its prior (-1 = prior0)

    Almost always this is `ziel` itself.  The difference arises where the coder
    does not set `flags` bit4 but the target row is inserted: then it uses the
    PPMd row of the next stream token, not prior0.  Measured on the full
    stream: 8 rows of 597,639,386, all at the end of the stream and all the
    last row of an A2 block that is directly followed by another A2 block (the
    coder looks only one block ahead for the prior).  All eight are masked, so
    the loss is not affected, but the prior is an input and the coder is the
    specification.

    bit4 decides: set -> prior0, else -> next stream position from the target
    row on.
    """
    n = len(ziel)
    pos = np.where(f["src"] >= 0, np.arange(n, dtype=np.int64), n)
    naechste_idx = np.minimum.accumulate(pos[::-1])[::-1]
    naechste_src = np.where(
        naechste_idx < n,
        f["src"][np.minimum(naechste_idx, n - 1)].astype(np.int64), -1)
    return np.where(f["prior0"], -1, np.where(ziel >= 0, ziel, naechste_src))


def gegenprobe_prior0(f: dict, ziel: np.ndarray, doc_rows: np.ndarray) -> int:
    """flags bit4 against the structure of the dump; 0 means they agree

    With the target rule above, `ziel < 0` follows from bit4 by construction,
    so comparing the two would prove nothing.  The check compares bit4 with
    "row j+1 is not a stream row" instead, with one allowed exception: the
    prompt block case (bit4 = 0, and a stream row of the same document still
    follows, but only behind inserted rows).  Anything else is a break: bit4
    set although a stream row follows directly, or bit4 clear without any
    further stream row in the document.

    The last row of each document is excluded: its target is not in the dump
    but comes from `doc-last-ziel`.
    """
    aus = np.ones(len(ziel), dtype=bool)
    aus[doc_rows[1:] - 1] = False
    n = len(ziel)
    src = f["src"]
    naechste = naechste_stromzeile(src)
    dok_ende = np.repeat(doc_rows[1:], np.diff(doc_rows))
    unmittelbar = np.zeros(n, dtype=bool)
    if n > 1:
        unmittelbar[:-1] = src[1:] >= 0
    ohne_j1 = ~unmittelbar
    erlaubt = ohne_j1 & ~f["prior0"] & (naechste < dok_ende)
    bruch = (ohne_j1 != f["prior0"]) & ~erlaubt
    return int(bruch[aus].sum())
