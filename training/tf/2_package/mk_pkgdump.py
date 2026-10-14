#!/usr/bin/env python3
r"""Builds a training package from a coder row dump (dump format v2)

    python3 mk_pkgdump.py <zeilen.bin> <target-directory> [options]

      --gate   <gate-mask.uint8>    zone mask written by the dump run, cut to
                                    the package (required)
      --tokens <ppmd-bytes.uint8>   token stream (for n_tokens; not copied)
      --n-tokens N                  number of tokens of the package
      --skip N --shift N            rows and stream positions in front of the
                                    package (header of an excerpt, or the
                                    metadata block of the full stream)
      --v 208 --nsent 8 --x 2
      --x0row / --redirmerge        build state of the dump (goes into the
                                    manifest)
      --arme syn,alias,z3,...       prompt/channel options of the build (go
                                    into the manifest of the weights)
      --fenster 2048                training window (bookkeeping + manifest)

Writes: `zeilen.dump` (copy/hard link, without header record and excerpt
offset), `doc-rows.int64`, `doc-last-ziel.int32`, `article-boundaries.int32`,
`dump-meta.json`, `kennzahlen.json`.

What is no longer produced, which is the point: `pk/` (title-lengths,
reinject-*, sent-*, z1t/syn/z3 prefixes), `gate-mask` and
`match-bytes`/`match-len` in the package.  All these tables were the Python
reconstruction of the coder's rules; the dump carries their result row by
row.  What remains is the prior table (not a coder product but the PPMd
state) and the token stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpv2


def sha256(pfad: str) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("ziel")
    p.add_argument("--gate", default=None)
    p.add_argument("--tokens", default=None)
    # On the full stream n_tokens need not be the length of the zone mask: the
    # data files of the package are cut at the metadata block.  The loader
    # derives n_tokens from `ppmd-bytes` and requires
    # `boundaries[-1] == n_tokens`; a wrong number here stops it at load time.
    p.add_argument("--n-tokens", dest="n_tokens", type=int, default=None)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--v", type=int, default=208)
    p.add_argument("--nsent", type=int, default=8)
    p.add_argument("--x", type=int, default=2)
    p.add_argument("--x0row", action="store_true")
    p.add_argument("--redirmerge", action="store_true")
    p.add_argument("--arme", default="")
    p.add_argument("--fenster", type=int, default=1024)
    # The piece cut counts fed rows, not coded tokens, on both sides: the
    # rotary table of the coder has 131,072 positions, and the number of
    # positions of a document is coded plus inserted rows.  On the full stream
    # 14 documents were above that (the longest 146,148 rows, together 20,029
    # rows beyond the limit).  Here the package build cuts.
    p.add_argument("--stueck", type=int, default=131072,
                   help="rows per piece (0 = do not cut)")
    p.add_argument("--prioren", default=None,
                   help="canonical path of the prior table (goes into the manifest)")
    p.add_argument("--prioren-sha", dest="prioren_sha", default=None)
    a = p.parse_args()

    os.makedirs(a.ziel, exist_ok=True)
    daten, kopf, n_kopf = dumpv2.lade(a.dump, skip=a.skip)
    f = dumpv2.felder(daten, shift=a.shift)
    n = len(f["src"])
    marke = (a.v + a.nsent) if a.x0row else 0

    print("Header record              : %s" % (kopf or "NONE (format v1)"))
    print("Rows                       : %d" % n)
    print("  stream / inserted / SENT : %d / %d / %d"
          % (int((~f["ein"]).sum()), int(f["ein"].sum()),
             int((f["tok"] >= a.v).sum())))
    print("  scored (flags bit1)      : %d" % int(f["maske"].sum()))
    print("  redirect page rows       : %d" % int(f["redir"].sum()))
    print("  prior0 rows (bit4)       : %d" % int(f["prior0"].sum()))
    assert int(np.asarray(daten["res"]).max()) == 0, "reserved != 0"
    p_max = int(max(f["p1"].max(), f["p2"].max()))
    if a.x0row:
        assert p_max == marke, ("X0ROW needs edge mark %d, the dump has %d"
                                % (marke, p_max))
    else:
        assert p_max < a.v + a.nsent, "the dump already carries the X0ROW mark"

    # Memory: p1/p2/mb/mc are not needed after this point, so their two key
    # figures are taken now and the columns released.  On the full stream they
    # are four int16 columns of 1.19 GB each, which would otherwise stay until
    # kennzahlen.json is written and sit exactly in the peak (zielspalten).
    starts, quelle = dumpv2.dokumentstarts(f, marke)
    _match_marke = int((f["mb"] == a.v).sum())
    _match_klasse15 = int((f["mc"] == 15).sum())
    for _sp in ("p1", "p2", "mb", "mc"):
        f.pop(_sp, None)
    doc_rows = np.concatenate([starts, [n]]).astype(np.int64)
    roh_dok = len(doc_rows) - 1
    roh_max = int(np.diff(doc_rows).max())

    # ---- piece cut by fed rows -------------------------------------------
    fortsetzung = np.zeros(len(doc_rows) - 1, dtype=bool)
    schnitte = 0
    if a.stueck:
        L0 = np.diff(doc_rows)
        stuecke = (L0 + a.stueck - 1) // a.stueck        # >= 1 per document
        schnitte = int((stuecke - 1).sum())
        if schnitte:
            neu_rows = np.concatenate([
                doc_rows[d] + np.arange(int(stuecke[d]), dtype=np.int64) * a.stueck
                for d in np.flatnonzero(stuecke > 1)] +
                [doc_rows[:-1][stuecke == 1]])
            neu_rows = np.sort(neu_rows)
            fortsetzung = ~np.isin(neu_rows, doc_rows[:-1])
            doc_rows = np.concatenate([neu_rows, [n]]).astype(np.int64)
        else:
            fortsetzung = np.zeros(len(doc_rows) - 1, dtype=bool)
    n_dok = len(doc_rows) - 1
    laengen = np.diff(doc_rows)
    if schnitte:
        print("Piece cut at %d rows: %d cuts, %d -> %d documents "
              "(longest %d -> %d rows)"
              % (a.stueck, schnitte, roh_dok, n_dok, roh_max,
                 int(laengen.max())))
    else:
        print("Piece cut: no document over %d rows" % a.stueck)
    print("  longest / shortest document: %d / %d rows"
          % (int(laengen.max()), int(laengen.min())))

    # ---- doc-last-ziel ---------------------------------------------------
    # The last fed row of a document targets the last, never fed token, i.e.
    # the next kept stream position.  Only the zone mask knows what is kept, so
    # this is done here and not in the loader.
    lz = doc_rows[1:] - 1
    ls = f["src"][lz]
    ohne = ls < 0
    if ohne.any():
        print("Documents without closing token (body entirely inside a zone): %d"
              % int(ohne.sum()))
    if a.gate:
        kept = ~(np.fromfile(a.gate, dtype=np.uint8) > 0)
        ziel, erste = dumpv2.dokumentschluss(f, doc_rows, kept)
        # A piece followed by another piece of the same original document has
        # no closing token: the transformer is reset at the cut, so its last
        # row predicts nothing.  That costs one scored position per cut (20 of
        # 551 million on the full stream) and is the conservative, provable
        # choice; the final cut point comes from the coder.
        weiter = np.zeros(n_dok, dtype=bool)
        if len(fortsetzung) == n_dok:
            weiter[:-1] = fortsetzung[1:]
        vorher = int((ziel >= 0).sum())
        ziel = np.where(weiter, -1, ziel)
        if int((ziel >= 0).sum()) != vorher:
            print("  piece boundaries without closing token: %d"
                  % (vorher - int((ziel >= 0).sum())))
        vor = np.zeros(len(kept), dtype=bool)
        vor[f["src"][~f["ein"]]] = True
        doppelt = int(vor[ziel[ziel >= 0]].sum())
        print("doc-last-ziel from gate-mask: %d targets, without closing token %d, "
              "of these already fed %d"
              % (int((ziel >= 0).sum()), int((ziel < 0).sum()), doppelt))
        assert doppelt == 0, (
            "%d document closing tokens are already fed rows; the "
            "zone mask does not match the dump" % doppelt)
        ziel = ziel.astype(np.int32)
        n_tokens = len(kept)
    else:
        print("WARNING: without --gate doc-last-ziel stays empty; %d scored "
              "positions (%.4f %%) are lost"
              % (n_dok, 100.0 * n_dok / max(1, int(f["maske"].sum()))))
        ziel = np.full(n_dok, -1, dtype=np.int32)
        n_tokens = None

    # ---- cross-check: flags bit4 against the derived target rule ---------
    z, zt, bew, sent = dumpv2.zielspalten(f, doc_rows, a.v, doc_ziel=ziel)
    if kopf:
        w = dumpv2.gegenprobe_prior0(f, z, doc_rows)
        # Only the direction "bit4 set, but the target rule finds a stream
        # target" is a break of the form.  The other direction (bit4 not set,
        # target row inserted) is the stack of A2 blocks at the end of the
        # stream: there the coder takes the PPMd row of the next stream token,
        # and `priorziel()` follows it.  Measured: 8 rows of 597,639,386, all
        # masked.
        hart = int(((f["prior0"]) & (z >= 0)).sum())
        print("Cross-check flags bit4 against the target rule: %d contradictions "
              "(of these hard, bit4 against a stream target: %d)" % (w, hart))
        assert hart == 0, (
            "%d rows: the coder says prior0, the target rule finds a "
            "stream target; the target rule does not fit this dump" % hart)
        pz = dumpv2.priorziel(f, z)
        weich = int(((~f["prior0"]) & (z < 0) & (pz >= 0)).sum())
        print("  soft cases (bit4 off, target row inserted): %d; the prior "
              "comes from the next stream position" % weich)
    bz = z[bew]
    doppelt = len(bz) - len(np.unique(bz))
    print("Probe0: scored %d, distinct targets %d, repeated %d"
          % (len(bz), len(np.unique(bz)), doppelt))
    assert doppelt == 0, "a target position is scored twice"

    # ---- article-boundaries in stream coordinates ------------------------
    # `erste` comes from dokumentschluss(); documents without a single stream
    # row (body entirely inside a gate zone, exactly 1 on the full stream)
    # carry -1 there.  In the dump mode `article-boundaries` is bookkeeping
    # only: the parent constructor needs n_articles+1 strictly increasing
    # boundaries with `[-1] == n_tokens`, while the content of the documents
    # comes from `doc-rows`.  Such gaps are therefore moved on by 1.
    if not a.gate:
        _, erste = dumpv2.dokumentschluss(
            f, doc_rows, np.ones(n_tokens or 1, dtype=bool))
    e = erste.copy()
    if e[0] < 0:
        e[0] = 0
    geschoben = 0
    for d in range(1, len(e)):
        if e[d] < 0 or e[d] <= e[d - 1]:
            e[d] = e[d - 1] + 1
            geschoben += 1
    if geschoben:
        print("article-boundaries: %d document starts synthetic (no "
              "stream token in the document)" % geschoben)
    erste = e
    assert (np.diff(erste) > 0).all(), "document starts not monotonic"
    print("Documents without closing token in total: %d" % int((ziel < 0).sum()))
    if a.n_tokens:
        n_tokens = a.n_tokens
    elif a.tokens:
        n_tokens = os.path.getsize(a.tokens)
    if n_tokens is None:
        n_tokens = int(max(f["src"].max(), ziel.max())) + 2
    bound = np.concatenate([[0], erste[1:], [n_tokens]]).astype(np.int32)
    assert (np.diff(bound.astype(np.int64)) > 0).all(), "boundaries not monotonic"
    assert len(bound) == n_dok + 1

    # ---- write -----------------------------------------------------------
    doc_rows.tofile(os.path.join(a.ziel, "doc-rows.int64"))
    ziel.tofile(os.path.join(a.ziel, "doc-last-ziel.int32"))
    bound.tofile(os.path.join(a.ziel, "article-boundaries.int32"))
    zpfad = os.path.join(a.ziel, "zeilen.dump")
    # Written block by block: a full copy of the dump (`np.array(daten)`) is
    # 7 GB on the full stream, on top of the memory the steps above need.  A
    # block of 4 Mi records is 50 MB.
    S = 1 << 22
    with open(zpfad, "wb") as fh:
        for i0 in range(0, len(daten), S):
            blk = np.array(daten[i0:i0 + S])
            if a.shift:
                sv = blk["src"]
                np.subtract(sv, np.int32(a.shift), out=sv, where=sv >= 0)
            fh.write(blk.tobytes())

    meta = {
        "dumpformat": kopf.get("version", 1) if kopf else 1,
        "quelle": os.path.abspath(a.dump),
        "quelle_sha256": sha256(a.dump),
        "zeilen_sha256": sha256(zpfad),
        "skip": a.skip, "shift": a.shift,
        "v": a.v, "nsent": a.nsent, "x": a.x,
        "x0row": bool(a.x0row), "redirmerge": bool(a.redirmerge),
        "arme": [s for s in a.arme.split(",") if s],
        "fenster": a.fenster,
        "stueck_zeilen": a.stueck,
        "schnitte": int(schnitte),
        "dokumente_vor_schnitt": int(roh_dok),
        "prioren": a.prioren,
        "prioren_sha256": a.prioren_sha,
        "n_zeilen": int(n), "n_dokumente": int(n_dok),
        "n_tokens": int(n_tokens), "randmarke": int(marke),
        "dokumentquelle": quelle,
    }
    with open(os.path.join(a.ziel, "dump-meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)

    laengen_ueber = int((np.diff(doc_rows) > 131072).sum())
    if laengen_ueber:
        print("*** %d documents over 2^17 = 131,072 ROWS (longest %d); "
              "that is the size of the coder's rope table"
              % (laengen_ueber, int(np.diff(doc_rows).max())))

    kenn = {
        "zeilen": int(n),
        "dokumente_ueber_2hoch17_zeilen": laengen_ueber,
        "stromzeilen": int((~f["ein"]).sum()),
        "eingefuegte_zeilen": int(f["ein"].sum()),
        "sentinelzeilen": int((f["tok"] >= a.v).sum()),
        "redirect_seitenzeilen": int(f["redir"].sum()),
        "prior0_zeilen": int(f["prior0"].sum()),
        "bewertet": int(bew.sum()),
        "dokumente": int(n_dok),
        "dokumente_koerper_ganz_in_zone": int(ohne.sum()),
        "dokumente_ohne_schlusstoken": int((ziel < 0).sum()),
        "laengstes_dokument_zeilen": int(laengen.max()),
        "kuerzestes_dokument_zeilen": int(laengen.min()),
        "match_marke_%d" % a.v: _match_marke,
        "match_klasse_15": _match_klasse15,
        "meta": meta,
    }
    with open(os.path.join(a.ziel, "kennzahlen.json"), "w", encoding="utf-8") as fh:
        json.dump(kenn, fh, indent=1, sort_keys=True)
    print("written to %s" % a.ziel)
    for k in ("zeilen", "stromzeilen", "eingefuegte_zeilen", "sentinelzeilen",
              "redirect_seitenzeilen", "bewertet", "dokumente",
              "laengstes_dokument_zeilen"):
        print("  %-28s %d" % (k, kenn[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
