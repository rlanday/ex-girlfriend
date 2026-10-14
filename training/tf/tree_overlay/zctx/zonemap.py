#!/usr/bin/env python3
"""zone map of the transformer's contribution

Question: should the transformer be switched off for fixed regions such as the
XML meta stream, so it can spend its capacity and time elsewhere?  This file
answers the preliminary question: **where** does the transformer contribute
at all, and **how much stream** lies in zones where it contributes nothing or
something negative.  It builds NO gating; it supplies the map on which one
can be decided.

Measured per token:

    delta_nats(i) = -ln p_TF(t_i)  -  -ln p_PPMD(t_i)

that is the contribution of the transformer AGAINST the PPMd prior it gets as
input.  Negative = the transformer is better.  This is not the contribution
against the finished coder (there fxcm, match and the SSE stages mix in as
well), but against the distribution the same slot would have delivered
without it, which is exactly the quantity a zone gate saves or gives away.

INPUTS (all from the same 10 MB excerpt, all aligned byte for byte)
  dump/ppmd-bytes.uint8     N token indices 0..V-1   (tf1_dump)
  dump/ppmd-probs.float16   N x V PPMd rows           (tf1_dump)
  dump/vocab-bitmap.u8      32 B, index -> byte value (tf1_dump)
  dump/article-boundaries.int32
  tf_probs.float16          N x V transformer rows    (dump build variant of
                            the compressor)
  run_priors.float16        N x V PPMd rows from the REAL run (dump build
                            variant of the compressor); only for the identity
                            check, not for the computation

TWO CONDITIONS, both checked here instead of assumed:
  1. DETERMINISM: the classification depends only on the coder stream, not
     on enwik9 and not on a heuristic with state.  Every rule below is a
     function of the token sequence; the decoder could run it at the same
     place with the same result.  That is the condition for a later gate to
     be buildable at all.
  2. DUMP IDENTITY: the PPMd rows of the standalone dump tool must equal
     those of the real compressor run.  The `check` command checks this byte
     for byte on a sample; without it, two different runs might be measured
     against each other.

ZONE RULES (on the BYTE VALUES of the coder stream, not on enwik9)
The coder stream is dictionary/WRT coded: words become code words >= 0x80,
the structure stays as plain punctuation.  Counted on the excerpt:
`[[` 9,524 / `]]` 9,522 occurrences, `\n` 7,169, `@` 20,198.

  TITEL     from the article separator up to and including TITLE_TERMINATOR
  META      from the end of the title to the start of the text (the XML
            header data: id, revision, timestamp, contributor, comment); the
            end is found by the marker sequence that `zonemap derive` DERIVES
            from the excerpt instead of guessing it
  LINK      between `[[` and `]]`, brackets included
  TABELLE   a line whose first non-blank character is in {* # | ! ; :},
            and everything between `{|` and `|}`
  ZIFFERN   runs of >= 4 digits/separators outside the zones above
            (population figures, coordinates)
  TEXT      the rest

Usage:
  python -m zctx.zonemap derive <dumpdir>              derive the markers
  python -m zctx.zonemap check  <dumpdir> <run_priors> dump identity
  python -m zctx.zonemap map    <dumpdir> <tf_probs>   the map
"""

from __future__ import annotations

import os
import sys

import numpy as np

# from zctx.zgrammar, where they are recomputed on the package
ARTICLE_SEPARATOR = [11, 11, 40, 175, 104, 42, 8, 11, 11, 11, 11, 40, 175, 107, 42]
TITLE_TERMINATOR = [40, 26, 175, 107, 42]

ZONES = ("TITEL", "META", "LINK", "TABELLE", "ZIFFERN", "TEXT")
LINE_MARKERS = set(b"*#|!;:")
DIGITS = set(b"0123456789")


def load_bitmap(directory: str) -> np.ndarray:
    """index -> byte value, ascending"""
    bm = np.fromfile(os.path.join(directory, "vocab-bitmap.u8"), dtype=np.uint8)
    values = [b * 8 + k for b in range(32) for k in range(8) if bm[b] >> k & 1]
    return np.array(values, dtype=np.uint8)


def find_all(tokens: np.ndarray, pattern: list) -> np.ndarray:
    """all start positions of a token sequence (vectorised, no Python loop)"""
    n, m = len(tokens), len(pattern)
    if n < m:
        return np.zeros(0, dtype=np.int64)
    hit = tokens[: n - m + 1] == pattern[0]
    for k in range(1, m):
        hit &= tokens[k : n - m + 1 + k] == pattern[k]
    return np.flatnonzero(hit)


def derive_text_marker(tokens: np.ndarray, boundaries: np.ndarray, width: int = 8) -> list:
    """the marker sequence that separates the XML header from the article text

    Derived, not guessed: the token sequence of length `width` that occurs
    exactly once in the region behind the end of the title in AS MANY
    articles AS POSSIBLE; that is `<text xml:space="preserve">` in its coded
    form.  A gate built on a guessed marker would fail silently.
    """
    starts = []
    for i in range(len(boundaries) - 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        t = find_all(tokens[a : min(a + 256, b)], TITLE_TERMINATOR)
        if len(t):
            starts.append(a + int(t[0]) + len(TITLE_TERMINATOR))
    counts: dict = {}
    for s in starts[:400]:
        window = tokens[s : s + 400]
        seen = set()
        for k in range(0, max(0, len(window) - width)):
            key = tuple(int(v) for v in window[k : k + width])
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    best, hits = max(counts.items(), key=lambda kv: kv[1])
    return [list(best), hits, len(starts[:400])]


def classify(tokens: np.ndarray, boundaries: np.ndarray, bitmap: np.ndarray,
             text_marker: list) -> np.ndarray:
    """(N,) uint8 zone index per token; a pure function of the token sequence"""
    n = len(tokens)
    bytes_ = bitmap[tokens]
    zone = np.full(n, ZONES.index("TEXT"), dtype=np.uint8)

    # --- LINK: between [[ and ]] ----------------------------------------
    br_open = (bytes_[:-1] == 0x5B) & (bytes_[1:] == 0x5B)
    br_close = (bytes_[:-1] == 0x5D) & (bytes_[1:] == 0x5D)
    depth = np.cumsum(np.pad(br_open.astype(np.int32), (0, 1))) - np.cumsum(
        np.pad(br_close.astype(np.int32), (1, 0))
    )
    zone[depth[:n] > 0] = ZONES.index("LINK")

    # --- TABELLE: lines with a list or table mark ------------------------
    nl = np.flatnonzero(bytes_ == 0x0A)
    line_start = np.concatenate([[0], nl + 1])
    line_end = np.concatenate([nl, [n]])
    for s, e in zip(line_start, line_end):
        k = s
        while k < e and bytes_[k] == 0x20:
            k += 1
        if k < e and int(bytes_[k]) in LINE_MARKERS:
            zone[s:e] = ZONES.index("TABELLE")

    # --- ZIFFERN: runs of >= 4 digits -------------------------------------
    isdig = np.isin(bytes_, np.fromiter(DIGITS, dtype=np.uint8))
    d = np.diff(np.concatenate([[0], isdig.view(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    for a, b in runs:
        if b - a >= 4 and zone[a] == ZONES.index("TEXT"):
            zone[a:b] = ZONES.index("ZIFFERN")

    # --- TITEL and META: per article --------------------------------------
    tm = list(text_marker) if text_marker else []
    for i in range(len(boundaries) - 1):
        a, b = int(boundaries[i]), int(boundaries[i + 1])
        t = find_all(tokens[a : min(a + 256, b)], TITLE_TERMINATOR)
        title_end = a + int(t[0]) + len(TITLE_TERMINATOR) if len(t) else a
        zone[a:title_end] = ZONES.index("TITEL")
        meta_end = title_end
        if tm:
            m = find_all(tokens[title_end : min(title_end + 2000, b)], tm)
            if len(m):
                meta_end = title_end + int(m[0]) + len(tm)
        zone[title_end:meta_end] = ZONES.index("META")
    return zone


def nats(probs_path: str, tokens: np.ndarray, v: int, chunk: int = 200_000) -> np.ndarray:
    """-ln p(true token) per position, streamed (the file is 2.7 GB)

    THE PAIRING IS SHIFTED BY ONE, as the table of `ppmd-probs.float16` in
    training/README.md warns (row i is the PPMD's distribution over token i+1,
    and an off-by-one is silent): row i predicts token i+1.  Pairing row i
    with `tokens[i]` gives 10.9 instead of 1.75 nats per token, which is
    noticed only against a known PPMd reference value.
    """
    tokens = np.asarray(tokens)[1:]          # the target of row i is token i+1
    n = len(tokens)
    out = np.empty(n, dtype=np.float32)
    with open(probs_path, "rb") as f:
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            raw = np.fromfile(f, dtype=np.float16, count=(b - a) * v)
            if len(raw) < (b - a) * v:
                out = out[:a]
                break
            rows = raw.reshape(b - a, v).astype(np.float32)
            p = rows[np.arange(b - a), tokens[a:b]]
            out[a:b] = -np.log(np.maximum(p, 1e-9))
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, d = sys.argv[1], sys.argv[2]
    tokens = np.fromfile(os.path.join(d, "ppmd-bytes.uint8"), dtype=np.uint8).astype(np.int64)
    boundaries = np.fromfile(os.path.join(d, "article-boundaries.int32"), dtype=np.int32)
    bitmap = load_bitmap(d)
    v = len(bitmap)

    if cmd == "derive":
        marker, hits, tried = derive_text_marker(tokens, boundaries)
        print("text marker (%d tokens): %s" % (len(marker), marker))
        print("found in %d of %d checked articles (%.1f %%)"
              % (hits, tried, 100.0 * hits / max(1, tried)))
        return 0

    if cmd == "check":
        run_priors = sys.argv[3]
        ref = os.path.join(d, "ppmd-probs.float16")
        a_sz, b_sz = os.path.getsize(ref), os.path.getsize(run_priors)
        print("tf1_dump  ppmd-probs : %d B" % a_sz)
        print("run       run_priors : %d B" % b_sz)
        row = v * 2
        n = min(a_sz, b_sz) // row
        rng = np.random.default_rng(1)
        idx = np.sort(rng.choice(n, size=min(2000, n), replace=False))
        bad = 0
        with open(ref, "rb") as fa, open(run_priors, "rb") as fb:
            for i in idx:
                fa.seek(int(i) * row); fb.seek(int(i) * row)
                if fa.read(row) != fb.read(row):
                    bad += 1
        print("sample of %d rows, unequal: %d" % (len(idx), bad))
        print("DUMP IDENTITY: %s" % ("OK" if bad == 0 else "VIOLATED"))
        return 0 if bad == 0 else 1

    if cmd == "map":
        tf_probs = sys.argv[3]
        marker, hits, tried = derive_text_marker(tokens, boundaries)
        zone = classify(tokens, boundaries, bitmap, marker)
        tf = nats(tf_probs, tokens, v)
        pp = nats(os.path.join(d, "ppmd-probs.float16"), tokens, v)
        # the zones belong to the PREDICTED token, so they shift by one as well
        zone = zone[1:]
        m = min(len(tf), len(pp), len(zone))
        tf, pp, zone = tf[:m], pp[:m], zone[:m]
        delta = tf - pp
        print("tokens scored : %d of %d" % (m, len(tokens)))
        print("PPMD total    : %.6f nats/token" % pp.mean())
        print("TF   total    : %.6f nats/token" % tf.mean())
        print("contribution  : %+.6f nats/token  (negative = TF better)" % delta.mean())
        print()
        print("%-9s %9s %7s %12s %12s %14s" %
              ("zone", "tokens", "share", "PPMD", "TF", "contrib/token"))
        rows = []
        for zi, name in enumerate(ZONES):
            sel = zone == zi
            k = int(sel.sum())
            if not k:
                continue
            rows.append((name, k, 100.0 * k / m, pp[sel].mean(), tf[sel].mean(),
                         delta[sel].mean(), delta[sel].sum()))
        for r in rows:
            print("%-9s %9d %6.2f%% %12.5f %12.5f %+14.6f" % r[:6])
        print()
        print("%-9s %14s %10s" % ("zone", "sum nats", "share"))
        tot = sum(r[6] for r in rows)
        for r in sorted(rows, key=lambda r: r[6]):
            print("%-9s %+14.1f %9.1f%%" % (r[0], r[6], 100.0 * r[6] / tot if tot else 0))
        print()
        dead = [r for r in rows if r[5] > -0.005]
        share = sum(r[2] for r in dead)
        print("TIME POOL: zones with contribution >= -0.005 nats/token: %s" %
              (", ".join(r[0] for r in dead) or "none"))
        print("           = %.2f %% of all tokens" % share)
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
