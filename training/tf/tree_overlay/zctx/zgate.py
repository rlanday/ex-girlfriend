#!/usr/bin/env python3
"""Digit gating: the CAUSAL zone rule, its calibration and its integration

Background: the zone map (`zctx/zonemap.py`, dump identity 0/2000) measured
that the transformer is **+0.396 nats per token WORSE** in the DIGIT zones than
the PPMd prior it replaces: 1.92 % of the tokens, +49,737 nats of damage on a
10 MB stream excerpt, i.e. -20.5 % of the whole gain of the transformer.
Switching it off there gains rate AND time.  This file takes the step from the
map to the evidence: the rule that a DECODER can apply at the same place with
the same result, and the document construction with which a transformer is
trained on the gated stream.

THE DIFFERENCE TO THE MAP IS CAUSALITY, and it costs something.  The map
classifies "runs of >= 4 digits".  That is a statement ABOUT THE WHOLE RUN and
therefore not available in the decoder: whoever sees the first character does
not yet know whether three more follow.  Only the past is available.  The
causal rule is therefore

    gate(i)  <=>  the last M tokens before the prediction are all digits

and it has two deviations from the map that have to be put in numbers:

  * RAMP-UP: the first M characters of a run stay ungated; their damage
    remains.
  * OVERHANG: the first character AFTER the run is gated as well, because at
    its prediction time the last M tokens are still digits.  This character
    is typically easy to predict; here the rule gives something away.

`zgate calib` measures both effects on the real material instead of
estimating them: for every M it computes the balance from the MEASURED nats
of both sides.

THE TWO ARMS

  a) "blind"     the gated position drops out of the model run completely.
                 The transformer no longer sees the digits, neither as target
                 nor as context.  Only this saves real time (a position that
                 is merely masked is still computed).
  b) "sentinel"  per skipped run ONE extra input token that carries the
                 LENGTH CLASS of the run.  Both sides can insert it
                 deterministically: as soon as the first non-digit character
                 is decoded, coder and decoder know the run length.

    The position sequence of a run t_s..t_e (arm b), with t_{s-1} before it:

        input t_{s-1}   -> target t_s      (scored: the transformer still
                                            predicts the FIRST digit)
        input SENT_c    -> target t_{e+1}  (NOT scored: gated, here the
                                            ensemble pays the prior)
        input t_{e+1}   -> target t_{e+2}  (scored, with the length known)

    Arm a is the same sequence without the sentinel row.

    The sentinel indices lie ABOVE the vocabulary (V, V+1, ...): they are
    pure model inputs, never occur in the stream and therefore cost NO
    alphabet tax.  Only K x d_model weights are paid for.
    Index 0 stays free; arm z4 uses it as its edge mark.

Usage:
  python -m zctx.zgate selftest
  python -m zctx.zgate calib   <dumpdir> <tf_probs.float16> [M list]
  python -m zctx.zgate pkgstat <pkgdir> [M list]
"""

from __future__ import annotations

import os
import sys

import numpy as np

DIGIT_BYTES = tuple(range(0x30, 0x3A))          # '0'..'9'

# Length classes of the sentinel.  The bounds are upper bounds (inclusive);
# the last class catches everything above.  Seven bounds cover the
# distribution of digit runs in enwik9 (1..3 short, 4 year, 5..8 number,
# above that table).
LENGTH_CLASS_BOUNDS = (1, 2, 3, 4, 6, 9, 15)
N_LENGTH_CLASSES = len(LENGTH_CLASS_BOUNDS) + 1  # = 8


def length_class(run_length: int) -> int:
    for k, bound in enumerate(LENGTH_CLASS_BOUNDS):
        if run_length <= bound:
            return k
    return len(LENGTH_CLASS_BOUNDS)


def load_bitmap(directory: str) -> np.ndarray:
    """token index -> byte value (identical to zonemap.load_bitmap)"""
    bm = np.fromfile(os.path.join(directory, "vocab-bitmap.u8"), dtype=np.uint8)
    values = [b * 8 + k for b in range(32) for k in range(8) if bm[b] >> k & 1]
    return np.array(values, dtype=np.uint8)


def digit_token_mask(bitmap: np.ndarray) -> np.ndarray:
    """(V,) bool: which TOKEN INDICES are digit bytes

    The rule works on token indices, not on bytes: the decoder has the index,
    and the mapping through the bitmap is a constant of the package.
    """
    return np.isin(bitmap.astype(np.int64), np.array(DIGIT_BYTES, dtype=np.int64))


def gate_from_digits(is_digit: np.ndarray, m: int) -> np.ndarray:
    """(n,) bool: gate[i] = the prediction AT position i is skipped

    Position i has input t_i and predicts t_{i+1}.  It is gated when t_i,
    t_{i-1}, ..., t_{i-m+1} are all digits.  A pure function of the past, so
    the decoder can evaluate it at the same place.
    """
    assert m >= 1
    n = len(is_digit)
    gate = is_digit.copy()
    for k in range(1, m):
        shifted = np.zeros(n, dtype=bool)
        if n > k:
            shifted[k:] = is_digit[:-k]
        gate &= shifted
    return gate


def runs_of(is_digit: np.ndarray) -> list:
    """[(start, end_exclusive)] of all digit runs"""
    d = np.diff(np.concatenate([[0], is_digit.view(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# --------------------------------------------------------------------------
# The exclusion: where NOT to gate although there are digits
# --------------------------------------------------------------------------
# The zone cross-check sharpened the rule: the digit damage of the
# transformer sits in TEXT and in the DIGIT zone, but in LINK and TABLE the
# transformer is STRONG on digits as well, and gating costs there (+3,032 /
# +4,905 nats at M=1).  Excluding these two zones raises the gain at M=2 from
# -62,862 to -69,044 nats, beyond the M=1 value, with half as many gated
# positions.
#
# Both zones can be recognised causally, by an automaton over the already
# decoded past, the same state that arm z4 gives the model as input.
#   LINK     bracket depth [[ ... ]] > 0, a prefix sum, purely causal
#   TABLE    the current line began (after spaces) with * # | ! ; :
LINE_MARKER_BYTES = tuple(b"*#|!;:")


def exclusion_mask(tokens: np.ndarray, bitmap: np.ndarray) -> np.ndarray:
    """(n,) bool: positions at which NOT to gate (state after t_i)

    The state AFTER reading t_i is evaluated, because exactly that is known
    when t_{i+1} is predicted.
    """
    b = bitmap[np.asarray(tokens, dtype=np.int64)]
    n = len(b)
    # LINK: depth after reading t_i.  `[[` is counted at the SECOND character,
    # and so is `]]`; both are the earliest causal point.
    op = np.zeros(n, dtype=np.int32)
    cl = np.zeros(n, dtype=np.int32)
    if n > 1:
        op[1:] = ((b[:-1] == 0x5B) & (b[1:] == 0x5B)).astype(np.int32)
        cl[1:] = ((b[:-1] == 0x5D) & (b[1:] == 0x5D)).astype(np.int32)
    # The counter has a FLOOR.  A plain prefix sum goes negative at an
    # unbalanced closer and never comes back, and the link exclusion is dead
    # for the rest of the stream.  Measured on the package: 11,978 openers
    # against 12,371 closers in the first 20 MB, where the LINK zone came to
    # 0.00 %.  d[i] = max(0, d[i-1] + open - close) can be vectorised as
    # s - min(0, running minimum of s).
    s = np.cumsum(op) - np.cumsum(cl)
    depth = s - np.minimum.accumulate(np.minimum(s, 0))
    excl = depth > 0

    # TABLE: start of line, then optional spaces, then a marker.  Fully
    # vectorised; a Python loop over the ~7 million lines of the 200 M
    # package would be the bottleneck of the whole tool.
    idx = np.arange(n, dtype=np.int64)
    last_nl = np.maximum.accumulate(np.where(b == 0x0A, idx, -1))
    nonspace = b != 0x20
    cs = np.cumsum(nonspace, dtype=np.int64)
    base = np.where(last_nl >= 0, cs[np.maximum(last_nl, 0)], 0)
    first_ns = nonspace & ((cs - base) == 1)
    marker = np.isin(b, np.array(LINE_MARKER_BYTES, dtype=np.uint8))
    last_marker = np.maximum.accumulate(np.where(first_ns & marker, idx, -1))
    excl |= last_marker > last_nl     # from the marker on; before it nobody knows
    return excl


def gate_mask(tokens: np.ndarray, bitmap: np.ndarray, m: int,
              exclude: bool = True) -> np.ndarray:
    """(n,) bool: the complete causal gate rule of the automaton

    gate(i) = the last M tokens are digits AND the state is neither "inside a
    link" nor "in a table or list line".
    """
    is_digit = digit_token_mask(bitmap)[np.asarray(tokens, dtype=np.int64)]
    gate = gate_from_digits(is_digit, m)
    if exclude:
        gate &= ~exclusion_mask(tokens, bitmap)
    return gate


# --------------------------------------------------------------------------
# Document construction (the training side)
# --------------------------------------------------------------------------

def gate_rows(gate: np.ndarray, sentinel: bool) -> tuple:
    """the rows of the gated document, as an INDEX LIST rather than as data

    `gate` is the gate mask over the positions of ONE document (length L-1:
    position i has input doc[i] and predicts doc[i+1]).  Three arrays of
    equal length are returned:

        src   per row the original position that prior AND target come from
        sent  True for a sentinel row
        cls   its length class (0 where sent=False)

    Loader and reference stay together because the sentinel row of a block
    [a,b) carries src = b-1, i.e. the prior row of the LAST gated position and
    with it the same target doc[b] that the transformer has just not
    predicted.  So for EVERY row target = doc[src+1] and prior = priors[src];
    there is no special case in the index arithmetic.

    A sentinel row is only inserted when a scored row follows its block: at
    the end of a document it carries nothing and only costs compute time, and
    the decoder recognises the end of the article just as well.
    """
    keep_idx = np.flatnonzero(~gate)
    if not sentinel:
        return keep_idx, np.zeros(len(keep_idx), dtype=bool), np.zeros(len(keep_idx), np.int64)

    d = np.diff(np.concatenate([[0], gate.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    keepb = ends < len(gate)           # a block at the end of the document gets no row
    starts, ends = starts[keepb], ends[keepb]
    if not len(ends):
        return keep_idx, np.zeros(len(keep_idx), dtype=bool), np.zeros(len(keep_idx), np.int64)

    # Vectorised through a sort key instead of list.insert: a kept position i
    # gets 2*i, the sentinel row of a block [a,b) gets 2*b-1.  It thus lands
    # exactly between the last gated position (2b-2) and the first kept one
    # after it (2b), the same order as the insertion, but in one sort step.
    # Measured: 22.3 ms -> 0.4 ms per micro-batch, about 10 % of the training
    # time that would otherwise have been spent in the loader.
    starts = starts.astype(np.int64)
    ends = ends.astype(np.int64)
    # the same classes as `length_class`, only vectorised: searchsorted on
    # the upper bounds returns exactly their index.
    cls_blocks = np.searchsorted(
        np.array(LENGTH_CLASS_BOUNDS, dtype=np.int64), ends - starts, side="left"
    ).astype(np.int64)

    key = np.concatenate([2 * keep_idx, 2 * ends - 1])
    src = np.concatenate([keep_idx, ends - 1])
    sent = np.concatenate([np.zeros(len(keep_idx), dtype=bool),
                           np.ones(len(ends), dtype=bool)])
    cls = np.concatenate([np.zeros(len(keep_idx), dtype=np.int64), cls_blocks])
    order = np.argsort(key, kind="stable")
    return src[order], sent[order], cls[order]


def build_document(document: np.ndarray, gate: np.ndarray,
                   vocabulary_size: int, sentinel: bool) -> tuple:
    """(input, output, loss_mask) of the gated document: the reference form

    `document` is the full token sequence of ONE document (length L), `gate`
    the gate mask of its L-1 positions (from `gate_mask`, computed on the
    WHOLE stream and cut to size here; the real decoder also runs over the
    whole stream, so the global mask is the right one).  Without gating the
    result would be exactly document[:-1] / document[1:] / all True.  The
    loader uses `gate_rows` directly; both produce the same row sequence.
    """
    doc = np.asarray(document, dtype=np.int64)
    n_pos = len(doc) - 1
    assert n_pos > 0
    src, sent, cls = gate_rows(np.asarray(gate, dtype=bool)[:n_pos], sentinel)
    inp = np.where(sent, vocabulary_size + cls, doc[src])
    out = doc[src + 1]
    return inp, out, ~sent


# --------------------------------------------------------------------------
# Calibration on the zone map (CPU, no cost, decides M)
# --------------------------------------------------------------------------

def _nats_cached(directory: str, probs_path: str, tokens: np.ndarray, v: int,
                 cache: str) -> np.ndarray:
    if os.path.exists(cache):
        return np.load(cache)
    from zctx.zonemap import nats

    out = nats(probs_path, tokens, v)
    np.save(cache, out)
    return out


def calib(dumpdir: str, tf_probs: str, ms: list) -> int:
    tokens = np.fromfile(os.path.join(dumpdir, "ppmd-bytes.uint8"),
                         dtype=np.uint8).astype(np.int64)
    bitmap = load_bitmap(dumpdir)
    v = len(bitmap)
    is_digit_token = digit_token_mask(bitmap)

    cdir = os.environ.get("ZGATE_CACHE", dumpdir)
    tf = _nats_cached(dumpdir, tf_probs, tokens, v, os.path.join(cdir, "_tf_nats.npy"))
    pp = _nats_cached(dumpdir, os.path.join(dumpdir, "ppmd-probs.float16"), tokens, v,
                      os.path.join(cdir, "_pp_nats.npy"))
    n = min(len(tf), len(pp))
    tf, pp = tf[:n], pp[:n]

    is_digit = is_digit_token[tokens]
    print("excerpt: %d tokens, %d scored positions, V=%d" % (len(tokens), n, v))
    print("digit share of the stream: %.3f %%" % (100.0 * is_digit.mean()))
    excl = exclusion_mask(tokens, bitmap)
    print("exclusion state (link or table line): %.3f %% of the positions"
          % (100.0 * excl.mean()))
    print("TF total    %.6f nats/token   PPMD total  %.6f   contribution %+.6f"
          % (tf.mean(), pp.mean(), (tf - pp).mean()))
    print()
    print("%-3s %-9s %10s %8s %14s %14s %12s" %
          ("M", "excl.", "gated", "share", "sum dNats", "dBytes", "per position"))
    for m in ms:
        for ex in (False, True):
            gate = gate_mask(tokens, bitmap, m, exclude=ex)[:n]
            k = int(gate.sum())
            if not k:
                continue
            delta = float((pp[gate] - tf[gate]).sum())
            print("%-3d %-9s %10d %7.3f%% %+14.1f %+14.1f %+12.6f"
                  % (m, "yes" if ex else "no", k, 100.0 * k / n, delta,
                     delta / np.log(2) / 8, delta / k))
    print()
    print("for comparison, the NON-causal map rule (runs >= 4, as in zonemap):")
    runs = runs_of(is_digit)
    zone = np.zeros(len(tokens), dtype=bool)
    for a, b in runs:
        if b - a >= 4:
            zone[a:b] = True
    zone_pred = zone[1:][:n]          # the zone belongs to the predicted token
    k = int(zone_pred.sum())
    delta = float((pp[zone_pred] - tf[zone_pred]).sum())
    print("    %10d %7.3f%% %+14.1f %+14.1f %+12.6f"
          % (k, 100.0 * k / n, delta, delta / np.log(2) / 8, delta / k))
    return 0


# --------------------------------------------------------------------------
# Counts and PPMd cost on the TRAINING PACKAGE (the quantity to offset against)
# --------------------------------------------------------------------------

EXCLUDE = os.environ.get("ZGATE_EXCLUDE", "1") != "0"


def pkgstat(pkgdir: str, ms: list, chunk_rows: int = 1_000_000) -> int:
    """PPMd nats and position count, split into gated / kept

    This is the quantity the training runs are offset against: at the gated
    positions the target system pays the prior that the transformer's slot
    would deliver without it.  Measured on the same package and with the
    same set of positions as `ppmd_loss.py` (article ends excluded, row i
    predicts token i+1).
    """
    tokens = np.fromfile(os.path.join(pkgdir, "ppmd-bytes.uint8"), dtype=np.uint8)
    n = len(tokens)
    bnd = np.fromfile(os.path.join(pkgdir, "article-boundaries.int32"),
                      dtype=np.int32).astype(np.int64)
    bitmap = load_bitmap(pkgdir)
    v = len(bitmap)
    size = os.path.getsize(os.path.join(pkgdir, "ppmd-probs.float16"))
    assert size == n * v * 2, (size, n, v)

    is_digit = digit_token_mask(bitmap)[tokens.astype(np.int64)]

    valid = np.ones(n, dtype=bool)
    valid[bnd[1:] - 1] = False           # last token of each article
    gates = {m: gate_mask(tokens, bitmap, m, exclude=EXCLUDE) for m in ms}

    priors = np.memmap(os.path.join(pkgdir, "ppmd-probs.float16"),
                       dtype=np.float16, mode="r", shape=(n, v))
    TINY = np.float32(6.0e-8)
    tot = {m: [0.0, 0, 0.0, 0] for m in ms}   # [nats_gate, k_gate, nats_keep, k_keep]
    for a in range(0, n - 1, chunk_rows):
        b = min(a + chunk_rows, n - 1)
        mvalid = valid[a:b]
        if not mvalid.any():
            continue
        rows = np.asarray(priors[a:b])
        tgt = tokens[a + 1:b + 1].astype(np.int64)
        p = rows[np.arange(b - a), tgt].astype(np.float32)
        nats = -np.log(np.maximum(p, TINY).astype(np.float64))
        for m in ms:
            g = gates[m][a:b] & mvalid
            kx = ~gates[m][a:b] & mvalid
            tot[m][0] += float(nats[g].sum()); tot[m][1] += int(g.sum())
            tot[m][2] += float(nats[kx].sum()); tot[m][3] += int(kx.sum())
        if (a // chunk_rows) % 25 == 0:
            print("  %5.1f%%" % (a / (n - 1) * 100), flush=True)

    print()
    print("package %s: %d tokens, %d articles, V=%d" % (pkgdir, n, len(bnd) - 1, v))
    print("digit share of the stream: %.4f %%" % (100.0 * is_digit.mean()))
    print()
    print("%-3s %12s %8s %14s %12s %14s" %
          ("M", "gated", "share", "PPMD gated", "kept", "PPMD kept"))
    for m in ms:
        ng, kg, nk, kk = tot[m]
        print("%-3d %12d %7.4f%% %14.6f %12d %14.6f"
              % (m, kg, 100.0 * kg / (kg + kk), ng / max(kg, 1), kk, nk / max(kk, 1)))
    import json
    out = {str(m): {"nats_gated": tot[m][0], "n_gated": tot[m][1],
                    "nats_kept": tot[m][2], "n_kept": tot[m][3]} for m in ms}
    print()
    print(json.dumps(out, indent=2))
    return 0


def selftest() -> None:
    v = 208
    bitmap = np.arange(v, dtype=np.uint8)          # index == byte, for the test
    isd = digit_token_mask(bitmap)
    assert isd[0x30] and isd[0x39] and not isd[0x2F] and not isd[0x3A]

    def gm(doc, m, exclude=False):
        return gate_mask(doc, bitmap, m, exclude=exclude)[: len(doc) - 1]

    # "ab12345cd" -> digit run of length 5
    doc = np.array([ord(c) for c in "ab12345cd"], dtype=np.int64)
    for m in (1, 2, 3):
        gate = gm(doc, m)
        inp, out, msk = build_document(doc, gate, v, sentinel=False)
        assert len(inp) == len(out) == len(msk) == int((~gate).sum())
        assert (inp == doc[:-1][~gate]).all() and (out == doc[1:][~gate]).all()

    # M = 1: the predictions of '2','3','4','5','c' are gated
    gate1 = gm(doc, 1)
    assert list(np.flatnonzero(gate1)) == [2, 3, 4, 5, 6]
    inp, out, msk = build_document(doc, gate1, v, sentinel=False)
    assert [chr(c) for c in inp] == ["a", "b", "c"]
    assert [chr(c) for c in out] == ["b", "1", "d"]

    # sentinel arm: one unscored row with the class of the run
    inp, out, msk = build_document(doc, gate1, v, sentinel=True)
    assert len(inp) == 4 and msk.sum() == 3
    sent = int(inp[2])
    assert sent >= v and sent - v == length_class(5)
    assert not msk[2] and chr(int(out[2])) == "c"
    assert [chr(c) for c in inp[[0, 1, 3]]] == ["a", "b", "c"]

    # a document without any digits stays unchanged
    plain = np.array([ord(c) for c in "hello world"], dtype=np.int64)
    i2, o2, m2 = build_document(plain, gm(plain, 1), v, sentinel=True)
    assert (i2 == plain[:-1]).all() and (o2 == plain[1:]).all() and m2.all()

    # a run AT THE END OF THE DOCUMENT gets no sentinel row
    tail = np.array([ord(c) for c in "ab1234"], dtype=np.int64)
    g = gm(tail, 1)
    i3, o3, m3 = build_document(tail, g, v, sentinel=True)
    i4, o4, m4 = build_document(tail, g, v, sentinel=False)
    assert (i3 < v).all(), "sentinel inserted at the end of the document"
    assert (i3 == i4).all() and (o3 == o4).all() and (m3 == m4).all()

    # gate_rows: target and prior come from src for EVERY row
    src, sentf, cls = gate_rows(gate1, sentinel=True)
    assert (doc[src + 1] == out).all()
    assert src.max() < len(doc) - 1 and (np.diff(src) > 0).all()

    # EXCLUSION: the same digits in a link and in a table line
    link = np.array([ord(c) for c in "x [[12345]] y"], dtype=np.int64)
    assert gm(link, 1).sum() > 0
    assert gm(link, 1, exclude=True).sum() == 0, "link not excluded"
    tab = np.array([ord(c) for c in "a\n| 12345 b"], dtype=np.int64)
    assert gm(tab, 1).sum() > 0
    assert gm(tab, 1, exclude=True).sum() == 0, "table line not excluded"
    # ... and NOT in ordinary text
    assert gm(np.array([ord(c) for c in "a\nim 12345 b"], dtype=np.int64), 1,
              exclude=True).sum() > 0

    # length classes are monotonic and cover everything
    assert length_class(1) == 0 and length_class(100) == N_LENGTH_CLASSES - 1
    assert all(length_class(k) <= length_class(k + 1) for k in range(1, 40))

    print("digit indices          %d of %d" % (int(isd.sum()), v))
    print("length classes         %d (bounds %s)" % (N_LENGTH_CLASSES, LENGTH_CLASS_BOUNDS))
    print("sentinel indices       %d .. %d (above V, never in the stream)"
          % (v, v + N_LENGTH_CLASSES - 1))
    print("exclusion              link depth > 0, table/list line from the marker on")
    print()
    print("OK: zgate self test")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
        return 0
    if cmd == "calib":
        ms = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else
                               ["1", "2", "3", "4"])]
        return calib(sys.argv[2], sys.argv[3], ms)
    if cmd == "pkgstat":
        ms = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else
                               ["1", "2", "3"])]
        return pkgstat(sys.argv[2], ms)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
