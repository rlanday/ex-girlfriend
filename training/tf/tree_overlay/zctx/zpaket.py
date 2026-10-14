#!/usr/bin/env python3
"""The complete context package as ONE arm ("pk")

Goal: test the whole package at once instead of each part in isolation,
including the parts already accepted as safe.  What is measured is therefore
not the additivity assumption but the total effect with all interactions.

Eight components, each accepted on its own, here in ONE document:

  1  z3 in the CAUSAL form     title . </title> . prefix . body
                               (the prefix stands BEHIND the title, because
                               the coder needs the title to look up the
                               incoming link contexts at all)
  4  A2  title re-injection at the "==" section boundaries, loss-masked
  7  A2n running heading number BEFORE the re-injected title
  5  z1t article number as text "#<n>" in the prompt
  8  SYNPROMPT  the redirect synonyms of the article in the prompt
  6  sentinel length class after every digit run, WITHOUT gating
  3  X=2 channel of the last bytes
  2  match channel (hot byte + length class)

THE PACKING.  The trainer hands the model exactly five tensors, so all extra
channels travel in the int64 `input_tokens` (built like zfeat / zlast /
zmatch).  The three separate models used INCOMPATIBLE bit fields (zlast:
8..23, zmatch: 8..20); here they sit side by side:

    bit  0.. 7   the token itself          (0..V+7; above V the sentinels)
    bit  8..15   predecessor t-1           (start of document: the edge mark)
    bit 16..23   predecessor t-2
    bit 24..32   match byte                (9 bits, 0..208; 208 = no mark)
    bit 33..36   match length class        (4 bits, 0..15)

37 of 63 bits.  The predecessor fields also carry sentinel values (up to
V+7=215), so they still fit in 8 bits.

THE POSITION BOOKKEEPING is the actual seam.  Beside the tokens each document
carries a vector `src`: the STREAM POSITION of the token, -1 for everything
inserted.  Everything else follows mechanically and without special cases:

    row j has input tok[j] and predicts tok[j+1]
    ziel[j] = src[j+1]                        (ziel = target)
    scored  <=>  ziel[j] >= start + T         (body; title and prompt excluded)
    prior row = priors[ziel[j] - 1], prior0 for inserted rows

So the number of SCORED positions is exactly the number of body tokens, the
same set that the masked reference arm `mref` scores.  The comparison is a
plain one-factor comparison without conversion; the composition trap of the
gating evaluation (gate_eval.py) does not arise.
"""

from __future__ import annotations

import os

import numpy as np

from zctx import zgate

TOKEN_BITS = 8
TOKEN_MASK = (1 << TOKEN_BITS) - 1

# The match constants are repeated here instead of being imported from
# zctx.zmatch: zmatch pulls in pysrc.model and with it fla/triton, and this
# module has to run on a CPU machine without GPU libraries (the required null
# tests run there).  `verify_zmatch()`, which the loader calls on the GPU pod,
# where zmatch can be imported, guards against the two drifting apart.
MATCH_BITS = 9
MATCH_MASK = (1 << MATCH_BITS) - 1
LEN_BITS = 4
LEN_MASK = (1 << LEN_BITS) - 1
N_MATCH_ROWS = 209                      # 208 token indices + the mark 208
N_LEN_CLASSES = 16
BITS_PER_CLASS = 8                      # 8 matching bits = 1 byte


def match_length_class(raw) -> np.ndarray:
    """raw bit length -> class 0..15 (identical to zmatch.length_class)"""
    return np.minimum(np.asarray(raw, dtype=np.int64) // BITS_PER_CLASS,
                      N_LEN_CLASSES - 1)


def verify_zmatch() -> None:
    """on the GPU machine: cross-check the constants that are kept twice"""
    from zctx import zmatch

    assert (MATCH_BITS, LEN_BITS, N_MATCH_ROWS, N_LEN_CLASSES, BITS_PER_CLASS) == (
        zmatch.MATCH_BITS, zmatch.LEN_BITS, zmatch.N_MATCH_ROWS,
        zmatch.N_LEN_CLASSES, zmatch.BITS_PER_CLASS
    ), "zpaket and zmatch have drifted apart"
    probe = np.arange(0, 256, 7)
    assert (match_length_class(probe) == zmatch.length_class(probe)).all()

# Number of predecessor slots.  X=2 is the measured point of the series
# (-0.001347 nats per token, -13.1 sigma); X=6 does not pay.
DEFAULT_X = 2
# On the FULL stream the edge mark is NOT a free index.  `tft.dataset check`
# reports `distinct_token_indices = 208` there (on pkg200m_v2, [0, 3] were
# free); byte 0x01 = token index 0 occurs 391,160 times in the coder stream.
# The mark is used legitimately 2*n_pieces = 486,896 times and spuriously
# 782,320 times, so the wrong use is 1.6x as frequent as the intended one, and
# both share ONE row.
# X0ROW gives the mark a row of its own at the top (V + N_SENTINEL); it still
# fits in TOKEN_BITS = 8 (216 < 256).  The coder must use the same setting;
# its weight loader checks it through the row count of embedding.last_table
# (X*(V+NSENT+1) instead of X*(V+NSENT)).
X0ROW = os.environ.get("PK_LAST_X0ROW", "0") != "0"
NO_PREDECESSOR = 0                      # stage-1 form: shared row 0


def edge_marker(vocabulary_size: int) -> int:
    """token value "no predecessor" """
    return (vocabulary_size + N_SENTINEL) if X0ROW else NO_PREDECESSOR


def last_rows(vocabulary_size: int) -> int:
    """rows of ONE predecessor slot of embedding.last_table"""
    return vocabulary_size + N_SENTINEL + (1 if X0ROW else 0)

N_SENTINEL = zgate.N_LENGTH_CLASSES     # 8 length classes above the vocabulary


def shifts(x: int = DEFAULT_X) -> dict:
    last = TOKEN_BITS
    match = TOKEN_BITS + TOKEN_BITS * x
    length = match + MATCH_BITS
    total = length + LEN_BITS
    assert total <= 62, "the packing overflows the int64 carrier (%d bits)" % total
    return {"last": last, "match": match, "len": length, "bits": total}


def pack(tokens: np.ndarray, match_byte: np.ndarray, match_len_class: np.ndarray,
         x: int = DEFAULT_X) -> np.ndarray:
    """int64 carrier for ONE document (input rows, i.e. document_in)

    `tokens` are the input tokens of the rows, `match_byte`/`match_len_class`
    the two channel values of the SAME rows (mark 208 and class 0 for inserted
    rows).  The predecessors are shifted in here from `tokens`; they are local
    to the document, exactly as the decoder would see them.
    """
    s = shifts(x)
    t = np.asarray(tokens, dtype=np.int64)
    assert (t >= 0).all() and (t <= TOKEN_MASK).all(), "token does not fit in 8 bits"
    mb = np.asarray(match_byte, dtype=np.int64)
    cl = np.asarray(match_len_class, dtype=np.int64)
    assert len(mb) == len(t) and len(cl) == len(t)
    assert (mb >= 0).all() and (mb < N_MATCH_ROWS).all()
    assert (cl >= 0).all() and (cl < N_LEN_CLASSES).all()
    out = t.copy()
    n = len(t)
    for k in range(1, x + 1):
        # the edge mark, not the fixed 0
        prev = np.full(n, edge_marker(_V_FOR_SENTINEL["v"]),
                       dtype=np.int64)
        if n > k:
            prev[k:] = t[:-k]
        out |= prev << (s["last"] + TOKEN_BITS * (k - 1))
    out |= mb << s["match"]
    out |= cl << s["len"]
    return out


def unpack(packed: np.ndarray, x: int = DEFAULT_X):
    """(token, [predecessors...], match_byte, length class): inverse of pack()"""
    s = shifts(x)
    q = np.asarray(packed, dtype=np.int64)
    token = q & TOKEN_MASK
    prev = [(q >> (s["last"] + TOKEN_BITS * k)) & TOKEN_MASK for k in range(x)]
    mb = (q >> s["match"]) & MATCH_MASK
    cl = (q >> s["len"]) & LEN_MASK
    return token, prev, mb, cl


# --------------------------------------------------------------------------
# A2n: the running heading number as text
# --------------------------------------------------------------------------

def byte_to_token(bitmap: np.ndarray) -> dict:
    werte = [b * 8 + k for b in range(32) for k in range(8) if bitmap[b] >> k & 1]
    return {int(v): i for i, v in enumerate(werte)}


def number_tokens(n: int, tok_of_byte: dict) -> np.ndarray:
    """the string #<n> as a token sequence, the same form as z1t

    No new symbol: the number sign and the digits are in the 208-value byte set
    anyway.  If one is missing, the build stops instead of silently working
    around it.
    """
    out = []
    for ch in "#%d" % n:
        b = ord(ch)
        assert b in tok_of_byte, (
            "byte %r is not in the 208-value set; A2n would pay an alphabet tax" % ch
        )
        out.append(tok_of_byte[b])
    return np.asarray(out, dtype=np.int64)


# --------------------------------------------------------------------------
# document construction: the ONLY source for length and content
# --------------------------------------------------------------------------

_V_FOR_SENTINEL = {"v": 208}


def set_vocabulary(v: int) -> None:
    _V_FOR_SENTINEL["v"] = int(v)


def sent_token(cls: int) -> int:
    return _V_FOR_SENTINEL["v"] + int(cls)


def sentinel_sites(gate_slice: np.ndarray, start: int, body: int):
    """[(stream position, class)]: one row behind every digit run

    The same run decomposition and the same classes as arm g1
    (zgate.gate_rows); the difference is that NOTHING is cut here: the
    transformer still predicts every position, and the length class is only an
    extra input.  A run that ends at the end of the document gets no row (it
    carries nothing there, as in g1).
    """
    out = []
    for a, b in zgate.runs_of(gate_slice):
        if int(b) >= body:
            continue
        out.append((start + int(b), zgate.length_class(int(b) - int(a))))
    return out


def build_document(tokens, start: int, end: int, title_len: int,
                   prompt: np.ndarray, a2_sites, a2_number: bool,
                   sentinels, tok_of_byte: dict, zone=None):
    """(tok, src) of the package document

    tok  the token sequence of the document (title . prompt . body with insertions)
    src  per place the stream position, -1 for everything inserted

    `sentinels` is a pair (positions, classes): before the STREAM POSITION
    `position` a sentinel row with token index V+class is inserted.  It thus
    stands directly behind the digit run and directly before the prediction of
    the token that follows it, the place where the length class is of use.
    """
    T = int(title_len)
    # Zone tokens (KOPFREST = page head, CMTIDX = comment index, HDRBLOCK =
    # metadata block) are NEITHER fed NOR scored, as in the coder.  The context
    # runs across the gap; there is no placeholder.  The predecessors (X2)
    # therefore follow from the FED sequence, because pack() shifts them on the
    # compacted row; the match channel is looked up per STREAM POSITION and so
    # stays the coder's value at that place.
    tsrc = np.arange(start, start + T, dtype=np.int64)
    ksrc = np.arange(start + T, end, dtype=np.int64)
    if zone is not None:
        if T:
            tsrc = tsrc[~zone[start : start + T]]
        ksrc = ksrc[~zone[start + T : end]]
    titel = (np.asarray(tokens[tsrc], dtype=np.int64) if len(tsrc)
             else np.zeros(0, dtype=np.int64))
    koerper = (np.asarray(tokens[ksrc], dtype=np.int64) if len(ksrc)
               else np.zeros(0, dtype=np.int64))

    stellen = []
    for k, p in enumerate(a2_sites, start=1):
        p = int(p)
        assert start + T <= p < end, "A2 site %d is not in the body" % p
        block = titel
        if a2_number:
            block = np.concatenate([number_tokens(k, tok_of_byte), titel])
        stellen.append((p, 0, block))
    for p, cls in zip(np.asarray(sentinels[0]).tolist(),
                      np.asarray(sentinels[1]).tolist()):
        p = int(p)
        assert start + T <= p < end, "sentinel site %d is not in the body" % p
        stellen.append((p, 1, np.asarray([sent_token(cls)], dtype=np.int64)))

    if stellen:
        stellen.sort(key=lambda z: (z[0], z[1]))
        # The insertion point is the INDEX of the stream position in the
        # (possibly gated) body sequence, not p-(start+T).
        _idx = np.searchsorted(ksrc, np.asarray([p for p, _, _ in stellen],
                                                dtype=np.int64))
        assert (ksrc[np.minimum(_idx, len(ksrc) - 1)]
                == np.asarray([p for p, _, _ in stellen])).all(), (
            "insertion point lies on a gated position")
        wo = np.concatenate([np.full(len(b), _idx[i], dtype=np.int64)
                             for i, (p, _, b) in enumerate(stellen)])
        was = np.concatenate([b for _, _, b in stellen])
        koerper = np.insert(koerper, wo, was)
        ksrc = np.insert(ksrc, wo, np.full(len(was), -1, dtype=np.int64))

    p_arr = np.asarray(prompt, dtype=np.int64)
    tok = np.concatenate([titel, p_arr, koerper])
    src = np.concatenate([
        tsrc,
        np.full(len(p_arr), -1, dtype=np.int64),
        ksrc,
    ])
    assert len(tok) == len(src)
    return tok, src


def selftest() -> None:
    """packing, edge case, document construction; needs neither torch nor a package"""
    rng = np.random.default_rng(20260903)
    n = 500
    t = rng.integers(1, 208, size=n, dtype=np.int64)
    mb = rng.integers(0, 209, size=n, dtype=np.int64)
    cl = rng.integers(0, 16, size=n, dtype=np.int64)
    q = pack(t, mb, cl)
    tt, prev, mbb, cll = unpack(q)
    assert (tt == t).all() and (mbb == mb).all() and (cll == cl).all()
    _rand = edge_marker(_V_FOR_SENTINEL["v"])   # 0 or V+NSENT
    assert (prev[0][1:] == t[:-1]).all() and prev[0][0] == _rand
    assert (prev[1][2:] == t[:-2]).all() and (prev[1][:2] == _rand).all()
    assert (q >= 0).all()
    assert shifts()["bits"] == 37

    # sentinel values must come through as token AND as predecessor
    set_vocabulary(208)
    t2 = np.array([5, 208 + 7, 9, 215], dtype=np.int64)
    q2 = pack(t2, np.full(4, 208), np.zeros(4))
    tt2, prev2, _, _ = unpack(q2)
    assert (tt2 == t2).all() and prev2[0][1] == 5 and prev2[0][3] == 9

    # document construction: title . prompt . body, one A2 site, one sentinel
    tokens = np.arange(100, 140, dtype=np.int64) % 208
    tok, src = build_document(tokens, 0, 40, 4, np.array([3, 4, 5]),
                              np.array([10]), True,
                              (np.array([20]), np.array([3])), {35: 7, 49: 9})
    assert len(tok) == 40 + 3 + 6 + 1, len(tok)
    assert (src[:4] == np.arange(0, 4)).all()
    assert (src[4:7] == -1).all()
    assert int((src < 0).sum()) == 3 + 6 + 1
    ziel = src[1:]
    assert int((ziel >= 4).sum()) == 36, "scored rows != body tokens"
    print("scored rows       %d (body tokens 36)" % int((ziel >= 4).sum()))
    print("carrier bits      %d of 63" % shifts()["bits"])
    print("OK: zpaket selftest")


if __name__ == "__main__":
    selftest()
