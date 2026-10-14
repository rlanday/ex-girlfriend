#!/usr/bin/env python3
"""arm z4 "lastbytes": the last X stream tokens as a direct model input

Background: the same idea (a bigram side input) worked on the LSTM head, so
it is tried on the transformer as well.

The transformer sees the immediate context only through attention and the KDA
recurrence.  A direct n-gram shortcut gives it the last X tokens as an
embedding at the same place where the token embedding enters, without the
detour through learned attention.

Built like z1 (zfeat.py), for the same reasons:

  * No new dump: the preceding tokens are in the stream, the loader only
    shifts them by k positions.  The decoder knows them anyway (they are the
    tokens it has just decoded), so they cost 0 bytes.
  * The values travel in the int64 `input_tokens` (the trainer hands the model
    exactly five tensors; a sixth would mean copying its training loop).
  * The output projection is initialised to zero: with the same seed the twin
    is numerically identical to the reference at step 0, so a measured
    difference cannot come from a different initial draw.

Packing (X = 4, the first point of the scale 2/4/8):

    bit  0.. 7   the token itself          (as in zfeat)
    bit  8..15   predecessor t-1
    bit 16..23   predecessor t-2
    bit 24..31   predecessor t-3
    bit 32..39   predecessor t-4

X <= 6 fits safely into the int64 carrier (8 + 6*8 = 56 bits).

**The edge case has a clean solution, not a convention:** at the start of a
document there is no predecessor.  The mark is token index **0**, which is
NOT used in the 208-value byte set of the package (`tft.dataset check`
reports `unused_token_indices [0, 3, 9]`).  No real symbol is overloaded and
no new one is invented, so the cost of a new symbol (7-44 KB per symbol) does
not arise.  `verify_sentinel()` checks this on the package instead of
assuming it.

CONTROL ARM (`z4c`; a control arm with information-free context is mandatory
for EVERY new model input):
the same X tokens, but their assignment to the X slots is shuffled by a FIXED
permutation.  This is a bijection, and the model could in principle learn it
back; the arm therefore separates "distance structure" from "mere capacity",
not "information" from "no information".  It is the cheaper half of the
question; the more expensive one (tokens from fixed DISTANT positions) is
open.
"""

from __future__ import annotations

import numpy as np

TOKEN_BITS = 8
TOKEN_MASK = (1 << TOKEN_BITS) - 1
LAST_SHIFT = TOKEN_BITS          # the first predecessor sits directly above the token
MAX_X = 6                        # 8 + 6*8 = 56 bits, safely inside the int64 carrier

# The token index that means "there is no predecessor".  Unused in the package.
NO_PREDECESSOR = 0

# fixed permutation of the control arm; derived deterministically from X so
# that it is reproducible without state (and would be a table of X bytes in C++)
def control_permutation(x: int) -> list:
    """a fixed, non-identity permutation of the X slots"""
    assert 2 <= x <= MAX_X
    rng = np.random.default_rng(20260901 + x)
    while True:
        p = rng.permutation(x)
        if not np.array_equal(p, np.arange(x)):
            return [int(v) for v in p]


def pack(tokens: np.ndarray, x: int, permute: bool = False) -> np.ndarray:
    """int64 carrier: token in the low 8 bits, the X predecessors above it

    `tokens` is the input sequence of ONE document (that is `document_in`).
    The predecessor of position i is tokens[i-k]; before the start of the
    document it is NO_PREDECESSOR.  The prefix of a z3 document counts as part
    of the document, because the decoder would have seen it at that point too.
    """
    assert 2 <= x <= MAX_X, "X must lie between 2 and %d" % MAX_X
    t = np.asarray(tokens, dtype=np.int64)
    assert (t >= 0).all() and (t <= TOKEN_MASK).all()
    n = len(t)
    out = t.copy()
    slots = control_permutation(x) if permute else list(range(x))
    for k in range(1, x + 1):
        prev = np.full(n, NO_PREDECESSOR, dtype=np.int64)
        if n > k:
            prev[k:] = t[:-k]
        out |= prev << (LAST_SHIFT + TOKEN_BITS * slots[k - 1])
    return out


def unpack(packed: np.ndarray, x: int):
    """(token, [predecessor_1 .. predecessor_X]); the inverse of pack()"""
    q = np.asarray(packed, dtype=np.int64)
    token = q & TOKEN_MASK
    prev = [
        (q >> (LAST_SHIFT + TOKEN_BITS * k)) & TOKEN_MASK for k in range(x)
    ]
    return token, prev


def verify_sentinel(data_dir: str) -> dict:
    """checks on the PACKAGE that token index 0 is really unused

    This is the one assumption the edge-case handling rests on.  It is
    recomputed, not assumed: a package that uses index 0 would silently
    confuse the predecessor channel with a real symbol.
    """
    import os

    tokens = np.fromfile(os.path.join(data_dir, "ppmd-bytes.uint8"), dtype=np.uint8)
    hits = int(np.count_nonzero(tokens == NO_PREDECESSOR))
    result = {
        "n_tokens": int(len(tokens)),
        "sentinel": NO_PREDECESSOR,
        "occurrences_of_sentinel": hits,
        "ok": hits == 0,
    }
    assert hits == 0, (
        "token index %d occurs %d times in the stream; it is NOT usable as "
        "edge mark.  A free index is needed (tft.dataset check lists them "
        "under unused_token_indices)." % (NO_PREDECESSOR, hits)
    )
    return result


def selftest(x: int = 4) -> None:
    """packing, edge case and control permutation, without torch and without a package"""
    rng = np.random.default_rng(7)
    t = rng.integers(1, 208, size=200, dtype=np.int64)

    p = pack(t, x)
    tok, prev = unpack(p, x)
    assert (tok == t).all(), "token does not survive the packing"
    for k in range(1, x + 1):
        expect = np.full(len(t), NO_PREDECESSOR, dtype=np.int64)
        expect[k:] = t[:-k]
        assert (prev[k - 1] == expect).all(), "predecessor %d wrong" % k
    assert (prev[0][:1] == NO_PREDECESSOR).all(), "document start not marked"

    # a document shorter than X must not run into an error
    short = pack(t[:2], x)
    _, sp = unpack(short, x)
    assert all((v[:1] == NO_PREDECESSOR).all() for v in sp)

    # the control arm carries THE SAME values, only in other slots
    c = pack(t, x, permute=True)
    _, cprev = unpack(c, x)
    order = control_permutation(x)
    assert order != list(range(x)), "control permutation is the identity"
    for k in range(1, x + 1):
        assert (cprev[order[k - 1]] == prev[k - 1]).all(), "control arm loses values"
    assert sorted(v.sum() for v in cprev) == sorted(v.sum() for v in prev)

    # bit width: nothing reaches the sign bit
    assert p.max() < (1 << (LAST_SHIFT + TOKEN_BITS * MAX_X))

    print("slots                  X = %d" % x)
    print("carrier bits           %d of 63" % (LAST_SHIFT + TOKEN_BITS * x))
    print("edge mark              token index %d (unused in the package)" % NO_PREDECESSOR)
    print("control permutation    %s" % (control_permutation(x),))
    print()
    print("OK: zlast self test")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        print(verify_sentinel(sys.argv[2]))
    else:
        selftest(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
