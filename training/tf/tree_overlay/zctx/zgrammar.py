#!/usr/bin/env python3
"""stream grammar for the z3 extractor: token sequences, not byte sequences

Everything in here is stated as TOKEN INDICES of our 208 byte set (see
section 4.6 of the top-level README.md), because that is what both sides see: the training
loader reads `ppmd-bytes.uint8`, and the coder's transformer sees the same
indices.  Nothing here may be derived from enwik9 raw text: the coder does not
have it at decode time (dump identity requirement).

The four constants were READ OFF the real 200M package, not guessed,
and `verify()` re-derives every one of them from a package before use.  The byte
column is only an aid to the human reader; the dictionary maps `<` -> 0x4c and
`>` -> 0x4e, so `<title>` is `L <df> <9b> N`.

    article separator  0x0b 0x0b 0x28 0xaf 0x68 0x2a 0x08 0x0b 0x0b 0x0b 0x0b
                       0x28 0xaf 0x6b 0x2a          "  <page>\n    <title>"
    title terminator   0x28 0x1a 0xaf 0x6b 0x2a     "</title>"
    link open          0x31                         "["   (WRT collapses "[[")
    paragraph          0x08                         "\n"

`article-boundaries.int32` points BEHIND the separator, so the
title starts exactly at a boundary and runs to the first title terminator.

    python -m zctx.zgrammar verify <package-dir>
"""

from __future__ import annotations

import os
import sys

import numpy as np

# token indices, decimal
ARTICLE_SEPARATOR = [11, 11, 40, 175, 104, 42, 8, 11, 11, 11, 11, 40, 175, 107, 42]
TITLE_TERMINATOR = [40, 26, 175, 107, 42]
LINK_OPEN = [49]
PARAGRAPH_TOKEN = 8

# a title longer than this is treated as "no usable title" (the real maximum in
# the 200M window is 59 tokens; the cap bounds the scan for the C++ port)
MAX_TITLE_TOKENS = 64

# a title shorter than this is not used as a retrieval key: one or two tokens is
# a single common word, its "mentions" are noise rather than incoming links
MIN_TITLE_TOKENS = 3


def _seq(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.uint8)


def find_first(tokens: np.ndarray, pattern: list[int], start: int, stop: int) -> int:
    """first index p in [start, stop-len(pattern)] with tokens[p:p+k] == pattern

    Plain, deterministic, no library search; this is the routine the C++ port
    mirrors.  Returns -1 if there is none.
    """
    k = len(pattern)
    if stop - start < k:
        return -1
    first = pattern[0]
    window = tokens[start : stop - k + 1]
    cand = np.flatnonzero(window == first)
    for c in cand:
        p = start + int(c)
        if bool((tokens[p : p + k] == _seq(pattern)).all()):
            return p
    return -1


def verify(directory: str) -> dict:
    """re-derive every grammar constant from a real package

    Fails loudly rather than silently extracting from a stream whose dictionary
    is not the one these constants came from (the same trap as a separator
    constant taken over from another dictionary).
    """
    tokens = np.fromfile(os.path.join(directory, "ppmd-bytes.uint8"), dtype=np.uint8)
    boundaries = np.fromfile(
        os.path.join(directory, "article-boundaries.int32"), dtype=np.int32
    ).astype(np.int64)
    info: dict = {"n_tokens": int(len(tokens)), "n_articles": int(len(boundaries)) - 1}

    sep = _seq(ARTICLE_SEPARATOR)
    k = len(sep)
    bad = 0
    for b in boundaries[1:-1]:
        b = int(b)
        if b < k or not bool((tokens[b - k : b] == sep).all()):
            bad += 1
    info["boundaries_without_separator"] = bad
    if bad:
        raise AssertionError(
            "%d of %d article boundaries are not preceded by the 15 token "
            "separator; this package was cut with a different dictionary"
            % (bad, len(boundaries) - 2)
        )

    lengths = []
    missing = 0
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        stop = min(start + MAX_TITLE_TOKENS + len(TITLE_TERMINATOR), int(boundaries[i + 1]))
        p = find_first(tokens, TITLE_TERMINATOR, start, stop)
        if p < 0:
            missing += 1
        else:
            lengths.append(p - start)
    info["titles_without_terminator"] = missing
    if lengths:
        a = np.asarray(lengths)
        info["title_len_min"] = int(a.min())
        info["title_len_median"] = float(np.median(a))
        info["title_len_mean"] = float(a.mean())
        info["title_len_max"] = int(a.max())
        info["titles_below_min"] = int((a < MIN_TITLE_TOKENS).sum())
    info["link_open_count"] = int((tokens == LINK_OPEN[0]).sum())
    info["paragraph_count"] = int((tokens == PARAGRAPH_TOKEN).sum())
    return info


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "verify":
        print(__doc__)
        return 2
    for key, value in verify(sys.argv[2]).items():
        print(str(key).ljust(32) + str(value))
    print("\nOK: grammar constants re-derived from the package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
