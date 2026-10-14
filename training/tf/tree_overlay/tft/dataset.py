#!/usr/bin/env python3
"""tools for the frozen training-data interface

The interface (the coder side dumps it, this training chain consumes it) is Ivanov's
make_training_data.sh output, 208 wide instead of 205, plus one file of ours:

  ppmd-bytes.uint8         N bytes,  vocabulary INDEX per stream byte, 0..V-1
  ppmd-probs.float16       N * V float16, row i = the ppmd's distribution over
                           token i+1 (written AFTER the ppmd was updated with
                           token i), in vocabulary index order
  article-boundaries.int32 int32 offsets, strictly increasing, FIRST entry 0,
                           LAST entry N
  vocab-bitmap.u8          32 bytes, bit (k*8+b) set <=> byte value k*8+b is in
                           the vocabulary; popcount must equal V   [OURS, new]

V is never written down: pysrc/data.py derives it as
filesize(priors) / 2 / N.  A wrong prior width therefore does not fail loudly,
it silently trains a differently shaped model, which `check` is here to
prevent.

    python -m tft.dataset check [dir]
    python -m tft.dataset cut <src-dir> <dst-dir> <max-tokens>
    python -m tft.dataset mini <dst-dir> [--tokens N] [--vocab 208]

No torch: numpy only, runs on any machine.
"""

from __future__ import annotations

import os
import sys

import numpy as np

DEFAULT_VOCAB = 208
FILES = {
    "tokens": "ppmd-bytes.uint8",
    "priors": "ppmd-probs.float16",
    "boundaries": "article-boundaries.int32",
    "bitmap": "vocab-bitmap.u8",
}


class CheckFailed(Exception):
    pass


def _path(directory: str, key: str) -> str:
    return os.path.join(directory, FILES[key])


def check(directory: str, expect_vocab: int | None = DEFAULT_VOCAB) -> dict:
    """validate a data package against the interface; raises CheckFailed"""
    problems: list[str] = []
    info: dict = {"dir": directory}

    for key in ("tokens", "priors", "boundaries"):
        if not os.path.isfile(_path(directory, key)):
            raise CheckFailed("missing " + _path(directory, key))

    n_tokens = os.path.getsize(_path(directory, "tokens"))
    info["n_tokens"] = n_tokens
    if n_tokens < 2:
        raise CheckFailed("need at least 2 tokens")

    priors_bytes = os.path.getsize(_path(directory, "priors"))
    if priors_bytes % 2:
        problems.append("priors file size is not a multiple of 2 (float16)")
    n_floats = priors_bytes // 2
    if n_floats % n_tokens:
        raise CheckFailed(
            "priors floats ("
            + str(n_floats)
            + ") not divisible by tokens ("
            + str(n_tokens)
            + "): the file has no whole number of rows; pysrc/data.py asserts this"
        )
    vocab = n_floats // n_tokens
    info["vocabulary_size_derived"] = vocab
    info["priors_bytes"] = priors_bytes
    if expect_vocab is not None and vocab != expect_vocab:
        problems.append(
            "derived vocabulary is " + str(vocab) + ", expected " + str(expect_vocab)
        )

    boundaries = np.fromfile(_path(directory, "boundaries"), dtype=np.int32)
    info["n_boundaries"] = int(len(boundaries))
    info["n_articles"] = int(len(boundaries)) - 1
    if len(boundaries) < 2:
        problems.append("boundaries need at least two entries (0 and n_tokens)")
    else:
        if boundaries[0] != 0:
            problems.append("boundaries[0] is " + str(boundaries[0]) + ", must be 0")
        if int(boundaries[-1]) != n_tokens:
            problems.append(
                "boundaries[-1] is "
                + str(int(boundaries[-1]))
                + ", must equal the token count "
                + str(n_tokens)
                + " (cutting a window must end ON an article boundary)"
            )
        if not (np.diff(boundaries.astype(np.int64)) > 0).all():
            problems.append("boundaries are not strictly increasing")
    if n_tokens > 0x7FFFFFFF:
        problems.append(
            "more than 2^31-1 tokens: the boundaries do not fit in int32 "
            "(his SaveArticleBoundaries fails loudly on this too)"
        )

    tokens = np.memmap(_path(directory, "tokens"), dtype=np.uint8, mode="r")
    tmax = int(tokens.max())
    info["token_max"] = tmax
    if tmax >= vocab:
        problems.append(
            "token index " + str(tmax) + " >= vocabulary " + str(vocab)
        )
    used = np.unique(tokens)
    info["distinct_token_indices"] = int(len(used))
    if len(used) < vocab:
        info["unused_token_indices"] = sorted(
            set(range(vocab)) - set(int(u) for u in used)
        )

    # bitmap (ours, optional but required for the real package)
    bitmap_path = _path(directory, "bitmap")
    if os.path.isfile(bitmap_path):
        raw = np.fromfile(bitmap_path, dtype=np.uint8)
        info["bitmap_bytes"] = int(len(raw))
        if len(raw) != 32:
            problems.append("vocab-bitmap.u8 has " + str(len(raw)) + " bytes, need 32")
        else:
            popcount = int(np.unpackbits(raw, bitorder="little").sum())
            info["bitmap_popcount"] = popcount
            if popcount != vocab:
                problems.append(
                    "bitmap popcount "
                    + str(popcount)
                    + " != derived vocabulary "
                    + str(vocab)
                )
            info["bitmap_byte_values"] = np.flatnonzero(
                np.unpackbits(raw, bitorder="little")
            ).tolist()
    else:
        info["bitmap"] = "MISSING (required for the training package)"

    # spot check the priors: rows must be non-negative and roughly normalised
    priors = np.memmap(
        _path(directory, "priors"), dtype=np.float16, mode="r", shape=(n_tokens, vocab)
    )
    rng = np.random.default_rng(0)
    sample_rows = rng.choice(n_tokens, size=min(256, n_tokens), replace=False)
    sample = np.asarray(priors[np.sort(sample_rows)], dtype=np.float64)
    if not np.isfinite(sample).all():
        problems.append("priors contain nan/inf")
    if (sample < 0).any():
        problems.append("priors contain negative values")
    sums = sample.sum(axis=1)
    info["prior_row_sum_min"] = float(sums.min())
    info["prior_row_sum_max"] = float(sums.max())
    if not (0.5 < sums.min() and sums.max() < 2.0):
        problems.append(
            "sampled prior rows do not sum near 1 (min "
            + format(float(sums.min()), ".4f")
            + ", max "
            + format(float(sums.max()), ".4f")
            + "): wrong row width or wrong dtype?"
        )

    lengths = np.diff(boundaries.astype(np.int64)) if len(boundaries) >= 2 else []
    if len(lengths):
        info["article_length_min"] = int(np.min(lengths))
        info["article_length_mean"] = float(np.mean(lengths))
        info["article_length_max"] = int(np.max(lengths))
        info["articles_over_2**17"] = int((lengths > (1 << 17)).sum())

    info["problems"] = problems
    if problems:
        raise CheckFailed("; ".join(problems))
    return info


def cut(src: str, dst: str, max_tokens: int, vocab: int | None = None) -> dict:
    """copy the first <= max_tokens tokens, truncated at an ARTICLE boundary

    Used twice: to make a small smoke-test window out of a big package, and as
    the fallback if a full-corpus dump lands on the pod instead of the 200M one.
    """
    os.makedirs(dst, exist_ok=True)
    n_tokens = os.path.getsize(_path(src, "tokens"))
    if vocab is None:
        vocab = os.path.getsize(_path(src, "priors")) // 2 // n_tokens

    boundaries = np.fromfile(_path(src, "boundaries"), dtype=np.int32)
    keep = boundaries[boundaries <= max_tokens]
    if len(keep) < 2:
        raise CheckFailed(
            "max_tokens " + str(max_tokens) + " is shorter than the first article"
        )
    new_n = int(keep[-1])

    tokens = np.memmap(_path(src, "tokens"), dtype=np.uint8, mode="r")
    np.asarray(tokens[:new_n]).tofile(_path(dst, "tokens"))

    priors = np.memmap(
        _path(src, "priors"), dtype=np.float16, mode="r", shape=(n_tokens, vocab)
    )
    out = np.memmap(
        _path(dst, "priors"), dtype=np.float16, mode="w+", shape=(new_n, vocab)
    )
    chunk = 1 << 20
    for start in range(0, new_n, chunk):
        end = min(start + chunk, new_n)
        out[start:end] = priors[start:end]
    out.flush()
    del out

    keep.astype(np.int32).tofile(_path(dst, "boundaries"))

    if os.path.isfile(_path(src, "bitmap")):
        np.fromfile(_path(src, "bitmap"), dtype=np.uint8).tofile(_path(dst, "bitmap"))

    return {
        "src_tokens": n_tokens,
        "dst_tokens": new_n,
        "dst_articles": int(len(keep)) - 1,
        "vocabulary_size": vocab,
    }


def mini(dst: str, n_tokens: int = 4096, vocab: int = DEFAULT_VOCAB, seed: int = 7) -> dict:
    """synthesize a tiny package in the exact interface format

    Not real data, a parser target.  The token stream is a low order Markov
    chain so that a "prior" built from it is not uniform, and the article
    lengths vary, including one length-1 article (the loader must skip it) and
    the trailing article that runs to the end of the stream.
    """
    os.makedirs(dst, exist_ok=True)
    rng = np.random.default_rng(seed)

    tokens = np.empty(n_tokens, dtype=np.uint8)
    state = 0
    for i in range(n_tokens):
        state = (state * 7 + int(rng.integers(0, 5))) % vocab
        tokens[i] = state
    tokens.tofile(_path(dst, "tokens"))

    # priors: a peaked distribution around the NEXT token, float16, row i is the
    # distribution over token i+1 (the last row predicts nothing and is unused)
    priors = np.memmap(
        _path(dst, "priors"), dtype=np.float16, mode="w+", shape=(n_tokens, vocab)
    )
    base = rng.random((vocab,)).astype(np.float32) * 0.001 + 0.0005
    targets = np.empty(n_tokens, dtype=np.int64)
    targets[: n_tokens - 1] = tokens[1:]
    targets[n_tokens - 1] = tokens[n_tokens - 1]
    chunk = 1 << 14
    for start in range(0, n_tokens, chunk):
        end = min(start + chunk, n_tokens)
        block = np.repeat(base[None, :], end - start, axis=0)
        block[np.arange(end - start), targets[start:end]] += 0.6
        block /= block.sum(axis=1, keepdims=True)
        priors[start:end] = block.astype(np.float16)
    priors.flush()
    del priors

    cuts = [0]
    position = 0
    while position < n_tokens:
        step = int(rng.integers(1, 600))
        position = min(position + step, n_tokens)
        cuts.append(position)
    if cuts[-1] != n_tokens:
        cuts.append(n_tokens)
    boundaries = np.array(sorted(set(cuts)), dtype=np.int32)
    boundaries.tofile(_path(dst, "boundaries"))

    # bitmap: the first `vocab` byte values, which is what a 208 byte set looks
    # like only by accident; the real one comes from our coder stream
    bits = np.zeros(256, dtype=np.uint8)
    bits[:vocab] = 1
    np.packbits(bits, bitorder="little").tofile(_path(dst, "bitmap"))

    return {
        "dir": dst,
        "n_tokens": n_tokens,
        "vocabulary_size": vocab,
        "n_articles": int(len(boundaries)) - 1,
    }


def _print(info: dict) -> None:
    for key, value in info.items():
        if isinstance(value, list) and len(value) > 12:
            value = str(value[:12])[:-1] + ", ... " + str(len(value)) + " total]"
        print(str(key).ljust(26) + str(value))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    command = args[0]
    try:
        if command == "check":
            directory = args[1] if len(args) > 1 else os.environ.get("TFT_DATA_DIR", "data")
            expect = int(args[2]) if len(args) > 2 else DEFAULT_VOCAB
            _print(check(directory, expect))
            print("\nOK: package matches the training-data interface")
        elif command == "cut":
            _print(cut(args[1], args[2], int(args[3])))
            print("\nOK: window written; run `check` on it before training")
        elif command == "mini":
            dst = args[1]
            n = int(args[args.index("--tokens") + 1]) if "--tokens" in args else 4096
            v = int(args[args.index("--vocab") + 1]) if "--vocab" in args else DEFAULT_VOCAB
            _print(mini(dst, n, v))
            print("\nOK: mini package written")
        else:
            print(__doc__)
            return 2
    except CheckFailed as e:
        print("CHECK FAILED: " + str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
