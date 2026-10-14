#!/usr/bin/env python3
"""z3, incoming-link context: the extractor, the format, and the checker

Idea: when earlier articles in the stream link to this one, give the
transformer the link text or a context chunk around it, aligned to the
paragraph: a link at the end of a paragraph takes the bytes before it, not
the bytes after it.

What this builds, per article i:
  * its title = the tokens between its article boundary and the first
    title terminator,
  * every EARLIER occurrence of the link opening token followed by the title
    (an incoming wiki link; the dictionary collapses the doubled bracket to a
    single one), restricted to positions < boundary[i] for causality: the
    decoder has decoded exactly that much when it needs the prefix,
  * one paragraph-aligned chunk per kept mention, directed away from the
    paragraph edge (the paragraph rule, SPEC_Z3.md section 4),
  * the concatenation, oldest chunk first, closed by the 15 token article
    separator so the model sees the familiar "an article starts here" mark.

Output (an extra stream in the training format):
  z3-prefix-tokens.uint8   concatenated prefixes, token indices
  z3-prefix-offsets.int32  n_articles + 1 offsets, offsets[0] == 0,
                           offsets[-1] == len(prefix tokens); the length field
                           of article i is offsets[i+1] - offsets[i] (0 = none)
  z3-prior0.float16        208 float16, the order 0 token distribution of the
                           window; the prior rows the loader feeds for prefix
                           positions (the coder cannot replay a past PPMD state,
                           so training must not use the original prior rows)
  z3-manifest.json         parameters, grammar, statistics, sha256 of the above

Everything is computed from the TOKEN STREAM alone (never from enwik9 raw text)
so the coder can reproduce it bit for bit; SPEC_Z3.md is the contract.

    python -m zctx.z3_extract build <package-dir> <out-dir> [--budget 256] ...
    python -m zctx.z3_extract check <package-dir> <out-dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from zctx import zgrammar as G

FILES = {
    "prefix_tokens": "z3-prefix-tokens.uint8",
    "prefix_offsets": "z3-prefix-offsets.int32",
    "prior0": "z3-prior0.float16",
    "manifest": "z3-manifest.json",
}

# defaults, all overridable on the command line; see SPEC_Z3.md section 6
DEFAULTS = {
    "budget": 256,        # total prefix tokens per article, separator INCLUDED
    "chunk": 64,          # token budget of one chunk, mention included
    "max_mentions": 8,    # never look at more than this many recent mentions
    "line_scan": 512,     # bound of the paragraph search on each side
    "mode": "link",       # "link" = opening token + title, "plain" = bare title
}

_FNV_OFFSET = np.uint64(1469598103934665603)
_FNV_PRIME = np.uint64(1099511628211)


def _hash_step(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (h ^ t.astype(np.uint64)) * _FNV_PRIME


def _key_hash(seq: np.ndarray) -> np.uint64:
    h = _FNV_OFFSET
    for t in seq:
        h = (h ^ np.uint64(int(t))) * _FNV_PRIME
    return h


def read_package(directory: str) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.fromfile(os.path.join(directory, "ppmd-bytes.uint8"), dtype=np.uint8)
    boundaries = np.fromfile(
        os.path.join(directory, "article-boundaries.int32"), dtype=np.int32
    ).astype(np.int64)
    assert boundaries[0] == 0 and int(boundaries[-1]) == len(tokens)
    return tokens, boundaries


def extract_titles(tokens: np.ndarray, boundaries: np.ndarray) -> list:
    """title token sequence per article, or None when there is no usable one"""
    titles: list = []
    term = len(G.TITLE_TERMINATOR)
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        stop = min(start + G.MAX_TITLE_TOKENS + term, int(boundaries[i + 1]))
        p = G.find_first(tokens, G.TITLE_TERMINATOR, start, stop)
        if p < 0 or p - start < G.MIN_TITLE_TOKENS:
            titles.append(None)
        else:
            titles.append(tokens[start:p].copy())
    return titles


def find_mentions(
    tokens: np.ndarray,
    titles: list,
    mode: str = "link",
    key_len: int = 8,
    log=print,
) -> dict:
    """all positions of "opening token + title" in the stream, per article

    Anchored on the link opening token so the scan touches ~2.5 % of the stream
    instead of all of it; mode="plain" drops the anchor and scans every position
    (documented variant, SPEC_Z3.md section 9).
    Returns {article: sorted positions of the OPENING token}.
    """
    n = len(tokens)
    open_len = len(G.LINK_OPEN) if mode == "link" else 0
    if mode == "link":
        anchors = np.flatnonzero(tokens == G.LINK_OPEN[0]).astype(np.int64)
    elif mode == "plain":
        anchors = np.arange(n, dtype=np.int64)
    else:
        raise ValueError("mode must be 'link' or 'plain'")
    log("  anchors: %d (%.2f %% of the stream)" % (len(anchors), 100.0 * len(anchors) / n))

    # bucket the titles by the length of their hash key
    by_key_len: dict = {}
    for i, t in enumerate(titles):
        if t is None:
            continue
        by_key_len.setdefault(min(len(t), key_len), []).append(i)

    starts = anchors + open_len
    h = np.full(len(starts), _FNV_OFFSET, dtype=np.uint64)
    alive = np.ones(len(starts), dtype=bool)
    hits: dict = {}

    with np.errstate(over="ignore"):
        for length in range(1, key_len + 1):
            idx = starts + (length - 1)
            alive &= idx < n
            safe = np.where(alive, idx, 0)
            h = _hash_step(h, tokens[safe])
            group = by_key_len.get(length)
            if not group:
                continue
            keys = np.array([int(_key_hash(titles[i][:length])) for i in group], dtype=np.uint64)
            order = np.argsort(keys, kind="stable")
            keys_sorted = keys[order]
            pos = np.searchsorted(keys_sorted, h, side="left")
            clipped = np.minimum(pos, len(keys_sorted) - 1)
            cand = np.flatnonzero(alive & (pos < len(keys_sorted)) & (keys_sorted[clipped] == h))
            log("    key length %d: %d titles, %d hash candidates" % (length, len(group), len(cand)))
            for c in cand:
                a = int(anchors[c])
                s = a + open_len
                k = int(pos[c])
                while k < len(keys_sorted) and keys_sorted[k] == h[c]:
                    article = group[int(order[k])]
                    title = titles[article]
                    if s + len(title) <= n and bool((tokens[s : s + len(title)] == title).all()):
                        hits.setdefault(article, []).append(a)
                    k += 1

    return {a: np.array(sorted(set(p)), dtype=np.int64) for a, p in hits.items()}


def paragraph_bounds(tokens: np.ndarray, m0: int, m1: int, line_scan: int, limit: int):
    """the paragraph (= line) containing the mention, bounded by line_scan

    A paragraph is the run between two newline tokens; the newlines themselves
    are not part of it.  The scan is bounded on both sides so the C++ port has
    a fixed worst case, and on the right additionally by `limit` (the article
    start), so that nothing the decoder has not decoded yet can leak in.
    """
    lo = max(0, m0 - line_scan)
    left = tokens[lo:m0]
    nl = np.flatnonzero(left == G.PARAGRAPH_TOKEN)
    start = lo + int(nl[-1]) + 1 if len(nl) else lo

    hi = min(len(tokens), m1 + line_scan, limit)
    right = tokens[m1:hi]
    nl = np.flatnonzero(right == G.PARAGRAPH_TOKEN)
    end = m1 + int(nl[0]) if len(nl) else hi
    return start, end


def chunk_for_mention(tokens: np.ndarray, m0: int, m1: int, budget: int, line_scan: int, limit: int):
    """the paragraph rule, exactly; see SPEC_Z3.md section 4

    m0..m1 is the mention (opening token included).  The chunk never leaves the
    paragraph, never reaches past `limit`, and always contains the mention.
    """
    span = m1 - m0
    if span >= budget:
        return m0, m0 + budget
    start, end = paragraph_bounds(tokens, m0, m1, line_scan, limit)
    room_before = m0 - start
    room_after = end - m1
    need = budget - span
    half = need // 2
    if room_after <= half:                      # mention at the paragraph END
        take_after = room_after
        take_before = min(room_before, need - take_after)
    elif room_before <= half:                   # mention at the paragraph START
        take_before = room_before
        take_after = min(room_after, need - take_before)
    else:                                       # in the middle: symmetric
        take_before = half
        take_after = need - half
    return m0 - take_before, m1 + take_after


def build_prefix(
    tokens: np.ndarray,
    mentions: np.ndarray,
    title_len: int,
    article_start: int,
    budget: int,
    chunk: int,
    max_mentions: int,
    line_scan: int,
    open_len: int,
) -> list:
    """the prefix of one article: newest mentions first, laid out oldest first"""
    earlier = mentions[mentions + open_len + title_len <= article_start]
    if len(earlier) == 0:
        return []
    sep = G.ARTICLE_SEPARATOR
    room = budget - len(sep)
    if room <= 0:
        return []

    taken: list = []
    used = 0
    for m0 in earlier[::-1][:max_mentions]:
        m0 = int(m0)
        m1 = m0 + open_len + title_len
        a, b = chunk_for_mention(tokens, m0, m1, chunk, line_scan, article_start)
        assert b <= article_start, "causality: chunk reaches past the article start"
        if any(a < tb and ta < b for ta, tb in taken):
            continue                              # overlaps a chunk already taken
        cost = (b - a) + (2 if taken else 0)      # two newlines between chunks
        if used + cost > room:
            break
        taken.append((a, b))
        used += cost
    if not taken:
        return []

    out: list = []
    for k, ab in enumerate(reversed(taken)):      # oldest first, newest adjacent
        if k:
            out.extend([G.PARAGRAPH_TOKEN, G.PARAGRAPH_TOKEN])
        out.extend(int(t) for t in tokens[ab[0] : ab[1]])
    out.extend(sep)
    return out


def build(directory: str, out_dir: str, params: dict, log=print) -> dict:
    t0 = time.time()
    tokens, boundaries = read_package(directory)
    n_articles = len(boundaries) - 1
    log("package: %d tokens, %d articles" % (len(tokens), n_articles))

    log("grammar self check ...")
    info = G.verify(directory)
    log("  titles without terminator: %d, below the %d token minimum: %d"
        % (info["titles_without_terminator"], G.MIN_TITLE_TOKENS, info.get("titles_below_min", 0)))

    titles = extract_titles(tokens, boundaries)
    n_usable = sum(1 for t in titles if t is not None)
    log("usable titles: %d of %d" % (n_usable, n_articles))

    log("mention search (mode=%s) ..." % params["mode"])
    mentions = find_mentions(tokens, titles, mode=params["mode"], log=log)
    open_len = len(G.LINK_OPEN) if params["mode"] == "link" else 0
    total_mentions = sum(len(v) for v in mentions.values())
    log("  %d articles have at least one mention anywhere, %d mentions total"
        % (len(mentions), total_mentions))

    log("prefix assembly ...")
    offsets = np.zeros(n_articles + 1, dtype=np.int64)
    parts: list = []
    n_with_prefix = 0
    chunk_lengths: list = []
    n_chunks: list = []
    for i in range(n_articles):
        title = titles[i]
        pre: list = []
        if title is not None and i in mentions:
            pre = build_prefix(
                tokens,
                mentions[i],
                len(title),
                int(boundaries[i]),
                params["budget"],
                params["chunk"],
                params["max_mentions"],
                params["line_scan"],
                open_len,
            )
        if pre:
            n_with_prefix += 1
            body = len(pre) - len(G.ARTICLE_SEPARATOR)
            chunk_lengths.append(body)
            n_chunks.append(1 + pre[:body].count(G.PARAGRAPH_TOKEN) // 2)
            parts.append(np.array(pre, dtype=np.uint8))
        offsets[i + 1] = offsets[i] + len(pre)

    prefix_tokens = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    assert int(offsets[-1]) == len(prefix_tokens)

    os.makedirs(out_dir, exist_ok=True)
    prefix_tokens.tofile(os.path.join(out_dir, FILES["prefix_tokens"]))
    offsets.astype(np.int32).tofile(os.path.join(out_dir, FILES["prefix_offsets"]))

    counts = np.bincount(tokens, minlength=256)[:208].astype(np.float64)
    prior0 = ((counts + 1.0) / (counts.sum() + 208.0)).astype(np.float16)
    prior0.tofile(os.path.join(out_dir, FILES["prior0"]))

    cl = np.asarray(chunk_lengths) if chunk_lengths else np.zeros(1)
    nc = np.asarray(n_chunks) if n_chunks else np.zeros(1)
    stats = {
        "n_tokens": int(len(tokens)),
        "n_articles": int(n_articles),
        "usable_titles": int(n_usable),
        "articles_with_mentions": int(len(mentions)),
        "articles_with_prefix": int(n_with_prefix),
        "hit_rate_percent": round(100.0 * n_with_prefix / n_articles, 3),
        "mentions_total": int(total_mentions),
        "prefix_tokens_total": int(len(prefix_tokens)),
        "prefix_tokens_percent_of_stream": round(100.0 * len(prefix_tokens) / len(tokens), 3),
        "prefix_len_mean_over_all_articles": round(float(len(prefix_tokens)) / n_articles, 2),
        "prefix_len_mean_where_present": round(float(cl.mean()) + len(G.ARTICLE_SEPARATOR), 2),
        "chunk_body_len_median": float(np.median(cl)),
        "chunk_body_len_max": int(cl.max()),
        "chunks_per_prefix_mean": round(float(nc.mean()), 2),
        "seconds": round(time.time() - t0, 1),
    }
    manifest = {
        "format_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_package": os.path.abspath(directory),
        "parameters": params,
        "grammar": {
            "ARTICLE_SEPARATOR": G.ARTICLE_SEPARATOR,
            "TITLE_TERMINATOR": G.TITLE_TERMINATOR,
            "LINK_OPEN": G.LINK_OPEN,
            "PARAGRAPH_TOKEN": G.PARAGRAPH_TOKEN,
            "MAX_TITLE_TOKENS": G.MAX_TITLE_TOKENS,
            "MIN_TITLE_TOKENS": G.MIN_TITLE_TOKENS,
        },
        "statistics": stats,
        "sha256": {
            name: _sha256(os.path.join(out_dir, name))
            for name in (FILES["prefix_tokens"], FILES["prefix_offsets"], FILES["prior0"])
        },
    }
    with open(os.path.join(out_dir, FILES["manifest"]), "w") as f:
        json.dump(manifest, f, indent=2)
    for key, value in stats.items():
        log("  %-38s %s" % (key, value))
    return manifest


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(directory: str, out_dir: str) -> dict:
    """the format checker: does the Z3 stream fit the package it claims to fit"""
    problems: list = []
    with open(os.path.join(out_dir, FILES["manifest"])) as f:
        manifest = json.load(f)
    boundaries = np.fromfile(
        os.path.join(directory, "article-boundaries.int32"), dtype=np.int32
    ).astype(np.int64)
    n_articles = len(boundaries) - 1

    pre = np.fromfile(os.path.join(out_dir, FILES["prefix_tokens"]), dtype=np.uint8)
    off = np.fromfile(os.path.join(out_dir, FILES["prefix_offsets"]), dtype=np.int32).astype(np.int64)
    if len(off) != n_articles + 1:
        problems.append("offsets have %d entries, need n_articles + 1 = %d" % (len(off), n_articles + 1))
    if len(off) and off[0] != 0:
        problems.append("offsets[0] is %d, must be 0" % off[0])
    if len(off) and int(off[-1]) != len(pre):
        problems.append("offsets[-1] is %d, prefix stream has %d tokens" % (int(off[-1]), len(pre)))
    if len(off) > 1 and (np.diff(off) < 0).any():
        problems.append("offsets are not monotone")
    if len(pre) and int(pre.max()) > 207:
        problems.append("prefix stream contains token index %d > 207" % int(pre.max()))

    sep = np.asarray(G.ARTICLE_SEPARATOR, dtype=np.uint8)
    budget = manifest["parameters"]["budget"]
    n_present = 0
    if not problems:
        for i in range(n_articles):
            length = int(off[i + 1] - off[i])
            if length == 0:
                continue
            n_present += 1
            if length < len(sep) + 1:
                problems.append("article %d has a prefix of %d tokens, shorter than the separator" % (i, length))
                break
            tail = pre[off[i + 1] - len(sep) : off[i + 1]]
            if not bool((tail == sep).all()):
                problems.append("article %d prefix does not end with the article separator" % i)
                break
            if length > budget:
                problems.append("article %d prefix is %d tokens, over the budget %d" % (i, length, budget))
                break

    prior0 = np.fromfile(os.path.join(out_dir, FILES["prior0"]), dtype=np.float16)
    if len(prior0) != 208:
        problems.append("z3-prior0 has %d entries, need 208" % len(prior0))
    elif not (0.5 < float(np.asarray(prior0, dtype=np.float64).sum()) < 2.0):
        problems.append("z3-prior0 does not sum near 1")

    info = {
        "articles": n_articles,
        "articles_with_prefix": n_present,
        "prefix_tokens": int(len(pre)),
        "longest_prefix": int(np.diff(off).max()) if len(off) > 1 else 0,
        "problems": problems,
    }
    if problems:
        raise AssertionError("; ".join(problems))
    return info


def render(directory: str, out_dir: str, article: int, context: int = 160) -> str:
    """human readable view of one prefix, for review, never for the algorithm

    Maps token indices back to byte values through the vocabulary bitmap; the
    dictionary code words stay as hex, which is enough to see paragraph shape
    and where the link sits.
    """
    bitmap = np.fromfile(os.path.join(directory, "vocab-bitmap.u8"), dtype=np.uint8)
    values = np.flatnonzero(np.unpackbits(bitmap, bitorder="little")).astype(np.uint8)
    pre = np.fromfile(os.path.join(out_dir, FILES["prefix_tokens"]), dtype=np.uint8)
    off = np.fromfile(os.path.join(out_dir, FILES["prefix_offsets"]), dtype=np.int32).astype(np.int64)
    tokens = np.memmap(os.path.join(directory, "ppmd-bytes.uint8"), dtype=np.uint8, mode="r")
    boundaries = np.fromfile(
        os.path.join(directory, "article-boundaries.int32"), dtype=np.int32
    ).astype(np.int64)

    def text(seq) -> str:
        out = []
        for b in values[np.asarray(seq)].tolist():
            out.append(chr(b) if 32 <= b < 127 else ("\\n" if b == 10 else "<%02x>" % b))
        return "".join(out)

    start = int(boundaries[article])
    lines = [
        "article %d, stream position %d, prefix %d tokens"
        % (article, start, int(off[article + 1] - off[article])),
        "PREFIX : " + text(pre[off[article] : off[article + 1]]),
        "ARTICLE: " + text(tokens[start : start + context]),
    ]
    return "\n".join(lines)


def _parse(args: list) -> dict:
    params = dict(DEFAULTS)
    for key in ("budget", "chunk", "max_mentions", "line_scan"):
        flag = "--" + key
        if flag in args:
            params[key] = int(args[args.index(flag) + 1])
    if "--mode" in args:
        params["mode"] = args[args.index("--mode") + 1]
    return params


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 3:
        print(__doc__)
        return 2
    if args[0] == "build":
        build(args[1], args[2], _parse(args))
        print("\nOK: Z3 stream written to " + args[2])
    elif args[0] == "check":
        for key, value in check(args[1], args[2]).items():
            print(str(key).ljust(28) + str(value))
        print("\nOK: Z3 stream matches the package")
    elif args[0] == "show":
        first = int(args[3]) if len(args) > 3 else 0
        count = int(args[4]) if len(args) > 4 else 1
        for a in range(first, first + count):
            print(render(args[1], args[2], a))
            print()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
