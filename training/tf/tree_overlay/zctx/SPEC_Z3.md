# SPEC_Z3: the link-context prefix, bit-exact

This file is the contract of the link-context rule (prompt `z3`). The Python
reference implementation `zctx/z3_extract.py` and the implementation in the
coder must produce the same prefix, token for token; where text and code
disagree, the text holds.

The rule works on the token stream only, never on raw `enwik9`, never on the
dictionary and never on a list of articles. At the point where the decoder needs
the prefix it has decoded exactly the tokens in front of the article boundary,
and the rule may not see more.

The grammar constants and figures below belong to the earlier stream form on
which the training chain of this directory was first run (`training/README.md`
section 2), and sections 5 and 8 describe the first training form, in which
the prefix stood in front of the article. In the current training form the
link context is part of the prompt block behind the title
(`training/README.md` section 4); the coder stamps the grammar constants of
its own stream from the dictionary.

## 1. Inputs

| name | type | source |
|---|---|---|
| `tokens[0..N)` | `uint8`, vocabulary index 0..207 | the coder stream (`ppmd-bytes.uint8`; in the coder, the tokens coded or decoded so far) |
| `boundaries[0..A]` | `int32`, strictly increasing, `[0] == 0`, `[A] == N` | `article-boundaries.int32`; in the coder, the positions behind the 15-token article separator, counted by the decoder itself |

## 2. Grammar constants (token indices, decimal)

```
ARTICLE_SEPARATOR = 11 11 40 175 104 42 8 11 11 11 11 40 175 107 42
TITLE_TERMINATOR  = 40 26 175 107 42
LINK_OPEN         = 49
PARAGRAPH_TOKEN   = 8
MAX_TITLE_TOKENS  = 64
MIN_TITLE_TOKENS  = 3
```

Read as bytes these are `"  <page>\n    <title>"`, `"</title>"`, `"["` and
`"\n"`; the dictionary maps `<` and `>` to other byte values, and the word
replacement folds `[[` into a single `[`. The constants depend on the
dictionary, like Ivanov's `kArticleSeparator`. `zctx/zgrammar.py verify`
re-derives each of them from a package and stops if a single article boundary
is not preceded by the separator.

## 3. Title of an article

```
title(i):
    start = boundaries[i]
    stop  = min(start + MAX_TITLE_TOKENS + 5, boundaries[i+1])
    p     = smallest p in [start, stop-5] with tokens[p..p+5) == TITLE_TERMINATOR
    no such p                          -> no title
    p - start < MIN_TITLE_TOKENS       -> no title
    otherwise title = tokens[start..p)
```

An article without a title gets no prefix. A title of one or two tokens is a
single frequent dictionary word, and its mentions are noise instead of
incoming links.

## 4. Mention and the paragraph rule

A **mention** of article i is a position `m0` with

```
tokens[m0] == LINK_OPEN  and  tokens[m0+1 .. m0+1+L) == title(i)
```

where `L = |title(i)|`. It covers `[m0, m1)` with `m1 = m0 + 1 + L`; the link
token is part of it.

A **paragraph** is the run between two `PARAGRAPH_TOKEN`s, without them. The
search is bounded by `LINE_SCAN` tokens on each side and on the right also by
`limit`, the start of the target article:

```
paragraph(m0, m1, limit):
    lo    = max(0, m0 - LINE_SCAN)
    start = (last j in [lo, m0) with tokens[j] == NL) + 1, else lo
    hi    = min(N, m1 + LINE_SCAN, limit)
    end   = (first j in [m1, hi) with tokens[j] == NL), else hi
```

The **chunk** with budget `C`:

```
chunk(m0, m1, C, limit):
    span = m1 - m0
    if span >= C:                       -> [m0, m0 + C)
    (start, end) = paragraph(m0, m1, limit)
    room_before = m0 - start
    room_after  = end - m1
    need = C - span
    half = need // 2
    if room_after <= half:              # mention at the end of the paragraph
        take_after  = room_after
        take_before = min(room_before, need - take_after)
    elif room_before <= half:           # mention at the start of the paragraph
        take_before = room_before
        take_after  = min(room_after, need - take_before)
    else:                               # mention in the middle
        take_before = half
        take_after  = need - half
    -> [m0 - take_before, m1 + take_after)
```

A link at the end of a paragraph takes the tokens before it, a link at the
start the tokens after it, a link in the middle both. The chunk never leaves
the paragraph and always contains the mention.

Causality is secured twice. Only mentions with `m1 <= boundaries[i]` are
considered, and `limit` cuts the chunk at `boundaries[i]` as well. With a
correct grammar the second guard is redundant (the separator carries a
newline); it is kept because a causality error produces an archive that does
not decode.

## 5. Selection of the mentions and the prefix

```
prefix(i):
    E = all mention positions m0 with m0 + 1 + L <= boundaries[i], ascending
    E empty -> prefix of length 0
    room = BUDGET - 15                  (15 = length of ARTICLE_SEPARATOR)
    taken = [], used = 0
    for m0 in E backwards, at most MAX_MENTIONS of them:
        (a, b) = chunk(m0, m0+1+L, CHUNK, boundaries[i])
        [a, b) overlaps a chunk already taken -> skip
        cost = (b - a) + (2 if taken else 0)
        used + cost > room -> stop
        taken += [(a, b)];  used += cost
    taken empty -> prefix of length 0
    prefix = oldest chunk ++ NL NL ++ ... ++ newest chunk ++ ARTICLE_SEPARATOR
```

The newest mentions are selected first, because they are closest to the
article. The oldest chunk is written first, so that the newest one ends up
next to the article, in stream order. The prefix ends with the article
separator, a token sequence the model already knows as the start of an
article, so the prefix needs no new symbol. An empty prefix writes no
separator.

## 6. Parameters

| parameter | default | meaning |
|---|---:|---|
| `BUDGET` | 256 | prefix length per article, separator included |
| `CHUNK` | 64 | budget of one chunk, mention included |
| `MAX_MENTIONS` | 8 | at most this many mentions are considered |
| `LINE_SCAN` | 512 | bound of the paragraph search on each side |
| `mode` | `link` | `plain` = title without the link token (section 9) |

On a window of 200 M tokens of the earlier stream form, 72.2 % of the articles
got a prefix, and the prefixes added 3.1 % tokens (mean length 160 tokens,
2.9 chunks). A smaller budget costs fewer tokens (128/64: +1.6 %), a larger
one more (512/64: +4.6 %); the share of articles with a prefix does not
depend on the budget.

## 7. Prior rows of the prefix positions

Every input position needs a PPMd prior row. The original rows of the copied
positions cannot be delivered by the decoder, which would have to keep 208
float16 values per token or rewind the PPMd model. Every prefix position
therefore gets the order-0 table of the stream, `z3-prior0.float16` (208
float16, 416 B), and the coder stamps the same table. Training on rows the
coder cannot deliver would measure a feature that cannot be built.

## 8. Output format

```
z3-prefix-tokens.uint8    all prefixes in sequence, token indices 0..207
z3-prefix-offsets.int32   A+1 offsets; offsets[0] == 0, offsets[A] == |tokens|,
                          monotone; prefix length of article i = off[i+1] - off[i]
z3-prior0.float16         208 values, sum about 1
z3-manifest.json          parameters, grammar, statistics, sha256
```

`python -m zctx.z3_extract check <package> <z3-dir>` checks the monotone
offsets, the end value, the token bound 207, that every non-empty prefix ends
with the separator and that none exceeds the budget.

## 9. Variants not built

* `mode = plain`, the title without the link token: more articles get a
  prefix, but from about three times as many mentions, mostly category lines,
  navigation boxes and running text without link character. It is an option;
  the default is `link`.
* Mentions after the article: not causal, not built.
* Deduplication by chunk content instead of position overlap: it would save
  budget in year and navigation articles, but adds a hash the coder would have
  to carry. Not built.

## 10. Acceptance of the coder-side implementation

The implementation in the coder is accepted when it produces, on the same
package, the same `z3-prefix-tokens.uint8` and offsets as the reference, byte
for byte. It is checked in three stages, for the same list of titles and
rejections, the same set of mention positions and the same prefix stream. The
three cases of the paragraph rule (end, start, middle) and the causality case
(chunk cut at the article boundary) are unit tests in `zctx/dryrun_zctx.py`.
