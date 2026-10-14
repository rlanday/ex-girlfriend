#!/usr/bin/env python3
"""zctx dry runs: everything that can be checked without a GPU

Proves on a machine without a GPU what can be proven there, and names what
only the GPU host can answer.  Uses the import stub `tft/stub_imports.py`
for the torch-only modules.

    python -m zctx.dryrun_zctx                    # synthetic data only
    python -m zctx.dryrun_zctx <lite-package>     # plus a real stream slice
                                                  # (tokens + boundaries only)
"""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from zctx import z3_extract, zfeat  # noqa: E402
from zctx import zgrammar as G  # noqa: E402

PASS: list = []
FAIL: list = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        PASS.append((name, detail))
        print("  ok    %-46s %s" % (name, detail))
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, repr(exc)))
        print("  FAIL  %-46s %r" % (name, exc))


# --------------------------------------------------------------------------- #
# a synthetic stream in the real grammar
# --------------------------------------------------------------------------- #
STRUCTURAL = set(G.ARTICLE_SEPARATOR) | set(G.TITLE_TERMINATOR) | set(G.LINK_OPEN) | {
    G.PARAGRAPH_TOKEN
}
FREE = [t for t in range(60, 208) if t not in STRUCTURAL]


def synth_package(directory: str, n_articles: int = 40, seed: int = 3) -> dict:
    """a token stream that obeys the grammar, with planted incoming links"""
    rng = np.random.default_rng(seed)
    titles = [
        [int(x) for x in rng.choice(FREE, size=int(rng.integers(3, 9)), replace=False)]
        for _ in range(n_articles)
    ]
    stream: list = []
    boundaries: list = [0]
    planted: dict = {}

    def paragraph(length: int) -> list:
        return [int(x) for x in rng.choice(FREE, size=length)]

    for i in range(n_articles):
        if i:
            stream.extend(G.ARTICLE_SEPARATOR)
            boundaries.append(len(stream))
        else:
            stream.extend(G.ARTICLE_SEPARATOR)
            boundaries[0] = len(stream)          # article 0 also starts after one
        stream.extend(titles[i])
        stream.extend(G.TITLE_TERMINATOR)
        # body: a few paragraphs, some of them carrying a link to a LATER article
        for _ in range(int(rng.integers(2, 6))):
            body = paragraph(int(rng.integers(10, 90)))
            target = int(rng.integers(0, n_articles))
            if target > i:
                where = int(rng.integers(0, len(body)))
                body = body[:where] + G.LINK_OPEN + titles[target] + body[where:]
                planted.setdefault(target, []).append(len(stream) + where)
            stream.extend(body)
            stream.append(G.PARAGRAPH_TOKEN)
    boundaries.append(len(stream))
    # the loader requires boundaries[0] == 0; shift the whole thing so the first
    # article starts at 0 by dropping the leading separator
    cut = boundaries[0]
    stream = stream[cut:]
    boundaries = [b - cut for b in boundaries]
    boundaries[0] = 0

    os.makedirs(directory, exist_ok=True)
    np.array(stream, dtype=np.uint8).tofile(
        os.path.join(directory, "ppmd-bytes.uint8")
    )
    np.array(boundaries, dtype=np.int32).tofile(
        os.path.join(directory, "article-boundaries.int32")
    )
    bits = np.zeros(256, dtype=np.uint8)
    bits[:208] = 1
    np.packbits(bits, bitorder="little").tofile(
        os.path.join(directory, "vocab-bitmap.u8")
    )
    return {"n_tokens": len(stream), "n_articles": len(boundaries) - 1,
            "planted_targets": len(planted)}


# --------------------------------------------------------------------------- #
def main() -> int:
    real = sys.argv[1] if len(sys.argv) > 1 else None
    work = tempfile.mkdtemp(prefix="zctx-dry-")
    print("zctx dry runs   work dir " + work)

    print("\n1  every zctx module compiles")

    def compile_all():
        n = 0
        for name in sorted(os.listdir(HERE)):
            if name.endswith(".py"):
                py_compile.compile(os.path.join(HERE, name), doraise=True)
                n += 1
        return "%d files" % n

    check("py_compile zctx/*.py", compile_all)

    print("\n2  Z1 feature core (numpy, no torch)")
    check("zfeat self test", lambda: (zfeat.selftest(), "ok")[1])

    def packing_edges():
        # the extremes the real corpus can reach
        tok = np.array([0, 207, 3], dtype=np.int64)
        art = np.array([0, 243_426, 1_048_575], dtype=np.int64)
        pos = np.array([0, 601_499_999, 1_073_741_823], dtype=np.int64)
        packed = zfeat.pack(tok, art, pos)
        t, a, p = zfeat.unpack(packed)
        assert (t == tok).all() and (a == art).all() and (p == pos).all()
        assert packed.max() < (1 << 62)
        return "corpus extremes round trip, max %d" % int(packed.max())

    check("packing survives the corpus extremes", packing_edges)

    def feature_monotone():
        table = zfeat.phase_table()
        f = zfeat.features(np.arange(0, 240_000, 1000), np.zeros(240, np.int64), table)
        assert f.shape == (240, zfeat.N_FEATURES)
        # the coarsest article level must sweep less than one full period over
        # the whole corpus, otherwise ordinals alias onto each other
        idx = zfeat.phase_indices(np.array([0, 243_426]), zfeat.ARTICLE_BITS, 1)
        assert int(idx[0, 0]) < int(idx[1, 0]), "level 0 aliases over enwik9"
        return "no aliasing on level 0 over 243427 articles"

    check("article levels do not alias", feature_monotone)

    print("\n3  the grammar on a synthetic stream")
    synth = os.path.join(work, "synth")
    info = synth_package(synth)
    check("synthetic package built", lambda: "%(n_tokens)d tokens, %(n_articles)d articles" % info)
    check("zgrammar.verify accepts it",
          lambda: "separator misses %d, titles without terminator %d"
                  % (G.verify(synth)["boundaries_without_separator"],
                     G.verify(synth)["titles_without_terminator"]))

    def grammar_rejects():
        broken = os.path.join(work, "broken")
        shutil.copytree(synth, broken)
        t = np.fromfile(os.path.join(broken, "ppmd-bytes.uint8"), dtype=np.uint8)
        b = np.fromfile(os.path.join(broken, "article-boundaries.int32"), dtype=np.int32)
        t[int(b[1]) - 3] = 200                  # damage one separator
        t.tofile(os.path.join(broken, "ppmd-bytes.uint8"))
        try:
            G.verify(broken)
        except AssertionError:
            return "a damaged separator is rejected"
        raise AssertionError("a damaged separator was NOT rejected")

    check("zgrammar.verify rejects a wrong dictionary", grammar_rejects)

    print("\n4  the paragraph rule, the three cases")

    def rule_cases():
        NL = G.PARAGRAPH_TOKEN
        # one paragraph of 100 free tokens between two newlines
        line = [NL] + [70] * 100 + [NL]
        toks = np.array(line, dtype=np.uint8)
        n = len(toks)
        # mention of length 4 at the END of the paragraph
        m0, m1 = 97, 101
        a, b = z3_extract.chunk_for_mention(toks, m0, m1, 32, 512, n)
        assert b <= 101 and a == 101 - 32, (a, b)
        end_case = (a, b)
        # ... at the START
        m0, m1 = 1, 5
        a2, b2 = z3_extract.chunk_for_mention(toks, m0, m1, 32, 512, n)
        assert a2 == 1 and b2 == 33, (a2, b2)
        start_case = (a2, b2)
        # ... in the MIDDLE
        m0, m1 = 50, 54
        a3, b3 = z3_extract.chunk_for_mention(toks, m0, m1, 32, 512, n)
        assert a3 == 50 - 14 and b3 == 54 + 14, (a3, b3)
        middle_case = (a3, b3)
        # the chunk never leaves the paragraph
        for a4, b4 in (end_case, start_case, middle_case):
            assert 1 <= a4 and b4 <= 101
        return "end %s, start %s, middle %s" % (end_case, start_case, middle_case)

    check("end -> before, start -> after, middle -> symmetric", rule_cases)

    def rule_limit():
        NL = G.PARAGRAPH_TOKEN
        toks = np.array([NL] + [70] * 200 + [NL], dtype=np.uint8)
        a, b = z3_extract.chunk_for_mention(toks, 100, 104, 64, 512, limit=110)
        assert b <= 110, (a, b)
        return "chunk clipped at the article start (%d, %d)" % (a, b)

    check("causality: the chunk stops at the article start", rule_limit)

    print("\n5  the Z3 stream: build, check, and what it must never do")
    out = os.path.join(work, "z3synth")
    manifest = z3_extract.build(synth, out, dict(z3_extract.DEFAULTS), log=lambda *a: None)
    check("build produced a stream",
          lambda: "%d of %d articles have a prefix, %d tokens"
                  % (manifest["statistics"]["articles_with_prefix"],
                     manifest["statistics"]["n_articles"],
                     manifest["statistics"]["prefix_tokens_total"]))
    check("planted links are found",
          lambda: (_assert(manifest["statistics"]["articles_with_prefix"] > 0,
                           "no planted link was retrieved"), "hit rate %.1f %%"
                   % manifest["statistics"]["hit_rate_percent"])[1])
    check("z3_extract.check accepts it",
          lambda: "%d prefixes, longest %d"
                  % (z3_extract.check(synth, out)["articles_with_prefix"],
                     z3_extract.check(synth, out)["longest_prefix"]))

    def check_rejects():
        bad = os.path.join(work, "z3broken")
        shutil.copytree(out, bad)
        off = np.fromfile(os.path.join(bad, z3_extract.FILES["prefix_offsets"]), dtype=np.int32)
        off[-1] = off[-1] + 7                    # length field no longer fits
        off.tofile(os.path.join(bad, z3_extract.FILES["prefix_offsets"]))
        try:
            z3_extract.check(synth, bad)
        except AssertionError:
            return "a wrong length field is rejected"
        raise AssertionError("a wrong length field was NOT rejected")

    check("z3_extract.check rejects a damaged stream", check_rejects)

    def causality_full():
        pre = np.fromfile(os.path.join(out, z3_extract.FILES["prefix_tokens"]), dtype=np.uint8)
        off = np.fromfile(os.path.join(out, z3_extract.FILES["prefix_offsets"]), dtype=np.int32)
        sep = np.asarray(G.ARTICLE_SEPARATOR, dtype=np.uint8)
        tokens, boundaries = z3_extract.read_package(synth)
        titles = z3_extract.extract_titles(tokens, boundaries)
        n_checked = 0
        for i in range(len(boundaries) - 1):
            length = int(off[i + 1] - off[i])
            if not length:
                continue
            body = pre[off[i] : off[i + 1] - len(sep)]
            title = titles[i]
            if title is None:
                continue
            # every kept chunk must be a literal slice of the stream BEFORE the
            # article start: search the body for the title and verify the
            # surrounding tokens exist earlier in the stream
            start = int(boundaries[i])
            assert z3_extract.G.find_first(body, list(G.LINK_OPEN) + [int(x) for x in title],
                                          0, len(body)) >= 0, i
            assert start > 0
            n_checked += 1
        return "%d prefixes contain the title behind the link token" % n_checked

    check("every prefix really quotes its own title", causality_full)

    print("\n6  the loader arithmetic (numpy model of the torch code)")

    def loader_arithmetic():
        tokens, boundaries = z3_extract.read_package(synth)
        pre = np.fromfile(os.path.join(out, z3_extract.FILES["prefix_tokens"]), dtype=np.uint8)
        off = np.fromfile(os.path.join(out, z3_extract.FILES["prefix_offsets"]), dtype=np.int32).astype(np.int64)
        supervised_plain = 0
        supervised_z3 = 0
        positions_plain = 0
        positions_z3 = 0
        for i in range(len(boundaries) - 1):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            body = end - start - 1
            if body <= 0:
                continue
            q = int(off[i + 1] - off[i])
            document = np.concatenate([pre[off[i]:off[i + 1]], tokens[start:end]])
            document_in = document[:-1]
            document_out = document[1:]
            assert len(document_in) == q + body == len(document_out), (i, q, body)
            assert bool((document_in[q:] == tokens[start:end - 1]).all())
            assert bool((document_out[q:] == tokens[start + 1:end]).all())
            supervised_plain += body
            supervised_z3 += (q + body) - q
            positions_plain += body
            positions_z3 += q + body
        assert supervised_plain == supervised_z3, "Z3 must not change the loss count"
        return "supervised identical (%d), positions +%.2f %%" % (
            supervised_plain, 100.0 * (positions_z3 - positions_plain) / positions_plain)

    check("Z3 keeps the supervised token count", loader_arithmetic)

    def z1_document_packing():
        tokens, boundaries = z3_extract.read_package(synth)
        start, end = int(boundaries[5]), int(boundaries[6])
        body = np.asarray(tokens[start:end - 1], dtype=np.int64)
        ordinal = np.full(len(body), 5, dtype=np.int64)
        stream = np.arange(start, end - 1, dtype=np.int64)
        packed = zfeat.pack(body, ordinal, stream)
        t, a, p = zfeat.unpack(packed)
        assert (t == body).all() and (a == 5).all() and (p == stream).all()
        return "%d positions round trip" % len(body)

    check("Z1 packing of a whole document", z1_document_packing)

    print("\n7  the torch modules (syntax and import surface only)")

    def stub_import():
        sys.path.insert(0, os.path.join(ROOT, "tft"))
        from tft import stub_imports  # noqa: F401

        import importlib

        importlib.import_module("pysrc.data")
        return "pysrc.data imports under the stub"

    tree = os.environ.get("TFT_TREE")
    if tree and os.path.isdir(os.path.join(tree, "pysrc")):
        sys.path.insert(0, tree)
        check("pysrc imports under the stub", stub_import)
    else:
        print("  skip  pysrc import                              "
              "(set TFT_TREE to external/fx2-cmix-transformer)")

    print("\n8  the evaluation")

    def eval_math():
        from zctx import eval_zctx

        ref = os.path.join(work, "ref.jsonl")
        arm = os.path.join(work, "arm.jsonl")
        with open(ref, "w") as f:
            for e in range(1, 11):
                f.write(json.dumps({"epoch": float(e), "train/loss": 1.5 - 0.02 * e,
                                    "train/epoch_loss": 1.5 - 0.02 * e}) + "\n")
        with open(arm, "w") as f:
            for e in range(1, 11):
                f.write(json.dumps({"epoch": float(e), "train/loss": 1.5 - 0.021 * e,
                                    "train/epoch_loss": 1.5 - 0.021 * e}) + "\n")
        result = eval_zctx.compare(ref, arm, delta_parameters=12288, sigma=0.002)
        assert result["delta_final"] < 0
        assert result["S_@41KB"] < 0, result
        assert result["weight_cost_in_S"] > 0
        return "d=%s  dS@41KB=%s  verdict=%s" % (
            result["delta_final"], result["S_@41KB"], result["verdict"])

    check("eval turns nats into bytes with the right sign", eval_math)

    if real:
        print("\n9  the real stream slice " + real)
        check("zgrammar.verify on real data",
              lambda: "%d articles, %d titles without terminator"
                      % (G.verify(real)["n_articles"], G.verify(real)["titles_without_terminator"]))
        realout = os.path.join(work, "z3real")
        m = z3_extract.build(real, realout, dict(z3_extract.DEFAULTS), log=lambda *a: None)
        check("z3 build on real data",
              lambda: "hit rate %.1f %%, +%.2f %% tokens"
                      % (m["statistics"]["hit_rate_percent"],
                         m["statistics"]["prefix_tokens_percent_of_stream"]))
        check("z3 check on real data",
              lambda: "longest prefix %d" % z3_extract.check(real, realout)["longest_prefix"])
    else:
        print("\n9  real stream slice: not given (pass a lite package directory)")

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, err in FAIL:
        print("  FAILED  " + name + "  " + err)
    shutil.rmtree(work, ignore_errors=True)
    return 1 if FAIL else 0


def _assert(condition, message):
    assert condition, message
    return True


if __name__ == "__main__":
    sys.exit(main())
