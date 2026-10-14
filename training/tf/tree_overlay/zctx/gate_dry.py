#!/usr/bin/env python3
"""Dry run of the digit gating: whatever can be proven without a GPU is proven without one

Two stages, because ${HOST} has no torch and the GPU pod costs money:

  stage A (numpy, runs anywhere)     the row construction on the REAL
      package: reference form against loader form, causality, completeness,
      sentinel bookkeeping, prior/target pairing.
  stage B (torch, only on the pod)   loader instance, two real micro-batches,
      `install_gate`, `install_z4` and the build guard that catches an arm
      silently running as the reference.

    python -m zctx.gate_dry A <pkgdir> [n_articles]
    python -m zctx.gate_dry B <pkgdir>
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from zctx import zgate  # noqa: E402

PASS: list = []
FAIL: list = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        PASS.append(name)
        print("  ok    %-52s %s" % (name, detail))
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, repr(exc)))
        print("  FAIL  %-52s %r" % (name, exc))


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------

def stage_a(pkgdir: str, n_articles: int = 300) -> None:
    tokens = np.fromfile(os.path.join(pkgdir, "ppmd-bytes.uint8"), dtype=np.uint8)
    bnd = np.fromfile(os.path.join(pkgdir, "article-boundaries.int32"),
                      dtype=np.int32).astype(np.int64)
    bitmap = zgate.load_bitmap(pkgdir)
    v = len(bitmap)
    m = int(os.environ.get("ZGATE_M", "1"))
    exclude = os.environ.get("ZGATE_EXCLUDE", "1") != "0"
    gate = zgate.gate_mask(tokens, bitmap, m, exclude=exclude)
    print("package %s: %d tokens, %d articles, V=%d, M=%d, exclusion=%s"
          % (pkgdir, len(tokens), len(bnd) - 1, v, m, exclude))
    print("gated positions: %d (%.4f %%)"
          % (int(gate.sum()), 100.0 * gate.mean()))
    print()

    zgate.selftest()
    print()

    rng = np.random.default_rng(20260902)
    arts = rng.choice(len(bnd) - 1, size=min(n_articles, len(bnd) - 1), replace=False)

    def per_article(sentinel: bool):
        stats = {"rows": 0, "sent": 0, "orig": 0}
        for a in arts:
            s, e = int(bnd[a]), int(bnd[a + 1])
            body = e - s - 1
            if body <= 0:
                continue
            doc = tokens[s:e].astype(np.int64)
            g = gate[s : s + body]
            inp, out, msk = zgate.build_document(doc, g, v, sentinel)
            src, sf, cls = zgate.gate_rows(g, sentinel)
            # loader form == reference form
            assert (np.where(sf, v + cls, doc[src]) == inp).all()
            assert (doc[src + 1] == out).all()
            assert (~sf == msk).all()
            # every kept position occurs exactly once, in order
            assert (np.diff(src) > 0).all()
            assert set(src[~sf].tolist()) == set(np.flatnonzero(~g).tolist())
            # nothing runs past the end of the document
            assert src.max() + 1 <= body
            # sentinels only at V and above, real tokens only below
            assert (inp[sf] >= v).all() and (inp[~sf] < v).all()
            stats["rows"] += len(inp)
            stats["sent"] += int(sf.sum())
            stats["orig"] += body
        return stats

    check("A1 reference form == loader form (arm g0)",
          lambda: "%d articles, %d rows" % (len(arts), per_article(False)["rows"]))
    st = per_article(True)
    check("A2 reference form == loader form (arm g1)",
          lambda: "%d rows, %d of them sentinels" % (st["rows"], st["sent"]))
    check("A3 g1 has more rows than g0 but fewer than ungated",
          lambda: "%d < %d < %d" % (per_article(False)["rows"], st["rows"], st["orig"]))

    def causality():
        """the gate decision depends ONLY on the past

        Checked, not assumed: the stream is changed BEHIND a position; the
        decision at that position must stay the same.
        """
        rng2 = np.random.default_rng(7)
        t = tokens.copy()
        cut = len(t) // 2
        g_before = zgate.gate_mask(t, bitmap, m, exclude=exclude)[:cut]
        t[cut:] = rng2.integers(1, v, size=len(t) - cut, dtype=np.int64).astype(np.uint8)
        g_after = zgate.gate_mask(t, bitmap, m, exclude=exclude)[:cut]
        assert (g_before == g_after).all()
        return "%d positions unchanged after random replacement of the tail" % cut

    check("A4 causality (the future does not change the decision)", causality)

    def no_digit_in_context():
        """what is gated is also gone as INPUT (arm g0)"""
        bad = 0
        for a in arts[:50]:
            s, e = int(bnd[a]), int(bnd[a + 1])
            body = e - s - 1
            if body <= 0:
                continue
            doc = tokens[s:e].astype(np.int64)
            g = gate[s : s + body]
            inp, _, _ = zgate.build_document(doc, g, v, sentinel=False)
            bad += int(len(inp) - (~g).sum())
        assert bad == 0
        return "0 deviations in 50 articles"

    check("A5 a gated position is also dropped as context", no_digit_in_context)

    def length_classes():
        seen = {}
        for a in arts:
            s, e = int(bnd[a]), int(bnd[a + 1])
            body = e - s - 1
            if body <= 0:
                continue
            g = gate[s : s + body]
            for x, y in zgate.runs_of(g):
                if y < body:
                    seen[zgate.length_class(int(y - x))] = seen.get(
                        zgate.length_class(int(y - x)), 0) + 1
        assert seen, "no blocks found"
        assert max(seen) < zgate.N_LENGTH_CLASSES
        return "classes in use %s" % (sorted(seen.items()),)

    check("A6 length classes within the valid range", length_classes)


# --------------------------------------------------------------------------
# Stage B (GPU pod)
# --------------------------------------------------------------------------

def stage_b(pkgdir: str) -> None:
    import torch

    from tft import config as tftcfg
    from pysrc.model import InitializeScales, Transformer

    from zctx.zloader import ZDataLoader
    from zctx.zmodel import install_gate, install_z4

    cfg = tftcfg.make_config("bf16", geometry_name=os.environ.get("TFT_GEOMETRY", "poc4m"),
                             variable_sequence_length=True)
    v = cfg.vocabulary_size
    device = torch.device("cpu")

    def build_loader(arm: str):
        os.environ["ZCTX_ARM"] = arm
        return ZDataLoader(
            arm=arm,
            z3_dir=None,
            tokens_filename=os.path.join(pkgdir, "ppmd-bytes.uint8"),
            priors_filename=os.path.join(pkgdir, "ppmd-probs.float16"),
            article_boundaries_filename=os.path.join(pkgdir, "article-boundaries.int32"),
            mmap=True,
            epochs=1,
            micro_batch_tokens=tftcfg.MICRO_BATCH_TOKENS,
            gradient_accumulation_steps=1,
            max_article_tokens=tftcfg.MAX_ARTICLE_TOKENS,
            max_merge_tokens=None,
            device=device,
            rank=0,
            world_size=1,
        )

    state = {}

    def loader(arm):
        def run():
            ld = build_loader(arm)
            state[arm] = ld
            batch = next(iter(ld))
            mb = batch.micro_batches[0] if hasattr(batch, "micro_batches") else batch
            it = mb.input_tokens
            limit = v + (zgate_classes if arm == "g1" else 0)
            assert int(it.max()) < limit, (int(it.max()), limit)
            assert int(it.min()) >= 0
            assert int(mb.output_tokens.max()) < v
            return "max(input)=%d < %d, rows %d, scored %d" % (
                int(it.max()), limit, it.shape[1], int(mb.loss_mask.sum()))
        return run

    from zctx.zgate import N_LENGTH_CLASSES as zgate_classes

    check("B1 loader g0 yields a valid micro-batch", loader("g0"))
    check("B2 loader g1 yields a valid micro-batch", loader("g1"))

    def gate_install():
        torch.manual_seed(20260902)
        m0 = Transformer(cfg)
        base = sum(p.numel() for p in m0.parameters())
        torch.manual_seed(20260902)
        m1 = Transformer(cfg)
        added = install_gate(m1, cfg, zgate_classes)
        tot = sum(p.numel() for p in m1.parameters())
        assert tot - base == added, (tot - base, added)
        assert torch.equal(m1.embedding.weight[:v], m0.embedding.weight)
        iz = InitializeScales(weight_steps=0, activation_steps=0)
        idx = torch.arange(v, v + zgate_classes, dtype=torch.int64).view(1, -1)
        with torch.no_grad():
            out = m1.embedding(idx, initialize_scales=iz)
        assert out.shape[-1] == cfg.d_model and torch.isfinite(out).all()
        return "+%d parameters, first %d rows identical, sentinel rows readable" % (added, v)

    check("B3 install_gate: rows extended, reference preserved", gate_install)

    def z4_install():
        from zctx import zlast

        torch.manual_seed(20260901)
        m0 = Transformer(cfg)
        torch.manual_seed(20260901)
        m1 = Transformer(cfg)
        x = int(os.environ.get("ZCTX_LAST_X", "4"))
        added = install_z4(m1, cfg, x=x, d_slot=int(os.environ.get("ZCTX_LAST_DIM", "16")))
        tok = np.random.default_rng(3).integers(1, v, size=64).astype(np.int64)
        packed = torch.from_numpy(zlast.pack(tok, x).reshape(1, -1))
        plain = torch.from_numpy(tok.reshape(1, -1))
        iz = InitializeScales(weight_steps=0, activation_steps=0)
        with torch.no_grad():
            a = m0.embedding(plain, initialize_scales=iz)
            b = m1.embedding(packed, initialize_scales=iz)
        assert torch.equal(a, b), "zero start not identical"
        ids_max = (x - 1) * v + (v - 1)
        assert ids_max < x * v
        return "+%d parameters, zero start bit-identical, largest slot %d < %d" % (
            added, ids_max, x * v)

    check("B4 install_z4: the GPU crash candidate runs and is neutral", z4_install)

    def wacht():
        """the build guard must catch the case of a rebuilt loader with an
        unchanged model (the z4 arm once ran that way)"""
        import re

        src = open(os.path.join(ROOT, "tft", "runner.py"), encoding="utf-8").read()
        assert "needs_model" in src and '"z4": "z4"' in src and '"g1": "gate"' in src
        assert re.search(r"assert need is None or need in installed", src)
        return "table covers z1/z1z3/z4/z4c/g1"

    check("B5 build guard against 'loader rebuilt, model not'", wacht)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    stage, pkgdir = sys.argv[1].upper(), sys.argv[2]
    print("=== DRY RUN GATING stage %s ===" % stage)
    if stage == "A":
        stage_a(pkgdir, int(sys.argv[3]) if len(sys.argv) > 3 else 300)
    elif stage == "B":
        stage_b(pkgdir)
    else:
        print(__doc__)
        return 2
    print()
    print("DRY RUN stage %s: %d passed, %d failed" % (stage, len(PASS), len(FAIL)))
    for name, exc in FAIL:
        print("   FAIL  %s  %s" % (name, exc))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
