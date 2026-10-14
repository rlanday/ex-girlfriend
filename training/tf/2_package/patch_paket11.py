#!/usr/bin/env python3
r"""Converts the training side to stage 2

    python3 patch_paket11.py <tree_train> [--pruefe]

The changes, each with its reason:

 1. `zctx/zpdump.py` new: the loader that takes its rows from the coder dump
    (arm `pkd`).  It does not touch `zploader.py`: the accepted stage-1 loader
    stays as the reference.
 2. `zctx/zpaket.py`, X0ROW: the edge mark of the X2 channel gets a row of
    its own at the top end (`V + NSENT`) instead of sharing row 0 with 391,160
    real occurrences of token index 0.  Option `PK_LAST_X0ROW`, default 0 ->
    stage-1 form unchanged.
 3. `zctx/zpmodel.py`: the predecessor table grows to `X * (V+NSENT+1)` rows;
    the token table stays at `V+NSENT` (the mark is never a token).
 4. `tft/config.py`: micro batch 2^18 (so the shortening rule of the loader no
    longer applies), and the optimizer batch stays at 2^20 by computing the
    gradient accumulation from both numbers instead of from a fixed reference
    GPU count.  In addition the position limit (rotary table) is separated
    from the piece limit: the piece limit stays 2^17 (a constant held in three
    places), the position limit becomes a value of its own.
 5. `tft/runner.py`: arm `pkd`, the warm start and resume path, and the build
    guard line for the new arm.
 6. `tft/export_weights_poc.py`: the manifest fields `paket`, `arme`,
    `prior0_sha`, `fenster`, `redirmerge`, `x0row`.  Without them a
    transformer trained without alias rows could be loaded by a coder that
    feeds alias rows, and nothing would fire.
 7. `tft/param_count.py`: geometry rows `poc4m_w1280` and further windows
    (otherwise row for row poc4m).  The window changes no tensor shape; one
    checkpoint warms every window (`window_size` enters only the attention
    mask, `pysrc/attention_function.py:164-171`).

Every change is idempotent: a second run finds its mark and leaves the file
alone.

Two pitfalls of this kind of patcher:
 (a) The mark must not be the first line of the replacement text; otherwise
     every change that contains its own anchor is skipped silently.
 (b) Whoever changes a replacement text must change its mark as well;
     otherwise an already patched tree takes the old version for the new one
     and the change silently does not arrive.
(German names: ersetze = replace, Marke = mark, Anker = anchor, alt/neu =
old/new, baum = tree.)
"""

from __future__ import annotations

import os
import shutil
import sys

MARKE = "NEIF-PAKET11"


def ersetze(pfad: str, alt: str, neu: str, name: str, marke: str = None) -> bool:
    """`marke` is a string that exists only after this change.

    Guessing the mark from the first line of the replacement text would skip
    every change that contains its own anchor, so each call names its mark.
    """
    s = open(pfad, encoding="utf-8").read()
    if marke is None:
        marke = neu
    if marke in s:
        print("  = %s already set" % name)
        return False
    assert s.count(alt) == 1, (
        "%s: anchor not unique (%d matches); the file has changed"
        % (name, s.count(alt)))
    open(pfad, "w", encoding="utf-8").write(s.replace(alt, neu))
    print("  + %s" % name)
    return True


def main() -> int:
    baum = sys.argv[1]
    hier = os.path.dirname(os.path.abspath(__file__))
    assert os.path.isdir(os.path.join(baum, "tft")), "not a tree_train: %s" % baum

    # ---- 1. the new loader ----------------------------------------------
    for name in ("dumpv2.py", "zpdump.py"):
        shutil.copyfile(os.path.join(hier, name),
                        os.path.join(baum, "zctx", name))
        print("  + zctx/%s" % name)

    # ---- 2. zpaket.py: X0ROW --------------------------------------------
    p = os.path.join(baum, "zctx", "zpaket.py")
    ersetze(p,
            'DEFAULT_X = 2\nNO_PREDECESSOR = 0                      '
            '# im 208er-Paket unbelegter Tokenindex',
            '''DEFAULT_X = 2
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
    return vocabulary_size + N_SENTINEL + (1 if X0ROW else 0)''',
            "zpaket.X0ROW", "def edge_marker(")
    ersetze(p,
            '        prev = np.full(n, NO_PREDECESSOR, dtype=np.int64)',
            '        # the edge mark, not the fixed 0\n'
            '        prev = np.full(n, edge_marker(_V_FOR_SENTINEL["v"]),\n'
            '                       dtype=np.int64)',
            "zpaket.pack edge mark", "prev = np.full(n, edge_marker(")
    # The self test checks the edge mark against the constant; with X0ROW it is
    # 216, not 0.  Without this edit `python -m zctx.zpaket` stops with an
    # AssertionError, and only on the rented host, shortly before the run.
    ersetze(p,
            "    assert (prev[0][1:] == t[:-1]).all() and prev[0][0] == NO_PREDECESSOR\n"
            "    assert (prev[1][2:] == t[:-2]).all() and (prev[1][:2] == NO_PREDECESSOR).all()",
            '    _rand = edge_marker(_V_FOR_SENTINEL["v"])   # 0 or V+NSENT\n'
            '    assert (prev[0][1:] == t[:-1]).all() and prev[0][0] == _rand\n'
            '    assert (prev[1][2:] == t[:-2]).all() and (prev[1][:2] == _rand).all()',
            "zpaket.selftest edge mark", "_rand = edge_marker(")
    s = open(p, encoding="utf-8").read()
    if "\nimport os\n" not in s:
        s = s.replace("import numpy as np", "import os\n\nimport numpy as np", 1)
        open(p, "w", encoding="utf-8").write(s)
        print("  + zpaket import os")

    # ---- 3. zpmodel.py: the predecessor table grows ---------------------
    p = os.path.join(baum, "zctx", "zpmodel.py")
    ersetze(p,
            '        self.v_plus = vocabulary_size + n_sentinel',
            '        self.v_plus = vocabulary_size + n_sentinel\n'
            '        # The predecessor table carries one more row per slot when the edge\n'
            '        # mark has a row of its own (X0ROW).  The token table stays at v_plus;\n'
            '        # the mark is never a token.\n'
            '        self.v_last = zpaket.last_rows(vocabulary_size)',
            "zpmodel.v_last", "self.v_last = zpaket.last_rows(")
    ersetze(p,
            '                self.x * self.v_plus, self.d_slot, init, quantization, cfg',
            '                self.x * self.v_last, self.d_slot, init, quantization, cfg',
            "zpmodel last_table rows", "self.x * self.v_last, self.d_slot")
    ersetze(p,
            '                torch.arange(self.x, dtype=torch.int64) * self.v_plus,',
            '                torch.arange(self.x, dtype=torch.int64) * self.v_last,',
            "zpmodel last_offsets", "dtype=torch.int64) * self.v_last")

    # ---- 4. config.py: micro batch and position limit --------------------
    p = os.path.join(baum, "tft", "config.py")
    ersetze(p,
            'MICRO_BATCH_TOKENS = (1 << 20) // 8\nREFERENCE_WORLD_SIZE = 8',
            '''# The micro batch is 2^18.  Every document then fits, and the shortening
# rule of the loader (drop A2, then the sentinels, then the prompt) no longer
# applies; the coder could not reproduce it causally (the total length is not
# known yet when the prompt is fed), and on pkgvoll_v2 it affected 14 of
# 243,426 articles.  The optimizer batch stays at his 2^20: the gradient
# accumulation is computed from both numbers instead of from a fixed
# reference GPU count (with 2^18 and 1 GPU that is 4 steps).
MICRO_BATCH_TOKENS = int(os.environ.get("TFT_MICRO_BATCH_TOKENS", 1 << 18))
OPTIMIZER_BATCH_TOKENS = int(os.environ.get("TFT_OPTIMIZER_BATCH_TOKENS", 1 << 20))
REFERENCE_WORLD_SIZE = 8

# The position limit (rotary table) is not the same number as the piece limit.
# The piece limit 2^17 is held in three places (here, `pysrc/data.py`,
# `predictor.cpp:kMaxArticleTokens`) and counts coded tokens in the coder; the
# number of positions of a document is coded plus inserted rows and lies above
# it.  Enlarging the micro batch without separating the two creates documents
# whose positions exceed the rotary table.  Default: the piece limit.
POSITION_LIMIT = int(os.environ.get("TFT_POSITION_LIMIT", MAX_ARTICLE_TOKENS))''',
            "config micro batch 2^18 + position limit", "POSITION_LIMIT = int(")
    ersetze(p,
            '''    assert REFERENCE_WORLD_SIZE % world_size == 0, (
        "use 1, 2, 4 or 8 GPUs so the optimizer batch stays at 2**20 tokens"
    )
    return REFERENCE_WORLD_SIZE // world_size''',
            '''    # computed from micro batch and optimizer batch, so that both can be set
    # independently
    gesamt = MICRO_BATCH_TOKENS * world_size
    assert OPTIMIZER_BATCH_TOKENS % gesamt == 0, (
        "optimizer batch %d is not a multiple of micro batch %d x %d GPUs"
        % (OPTIMIZER_BATCH_TOKENS, MICRO_BATCH_TOKENS, world_size)
    )
    return OPTIMIZER_BATCH_TOKENS // gesamt''',
            "config gradient_accumulation_steps", "return OPTIMIZER_BATCH_TOKENS // gesamt")
    ersetze(p,
            '        max_variable_sequence_length=MAX_ARTICLE_TOKENS\n'
            '        if variable_sequence_length\n'
            '        else None,',
            '        max_variable_sequence_length=POSITION_LIMIT\n'
            '        if variable_sequence_length\n'
            '        else None,',
            "config max_variable_sequence_length", "max_variable_sequence_length=POSITION_LIMIT")

    # ---- 5. runner.py: arm pkd + warm start ------------------------------
    p = os.path.join(baum, "tft", "runner.py")
    ersetze(p,
            '''    arm = os.environ.get("ZCTX_ARM", "none")
    if arm == "pk":''',
            '''    arm = os.environ.get("ZCTX_ARM", "none")
    # arm `pkd` reads its rows from the coder's dump instead of rebuilding them
    # from pk tables (the coder is the specification).  There,
    # `article_boundaries_filename` holds the document boundaries of the dump.
    if arm == "pkd":
        from zctx.zpdump import PaketDumpDataLoader

        dump_dir = os.environ["PK_DUMP_DIR"]
        arguments["article_boundaries_filename"] = os.path.join(
            dump_dir, "article-boundaries.int32")
        # The coder has already made the 2^17 piece cut; the loader must not
        # cut a second time.  It counts in stream positions, and a document
        # can span more than 2^17 of them when the cut falls on a gated
        # position (measured: 131,086 instead of 131,072).  A second cut would
        # create documents that do not exist in the dump.
        arguments["max_article_tokens"] = tftcfg.MICRO_BATCH_TOKENS
        return PaketDumpDataLoader(dump_dir=dump_dir, **arguments)

    if arm == "pk":''',
            "runner arm pkd", 'arguments["max_article_tokens"] = tftcfg.MICRO_BATCH_TOKENS')
    ersetze(p,
            '''    needs_model = {"z1": "z1", "z1z3": "z1", "z4": "z4", "z4c": "z4", "g1": "gate",
                   "m1": "match", "pk": "paket"}''',
            '''    needs_model = {"z1": "z1", "z1z3": "z1", "z4": "z4", "z4c": "z4", "g1": "gate",
                   "m1": "match", "pk": "paket", "pkd": "paket"}''',
            "runner needs_model pkd", "\"pkd\": \"paket\"")
    ersetze(p,
            '''    arm_env = os.environ.get("ZCTX_ARM", "none")
    needs_model''',
            '''    arm_env = os.environ.get("ZCTX_ARM", "none")
    if arm_env == "pkd":
        arm_env = "pk"          # NEIF-PAKET11: the same model change as `pk`
    needs_model''',
            "runner arm_env pkd", 'arm_env = "pk"          # NEIF-PAKET11')
    ersetze(p,
            '''    load_from = {"bf16": None, "qat": tftcfg.model_path("bf16", g.name),
                 "fp32": tftcfg.model_path("qat", g.name)}[phase]''',
            '''    # Warm start: a new training (new package, new set of documents) begins
    # from the weights of stage 1 instead of random numbers.  The epoch index
    # starts at 0; the weights are taken over shape-aware (X0ROW grows
    # embedding.last_table, and naively appending rows would shift slot 1;
    # see paket11_warmstart.py).
    load_from = {"bf16": None, "qat": tftcfg.model_path("bf16", g.name),
                 "fp32": tftcfg.model_path("qat", g.name)}[phase]
    warm = os.environ.get("TFT_WARMSTART")
    if warm and phase == "bf16":
        from zctx import zpaket
        import paket11_warmstart as p11w

        p11w.gewichte_laden(
            raw_model, warm,
            x=int(os.environ.get("PK_LAST_X", str(zpaket.DEFAULT_X))),
            v_plus=cfg.vocabulary_size + zpaket.N_SENTINEL,
            x0row=zpaket.X0ROW, device=device, laut=main_process)
        load_from = None''',
            "runner warm start", "p11w.gewichte_laden(")
    ersetze(p,
            '''    if os.environ.get("KINJ_OPTSTATE", "0") != "0" and main_process:
        import pysrc.train as _pt

        _mk = _pt.make_optimizers

        def _pk_make_optimizers(_model, _cfg):
            opts = _mk(_model, _cfg)
            _CHECKPOINT["optimizers"] = opts
            print("context package: optimizer state is saved as well", flush=True)
            return opts

        _pt.make_optimizers = _pk_make_optimizers''',
            '''    if os.environ.get("KINJ_OPTSTATE", "0") != "0" and main_process:
        import pysrc.train as _pt

        _mk = _pt.make_optimizers

        def _pk_make_optimizers(_model, _cfg):
            opts = _mk(_model, _cfg)
            _CHECKPOINT["optimizers"] = opts
            print("context package: optimizer state is saved as well", flush=True)
            # take over the Muon/AdamW moments of stage 1.  A parameter whose
            # shape has changed (X0ROW) gets no old state; its moment would
            # refer to the old row layout.
            wopt = os.environ.get("TFT_WARMSTART_OPT")
            if wopt:
                import paket11_warmstart as p11w

                p11w.optzustand_laden(opts, wopt, laut=True)
            return opts

        _pt.make_optimizers = _pk_make_optimizers''',
            "runner warm start optimizer state", "p11w.optzustand_laden(")

    # ---- 6. export_weights_poc.py: manifest fields -----------------------
    p = os.path.join(baum, "tft", "export_weights_poc.py")
    if os.path.exists(p):
        # `paket_from_env()` only recognises `ZCTX_ARM=pk`.  The
        # dump mode is called `pkd` and has the SAME model form; without this
        # edit the export builds a plain model, and `load_state_dict` stops at
        # the V+8 embedding row (found in the smoke run).
        ersetze(p,
                '    if os.environ.get("ZCTX_ARM", "none") != "pk":',
                '    if os.environ.get("ZCTX_ARM", "none") not in ("pk", "pkd"):',
                "export paket_from_env knows pkd",
                'not in ("pk", "pkd")')
        # The shape check of the PREDECESSOR table computes
        # `X * (V + NSENT)`.  With X0ROW it is `X * (V + NSENT + 1)`, 434
        # instead of 432 rows.  `zpaket.last_rows()` is the ONE place that
        # holds the number.
        ersetze(
            p,
            '            assert by_name["embedding.last_table.weight.q"].shape == (\n'
            '                X * want_rows, d_slot), by_name["embedding.last_table.weight.q"].shape',
            '            from zctx import zpaket as _zpl\n'
            '            _last_rows = _zpl.last_rows(cfg.vocabulary_size)\n'
            '            assert by_name["embedding.last_table.weight.q"].shape == (\n'
            '                X * _last_rows, d_slot), (\n'
            '                by_name["embedding.last_table.weight.q"].shape,\n'
            '                "expected %d rows per slot (X0ROW=%s)"\n'
            '                % (_last_rows, _zpl.X0ROW))',
            "export last_table shape check with X0ROW",
            "_last_rows = _zpl.last_rows(")
        ersetze(p,
                '    bitmap_path = tftcfg.data_paths()["vocab_bitmap_filename"]\n'
                '    payload = dict(result)',
                '''    bitmap_path = tftcfg.data_paths()["vocab_bitmap_filename"]
    payload = dict(result)
    # These fields tie the weights to their training form (package, arms,
    # prior0, window, redirect merge, X0ROW); the measurement builds of the
    # development tree check them when they load a container.
    payload["paket"] = payload.get("paket") or _p11_paket()
    payload["arme"] = _p11_arme()
    payload["prior0_sha"] = _p11_prior0_sha()
    payload["fenster"] = int(getattr(_p11_geom(), "window_size", 0))
    payload["redirmerge"] = _p11_meta("redirmerge",
                                      os.environ.get("PK_REDIRMERGE", "0") != "0")
    payload["x0row"] = _p11_meta("x0row",
                                 os.environ.get("PK_LAST_X0ROW", "0") != "0")''',
                "export manifest fields", "_p11_prior0_sha()")
        s = open(p, encoding="utf-8").read()
        if "_p11_arme" not in s.split("def write_sidecar")[0]:
            zusatz = '''

# --- the manifest fields -------------------------------------------------

def _p11_geom():
    from tft import config as _c
    return _c.geometry(os.environ.get("TFT_GEOMETRY"))


def _p11_meta(feld, fallback):
    """build state from `dump-meta.json` of the package.

    The environment is the wrong source: in dump mode `PK_REDIRMERGE` controls
    nothing (the feature is in the dump), and the manifest has to state what the
    package carries.
    """
    d = os.environ.get("PK_DUMP_DIR")
    if d:
        import json as _j
        try:
            with open(os.path.join(d, "dump-meta.json"), encoding="utf-8") as f:
                return _j.load(f).get(feld, fallback)
        except OSError:
            pass
    return fallback


def _p11_paket() -> str:
    return os.environ.get("PK_PAKET_NAME") or os.path.basename(
        (os.environ.get("PK_DUMP_DIR") or os.environ.get("PK_DIR") or "").rstrip("/"))


def _p11_arme() -> list:
    """the arm list of the package, the training form recorded in the manifest.

    It must state what the training really fed: an arm added to the package
    without an entry here reopens the gap this field exists to close.
    """
    dumpdir = os.environ.get("PK_DUMP_DIR")
    if dumpdir:
        import json as _j
        with open(os.path.join(dumpdir, "dump-meta.json"), encoding="utf-8") as f:
            return list(_j.load(f).get("arme", []))
    an = lambda n, d="1": os.environ.get(n, d) != "0"
    arme = []
    for name, var, d in (("paket", "PK_PAKET", "1"), ("match", "PK_MATCH", "1"),
                         ("z3", "PK_Z3", "1"), ("num", "PK_NUM", "0"),
                         ("z1t", "PK_Z1T", "1"), ("syn", "PK_SYN", "1"),
                         ("alias", "PK_ALIAS", "0"), ("a2", "PK_A2", "1"),
                         ("a2n", "PK_A2N", "1"), ("sent", "PK_SENT", "1"),
                         ("sent_feed", "PK_SENT_FEED", "1"),
                         ("titelmask", "PK_TITLEMASK", "1"),
                         ("x0row", "PK_LAST_X0ROW", "0")):
        if an(var, d):
            arme.append(name)
    return arme


def _p11_prior0_sha():
    """sha256 of the prior0 table of the package"""
    import hashlib
    weg = os.environ.get("PK_PRIOR0")
    if not weg:
        d = os.environ.get("PK_DUMP_DIR") or os.environ.get("PK_DIR")
        weg = os.path.join(d, "z3-prior0.float16") if d else None
    if not weg or not os.path.exists(weg):
        return None
    h = hashlib.sha256()
    with open(weg, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

'''
            marke = "\ndef write_sidecar"
            assert s.count(marke) == 1
            s = s.replace(marke, zusatz + "\ndef write_sidecar")
            open(p, "w", encoding="utf-8").write(s)
            print("  + export manifest helper functions")
    else:
        print("  ! tft/export_weights_poc.py missing in this tree")

    # ---- 7. geometry rows poc4m_w1280 and further windows ---------------
    p = os.path.join(baum, "tft", "param_count.py")
    s = open(p, encoding="utf-8").read()
    if 'name="poc4m_w8192"' not in s:
        anker = "\n# --- the depth ladder"
        assert s.count(anker) == 1, "anchor of the geometry list not unique"
        neu = "".join('''
_add(
    Geometry(
        name="poc4m_w%d",
        vocabulary_size=208,
        d_model=192,
        n_layers=8,
        d_head=64,
        n_query_heads=3,
        n_key_value_heads=3,
        d_mlp=768,
        kimi_linear=[True, True, True, False] * 2,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        window_size=%d,
        note="poc4m with window %d; only window_size differs, no tensor "
             "shape depends on it, and one checkpoint serves every window "
             "(pysrc/attention_function.py:164-171).",
    )
)
''' % (w, w, w) for w in (1280, 1536, 2048, 3072, 4096, 6144, 8192))
        open(p, "w", encoding="utf-8").write(s.replace(anker, neu + anker))
        print("  + param_count poc4m_w rows 1280..8192")
    else:
        print("  = param_count poc4m_w rows already present")

    # ---- 8. warm start module onto the search path -----------------------
    shutil.copyfile(os.path.join(hier, "warmstart.py"),
                    os.path.join(baum, "paket11_warmstart.py"))
    print("  + paket11_warmstart.py")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
