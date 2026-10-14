#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""The trainer loader against the RAW coder dump, token by token, on 20 documents

    TRAIN_TREE=<training tree> [PYLIB=<python packages>] \
        loader_vs_dump.py <package dir> <raw dump paket-zeilen.bin> <token_start> <skip>

Checked per row: tok, p1, p2, mb, mc (the loader input, unpacked through
dumpv2.felder) and src (coder src - token_start), and that the packed loader
input is built from exactly these fields.  Per scored row: the target token
equals the token of the stream (ppmd-bytes of the package, computed from the
stream, not from the dump), and every target except the closing token of a
document is a stream row of the coder with the same token.  The priors file
must exist with the right length; its values are not read.  Prints
"PASSED" when every count except the row totals is 0.  Keys of the printed
counts (German): zeilen = rows, bewertet = scored, eingabe = packed input,
ziel_tok_gegen_bett / _koder = target token against the stream / the coder,
schlussziele_ohne_dumpzeile = closing targets without a dump row.
"""
import os, sys
if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])
import importlib.abc, importlib.machinery, types
class _A(types.ModuleType):
    __path__ = []
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return type(n, (), {"__init__": lambda *a, **k: None, "__call__": lambda *a, **k: None})
class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    W = ("fla", "flash_attn", "causal_conv1d", "mamba_ssm")
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.W: return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec): return _A(spec.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, _F())
sys.path.insert(0, os.environ["TRAIN_TREE"])
for k, v in (("ZCTX_ARM", "pkd"), ("PK_LAST_X0ROW", "1"), ("PK_REDIRMERGE", "1"), ("PK_SLOT_DIM", "16"),
             ("TFT_MICRO_BATCH_TOKENS", "262144"), ("TFT_OPTIMIZER_BATCH_TOKENS", "1048576")):
    os.environ.setdefault(k, v)
PKG, ROH, TS, SKIP = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.environ["PK_DUMP_DIR"] = PKG
import numpy as np, torch
from tft import config as tftcfg
from zctx.zpdump import PaketDumpDataLoader
from zctx import dumpv2
pri = PKG + "/ppmd-probs.float16"
PK = np.fromfile(PKG + "/ppmd-bytes.uint8", dtype=np.uint8)
lader = PaketDumpDataLoader(dump_dir=PKG, tokens_filename=PKG + "/ppmd-bytes.uint8", priors_filename=pri,
    article_boundaries_filename=PKG + "/article-boundaries.int32", mmap=True, epochs=1,
    micro_batch_tokens=tftcfg.MICRO_BATCH_TOKENS, gradient_accumulation_steps=tftcfg.gradient_accumulation_steps(1),
    max_article_tokens=tftcfg.MICRO_BATCH_TOKENS, max_merge_tokens=None, device=torch.device("cpu"), rank=0, world_size=1)
SATZ = np.dtype([("src","<i4"),("tok","u1"),("p1","u1"),("p2","u1"),("mb","u1"),("mc","u1"),("fl","u1"),("res","<u2")])
roh = np.memmap(ROH, dtype=SATZ, mode="r")
rsrc = np.asarray(roh["src"]); rtok = np.asarray(roh["tok"])
pos2row = {}  # only for the stream rows that are needed
dz, _, _ = dumpv2.lade(PKG + "/zeilen.dump", mmap=True)
dr = np.fromfile(PKG + "/doc-rows.int64", dtype=np.int64)
n = lader.n_articles
docs = sorted(set([0, 1, 2, 5, 10, 20, 26, 27, 28] + [int(x) for x in np.linspace(30, n - 1, 11)]))
print(f"loader: {n} documents, {lader.n_tokens} tokens; sample of {len(docs)} documents", flush=True)
# package row i = raw row i + SKIP + 1 (mk_pkgdump drops SKIP + 1 rows in front)
V = 208
gesamt = {"zeilen": 0, "tok": 0, "p1": 0, "p2": 0, "mb": 0, "mc": 0, "src": 0, "bewertet": 0}
strom_rows = np.flatnonzero(rsrc >= TS)
strom_src = rsrc[strom_rows]
for a in docs:
    eingabe, ziel, ziel_tok, bewerte, src, prior = lader.zeilen(a)
    r0, r1 = int(dr[a]), int(dr[a + 1])
    f = dumpv2.felder(dz[r0:r1])
    R = roh[r0 + SKIP + 1: r1 + SKIP + 1]
    L = r1 - r0; gesamt["zeilen"] += L
    for k in ("tok", "p1", "p2", "mb", "mc"):
        gesamt[k] += int((np.asarray(f[k]) != np.asarray(R[k])).sum())
    rs = np.asarray(R["src"]); ls = np.asarray(src)
    gesamt["src"] += int((np.where(rs >= 0, rs - TS, rs) != ls).sum())
    # the loader input must be packed from exactly these fields
    from zctx.zpdump import packe
    e2 = packe(np.asarray(R["tok"]), np.asarray(R["p1"]), np.asarray(R["p2"]), np.asarray(R["mb"]), np.asarray(R["mc"]), lader.pk_x)
    gesamt.setdefault("eingabe", 0); gesamt["eingabe"] += int((np.asarray(e2) != np.asarray(eingabe)).sum())
    b = np.flatnonzero(np.asarray(bewerte)); gesamt["bewertet"] += len(b)
    zz = np.asarray(ziel)[b]; zt = np.asarray(ziel_tok)[b]
    # (1) target token == stream raster (ppmd-bytes of the package, computed from the stream, not from the dump)
    gesamt["ziel_tok_gegen_bett"] = gesamt.get("ziel_tok_gegen_bett", 0) + int((PK[zz] != zt).sum())
    # (2) every target except the closing target of the document is a coder stream row with the same token
    innen = zz != int(lader._doc_ziel[a])
    z = zz[innen] + TS
    idx = np.minimum(np.searchsorted(strom_src, z), len(strom_src) - 1)
    gesamt["ziel_tok_gegen_koder"] = gesamt.get("ziel_tok_gegen_koder", 0) + int(((strom_src[idx] != z) | (rtok[strom_rows[idx]] != zt[innen])).sum())
    gesamt["schlussziele_ohne_dumpzeile"] = gesamt.get("schlussziele_ohne_dumpzeile", 0) + int((~innen).sum())
print("loader against the raw coder dump, mismatches:", gesamt, flush=True)
print("LOADER VS DUMP", "PASSED" if all(v == 0 for k, v in gesamt.items() if k not in ("zeilen", "bewertet", "schlussziele_ohne_dumpzeile")) else "MISMATCH")
