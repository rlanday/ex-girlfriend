#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Loader probe over the WHOLE package, on the CPU

    TRAIN_TREE=<training tree> [PYLIB=<python packages>] loader_probe.py <package dir>
                                                         [--soll-bewertet N]

Runs the real trainer loader (arm `pkd` of the training tree assembled as in
`training/README.md` section 1) over every micro-batch of the package in
order and checks each one: target tokens inside 0..V-1, finite priors, prior
row sums of the scored rows between 0.9 and 1.1.  It also counts scored prior
rows with a float16 zero (underflow, information only).  The GPU libraries
that `tft.config` imports through `pysrc.model` (fla, flash_attn, ...) are
replaced by empty stand-ins; the loader path does not use them.

The epoch must be complete: the trainer loader builds the micro-batches in a
prefetch thread whose exception only ends the epoch early (`pysrc/data.py`
`prefetch()`, end mark in `finally`), so the probe counts the micro-batches
against `n_ordered_micro_batches()` of the loader and, with `--soll-bewertet`
(the scored count of the Z table), the scored rows; a difference fails.
(German names: Attrappe = stand-in, Wurzeln = top-level packages, lader =
loader, bewertet = scored, nullzeilen = rows with a zero.)
"""
import os, sys
if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])
# stand-ins ONLY for the GPU model libraries
import importlib.abc, importlib.machinery, types
class _Attrappe(types.ModuleType):
    __path__ = []
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return type(n, (), {"__init__": lambda *a, **k: None, "__call__": lambda *a, **k: None})
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    WURZELN = ("fla", "flash_attn", "causal_conv1d", "mamba_ssm")
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.WURZELN:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec): return _Attrappe(spec.name)
    def exec_module(self, m): pass
sys.meta_path.insert(0, _Finder())
sys.path.insert(0, os.environ["TRAIN_TREE"])
for k, v in (("ZCTX_ARM", "pkd"), ("PK_LAST_X0ROW", "1"), ("PK_REDIRMERGE", "1"), ("PK_SLOT_DIM", "16"),
             ("TFT_MICRO_BATCH_TOKENS", "262144"), ("TFT_OPTIMIZER_BATCH_TOKENS", "1048576")):
    os.environ.setdefault(k, v)
PKG = sys.argv[1]
SOLL_BEW = int(sys.argv[sys.argv.index("--soll-bewertet") + 1]) if "--soll-bewertet" in sys.argv else None
os.environ["PK_DUMP_DIR"] = PKG
import torch, time
# OpenMP teams of OMP_NUM_THREADS (4 at most), no inter-op pool; the
# prefetch thread of the loader gets a team of its own, so build_package.sh
# sets OMP_NUM_THREADS to half its thread limit and one BLAS thread
torch.set_num_threads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))
torch.set_num_interop_threads(1)
from tft import config as tftcfg
from zctx.zpdump import PaketDumpDataLoader
lader = PaketDumpDataLoader(
    dump_dir=PKG, tokens_filename=os.path.join(PKG, "ppmd-bytes.uint8"),
    priors_filename=os.path.join(PKG, "ppmd-probs.float16"),
    article_boundaries_filename=os.path.join(PKG, "article-boundaries.int32"),
    mmap=True, epochs=1, micro_batch_tokens=tftcfg.MICRO_BATCH_TOKENS,
    gradient_accumulation_steps=tftcfg.gradient_accumulation_steps(1),
    max_article_tokens=tftcfg.MICRO_BATCH_TOKENS, max_merge_tokens=None,
    device=torch.device("cpu"), rank=0, world_size=1)
SOLL_MB = lader.n_ordered_micro_batches()
print("loader ready: %d documents, %d tokens, V %d, %d micro-batches" % (lader.n_articles, lader.n_tokens, lader.vocabulary_size, SOLL_MB), flush=True)
if SOLL_BEW is None:
    print("  --   without --soll-bewertet only the micro-batch count shows a shortened epoch", flush=True)
V = lader.vocabulary_size
nullzeilen = nullwerte = 0; nullbsp = None
n = bew = pos = 0; smin, smax = 9.0, 0.0; spmax = 0; t0 = time.time()
for artikel, mb in lader.iter_ordered_micro_batches():
    n += 1
    lm = mb.loss_mask; ot = mb.output_tokens
    b = int(lm.sum()); bew += b; pos += int(mb.input_tokens.numel())
    zt = ot[lm]; assert (zt >= 0).all() and (zt < V).all(), "target token outside 0..V-1"
    ps = mb.priors[lm]
    assert torch.isfinite(ps).all(), "NaN/Inf in the prior columns"
    s = ps.float().sum(-1); smin = min(smin, float(s.min())); smax = max(smax, float(s.max()))
    assert float(s.min()) > 0.9 and float(s.max()) < 1.1, "prior row sum %.4f..%.4f" % (float(s.min()), float(s.max()))
    nz = (ps == 0).any(-1); nullzeilen += int(nz.sum()); nullwerte += int((ps == 0).sum())
    if nz.any() and nullbsp is None: nullbsp = (n, int(nz.nonzero()[0][0]))
    gr = mb.cumulative_sequence_lengths[0]; spmax = max(spmax, int((gr[1:] - gr[:-1]).max()))
    if n % 200 == 0:
        print("  %d micro-batches, scored %d, %.0f s" % (n, bew, time.time() - t0), flush=True)
print("scored prior rows with at least one 0 (float16 underflow): %d, zero values %d, first example (micro-batch, pos) %s" % (nullzeilen, nullwerte, nullbsp), flush=True)
# spmax: the longest segment of cumulative_sequence_lengths over all
# micro-batches; that is often the filler segment at the end of a micro-batch
# (micro-batch length minus the documents in it), not a document
fehler = []
if n != SOLL_MB:
    fehler.append("%d micro-batches != %d of the loader: the epoch ended early (an exception in the prefetch thread "
                  "of the trainer loader ends it without an error)" % (n, SOLL_MB))
if SOLL_BEW is not None and bew != SOLL_BEW:
    fehler.append("scored %d != %d of the Z table" % (bew, SOLL_BEW))
if fehler:
    print("LOADER PROBE FAILED: " + "; ".join(fehler), flush=True)
    sys.exit(1)
print("LOADER PROBE PASSED: %d micro-batches (= loader), scored %d%s, longest segment of a micro-batch %d (filler included), prior sums %.5f..%.5f, %.0f s"
      % (n, bew, " (= Z table)" if SOLL_BEW is not None else "", spmax, smin, smax, time.time() - t0), flush=True)
