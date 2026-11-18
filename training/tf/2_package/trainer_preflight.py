#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Preflight of the trainer imports, before the coder runs

    TRAIN_TREE=<training tree> [PYLIB=<python packages>] trainer_preflight.py

Imports what loader_vs_dump.py, loader_probe.py and `python -m tft.dataset
check` import in stage 3 and 5 of build_package.sh (numpy, torch, tft.config,
tft.dataset, zctx.zpdump, zctx.dumpv2), with the same stand-ins for the GPU
model libraries.  A missing package (e.g. jaxtyping) stops the recipe here and
not hours later behind the dump and prior runs.  PYLIB names a directory with
additional packages (`pip install --target <dir> ...`).
"""
import os, sys
if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])
import importlib, importlib.abc, importlib.machinery, types


class _Attrappe(types.ModuleType):
    __path__ = []

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return type(n, (), {"__init__": lambda *a, **k: None, "__call__": lambda *a, **k: None})


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    WURZELN = ("fla", "flash_attn", "causal_conv1d", "mamba_ssm")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.WURZELN:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _Attrappe(spec.name)

    def exec_module(self, m):
        pass


sys.meta_path.insert(0, _Finder())
sys.path.insert(0, os.environ["TRAIN_TREE"])
fehlt = []
for mod in ("numpy", "torch", "tft.config", "tft.dataset", "zctx.zpdump", "zctx.dumpv2"):
    try:
        importlib.import_module(mod)
    except Exception as e:          # report every missing piece, not only the first
        fehlt.append(f"{mod}: {type(e).__name__}: {e}")
if fehlt:
    print("trainer preflight FAILED (python " + sys.executable + "):")
    for f in fehlt:
        print("   " + f)
    sys.exit(1)
import numpy, torch
print(f"trainer preflight passed: python {sys.version.split()[0]}, numpy {numpy.__version__}, torch {torch.__version__}")
