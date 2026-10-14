#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""File check of a dump or prior run, directly after the run

    VOCAB_BITMAP=<vocab-bitmap.u8> dump_check.py <run dir> <coder stream> [--nur-dateien]

Checks the files themselves: zone mask and match-*.uint8 == raster length
(segment header + stream), paket-zeilen.bin a multiple of 12, stream rows
strictly increasing and without duplicates, 0 token mismatches against the
stream, and the LAST stream row == end of the raster - 1 (a lost tail buffer
shows up here).  `--nur-dateien` (files only) checks only the lengths, for the
prior run.  Exit 0 = passed; the result is printed as JSON (keys in German:
saetze = records, stromsaetze = stream records, eingefuegt = inserted,
fehler = errors, GRUEN = passed).
"""
import sys, os, json, numpy as np
D, bett = sys.argv[1], sys.argv[2]; nur = "--nur-dateien" in sys.argv
n = os.path.getsize(bett); NR = n + 6
res = {"raster_len_soll": NR}; fehler = []
for m in ("zonenmaske.u8", "match-bytes.uint8", "match-len.uint8"):
    s = os.path.getsize(os.path.join(D, m)); res[m] = s
    if s != NR: fehler.append(f"{m} {s} != {NR}")
if not nur:
    SATZ = np.dtype([("src","<i4"),("tok","u1"),("p1","u1"),("p2","u1"),("mb","u1"),("mc","u1"),("fl","u1"),("res","<u2")])
    f = os.path.join(D, "paket-zeilen.bin"); sz = os.path.getsize(f); res["paket_bytes"] = sz
    if sz % 12: fehler.append("paket-zeilen.bin not a multiple of 12")
    if sz % 1048584 == 0: res["hinweis"] = "size is a multiple of the 1 MiB buffer (suspected missing tail)"
    bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
    menge = np.array([v for v in range(256) if (bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
    lut = np.full(256, 255, np.uint8); lut[menge] = np.arange(len(menge), dtype=np.uint8)
    b = np.fromfile(bett, dtype=np.uint8)
    kopf = np.array([1] + [0x80 | ((n >> k) & 0x7F) for k in (28, 21, 14, 7, 0)], dtype=np.int64)
    raster = np.concatenate([lut[kopf], lut[b]]); del b
    r = np.memmap(f, dtype=SATZ, mode="r"); N = len(r); CH = 1 << 25
    nstrom = nein = 0; abw = 0; mono = 0; prev = -1; seen = np.zeros(NR, np.bool_); dup = 0
    for a in range(0, N, CH):
        c = np.asarray(r[a:a+CH]); src = c["src"].astype(np.int64)
        s = src >= 0; nein += int((~s).sum()); ss = src[s]; nstrom += len(ss)
        if len(ss):
            d = np.diff(np.concatenate([[prev], ss])); mono += int((d <= 0).sum()); prev = int(ss[-1])
            if ss.max() >= NR: fehler.append("src behind the raster"); break
            abw += int((c["tok"][s] != raster[ss]).sum()); dup += int(seen[ss].sum()); seen[ss] = True
    res.update({"saetze": N, "stromsaetze": nstrom, "eingefuegt": nein, "letzte_stromzeile": prev,
                "letzte_stromzeile_soll": NR - 1, "token_abweichungen": abw, "doppelt": dup,
                "monotonie_brueche": mono, "positionen_ohne_zeile": int(NR - seen.sum())})
    if prev != NR - 1: fehler.append(f"last stream row {prev} != {NR-1}")
    if abw: fehler.append(f"{abw} token mismatches")
    if dup: fehler.append(f"{dup} duplicate positions")
    if mono: fehler.append(f"{mono} breaks of monotonicity")
res["fehler"] = fehler; res["GRUEN"] = not fehler
print(json.dumps(res, indent=1)); sys.exit(0 if not fehler else 3)
