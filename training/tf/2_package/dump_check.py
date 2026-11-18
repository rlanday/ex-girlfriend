#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""File check of a dump or prior run, directly after the run

    VOCAB_BITMAP=<vocab-bitmap.u8> dump_check.py <run dir> <coder stream> [--anker <hex>] [--nur-dateien]
    VOCAB_BITMAP=<vocab-bitmap.u8> dump_check.py <run dir> <coder stream> --prioren <prior dir> [--kopf-laenge N]

<coder stream> is the stream the run coded (on a probe bed the prefix of the
coder stream that belongs to it); the raster is that stream behind its 6-byte
segment header.

Dump run: zone mask and match-*.uint8 == raster length, paket-zeilen.bin a
multiple of 12, stream rows strictly increasing and without duplicates, 0
token mismatches against the stream, and the LAST stream row == end of the
raster - 1 (a lost tail buffer shows up here).  This is the completeness
criterion of the dump run.  The dump writes no row for the last byte of a
page anchor (on C40 4,796 of the 4,852 positions without a row behind the
cut); a probe bed ends behind a page anchor, so with `--anker <hex>` (the
page anchor of the tree) a coder stream that ends with the anchor has its
last stream row at the end of the raster - 2.  `--nur-dateien` (files only) checks only the
lengths.

Prior run (`--prioren <dir>` holding ppmd-probs.float16, ppmd-bytes.uint8,
ppmd-quelle.u8): one raster row per coder step, the origin raster index
parallel to the bytes, 2V B of priors per row (V = values of VOCAB_BITMAP), the rows of coded bytes
(origin 0) exactly the raster in order (count and every token), the injected
rows (origin 1) counted.  This is the completeness criterion of the prior run.
The 6-byte segment header of a "-n" run carries the length of ITS input, the
bed (canonical stream), not the length of the coder stream: `--kopf-laenge N`
(N = size of the bed) gives the header the prior raster must carry.  The header
lies in front of the metadata-block restart of PPMd and in front of the cut.

Exit 0 = passed; the result is printed as JSON (keys in German: saetze =
records, stromsaetze = stream records, eingefuegt = inserted, fehler = errors,
GRUEN = passed; prior_* = prior raster).
"""
import sys, os, json, numpy as np
D, bett = sys.argv[1], sys.argv[2]; nur = "--nur-dateien" in sys.argv
PRI = sys.argv[sys.argv.index("--prioren") + 1] if "--prioren" in sys.argv else None
n = os.path.getsize(bett); NR = n + 6
KL = int(sys.argv[sys.argv.index("--kopf-laenge") + 1]) if "--kopf-laenge" in sys.argv else n
ANK = bytes.fromhex(sys.argv[sys.argv.index("--anker") + 1]) if "--anker" in sys.argv else b""
res = {"raster_len_soll": NR, "kopf_laenge": KL}; fehler = []
# the byte set (VOCAB_BITMAP): V values, a prior row holds V float16 = 2V bytes
_bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
MENGE = np.array([v for v in range(256) if (_bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
ZB = 2 * len(MENGE)


def raster_lesen():
    menge = MENGE
    lut = np.full(256, 255, np.uint8); lut[menge] = np.arange(len(menge), dtype=np.uint8)
    b = np.fromfile(bett, dtype=np.uint8)
    kopf = np.array([1] + [0x80 | ((KL >> k) & 0x7F) for k in (28, 21, 14, 7, 0)], dtype=np.int64)
    r = np.concatenate([lut[kopf], lut[b]])
    if (r == 255).any():
        fehler.append(f"coder stream byte outside the byte set ({len(menge)} values)")
    return r


if PRI:
    pb, pq, pp = (os.path.join(PRI, f) for f in ("ppmd-bytes.uint8", "ppmd-quelle.u8", "ppmd-probs.float16"))
    for f in (pb, pq, pp):
        if not os.path.exists(f):
            fehler.append(f"{f} missing")
    if not fehler:
        nb, nq, npp = os.path.getsize(pb), os.path.getsize(pq), os.path.getsize(pp)
        res.update({"prior_zeilen": nb, "prior_quelle_bytes": nq, "prior_probs_bytes": npp})
        if nq != nb: fehler.append(f"ppmd-quelle {nq} != ppmd-bytes {nb}")
        if npp != nb * ZB: fehler.append(f"ppmd-probs {npp} != ppmd-bytes {nb} x {ZB}")
        q = np.fromfile(pq, dtype=np.uint8); by = np.memmap(pb, dtype=np.uint8, mode="r")
        if not set(int(v) for v in np.unique(q)) <= {0, 1}: fehler.append("ppmd-quelle holds values other than 0/1")
        strom = np.flatnonzero(q == 0)
        res["prior_strom"] = int(len(strom)); res["prior_eingespeist"] = int((q == 1).sum())
        if len(strom) != NR:
            fehler.append(f"stream rows of the prior raster {len(strom)} != raster {NR} (incomplete run?)")
        elif len(q) == len(by):
            raster = raster_lesen()
            abw = int((np.asarray(by)[strom] != raster).sum())
            res["prior_token_abweichungen"] = abw
            if abw: fehler.append(f"{abw} token mismatches of the prior stream rows against the coder stream")
    res["fehler"] = fehler; res["GRUEN"] = not fehler
    print(json.dumps(res, indent=1)); sys.exit(0 if not fehler else 3)

for m in ("zonenmaske.u8", "match-bytes.uint8", "match-len.uint8"):
    p = os.path.join(D, m)
    s = os.path.getsize(p) if os.path.exists(p) else -1; res[m] = s
    if s != NR: fehler.append(f"{m} {s} != {NR}")
if not nur:
    SATZ = np.dtype([("src","<i4"),("tok","u1"),("p1","u1"),("p2","u1"),("mb","u1"),("mc","u1"),("fl","u1"),("res","<u2")])
    f = os.path.join(D, "paket-zeilen.bin"); sz = os.path.getsize(f); res["paket_bytes"] = sz
    if sz % 12: fehler.append("paket-zeilen.bin not a multiple of 12")
    if sz % 1048584 == 0: res["hinweis"] = "size is a multiple of the 1 MiB buffer (suspected missing tail)"
    raster = raster_lesen()
    r = np.memmap(f, dtype=SATZ, mode="r"); N = len(r); CH = 1 << 25
    nstrom = nein = 0; abw = 0; mono = 0; prev = -1; seen = np.zeros(NR, np.bool_); dup = 0
    for a in range(0, N, CH):
        c = np.asarray(r[a:a+CH]); src = c["src"].astype(np.int64)
        s = src >= 0; nein += int((~s).sum()); ss = src[s]; nstrom += len(ss)
        if len(ss):
            d = np.diff(np.concatenate([[prev], ss])); mono += int((d <= 0).sum()); prev = int(ss[-1])
            if ss.max() >= NR: fehler.append("src behind the raster"); break
            abw += int((c["tok"][s] != raster[ss]).sum()); dup += int(seen[ss].sum()); seen[ss] = True
    # the last byte of a page anchor gets no row: a stream that ends with the
    # anchor (a probe bed) has its last stream row one position earlier
    with open(bett, "rb") as fh:
        fh.seek(max(0, n - len(ANK)))
        ende_anker = bool(ANK) and n >= len(ANK) and fh.read() == ANK
    soll_letzte = NR - 2 if ende_anker else NR - 1
    res.update({"saetze": N, "stromsaetze": nstrom, "eingefuegt": nein, "letzte_stromzeile": prev,
                "letzte_stromzeile_soll": soll_letzte, "strom_endet_mit_anker": ende_anker,
                "token_abweichungen": abw, "doppelt": dup,
                "monotonie_brueche": mono, "positionen_ohne_zeile": int(NR - seen.sum())})
    if prev != soll_letzte: fehler.append(f"last stream row {prev} != {soll_letzte}")
    if ende_anker and seen[NR - 1]: fehler.append("the last byte of the page anchor at the stream end has a row")
    if abw: fehler.append(f"{abw} token mismatches")
    if dup: fehler.append(f"{dup} duplicate positions")
    if mono: fehler.append(f"{mono} breaks of monotonicity")
res["fehler"] = fehler; res["GRUEN"] = not fehler
print(json.dumps(res, indent=1)); sys.exit(0 if not fehler else 3)
