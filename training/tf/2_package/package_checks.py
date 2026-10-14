#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Z table: the assurances of the training package, checked on the full-stream tables

    package_checks.py <package dir> --prioren <ppmd-probs.float16> --gate <gate.u8>
                      --strombytes <N> --token-start <S> [--stichproben 64]
                      [--ppmd-bytes <file>] [--grenzen <pages.tsv> --token-start-strom <S-6>]
                      [--prioren-quelle <file>] [--v7-z5 <json>] [--prioren-vgl <file>]
                      [--json <out>]

Prints the table as Markdown and as JSON and exits with 1 when an assurance
fails.  The rows:

 Z1  no document spans 2^17 positions or more, prompt rows included (the
     position limit of the trainer; documents above it produce NaN).
 Z2  the metadata block stands in front of the pages: the dump starts at the
     first page, `token_start` = end of the block, `n_tokens` = length behind
     the cut, and the gate zone of the block is gone after the cut.
 Z3  document tables consistent, doc-rows spans the dump exactly.
 Z4  every scored target position exactly once, no sentinel as target token.
 Z4b stream positions without a gradient are only document edges.
 Z5  three prompt classes, two masking rules: alias (bit5 and not bit6) and
     author (bit6) rows are masked except the last row of each block;
     metadata/#rd rows (bit7) are masked except the last row of a group
     that is followed by a stream row.
 Z5b block counts against a reference table (`--v7-z5`, the JSON of an
     earlier package); RED is expected when the article order changed.
 Z6  zone mask in the dump raster, ids 0/3 only; there is no title zone.
 Z7  ppmd-probs.float16 complete (rows x 416 B), sampled row sums near 1.
 Z7b alignment: row i of the prior table predicts token i+1.
 Z7c optional comparison with a second prior table (`--prioren-vgl`).
 Z7d the priors are a hard link of `--prioren-quelle` (no copy).
 Z9  title rows carry no mask bit (the title is predicted).
Z8 (manifest fields) is not checked here.

German names in the JSON: zusicherung = assurance, befund = finding,
zeilen = rows, alle_gruen = all passed.
"""
import argparse, hashlib, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dumpv2

MBT = 1 << 17          # position limit of the trainer (TFT_POSITION_LIMIT)

p = argparse.ArgumentParser()
p.add_argument("paket")
p.add_argument("--prioren", required=True)
p.add_argument("--gate", required=True)
p.add_argument("--strombytes", type=int, required=True)
p.add_argument("--token-start", dest="token_start", type=int, required=True)
p.add_argument("--stichproben", type=int, default=64)
p.add_argument("--grenzen", default=None,
               help="page table for Z9 (title spans)")
p.add_argument("--token-start-strom", dest="token_start_strom", type=int, default=0,
               help="end of the block in STREAM coordinates (Z9 converts the title spans)")
p.add_argument("--json", default=None)
p.add_argument("--v7-z5", dest="v7_z5", default=None, help="Z-table JSON with the reference counts (prompt blocks)")
p.add_argument("--prioren-quelle", dest="prioren_quelle", default=None, help="file of which ppmd-probs must be a HARD LINK")
p.add_argument("--ppmd-bytes", dest="ppmd_bytes", default=None)
p.add_argument("--prioren-vgl", dest="prioren_vgl", default=None,
               help="comparison prior table for Z7c")
a = p.parse_args()
D = a.paket.rstrip("/")

zeilen = {}
def z(nr, text, ok, detail):
    zeilen[nr] = {"zusicherung": text, "ok": bool(ok), "befund": detail}
    print(f"{nr}  {'GREEN' if ok else '*** RED ***':<11} {text}\n     {detail}")

# ---------------------------------------------------------------- tables
bnd = np.fromfile(f"{D}/article-boundaries.int32", dtype=np.int32).astype(np.int64)
rows = np.fromfile(f"{D}/doc-rows.int64", dtype=np.int64)
lz   = np.fromfile(f"{D}/doc-last-ziel.int32", dtype=np.int32)
tl   = np.fromfile(f"{D}/title-lengths.int32", dtype=np.int32)
daten, kopf, nk = dumpv2.lade(f"{D}/zeilen.dump", skip=0)
f = dumpv2.felder(daten, shift=0)
n = len(f["src"])
gate = np.fromfile(a.gate, dtype=np.uint8)
n_tokens = a.strombytes - a.token_start

# ------------------------------------------------------------------- Z1
sp = np.diff(bnd)
z("Z1", f"no document boundary spans >= 2^17 ({MBT:,}) positions, prompt rows included",
  sp.max() < MBT,
  f"max(diff(article-boundaries)) = {int(sp.max()):,} in document {int(sp.argmax())}; "
  f"{int((sp >= MBT).sum())} documents >= 2^17; documents {len(sp):,}")

# ------------------------------------------------------------------- Z2
# the stream of the package starts AFTER the metadata block (cut in front)
src_min = int(f["src"][f["src"] >= 0].min())
keine_hdr = 3 not in set(np.unique(gate).tolist())
z("Z2", "metadata block in front: the training dump starts at the FIRST PAGE; "
        "token_start = end of the block, n_tokens = length BEHIND the cut (NOT an upper bound); "
        "the gate zone was in front and is gone after the cut",
  src_min == 0 and keine_hdr and bnd[-1] == n_tokens and len(gate) == n_tokens,
  f"token_start {a.token_start:,}; smallest stream position in the dump {src_min} (expected 0); "
  f"metadata block zone in the cut gate: {'none' if keine_hdr else 'STILL THERE'}; "
  f"article-boundaries[-1] {int(bnd[-1]):,} == n_tokens {n_tokens:,}; "
  f"gate length {len(gate):,}")

# ------------------------------------------------------------------- Z3
z("Z3", "document count consistent and doc-rows spans the dump exactly",
  len(rows) - 1 == len(lz) == len(tl) and rows[0] == 0 and rows[-1] == n,
  f"len(doc-rows)-1 {len(rows)-1:,} == len(doc-last-ziel) {len(lz):,} "
  f"== len(title-lengths) {len(tl):,}; doc-rows[0] {int(rows[0])}, "
  f"doc-rows[-1] {int(rows[-1]):,} == rows {n:,}")

# ------------------------------------------------------------------- Z4
ziel, ziel_tok, bew, sent = dumpv2.zielspalten(f, rows, 208, doc_ziel=lz.astype(np.int32))
bz = ziel[bew]
u = np.unique(bz)
doppelt = len(bz) - len(u)
# "No sentinel targets" means: the TARGET TOKEN of a scored row is not a
# sentinel.  Not: "a sentinel follows the row"; in this training form that is
# the normal case (the prompt blocks stand between the stream rows), and the
# target rule skips them.  Checked is therefore `ziel_tok`.
sentziel = int((bew & (ziel_tok >= 208)).sum())
del bz
# Stream positions WITHOUT a gradient = no scored row predicts them.
# Expected: only document starts.
gedeckt = np.zeros(n_tokens, dtype=bool)
gedeckt[u[(u >= 0) & (u < n_tokens)]] = True
ohne = np.flatnonzero(~gedeckt)
del gedeckt
dok_anf = set(int(x) for x in bnd[:-1])
# Documents without a closing token (doc-last-ziel = -1, the document ends on
# an appended block): their last stream position has no predicting row.
dok_ende_ohne = set(int(bnd[d + 1]) - 1 for d in np.flatnonzero(lz < 0))
ohne_anf = sum(1 for x in ohne if int(x) in dok_anf)
ohne_end = sum(1 for x in ohne if int(x) in dok_ende_ohne and int(x) not in dok_anf)
ohne_rest = [int(x) for x in ohne if int(x) not in dok_anf and int(x) not in dok_ende_ohne]
GRAD = {"ohne_gradient": int(len(ohne)), "davon_dokumentanfang": int(ohne_anf),
        "davon_dokumentende_ohne_schlusstoken": int(ohne_end),
        "dokumente_ohne_schlusstoken": int((lz < 0).sum()),
        "rest": ohne_rest[:50], "rest_n": len(ohne_rest)}
z("Z4", "probe 0: every scored target position exactly once, no sentinel targets",
  doppelt == 0 and sentziel == 0,
  f"scored {int(bew.sum()):,}, distinct targets {len(u):,}, repeated {doppelt}, "
  f"sentinel targets {sentziel}")
z("Z4b", "stream positions without a gradient are only document edges "
         "(start without a prompt block; end of a document without a closing token)",
  len(ohne_rest) == 0,
  f"without a gradient {len(ohne):,} of n_tokens {n_tokens:,}; of these document start "
  f"(article-boundaries) {ohne_anf:,}; document end without closing token {ohne_end} "
  f"(doc-last-ziel<0: {int((lz < 0).sum())} documents); rest {len(ohne_rest)} {ohne_rest[:10]}")
del u, ohne

# ------------------------------------------------------------------- Z5
fl = f["fl"]
al = (fl & 32) != 0
au = (fl & 64) != 0
alias_rein = al & ~au
def gruppen(maske):
    i = np.flatnonzero(maske)
    if not len(i): return 0, np.array([], dtype=np.int64)
    b = np.flatnonzero(np.diff(i) != 1)
    return len(b) + 1, np.concatenate((i[b], [i[-1]]))
n_al, en_al = gruppen(alias_rein)
n_au, en_au = gruppen(au)
unmask_al = int(((fl[en_al] & 2) == 0).sum()) if n_al else 0
unmask_au = int(((fl[en_au] & 2) == 0).sum()) if n_au else 0
al_unmask_ges = int((alias_rein & ((fl & 2) == 0)).sum())
au_unmask_ges = int((au & ((fl & 2) == 0)).sum())
me = (fl & 128) != 0
n_me, en_me = gruppen(me)
me_unmask = int((me & ((fl & 2) == 0)).sum())
me_ohne_b4 = int((me & ((fl & 16) == 0)).sum())
# Per metadata/#rd group exactly the last row is unmasked, if a stream row
# follows it and it is not a sentinel.  All other metadata rows stay masked.
nach = np.minimum(en_me + 1, n - 1)
en_soll = ((fl[nach] & 1) == 0) & (f["tok"][en_me] < 208)
me_unmask_ende = int(((fl[en_me] & 2) == 0).sum())
me_unmask_ende_soll = int(en_soll.sum())
me_unmask_ende_ok = bool((((fl[en_me] & 2) == 0) == en_soll).all())
sent_z = f["tok"] >= 208
sent_unmask = int((sent_z & ((fl & 2) == 0)).sum())
Z5X = {"alias_zeilen": int(alias_rein.sum()), "alias_bloecke": n_al, "alias_unmask": al_unmask_ges,
       "autor_zeilen": int(au.sum()), "autor_bloecke": n_au, "autor_unmask": au_unmask_ges,
       "meta_zeilen": int(me.sum()), "meta_gruppen": n_me, "meta_unmask": me_unmask,
       "meta_unmask_ende": me_unmask_ende, "meta_unmask_soll": me_unmask_ende_soll,
       "sentinelzeilen": int(sent_z.sum()), "sentinel_unmask": sent_unmask}
z("Z5", "three prompt classes, all src=-1: ALIAS (bit5&!bit6) and AUTHOR (bit6) "
        "masked EXCEPT the last row of each block; METADATA/#rd (bit7): ONLY the last row of each group "
        "unmasked (a stream row behind it, tok<V), all others masked",
  al_unmask_ges == n_al and au_unmask_ges == n_au
  and unmask_al == n_al and unmask_au == n_au
  and int((alias_rein & (f["src"] >= 0)).sum()) == 0
  and int((au & (f["src"] >= 0)).sum()) == 0
  and me_unmask == me_unmask_ende == me_unmask_ende_soll and me_unmask_ende_ok
  and int((me & (f["src"] >= 0)).sum()) == 0
  and me_ohne_b4 == n_me,
  f"ALIAS {int(alias_rein.sum()):,} rows in {n_al:,} blocks, unmasked {al_unmask_ges:,} "
  f"(of these last rows of a block {unmask_al:,}); "
  f"AUTHOR {int(au.sum()):,} rows in {n_au:,} blocks, unmasked {au_unmask_ges:,} "
  f"(of these last rows of a block {unmask_au:,}); "
  f"METADATA/#rd {int(me.sum()):,} rows in {n_me:,} groups, unmasked {me_unmask:,} "
  f"(of these group ends {me_unmask_ende:,}, expected {me_unmask_ende_soll:,}, rule exact per row "
  f"{'yes' if me_unmask_ende_ok else 'NO'}), without bit4 {me_ohne_b4:,} (expected {n_me:,}); "
  f"sentinel rows (tok>=V) {int(sent_z.sum()):,}, of these unmasked {sent_unmask:,}; all src == -1")
del sent_z

# ------------------------------------------------------------------- Z6
ids = sorted(int(v) for v in np.unique(gate))
z("Z6", "zone mask in the dump raster (max(src)+1 = stream bytes), ids 0/3 only; "
        "there is no title zone (5) in this form",
  len(gate) == n_tokens and int(f["src"].max()) + 1 <= n_tokens and set(ids) <= {0, 3}
  and 5 not in set(ids),
  f"gate {len(gate):,} B == n_tokens {n_tokens:,}; max(src)+1 {int(f['src'].max())+1:,}; "
  f"ids {ids} (after the cut in front id 3 is empty; id 5 does not occur)")

# ------------------------------------------------------------------- Z7
pg = os.path.getsize(a.prioren)
soll = n_tokens * 416
rng = np.random.default_rng(6)
stellen = np.unique(np.concatenate([[0, n_tokens - 1],
                                    rng.integers(0, n_tokens, a.stichproben)]))
summen = []
with open(a.prioren, "rb") as fh:
    for i in stellen:
        fh.seek(int(i) * 416)
        v = np.frombuffer(fh.read(416), dtype=np.float16).astype(np.float64)
        summen.append(float(v.sum()))
summen = np.array(summen)
gut = int(((summen > 0.9) & (summen < 1.1)).sum())
z("Z7", f"ppmd-probs.float16 complete: length = rows x 416, >= {a.stichproben} "
        "samples with row sum ~ 1 (catches a placeholder file)",
  pg == soll and gut == len(summen),
  f"file {pg:,} B, expected {soll:,} ({n_tokens:,} x 416); {len(summen)} samples, "
  f"{gut} with 0.9 < sum < 1.1; min {summen.min():.4f} max {summen.max():.4f}")

# ------------------------------------------------------------------- Z7b/Z7c
# Z7b alignment: row i predicts token i+1, measured on the table itself
#     against the ppmd-bytes of the package.
# Z7c five fixed positions: row sha against the ZERO row (416 zero bytes) and
#     against a comparison table; plus the share of bit-identical rows over
#     all samples.
Z7X = {}
if a.ppmd_bytes:
    pb = np.memmap(a.ppmd_bytes, dtype=np.uint8, mode="r")
    rng2 = np.random.default_rng(20)
    st2 = np.sort(rng2.integers(2, n_tokens - 3, 8000))
    ce = {k: 0.0 for k in (-1, 0, 1, 2)}; treff = {k: 0 for k in ce}
    gleich = 0; dmax = 0.0; dsum = 0.0
    with open(a.prioren, "rb") as fh, open(a.prioren_vgl or a.prioren, "rb") as fv:
        for i in st2:
            fh.seek(int(i) * 416); v = np.frombuffer(fh.read(416), dtype=np.float16).astype(np.float64)
            fv.seek(int(i) * 416); w = np.frombuffer(fv.read(416), dtype=np.float16).astype(np.float64)
            for k in ce:
                t = int(pb[int(i) + k])
                ce[k] += -np.log2(max(v[t], 1e-9)); treff[k] += int(v.argmax() == t)
            gleich += int((v == w).all()); d = float(np.abs(v - w).sum()); dsum += d; dmax = max(dmax, d)
    m2 = len(st2)
    ce = {k: ce[k] / m2 for k in ce}
    beste = min(ce, key=ce.get)
    z("Z7b", "alignment of the prior table: row i predicts token i+1 (measured, 8,000 rows)",
      beste == 1 and ce[1] < 4.0,
      "CE bit/token per offset " + ", ".join(f"{k:+d}: {ce[k]:.3f} (argmax {100*treff[k]/m2:.1f} %)" for k in sorted(ce))
      + f"; best offset {beste:+d}")
    Z7X = {"ce": {str(k): ce[k] for k in ce}}
if a.ppmd_bytes and a.prioren_vgl:
    null = hashlib.sha256(b"\0" * 416).hexdigest()
    fuenf = [0, n_tokens // 4, n_tokens // 2, 3 * n_tokens // 4, n_tokens - 1]
    tab = []
    with open(a.prioren, "rb") as fh, open(a.prioren_vgl, "rb") as fv:
        for i in fuenf:
            fh.seek(i * 416); r20 = fh.read(416); fv.seek(i * 416); r25 = fv.read(416)
            h20, h25 = hashlib.sha256(r20).hexdigest(), hashlib.sha256(r25).hexdigest()
            s20 = float(np.frombuffer(r20, dtype=np.float16).astype(np.float64).sum())
            tab.append({"zeile": i, "sha20": h20[:16], "sha25": h25[:16], "summe20": round(s20, 4),
                        "null": h20 == null, "gleich25": h20 == h25})
    kein_null = not any(t["null"] for t in tab)
    verschieden = sum(1 for t in tab if not t["gleich25"])
    Z7X = {"ce": {str(k): ce[k] for k in ce}, "fuenf": tab, "bitgleich_anteil": gleich / m2,
           "mittel_l1": dsum / m2, "max_l1": dmax}
    z("Z7c", "priors at five positions: no zero row; distinguishable from the comparison table",
      kein_null and gleich < m2,
      "; ".join(f"row {t['zeile']:,}: sha {t['sha20']} / comparison {t['sha25']} sum {t['summe20']}"
                f"{' ZERO' if t['null'] else ''}{' =cmp' if t['gleich25'] else ' !=cmp'}" for t in tab)
      + f" | {verschieden}/5 positions differ; over {m2:,} samples bit-identical {100*gleich/m2:.1f} %, "
      f"mean L1 {dsum/m2:.4f}, max L1 {dmax:.3f}")

# ------------------------------------------------------------------- Z7d
if a.prioren_quelle:
    st_a, st_b = os.stat(a.prioren), os.stat(a.prioren_quelle)
    z("Z7d", "the priors are a HARD LINK of the source file (no copy)",
      (st_a.st_ino, st_a.st_dev) == (st_b.st_ino, st_b.st_dev) and st_a.st_size == soll,
      f"inode {st_a.st_ino} == {st_b.st_ino} ({a.prioren_quelle}), link count {st_a.st_nlink}, size {st_a.st_size:,}")

# ------------------------------------------------------------------- Z5b
if a.v7_z5:
    v7 = json.load(open(a.v7_z5))["z5"]
    keys = ("alias_zeilen", "alias_bloecke", "alias_unmask", "autor_zeilen", "autor_bloecke", "autor_unmask")
    diff = {k: (v7[k], Z5X[k]) for k in keys if v7[k] != Z5X[k]}
    z("Z5b", "prompt blocks AUTHOR/ALIAS unchanged against the reference; sentinel rows still masked",
      not diff and Z5X["sentinel_unmask"] == 0,
      "; ".join(f"{k} reference {v7[k]:,} / here {Z5X[k]:,}" for k in keys)
      + f"; sentinel reference {v7['sentinelzeilen']:,} / here {Z5X['sentinelzeilen']:,}, unmasked {Z5X['sentinel_unmask']}"
      + (f"; DIFFERENCE {diff}" if diff else ""))

# ------------------------------------------------------------------- Z9
# The assurance the whole training form rests on: the title is PREDICTED.
# The title spans come from the page table (there is no title zone to read
# them from).
if a.grenzen:
    G = np.loadtxt(a.grenzen, delimiter="\t", skiprows=1, dtype=np.int64)
    ts = G[:, 3] - a.token_start_strom
    tlen = G[:, 4]
    gilt = (ts >= 0) & (ts < n_tokens)
    obergrenze = int(f["src"].max()) + 2
    titelbyte = np.zeros(obergrenze, bool)
    for s0, l0 in zip(ts[gilt], tlen[gilt]):
        titelbyte[int(s0):int(s0) + int(l0)] = True
    ist_titel = (f["src"] >= 0) & titelbyte[np.clip(f["src"], 0, obergrenze - 1)]
    tm = int((ist_titel & ((fl & 2) != 0)).sum())
    z("Z9", "title rows carry NO mask bit (the title is predicted)",
      tm == 0 and int(ist_titel.sum()) > 0,
      f"title rows in the dump {int(ist_titel.sum()):,}, of these masked {tm} (expected 0); "
      f"rows added to the scored set = {int((ist_titel & ((fl & 2) == 0)).sum()):,}")

alle = all(v["ok"] for v in zeilen.values())
print("\n| # | assurance | finding | result |")
print("|---|---|---|---|")
for nr in sorted(zeilen):
    v = zeilen[nr]
    print(f"| {nr} | {v['zusicherung']} | {v['befund']} | {'GREEN' if v['ok'] else 'RED'} |")
print(f"| Z8 | manifest fields | not checked by this tool | - |")
print(f"\nRESULT Z1-Z7+Z9: {'ALL GREEN' if alle else '*** AT LEAST ONE RED ***'}")
if a.json:
    json.dump({"zeilen": zeilen, "alle_gruen": alle, "n_tokens": n_tokens,
               "gradient": GRAD, "z5": Z5X, "z7": Z7X,
               "laengstes_dokument_zeilen": int(np.diff(rows).max()),
               "bewertet": int(bew.sum()),
               "token_start": a.token_start, "strombytes": a.strombytes,
               "zeilen_im_dump": int(n), "dokumente": int(len(rows) - 1)},
              open(a.json, "w"), indent=1, sort_keys=True)
sys.exit(0 if alle else 1)
