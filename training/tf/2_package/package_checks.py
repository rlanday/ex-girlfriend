#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Z table: the assurances of the training package, checked on the full-stream tables

    package_checks.py <package dir> --prioren <ppmd-probs.float16> --gate <gate.u8>
                      --strombytes <N> --token-start <S> [--stichproben 64]
                      --ppmd-bytes <file>
                      ( --grenzen <pages.tsv> --token-start-strom <S-6>
                        --slots "<slot content line>" --zwang-tafel <tf1_block_set2.h>
                      | --textform <MANIFEST.json> )
                      [--prioren-quelle <file>] [--v7-z5 <json>] [--prioren-vgl <file>]
                      [--json <out>]

Prints the table as Markdown and as JSON and exits with 1 when an assurance
fails (2 when the text-form inputs are missing).  The text-form facts the rows
need come from the build (page table, the slot sizes of the dump run, the
table of the forced title prefixes of the tree) or, on the training host,
from the field `textform` that the build wrote into MANIFEST.json; the JSON
output carries that field.  The rows:

 Z1  no document spans 2^17 positions or more, prompt rows included (the
     position limit of the trainer; documents above it produce NaN).
 Z2  the metadata block stands in front of the pages: the dump starts at the
     first page, `token_start` = end of the block, `n_tokens` = length behind
     the cut, and the gate zone of the block is gone after the cut.  Text
     form: behind the cut at most one zone-3 run remains, the title line of
     page 0 behind the text start: it starts at the title of page 0 and is
     at most title length + 2 long (title, line feed, run-on byte).
 Z3  document tables consistent, doc-rows spans the dump exactly.
 Z4  every scored target position exactly once, no sentinel as target token.
 Z4b stream positions without a gradient are only document edges, or
     positions whose every predicting row the coder masked on purpose, or
     positions inside a gated zone behind the cut (title line of page 0);
     no other position may lack a predicting row.  Masked on purpose may
     only be positions in slot 2 (the fully forced name bytes of the author
     article; slot 2 = package positions s1+2 .. s1+2+s2-1, the end-mark
     pairs at both ends are checked; the end-mark byte is read from the tree,
     `--endmarke`, and travels in the text-form facts) or bytes of a forced
     title prefix.
 Z5  three prompt classes, two masking rules: alias (bit5 and not bit6) and
     author (bit6) rows are masked except the last row of each block;
     metadata/#rd rows (bit7) are masked except the last row of a group
     that is followed by a stream row.  Text form: in front of a forced
     title prefix (`Image:`, `Help:`, `MediaWiki:`, `Wikipedia:` on the page
     ranges of tf1/tf1_block_set2.h) that last row predicts a forced byte
     and is masked; every such page has exactly one.
 Z5b block counts against a reference table (`--v7-z5`, the JSON of an
     earlier package); RED is expected when the article order changed.
 Z6  zone mask in the dump raster, ids 0/3 only; there is no title zone.
 Z7  ppmd-probs.float16 complete (rows x 2V B, V = values of the byte set
     vocab-bitmap.u8 of the package), sampled row sums near 1.
 Z7b alignment: row i of the prior table predicts token i+1.
 Z7c optional comparison with a second prior table (`--prioren-vgl`).
 Z7d the priors are a hard link of `--prioren-quelle` (no copy).
 Z9  title rows carry no mask bit (the title is predicted), with one
     exception: the coder takes the forced title prefixes from the table and
     does not code them, so on the pages of the prefix ranges the title rows
     that predict prefix bytes 1 .. L-1 are masked (L-1 rows per page, the
     row at prefix byte L-1 is scored again); their tokens are the table
     bytes.  Exactly these rows, all of them, nothing else.
Z8 (manifest fields) is not checked here.

German names in the JSON: zusicherung = assurance, befund = finding,
zeilen = rows, alle_gruen = all passed.
"""
import argparse, hashlib, json, os, re, sys
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
p.add_argument("--slots", default=None,
               help="the line 'slot content a / b / c B' of the dump run (sizes of the text start slots)")
p.add_argument("--zwang-tafel", dest="zwang_tafel", default=None,
               help="tf1/tf1_block_set2.h of the tree: page ranges and bytes of the forced title prefixes")
p.add_argument("--endmarke", default=None,
               help="end-mark byte of the text-start slots (kEnde of src/neif_textanfang_pack.h), e.g. 0x0B")
p.add_argument("--textform", default=None,
               help="MANIFEST.json (field 'textform') or a JSON with that field: the text-form facts "
                    "of the build, in place of --grenzen/--slots/--zwang-tafel")
a = p.parse_args()
D = a.paket.rstrip("/")


def zwang_tafel_lesen(pfad):
    """page ranges, prefix lengths and bytes of the forced title prefixes,
    read from the header of the tree (not copied into this tool)"""
    s = open(pfad, encoding="utf-8").read()

    def wert(name):
        m = re.search(r"#define\s+" + name + r"\s+(.+)", s)
        if not m:
            sys.exit(f"package_checks: {name} not found in {pfad}")
        return m.group(1)
    n = int(wert("NEIF_TF1_ZWANG_N"))
    von = [int(x) for x in re.findall(r"\d+", wert("NEIF_TF1_ZWANG_VON"))]
    bis = [int(x) for x in re.findall(r"\d+", wert("NEIF_TF1_ZWANG_BIS"))]
    ln = [int(x) for x in re.findall(r"\d+", wert("NEIF_TF1_ZWANG_LEN"))]
    by = [[int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)", g)]
          for g in re.findall(r"\{([^{}]*)\}", wert("NEIF_TF1_ZWANG_BYTES"))]
    if not (len(von) == len(bis) == len(ln) == len(by) == n and n > 0):
        sys.exit(f"package_checks: forced prefix table of {pfad} inconsistent ({n} {von} {bis} {ln})")
    for b in range(n):
        if not (von[b] <= bis[b] and (b == 0 or von[b] > bis[b - 1]) and 2 <= ln[b] <= len(by[b])):
            sys.exit(f"package_checks: forced prefix range {b} of {pfad} not sorted/disjoint or too short")
    return {"tafel_sha256": hashlib.sha256(s.encode("utf-8")).hexdigest(),
            "von": von, "bis": bis, "len": ln, "bytes": [by[b][:ln[b]] for b in range(n)],
            "seiten_der_tafel": int(sum(y - x + 1 for x, y in zip(von, bis)))}


# ------------------------------------------------- text-form facts (build or manifest)
TF = None
if a.textform:
    TF = json.load(open(a.textform, encoding="utf-8"))
    TF = TF.get("textform", TF)
elif a.grenzen and a.slots and a.zwang_tafel:
    G0 = np.loadtxt(a.grenzen, delimiter="\t", skiprows=1, dtype=np.int64)
    if not (G0[:, 0] == np.arange(len(G0))).all():
        sys.exit("package_checks: the page table does not list the pages 0, 1, 2, ... in order")
    _sl = [int(x) for x in re.findall(r"\d+", a.slots)]
    if len(_sl) < 3:
        sys.exit(f"package_checks: --slots without three sizes: {a.slots!r}")
    _zt = zwang_tafel_lesen(a.zwang_tafel)
    _n_tok = a.strombytes - a.token_start
    _titel = []
    for b in range(len(_zt["von"])):
        L = _zt["len"][b]
        for k in range(_zt["von"][b], min(_zt["bis"][b], len(G0) - 1) + 1):
            st = int(G0[k, 3]) - a.token_start_strom
            if st < 0 or st + L > _n_tok:
                continue
            if int(G0[k, 4]) < L:
                sys.exit(f"package_checks: page {k} lies in forced prefix range {b} but its title "
                         f"has {int(G0[k, 4])} B < {L}")
            _titel.append([st, L, b, k])
    if a.endmarke is None:
        sys.exit("package_checks: --endmarke missing (the end-mark byte of the slots, read from the tree)")
    TF = {"slots": _sl[:3],
          "seite0_titel": [int(G0[0, 3]) - a.token_start_strom, int(G0[0, 4])],
          "slot_endmarke": int(a.endmarke, 0),
          "zwang": dict(_zt, titel=_titel)}
else:
    sys.exit("package_checks: text-form inputs missing: --grenzen, --slots and --zwang-tafel "
             "(build) or --textform <MANIFEST.json> (training host)")
SL1, SL2 = int(TF["slots"][0]), int(TF["slots"][1])
SLOT2 = (SL1 + 2, SL1 + 2 + SL2)               # [from, to) in package positions
ZW = np.array([t[:3] for t in TF["zwang"]["titel"]], dtype=np.int64).reshape(-1, 3)
ZW_START, ZW_LEN, ZW_B = ZW[:, 0], ZW[:, 1], ZW[:, 2]
# all forced bytes (Z4b) and the title positions whose rows predict bytes 1..L-1 (Z9)
ZW_BYTES = (np.concatenate([np.arange(s, s + L) for s, L in zip(ZW_START, ZW_LEN)])
            if len(ZW) else np.zeros(0, np.int64))
ZW_ZEILEN = (np.concatenate([np.arange(s, s + L - 1) for s, L in zip(ZW_START, ZW_LEN)])
             if len(ZW) else np.zeros(0, np.int64))
_vb = np.fromfile(os.path.join(D, "vocab-bitmap.u8"), dtype=np.uint8)
LUT = np.full(256, 255, np.int16)
LUT[[v for v in range(256) if (_vb[v >> 3] >> (v & 7)) & 1]] = np.arange(
    sum(1 for v in range(256) if (_vb[v >> 3] >> (v & 7)) & 1), dtype=np.int16)
V = int((LUT != 255).sum())          # vocabulary = values of the byte set of the package
ZB = 2 * V                           # bytes of a prior row (V float16)
ENDE = int(TF.get("slot_endmarke", 0x0B))   # manifests before version 2 carry no end mark: 0x0B
print(f"text form: slots {TF['slots']}, slot 2 at package positions {SLOT2[0]:,}..{SLOT2[1]-1:,}; "
      f"title of page 0 {TF['seite0_titel']}; forced title prefixes: {len(ZW):,} titles in this package "
      f"(table {TF['zwang']['seiten_der_tafel']:,} pages, ranges {TF['zwang']['von']}..{TF['zwang']['bis']}, "
      f"lengths {TF['zwang']['len']}), {len(ZW_ZEILEN):,} title rows of prefix bytes 0..L-2")

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
# zone-3 runs left in the cut gate: in the text form exactly the title line of
# page 0 behind the text start (title, line feed and the run-on byte of the
# gate); the metadata block itself must be gone
_nz = np.diff(np.concatenate([[0], (gate == 3).astype(np.int8), [0]]))
_a3, _b3 = np.flatnonzero(_nz == 1), np.flatnonzero(_nz == -1)
laeufe3 = [(int(x), int(y - x)) for x, y in zip(_a3, _b3)]
_s0, _l0 = int(TF["seite0_titel"][0]), int(TF["seite0_titel"][1])
keine_hdr = len(laeufe3) == 0 or (len(laeufe3) == 1 and laeufe3[0][0] == _s0 and laeufe3[0][1] <= _l0 + 2)
z("Z2", "metadata block in front: the training dump starts at the FIRST PAGE; "
        "token_start = end of the block, n_tokens = length BEHIND the cut (NOT an upper bound); "
        "the gate zone of the block is gone after the cut (text form: at most one zone-3 run, "
        "the title line of page 0)",
  src_min == 0 and keine_hdr and bnd[-1] == n_tokens and len(gate) == n_tokens,
  f"token_start {a.token_start:,}; smallest stream position in the dump {src_min} (expected 0); "
  f"zone-3 runs in the cut gate {len(laeufe3)} (start, length) {laeufe3[:4]}, title of page 0 at "
  f"{_s0:,}, {_l0} B: "
  f"{'only the title line of page 0' if keine_hdr else 'MORE OR OTHER ZONE-3 RUNS (metadata block still there?)'}; "
  f"article-boundaries[-1] {int(bnd[-1]):,} == n_tokens {n_tokens:,}; "
  f"gate length {len(gate):,}")

# ------------------------------------------------------------------- Z3
z("Z3", "document count consistent and doc-rows spans the dump exactly",
  len(rows) - 1 == len(lz) == len(tl) and rows[0] == 0 and rows[-1] == n,
  f"len(doc-rows)-1 {len(rows)-1:,} == len(doc-last-ziel) {len(lz):,} "
  f"== len(title-lengths) {len(tl):,}; doc-rows[0] {int(rows[0])}, "
  f"doc-rows[-1] {int(rows[-1]):,} == rows {n:,}")

# ------------------------------------------------------------------- Z4
ziel, ziel_tok, bew, sent = dumpv2.zielspalten(f, rows, V, doc_ziel=lz.astype(np.int32))
bz = ziel[bew]
u = np.unique(bz)
doppelt = len(bz) - len(u)
# "No sentinel targets" means: the TARGET TOKEN of a scored row is not a
# sentinel.  Not: "a sentinel follows the row"; in this training form that is
# the normal case (the prompt blocks stand between the stream rows), and the
# target rule skips them.  Checked is therefore `ziel_tok`.
sentziel = int((bew & (ziel_tok >= V)).sum())
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
# positions some row predicts, but only rows the coder masked (flags bit1):
# an intended loss mask (text form: forced name bytes of the author article)
_mz = ziel[(ziel >= 0) & ~f["maske"]]
gezielt_maskiert = np.zeros(n_tokens, dtype=bool)
gezielt_maskiert[_mz[_mz < n_tokens]] = True
del _mz
ohne_anf = sum(1 for x in ohne if int(x) in dok_anf)
ohne_end = sum(1 for x in ohne if int(x) in dok_ende_ohne and int(x) not in dok_anf)
_kand = [int(x) for x in ohne if int(x) not in dok_anf and int(x) not in dok_ende_ohne]
ohne_maske = [x for x in _kand if gezielt_maskiert[x]]
# a gated zone behind the cut (text form: title line of page 0) has no rows
ohne_zone = [x for x in _kand if not gezielt_maskiert[x] and x < len(gate) and gate[x] != 0]
ohne_rest = [x for x in _kand if not gezielt_maskiert[x] and not (x < len(gate) and gate[x] != 0)]
del _kand, gezielt_maskiert
# masked on purpose only in slot 2 or on a forced title prefix
_om = np.array(ohne_maske, dtype=np.int64)
_im_slot2 = (_om >= SLOT2[0]) & (_om < SLOT2[1])
_im_zwang = np.isin(_om, ZW_BYTES)
maske_fremd = _om[~(_im_slot2 | _im_zwang)]
# slot 2 lies between two end-mark pairs (slot 1 starts at package position 0)
_ende = int(LUT[ENDE])
if a.ppmd_bytes:
    _pb = np.memmap(a.ppmd_bytes, dtype=np.uint8, mode="r")
    _marken = [SL1, SL1 + 1, SLOT2[1], SLOT2[1] + 1]
    slot2_marken_ok = _ende >= 0 and SLOT2[1] + 1 < len(_pb) and all(int(_pb[i]) == _ende for i in _marken)
    del _pb
else:
    slot2_marken_ok = False
GRAD = {"ohne_gradient": int(len(ohne)), "davon_dokumentanfang": int(ohne_anf),
        "davon_dokumentende_ohne_schlusstoken": int(ohne_end),
        "davon_vom_koder_maskiert": len(ohne_maske),
        "davon_maskiert_in_slot2": int(_im_slot2.sum()),
        "davon_maskiert_erzwungene_titelbytes": int((_im_zwang & ~_im_slot2).sum()),
        "maskiert_ausserhalb": [int(x) for x in maske_fremd[:50]], "maskiert_ausserhalb_n": int(len(maske_fremd)),
        "slot2_von_bis": [SLOT2[0], SLOT2[1] - 1], "slot2_endmarken_ok": bool(slot2_marken_ok),
        "davon_in_gegateter_zone": len(ohne_zone),
        "maskiert_von_bis": [ohne_maske[0], ohne_maske[-1]] if ohne_maske else [],
        "dokumente_ohne_schlusstoken": int((lz < 0).sum()),
        "rest": ohne_rest[:50], "rest_n": len(ohne_rest)}
z("Z4", "probe 0: every scored target position exactly once, no sentinel targets",
  doppelt == 0 and sentziel == 0,
  f"scored {int(bew.sum()):,}, distinct targets {len(u):,}, repeated {doppelt}, "
  f"sentinel targets {sentziel}")
z("Z4b", "stream positions without a gradient are only document edges "
         "(start without a prompt block; end of a document without a closing token) "
         "or masked by the coder on purpose (all predicting rows carry the mask bit: only in slot 2, "
         "between its end marks, or on a forced title prefix) or inside a gated zone",
  len(ohne_rest) == 0 and len(maske_fremd) == 0 and slot2_marken_ok,
  f"without a gradient {len(ohne):,} of n_tokens {n_tokens:,}; of these document start "
  f"(article-boundaries) {ohne_anf:,}; document end without closing token {ohne_end} "
  f"(doc-last-ziel<0: {int((lz < 0).sum())} documents); masked by the coder {len(ohne_maske):,}"
  f"{' in ' + str(GRAD['maskiert_von_bis']) if ohne_maske else ''} = slot 2 "
  f"[{SLOT2[0]:,}..{SLOT2[1]-1:,}] {GRAD['davon_maskiert_in_slot2']:,} + forced title bytes "
  f"{GRAD['davon_maskiert_erzwungene_titelbytes']:,} + elsewhere {len(maske_fremd)} {GRAD['maskiert_ausserhalb'][:10]}; "
  f"end marks 0x{ENDE:02X} 0x{ENDE:02X} around slot 2 {'found' if slot2_marken_ok else 'NOT FOUND (or no --ppmd-bytes)'}; "
  f"in a gated zone {len(ohne_zone)}; "
  f"no predicting row at all {len(ohne_rest)} {ohne_rest[:10]}")
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
# Text form: a group directly in front of a forced title prefix (the stream
# row behind it is byte 0 of the prefix) predicts a forced byte and stays
# masked; each forced title in the dump has exactly one such group.
nach = np.minimum(en_me + 1, n - 1)
vor_zwang = ((fl[nach] & 1) == 0) & (f["src"][nach] >= 0) & np.isin(f["src"][nach], ZW_START)
en_soll = ((fl[nach] & 1) == 0) & (f["tok"][en_me] < V) & ~vor_zwang
me_unmask_ende = int(((fl[en_me] & 2) == 0).sum())
me_unmask_ende_soll = int(en_soll.sum())
me_unmask_ende_ok = bool((((fl[en_me] & 2) == 0) == en_soll).all())
# forced titles whose byte 0 is a stream row of this dump; each needs its group
# (presence table over the stream positions, filled in chunks: no copy of src)
_da = np.zeros(n_tokens + 1, dtype=bool)
for _i0 in range(0, n, 1 << 26):
    _s = f["src"][_i0:_i0 + (1 << 26)]
    _da[_s[(_s >= 0) & (_s <= n_tokens)]] = True
zw_im_dump = int(_da[ZW_START].sum()) if len(ZW) else 0
del _da, _s
zw_gruppen = int(vor_zwang.sum())
zw_gruppen_maskiert = int((vor_zwang & ((fl[en_me] & 2) != 0)).sum())
zw_ok = zw_gruppen == zw_gruppen_maskiert == zw_im_dump
sent_z = f["tok"] >= V
sent_unmask = int((sent_z & ((fl & 2) == 0)).sum())
Z5X = {"alias_zeilen": int(alias_rein.sum()), "alias_bloecke": n_al, "alias_unmask": al_unmask_ges,
       "autor_zeilen": int(au.sum()), "autor_bloecke": n_au, "autor_unmask": au_unmask_ges,
       "meta_zeilen": int(me.sum()), "meta_gruppen": n_me, "meta_unmask": me_unmask,
       "meta_unmask_ende": me_unmask_ende, "meta_unmask_soll": me_unmask_ende_soll,
       "meta_gruppen_vor_zwang": zw_gruppen, "meta_gruppen_vor_zwang_maskiert": zw_gruppen_maskiert,
       "zwang_titel_im_dump": zw_im_dump,
       "sentinelzeilen": int(sent_z.sum()), "sentinel_unmask": sent_unmask}
z("Z5", "three prompt classes, all src=-1: ALIAS (bit5&!bit6) and AUTHOR (bit6) "
        "masked EXCEPT the last row of each block; METADATA/#rd (bit7): ONLY the last row of each group "
        "unmasked (a stream row behind it, tok<V, no forced title prefix behind it), all others masked; "
        "one masked group end in front of every forced title prefix",
  al_unmask_ges == n_al and au_unmask_ges == n_au
  and unmask_al == n_al and unmask_au == n_au
  and int((alias_rein & (f["src"] >= 0)).sum()) == 0
  and int((au & (f["src"] >= 0)).sum()) == 0
  and me_unmask == me_unmask_ende == me_unmask_ende_soll and me_unmask_ende_ok
  and int((me & (f["src"] >= 0)).sum()) == 0
  and me_ohne_b4 == n_me and zw_ok,
  f"ALIAS {int(alias_rein.sum()):,} rows in {n_al:,} blocks, unmasked {al_unmask_ges:,} "
  f"(of these last rows of a block {unmask_al:,}); "
  f"AUTHOR {int(au.sum()):,} rows in {n_au:,} blocks, unmasked {au_unmask_ges:,} "
  f"(of these last rows of a block {unmask_au:,}); "
  f"METADATA/#rd {int(me.sum()):,} rows in {n_me:,} groups, unmasked {me_unmask:,} "
  f"(of these group ends {me_unmask_ende:,}, expected {me_unmask_ende_soll:,}, rule exact per row "
  f"{'yes' if me_unmask_ende_ok else 'NO'}), without bit4 {me_ohne_b4:,} (expected {n_me:,}); "
  f"group ends in front of a forced title prefix {zw_gruppen:,}, masked {zw_gruppen_maskiert:,}, "
  f"forced titles in the dump {zw_im_dump:,}; "
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
soll = n_tokens * ZB
rng = np.random.default_rng(6)
stellen = np.unique(np.concatenate([[0, n_tokens - 1],
                                    rng.integers(0, n_tokens, a.stichproben)]))
summen = []
with open(a.prioren, "rb") as fh:
    for i in stellen:
        fh.seek(int(i) * ZB)
        v = np.frombuffer(fh.read(ZB), dtype=np.float16).astype(np.float64)
        summen.append(float(v.sum()))
summen = np.array(summen)
gut = int(((summen > 0.9) & (summen < 1.1)).sum())
z("Z7", f"ppmd-probs.float16 complete: length = rows x {ZB}, >= {a.stichproben} "
        "samples with row sum ~ 1 (catches a placeholder file)",
  pg == soll and gut == len(summen),
  f"file {pg:,} B, expected {soll:,} ({n_tokens:,} x {ZB}); {len(summen)} samples, "
  f"{gut} with 0.9 < sum < 1.1; min {summen.min():.4f} max {summen.max():.4f}")

# ------------------------------------------------------------------- Z7b/Z7c
# Z7b alignment: row i predicts token i+1, measured on the table itself
#     against the ppmd-bytes of the package.
# Z7c five fixed positions: row sha against the ZERO row (2V zero bytes) and
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
            fh.seek(int(i) * ZB); v = np.frombuffer(fh.read(ZB), dtype=np.float16).astype(np.float64)
            fv.seek(int(i) * ZB); w = np.frombuffer(fv.read(ZB), dtype=np.float16).astype(np.float64)
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
    null = hashlib.sha256(b"\0" * ZB).hexdigest()
    fuenf = [0, n_tokens // 4, n_tokens // 2, 3 * n_tokens // 4, n_tokens - 1]
    tab = []
    with open(a.prioren, "rb") as fh, open(a.prioren_vgl, "rb") as fv:
        for i in fuenf:
            fh.seek(i * ZB); r20 = fh.read(ZB); fv.seek(i * ZB); r25 = fv.read(ZB)
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
    # text form: the title line of page 0 lies in the short zone-3 run behind
    # the slots (the gate runs on over it); the gate masks it by design
    in_zone = np.zeros(len(ist_titel), bool)
    sv = f["src"] >= 0
    in_zone[sv] = gate[np.clip(f["src"][sv], 0, len(gate) - 1)] != 0
    # rows of the forced title prefixes: title bytes 0..L-2 of a page in a
    # prefix range (they predict the forced bytes 1..L-1)
    zwz = np.zeros(obergrenze, bool)
    zwz[ZW_ZEILEN[ZW_ZEILEN < obergrenze]] = True
    ist_zwang = ist_titel & zwz[np.clip(f["src"], 0, obergrenze - 1)]
    del zwz
    maskiert = (fl & 2) != 0
    tz = int((ist_titel & in_zone).sum())
    tm = int((ist_titel & ~in_zone & ~ist_zwang & maskiert).sum())
    zw_zeilen = int(ist_zwang.sum())
    zw_maskiert = int((ist_zwang & maskiert).sum())
    # expected: L-1 rows for every forced title of the package
    _zi = np.flatnonzero(ist_zwang)
    _soll = int((ZW_LEN - 1).sum())
    # the tokens of the forced rows are the table bytes (catches a wrong page
    # or position mapping)
    _tab = {}
    for b, byts in enumerate(TF["zwang"]["bytes"]):
        _tab[b] = [int(LUT[x]) for x in byts]
    _soll_tok = np.zeros(obergrenze, np.int16) - 1
    for s0, L, b in zip(ZW_START, ZW_LEN, ZW_B):
        if s0 + L - 1 <= obergrenze:
            _soll_tok[s0:s0 + L - 1] = _tab[int(b)][:L - 1]
    zw_tok_falsch = int((f["tok"][_zi] != _soll_tok[f["src"][_zi]]).sum()) if len(_zi) else 0
    del _zi, _soll_tok
    z("Z9", "title rows carry NO mask bit (the title is predicted), except in a gated zone and on the "
            "forced title prefixes (rows of prefix bytes 0..L-2 masked, all of them, tokens = table bytes)",
      tm == 0 and int(ist_titel.sum()) > 0 and zw_maskiert == zw_zeilen == _soll and zw_tok_falsch == 0,
      f"title rows in the dump {int(ist_titel.sum()):,}, of these in a gated zone (page 0 of the text form) {tz}, "
      f"rows of forced prefixes {zw_zeilen:,} (expected {_soll:,} = sum of L-1 over {len(ZW):,} forced titles), "
      f"masked {zw_maskiert:,}, token != table byte {zw_tok_falsch}; "
      f"masked elsewhere {tm} (expected 0); "
      f"rows added to the scored set = {int((ist_titel & ~maskiert).sum()):,}")

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
               "zeilen_im_dump": int(n), "dokumente": int(len(rows) - 1),
               # the text-form facts of this check; paket_manifest.py moves
               # them into MANIFEST.json (field textform) for the acceptance
               "textform": TF},
              open(a.json, "w"), indent=1, sort_keys=True)
sys.exit(0 if alle else 1)
