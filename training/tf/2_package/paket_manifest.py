#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Manifest of a training package (assurance Z8), written by build_package.sh

    paket_manifest.py --pkg <dir> --priors <dir> --work <dir> --tree-commit C --tree-sha S
                      --tools <dir> --arme a,b,... --geometrie <arms_end.txt>
                      --token-start N --n-tokens N --skip N --strombytes N --praefix N
                      --zeile name=<line file> ... --eingang name=<file> ...
                      --json <run json> ... --zeilenvgl <zeilen_vgl.json>
                      --ppmd <ppmd lines> --slots "<slot content line>"
                      --stempel name=<stamp file> ... [--trainerbaum <sha256>]

Writes <pkg>/MANIFEST.json: where the package comes from (tree commit and
content hash, the build lines as text and sha256, the tool binaries), what it
is (arms and geometry of the end line, cut, n_tokens, prior0), the inputs with
size and sha256 (enwik9, dictionary, order, canonical stream, .vs3, coder
stream, bed), every table of the package and of the priors with size and
sha256, and the counts of the runs and checks.  abnahme_paket.py checks a
package against this file.  The priors file is not hashed again: its sha256
comes from the probe step (PROBS_SHA256).

`--stempel` names the stamps of the runs and stages (build_package.sh): they
go into the manifest, and every tool sha256 a run stamp names must be the one
of the tool binaries listed here, else the manifest is not written (a package
whose runs came from other binaries than the ones it names is refused).  The
text-form facts of the Z table (field `textform` of ztable.json: slot sizes,
title of page 0, forced title prefixes) become the field `textform`, which
abnahme_paket.py hands to package_checks.py on the training host.  Version 2
of the manifest: the package carries its page table (seiten.tsv, the title
rows of Z9) and the manifest names the content sha256 of the training tree
the build checked the package with (`--trainerbaum`).
"""
import argparse, hashlib, json, os, sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def kv(p):
    d = {}
    for z in open(p, encoding="utf-8"):
        if "=" in z:
            k, v = z.rstrip("\n").split("=", 1)
            d[k] = v
    return d


def feat(p):
    """NEIF_LIST of a `make features` output file, or None"""
    if not os.path.exists(p):
        return None
    for z in open(p, encoding="utf-8", errors="replace"):
        if z.startswith("NEIF_LIST"):
            return z.split("=", 1)[1].strip()
    return None


ap = argparse.ArgumentParser()
for k in ("pkg", "priors", "work", "tree-commit", "tree-sha", "tools", "arme", "geometrie", "zeilenvgl", "ppmd", "slots"):
    ap.add_argument("--" + k, required=True)
for k in ("token-start", "n-tokens", "skip", "strombytes", "praefix"):
    ap.add_argument("--" + k, type=int, required=True)
ap.add_argument("--zeile", action="append", default=[])
ap.add_argument("--eingang", action="append", default=[])
ap.add_argument("--json", action="append", default=[])
ap.add_argument("--stempel", action="append", default=[])
ap.add_argument("--trainerbaum", default=None)
a = ap.parse_args()

m = {"manifest": "training package, text form", "version": 2}
m["trainerbaum_sha256"] = a.trainerbaum
m["baum"] = {"commit": a.tree_commit, "src_inhalt_sha256": a.tree_sha}
m["zeilen"] = {}
for z in a.zeile:
    n, p = z.split("=", 1)
    m["zeilen"][n] = {"datei": os.path.abspath(p), "sha256": sha(p),
                      "zeile": open(p, encoding="utf-8").read().replace("\r", " ").replace("\n", " ").strip(),
                      "neif_list": feat(os.path.join(a.tools, f"features_{n}.txt"))}
m["werkzeuge"] = {f: sha(os.path.join(a.tools, f)) for f in sorted(os.listdir(a.tools))
                  if f.startswith("cmix_") and os.path.isfile(os.path.join(a.tools, f))}
g = kv(a.geometrie)
m["arme"] = [s for s in a.arme.split(",") if s]
m["geometrie"] = {k: (int(g[k]) if g.get(k, "").lstrip("-").isdigit() else g.get(k)) for k in
                  ("v", "nsent", "x", "dslot", "x0row", "redirmerge", "fenster", "nl", "prior0_set")}
m["geometrie"]["prior0_set_der_endzeile"] = m["geometrie"].pop("prior0_set")
m["schnitt"] = {"token_start": a.token_start, "n_tokens": a.n_tokens, "skip": a.skip,
                "strombytes_mit_kopf": a.strombytes, "probe_praefix_bytes": a.praefix or None}
m["textanfang"] = a.slots
m["ppmd_zeilen"] = [z.rstrip("\n") for z in open(a.ppmd, encoding="utf-8")]
m["eingaenge"] = {}
for e in a.eingang:
    n, p = e.split("=", 1)
    m["eingaenge"][n] = {"pfad": os.path.abspath(p), "bytes": os.path.getsize(p), "sha256": sha(p)}
tab = {}
for f in sorted(os.listdir(a.pkg)):
    p = os.path.join(a.pkg, f)
    if os.path.isfile(p) and f not in ("MANIFEST.json",):
        tab[f] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
pr = {}
for f in ("ppmd-bytes.uint8", "z3-prior0.float16", "ppmd-probs.float16"):
    p = os.path.join(a.priors, f)
    if os.path.exists(p):
        if f == "ppmd-probs.float16":
            pr[f] = {"bytes": os.path.getsize(p), "sha256": open(os.path.join(a.priors, "PROBS_SHA256")).read().strip()}
        else:
            pr[f] = {"bytes": os.path.getsize(p), "sha256": sha(p)}
m["tabellen_paket"] = tab
m["tabellen_prioren"] = pr
m["prior0_sha256"] = tab.get("z3-prior0.float16", {}).get("sha256")
zahlen = {}
for j in a.json:
    try:
        d = json.load(open(j))
    except Exception as ex:  # noqa: BLE001
        sys.exit(f"cannot read {j}: {ex}")
    zahlen[os.path.basename(j)] = d
zv = json.load(open(a.zeilenvgl))
m["zeilenvergleich"] = {"endzeile": zv.get("endzeile"), "quittiert": zv.get("quittung"),
                        "zeilen": {os.path.basename(k): {"offen": v.get("offen"),
                                                         "abweichungen": [(x["schluessel"], x["klasse"], x["zeile"], x["ende"])
                                                                          for x in v.get("abweichungen", [])
                                                                          if x["klasse"] in ("form", "form-k")]}
                                   for k, v in zv.get("zeilen", {}).items()}}
zt = zahlen.get("ztable.json", {})
m["textform"] = zt.pop("textform", None)
if not m["textform"]:
    sys.exit("ztable.json carries no field textform (package_checks.py of the text form)")
m["stempel"] = {}
for e in a.stempel:
    n, p = e.split("=", 1)
    zeilen = [z.rstrip("\n") for z in open(p, encoding="utf-8")]
    m["stempel"][n] = zeilen
    for z in zeilen:
        t = z.split()
        if t[:1] == ["werkzeug"] and m["werkzeuge"].get(t[1]) != t[2]:
            sys.exit(f"stamp {n}: the run used {t[1]} {t[2][:16]}, the tools of this call are "
                     f"{str(m['werkzeuge'].get(t[1]))[:16]}: the package is not from these tools")
z5 = zt.get("z5", {})
m["alias_form"] = ("NEU (alias and author rows: src = -1, bit5; masked except the last row of a block)"
                   if z5.get("alias_zeilen", 0) > 0 else "no alias rows")
m["container"] = None
m["container_hinweis"] = ("the package has no weight container: the trainer writes arms, prior0 sha and geometry of "
                          "this manifest into the container it exports, the coder of the end line checks them")
m["zahlen"] = zahlen
out = os.path.join(a.pkg, "MANIFEST.json")
json.dump(m, open(out, "w", encoding="utf-8"), indent=1, sort_keys=True, ensure_ascii=False)
print(f"written {out}: {len(tab)} package tables, {len(pr)} prior tables, arms {len(m['arme'])}")
