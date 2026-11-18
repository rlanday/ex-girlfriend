#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
"""Acceptance of a training package on the training side, against its manifest

    [TRAIN_TREE=<training tree>] abnahme_paket.py <package dir> [--ohne-prioren-sha]
                                                  [--ohne-speicherpruefung]

The package directory holds the tables, the priors (ppmd-probs.float16,
ppmd-bytes.uint8) and MANIFEST.json (paket_manifest.py).  Every expected value
comes from the manifest of this package; nothing is written in here for a
particular package form.  Checked, in this order (a finding does not stop the
run, the findings are counted at the end):

  A  integrity: size and sha256 of every table against the manifest (the
     priors file too, unless --ohne-prioren-sha; that is the check after a
     transfer)
  B  manifest against the files: n_tokens = ppmd-bytes = last article
     boundary = zone mask, priors = n_tokens x 2V (V of the geometry), the arms and the geometry
     of dump-meta.json = those of the manifest, the arms carry a text form arm,
     prior0 sha256
  C  to G  the Z table of package_checks.py with the cut and the text-form
     facts of the manifest (Z1 document span < 2^17, Z2 cut behind the
     metadata block, Z3 tables, Z4 probe 0 and positions without a gradient,
     Z5 prompt masks with the forced title prefixes, Z6 zone mask, Z7 priors
     complete and aligned, Z9 title rows with the page table seiten.tsv of the
     package).  The Z table holds the whole dump in memory: about 100 B per
     token (44.1 GiB measured on 97.6 % of the stream); the acceptance stops
     before it when less memory is available (--ohne-speicherpruefung skips
     that check)
  H  the interface check of the trainer (`python -m tft.dataset check`),
     when TRAIN_TREE is set
  Z8 the manifest fields: tree commit and content hash, the build lines with
     sha256, the arms, the inputs with sha256, n_tokens, the tables, the
     stamps of the runs and stages (their tool sha256 = the tools named)

Exit 0 = accepted; the last line says so.
"""
import hashlib, json, os, subprocess, sys

import numpy as np

PK = sys.argv[1].rstrip("/")
OHNE_P = "--ohne-prioren-sha" in sys.argv
OHNE_SP = "--ohne-speicherpruefung" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
fehler = []


def sage(ok, text):
    print(("  OK   " if ok else "  FAIL ") + text, flush=True)
    if not ok:
        fehler.append(text)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


M = json.load(open(os.path.join(PK, "MANIFEST.json"), encoding="utf-8"))
S = M["schnitt"]
NT, TS = int(S["n_tokens"]), int(S["token_start"])
print(f"Acceptance {PK}: n_tokens {NT:,}, token_start {TS:,}, tree {M['baum']['commit'][:12]}"
      + (f", PROBE on {S['probe_praefix_bytes']:,} B" if S.get("probe_praefix_bytes") else ""))

print("A  integrity")
tabs = dict(M["tabellen_paket"])
tabs.update({k: v for k, v in M["tabellen_prioren"].items()})
for f, soll in sorted(tabs.items()):
    p = os.path.join(PK, f)
    if f in ("pkg.tar", "SHA256SUMS_klein", "ppmd-quelle.u8"):
        continue                    # not part of the package directory on the training host
    if not os.path.exists(p):
        sage(False, f"{f} missing")
        continue
    sage(os.path.getsize(p) == soll["bytes"], f"{f} size {os.path.getsize(p):,} (manifest {soll['bytes']:,})")
    if f == "ppmd-probs.float16" and OHNE_P:
        print("  --   ppmd-probs.float16: sha256 not computed (--ohne-prioren-sha)")
        continue
    sage(sha(p) == soll["sha256"], f"{f} sha256 {soll['sha256'][:16]}")

print("B  manifest against the files")
pb = os.path.getsize(os.path.join(PK, "ppmd-bytes.uint8"))
ab = np.fromfile(os.path.join(PK, "article-boundaries.int32"), dtype=np.int32)
zm = os.path.getsize(os.path.join(PK, "zonenmaske.u8"))
pp = os.path.getsize(os.path.join(PK, "ppmd-probs.float16"))
sage(pb == NT, f"ppmd-bytes {pb:,} = n_tokens {NT:,}")
sage(int(ab[-1]) == NT, f"last article boundary {int(ab[-1]):,} = n_tokens")
sage(zm == NT, f"zone mask {zm:,} = n_tokens")
ZB = 2 * int(M["geometrie"]["v"])
sage(pp == NT * ZB, f"priors {pp:,} = n_tokens x {ZB}")
sage(S["strombytes_mit_kopf"] - TS == NT, "coded bytes - token_start = n_tokens")
dm = json.load(open(os.path.join(PK, "dump-meta.json")))
sage(dm.get("arme") == M["arme"], f"arms of dump-meta.json = manifest ({len(M['arme'])} arms)")
sage(any(x.startswith("textform") for x in M["arme"]), f"a text form arm in the list ({[x for x in M['arme'] if x.startswith('textform')]})")
G = M["geometrie"]
for k in ("v", "nsent", "x", "fenster"):
    sage(int(dm.get(k)) == int(G[k]), f"{k} dump-meta {dm.get(k)} = end line {G[k]}")
sage(bool(dm.get("x0row")) == bool(G["x0row"]) and bool(dm.get("redirmerge")) == bool(G["redirmerge"]),
     "x0row/redirmerge of dump-meta = end line")
sage(sha(os.path.join(PK, "z3-prior0.float16")) == M["prior0_sha256"], f"prior0 sha256 {str(M['prior0_sha256'])[:16]}")
sage(dm.get("n_tokens") == NT, f"dump-meta n_tokens {dm.get('n_tokens')} = n_tokens")

print("C-G  Z table (package_checks.py)")
# memory: the Z table holds the dump; about 100 B per token plus 2 GiB
# (measured 46,258,328 KiB at 479.9 million tokens, 97.6 % of the stream)
bedarf = -(-(NT * 100 + (2 << 30)) // (1 << 30))      # GiB, rounded up
try:
    frei = int(next(z.split()[1] for z in open("/proc/meminfo") if z.startswith("MemAvailable:"))) >> 20
except (OSError, StopIteration, ValueError):
    frei = -1
    print("  --   /proc/meminfo not readable: the memory of the Z table is not checked", file=sys.stderr)
if frei >= 0 and frei < bedarf and not OHNE_SP:
    sage(False, f"memory for the Z table: {frei} GiB available, about {bedarf} GiB needed "
                f"(free memory, or --ohne-speicherpruefung)")
    print(f"ACCEPTANCE FAILED: memory ({frei} GiB available, about {bedarf} GiB needed for the Z table)")
    sys.exit(1)
print(f"  memory: {frei} GiB available, about {bedarf} GiB needed by the Z table")
cmd = [sys.executable, os.path.join(HERE, "package_checks.py"), PK, "--prioren", os.path.join(PK, "ppmd-probs.float16"),
       "--ppmd-bytes", os.path.join(PK, "ppmd-bytes.uint8"), "--gate", os.path.join(PK, "zonenmaske.u8"),
       "--strombytes", str(S["strombytes_mit_kopf"]), "--token-start", str(TS), "--stichproben", "64",
       "--textform", os.path.join(PK, "MANIFEST.json")]
# Z9 (title rows) needs the page table; packages of manifest version 2 carry it
SEITEN = os.path.join(PK, "seiten.tsv")
if os.path.exists(SEITEN):
    cmd += ["--grenzen", SEITEN, "--token-start-strom", str(TS - 6)]
elif int(M.get("version", 1)) >= 2:
    sage(False, "seiten.tsv missing: the page table of a version-2 package (Z9 not checkable)")
else:
    print("  --   no seiten.tsv (package of manifest version 1): Z9 not checked")
r = subprocess.run(cmd, capture_output=True, text=True)
rot = [z for z in r.stdout.splitlines() if z.startswith("| Z") and z.rstrip().endswith("RED |")]
print("\n".join("       " + z for z in r.stdout.splitlines() if z.startswith("Z") and "  " in z[:6]))
sage(r.returncode == 0, f"Z table rc={r.returncode}{', red: ' + ' '.join(z.split('|')[1].strip() for z in rot) if rot else ''}")
if os.path.exists(SEITEN):
    sage(any(z.startswith("| Z9 ") for z in r.stdout.splitlines()), "Z9 (title rows) ran with the page table of the package")

print("H  interface check of the trainer")
if os.environ.get("TRAIN_TREE"):
    r = subprocess.run([sys.executable, "-m", "tft.dataset", "check", PK], cwd=os.environ["TRAIN_TREE"],
                       capture_output=True, text=True)
    sage(r.returncode == 0 and "package matches the training-data interface" in r.stdout,
         f"tft.dataset check rc={r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ''}")
else:
    print("  --   TRAIN_TREE not set: skipped")

print("Z8 manifest fields")
sage(M["baum"]["commit"] not in ("", "unknown") and len(M["baum"]["src_inhalt_sha256"]) == 64,
     f"tree commit {M['baum']['commit'][:12]} and content sha256 {M['baum']['src_inhalt_sha256'][:16]}")
for n in ("stream", "dump", "priors", "end", "bed"):
    z = M["zeilen"].get(n)
    sage(bool(z and z.get("zeile") and len(z.get("sha256", "")) == 64), f"build line '{n}' with text and sha256")
for n in ("enwik9", "dict", "order", "canonical_stream", "vs3", "coder_stream", "bed"):
    e = M["eingaenge"].get(n)
    sage(bool(e and len(e.get("sha256", "")) == 64 and e.get("bytes", 0) > 0), f"input '{n}' with size and sha256")
sage(len(M.get("werkzeuge", {})) >= 4, f"tool binaries {sorted(M.get('werkzeuge', {}))}")
sage(bool(M.get("arme")) and bool(M.get("alias_form")), "arms and alias form")
ST = M.get("stempel") or {}
sage(sorted(ST) == ["bed", "dump", "priors", "stage3", "stage4", "stream"],
     f"stamps of the runs and stages {sorted(ST)}")
wz = [z.split() for v in ST.values() for z in v if z.startswith("werkzeug ")]
sage(bool(wz) and all(M["werkzeuge"].get(t[1]) == t[2] for t in wz),
     f"tool sha256 of the run stamps = tools of the manifest ({len(wz)} entries)")
TFM = M.get("textform") or {}
sage(bool(TFM.get("slots")) and "zwang" in TFM and "titel" in TFM["zwang"],
     f"text-form facts: slots {TFM.get('slots')}, forced title prefixes "
     f"{len((TFM.get('zwang') or {}).get('titel', []))} titles")

print()
if fehler:
    print(f"ACCEPTANCE FAILED: {len(fehler)} finding(s)")
    for f in fehler:
        print("   * " + f)
    sys.exit(1)
print("ACCEPTANCE PASSED (A-H, Z8)")
