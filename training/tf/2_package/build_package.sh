#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
#
# Recipe of the training package of the shipped transformer (pkgvoll_v12k).
#
# It runs the commands of the chain that produced the package, in the same
# order, on one machine and without our hosts: coder stream -> dump run ->
# cut at the end of the metadata block -> mk_pkgdump (29 arms, window 2048) ->
# tables -> ppmd-bytes and prior0 -> prior run -> trimming of the priors ->
# checks.  The original ran the dump part and the prior part on two machines
# in parallel; each step names the script of the chain it comes from
# (lauf12.sh, kette12_d.sh, montage12.sh, kette12_p4.sh, kette12_p4_fort.sh).
# Steps that only served our machines (RAM admission gate, HTTP transfer of
# the package between hosts, waiting for disk space, the manifest writer with
# host names) are left out; each place says so.
#
# The coder-side runs use the three tools of ../1_dump (the source of this
# entry with trainmodes.patch, built by build_dump_tools.sh); they give the
# outputs of the development tree that ran the chain (../1_dump/README.md,
# section 4).
#
# Needs a 64-bit Linux host with the toolchain of build.sh (clang++-18,
# ld.lld-18, make) and patch, Python 3 with NumPy (and torch for the loader
# checks), about 250 GB free for the priors (231 GB before trimming) and, by
# estimate, 35 GB for the rest.
#
# Variables (paths):
#   ENTRY          the unpacked source package of the entry (default: the
#                  directory that holds training/)
#   ENWIK9         enwik9
#   DICT           dictionary of the entry (default $ENTRY/dictionary/english.dic,
#                  sha256 0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535)
#   ORDER          article order residual of the training stream (default
#                  $ENTRY/src/readalike_prepr/data/new_article_order,
#                  sha256 c9f8b1001bfbc041a087eab65e5caa609eb87bb8ecf297cfcdbcdaffa7a1dd19)
#   VOCAB_BITMAP   the 208-value byte set, 32 B (default: vocab-bitmap.u8 next to
#                  this script, sha256 d257fd4a869ca4f61d67bb487f4276517a4d81540098f76828403b298b29a70c)
#   TRAIN_TREE     the training tree, assembled as in training/README.md section 1
#   PYLIB          optional extra Python package directory for the loader checks
#   EARLIER_BED    optional: coder stream of the earlier package form, for the
#                  self-test of pages.py (skipped when unset)
#   REF_Z5_JSON    optional: Z-table JSON of an earlier package, for row Z5b
#   WORK           work directory (default ./work)
#   PRIORS         directory of the prior table (default $WORK/priors)
#   PKG            the training package (default $WORK/pkgvoll_v12k)
#   DEVTREE, DEVTREE_ENDBAU  optional, only for prior0_diff.py (a new prior0
#                  table as a change of the development tree, commit 786964b1,
#                  and its scripts/endbau.sh; not in this package)
#   P0_TEXT        description line for prior0_diff.py (comments only)
set -uo pipefail
ulimit -c 0
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAINING="${TRAINING:-$(cd "$HERE/../.." && pwd)}"
ENTRY="${ENTRY:-$(cd "$TRAINING/.." && pwd)}"
: "${ENWIK9:?ENWIK9 missing}"
DICT="${DICT:-$ENTRY/dictionary/english.dic}"
ORDER="${ORDER:-$ENTRY/src/readalike_prepr/data/new_article_order}"
VOCAB_BITMAP="${VOCAB_BITMAP:-$HERE/vocab-bitmap.u8}"
: "${TRAIN_TREE:?TRAIN_TREE missing}"
[ "$(sha256sum "$VOCAB_BITMAP" | cut -d" " -f1)" = d257fd4a869ca4f61d67bb487f4276517a4d81540098f76828403b298b29a70c ] \
  || { echo "VOCAB_BITMAP is not the byte set of the training package" >&2; exit 2; }
export VOCAB_BITMAP TRAIN_TREE
[ -n "${PYLIB:-}" ] && export PYLIB
WORK="$(mkdir -p "${WORK:-./work}" && cd "${WORK:-./work}" && pwd)"
PRIORS="${PRIORS:-$WORK/priors}"
PKG="${PKG:-$WORK/pkgvoll_v12k}"
P0_TEXT="${P0_TEXT:-pkgvoll_v12k: prior0 of the training package of the shipped transformer}"
FEAT="$TRAINING/tf/1_dump"
mkdir -p "$WORK/logs" "$WORK/runs"
L="$WORK/logs/build_package.log"

# expected values of the original run
SHA_BK=8e0b264a777cb56a6be48c58002450c4711b1cebf3ffcb0129a32ef34273c3ba   # coder stream, packed
SHA_BU=e556d3f837c66c69e0a4f3a1e972b05bdb034633f6a8cadd6d91ef9c34135111   # coder stream, unpacked bed
SHA_DUMP=58fc6c74c09400192c28cf3608e6eeb0387ba4f40b4255d39dbc8ac5e0bf66a0 # paket-zeilen.bin of the dump run
SHA_P0_SET9=591218150f09541909a980fca4c23fa044556bc1dcd2a5f592010103070deed6

say()  { echo "$(date '+%F %T') $*" | tee -a "$L"; }
fail() { say "!!! $*"; exit "${2:-1}"; }
sha()  { sha256sum "$1" | cut -d' ' -f1; }

# ---- 0. the three tools: a copy of the entry's source with trainmodes.patch,
#         plain builds (the chain built the development tree, see ../1_dump)
T="$WORK/tools"
if [ ! -x "$T/cmix_priors" ]; then
  bash "$FEAT/build_dump_tools.sh" "$ENTRY" "$T" > "$WORK/logs/build_tools.log" 2>&1 || fail "build_dump_tools.sh"
fi
say "tools: $(cd "$T" && sha256sum cmix_stream cmix_dump cmix_priors | cut -c1-16 | tr '\n' ' ')"

# ---- 1. coder stream (stream run of the chain): the packed stream
#         <archive>.cmix.temp and <archive>.cmix.temp.vorpack, the stream
#         before the metadata block is packed
S="$WORK/stream"
if [ ! -s "$S/archiv.cmix.temp.vorpack" ]; then
  mkdir -p "$S"
  ( cd "$S" && cp "$T/cmix_stream" . && cp "$DICT" .dict && cp "$ORDER" .new_article_order \
    && ./cmix_stream -e "$ENWIK9" archiv ) > "$WORK/logs/stream.log" 2>&1 || fail "stream run"
fi
BK="$S/archiv.cmix.temp"; BU="$S/archiv.cmix.temp.vorpack"
[ "$(sha "$BK")" = "$SHA_BK" ] || say "note: coder stream sha differs from the original run"
[ "$(sha "$BU")" = "$SHA_BU" ] || say "note: unpacked bed sha differs from the original run"
say "stream: $(stat -c%s "$BK") B packed, $(stat -c%s "$BU") B unpacked"

# ---- one coder run in "-n" mode (lauf12.sh).  The run reads the unpacked bed;
#      it is complete when the coder's own count of coded stream bytes equals
#      the stream size + 6 (segment header).  The tools write into the run
#      directory; the two prior files are symlinks into PRIOR_DIR, created
#      before the run.  The original started the binary through the RAM
#      admission gate of our machines; left out here.
lauf() {   # lauf <tool> <bed> <run name>; PRIOR_DIR and SOLL from the caller
  local tool=$1 bett=$2 d="$WORK/runs/$3"
  [ -e "$d/.FERTIG" ] && { echo "== $3 already complete, nothing done"; echo "== RC=0 (earlier run)"; return 0; }
  mkdir -p "$d" "$PRIOR_DIR"; cd "$d" || return 1
  cp "$T/cmix_$tool" .
  cp "$DICT" .dict
  if [ "$tool" = priors ]; then
    ln -sf "$PRIOR_DIR/ppmd-probs.float16" ppmd-probs.float16
    ln -sf "$PRIOR_DIR/ppmd-bytes.uint8" ppmd-bytes.uint8
  fi
  # the output is a symlink to /dev/null INSIDE the run directory: the coder
  # derives its temp file name from the output name, and two runs writing to
  # /dev/null would share /dev/null.cmix.temp
  ln -sf /dev/null aus
  echo "== START $(date +%F' '%T) tool=$tool bed=$bett dir=$d prior=$PRIOR_DIR"
  /usr/bin/time -v nice -n 5 "./cmix_$tool" -n "$bett" ./aus 2> stderr.txt
  RC=$?
  echo "== RC=$RC $(date +%F' '%T)"
  rm -f aus.cmix.temp ppm.page
  IST=$(tr '\r' '\n' < stderr.txt | sed -n 's/^coded stream bytes \([0-9]*\)$/\1/p' | tail -1)
  if [ "$IST" = "$SOLL" ]; then echo "== COMPLETE stream=$IST (expected $SOLL)"; date > "$d/.FERTIG"
  else echo "== INCOMPLETE stream=$IST expected $SOLL: RUN INVALID"; fi
}
SOLL=$(( $(stat -c%s "$BK") + 6 ))

# ---- 2. dump run (kette12_d.sh)
D="$WORK/runs/dump"
( PRIOR_DIR="$D" lauf dump "$BU" dump ) > "$WORK/logs/dump.log" 2>&1
say "dump: $(grep -a '^== \(RC\|COMPLETE\|INCOMPLETE\)' "$WORK/logs/dump.log" | tr '\n' ' ')"
[ -e "$D/.FERTIG" ] || fail "dump run not complete" 5
[ "$(sed -n 's/^== RC=\([0-9]*\).*/\1/p' "$WORK/logs/dump.log" | tail -1)" = 0 ] || fail "dump run RC" 5
python3 "$HERE/dump_check.py" "$D" "$BK" > "$WORK/logs/dump_check_dump.json" 2>&1 || fail "file check of the dump run" 5
[ "$(sha "$D/paket-zeilen.bin")" = "$SHA_DUMP" ] || say "note: dump sha differs from the original run"
# (the chain also checked self-report lines of the development tree here:
#  redirect table and author pairing counts, block counts of the prompt
#  automaton; the tools do not print them)
# token_start = end of the metadata block (zone 3) + 1, in dump coordinates
TS=$(python3 - "$D/zonenmaske.u8" <<'PY'
import sys, numpy as np
m = np.fromfile(sys.argv[1], dtype=np.uint8); i = np.flatnonzero(m == 3)
assert len(i) and i[-1]-i[0]+1 == len(i); print(int(i[-1]) + 1)
PY
)
NT=$(( SOLL - TS ))
say "token_start $TS, n_tokens $NT"
if [ -n "${EARLIER_BED:-}" ]; then
  python3 "$HERE/pages.py" --selbstprobe "$EARLIER_BED" >> "$L" 2>&1 || fail "pages.py self-test" 6
else
  say "pages.py self-test skipped (EARLIER_BED unset)"
fi
python3 "$HERE/pages.py" "$BK" "$WORK/pages.tsv" "$TS" >> "$L" 2>&1 || fail "page table" 6

# ---- 3. the package from the dump (montage12.sh)
W="$WORK/montage"; mkdir -p "$W" "$PKG"
[ -e "$PKG/zeilen.dump" ] && fail "zeilen.dump already in $PKG" 9
# 3.1 cut point = end of the metadata block + 1
python3 - "$D/zonenmaske.u8" <<'PY' | tee "$W/cut.txt"
import sys, numpy as np
m = np.fromfile(sys.argv[1], dtype=np.uint8)
h = np.bincount(m, minlength=6)
print("# histogram", {i: int(h[i]) for i in range(6) if h[i]})
assert int(h[5]) == 0
i = np.flatnonzero(m == 3); a, b = int(i[0]), int(i[-1]); assert b - a + 1 == len(i)
print(f"# metadata block {a:,}..{b:,} ({len(i):,} B), stream bytes {len(m):,}, n_tokens {len(m)-(b+1):,}")
print(b + 1)
PY
SCHNITT=$(tail -1 "$W/cut.txt")
NTOK=$(( $(stat -c%s "$D/zonenmaske.u8") - SCHNITT ))
[ "$SCHNITT" = "$TS" ] && [ "$NTOK" = "$NT" ] || fail "token_start/n_tokens differ" 2
# 3.2 cut the zone mask
python3 - "$D/zonenmaske.u8" "$SCHNITT" "$W/gate_cut.u8" <<'PY'
import sys, numpy as np
m = np.fromfile(sys.argv[1], dtype=np.uint8); g = m[int(sys.argv[2]):]; g.tofile(sys.argv[3])
h = {int(v): int(c) for v, c in zip(*np.unique(g, return_counts=True))}
print(f"cut {len(g):,} B, histogram {h}"); assert set(h) <= {0}
PY
# 3.3 skip = dump rows in front of the first stream row of the package
SKIP=$(python3 - "$D/paket-zeilen.bin" "$SCHNITT" <<'PY'
import sys, numpy as np
roh = np.memmap(sys.argv[1], dtype=np.uint8, mode='r'); n = len(roh)//12
src = np.array(roh[:n*12].reshape(n,12)[:, :4]).view(np.int32).ravel()
print(int(np.flatnonzero(src >= int(sys.argv[2]))[0]) - 1)
PY
)
say "skip $SKIP"
# 3.4 mk_pkgdump (the OOM killer is told to take this process first)
( echo 1000 > /proc/self/oom_score_adj
  exec /usr/bin/time -v nice -n 5 python3 "$HERE/mk_pkgdump.py" \
    "$D/paket-zeilen.bin" "$PKG" --gate "$W/gate_cut.u8" --n-tokens "$NTOK" \
    --skip "$SKIP" --shift "$SCHNITT" --v 208 --nsent 8 --x 2 --x0row --redirmerge \
    --arme syn,alias,a2,a2n,sent,z3,x0row,redirmerge,aliasprompt,autor,meta,metavorn,metasent,rd,titelersatz2,subdedup,xdedup,tfonly,metaform3,frueh,redirfest,uid_erst,tsprog2,metakons,kapfrei1,aliastab1,kalblock1,c8tune1,leiter2_o25 \
    --fenster 2048 ) > "$WORK/logs/mk_pkgdump.log" 2>&1 || fail "mk_pkgdump" 3
[ -s "$PKG/zeilen.dump" ] || fail "no zeilen.dump" 3
# 3.5 title-lengths from the page table
python3 "$HERE/title_lengths.py" "$PKG/article-boundaries.int32" "$WORK/pages.tsv" \
  "$PKG/title-lengths.int32" --bett-off $(( SCHNITT - 6 )) >> "$L" 2>&1 || fail "title-lengths" 3
# 3.6 zone mask and byte set into the package
cp "$W/gate_cut.u8" "$PKG/zonenmaske.u8"
cp "$VOCAB_BITMAP" "$PKG/vocab-bitmap.u8"
# 3.7 ppmd-bytes of the package from the stream raster, prior0; dump stream rows against the stream
python3 "$HERE/ppmd_bytes.py" "$D/paket-zeilen.bin" "$SCHNITT" "$NTOK" "$PKG" "$BK" >> "$L" 2>&1 || fail "ppmd-bytes/prior0" 7
P0=$(sha "$PKG/z3-prior0.float16")
say "prior0 of the package $P0 $([ "$P0" = "$SHA_P0_SET9" ] && echo "(= set 9)" || echo "(NEW: a new set number is needed)")"
# (the chain also ran two diagnostic probes here that decode prompt groups with
#  a WRT tool of the development tree; they are not part of this package)
# 3.8 loader against the raw dump; the loader needs a priors file of the right
#     length, whose values this check does not read (a sparse placeholder)
truncate -s $(( $(stat -c%s "$PKG/ppmd-bytes.uint8") * 416 )) "$PKG/ppmd-probs.float16"
python3 "$HERE/loader_vs_dump.py" "$PKG" "$D/paket-zeilen.bin" "$TS" "$SKIP" > "$WORK/logs/loader_vs_dump.log" 2>&1 || fail "loader_vs_dump" 3
say "loader_vs_dump: $(tail -1 "$WORK/logs/loader_vs_dump.log")"
rm -f "$PKG/ppmd-probs.float16"
# 3.9 the package as one tar (sorted by name) and the checksums of the small files
( cd "$PKG" && tar -cf pkgv12k.tar --sort=name article-boundaries.int32 doc-last-ziel.int32 doc-rows.int64 \
    title-lengths.int32 z3-prior0.float16 dump-meta.json kennzahlen.json vocab-bitmap.u8 zonenmaske.u8 zeilen.dump \
  && sha256sum article-boundaries.int32 doc-last-ziel.int32 doc-rows.int64 title-lengths.int32 z3-prior0.float16 \
    dump-meta.json kennzahlen.json vocab-bitmap.u8 zonenmaske.u8 zeilen.dump ppmd-bytes.uint8 pkgv12k.tar > SHA256SUMS_klein ) \
  || fail "tar" 3
say "tar: $(grep pkgv12k.tar "$PKG/SHA256SUMS_klein")"
# (the chain served the tar and the checksums over HTTP to the prior machine; not needed here)

# ---- 4. prior run (kette12_p4.sh); the original waited for 250 GB of free disk
mkdir -p "$PRIORS"
F=$(df -k --output=avail "$PRIORS" | tail -1 | tr -d ' ')
[ "$F" -ge $(( 250 * 1024 * 1024 )) ] || say "warning: less than 250 GB free for the priors"
[ -e "$PRIORS/ppmd-probs.float16" ] && fail "$PRIORS/ppmd-probs.float16 exists" 7
D="$WORK/runs/priors"
( PRIOR_DIR="$PRIORS" lauf priors "$BU" priors ) > "$WORK/logs/priors.log" 2>&1
say "prior run: $(grep -a '^== \(RC\|COMPLETE\|INCOMPLETE\)' "$WORK/logs/priors.log" | tr '\n' ' ')"
[ -e "$D/.FERTIG" ] || fail "prior run not complete" 5
[ "$(sed -n 's/^== RC=\([0-9]*\).*/\1/p' "$WORK/logs/priors.log" | tail -1)" = 0 ] || fail "prior run RC" 5
# the PPMd restart behind the metadata block must report order 25 -> 25
PZ=$(tr '\r' '\n' < "$D/stderr.txt" | grep -aE "^ppmd: metablock end at model byte [0-9]+, order 25 -> 25" | tail -1)
[ -n "$PZ" ] || fail "PPMd restart line 'order 25 -> 25' missing" 5
python3 "$HERE/dump_check.py" "$D" "$BK" --nur-dateien > "$WORK/logs/dump_check_priors.json" 2>&1 || fail "file check of the prior run" 5
PB=$(stat -c%s "$PRIORS/ppmd-bytes.uint8"); PP=$(stat -c%s "$PRIORS/ppmd-probs.float16")
[ "$PP" = "$(( PB * 416 ))" ] || fail "raw priors $PP B != ppmd-bytes $PB x 416" 5

# ---- 5. trimming: the raw raster has one row per coder step (stream bytes plus
#         bytes injected through all models); the package keeps the stream rows
#         behind the metadata block (kette12_p4.sh)
O="$WORK/trim"; rm -rf "$O"; mkdir -p "$O"
NSTROM=$(stat -c%s "$D/zonenmaske.u8")
SCHNITT=$(python3 - "$D/zonenmaske.u8" <<'PY'
import sys, numpy as np
m = np.fromfile(sys.argv[1], dtype=np.uint8); i = np.flatnonzero(m == 3)
assert len(i) and i[-1]-i[0]+1 == len(i), "metadata block is not ONE run"
print(int(i[-1]) + 1)
PY
)
say "cut (stream raster) $SCHNITT, stream bytes $NSTROM"
python3 "$HERE/prior_mask.py" "$D" "$NSTROM" "$O/okbmaske.u8" >> "$L" 2>&1 || fail "prior_mask" 6
# trim mask: everything up to the PPMd restart goes (the metadata block and the
# extra timestamp rows fed before the restart), behind it the injected rows
python3 - "$O/okbmaske.u8" "$SCHNITT" "$NSTROM" "$O/trimmaske.u8" "$PZ" "$PRIORS/ppmd-bytes.uint8" "$BK" "$O/ppmd-bytes_strom.uint8" <<'PY' 2>&1 | tee -a "$L"
import sys, os, re, numpy as np
mp, schnitt, nstrom, out, pz, rawp, koder, stp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
m = np.fromfile(mp, dtype=np.uint8)
null = np.flatnonzero(m == 0); assert len(null) == nstrom
grenze = int(null[schnitt])
g2 = int(re.search(r"model byte (\d+)", pz).group(1))
raw = np.fromfile(rawp, dtype=np.uint8)
extra = len(raw) - len(m)
print(f"coder rows {len(m):,}, raw rows {len(raw):,}, extra rows {extra:,}; mask boundary {grenze:,}, restart {g2:,}, difference {g2-grenze:,}")
assert grenze == schnitt, "injected rows before the cut, unexpected"
assert g2 - grenze == extra, "the extra rows are not all before the restart"
bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
menge = np.array([v for v in range(256) if (bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
lut = np.full(256, 255, np.uint8); lut[menge] = np.arange(len(menge), dtype=np.uint8)
bett = np.fromfile(koder, dtype=np.uint8); n = len(bett)
kopf = np.array([1] + [0x80 | ((n >> s) & 0x7F) for s in (28, 21, 14, 7, 0)], dtype=np.uint8)
soll = np.concatenate([lut[kopf], lut[bett]]); assert len(soll) == nstrom and (soll != 255).all()
hinten = raw[g2:]; mh = m[grenze:]
assert len(hinten) == len(mh)
st_h = hinten[mh == 0]
ok_h = bool(len(st_h) == nstrom - schnitt and (st_h == soll[schnitt:]).all())
print(f"stream rows behind the restart equal the stream: {ok_h} ({len(st_h):,})")
assert ok_h, "raw raster behind the restart does not match the coder stream"
np.concatenate([soll[:schnitt], st_h]).tofile(stp)
t = np.concatenate([np.ones(g2, np.uint8), mh]); t.tofile(out)
print(f"trim mask: {len(t):,} rows, remove {int(t.sum()):,}, keep {int((t==0).sum()):,}")
PY
[ "${PIPESTATUS[0]}" = 0 ] || fail "trim mask" 6
[ "$(stat -c%s "$PRIORS/ppmd-bytes.uint8")" = "$(stat -c%s "$O/trimmaske.u8")" ] || fail "raster length != mask length" 6
cp "$PRIORS/ppmd-bytes.uint8" "$O/ppmd-bytes_koderraster.uint8"
python3 "$HERE/trim.py" "$O/trimmaske.u8" "$O/ppmd-bytes_koderraster.uint8" "$PRIORS/ppmd-bytes_paket.uint8" 1 >> "$L" 2>&1 || fail "trim bytes" 6
nice -n 5 python3 "$HERE/trim_inplace.py" "$O/trimmaske.u8" "$PRIORS/ppmd-probs.float16" 416 >> "$L" 2>&1 || fail "trim_inplace" 6
rm -f "$PRIORS/ppmd-bytes.uint8"; mv "$PRIORS/ppmd-bytes_paket.uint8" "$PRIORS/ppmd-bytes.uint8"

# ---- 6. probes of the trimmed priors (kette12_p4.sh); prior0 is computed
#         again from the trimmed bytes
python3 - "$PRIORS" "$O" "$BK" "$WORK" "$SCHNITT" <<'PY' 2>&1 | tee -a "$L"
import sys, os, hashlib, json, numpy as np
Z, O, BETT, W, SCHNITT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    return h.hexdigest()
bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
menge = np.array([v for v in range(256) if (bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
lut = np.full(256, 255, np.uint8); lut[menge] = np.arange(len(menge), dtype=np.uint8)
res = {}
st = np.fromfile(O + "/ppmd-bytes_strom.uint8", dtype=np.uint8)
bett = np.fromfile(BETT, dtype=np.uint8); n = len(bett)
kopf = np.array([1] + [0x80 | ((n >> s) & 0x7F) for s in (28, 21, 14, 7, 0)], dtype=np.uint8)
res["strom_raster_gleich_segmentkopf_plus_koderstrom_v12k"] = bool(len(st) == n + 6 and (st[:6] == lut[kopf]).all() and (st[6:] == lut[bett]).all())
bt = np.fromfile(Z + "/ppmd-bytes.uint8", dtype=np.uint8); N = len(bt); res["n_tokens"] = N; res["token_start"] = SCHNITT
res["paket_gleich_strom_ab_schnitt"] = bool((bt == st[SCHNITT:]).all())
c = np.bincount(bt, minlength=256)[:208].astype(np.float64)
p0 = ((c + 1.0) / (c.sum() + 208.0)).astype(np.float16); p0.tofile(Z + "/z3-prior0.float16")
res["prior0_sha"] = sha(Z + "/z3-prior0.float16")
res["prior0_gleich_591218_set9"] = res["prior0_sha"] == "591218150f09541909a980fca4c23fa044556bc1dcd2a5f592010103070deed6"
res["ppmd_bytes_sha"] = sha(Z + "/ppmd-bytes.uint8")
P = Z + "/ppmd-probs.float16"; sz = os.path.getsize(P)
res["probs_bytes"] = sz; res["probs_laenge_ok"] = sz == N * 416
res["probs_belegt_anteil"] = round(os.stat(P).st_blocks * 512 / sz, 4)
a = np.memmap(P, dtype=np.float16, mode="r").reshape(-1, 208)
idx = np.unique(np.linspace(0, N - 2, 4000).astype(np.int64))
rows = a[idx].astype(np.float64); s = rows.sum(1)
res["stichproben"] = int(len(idx)); res["zeilensumme_min"] = float(s.min()); res["zeilensumme_max"] = float(s.max())
res["stichproben_null"] = int((rows.max(1) == 0).sum())
pn = rows[np.arange(len(idx)), bt[idx + 1]]
res["ce_bit"] = float(-np.log2(np.maximum(pn, 1e-12)).mean())
res["probs_sha"] = sha(P)
ok = (res["strom_raster_gleich_segmentkopf_plus_koderstrom_v12k"] and res["paket_gleich_strom_ab_schnitt"] and res["probs_laenge_ok"]
      and res["probs_belegt_anteil"] > 0.99 and res["stichproben_null"] == 0
      and 0.99 < res["zeilensumme_min"] and res["zeilensumme_max"] < 1.01)
res["PROBEN_OK"] = bool(ok)
json.dump(res, open(W + "/proben_p.json", "w"), indent=1); print(json.dumps(res, indent=1))
open(Z + "/PROBS_SHA256", "w").write(res["probs_sha"] + "\n")
sys.exit(0 if ok else 4)
PY
[ "${PIPESTATUS[0]}" = 0 ] || fail "probes failed (proben_p.json)" 4
head -c 1048576 "$PRIORS/ppmd-probs.float16" | sha256sum | cut -d" " -f1 > "$PRIORS/ERSTES_MIB.sha256"
say "probes passed; first MiB $(cut -c1-16 "$PRIORS/ERSTES_MIB.sha256"), priors $(cut -c1-16 "$PRIORS/PROBS_SHA256")"

# ---- 7. priors against the package (kette12_p4.sh, kette12_p4_fort.sh): the
#         original fetched the package from the dump machine here and checked
#         it against SHA256SUMS_klein; on one machine it is already in $PKG
[ "$TS" = "$SCHNITT" ] || fail "token_start package $TS != priors $SCHNITT" 4
( cd "$PKG" && sha256sum -c --quiet SHA256SUMS_klein ) || fail "package files != SHA256SUMS_klein" 3
B2=$(sha "$PRIORS/ppmd-bytes.uint8"); BS=$(awk '$2=="ppmd-bytes.uint8"{print $1}' "$PKG/SHA256SUMS_klein")
P2=$(sha "$PRIORS/z3-prior0.float16"); P3=$(sha "$PKG/z3-prior0.float16")
say "ppmd-bytes priors/package $([ "$B2" = "$BS" ] && echo EQUAL || echo DIFFERENT); prior0 priors/package $([ "$P2" = "$P3" ] && echo EQUAL || echo DIFFERENT)"
[ "$B2" = "$BS" ] && [ "$P2" = "$P3" ] || fail "raster/prior0 of priors and package differ" 4
if [ "$P3" = "$SHA_P0_SET9" ]; then SET=9; else SET=12; fi
if [ -n "${DEVTREE:-}" ] && [ -n "${DEVTREE_ENDBAU:-}" ]; then
  python3 "$HERE/prior0_diff.py" "$DEVTREE" "$DEVTREE_ENDBAU" "$PRIORS/z3-prior0.float16" "$SET" "$WORK/p0diff" "$P0_TEXT" >> "$L" 2>&1 || fail "prior0 diff" 4
  say "prior0 ${P3:0:16} -> set $SET, diff $WORK/p0diff/prior0_set_v$SET.diff"
elif [ "$SET" = 9 ]; then
  say "prior0 ${P3:0:16} = set 9, the table of the entry (src/tf1/tf1_prior0_set_v9.h)"
else
  say "prior0 ${P3:0:16} is new: the coder needs it in place of the table in src/tf1/tf1_prior0_set_v9.h"
fi

# ---- 8. checks of the whole package: a directory of hard links (package + priors)
PK="$WORK/pkg_check"; rm -rf "$PK"; mkdir -p "$PK"
for f in article-boundaries.int32 doc-last-ziel.int32 doc-rows.int64 title-lengths.int32 z3-prior0.float16 \
         dump-meta.json kennzahlen.json vocab-bitmap.u8 zonenmaske.u8 zeilen.dump; do ln -f "$PKG/$f" "$PK/$f"; done
ln -f "$PRIORS/ppmd-probs.float16" "$PK/ppmd-probs.float16"; ln -f "$PRIORS/ppmd-bytes.uint8" "$PK/ppmd-bytes.uint8"
# 8.1 Z table; Z5b is RED by design when the reference is an older package
Z5=(); [ -n "${REF_Z5_JSON:-}" ] && Z5=(--v7-z5 "$REF_Z5_JSON")
( echo 1000 > /proc/self/oom_score_adj
  exec nice -n 5 /usr/bin/time -f "rss_kb %M wall %e" python3 "$HERE/package_checks.py" "$PK" \
    --prioren "$PK/ppmd-probs.float16" --ppmd-bytes "$PK/ppmd-bytes.uint8" --gate "$PK/zonenmaske.u8" \
    --strombytes "$SOLL" --token-start "$TS" --grenzen "$WORK/pages.tsv" --token-start-strom $(( TS - 6 )) \
    ${Z5[@]+"${Z5[@]}"} --prioren-quelle "$PRIORS/ppmd-probs.float16" --stichproben 64 --json "$WORK/ztable.json" ) \
  > "$WORK/logs/ztable.log" 2>&1
ZT=$?
ROTZ=$(grep "^| Z" "$WORK/logs/ztable.log" | grep "RED" | awk -F"|" '{gsub(/ /,"",$2); print $2}' | tr '\n' ' ')
say "Z table rc=$ZT, red rows: '${ROTZ}'"
{ [ $ZT = 0 ] || [ "$(echo $ROTZ)" = "Z5b" ]; } || fail "Z table red apart from Z5b: $ROTZ" 5
# 8.2 interface check of the trainer
( cd "$TRAIN_TREE" && PYTHONPATH="${PYLIB:-}" nice -n 5 python3 -m tft.dataset check "$PK" ) > "$WORK/logs/dataset_check.log" 2>&1
DC=$?; grep -q "package matches the training-data interface" "$WORK/logs/dataset_check.log" || DC=77
say "tft.dataset check rc=$DC: $(tail -2 "$WORK/logs/dataset_check.log" | tr '\n' ' ')"
[ $DC = 0 ] || fail "dataset check" 5
# 8.3 the real loader over every micro-batch
( echo 1000 > /proc/self/oom_score_adj; exec nice -n 5 python3 "$HERE/loader_probe.py" "$PK" ) > "$WORK/logs/loader_probe.log" 2>&1
LP=$?; say "loader_probe rc=$LP: $(tail -1 "$WORK/logs/loader_probe.log")"
[ $LP = 0 ] || fail "loader probe" 5
# (the chain wrote a JSON manifest with the build lines, binary checksums and
#  host names here, and served the priors over HTTP; the facts are recorded in
#  training/tf/1_dump/README.md and training/MANIFEST.tsv)
( cd "$PK" && sha256sum -- * > SHA256SUMS_all 2>/dev/null )
say "=== training package complete: $PKG, priors $PRIORS, checks in $PK (prior0 set $SET)"
