#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
#
# Recipe of the training package of the TEXT FORM: the metadata block and
# behind it the census metadata block, then the three articles of the text
# start (dictionary article, contributor article, census pseudo article), then
# the pages.
#
# Two forms:
#   ENTRY=<entry directory>  the tools are built from the source of the entry
#                  with ../1_dump/trainmodes.patch (build_dump_tools.sh
#                  <entry> ...); the values of the stream form (page anchor,
#                  slot end mark, arms and geometry, forced title prefixes)
#                  come from ../1_dump/stream_form.txt and
#                  ../1_dump/forced_title_prefixes.h, which hold what the
#                  development tree gives for this entry; there is no line
#                  check (no build lines)
#   TREE=<development tree>  the form that built the shipped package: tools
#                  from the development tree with the build lines of ../1_dump,
#                  values from its source, the line check against the end line
#
# Chain, on one machine:
#   0  tools, the line check against the end line (TREE: zeilen_vgl.py,
#      quittances bound to their places), the manifest arms of the end line
#      (TREE: manifest_arms.sh), the values the tree and the input fix (byte
#      set, page anchor, slot end mark, page count), the training tree
#   1  bed run: "-e enwik9" with the bed line writes the canonical stream (the
#      bed of the "-n" runs), the census pseudo article of slot 3 (.vs3) and the
#      coder stream; a second "-e" run with the stream line writes the coder
#      stream again (cross-check); page anchors of the coder stream and of
#      the canonical stream against the page count of the input
#   2  dump run and prior run, "-n" on the bed, in parallel, under a disk watch
#   3  cut at the end of the first zone-3 run, mk_pkgdump, page table and title
#      lengths, ppmd-bytes and prior0, the loader against the raw dump, tar
#   4  trimming of the priors with their origin raster, probes
#   5  Z table, the loader against the raw dump, trainer interface check,
#      loader probe, manifest (Z8), acceptance
# No step reads a value of an earlier stream form: stream lengths, the cut and
# the arms come from this run, the byte set, the page anchor, the slot end mark
# and the forced title prefixes from the source of the tree, the page count
# from enwik9.
#
# Variables (paths; the four line files default to ../1_dump):
#   ENTRY          the unpacked source package of the entry (or TREE); the
#                  content sha256 of its src/ and makefile and the sha256 of
#                  the patch go into the manifest
#   TREE           development tree, holds src/cmix-neifv9 (or ENTRY); its
#                  git commit (or TREE_COMMIT) goes into the manifest
#   ENWIK9         enwik9 (required)
#   DICT           dictionary of the coder stream, .dict of the "-e" runs
#                  (required; expected sha256 0ab9f2d2..., DICT_SHA overrides)
#   ORDER          article order residual (required: the file
#                  src/readalike_prepr/data/new_article_order of the entry,
#                  sha256 0eec59a3...; ORDER_SHA overrides)
#   FEAT_STREAM FEAT_DUMP FEAT_PRIORS   build lines of the three tools
#   FEAT_END       the end line: the coder line that will run the trained
#                  container (arms, geometry, line check)
#   ZV             line check (default $TREE/scripts/zeilen_vgl.py)
#   QUITTUNG       keys whose form difference to the end line is accepted
#                  (comma list, default: none); every other one stops the run
#   QUITTUNG_SOLL  the places at which each accepted key may act in a feeder
#                  source (default: quittungen.tsv next to this script); a
#                  further place stops the run
#   VOCAB_BITMAP   the byte set (default: read from src/neif_vocfix_set.h of
#                  the tree; a given file must equal it)
#   PREFIX         optional, bytes: the "-n" runs read only the first PREFIX
#                  bytes of the bed, cut back to the end of the last page
#                  anchor in them (a probe package of whole pages; page
#                  counts and the expected full-stream values are not checked)
#   GATE           optional command prefix for every coder run (an admission
#                  gate of the machine; empty on a dedicated host)
#   PYTHON_TORCH   python with numpy and torch for the trainer checks
#                  (default python3)
#   TRAIN_TREE     training tree (default: assembled anew into
#                  $WORK/train_tree by every call, as in training/README.md
#                  section 1; a given tree is used as it is)
#   PYLIB          optional: directory with python packages PYTHON_TORCH lacks
#                  (pip install --target <dir> jaxtyping ...); stage 0 checks
#                  the trainer imports (trainer_preflight.py)
#   WORK           work directory (default ./work); PRIORS (default
#                  $WORK/priors), PKG (default $WORK/pkg).  PRIORS may lie on
#                  another file system; the check directory of stage 5 then
#                  lies in PRIORS (pkg_check), else in WORK
#   JOBS           parallel make jobs of each of the four tool builds (default 8)
#   PAGE_DIR       optional: directory on a file system with sparse files
#                  (ext4, xfs); the PPMd paging file ppm.page (14.7 GB
#                  apparent) of each "-n" run becomes a symlink into it.
#                  Only needed when WORK lies on a file system that
#                  allocates the whole file (NTFS over drvfs in WSL).
#   PY_THREADS     threads of the python/numpy/torch steps (default 4: the
#                  limit on shared servers); the torch steps run one BLAS
#                  thread and two OpenMP teams of PY_THREADS/2 (the trainer
#                  loader prefetches in a thread of its own, which gets a team
#                  of its own); the coder runs use one thread; the line check
#                  runs PY_THREADS preprocessors
#   DISK_MIN_GIB   room the recipe demands in WORK, free plus already used
#                  by WORK, in GiB (default 245 for the full stream, 25 for
#                  a probe)
#   DISK_RESERVE_GIB  free room below which the disk watch of stage 2 stops
#                  the runs (default 4)
#   ENWIK9_SHA     expected sha256 of ENWIK9 (default the one of enwik9)
# Disk (full stream, stage 0 stops below DISK_MIN_GIB; GiB = 2^30 B): raw
# priors about 200 GiB (rows = coded bytes + injected bytes, 2V = 416 B each;
# projection in REZEPT.md), the two "-n" runs up to 2 x 13.7 GiB of ppm.page
# while they run, dump 7.5 GiB, stage-1 runs 3.7 GiB, package 14 GiB (tables
# and tar, after the runs): peak about 240 GiB at the end of stage 2.  Stage 2
# runs under a disk watch (exit 12 before the disk is full), stages 3, 4 and 5
# check their room before they write.
# Memory (full stream): mk_pkgdump about 93 B per token, the Z table about
# 100 B per token (measured 41.2 and 44.1 GiB on 97.6 % of the stream), plus
# 2 GiB; stage 0 stops below 50 GiB total, stages 3.2, 5.1 and 5.6 stop when
# less is available (exit 11; resume with the same command once it is).
# Resume: every finished run and stage carries a stamp (STEMPEL, STAGE3.stempel,
# STAGE4.stempel) with the sha256 of its tools, build lines, code and inputs;
# the same command recomputes it and stops (exit 10) on any difference, a
# finished run or stage of another tree, line or input is never reused.  The
# prior run and stage 3 also record where they wrote (PRIORS, PKG) and what
# (ppmd-quelle.u8, SHA256SUMS_klein): a resume whose PRIORS or PKG does not
# hold that output stops (exit 10) and names the directory of the earlier call.  The trimming of stage 4 leaves two marks
# (STAGE4.begonnen before the priors are changed, STAGE4.getrimmt after), so
# an interruption inside stage 4 resumes there.  The training tree and every
# check of stage 5 run anew on every call.
# Exit codes: 1 tool build, 2 inputs, lines, line check, arms, values of the
# tree, room, memory in total, python imports, 3 montage, training tree,
# manifest, 4 probes of the priors, 5 runs, Z table, trainer checks,
# acceptance, 6 page anchors, page count, page table, trimming, 7 priors file
# without a finished prior run, ppmd-bytes/prior0, 9 zeilen.dump without a
# finished stage 3, 10 stamp, 11 memory before stage 3.2/5.1/5.6, 12 disk.
#
# The whole recipe is one brace group: bash reads all of it before the first
# step runs and stops at its exit, so a change of this file during a run (a
# checkout in TREE) does not reach the running recipe, and no text behind
# the group is ever read.
{
set -uo pipefail
ulimit -c 0
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAINING="${TRAINING:-$(cd "$HERE/../.." && pwd)}"
FEAT="$TRAINING/tf/1_dump"
if [ -n "${ENTRY:-}" ]; then
  [ -z "${TREE:-}" ] || { echo "ENTRY and TREE are both set: choose one form" >&2; exit 2; }
  FORM=entry
  ENTRY="$(cd "$ENTRY" && pwd)"
  PATCH="$FEAT/trainmodes.patch"
  WERTE="$FEAT/stream_form.txt"
  ZWANG="$FEAT/forced_title_prefixes.h"
  TREE="$ENTRY"; SRC=""
else
  : "${TREE:?TREE (development tree with src/cmix-neifv9) or ENTRY (source package of the entry) missing}"
  FORM=tree
  TREE="$(cd "$TREE" && pwd)"
  SRC="$TREE/src/cmix-neifv9"
  ZWANG="$SRC/src/tf1/tf1_block_set2.h"
fi
: "${ENWIK9:?ENWIK9 missing}"
: "${DICT:?DICT missing (dictionary of the coder stream, sha256 0ab9f2d2...)}"
ORDER="${ORDER:?ORDER missing (article order residual, sha256 0eec59a3...)}"
DICT_SHA="${DICT_SHA:-0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535}"
ORDER_SHA="${ORDER_SHA:-0eec59a305774b5a46b17b83a31274aafd5081ba387a3536def3372a442ba59d}"
FEAT_STREAM="${FEAT_STREAM:-$FEAT/features_stream.txt}"
FEAT_DUMP="${FEAT_DUMP:-$FEAT/features_dump.txt}"
FEAT_PRIORS="${FEAT_PRIORS:-$FEAT/features_priors.txt}"
FEAT_END="${FEAT_END:-$FEAT/features_end.txt}"
ZV="${ZV:-$TREE/scripts/zeilen_vgl.py}"
QUITTUNG="${QUITTUNG:-}"
QUITTUNG_SOLL="${QUITTUNG_SOLL:-$HERE/quittungen.tsv}"
PREFIX="${PREFIX:-}"
GATE="${GATE:-}"
PAGE_DIR="${PAGE_DIR:-}"
PYTHON_TORCH="${PYTHON_TORCH:-python3}"
PY_THREADS="${PY_THREADS:-4}"
DISK_RESERVE_GIB="${DISK_RESERVE_GIB:-4}"
ENWIK9_SHA="${ENWIK9_SHA:-159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc}"
WORK="$(mkdir -p "${WORK:-./work}" && cd "${WORK:-./work}" && pwd)"
PRIORS="${PRIORS:-$WORK/priors}"
PKG="${PKG:-$WORK/pkg}"
mkdir -p "$PRIORS" "$PKG" "$WORK/logs" "$WORK/runs" "$WORK/stempel"
PRIORS="$(cd "$PRIORS" && pwd)"; PKG="$(cd "$PKG" && pwd)"
SELF="$HERE/$(basename "$0")"
if [ "$FORM" = entry ]; then
  # no build lines: each tool is the source of the entry with the patch and one
  # TRAIN_MODE; these one-line files name it where the recipe records a line
  PATCH_SHA=$(sha256sum "$PATCH" | cut -d' ' -f1)
  for t_ in stream:1 dump:2 priors:3 bed:4 end:none; do
    echo "the source of the entry ($ENTRY) with trainmodes.patch (sha256 $PATCH_SHA), TRAIN_MODE=${t_#*:}" \
      > "$WORK/logs/mode_${t_%%:*}.txt"
  done
  FEAT_STREAM="$WORK/logs/mode_stream.txt"; FEAT_DUMP="$WORK/logs/mode_dump.txt"
  FEAT_PRIORS="$WORK/logs/mode_priors.txt"; FEAT_END="$WORK/logs/mode_end.txt"
fi
export PYTHONDONTWRITEBYTECODE=1
# python and numpy steps: at most PY_THREADS threads each; the torch steps run
# with one BLAS thread and OpenMP teams of PY_THREADS/2: the loader of the
# trainer prefetches in a second thread, and every thread that enters torch
# gets an OpenMP team of its own (measured: 2 teams of 4 = 8 active threads);
# the coder runs set their own single thread below
export OMP_NUM_THREADS="$PY_THREADS" MKL_NUM_THREADS="$PY_THREADS" OPENBLAS_NUM_THREADS="$PY_THREADS" \
       NUMEXPR_NUM_THREADS="$PY_THREADS" ZV_JOBS="${ZV_JOBS:-$PY_THREADS}"
L="$WORK/logs/build_package.log"
say()  { echo "$(date '+%F %T') $*" | tee -a "$L"; }
fail() { say "!!! $*"; exit "${2:-1}"; }
sha()  { sha256sum "$1" | cut -d' ' -f1; }
kv()   { sed -n "s/^$1=//p" "$2" | tail -1; }   # kv <key> <file of key=value lines>
gib()  { echo $(( $1 / 1024 / 1024 )); }        # KiB -> GiB
mem_kb() { awk -v k="$1:" '$1==k{print $2}' /proc/meminfo; }
frei_gib() { df -Pk "$1" | awk 'NR==2{print int($4/1048576)}'; }
PY_TEAM=$(( PY_THREADS > 1 ? PY_THREADS / 2 : 1 ))
torchlauf() { OMP_NUM_THREADS="$PY_TEAM" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$@"; }

say "=== build_package.sh (text form): $FORM $TREE, work $WORK${PREFIX:+, probe PREFIX $PREFIX}"
if [ "$FORM" = entry ]; then
  for f in "$ENTRY/makefile" "$ENTRY/src" "$PATCH" "$WERTE" "$ZWANG"; do [ -e "$f" ] || fail "missing: $f" 2; done
else
  [ -f "$SRC/makefile" ] || fail "no $SRC/makefile" 2
fi
for f in "$FEAT_STREAM" "$FEAT_DUMP" "$FEAT_PRIORS" "$FEAT_END" "$ENWIK9" "$DICT" "$ORDER" "$QUITTUNG_SOLL" "$ZWANG"; do
  [ -s "$f" ] || fail "missing or empty: $f" 2
done
[ "$PY_THREADS" -ge 1 ] 2>/dev/null || fail "PY_THREADS is not a thread count" 2
[ "$DISK_RESERVE_GIB" -ge 0 ] 2>/dev/null || fail "DISK_RESERVE_GIB is not a number" 2
python3 -c 'import numpy' 2>/dev/null || fail "python3 has no numpy (the dump checks, the trimming and the Z table need it)" 2
ENWIK9_IST=$(sha "$ENWIK9")
[ "$ENWIK9_IST" = "$ENWIK9_SHA" ] || fail "ENWIK9 sha256 $ENWIK9_IST != $ENWIK9_SHA" 2
[ "$(sha "$DICT")" = "$DICT_SHA" ] || fail "DICT sha256 $(sha "$DICT") != $DICT_SHA" 2
[ "$(sha "$ORDER")" = "$ORDER_SHA" ] || fail "ORDER sha256 $(sha "$ORDER") != $ORDER_SHA (text form: order v4)" 2
if [ -n "$PREFIX" ]; then [ "$PREFIX" -gt 0 ] 2>/dev/null || fail "PREFIX is not a byte count" 2; fi

# ---- 0.0 room and memory, before anything runs.  Disk: free plus what WORK
#          (and PRIORS) already hold, against the measured need (header).  The
#          check directory of stage 5 holds hard links of the priors: it lies
#          on the file system of PRIORS (in PRIORS when that is not the one of
#          WORK; the small package files are copied there, not linked).
DISK_MIN_GIB="${DISK_MIN_GIB:-$([ -n "$PREFIX" ] && echo 25 || echo 245)}"
[ "$DISK_MIN_GIB" -ge 0 ] 2>/dev/null || fail "DISK_MIN_GIB is not a number" 2
platz_gib() {   # platz_gib <dir>: free on its file system + what the directory already holds
  local frei; frei=$(df -Pk "$1" | awk 'NR==2{print $4}')
  echo $(( (frei + $(du -sk "$1" 2>/dev/null | cut -f1)) / 1024 / 1024 ))
}
if [ "$(stat -c %d "$WORK")" = "$(stat -c %d "$PRIORS")" ]; then
  PK="$WORK/pkg_check"; PK_KOPIE=0
  case "$PRIORS/" in
    "$WORK"/*) P_GIB=$(platz_gib "$WORK"); P_TXT="WORK" ;;
    *) P_GIB=$(( $(platz_gib "$WORK") + $(du -sk "$PRIORS" | cut -f1) / 1024 / 1024 )); P_TXT="WORK + PRIORS" ;;
  esac
else
  # PRIORS on another file system: the raw priors (85 % of the need) and the
  # copies of the small package files in the check directory (3 %) go there
  PK="$PRIORS/pkg_check"; PK_KOPIE=1
  P_MIN=$(( DISK_MIN_GIB * 88 / 100 )); PR_GIB=$(platz_gib "$PRIORS")
  [ "$PR_GIB" -ge "$P_MIN" ] || fail "PRIORS ($PRIORS, another file system than WORK) has room for $PR_GIB GiB, the raw priors and the check directory need $P_MIN GiB; DISK_MIN_GIB overrides" 2
  P_GIB=$(( $(platz_gib "$WORK") + P_MIN )); P_TXT="WORK (+ $P_MIN GiB in PRIORS)"
fi
[ "$P_GIB" -ge "$DISK_MIN_GIB" ] || fail "$P_TXT: room for $P_GIB GiB (free + already used), the recipe needs $DISK_MIN_GIB GiB (full stream: raw priors about 200 GiB, ppm.page of the two -n runs up to 27 GiB, dump 7.5 GiB, stage-1 runs 3.7 GiB, package 14 GiB); DISK_MIN_GIB overrides" 2
say "room: $P_GIB GiB in $P_TXT (needs $DISK_MIN_GIB GiB); check directory $PK$([ $PK_KOPIE = 1 ] && echo ' (PRIORS on another file system than WORK: package files copied there)'); memory total $(gib "$(mem_kb MemTotal)") GiB, available $(gib "$(mem_kb MemAvailable)") GiB; python threads $PY_THREADS"
if [ -z "$PREFIX" ]; then
  [ "$(gib "$(mem_kb MemTotal)")" -ge 50 ] || fail "memory $(gib "$(mem_kb MemTotal)") GiB: the full package needs about 45 GiB (mk_pkgdump) and 48 GiB (Z table), at least 50 GiB total, 64 GiB recommended" 2
fi
# mem_vor <bytes per token> <step>: stop before a large python step when less
# is available than n_tokens x <bytes per token> + 2 GiB (measured on 97.6 %
# of the stream, see the header); a probe package checks its own small need
mem_vor() {
  local av need; av=$(gib "$(mem_kb MemAvailable)")
  need=$(( (NTOK * $1 + (2 << 30) + (1 << 30) - 1) >> 30 ))
  [ "$av" -ge "$need" ] || fail "$2 needs about $need GiB ($1 B per token, $NTOK tokens, + 2 GiB), $av GiB available: free memory and run the same command again (finished runs and stages are kept)" 11
  say "memory for $2: $av GiB available, about $need GiB needed"
}
# platz_vor <GiB> <dir> <step>: stop before a step writes more than the room
platz_vor() {
  local fr; fr=$(frei_gib "$2")
  [ "$fr" -ge "$(( $1 + DISK_RESERVE_GIB ))" ] || fail "$3 writes about $1 GiB into $2, $fr GiB free (+ reserve $DISK_RESERVE_GIB GiB): free disk and run the same command again" 12
}

# ---- stamps (resume).  stempel_soll <name> <lines...> writes what this call
#      would build into WORK/stempel/<name>.soll; stempel_pruefen <name>
#      <stamp of the finished run or stage> compares and stops on a difference.
stempel_soll() { local n=$1; shift; printf '%s\n' "$@" > "$WORK/stempel/$n.soll"; }
stempel_pruefen() {   # stempel_pruefen <key> <stamp file> <what>
  local ist=$2 n=$3 soll="$WORK/stempel/$1.soll"
  if [ ! -f "$ist" ]; then
    fail "$n is finished but carries no stamp ($ist; made by an earlier recipe?): it is not reused, start with a fresh WORK" 10
  fi
  if ! cmp -s "$ist" "$soll"; then
    say "!!! $n was made with other tools, lines, code or inputs than this call (< finished, > this call):"
    diff "$ist" "$soll" | grep '^[<>]' | sed 's/^/      /' | tee -a "$L"
    fail "$n does not belong to this call: start with a fresh WORK, or remove $n and every result behind it (runs/dump, runs/priors, $PRIORS, $PKG, montage, trim, STAGE3.*, STAGE4.*)" 10
  fi
  say "stamp of $n matches: $(sha "$ist" | cut -c1-16)"
}
abschnitt() {  # abschnitt <from marker> <to marker>: text of a stage of this script
  awk -v a="$1" -v b="$2" 'index($0,a)==1{p=1} index($0,b)==1{p=0} p' "$SELF"
}
if [ "$FORM" = entry ]; then
  # content hash of src/ and makefile of the entry (sorted per-file sha256,
  # C locale); the commit field names the patch
  TREE_SHA=$(cd "$ENTRY" && find src makefile -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  TREE_COMMIT="${TREE_COMMIT:-entry+trainmodes.patch:$PATCH_SHA}"
  say "entry $ENTRY, content sha256 ${TREE_SHA:0:16} (src/, makefile), trainmodes.patch ${PATCH_SHA:0:16}"
  cp "$WORK/logs/mode_bed.txt" "$WORK/logs/features_bed_line.txt"
else
  TREE_COMMIT="${TREE_COMMIT:-$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unknown)}"
  # content hash of src/cmix-neifv9 (sorted per-file sha256, C locale)
  TREE_SHA=$(cd "$SRC" && find . -type f ! -path './.git/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  say "tree commit $TREE_COMMIT, content sha256 ${TREE_SHA:0:16} (src/cmix-neifv9)"
  { tr -d '\r\n' < "$FEAT_STREAM"; echo " CENSBLOCK_PRUEF=1"; } > "$WORK/logs/features_bed_line.txt"
fi
for f in "$FEAT_STREAM" "$FEAT_DUMP" "$FEAT_PRIORS" "$FEAT_END"; do say "line $(sha "$f" | cut -c1-16) $f"; done

# ---- 0.05 values the tree and the input fix (no value of a stream form is
#           written into this recipe): the byte set of the coder stream
#           (src/neif_vocfix_set.h), the page anchor (page_anchors.py
#           --anker-aus-baum: three definitions in the tree must agree), the
#           end mark of the text-start slots (kEnde, src/neif_textanfang_pack.h),
#           and the page count of the input (the '</page>' of enwik9)
VOCAB_BAUM="$WORK/logs/vocab-bitmap_baum.u8"
if [ "$FORM" = entry ]; then
  # the values of the stream form of the entry: the byte set of this directory
  # and ../1_dump/stream_form.txt (read from the development tree)
  cp "$HERE/vocab-bitmap.u8" "$VOCAB_BAUM"
  V_BAUM=$(python3 -c 'import sys; print(sum(bin(x).count("1") for x in open(sys.argv[1], "rb").read()))' "$VOCAB_BAUM")
  ANKER_HEX=$(kv anchor "$WERTE"); ENDMARKE=$(kv slot_end_mark "$WERTE")
  [ -n "$ANKER_HEX" ] && [ -n "$ENDMARKE" ] && [ "$V_BAUM" -gt 0 ] 2>/dev/null \
    || fail "values of the stream form ($WERTE, $HERE/vocab-bitmap.u8) unreadable" 2
  [ -z "${VOCAB_BITMAP:-}" ] || cmp -s "$VOCAB_BITMAP" "$VOCAB_BAUM" \
    || fail "VOCAB_BITMAP $VOCAB_BITMAP is not the byte set of the entry ($HERE/vocab-bitmap.u8)" 2
  VOCAB_BITMAP="$VOCAB_BAUM"
else
V_BAUM=$(python3 - "$SRC/src/neif_vocfix_set.h" "$VOCAB_BAUM" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="latin-1").read()
m = re.search(r"#define\s+NEIF_VOCFIX_BITMAP_INIT\s*\{([^}]*)\}", s)
n = re.search(r"#define\s+NEIF_VOCFIX_SIZE\s+(\d+)", s)
b = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{1,2})", m.group(1))) if m else b""
if len(b) != 32 or not n or sum(bin(x).count("1") for x in b) != int(n.group(1)):
    sys.exit("NEIF_VOCFIX_BITMAP_INIT / NEIF_VOCFIX_SIZE unreadable or inconsistent")
open(sys.argv[2], "wb").write(b)
print(int(n.group(1)))
PY
) || fail "byte set of the tree ($SRC/src/neif_vocfix_set.h)" 2
if [ -n "${VOCAB_BITMAP:-}" ]; then
  cmp -s "$VOCAB_BITMAP" "$VOCAB_BAUM" || fail "VOCAB_BITMAP $VOCAB_BITMAP is not the byte set of the tree (src/neif_vocfix_set.h)" 2
else
  VOCAB_BITMAP="$VOCAB_BAUM"
fi
export VOCAB_BITMAP
ANKER_HEX=$(python3 "$HERE/page_anchors.py" --anker-aus-baum "$SRC" 2> "$WORK/logs/anker.log") \
  || fail "page anchor of the tree: $(tr '\n' ' ' < "$WORK/logs/anker.log")" 6
ENDMARKE=$(sed -n 's/.*static const unsigned char kEnde *= *\(0x[0-9A-Fa-f][0-9A-Fa-f]*\);.*/\1/p' "$SRC/src/neif_textanfang_pack.h" | head -1)
[ -n "$ENDMARKE" ] || fail "end mark of the text-start slots (kEnde) not found in $SRC/src/neif_textanfang_pack.h" 2
fi
export VOCAB_BITMAP
SEITEN=$(python3 "$HERE/page_anchors.py" --seiten-zaehlen "$ENWIK9") && [ "$SEITEN" -gt 0 ] 2>/dev/null \
  || fail "page count of $ENWIK9" 6
say "from the tree: byte set $V_BAUM values ($(sha "$VOCAB_BITMAP" | cut -c1-16)), page anchor $ANKER_HEX, slot end mark $ENDMARKE; from the input: $SEITEN pages"

# ---- 0.1 tools: plain builds of the tree with the four lines; rebuilt when a
#          line file or the tree changes
T="$WORK/tools"
STAMP="$TREE_SHA $(cat "$FEAT_STREAM" "$FEAT_DUMP" "$FEAT_PRIORS" | sha256sum | cut -c1-64)"
if [ ! -x "$T/cmix_priors" ] || [ "$(cat "$T/STAMP" 2>/dev/null)" != "$STAMP" ]; then
  rm -rf "$T"
  if [ "$FORM" = entry ]; then
    bash "$FEAT/build_dump_tools.sh" "$ENTRY" "$T" bed stream dump priors > "$WORK/logs/build_tools.log" 2>&1 \
      || fail "build_dump_tools.sh (see $WORK/logs/build_tools.log)" 1
    # the line record of each tool (the run stamps and the manifest read it)
    for t_ in bed stream dump priors; do cp "$WORK/logs/mode_$t_.txt" "$T/features_$t_.txt"; done
  else
    FEAT_STREAM="$FEAT_STREAM" FEAT_DUMP="$FEAT_DUMP" FEAT_PRIORS="$FEAT_PRIORS" \
      bash "$FEAT/build_dump_tools.sh" --tree "$TREE" "$T" > "$WORK/logs/build_tools.log" 2>&1 \
      || fail "build_dump_tools.sh (see $WORK/logs/build_tools.log)" 1
  fi
  echo "$STAMP" > "$T/STAMP"
fi
say "tools: $(cd "$T" && sha256sum cmix_bed cmix_stream cmix_dump cmix_priors | cut -c1-16 | tr '\n' ' ')"

# ---- 0.2 manifest arms and package geometry of the END line, read from the
#          source of the tree (neif_tf1_manifest::Arme()).  The trainer copies
#          the arms into the weight container and the coder of the end line
#          compares them as a set: the package must carry exactly this list.
if [ "$FORM" = entry ]; then
  # the arms and the geometry of the entry, as the development tree gives them
  cp "$WERTE" "$WORK/logs/arms_end.txt"; cp "$WERTE" "$WORK/logs/arms_dump.txt"
else
bash "$HERE/manifest_arms.sh" "$SRC" "$FEAT_END" > "$WORK/logs/arms_end.txt" 2>&1 || fail "manifest_arms.sh on the end line: $(tail -3 "$WORK/logs/arms_end.txt")" 2
bash "$HERE/manifest_arms.sh" "$SRC" "$FEAT_DUMP" > "$WORK/logs/arms_dump.txt" 2>&1 || fail "manifest_arms.sh on the dump line: $(tail -3 "$WORK/logs/arms_dump.txt")" 2
fi
ARME_END=$(kv arme "$WORK/logs/arms_end.txt")
[ -n "$ARME_END" ] || fail "no arm list for the end line" 2
for k in arme v nsent x dslot x0row redirmerge fenster nl; do
  [ "$(kv $k "$WORK/logs/arms_end.txt")" = "$(kv $k "$WORK/logs/arms_dump.txt")" ] \
    || fail "dump line and end line differ in $k: $(kv $k "$WORK/logs/arms_dump.txt") / $(kv $k "$WORK/logs/arms_end.txt")" 2
done
if [ -n "${ARME:-}" ] && [ "$ARME" != "$ARME_END" ]; then fail "ARME given ($ARME) != Arme() of the end line ($ARME_END)" 2; fi
ARME="$ARME_END"
G_V=$(kv v "$WORK/logs/arms_end.txt"); G_NSENT=$(kv nsent "$WORK/logs/arms_end.txt"); G_X=$(kv x "$WORK/logs/arms_end.txt")
G_FENSTER=$(kv fenster "$WORK/logs/arms_end.txt"); G_X0ROW=$(kv x0row "$WORK/logs/arms_end.txt"); G_RM=$(kv redirmerge "$WORK/logs/arms_end.txt")
G_DSLOT=$(kv dslot "$WORK/logs/arms_end.txt")
[ "$G_V" = "$V_BAUM" ] || fail "vocabulary of the end line (NEIF_TF1_V $G_V) != values of the byte set of the tree ($V_BAUM)" 2
ZB=$(( 2 * G_V ))          # bytes of a prior row: V float16
MKOPT=(--v "$G_V" --nsent "$G_NSENT" --x "$G_X" --fenster "$G_FENSTER")
[ "$G_X0ROW" = 1 ] && MKOPT+=(--x0row)
[ "$G_RM" = 1 ] && MKOPT+=(--redirmerge)
# the trainer steps read the build state of the dump rows from the environment
export PK_LAST_X0ROW="$G_X0ROW" PK_REDIRMERGE="$G_RM" PK_SLOT_DIM="$G_DSLOT"
say "arms of the end line: $ARME"
say "geometry: v $G_V nsent $G_NSENT x $G_X dslot $G_DSLOT window $G_FENSTER x0row $G_X0ROW redirmerge $G_RM nl $(kv nl "$WORK/logs/arms_end.txt")"

# ---- 0.3 line check: every key of `make features` and every NEIF macro the
#          preprocessor sees, dump/prior/stream/bed line against the end line;
#          an accepted key (QUITTUNG) only at the places QUITTUNG_SOLL names
if [ "$FORM" = entry ]; then
  # no build lines: the modes of the patch are the line check (generated from
  # the line of the entry plus the switches of each mode)
  printf '{"endzeile": "%s", "quittung": [], "zeilen": {}}\n' "the source of the entry, trainmodes.patch $PATCH_SHA" \
    > "$WORK/logs/zeilen_vgl.json"
  say "line check: none (entry form, trainmodes.patch ${PATCH_SHA:0:16})"
else
[ -f "$ZV" ] || fail "line check $ZV missing (ZV=...)" 2
QARG=(--quittung-soll "$QUITTUNG_SOLL"); [ -n "$QUITTUNG" ] && QARG+=(--quittung "$QUITTUNG")
python3 "$ZV" "$SRC" "$FEAT_END" "$FEAT_DUMP" "$FEAT_PRIORS" "$FEAT_STREAM" "$WORK/logs/features_bed_line.txt" \
  "${QARG[@]}" --json "$WORK/logs/zeilen_vgl.json" > "$WORK/logs/zeilen_vgl.txt" 2>&1
ZRC=$?
say "line check rc=$ZRC: $(grep -a '^== \|ERGEBNIS' "$WORK/logs/zeilen_vgl.txt" | sed 's/^== //' | cut -c1-140 | tr '\n' ' ')"
grep -a 'QUITTUNG GEBROCHEN\|QUITTUNG UNGEBUNDEN\|Fundstelle nicht in der Sollliste' "$WORK/logs/zeilen_vgl.txt" | sort -u | sed 's/^ */      /' | tee -a "$L"
[ $ZRC = 0 ] || fail "line check: form difference to the end line, an accepted key at a place $QUITTUNG_SOLL does not name, or a preprocessor error, see $WORK/logs/zeilen_vgl.txt" 2
fi

# ---- 0.4 the trainer tree (training/README.md section 1) for the loader
#          checks of stages 3 and 5, assembled anew on every call (a resume
#          checks with the code of this call), and a preflight of its imports
#          with PYTHON_TORCH (a missing package stops here, not behind the runs)
if [ -z "${TRAIN_TREE:-}" ]; then
  TRAIN_TREE="$WORK/train_tree"
  rm -rf "$TRAIN_TREE.neu"
  cp -a "$TRAINING/tf/tree_upstream" "$TRAIN_TREE.neu" && cp -a "$TRAINING/tf/tree_overlay/." "$TRAIN_TREE.neu/" \
    && python3 "$HERE/patch_paket11.py" "$TRAIN_TREE.neu" > "$WORK/logs/train_tree.log" 2>&1 || fail "training tree" 3
  rm -rf "$TRAIN_TREE" && mv "$TRAIN_TREE.neu" "$TRAIN_TREE" || fail "training tree" 3
  TT_WIE="assembled anew"
else
  TT_WIE="given"
fi
export TRAIN_TREE
TT_SHA=$(cd "$TRAIN_TREE" && find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)
say "training tree $TRAIN_TREE ($TT_WIE): content sha256 ${TT_SHA:0:16}"
# pysrc/model.py imports jaxtyping for type annotations only; without it (and
# without a PYLIB) a stand-in as in tf/4_export/export.sh
if [ -z "${PYLIB:-}" ] && ! "$PYTHON_TORCH" -c "import jaxtyping" 2>/dev/null; then
  PYLIB="$WORK/pylib_stub"; mkdir -p "$PYLIB"
  printf 'class _T:\n    def __class_getitem__(cls, item):\n        return object\n\n\ndef __getattr__(name):\n    return _T\n' > "$PYLIB/jaxtyping.py"
  say "jaxtyping missing in $PYTHON_TORCH: stand-in for the annotations in $PYLIB"
fi
[ -n "${PYLIB:-}" ] && export PYLIB
torchlauf "$PYTHON_TORCH" "$HERE/trainer_preflight.py" > "$WORK/logs/trainer_preflight.log" 2>&1 \
  || fail "trainer imports with PYTHON_TORCH=$PYTHON_TORCH${PYLIB:+ PYLIB=$PYLIB}: $(tr '\n' ' ' < "$WORK/logs/trainer_preflight.log" | cut -c1-400)" 2
say "$(tail -1 "$WORK/logs/trainer_preflight.log")"

# ---- 1. bed run and stream run ("-e enwik9", in parallel)
erun() {   # erun <tool> <dir>: "-e enwik9" with .dict and the order in the run directory
  local tool=$1 d=$2 rc
  [ -e "$d/.FERTIG" ] && return 0
  rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 1
  cp "$T/cmix_$tool" ./cmix; cp "$DICT" .dict; cp "$ORDER" .new_article_order
  /usr/bin/time -v nice -n 5 $GATE ./cmix -e "$ENWIK9" out > enc.log 2>&1; rc=$?
  echo "$rc" > rc.txt
  rm -f .main .main_reordered .ready4cmix un1
  [ $rc = 0 ] && cp "$WORK/stempel/$tool.soll" STEMPEL && date > .FERTIG
  return $rc
}
# lauf_soll <tool> <input lines...>: the stamp of a run of this call (tool
# binary, its resolved build line, the tree, the run command, the inputs)
lauf_soll() {
  local tool=$1 fn=$2; shift 2
  stempel_soll "$tool" "lauf $tool" "werkzeug cmix_$tool $(sha "$T/cmix_$tool")" \
    "zeile $(sha "$T/features_$tool.txt")" "baum $TREE_SHA" \
    "befehl $(declare -f "$fn" | sha256sum | cut -c1-64)" "$@"
}
B="$WORK/runs/bed"; S="$WORK/runs/stream"
for t in bed stream; do
  lauf_soll $t erun "eingang enwik9 $ENWIK9_IST" "eingang dict $DICT_SHA" "eingang order $ORDER_SHA"
done
[ -e "$B/.FERTIG" ] && stempel_pruefen bed "$B/STEMPEL" "the bed run $B"
[ -e "$S/.FERTIG" ] && stempel_pruefen stream "$S/STEMPEL" "the stream run $S"
( erun bed "$B" ) & P1=$!
( erun stream "$S" ) & P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
[ $R1 = 0 ] || fail "bed run rc=$R1 ($B/enc.log)" 5
[ $R2 = 0 ] || fail "stream run rc=$R2 ($S/enc.log)" 5
KANON="$B/censpruef_kanon.bin"; VS3="$B/censpruef_vs3.bin"; BK="$B/censpruef_strom.bin"
for f in "$KANON" "$VS3" "$BK" "$S/out.cmix.temp"; do [ -s "$f" ] || fail "bed/stream run wrote no $f" 5; done
[ "$(sha "$BK")" = "$(sha "$S/out.cmix.temp")" ] || fail "coder stream of the bed run != coder stream of the stream run" 5
say "canonical stream (bed) $(stat -c%s "$KANON") B $(sha "$KANON" | cut -c1-16); .vs3 $(stat -c%s "$VS3") B $(sha "$VS3" | cut -c1-16); coder stream $(stat -c%s "$BK") B $(sha "$BK" | cut -c1-16) (= stream run)"
[ -s "$B/.main_phda9prepr" ] && say ".main_phda9prepr $(stat -c%s "$B/.main_phda9prepr") B $(sha "$B/.main_phda9prepr" | cut -c1-16)"
# ---- 1.1 page anchors of the WHOLE coder stream and of the canonical
#          stream, before the long runs and also on a probe bed: one anchor
#          per page of the input in both (the stages between them keep every
#          anchor) and a tail behind the last one; a stream form that moved
#          an anchor stops here, not in stage 3.  The tail itself may differ
#          between the two streams (the rule stages rewrite up
#          to the stream end): reported, not demanded
python3 "$HERE/page_anchors.py" "$BK" "$WORK/logs/page_anchors_coder_stream.tsv" --anker "$ANKER_HEX" \
  --seiten "$SEITEN" --anker-wie "$KANON" > "$WORK/logs/page_anchors.log" 2>&1 \
  || fail "page anchors of the coder stream or of the canonical stream: $(grep -a '^!' "$WORK/logs/page_anchors.log" | tr '\n' ' ')" 6
say "coder stream: $(grep -a '^# anchors\|^# last anchor' "$WORK/logs/page_anchors.log" | sed 's/^# //' | cut -c1-200 | tr '\n' ' ')"
BETT="$KANON"; BETT_N=0
if [ -n "$PREFIX" ]; then
  # a probe bed ends behind the last page anchor within PREFIX bytes: stream
  # stages that look ahead inside a page (the rule stages: the
  # words behind "a"/"an", closing brackets) then see the same bytes in the
  # prefix as in the whole stream, and the dump of the prefix is a prefix of
  # the dump of the whole stream
  BETT_N=$(python3 -c 'import sys; d = open(sys.argv[1], "rb").read(int(sys.argv[2])); a = bytes.fromhex(sys.argv[3]); i = d.rfind(a); print(i + len(a) if i >= 0 else 0)' \
             "$KANON" "$PREFIX" "$ANKER_HEX") && [ "$BETT_N" -gt 0 ] 2>/dev/null \
    || fail "PREFIX $PREFIX: no page anchor in the first $PREFIX B of the canonical stream (a probe bed holds whole pages)" 2
  BETT="$WORK/runs/bed_prefix.bin"
  head -c "$BETT_N" "$KANON" > "$BETT"
  say "probe bed: first $BETT_N B of the canonical stream (PREFIX $PREFIX, cut behind the last page anchor), $(sha "$BETT" | cut -c1-16)"
fi
BETT_SHA=$(sha "$BETT"); VS3_SHA=$(sha "$VS3")

# ---- 2. dump run and prior run ("-n" on the bed, in parallel).  Each run
#         directory holds .dict and .vs3 (slot 3 is read from ./.vs3: without it
#         the census pseudo article is packed EMPTY and nothing complains).
#         NEIF_TF1_WEIGHTS only switches the transformer path on: the dump-only
#         and PPMd-only builds never read a container.
nrun() {   # nrun <tool> <dir>
  local tool=$1 d=$2 rc
  [ -e "$d/.FERTIG" ] && return 0
  rm -rf "$d"; mkdir -p "$d"; cd "$d" || return 1
  cp "$T/cmix_$tool" ./cmix; cp "$DICT" .dict; cp "$VS3" .vs3
  [ "$(sha .vs3)" = "$(sha "$VS3")" ] || { echo ".vs3 missing next to the run" > rc_reason.txt; return 9; }
  ln -sf /dev/null aus
  if [ -n "$PAGE_DIR" ]; then
    mkdir -p "$PAGE_DIR/$tool" && rm -f "$PAGE_DIR/$tool/ppm.page" && ln -sf "$PAGE_DIR/$tool/ppm.page" ppm.page || return 1
  fi
  (
    unset NEIF_TF1_PAKET_DUMP NEIF_TF1_GATE_DUMP NEIF_TF1_MATCH_DUMP NEIF_TF1_SAVE_PRIORS NEIF_PPMD_ONLY_BYTES NEIF_PPMD_ONLY_QUELLE
    export OMP_NUM_THREADS=1 NEIF_TF1_WEIGHTS=no-container-in-this-mode
    if [ "$tool" = dump ]; then
      export NEIF_TF1_PAKET_DUMP=./ NEIF_TF1_GATE_DUMP="$d/zonenmaske.u8" NEIF_TF1_MATCH_DUMP=./ NEIF_TF1_SAVE_PRIORS=/dev/null
    else
      export NEIF_TF1_SAVE_PRIORS="$PRIORS/ppmd-probs.float16" NEIF_PPMD_ONLY_BYTES="$PRIORS/ppmd-bytes.uint8" \
             NEIF_PPMD_ONLY_QUELLE="$PRIORS/ppmd-quelle.u8"
    fi
    /usr/bin/time -v nice -n 5 $GATE ./cmix -n "$BETT" ./aus > stderr.txt 2>&1
  ); rc=$?
  echo "$rc" > rc.txt
  rm -f aus.cmix.temp ppm.page
  [ -n "$PAGE_DIR" ] && rm -f "$PAGE_DIR/$tool/ppm.page"
  cmp -s .vs3 "$VS3" || { echo ".vs3 changed by the run" > rc_reason.txt; return 8; }
  # where the prior run wrote, named by its origin raster (never changed later)
  if [ $rc = 0 ] && [ "$tool" = priors ]; then
    echo "$PRIORS $(stat -c%s "$PRIORS/ppmd-quelle.u8") $(sha "$PRIORS/ppmd-quelle.u8")" > AUSGABE || return 1
  fi
  [ $rc = 0 ] && cp "$WORK/stempel/$tool.soll" STEMPEL && date > .FERTIG
  return $rc
}
# platzwache <stop file> <mark file>: the disk watch of stage 2.  Every 5 s
# the free room of every file system the runs write to (WORK, PRIORS,
# PAGE_DIR); below DISK_RESERVE_GIB it stops the own coder runs (processes
# named cmix whose working directory is a run directory of this WORK, and a
# gate.sh of such a run that still waits for admission) and writes the mark.
# Every 10 minutes a projection from the progress of the prior run (raw
# priors so far / percent done) warns when the room will not last.
platzwache() {
  local stop=$1 marke=$2 d fr t=0 p s rest pid cwd
  local dirs=("$WORK" "$PRIORS"); [ -n "$PAGE_DIR" ] && dirs+=("$PAGE_DIR")
  while [ ! -e "$stop" ]; do
    for d in "${dirs[@]}"; do
      fr=$(frei_gib "$d")
      [ -n "$fr" ] || continue
      if [ "$fr" -lt "$DISK_RESERVE_GIB" ]; then
        echo "$d $fr" > "$marke"
        for pid in /proc/[0-9]*; do
          cwd=$(readlink "$pid/cwd" 2>/dev/null)
          [ "$cwd" = "$DD" ] || [ "$cwd" = "$DP" ] || continue
          case "$(cat "$pid/comm" 2>/dev/null)" in
            cmix) kill "${pid#/proc/}" 2>/dev/null ;;
            bash) tr '\0' '\n' < "$pid/cmdline" 2>/dev/null | sed -n 2p | grep -q 'gate\.sh$' && kill "${pid#/proc/}" 2>/dev/null ;;
          esac
        done
        return 0
      fi
    done
    if [ $((t % 120)) = 60 ] && [ -s "$DP/progress.log" ] && [ -e "$PRIORS/ppmd-probs.float16" ]; then
      p=$(tail -1 "$DP/progress.log" | awk '{printf "%d", $1 * 100}')      # percent x 100
      s=$(stat -c%s "$PRIORS/ppmd-probs.float16")
      if [ "${p:-0}" -ge 100 ] && [ "$p" -lt 10000 ]; then
        rest=$(( s / p * (10000 - p) / 1073741824 ))
        fr=$(frei_gib "$PRIORS")
        say "disk watch: prior run at $((p / 100)) %, raw priors $((s / 1073741824)) GiB, projected rest $rest GiB, free $fr GiB in PRIORS"
        [ "$rest" -gt $(( fr - DISK_RESERVE_GIB )) ] && say "!!! disk watch: the projection ($rest GiB more raw priors) exceeds the free room of PRIORS ($fr GiB - reserve $DISK_RESERVE_GIB GiB): free disk now, the runs stop at the reserve"
      fi
    fi
    sleep 5; t=$((t + 1))
  done
}
DD="$WORK/runs/dump"; DP="$WORK/runs/priors"
for t in dump priors; do
  lauf_soll $t nrun "eingang bed $BETT_SHA" "eingang vs3 $VS3_SHA" "eingang dict $DICT_SHA" "praefix ${PREFIX:-0}"
done
[ -e "$DD/.FERTIG" ] && stempel_pruefen dump "$DD/STEMPEL" "the dump run $DD"
if [ -e "$DP/.FERTIG" ]; then
  stempel_pruefen priors "$DP/STEMPEL" "the prior run $DP"
  # the output of the finished prior run must be in this call's PRIORS: the
  # origin raster ppmd-quelle.u8 is never changed, its size and sha256 name it
  if [ -s "$DP/AUSGABE" ]; then
    read -r A_DIR A_N A_SHA < "$DP/AUSGABE"
    if [ ! -s "$PRIORS/ppmd-quelle.u8" ] || [ "$(stat -c%s "$PRIORS/ppmd-quelle.u8")" != "$A_N" ] \
       || [ "$(sha "$PRIORS/ppmd-quelle.u8")" != "$A_SHA" ]; then
      fail "the finished prior run $DP wrote its priors into $A_DIR; PRIORS=$PRIORS does not hold them (ppmd-quelle.u8 $A_N B ${A_SHA:0:16}): give PRIORS=$A_DIR again, or remove $DP and start the prior run anew" 10
    fi
  fi
else
  [ -e "$PRIORS/ppmd-probs.float16" ] && fail "$PRIORS/ppmd-probs.float16 exists without a finished prior run (remove it or the run directory)" 7
fi
PWSTOP="$WORK/logs/platzwache.stop"; PWMARKE="$WORK/logs/platzwache.voll"; rm -f "$PWSTOP" "$PWMARKE"
if [ ! -e "$DD/.FERTIG" ] || [ ! -e "$DP/.FERTIG" ]; then
  ( platzwache "$PWSTOP" "$PWMARKE" ) & PW=$!
else
  PW=""
fi
( nrun dump "$DD" ) & P1=$!
( nrun priors "$DP" ) & P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
touch "$PWSTOP"; [ -n "$PW" ] && wait "$PW"
if [ -e "$PWMARKE" ]; then
  # the partial output of a stopped run is of no use: free the room now (the
  # output of a run that finished stays)
  [ -e "$DP/.FERTIG" ] || rm -f "$PRIORS/ppmd-probs.float16" "$PRIORS/ppmd-bytes.uint8" "$PRIORS/ppmd-quelle.u8"
  [ -e "$DD/.FERTIG" ] || rm -f "$DD/paket-zeilen.bin" "$DD/match-bytes.uint8" "$DD/match-len.uint8" "$DD/zonenmaske.u8"
  fail "disk: $(cut -d' ' -f1 "$PWMARKE") had $(cut -d' ' -f2 "$PWMARKE") GiB free (< reserve $DISK_RESERVE_GIB GiB): the dump and prior runs were stopped and their partial output removed; free disk (stage 2 needs about $([ -n "$PREFIX" ] && echo 22 || echo 230) GiB) and run the same command again" 12
fi
say "dump run rc=$R1 $(cat "$DD/rc_reason.txt" 2>/dev/null) ($(grep -a 'Elapsed\|Maximum resident' "$DD/stderr.txt" | sed 's/^[[:space:]]*//' | tr '\n' ' '))"
say "prior run rc=$R2 $(cat "$DP/rc_reason.txt" 2>/dev/null) ($(grep -a 'Elapsed\|Maximum resident' "$DP/stderr.txt" | sed 's/^[[:space:]]*//' | tr '\n' ' '))"
[ $R1 = 0 ] || fail "dump run rc=$R1 ($DD/stderr.txt)" 5
[ $R2 = 0 ] || fail "prior run rc=$R2 ($DP/stderr.txt)" 5
# slot 3 really packed: the text start line names the slot sizes, slot 3 = .vs3
for d in "$DD" "$DP"; do
  SL=$(tr '\r' '\n' < "$d/stderr.txt" | grep -a -o 'slot content [0-9]* / [0-9]* / [0-9]* B' | tail -1)
  [ -n "$SL" ] || fail "no 'slot content' line in $d/stderr.txt (text start not packed?)" 5
  S3=$(echo "$SL" | awk '{print $7}')
  [ "$S3" = "$(stat -c%s "$VS3")" ] || fail "$d: slot 3 holds $S3 B, .vs3 has $(stat -c%s "$VS3") B" 5
done
say "text start: $SL (slot 3 = .vs3)"
# the coder stream the runs saw: the zone mask has one byte per coded byte
# (6-byte segment header + stream); the full bed must give the whole coder
# stream, a probe bed a prefix of it (checked token by token below)
NR=$(stat -c%s "$DD/zonenmaske.u8")
if [ -z "$PREFIX" ]; then
  [ "$NR" = "$(( $(stat -c%s "$BK") + 6 ))" ] || fail "zone mask $NR B != coder stream $(stat -c%s "$BK") + 6: the dump run is incomplete" 5
  KOD="$BK"
else
  KOD="$WORK/runs/coder_prefix.bin"
  head -c $(( NR - 6 )) "$BK" > "$KOD"
fi
# completeness and identity: zone mask and match channel one byte per coded
# byte, every stream row of the dump carries the token of the coder stream,
# the last stream row is the last coded byte (the byte in front of it when
# the coded stream ends with a page anchor, a probe bed: the dump writes no
# row for the last byte of a page anchor)
python3 "$HERE/dump_check.py" "$DD" "$KOD" --anker "$ANKER_HEX" > "$WORK/logs/dump_check_dump.json" 2>&1 || fail "file check of the dump run ($WORK/logs/dump_check_dump.json)" 5
if [ ! -e "$WORK/STAGE4.ok" ] && [ ! -e "$WORK/STAGE4.begonnen" ] && [ ! -e "$WORK/STAGE4.getrimmt" ]; then
  python3 "$HERE/dump_check.py" "$DD" "$KOD" --prioren "$PRIORS" --kopf-laenge "$(stat -c%s "$BETT")" > "$WORK/logs/dump_check_priors.json" 2>&1 || fail "file check of the prior run ($WORK/logs/dump_check_priors.json)" 5
else
  say "file check of the prior run: done before the trimming began (logs/dump_check_priors.json); the raw raster is trimmed in place"
  [ -s "$WORK/logs/dump_check_priors.json" ] || fail "the trimming began but logs/dump_check_priors.json is missing: start with a fresh WORK" 5
fi
say "dump run complete: $NR coded bytes, $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["saetze"], "rows,", d["stromsaetze"], "stream rows,", d["eingefuegt"], "inserted")' "$WORK/logs/dump_check_dump.json")"
say "prior run complete: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["prior_zeilen"], "raster rows,", d["prior_strom"], "stream rows,", d["prior_eingespeist"], "injected")' "$WORK/logs/dump_check_priors.json")"
# PPMd lines: restarts in the prior run, the warm-ups in both runs
tr '\r' '\n' < "$DP/stderr.txt" | grep -a -E '^ppmd: |^ppmd warm-up ' > "$WORK/logs/ppmd_lines_priors.txt"
tr '\r' '\n' < "$DD/stderr.txt" | grep -a -E '^ppmd warm-up ' > "$WORK/logs/ppmd_lines_dump.txt"
grep -q '^ppmd: metablock end at model byte' "$WORK/logs/ppmd_lines_priors.txt" || fail "prior run: no PPMd restart line behind the metadata block" 5
W1=$(grep 'end of slot 1' "$WORK/logs/ppmd_lines_priors.txt" | tail -1); W2=$(grep 'end of slot 1' "$WORK/logs/ppmd_lines_dump.txt" | tail -1)
[ "$W1" = "$W2" ] || fail "warm-up at the end of slot 1 differs between the prior run ($W1) and the dump run ($W2)" 5
while read -r z; do say "  $z"; done < "$WORK/logs/ppmd_lines_priors.txt"

KLEIN="article-boundaries.int32 doc-last-ziel.int32 doc-rows.int64 title-lengths.int32 z3-prior0.float16 dump-meta.json kennzahlen.json vocab-bitmap.u8 zonenmaske.u8 zeilen.dump seiten.tsv"
W="$WORK/montage"

# Stages 3 and 4 leave a mark (STAGE3.ok, STAGE4.ok) and a stamp
# (STAGE3.stempel, STAGE4.stempel): a run that stopped in stage 5 is resumed
# with the same command; the priors are trimmed in place and must not be
# trimmed twice.  The stamp of a stage names the stamps of the runs it read,
# the sha256 of its scripts and of its part of this script, the arms, the
# geometry and the output directories; a resume with anything else stops.
# The loader check against the raw dump runs in stage 3 and again in stage 5
# (with the training tree of the call), so it is not part of the stamp.
Z3=("stufe 3" "lauf dump $(sha "$DD/STEMPEL")" "lauf bed $(sha "$B/STEMPEL")" "arme $ARME"
    "geometrie ${MKOPT[*]}" "vocab $(sha "$VOCAB_BITMAP")" "praefix ${PREFIX:-0}"
    "seiten $SEITEN anker $ANKER_HEX"
    "code $(abschnitt '# ---- 3. ' '# ---- 4. ' | sha256sum | cut -c1-64)")
for f in zone_cut.py mk_pkgdump.py dumpv2.py pages.py page_anchors.py title_lengths.py ppmd_bytes.py; do
  Z3+=("skript $f $(sha "$HERE/$f")")
done
stempel_soll stage3 "${Z3[@]}"
Z4=("stufe 4" "stufe3 $(sha "$WORK/stempel/stage3.soll")" "lauf priors $(sha "$DP/STEMPEL")" "breite $ZB"
    "code $(abschnitt '# ---- 4. ' '# ---- 5. ' | sha256sum | cut -c1-64)")
for f in trim_quelle.py trim.py trim_inplace.py; do Z4+=("skript $f $(sha "$HERE/$f")"); done
stempel_soll stage4 "${Z4[@]}"
if [ -e "$WORK/STAGE3.ok" ]; then
  stempel_pruefen stage3 "$WORK/STAGE3.stempel" "stage 3 (package $PKG)"
  # shellcheck disable=SC1091
  . "$WORK/STAGE3.ok"
  # the package of stage 3 must be in this call's PKG (its checksum list names it)
  if [ -s "$WORK/STAGE3.pkg" ]; then
    read -r A_DIR A_SHA < "$WORK/STAGE3.pkg"
    [ -s "$PKG/SHA256SUMS_klein" ] && [ "$(sha "$PKG/SHA256SUMS_klein")" = "$A_SHA" ] \
      || fail "stage 3 wrote its package into $A_DIR; PKG=$PKG does not hold it (SHA256SUMS_klein ${A_SHA:0:16}): give PKG=$A_DIR again, or start with a fresh WORK" 10
  fi
  say "stage 3 done earlier: token_start $TS, n_tokens $NTOK, skip $SKIP"
else
# ---- 3. the package from the dump
mkdir -p "$W" "$PKG"
[ -e "$PKG/zeilen.dump" ] && fail "zeilen.dump already in $PKG (remove $PKG to build the package again)" 9
# room: zeilen.dump and the tar (about twice the raw dump) and four tables of
# one byte per coded byte, plus 1 GiB
platz_vor $(( (2 * $(stat -c%s "$DD/paket-zeilen.bin") + 4 * NR) / 1073741824 + 1 )) "$PKG" "stage 3 (package)"
# 3.1 cut = end of the first zone-3 run (metadata block + census metablock) + 1;
#     a further zone-3 run (title line of page 0 behind the slots) stays
python3 "$HERE/zone_cut.py" cut "$DD/zonenmaske.u8" > "$W/cut.txt" || fail "zone_cut cut" 3
cat "$W/cut.txt" >> "$L"
TS=$(tail -1 "$W/cut.txt")
NTOK=$(( NR - TS ))
python3 "$HERE/zone_cut.py" gate "$DD/zonenmaske.u8" "$TS" "$W/gate_cut.u8" >> "$L" 2>&1 || fail "zone_cut gate" 3
SKIP=$(python3 "$HERE/zone_cut.py" skip "$DD/paket-zeilen.bin" "$TS" 2>> "$L") || fail "zone_cut skip" 3
say "cut (token_start) $TS, n_tokens $NTOK, skip $SKIP"
# 3.2 mk_pkgdump, with the arms and the geometry of the end line
mem_vor 93 "mk_pkgdump (stage 3.2; measured 41.2 GiB on 97.6 % of the stream)"
( echo 1000 > /proc/self/oom_score_adj
  exec /usr/bin/time -v nice -n 5 python3 "$HERE/mk_pkgdump.py" \
    "$DD/paket-zeilen.bin" "$PKG" --gate "$W/gate_cut.u8" --n-tokens "$NTOK" \
    --skip "$SKIP" --shift "$TS" "${MKOPT[@]}" --arme "$ARME" ) > "$WORK/logs/mk_pkgdump.log" 2>&1 || fail "mk_pkgdump" 3
[ -s "$PKG/zeilen.dump" ] || fail "no zeilen.dump" 3
grep -a -E 'longest|no document over|Probe0|contradictions|without closing' "$WORK/logs/mk_pkgdump.log" | sed 's/^/  /' >> "$L"
# 3.3 page table of the coder stream (anchor of the tree, page count of the
#     input) and the title lengths; the page table goes into the package
#     (seiten.tsv: the title rows of Z9 on the training host)
python3 "$HERE/pages.py" "$KOD" "$WORK/pages.tsv" "$TS" --zonen "$DD/zonenmaske.u8" --anker "$ANKER_HEX" \
  --seiten "$SEITEN" ${PREFIX:+--praefix} >> "$L" 2>&1 || fail "page table" 6
cp "$WORK/pages.tsv" "$PKG/seiten.tsv"
python3 "$HERE/title_lengths.py" "$PKG/article-boundaries.int32" "$WORK/pages.tsv" \
  "$PKG/title-lengths.int32" --bett-off $(( TS - 6 )) >> "$L" 2>&1 || fail "title-lengths" 3
# 3.4 zone mask and byte set into the package
cp "$W/gate_cut.u8" "$PKG/zonenmaske.u8"
cp "$VOCAB_BITMAP" "$PKG/vocab-bitmap.u8"
# 3.5 ppmd-bytes of the package from the coder stream, prior0; dump stream rows against the stream
python3 "$HERE/ppmd_bytes.py" "$DD/paket-zeilen.bin" "$TS" "$NTOK" "$PKG" "$KOD" >> "$L" 2>&1 || fail "ppmd-bytes/prior0" 7
say "prior0 of the package $(sha "$PKG/z3-prior0.float16")"
# 3.6 the loader against the raw dump; the loader needs a priors file of the
#     right length, whose values this check does not read (a sparse placeholder)
truncate -s $(( $(stat -c%s "$PKG/ppmd-bytes.uint8") * ZB )) "$PKG/ppmd-probs.float16"
torchlauf "$PYTHON_TORCH" "$HERE/loader_vs_dump.py" "$PKG" "$DD/paket-zeilen.bin" "$TS" "$SKIP" > "$WORK/logs/loader_vs_dump.log" 2>&1 || fail "loader_vs_dump ($WORK/logs/loader_vs_dump.log)" 3
say "loader_vs_dump: $(tail -1 "$WORK/logs/loader_vs_dump.log")"
rm -f "$PKG/ppmd-probs.float16"
# 3.7 the package as one tar (sorted by name) and the checksums of the small files
# shellcheck disable=SC2086
( cd "$PKG" && tar -cf pkg.tar --sort=name $KLEIN && sha256sum $KLEIN ppmd-bytes.uint8 pkg.tar > SHA256SUMS_klein ) || fail "tar" 3
say "tar: $(grep pkg.tar "$PKG/SHA256SUMS_klein")"
cp "$WORK/stempel/stage3.soll" "$WORK/STAGE3.stempel"
echo "$PKG $(sha "$PKG/SHA256SUMS_klein")" > "$WORK/STAGE3.pkg"
printf 'TS=%s\nNTOK=%s\nSKIP=%s\n' "$TS" "$NTOK" "$SKIP" > "$WORK/STAGE3.ok"
fi

if [ -e "$WORK/STAGE4.ok" ]; then
  stempel_pruefen stage4 "$WORK/STAGE4.stempel" "stage 4 (priors $PRIORS, trimmed)"
  say "stage 4 done earlier: priors trimmed and probed"
else
# ---- 4. trimming of the priors: the raw raster has one row per coder step
#         (coded bytes plus bytes injected through all models); the package
#         keeps the rows of coded bytes behind the cut.  ppmd-quelle.u8 says
#         which row is which.  Marks: STAGE4.begonnen is written before the
#         priors are changed (from then on the raw raster check of stage 2 is
#         skipped, the trimming resumes with its journal), STAGE4.getrimmt
#         after the trimming and the swap of the bytes (a resume probes
#         again).  The raw bytes stay in PRIORS as ppmd-bytes_roh.uint8.
O="$WORK/trim"
NQ=$(stat -c%s "$PRIORS/ppmd-quelle.u8")
RB="$PRIORS/ppmd-bytes.uint8"
if [ -e "$WORK/STAGE4.getrimmt" ]; then
  stempel_pruefen stage4 "$WORK/STAGE4.getrimmt" "the trimmed priors of an earlier call ($PRIORS)"
  say "stage 4: priors trimmed in an earlier call, the probes run again"
  RB="$PRIORS/ppmd-bytes_roh.uint8"; mkdir -p "$O"
elif [ -e "$WORK/STAGE4.begonnen" ]; then
  stempel_pruefen stage4 "$WORK/STAGE4.begonnen" "the trimming begun in an earlier call ($PRIORS)"
  say "stage 4: the trimming of $PRIORS began in an earlier call and is resumed"
  [ -e "$PRIORS/ppmd-bytes_roh.uint8" ] && RB="$PRIORS/ppmd-bytes_roh.uint8"
  mkdir -p "$O"
else
  [ -e "$PRIORS/ppmd-bytes_roh.uint8" ] && fail "$PRIORS/ppmd-bytes_roh.uint8 exists but no stage-4 mark: the priors are in an unknown state, start with a fresh WORK" 6
  rm -rf "$O"; mkdir -p "$O"
fi
[ -s "$RB" ] && [ "$(stat -c%s "$RB")" = "$NQ" ] || fail "raw ppmd-bytes $RB missing or not of the length of the origin raster ($NQ rows)" 6
# room: the trim mask and the stream bytes in WORK, the package bytes in PRIORS
platz_vor $(( 2 * NQ / 1073741824 + 1 )) "$O" "stage 4 (trim mask)"
platz_vor $(( NTOK / 1073741824 + 1 )) "$PRIORS" "stage 4 (package bytes)"
python3 "$HERE/trim_quelle.py" "$PRIORS/ppmd-quelle.u8" "$TS" "$NR" "$RB" "$KOD" \
  "$O/trimmaske.u8" "$O/ppmd-bytes_strom.uint8" 2>&1 | tee -a "$L"
[ "${PIPESTATUS[0]}" = 0 ] || fail "trim mask (ppmd-quelle)" 6
[ "$NQ" = "$(stat -c%s "$O/trimmaske.u8")" ] || fail "raster length != mask length" 6
if [ ! -e "$WORK/STAGE4.getrimmt" ]; then
  python3 "$HERE/trim.py" "$O/trimmaske.u8" "$RB" "$PRIORS/ppmd-bytes_paket.uint8" 1 >> "$L" 2>&1 || fail "trim bytes" 6
  [ -e "$WORK/STAGE4.begonnen" ] || cp "$WORK/stempel/stage4.soll" "$WORK/STAGE4.begonnen"
  nice -n 5 python3 "$HERE/trim_inplace.py" "$O/trimmaske.u8" "$PRIORS/ppmd-probs.float16" "$ZB" >> "$L" 2>&1 || fail "trim_inplace (resume with the same command: it continues with its journal)" 6
  # swap in one directory (renames): the raw bytes keep their content under a
  # new name, the package bytes take the name; a resume finds either state
  [ "$RB" = "$PRIORS/ppmd-bytes.uint8" ] && { mv "$PRIORS/ppmd-bytes.uint8" "$PRIORS/ppmd-bytes_roh.uint8" || fail "rename of the raw bytes" 6; }
  mv "$PRIORS/ppmd-bytes_paket.uint8" "$PRIORS/ppmd-bytes.uint8" || fail "rename of the package bytes" 6
  cp "$WORK/stempel/stage4.soll" "$WORK/STAGE4.getrimmt"
  say "stage 4: priors trimmed ($(stat -c%s "$PRIORS/ppmd-probs.float16") B), raw bytes kept as ppmd-bytes_roh.uint8"
fi

# ---- 4.1 probes of the trimmed priors; prior0 again from the trimmed bytes
python3 - "$PRIORS" "$O" "$KOD" "$WORK" "$TS" <<'PY' 2>&1 | tee -a "$L"
import sys, os, hashlib, json, numpy as np
Z, O, BETT, W, SCHNITT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    return h.hexdigest()
bm = open(os.environ["VOCAB_BITMAP"], "rb").read(32)
menge = np.array([v for v in range(256) if (bm[v >> 3] >> (v & 7)) & 1], dtype=np.int64)
V = len(menge)
lut = np.full(256, 255, np.uint8); lut[menge] = np.arange(V, dtype=np.uint8)
res = {"vokabular": V}
st = np.fromfile(O + "/ppmd-bytes_strom.uint8", dtype=np.uint8)
bett = np.fromfile(BETT, dtype=np.uint8); n = len(bett)
kopf = np.array([1] + [0x80 | ((n >> s) & 0x7F) for s in (28, 21, 14, 7, 0)], dtype=np.uint8)
res["stream_raster_equals_header_plus_coder_stream"] = bool(len(st) == n + 6 and (st[:6] == lut[kopf]).all() and (st[6:] == lut[bett]).all())
bt = np.fromfile(Z + "/ppmd-bytes.uint8", dtype=np.uint8); N = len(bt); res["n_tokens"] = N; res["token_start"] = SCHNITT
res["package_equals_stream_behind_cut"] = bool(len(bt) == len(st) - SCHNITT and (bt == st[SCHNITT:]).all())
c = np.bincount(bt, minlength=256)[:V].astype(np.float64)
p0 = ((c + 1.0) / (c.sum() + float(V))).astype(np.float16); p0.tofile(Z + "/z3-prior0.float16")
res["prior0_sha"] = sha(Z + "/z3-prior0.float16")
res["ppmd_bytes_sha"] = sha(Z + "/ppmd-bytes.uint8")
P = Z + "/ppmd-probs.float16"; sz = os.path.getsize(P)
res["probs_bytes"] = sz; res["probs_length_ok"] = sz == N * 2 * V
res["probs_allocated_share"] = round(os.stat(P).st_blocks * 512 / sz, 4)
a = np.memmap(P, dtype=np.float16, mode="r").reshape(-1, V)
idx = np.unique(np.linspace(0, N - 2, 4000).astype(np.int64))
rows = a[idx].astype(np.float64); s = rows.sum(1)
res["samples"] = int(len(idx)); res["row_sum_min"] = float(s.min()); res["row_sum_max"] = float(s.max())
res["samples_zero"] = int((rows.max(1) == 0).sum())
pn = rows[np.arange(len(idx)), bt[idx + 1]]
res["ce_bit"] = float(-np.log2(np.maximum(pn, 1e-12)).mean())
res["probs_sha"] = sha(P)
ok = (res["stream_raster_equals_header_plus_coder_stream"] and res["package_equals_stream_behind_cut"] and res["probs_length_ok"]
      and res["probs_allocated_share"] > 0.99 and res["samples_zero"] == 0
      and 0.99 < res["row_sum_min"] and res["row_sum_max"] < 1.01)
res["PROBES_OK"] = bool(ok)
json.dump(res, open(W + "/proben_p.json", "w"), indent=1); print(json.dumps(res, indent=1))
open(Z + "/PROBS_SHA256", "w").write(res["probs_sha"] + "\n")
sys.exit(0 if ok else 4)
PY
[ "${PIPESTATUS[0]}" = 0 ] || fail "probes failed (proben_p.json)" 4
say "probes passed; priors $(cut -c1-16 "$PRIORS/PROBS_SHA256")"
( cd "$PKG" && sha256sum -c --quiet SHA256SUMS_klein ) || fail "package files != SHA256SUMS_klein" 3
B2=$(sha "$PRIORS/ppmd-bytes.uint8"); BS=$(awk '$2=="ppmd-bytes.uint8"{print $1}' "$PKG/SHA256SUMS_klein")
P2=$(sha "$PRIORS/z3-prior0.float16"); P3=$(sha "$PKG/z3-prior0.float16")
[ "$B2" = "$BS" ] && [ "$P2" = "$P3" ] || fail "raster/prior0 of priors and package differ" 4
say "ppmd-bytes and prior0: priors = package; prior0 ${P3:0:16}: the coder needs it as a prior0 set (tf1_prior0_set.h) before a TF coder run"
cp "$WORK/stempel/stage4.soll" "$WORK/STAGE4.stempel"
date > "$WORK/STAGE4.ok"
fi

# ---- 5. checks of the whole package: a directory of hard links (package +
#         priors) on the file system of PRIORS; the small package files are
#         copied when PKG lies on another one
( cd "$PKG" && sha256sum -c --quiet SHA256SUMS_klein ) > "$WORK/logs/pkg_sha.log" 2>&1 \
  || fail "package files in $PKG != SHA256SUMS_klein ($(head -3 "$WORK/logs/pkg_sha.log" | tr '\n' ' '))" 3
rm -rf "$PK"; mkdir -p "$PK"
if [ "$(stat -c %d "$PKG")" = "$(stat -c %d "$PK")" ]; then
  for f in $KLEIN; do ln -f "$PKG/$f" "$PK/$f" || fail "hard link $PKG/$f" 3; done
else
  KB=0; for f in $KLEIN; do KB=$(( KB + $(stat -c%s "$PKG/$f") )); done
  platz_vor $(( KB / 1073741824 + 1 )) "$PK" "stage 5 (copies of the package files next to the priors)"
  for f in $KLEIN; do cp "$PKG/$f" "$PK/$f" || fail "copy $PKG/$f" 3; done
fi
ln -f "$PRIORS/ppmd-probs.float16" "$PK/ppmd-probs.float16" && ln -f "$PRIORS/ppmd-bytes.uint8" "$PK/ppmd-bytes.uint8" \
  || fail "hard links of the priors into $PK" 3
# 5.1 Z table; the text-form facts: page table, slot sizes of the dump run,
#     the slot end mark and the forced title prefixes of the tree
#     (TREE: tf1/tf1_block_set2.h; ENTRY: ../1_dump/forced_title_prefixes.h)
mem_vor 100 "the Z table (stage 5.1; measured 44.1 GiB on 97.6 % of the stream)"
( echo 1000 > /proc/self/oom_score_adj
  exec nice -n 5 /usr/bin/time -f "rss_kb %M wall %e" python3 "$HERE/package_checks.py" "$PK" \
    --prioren "$PK/ppmd-probs.float16" --ppmd-bytes "$PK/ppmd-bytes.uint8" --gate "$PK/zonenmaske.u8" \
    --strombytes "$NR" --token-start "$TS" --grenzen "$WORK/pages.tsv" --token-start-strom $(( TS - 6 )) \
    --slots "$SL" --zwang-tafel "$ZWANG" --endmarke "$ENDMARKE" \
    --prioren-quelle "$PRIORS/ppmd-probs.float16" --stichproben 64 --json "$WORK/ztable.json" ) \
  > "$WORK/logs/ztable.log" 2>&1
ZT=$?
ROTZ=$(grep "^| Z" "$WORK/logs/ztable.log" | grep "RED" | awk -F"|" '{gsub(/ /,"",$2); print $2}' | tr '\n' ' ')
say "Z table rc=$ZT, red rows: '${ROTZ}' ($(grep -a '^text form\|^rss_kb' "$WORK/logs/ztable.log" | cut -c1-300 | tr '\n' ' '))"
[ $ZT = 0 ] || fail "Z table red: $ROTZ ($WORK/logs/ztable.log)" 5
# 5.2 the loader against the raw dump with the training tree of this call
#     (stage 3 ran it with the tree of the call that built the package)
torchlauf "$PYTHON_TORCH" "$HERE/loader_vs_dump.py" "$PK" "$DD/paket-zeilen.bin" "$TS" "$SKIP" > "$WORK/logs/loader_vs_dump5.log" 2>&1
LV=$?; grep -q "LOADER VS DUMP PASSED" "$WORK/logs/loader_vs_dump5.log" || LV=77
say "loader_vs_dump (stage 5) rc=$LV: $(tail -1 "$WORK/logs/loader_vs_dump5.log")"
[ $LV = 0 ] || fail "loader_vs_dump ($WORK/logs/loader_vs_dump5.log)" 5
# 5.3 interface check of the trainer
( cd "$TRAIN_TREE" && exec env PYTHONPATH="${PYLIB:-}" OMP_NUM_THREADS="$PY_TEAM" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 nice -n 5 "$PYTHON_TORCH" -m tft.dataset check "$PK" ) > "$WORK/logs/dataset_check.log" 2>&1
DC=$?; grep -q "package matches the training-data interface" "$WORK/logs/dataset_check.log" || DC=77
say "tft.dataset check rc=$DC: $(tail -2 "$WORK/logs/dataset_check.log" | tr '\n' ' ')"
[ $DC = 0 ] || fail "dataset check" 5
# 5.4 the real loader over every micro-batch; the epoch must be complete (the
#     prefetch thread of the trainer loader ends it early on an exception):
#     micro-batches = the loader's own count, scored rows = the Z table
BEW_SOLL=$(python3 -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["bewertet"]))' "$WORK/ztable.json") \
  || fail "no scored count in $WORK/ztable.json" 5
( echo 1000 > /proc/self/oom_score_adj; exec nice -n 5 env OMP_NUM_THREADS="$PY_TEAM" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$PYTHON_TORCH" "$HERE/loader_probe.py" "$PK" --soll-bewertet "$BEW_SOLL" ) > "$WORK/logs/loader_probe.log" 2>&1
LP=$?; grep -q "^LOADER PROBE PASSED" "$WORK/logs/loader_probe.log" || [ $LP != 0 ] || LP=77
say "loader_probe rc=$LP: $(tail -1 "$WORK/logs/loader_probe.log")"
[ $LP = 0 ] || fail "loader probe" 5

# ---- 5.5 manifest (Z8): what the package is and where it comes from
python3 "$HERE/paket_manifest.py" --pkg "$PKG" --priors "$PRIORS" --work "$WORK" \
  --tree-commit "$TREE_COMMIT" --tree-sha "$TREE_SHA" --tools "$T" --arme "$ARME" \
  --geometrie "$WORK/logs/arms_end.txt" --token-start "$TS" --n-tokens "$NTOK" --skip "$SKIP" \
  --strombytes "$NR" --praefix "$BETT_N" \
  --zeile stream="$FEAT_STREAM" --zeile dump="$FEAT_DUMP" --zeile priors="$FEAT_PRIORS" --zeile end="$FEAT_END" \
  --zeile bed="$WORK/logs/features_bed_line.txt" \
  --eingang enwik9="$ENWIK9" --eingang dict="$DICT" --eingang order="$ORDER" \
  --eingang canonical_stream="$KANON" --eingang vs3="$VS3" --eingang coder_stream="$BK" --eingang bed="$BETT" \
  --eingang coder_stream_of_the_runs="$KOD" \
  --json "$WORK/ztable.json" --json "$WORK/proben_p.json" --json "$WORK/logs/dump_check_dump.json" \
  --json "$WORK/logs/dump_check_priors.json" --zeilenvgl "$WORK/logs/zeilen_vgl.json" \
  --ppmd "$WORK/logs/ppmd_lines_priors.txt" --slots "$SL" \
  --stempel bed="$B/STEMPEL" --stempel stream="$S/STEMPEL" --stempel dump="$DD/STEMPEL" --stempel priors="$DP/STEMPEL" \
  --stempel stage3="$WORK/STAGE3.stempel" --stempel stage4="$WORK/STAGE4.stempel" --trainerbaum "$TT_SHA" \
  > "$WORK/logs/manifest.log" 2>&1 || fail "manifest ($WORK/logs/manifest.log)" 3
cp "$PKG/MANIFEST.json" "$PK/MANIFEST.json"
# 5.6 the acceptance of the training side against the manifest (A to H, Z8);
#     it runs the Z table again (with the page table of the package for Z9)
mem_vor 100 "the acceptance (stage 5.6, the Z table again)"
( echo 1000 > /proc/self/oom_score_adj; exec env OMP_NUM_THREADS="$PY_TEAM" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$PYTHON_TORCH" "$HERE/abnahme_paket.py" "$PK" --ohne-prioren-sha ) > "$WORK/logs/abnahme.log" 2>&1
AB=$?; say "acceptance rc=$AB: $(tail -1 "$WORK/logs/abnahme.log")"
[ $AB = 0 ] || fail "acceptance ($WORK/logs/abnahme.log)" 5
( cd "$PK" && sha256sum -- * > SHA256SUMS_all 2>/dev/null )
say "=== training package complete${PREFIX:+ (PROBE on $BETT_N B, PREFIX $PREFIX)}: $PKG, priors $PRIORS, checks in $PK"
exit 0
}
