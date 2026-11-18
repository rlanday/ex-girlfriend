#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
#
# Builds the tools of stage (b) and the trace tool of BITLSTM.
#
#   build_dump_tools.sh <entry directory> <output directory> [tool ...]
#   build_dump_tools.sh --tree <development tree> <output directory>
#
# <entry directory>: the unpacked source package of this entry.  A copy of its
#   src/ and makefile gets trainmodes.patch of this directory; each tool is a
#   plain build of that copy (`make cmix`, clang++-18, no profile, no LTO, no
#   packing) with one value of TRAIN_MODE:
#     cmix_stream  1  "-e" writes the coder stream (out.cmix.temp) and stops
#                     before the coder
#     cmix_dump    2  "-n" on the bed: dump rows, zone mask, match channel
#     cmix_priors  3  "-n" on the bed: PPMd priors, their byte tokens and origin
#     cmix_bed     4  "-e" writes the canonical stream (the bed of the "-n"
#                     runs), the census pseudo article of slot 3 (.vs3 next to
#                     the bed) and the coder stream, and stops before the coder
#     cmix_trace   5  "-e" with the transformer: the BITLSTM trace
#                     (../../bitl/README.md, section 1)
#   Without TRAIN_MODE the patched copy builds the compressor of the package
#   unchanged.  [tool ...] limits the build to the named tools (default all).
#
# --tree: the development tree holds src/cmix-neifv9; each tool is a plain
#   build (`make cmix`, clang++-17) of a copy of src/cmix-neifv9 with one build
#   line of this directory:
#     cmix_bed     features_stream.txt + CENSBLOCK_PRUEF=1
#     cmix_stream  features_stream.txt (PREPARE_ONLY)
#     cmix_dump    features_dump.txt (TF1_DUMPONLY, TF1_PAKET_DUMP)
#     cmix_priors  features_priors.txt (PPMD_ONLY)
#   A build line that does not build, a "macro redefined" or an "#error" in a
#   build log stops the script.  FEAT_STREAM, FEAT_DUMP and FEAT_PRIORS name
#   other line files (build_package.sh passes the ones it was given).
#
# Output: cmix_<tool>, build_<tool>.log, and a line "cmix_<tool> <sha256>" per
# tool on stdout; --tree also features_<tool>.txt (the resolved line,
# `make features`).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--tree" ]; then
  [ $# -eq 3 ] || { echo "usage: $0 --tree <development tree> <output directory>" >&2; exit 2; }
  TREE="$(cd "$2" && pwd)"
  SRC="$TREE/src/cmix-neifv9"
  [ -f "$SRC/makefile" ] || { echo "$SRC/makefile is missing" >&2; exit 1; }
  mkdir -p "$3"; OUT="$(cd "$3" && pwd)"
  for t in make clang++-17; do
    command -v "$t" >/dev/null 2>&1 || { echo "$t is missing" >&2; exit 1; }
  done
  # the line files: FEAT_STREAM, FEAT_DUMP, FEAT_PRIORS override the files of
  # this directory (build_package.sh passes its own)
  datei() { case $1 in stream) echo "${FEAT_STREAM:-$HERE/features_stream.txt}" ;;
            dump) echo "${FEAT_DUMP:-$HERE/features_dump.txt}" ;; priors) echo "${FEAT_PRIORS:-$HERE/features_priors.txt}" ;; esac; }
  zeile() { tr -d '\r\n' < "$(datei "$1")"; }
  for f in stream dump priors; do [ -s "$(datei $f)" ] || { echo "$(datei $f) is missing" >&2; exit 1; }; done
  baue() {   # baue <tool> <line>
    local tool=$1 line=$2 W="$OUT/build_$1"
    rm -rf "$W"; mkdir -p "$W"
    cp -a "$SRC/." "$W/"
    ( cd "$W" && make -s features FEATURES="$line" ) > "$OUT/features_$tool.txt" 2>&1 \
      || { echo "make features of cmix_$tool failed, see $OUT/features_$tool.txt" >&2; return 1; }
    if ! ( cd "$W" && make -s -j"${JOBS:-8}" cmix FEATURES="$line" ) > "$OUT/build_$tool.log" 2>&1; then
      echo "build of cmix_$tool failed, see $OUT/build_$tool.log" >&2; return 1
    fi
    if grep -qa 'macro redefined\|#error' "$OUT/build_$tool.log"; then
      echo "cmix_$tool: 'macro redefined' or '#error' in $OUT/build_$tool.log" >&2; return 1
    fi
    cp "$W/cmix" "$OUT/cmix_$tool"
    rm -rf "$W"
  }
  pids=()
  baue bed "$(zeile stream) CENSBLOCK_PRUEF=1" & pids+=($!)
  baue stream "$(zeile stream)" & pids+=($!)
  baue dump "$(zeile dump)" & pids+=($!)
  baue priors "$(zeile priors)" & pids+=($!)
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ $rc = 0 ] || exit 1
  for tool in bed stream dump priors; do
    echo "cmix_$tool $(sha256sum "$OUT/cmix_$tool" | cut -d' ' -f1)"
  done
  exit 0
fi

[ $# -ge 2 ] || { echo "usage: $0 <entry directory> <output directory> [tool ...] | $0 --tree <development tree> <output directory>" >&2; exit 2; }
SRC="$(cd "$1" && pwd)"
mkdir -p "$2"; OUT="$(cd "$2" && pwd)"
shift 2
TOOLS=("$@"); [ ${#TOOLS[@]} -gt 0 ] || TOOLS=(stream dump priors bed trace)
PATCH="$HERE/trainmodes.patch"
for t in make patch clang++-18 ld.lld-18; do
  command -v "$t" >/dev/null 2>&1 || { echo "$t is missing" >&2; exit 1; }
done
for f in src makefile encoder_split.lds runtime_exports.map; do
  [ -e "$SRC/$f" ] || { echo "$SRC/$f is missing" >&2; exit 1; }
done
modus() { case $1 in stream) echo 1 ;; dump) echo 2 ;; priors) echo 3 ;; bed) echo 4 ;; trace) echo 5 ;;
          *) echo "unknown tool $1 (stream, dump, priors, bed, trace)" >&2; exit 2 ;; esac; }
for tool in "${TOOLS[@]}"; do
  mode=$(modus "$tool")
  W="$OUT/build_$tool"
  rm -rf "$W"; mkdir -p "$W"
  cp -a "$SRC/src" "$SRC/makefile" "$SRC/encoder_split.lds" "$SRC/runtime_exports.map" "$W/"
  ( cd "$W" && patch -p1 --no-backup-if-mismatch -s < "$PATCH" ) \
    || { echo "trainmodes.patch does not apply to $SRC" >&2; exit 1; }
  make -C "$W" -j"${JOBS:-8}" cmix TRAIN_MODE=$mode CC="clang++-18 -DTRAIN_MODE=$mode" > "$OUT/build_$tool.log" 2>&1 \
    || { echo "build of cmix_$tool failed, see $OUT/build_$tool.log" >&2; exit 1; }
  if grep -qa 'macro redefined\|#error' "$OUT/build_$tool.log"; then
    echo "cmix_$tool: 'macro redefined' or '#error' in $OUT/build_$tool.log" >&2; exit 1
  fi
  cp "$W/cmix" "$OUT/cmix_$tool"
  rm -rf "$W"
  echo "cmix_$tool $(sha256sum "$OUT/cmix_$tool" | cut -d' ' -f1)"
done
