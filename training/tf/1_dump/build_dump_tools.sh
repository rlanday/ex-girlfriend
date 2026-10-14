#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
#
# Builds the three tools of stage (b) from a copy of the source of this entry
# with trainmodes.patch applied.  The source of the entry is not changed.
#
#   build_dump_tools.sh <entry directory> <output directory>
#
#   <entry directory>   the unpacked source package (it holds src/ and makefile)
#   <output directory>  receives cmix_stream, cmix_dump, cmix_priors and the
#                       build logs; build_<tool>/ holds each patched copy
#
# Toolchain as for build.sh (install.sh): make, clang++-18, ld.lld-18, plus
# patch.  Plain builds without profile, LTO or packing, like the builds of the
# original runs (clang 17); README.md, section 4, compares the outputs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ $# -eq 2 ] || { echo "usage: $0 <entry directory> <output directory>" >&2; exit 2; }
SRC="$(cd "$1" && pwd)"
mkdir -p "$2"; OUT="$(cd "$2" && pwd)"
PATCH="$HERE/trainmodes.patch"
for t in make patch clang++-18 ld.lld-18; do
  command -v "$t" >/dev/null 2>&1 || { echo "$t is missing" >&2; exit 1; }
done
for f in src makefile encoder_split.lds runtime_exports.map; do
  [ -e "$SRC/$f" ] || { echo "$SRC/$f is missing" >&2; exit 1; }
done

# mode 1 stream, 2 dump, 3 priors (src/train_mode.h after the patch)
for tm in stream:1 dump:2 priors:3; do
  tool=${tm%%:*}; mode=${tm##*:}
  W="$OUT/build_$tool"
  rm -rf "$W"; mkdir -p "$W"
  cp -a "$SRC/src" "$SRC/makefile" "$SRC/encoder_split.lds" "$SRC/runtime_exports.map" "$W/"
  ( cd "$W" && patch -p1 --no-backup-if-mismatch -s < "$PATCH" ) \
    || { echo "patch does not apply to $SRC/src" >&2; exit 1; }
  make -C "$W" cmix CC="clang++-18 -DTRAIN_MODE=$mode" > "$OUT/build_$tool.log" 2>&1 \
    || { echo "build of cmix_$tool failed, see $OUT/build_$tool.log" >&2; exit 1; }
  cp "$W/cmix" "$OUT/cmix_$tool"
  echo "cmix_$tool $(sha256sum "$OUT/cmix_$tool" | cut -d' ' -f1)"
done
