#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
#
# The manifest arms and the package geometry of a coder line, read from the
# source, not written by hand.
#
#   manifest_arms.sh <source tree> <line file>
#
# <source tree> holds the makefile and src/tf1/tf1_manifest.h (src/cmix-neifv9
# of the development tree, or the source of an entry).  The script resolves
# the line with `make features` (no build), compiles a small probe with the
# resulting defines against src/tf1/tf1_manifest.h and prints what the coder
# of this line checks a weight container against (neif_tf1_manifest::Arme())
# and the constants the package rows depend on:
#
#   arme=<comma list>   the arm list; the package must carry exactly this list
#                       (the coder compares the sets, tf1_manifest.h Check())
#   v nsent x dslot     vocabulary width, sentinel rows, X2 slots, slot width
#   x0row redirmerge    build state of the dump rows (mk_pkgdump --x0row/--redirmerge)
#   fenster nl          attention window and number of layers
#   prior0_set          stamped prior0 table
#
# Exit 0 with the lines above on stdout; 2 when the line does not resolve or
# the probe does not compile.  Needs make and clang++-17 (CXX_PROBE overrides).
set -uo pipefail
[ $# -eq 2 ] || { sed -n 5,24p "$0"; exit 2; }
SRC="$(cd "$1" && pwd)" || exit 2
ZEILE="$(tr -d '\r\n' < "$2")" || exit 2
CXX="${CXX_PROBE:-clang++-17}"
[ -f "$SRC/makefile" ] && [ -f "$SRC/src/tf1/tf1_manifest.h" ] || { echo "no makefile or src/tf1/tf1_manifest.h in $SRC" >&2; exit 2; }
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
out=$(cd "$SRC" && make -s features FEATURES="$ZEILE" 2>&1) || { echo "make features failed: $(grep -m1 -i 'error' <<<"$out")" >&2; exit 2; }
grep -q '^makefile:[0-9]*: \*\*\*' <<<"$out" && { echo "make features: $(grep -m1 '\*\*\*' <<<"$out")" >&2; exit 2; }
DEF=$(sed -n 's/^NEIF_DEFINES *= *//p' <<<"$out")
[ -n "$DEF" ] || { echo "make features printed no NEIF_DEFINES" >&2; exit 2; }
grep -q -- '-DNEIF_TF1=1' <<<"$DEF" || { echo "the line builds no transformer (NEIF_TF1 not 1): no manifest arms" >&2; exit 2; }
cat > "$T/probe.cpp" <<'CPP'
#include <cstdio>
#include "tf1/tf1_manifest.h"
#ifndef NEIF_TF1_PAKET_X0ROW
#define NEIF_TF1_PAKET_X0ROW 0
#endif
#ifndef NEIF_TF1_REDIRMERGE
#define NEIF_TF1_REDIRMERGE 0
#endif
int main() {
  std::printf("arme=%s\n", neif_tf1_manifest::Join(neif_tf1_manifest::Arme()).c_str());
  std::printf("v=%d\nnsent=%d\nx=%d\ndslot=%d\n", (int)NEIF_TF1_V, (int)NEIF_TF1_PAKET_NSENT,
              (int)NEIF_TF1_PAKET_X, (int)NEIF_TF1_PAKET_DSLOT);
  std::printf("x0row=%d\nredirmerge=%d\nfenster=%d\nnl=%d\nprior0_set=%d\n", (int)(NEIF_TF1_PAKET_X0ROW),
              (int)(NEIF_TF1_REDIRMERGE), (int)NEIF_TF1_ATTN_WIN, (int)NEIF_TF1_NL, (int)NEIF_TF1_PRIOR0_SET);
  return 0;
}
CPP
# shellcheck disable=SC2086
if ! (cd "$SRC" && $CXX -std=c++17 -march=x86-64-v3 -O0 -w $DEF -Isrc -Isrc/tf1 "$T/probe.cpp" -o "$T/probe") > "$T/cc.log" 2>&1; then
  # the probe uses inline code only; symbols the headers declare but the
  # probe never calls may stay unresolved
  # shellcheck disable=SC2086
  (cd "$SRC" && $CXX -std=c++17 -march=x86-64-v3 -O0 -w $DEF -Isrc -Isrc/tf1 "$T/probe.cpp" -o "$T/probe" \
     -Wl,--unresolved-symbols=ignore-all) > "$T/cc2.log" 2>&1 \
    || { echo "probe does not compile:" >&2; head -20 "$T/cc.log" >&2; exit 2; }
fi
"$T/probe" || { echo "probe failed" >&2; exit 2; }
