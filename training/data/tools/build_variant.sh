#!/usr/bin/env bash
# build_variant.sh - builds a compressor variant with the recipe of the fork
# base (clang-17 + PGO + LTO + strip + UPX 5.1.1 --ultra-brute) and a set of
# build features, and records the size breakdown in variant.json.
#
#   bash scripts/build_variant.sh <name> "<FEATURES>" [options]
#
# Examples
#   bash scripts/build_variant.sh base   "DET"
#   bash scripts/build_variant.sh no_a1  "DET A1=0"
#   bash scripts/build_variant.sh slim   "DET NOMANIFEST NOPARAMS"
#   bash scripts/build_variant.sh order2 "DET" --order <orders>/v2.bin
#   bash scripts/build_variant.sh rel1   "RELEASE" --release   <- submission build
#
# Submission build:
#   --release is the short form of the packaging stages (a) to (e):
#     (a) clang-18 + lld-18   (b)+(c) NOMANIFEST NOPARAMS B5C_SLIM (from the
#     FEATURES alias RELEASE)   (d) --as-needed --build-id=none (makefile)
#     (e) sstrip before packing
#   Together -4,648 B in S.  A binary built this way has no -n mode and writes
#   no <out>.features.txt, so it cannot be used for measurements.  Therefore
#   --release requires the alias RELEASE in FEATURES.
#
# Options
#   --order <file>    another article order (replaces
#                     src/readalike_prepr/data/new_article_order)
#   --dict  <file>    another dictionary (replaces dictionary/english.dic)
#   --src   <dir>     source directory (default: $MIRROR/src/cmix-lex)
#   --out   <dir>     target directory (default: $VARROOT/<name>)
#   --release         submission build: --llvm 18 --sstrip (see above);
#                     requires RELEASE in the FEATURES string
#   --llvm N          toolchain clang++-N / llvm-*-N (default 17; the makefile
#                     picks lld-N as the linker)
#   --sstrip          stage (e): drop the section header table before packing
#                     (-484 B packed = -968 B in S, same output)
#   --packer upx|stub packing stage (f): upx, or the self-extracting stub
#                     (the stub is the packer of the submitted build)
#   --reuse-pgo       use the cached PGO profile (faster, and variants stay
#                     exactly comparable; without it a new profile is made)
#   --selftest        run enwik8_1m -c after the build (~3 min, 8 GB)
#   --jobs N          make -j N (default: nproc, at most 16)
#   --force           delete an existing target directory
#   --dry-run         only show the resolved flags, do not build
#
# Result
#   <out>/src/run/cmix   packed compressor binary
#   <out>/cmix           copy of it (the compressor counted in S)
#   <out>/variant.json   features, flags, size breakdown, sha256, times
#   <out>/build.log      complete build log
#
# $MIRROR (the repository checkout) is only read; the build always runs in a
# working copy under <out>/src.

set -uo pipefail

MIRROR="${MIRROR:?set MIRROR to the repository checkout}"
VARROOT="${VARROOT:?set VARROOT to the build directory}"
DATA="${DATA:?set DATA to the data directory}"
CACHE="${CACHE:-$VARROOT/_cache}"

REF_CMIX_PACKED=459938          # record compressor
REF_ARCHIVE9=109190109
REF_S=109650047
TARGET_S=108553546

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }
die() { printf '[%s] ERROR: %s\n' "$(date +'%F %T')" "$*" >&2; exit 1; }
fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

usage() { sed -n '2,40p' "$0"; exit 1; }

[ $# -ge 2 ] || usage
NAME="$1"; shift
FEATURES="$1"; shift
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid variant name '$NAME'"

SRC_IN="$MIRROR/src/cmix-lex"
OUT=""
ORDER_FILE=""
DICT_FILE=""
REUSE_PGO=0
SELFTEST=0
JOBS=""
FORCE=0
DRYRUN=0
# packaging stages; the defaults keep the recipe of the fork base
LLVM_VER=17
B5C_SSTRIP=0
B5C_PACKER=upx
RELEASE_MODE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --order)     ORDER_FILE="${2:-}"; shift 2 ;;
    --dict)      DICT_FILE="${2:-}"; shift 2 ;;
    --src)       SRC_IN="${2:-}"; shift 2 ;;
    --out)       OUT="${2:-}"; shift 2 ;;
    --jobs)      JOBS="${2:-}"; shift 2 ;;
    --llvm)      LLVM_VER="${2:-}"; shift 2 ;;
    --sstrip)    B5C_SSTRIP=1; shift ;;
    --packer)    B5C_PACKER="${2:-}"; shift 2 ;;
    --release)   RELEASE_MODE=1; LLVM_VER=18; B5C_SSTRIP=1; shift ;;
    --reuse-pgo) REUSE_PGO=1; shift ;;
    --selftest)  SELFTEST=1; shift ;;
    --force)     FORCE=1; shift ;;
    --dry-run)   DRYRUN=1; shift ;;
    -h|--help)   usage ;;
    *) die "unknown option '$1'" ;;
  esac
done

# Guard: a binary without -n and without a manifest is useless for
# measurements, and that would only show much later.
case " $(printf '%s' "$FEATURES" | tr 'a-z' 'A-Z') " in
  *" RELEASE "*) FEATURES_HAS_RELEASE=1 ;;
  *)             FEATURES_HAS_RELEASE=0 ;;
esac
if [ "$RELEASE_MODE" = 1 ] && [ "$FEATURES_HAS_RELEASE" = 0 ]; then
  die "--release without 'RELEASE' in the FEATURES string. Use:
     bash scripts/build_variant.sh $NAME \"RELEASE\" --release
   (RELEASE == V2 + NOMANIFEST + NOPARAMS + B5C_SLIM + release linker flags;
    the binary then has no -n mode and writes no <out>.features.txt)"
fi
if [ "$FEATURES_HAS_RELEASE" = 1 ] && [ "$RELEASE_MODE" = 0 ]; then
  log "WARNING: FEATURES contains RELEASE, but --release is missing -> this leaves out"
  log "         stage (a) clang-18 (-1,192 B in S) and stage (e) sstrip (-968 B in S)."
fi
case "$B5C_PACKER" in upx|stub) ;; *) die "--packer must be upx or stub" ;; esac
[ "$B5C_PACKER" = stub ] && log "note: --packer stub packs with the self-extracting stub, as the submitted build does"

[ -n "$OUT" ] || OUT="$VARROOT/$NAME"
[ -n "$JOBS" ] || JOBS=$(nproc 2>/dev/null || echo 4)
[ "$JOBS" -gt 16 ] 2>/dev/null && JOBS=16
[ -d "$SRC_IN" ] || die "source directory missing: $SRC_IN"
[ -z "$ORDER_FILE" ] || [ -f "$ORDER_FILE" ] || die "order file missing: $ORDER_FILE"
[ -z "$DICT_FILE" ]  || [ -f "$DICT_FILE" ]  || die "dictionary missing: $DICT_FILE"

SRC="$OUT/src"

# ------------------------------------------------------------ resolve flags
resolve() {
  make -s -C "$1" features FEATURES="$FEATURES" CC="clang++-$LLVM_VER" 2>/dev/null | \
    sed -n "s/^$2 *= *//p" | head -n1
}

if [ "$DRYRUN" = 1 ]; then
  echo "variant    : $NAME"
  echo "FEATURES   : $FEATURES"
  make -s -C "$SRC_IN" features FEATURES="$FEATURES" CC="clang++-$LLVM_VER" 2>&1 | grep -v '^ARCH_FLAGS = \|^FP_MODEL   = \|^NEIF_LIST  = '
  echo "target     : $OUT"
  [ -n "$ORDER_FILE" ] && echo "Order      : $ORDER_FILE"
  [ -n "$DICT_FILE" ]  && echo "Dictionary : $DICT_FILE"
  exit 0
fi

# ------------------------------------------------------------------- Stage
if [ -d "$OUT" ]; then
  [ "$FORCE" = 1 ] || die "target directory already exists: $OUT (--force to overwrite)"
  log "deleting $OUT"
  rm -rf "$OUT"
fi
mkdir -p "$SRC" "$CACHE"

log "variant '$NAME'  FEATURES=\"$FEATURES\""
log "sources: $SRC_IN -> $SRC"
rsync -a --delete --exclude 'tools/' --exclude 'pgo_data/' --exclude 'run/' \
      --exclude '*.o' --exclude 'cmix' --exclude 'cmix_det' --exclude 'cmix_costmap' \
      "$SRC_IN/" "$SRC/"

# UPX 5.1.1 from the cache, or installed once
mkdir -p "$SRC/tools"
if [ -x "$CACHE/tools/upx" ]; then
  cp -a "$CACHE/tools/upx" "$SRC/tools/upx"
else
  log "installing UPX 5.1.1 (once)"
  ( cd "$SRC" && bash install_tools/install_upx.sh ) >> "$OUT/build.log" 2>&1 \
    || die "install_upx.sh failed, see $OUT/build.log"
  mkdir -p "$CACHE/tools" && cp -a "$SRC/tools/upx" "$CACHE/tools/upx"
fi
UPX_VER="$("$SRC/tools/upx" --version 2>/dev/null | head -n1)"
[ "$UPX_VER" = "upx 5.1.1" ] || die "wrong UPX version: '$UPX_VER' (expected 'upx 5.1.1')"

# ld.lld-N is checked as well: the recipe uses -flto, so the linker does the
# code generation and must match the compiler version (clang-18 with lld-17
# does not link at all).  The makefile derives -fuse-ld=lld-N from CC.
for t in "clang++-$LLVM_VER" "llvm-profdata-$LLVM_VER" "llvm-strip-$LLVM_VER" "ld.lld-$LLVM_VER"; do
  command -v "$t" >/dev/null || die "$t missing (apt-get install clang-$LLVM_VER lld-$LLVM_VER llvm-$LLVM_VER)"
done
if [ "$B5C_SSTRIP" = 1 ]; then
  command -v "llvm-objcopy-$LLVM_VER" >/dev/null || die "llvm-objcopy-$LLVM_VER missing (needed for --sstrip)"
fi
command -v objcopy >/dev/null || die "objcopy (binutils) missing; the recipe depends on it"
CLANG_VER="$("clang++-$LLVM_VER" --version | head -n1)"
LLD_VER="$("ld.lld-$LLVM_VER" --version 2>/dev/null | head -n1)"
OBJCOPY_VER="$(objcopy --version 2>/dev/null | head -n1)"
log "Toolchain : $CLANG_VER | $LLD_VER | $OBJCOPY_VER"
log "packaging : sstrip=$B5C_SSTRIP packer=$B5C_PACKER"

# install the order or dictionary variant
ORDER_DEFAULT="$SRC/src/readalike_prepr/data/new_article_order"
DICT_DEFAULT="$SRC/dictionary/english.dic"
if [ -n "$ORDER_FILE" ]; then
  log "replacing order file: $ORDER_FILE ($(fsize "$ORDER_FILE") B)"
  cp -f "$ORDER_FILE" "$ORDER_DEFAULT" || die "order file cannot be copied"
fi
if [ -n "$DICT_FILE" ]; then
  log "replacing dictionary: $DICT_FILE ($(fsize "$DICT_FILE") B)"
  cp -f "$DICT_FILE" "$DICT_DEFAULT" || die "dictionary cannot be copied"
fi

# PGO profile from the cache (keeps variants exactly comparable, saves ~2 min)
PGO_MODE="frisch"
if [ "$REUSE_PGO" = 1 ] && [ -s "$CACHE/pgo_data/default.profdata" ]; then
  mkdir -p "$SRC/pgo_data"
  cp -f "$CACHE/pgo_data/default.profdata" "$SRC/pgo_data/default.profdata"
  PGO_MODE="wiederverwendet"
fi
# Four inputs determine the product and come from the environment, so they are
# recorded in the log and in variant.json; with them a control build can be
# repeated byte for byte.
# Their weight, each measured alone on the same tree:
#   CACHE          _cache instead of an older cache       +1,153 B in the compressor
#   SIZEOPT_TUNE   empty instead of -mtune=k8             +2,549 B
#   B5C_STUB_DIR   b5c_stub instead of b5c2_stub          +2,930 B in cmix_orig
#   B6_HINTS_FILE  8-block instead of 6-block hint package  5,861 B less in comp_hints
PGO_PROFDATA=""; PGO_MD5=""
if [ "$PGO_MODE" = "wiederverwendet" ]; then
  PGO_PROFDATA="$CACHE/pgo_data/default.profdata"
  PGO_MD5="$(md5sum "$PGO_PROFDATA" 2>/dev/null | cut -d' ' -f1)"
fi
B5C_STUB_DIR_EFF="${B5C_STUB_DIR:-./tools_src/b5c_stub}"
B6_HINTS_FILE_EFF="${B6_HINTS_FILE:-$SRC/dictionary/english.hints}"
B6_HINTS_MD5="$(md5sum "$B6_HINTS_FILE_EFF" 2>/dev/null | cut -d' ' -f1)"
log "PGO profile: $PGO_MODE${PGO_MD5:+  Cache=$CACHE  profdata-md5=$PGO_MD5}"
log "SIZEOPT_TUNE : ${SIZEOPT_TUNE:-<empty>}"
log "B5C_STUB_DIR : $B5C_STUB_DIR_EFF$([ -n "${B5C_STUB_DIR:-}" ] || echo '   (Default!)')"
log "B6_HINTS_FILE: $B6_HINTS_FILE_EFF  md5=${B6_HINTS_MD5:-<missing>}$([ -n "${B6_HINTS_FILE:-}" ] || echo '   (Default!)')"

NEIF_LIST="$(resolve "$SRC" NEIF_LIST)"
ARCH_FLAGS_R="$(resolve "$SRC" ARCH_FLAGS)"
FP_MODEL_R="$(resolve "$SRC" FP_MODEL)"
log "resolved  : $NEIF_LIST"
log "ARCH_FLAGS: $ARCH_FLAGS_R"
log "FP_MODEL  : $FP_MODEL_R"

# -------------------------------------------------------------------- Build
T0=$(date +%s)
log "build running (PGO + LTO + UPX). Log: $OUT/build.log"
(
  cd "$SRC" || exit 1
  export FEATURES
  export REUSE_PGO="$REUSE_PGO"
  export FORCE_SELFEXTRACT_REBUILD=1
  export MAKEFLAGS="-j$JOBS"
  # pass the packaging stages on to build_and_construct_comp.sh
  export LLVM_VER="$LLVM_VER"
  export B5C_SSTRIP="$B5C_SSTRIP"
  export B5C_PACKER="$B5C_PACKER"
  bash -x ./build_and_construct_comp.sh
) >> "$OUT/build.log" 2>&1 || die "build failed, see $OUT/build.log"
T1=$(date +%s)

RUN="$SRC/run"
[ -x "$RUN/cmix" ] || die "packaged binary $RUN/cmix missing"
cp -f "$RUN/cmix" "$OUT/cmix"; chmod +x "$OUT/cmix"

# put the PGO profile into the cache (for --reuse-pgo of later variants)
if [ -s "$SRC/pgo_data/default.profdata" ] && [ ! -s "$CACHE/pgo_data/default.profdata" ]; then
  mkdir -p "$CACHE/pgo_data"
  cp -f "$SRC/pgo_data/default.profdata" "$CACHE/pgo_data/default.profdata"
fi

# -------------------------------------------------------------------- sizes
ORIG=$(fsize "$RUN/cmix_orig")
CDICT=$(fsize "$RUN/comp_dict")
CORDER=$(fsize "$RUN/comp_order")
HDR=$(fsize "$RUN/header.dat")
PACKED=$(fsize "$RUN/cmix")
SUM=$((ORIG + CDICT + CORDER + HDR))
# size before UPX, taken from the UPX output in the build log
PREUPX=$(grep -aE '^ *[0-9]+ *-> *[0-9]+ ' "$OUT/build.log" | tail -n1 | awk '{print $1}')
[ -n "${PREUPX:-}" ] || PREUPX=0
FIXED_S=$((2 * (ORIG + CDICT) + CORDER + 24))

printf '\n=== size breakdown of variant %s ===\n' "$NAME"
printf '%-36s %12s %12s\n' "part" "bytes" "record"
printf '%-36s %12d %12s\n' "cmix before UPX"                  "$PREUPX" "-"
printf '%-36s %12d %12s\n' "cmix_orig (UPX, = .decomp_bin)"   "$ORIG" "-"
printf '%-36s %12d %12s\n' "comp_dict (= .dict.comp)"         "$CDICT" "-"
printf '%-36s %12d %12d\n' "comp_order"                       "$CORDER" 199754
printf '%-36s %12d %12d\n' "header.dat"                       "$HDR" 12
printf '%-36s %12d %12d\n' "SUM = run/cmix"                   "$SUM" "$REF_CMIX_PACKED"
printf '%-36s %12d\n'      "checked (stat run/cmix)"          "$PACKED"
printf 'delta compressor against the record: %+d B\n' "$((PACKED - REF_CMIX_PACKED))"
printf 'fixed S share 2*(orig+dict)+order+24: %d B  (record 720122)\n\n' "$FIXED_S"

# ------------------------------------------------------------------ Selftest
ST_OUT=0; ST_USER=""; ST_INPUT=""
if [ "$SELFTEST" = 1 ]; then
  if [ -f "$DATA/enwik8_1m" ]; then
    ST_INPUT="$DATA/enwik8_1m"
    log "self-test: cmix_orig -c dictionary/english.dic enwik8_1m out"
    T="$OUT/selftest"; rm -rf "$T"; mkdir -p "$T"
    ( cd "$T" && /usr/bin/time -f '%U' "$RUN/cmix_orig" -c "$DICT_DEFAULT" "$ST_INPUT" out ) \
      > "$T/log" 2>&1
    ST_OUT=$(fsize "$T/out")
    ST_USER=$(tail -n1 "$T/log")
    log "self-test: $ST_OUT B (user ${ST_USER}s)"
  else
    log "self-test skipped ($DATA/enwik8_1m missing)"
  fi
fi

# --------------------------------------------------------------- variant.json
V_NAME="$NAME" V_FEATURES="$FEATURES" V_LIST="$NEIF_LIST" \
V_ARCH="$ARCH_FLAGS_R" V_FP="$FP_MODEL_R" V_SRC="$SRC_IN" V_OUT="$OUT" \
V_ORDER="${ORDER_FILE:-}" V_DICT="${DICT_FILE:-}" \
V_ORDER_SHA="$(sha "$ORDER_DEFAULT")" V_ORDER_BYTES="$(fsize "$ORDER_DEFAULT")" \
V_DICT_SHA="$(sha "$DICT_DEFAULT")" V_DICT_BYTES="$(fsize "$DICT_DEFAULT")" \
V_CLANG="$CLANG_VER" V_UPX="$UPX_VER" V_PGO="$PGO_MODE" \
V_PGO_CACHE="${PGO_PROFDATA:-}" V_PGO_MD5="${PGO_MD5:-}" \
V_SIZEOPT_TUNE="${SIZEOPT_TUNE:-}" \
V_STUB_DIR="${B5C_STUB_DIR_EFF:-}" V_HINTS_FILE="${B6_HINTS_FILE_EFF:-}" \
V_HINTS_MD5="${B6_HINTS_MD5:-}" \
V_LLD="$LLD_VER" V_OBJCOPY="$OBJCOPY_VER" V_LLVM_VER="$LLVM_VER" \
V_SSTRIP="$B5C_SSTRIP" V_PACKER="$B5C_PACKER" V_RELEASE="$RELEASE_MODE" \
V_PREUPX="$PREUPX" V_ORIG="$ORIG" V_CDICT="$CDICT" V_CORDER="$CORDER" \
V_HDR="$HDR" V_SUM="$SUM" V_PACKED="$PACKED" V_FIXED_S="$FIXED_S" \
V_SHA="$(sha "$RUN/cmix")" V_SECONDS="$((T1 - T0))" V_HOST="$(hostname)" \
V_ST_OUT="$ST_OUT" V_ST_USER="${ST_USER:-}" V_ST_INPUT="${ST_INPUT:-}" \
V_REF="$REF_CMIX_PACKED" \
python3 - "$OUT/variant.json" <<'PY'
import json, os, sys, datetime

def i(k, d=0):
    try: return int(os.environ.get(k, "") or d)
    except ValueError: return d
def f(k):
    try: return float(os.environ.get(k, "") or 0)
    except ValueError: return None
def s(k):
    v = os.environ.get(k, "")
    return v if v else None

packed, ref = i("V_PACKED"), i("V_REF")
d = {
  "name": s("V_NAME"),
  "kind": "variant",
  "features": s("V_FEATURES") or "",
  "features_resolved": s("V_LIST"),
  "arch_flags": s("V_ARCH"),
  "fp_model": s("V_FP"),
  "pgo": s("V_PGO"),
  # without these six a control build cannot be repeated
  "pgo_profdata": s("V_PGO_CACHE"),
  "pgo_profdata_md5": s("V_PGO_MD5"),
  "sizeopt_tune": s("V_SIZEOPT_TUNE"),
  "b5c_stub_dir": s("V_STUB_DIR"),
  "b6_hints_file": s("V_HINTS_FILE"),
  "b6_hints_md5": s("V_HINTS_MD5"),
  "clang": s("V_CLANG"),
  "lld": s("V_LLD"),
  "objcopy": s("V_OBJCOPY"),
  "upx": s("V_UPX"),
  # packaging stages
  "packaging": {
    "llvm_version": i("V_LLVM_VER"),
    "sstrip": i("V_SSTRIP") == 1,
    "packer": s("V_PACKER"),
    "release_mode": i("V_RELEASE") == 1,
  },
  "src": s("V_SRC"),
  "dir": s("V_OUT"),
  "binary": (s("V_OUT") or "") + "/cmix",
  "binary_sha256": s("V_SHA"),
  "order_file": s("V_ORDER"),
  "order_sha256": s("V_ORDER_SHA"),
  "order_bytes": i("V_ORDER_BYTES"),
  "dict_file": s("V_DICT"),
  "dict_sha256": s("V_DICT_SHA"),
  "dict_bytes": i("V_DICT_BYTES"),
  "sizes": {
    "cmix_pre_upx": i("V_PREUPX"),
    "cmix_orig": i("V_ORIG"),
    "comp_dict": i("V_CDICT"),
    "comp_order": i("V_CORDER"),
    "header_dat": i("V_HDR"),
    "sum": i("V_SUM"),
    "packed": packed,
  },
  "compressor_bytes": packed,
  "delta_compressor_record": packed - ref,
  "fixed_s_share": i("V_FIXED_S"),
  "record": {"compressor": ref, "archive9": 109190109, "S": 109650047,
             "fixed_s_share": 720122},
  "build_seconds": i("V_SECONDS"),
  "host": s("V_HOST"),
  "built_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
}
st_in = s("V_ST_INPUT")
if st_in:
    d["selftest"] = {"input": st_in, "out_bytes": i("V_ST_OUT"),
                     "user_s": f("V_ST_USER")}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(d, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("variant.json written: " + sys.argv[1])
PY

log "done in $((T1 - T0)) s -> $OUT/cmix ($PACKED B)"
log "next step: bash scripts/full_run.sh $NAME --variant $NAME"
