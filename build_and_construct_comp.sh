#!/bin/bash

#SEED="$1"
#UPDATE_LIMIT="$2"

SEED="923"
UPDATE_LIMIT="3000"

CFLAGS_DEFINES="-DSEED=$SEED -DUPDATE_LIMIT=$UPDATE_LIMIT"

# The weight container that gets embedded as the fourth block.
WEIGHTS_FILE="weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4"

# ---------------------------------------------------------------------------
# Packaging.  There are no options: the toolchain, the section stripping and
# the packer are fixed here, because the submitted binary is defined by exactly
# this combination.  clang-18 + lld-18 is not a preference -- built with 17 the
# packed binary is 596 B larger, and it counts twice in the score.
# ---------------------------------------------------------------------------
LLVM_VER="18"
CXX_BIN="clang++-$LLVM_VER"
PROFDATA_BIN="llvm-profdata-$LLVM_VER"
STRIP_BIN="llvm-strip-$LLVM_VER"
OBJCOPY_SS="llvm-objcopy-$LLVM_VER"
PAK_SSTRIP="1"
# The packer is the context-mixing stub of tools_src/packer.
PAK_PACKER="cmstub"

# building with PGO
if [[ "1" == "1" && -s ./pgo_data/default.profdata ]]; then
  echo "Reusing existing PGO profile ./pgo_data/default.profdata"
else
  rm -rf pgo_data
  mkdir -p pgo_data
  make CC="$CXX_BIN" CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_gen -j || exit 1

  ./cmix -c ./prof_input/input ./prof_comp > ./prof_output
  # The real -e run enters compression through the dictionary/preprocessor path;
  # include a small sample of that route so PGO does not overfit to nodict only.
  # Only the run with the dictionary may get a corpus-representative stream
  # (K1_TRAIN_FILE), because that is the route the -e path takes; the run
  # without dictionary above covers a path -e never takes.
  K1_TRAIN="./prof_input/input"
  ./cmix -c ./dictionary/english.dic "$K1_TRAIN" ./prof_comp_dict > ./prof_output_dict
  rm ./prof_comp ./prof_output ./prof_comp_dict ./prof_output_dict
  "$PROFDATA_BIN" merge -output=default.profdata ./pgo_data/*
  mv default.profdata pgo_data/
fi

# `|| exit 1`: without it the script would continue after a failed make (for
# example a stop by the section guard) and pack the binary of a previous run.
make CC="$CXX_BIN" CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_use -j || exit 1
# Drop non-runtime ELF metadata before packing. The binary is already linked with
# -s; these sections are extra loader/comment notes and do not affect behavior.
# Keep the image WITH section headers for the section guard, which runs in
# the packer's build_packer.sh (after sstrip, objdump -d no longer works).
cp -f cmix cmix.cutguard 2>/dev/null || true
"$STRIP_BIN" --strip-all cmix || true
# This must be GNU objcopy.  GNU objcopy and llvm-objcopy produce different
# program headers (9 bytes) and therefore different binaries; the GNU result is
# 64 B smaller once packed (128 B in S), and the release sha256 rests on it.
objcopy --remove-section=.comment \
  --remove-section=.note.gnu.property \
  --remove-section=.note.gnu.build-id \
  --remove-section=.note.ABI-tag \
  cmix 2>/dev/null || true

# sstrip: drop the section header table and .shstrtab.  llvm-objcopy
# --strip-sections first (same result), tools_src/sstrip.c as the fallback.
if [[ "$PAK_SSTRIP" == "1" ]]; then
  if "$OBJCOPY_SS" --strip-sections cmix 2>/dev/null; then
    echo "sstrip: $OBJCOPY_SS --strip-sections"
  else
    make CC="$CXX_BIN" sstrip || exit 1
    ./sstrip cmix cmix.sstripped || exit 1
    mv cmix.sstripped cmix
    chmod +x cmix
  fi
fi
# Own self-extractor instead of UPX.  The stub unpacks the image into an
# anonymous memory file and starts it with memfd_create + execveat
# (fallback: a file in ./ or /tmp and execve).  The packed stream is cut
# after the decompressor part (header B5JPAYLOAD_V1), so the decompressor
# embedded in the archive carries no encoder-only code.  The coder is the
# context-mixing packer of tools_src/packer; packing needs gcc for the host
# coder; unpacking takes about 10 s and about 180 MB per start, freed before
# the execveat.
PAK_STUB_DIR="./tools_src/packer"
command -v gcc >/dev/null || { echo "PAK_PACKER=cmstub: gcc is missing (host coder)." >&2; exit 1; }
if [[ ! -x "$PAK_STUB_DIR/build_packer.sh" ]]; then
  echo "PAK_PACKER=$PAK_PACKER: $PAK_STUB_DIR/build_packer.sh is missing." >&2
  exit 1
fi
CC="clang-$LLVM_VER" bash "$PAK_STUB_DIR/build_packer.sh" "$PWD/cmix" ./pak_pack || exit 1
mv ./pak_pack/cmix.pak cmix || exit 1
chmod +x cmix
# Keep the unpacked image with section headers for checks of the instruction
# set and the imported symbols; it is not part of the result.
mv -f cmix.cutguard cmix.unpacked 2>/dev/null || true

# ---------------------------------------------------------------------------
#  Build helper.
#
# comp_hints and comp_order are coded by a second binary, `cmix_aux`, built
# from the same sources with -DAUX_BUILD=1 (src/runner.cpp).  It codes the
# two blocks in full-alphabet mode and marks that with bit 0x40 in the header
# of the coded block, which the shipped binary reads.  The helper is used only
# during the build and is never shipped.
#
# It is built lazily, only when a block has to be rebuilt.  The function sets
# the global variable AUX_BIN (no command substitution: that would run in
# a subshell and build the helper a second time).
AUX_BIN=""
build_aux_helper() {
  if [[ -n "$AUX_BIN" ]]; then return 0; fi
  echo "building build helper cmix_aux (-DAUX_BUILD=1)" >&2
  # Put the finished main binary aside: `make prof_use` starts with `clean`.
    ( cd "$ROOT" && make CC="$CXX_BIN" CFLAGS_DEFINES="$CFLAGS_DEFINES" cmix_aux ) >&2
  local rc=$?
  if [[ $rc -ne 0 || ! -x "$ROOT/cmix_aux" ]]; then
    echo "building the helper cmix_aux failed (rc=$rc)" >&2
    return 1
  fi
  AUX_BIN="$ROOT/cmix_aux"
  return 0
}

# this is a directory where the compressor binary will be placed
DIR=run
mkdir -p ./$DIR
ROOT=$(pwd)
cp ./cmix $DIR/cmix_orig
# git diff > $DIR/patch
# exit
# building a selfextracting binary
pushd $DIR
# The binary carries, in the dictionary slot, not the compressed dictionary
# but the hint package (comp_hints); the -e path derives .dict from it plus
# enwik9 and puts .dict.comp into the archive (src/runner.cpp,
# src/preprocess/dict_derive.cpp).  The hint package belongs to the
# dictionary: a different dictionary/english.dic needs a new one.
B6_HINTS_FILE="${B6_HINTS_FILE:-$ROOT/dictionary/english.hints}"
if [[ ! -s "$B6_HINTS_FILE" ]]; then
  echo "hint package '$B6_HINTS_FILE' missing." >&2
  echo "see training/data/README.md for how it is produced" >&2
  exit 1
fi
SELFEXTRACT_SRC="$B6_HINTS_FILE"
SELFEXTRACT_BLOCK=comp_hints

# Reuse the embedded dictionary/order streams by default. PPM mmap/RSS-only
# changes do not alter compression behavior, and recompressing these payloads
# costs time without changing the final self-extract contents. Set
# FORCE_SELFEXTRACT_REBUILD=1 when the dictionary, order, or coder changes.
if [[ "0" == "1" || ! -s ./$SELFEXTRACT_BLOCK ]]; then
# The two blocks are not an enwik9 coder stream: 20.5 % (comp_hints) and 7.8 %
# (comp_order) of their bytes lie outside the stamped 208-value byte set, which
# the 208-value mode would code at 12 to 14 bits each (+116 KB of S).  Hence the
# full-alphabet helper; the coding of enwik9 and .dict.comp is not affected.
  build_aux_helper || { echo "error: the build helper does not compile - stopping" >&2; exit 1; }
  "$AUX_BIN" -c "$SELFEXTRACT_SRC" ./$SELFEXTRACT_BLOCK \
    || { echo "ERROR: cmix_aux -c (comp_hints) rc=$? - abort" >&2; exit 1; }
else
  echo "Reusing cached ./$SELFEXTRACT_BLOCK ($(wc -c < ./$SELFEXTRACT_BLOCK) bytes)"
fi
if [[ "0" == "1" || ! -s ./comp_order ]]; then
  build_aux_helper || { echo "error: the build helper does not compile - stopping" >&2; exit 1; }
  "$AUX_BIN" -c $ROOT/src/readalike_prepr/data/new_article_order ./comp_order \
    || { echo "ERROR: cmix_aux -c (comp_order) rc=$? - abort" >&2; exit 1; }
else
  echo "Reusing cached ./comp_order ($(wc -c < ./comp_order) bytes)"
fi
# If the helper crashed, empty blocks would be embedded and the build would
# still look fine.  The `||` checks above catch a failing helper, the size check
# here catches "rc 0 but empty".
for _f in "./$SELFEXTRACT_BLOCK" ./comp_order; do
  _sz=$(wc -c < "$_f" 2>/dev/null || echo 0)
  if [ "${_sz:-0}" -lt 1024 ]; then
    echo "error: $_f is only ${_sz} B - empty or truncated, build aborted" >&2
    exit 1
  fi
done

# The weight container is the fourth block.  It is appended after packing
# (cmix_orig is the packed image), because the container is entropy-coded
# already and a second coding pass would only make it larger.  WEIGHTS_FILE
# names the stamped container; unset, no fourth block is appended.
WEIGHTS_PATH=""
WEIGHTS_SIZE=0
if [[ -n "${WEIGHTS_FILE:-}" ]]; then
  if [[ ! -s "$ROOT/$WEIGHTS_FILE" ]]; then
    echo "weight container '$WEIGHTS_FILE' is missing in $ROOT" >&2
    exit 1
  fi
  cp -f "$ROOT/$WEIGHTS_FILE" "./$WEIGHTS_FILE"
  WEIGHTS_PATH="./$WEIGHTS_FILE"
  WEIGHTS_SIZE=$(wc -c < "$WEIGHTS_PATH")
fi

# creating a header with size of the above files
if [[ -n "$WEIGHTS_PATH" ]]; then
  ./cmix_orig -h $(wc -c < ./$SELFEXTRACT_BLOCK) $(wc -c < ./comp_order) 0 "$WEIGHTS_SIZE"
else
  ./cmix_orig -h $(wc -c < ./$SELFEXTRACT_BLOCK) $(wc -c < ./comp_order) 0
fi

# merging the above files and setting permissions for the final executable file
cat ./cmix_orig ./$SELFEXTRACT_BLOCK ./comp_order $WEIGHTS_PATH header.dat > ./cmix
chmod +x ./cmix
