#!/bin/bash

#SEED="$1"
#UPDATE_LIMIT="$2"

SEED="923"
UPDATE_LIMIT="3000"

CFLAGS_DEFINES="-DSEED=$SEED -DUPDATE_LIMIT=$UPDATE_LIMIT"

# The weight container that gets embedded as the fourth block.
WEIGHTS_FILE="weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5"

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
# Function order (cmstub only): the context-mixing packer codes
# the image better when functions with similar code lie next to each other.
# The order is computed from this first link - its stripped image after the
# packer's address transform and its linker map (symorder.py, deterministic)
# - and a second build links the functions in that order.  Same code, other
# layout; 2.327 B in S on cmix-neif-1.0.
if [[ "$PAK_PACKER" == "cmstub" && "1" == "1" ]]; then
  PAK_STUB_DIR="./tools_src/packer"
  cp -f cmix symorder.img
  "$STRIP_BIN" --strip-all symorder.img
  objcopy --remove-section=.comment \
    --remove-section=.note.gnu.property \
    --remove-section=.note.gnu.build-id \
    --remove-section=.note.ABI-tag \
    symorder.img
  "$OBJCOPY_SS" --strip-sections symorder.img
  gcc -O2 -w -o symorder_tool "$PAK_STUB_DIR/pack_tool.c" || exit 1
  ./symorder_tool xform symorder.img symorder.x || exit 1
  python3 "$PAK_STUB_DIR/symorder.py" symorder.x cmix.map > symbol_order.txt || exit 1
  rm -f symorder.img symorder.x symorder_tool
  make CC="$CXX_BIN" CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_use -j \
    LINK_ORDER="-Wl,--symbol-ordering-file=$PWD/symbol_order.txt -Wl,--no-warn-symbol-ordering" || exit 1
fi
# Drop non-runtime ELF metadata before packing. The binary is already linked with
# -s; these sections are extra loader/comment notes and do not affect behavior.
# Keep the image WITH section headers for the section guard, which runs in
# the packer's build_packer.sh (after sstrip, objdump -d no longer works).
cp -f cmix cmix.cutguard 2>/dev/null || true
# The map of this link, kept next to the image: the helper build below links
# again and overwrites cmix.map, so the guard could not be re-run later.
cp -f cmix.map cmix.unpacked.map 2>/dev/null || true
"$STRIP_BIN" --strip-all cmix || true
# This must be GNU objcopy.  GNU objcopy and llvm-objcopy produce different
# program headers (9 bytes) and therefore different binaries; the GNU result is
# 64 B smaller once packed (128 B in S), and the release sha256 rests on it.
# .eh_frame/.eh_frame_hdr: GNU objcopy zeroes them in place (same length,
# 56 B of headers and content change).  The program has no exceptions and no
# unwind tables of its own; without a frame table an exception from libstdc++
# ends in terminate as it does today.
objcopy --remove-section=.comment \
  --remove-section=.note.gnu.property \
  --remove-section=.note.gnu.build-id \
  --remove-section=.note.ABI-tag \
  --remove-section=.eh_frame \
  --remove-section=.eh_frame_hdr \
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
# anonymous memory file and starts it with memfd_create + execveat.  The coder
# is the context-mixing packer of tools_src/packer; packing needs gcc for the
# host coder; unpacking takes about 50 s and about 1.5 GiB per start on a Zen 2 core, all of it
# released by the execveat.  Its packed stream is cut after the decompressor
# part (header: six u32 at file offset 120 of the stub), so the decompressor
# embedded in the archive carries no encoder-only code.
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
# enwik9 (src/runner.cpp, src/preprocess/dict_derive.cpp); the decompressor
# takes the dictionary from the first article of the stream.  The hint package
# belongs to the dictionary: a different dictionary/english.dic needs a new one.
B6_HINTS_FILE="${B6_HINTS_FILE:-$ROOT/dictionary/english.hints}"
if [[ ! -s "$B6_HINTS_FILE" ]]; then
  echo "hint package '$B6_HINTS_FILE' missing." >&2
  echo "see section 8 of training/README.md for how it is produced" >&2
  exit 1
fi
# The hint package has the form v2 (four parts, each a varint length and its
# bytes, nothing after them; src/preprocess/dict_derive.cpp).  A package of an
# older form would be read as a different dictionary, so it stops the build.
_hp=$(od -An -v -tu1 "$B6_HINTS_FILE" | awk '{ for (i = 1; i <= NF; i++) b[n++] = $i }
  END { p = 0; k = 0
        while (p < n) { v = 0; s = 1
          do { c = b[p++]; v += (c % 128) * s; s *= 128 } while (c >= 128 && p < n)
          p += v; k++ }
        print (p == n) ? k : -1 }')
if [[ "$_hp" != 4 ]]; then
  echo "hint package '$B6_HINTS_FILE' is not of form v2 (parts: $_hp, expected 4)" >&2
  exit 1
fi
SELFEXTRACT_SRC="$B6_HINTS_FILE"
SELFEXTRACT_BLOCK=comp_hints

# By default the embedded blocks are rebuilt only when they are missing.
# Coding them again takes time and gives the same blocks as long as the
# hint package, the article order and the build helper that codes them
# stay the same.
if [[ "0" == "1" || ! -s ./$SELFEXTRACT_BLOCK ]]; then
# The inputs of the two blocks are coded data and no enwik9 coder stream;
# about 18 % of their bytes lie outside the stamped 208-value byte set of the
# coder of enwik9.  Hence the full-alphabet helper; the coding of enwik9
# itself is not affected.
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

# creating a header with the sizes of the above files: three little-endian
# int32 (hint package, article order, weight container or 0)
python3 -c 'import struct, sys; open("header.dat", "wb").write(struct.pack("<3i", *map(int, sys.argv[1:])))' \
  "$(wc -c < ./$SELFEXTRACT_BLOCK)" "$(wc -c < ./comp_order)" "$WEIGHTS_SIZE"

# merging the above files and setting permissions for the final executable file
cat ./cmix_orig ./$SELFEXTRACT_BLOCK ./comp_order $WEIGHTS_PATH header.dat > ./cmix
chmod +x ./cmix
