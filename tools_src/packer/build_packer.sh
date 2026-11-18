#!/bin/bash
# Packs the stripped binary with the self-extracting stub of this directory.
# Called by build_and_construct_comp.sh; writes <name>.pak into the working
# directory.
#
# Layout: [stage 1][one context-mixing stream]; the header (six u32) sits at
# file offset 120 of stage 1, right behind the ELF and program header.
# The coder is the x86-aware context-mixing coder of cm_core.h (instruction-
# level contexts from a length decoder over the already decoded bytes) after
# an exact address transform (x86t.h: RIP-relative disp32 and rel32 of
# call/jmp/jcc -> absolute addresses).  The unpacker (stage 2, stub.c)
# is stored compressed with the byte-mask context mixer of boot_cm.h and
# unpacked by stage 1 (stub_boot.c), which also carries the header in plain
# form; stage 2 writes the image into an anonymous memory file and starts it.
# The linker script puts all encoder-only code into a section at the end of
# the file; up to Nd the image is a complete decompressor, so the first K
# bytes of the stream decode the decompressor alone, and selfextract_comp()
# truncates .decomp_bin at payload_off+K (runner.cpp decomp_prefix_len); the
# encoder part is carried only once, in the packed program.
#
# The stub passes the real package path through the memfd image
# ([path][u32 length][16-B mark "B5C2SELFPATH_V1"]).  Without it selfextract_*
# does not find the appended blocks.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
IN="${1:?Usage: build_pak_stub.sh <cmix-binary> [workdir]}"
WD="${2:-.}"
CC="${CC:-clang-18}"
OBJCOPY_SS="${OBJCOPY_SS:-}"
if [ -z "$OBJCOPY_SS" ]; then
  case "$CC" in
    *-1[0-9]) OBJCOPY_SS="llvm-objcopy-${CC##*-}" ;;
    *)        OBJCOPY_SS="llvm-objcopy" ;;
  esac
  command -v "$OBJCOPY_SS" >/dev/null 2>&1 || OBJCOPY_SS="llvm-objcopy"
fi
sstrip() { cp "$1" "$2" && "$OBJCOPY_SS" --strip-sections "$2"; }
command -v gcc >/dev/null || { echo "gcc is missing (CM host coder)" >&2; exit 1; }


mkdir -p "$WD"; cd "$WD"; WD="$PWD"
N="$(basename "$IN")"
IN_ABS="$(cd "$(dirname "$IN")" && pwd)/$(basename "$IN")"

sstrip "$IN_ABS" "$N.ss"; chmod +x "$N.ss"
NF=$(stat -c%s "$N.ss")

# --- Nd: file offset of the LAST file-backed PT_LOAD (= .encoder_only) ---------
read ND SZ < <(readelf -lW "$N.ss" | awk '$1=="LOAD" && strtonum($5)>0 {o=$2;s=$5} END{print strtonum(o), strtonum(s)}')
[ -n "${ND:-}" ] && [ "$ND" -gt 0 ] || { echo "CUT: no file-backed PT_LOAD found" >&2; exit 1; }
if [ "$((ND+SZ))" != "$NF" ]; then
  echo "CUT: last LOAD does not end at end of file ($((ND+SZ)) != $NF)" >&2; exit 1
fi
# Guard against the silent misbuild: without the linker script the last LOAD
# is the ordinary text/data segment and Nd would be far too small, so the
# truncated .decomp_bin would hold no complete decompressor.
# The encoder block is about 200 KB; below 32 KB something is off.
if [ "$SZ" -lt 32768 ]; then
  echo "CUT: block only $SZ B - linker script v7v7.lds has no effect" >&2; exit 1
fi
if [ "$ND" -lt $((NF/4)) ]; then
  echo "CUT: Nd=$ND is < Nf/4 ($NF) - the last LOAD is NOT .encoder_only" >&2; exit 1
fi

# --- STATIC BUILD GUARD ------------------------------------------------------
# The two guards above only catch a total failure of the linker script.  They
# do not catch one of the ~150 functions named in the script being on the
# decompression path; the decompressor would then jump into the zero-filled
# range, and only a full decompression would show it (SIGSEGV after many
# hours).  So the shipped build itself is examined: reachability from the
# argc==1 branch of main over the complete call graph (cut_dpath.py).  A
# decoder-only build cannot serve as reference, because the compiler inlines
# the static functions there and their names disappear.
#
# It needs two inputs from the caller:
#   CUT_GUARD_ELF - the linked image with section headers (before sstrip;
#                   default cmix.cutguard, build_and_construct_comp.sh copies it)
#   CUT_MAP       - the linker map (default cmix.map from -Wl,-Map=cmix.map;
#                   does not change the binary)
# If either is missing, the build stops.
GUARD_ELF="$(dirname "$IN_ABS")/cmix.cutguard"
GUARD_MAP="$(dirname "$IN_ABS")/cmix.map"
# The linker script sits in the package next to the built binary.
GUARD_LDS="$(dirname "$IN_ABS")/encoder_split.lds"
if [ "1" != "0" ]; then
  command -v python3 >/dev/null 2>&1 || {
    echo "CUT guard: python3 is missing - the dud check cannot run." >&2
    echo "  (CUT_WACHT=0 overrides it - ONLY for builds that are not" >&2
    echo "   shipped, and only with a note in the build log.)" >&2
    exit 1; }
  command -v objdump >/dev/null 2>&1 || { echo "CUT guard: objdump is missing" >&2; exit 1; }
  [ -s "$GUARD_ELF" ] || {
    echo "CUT guard: $GUARD_ELF is missing." >&2
    echo "  The caller must copy the linked image there BEFORE sstrip" >&2
    echo "  (build_and_construct_comp.sh: cp cmix cmix.cutguard after make prof_use)." >&2
    exit 1; }
  [ -s "$GUARD_MAP" ] || {
    echo "CUT guard: linker map $GUARD_MAP is missing." >&2
    echo "  In the makefile -Wl,-Map=cmix.map must be on the cmix target (byte-neutral)." >&2
    exit 1; }
  echo "CUT guard: checking $GUARD_ELF against ${GUARD_LDS:-<linker script unknown>}"
  if ! python3 "$SRC/cut_dpath.py" "$GUARD_ELF" "$GUARD_MAP" "$GUARD_LDS"; then
    echo "" >&2
    echo "CUT guard: BUILD ABORTED.  The functions named above are in the" >&2
    echo "  encoder block, but are needed by the DECOMPRESSOR.  If the" >&2
    echo "  package were shipped like this, compression would run through fine and" >&2
    echo "  only the verify would crash (after 34:25 h, SIGSEGV)." >&2
    exit 1
  fi
fi
# --- End build guard ---------------------------------------------------------

gcc -O2 -w -o cmcut_tool "$SRC/pack_tool.c"

# Exact x86 address transform over the executable PT_LOADs (the encoder
# block is a PT_LOAD of its own, so the prefix does not depend on it).
./cmcut_tool xform "$N.ss" "$N.x"

# Stage 2 (the unpacker, stub.c) as a flat image at S2BASE, compressed
# with the bootstrap coder; stage 1 (stub_boot.c: the bootstrap decoder in
# assembly, linker script s1.lds) carries it plus the header.  The x86-CM
# stream of the image is coded after the warm-up on the flat stage-2
# image (cm_core.h cm_warm), so each stub gets its own stream.
# Stage 2 is compiled with gcc -Os, stage 1 with $CC -Oz: gcc's code of
# stage 2 is smaller once coded (stage 2 travels compressed, stage 1 plain),
# 233 B in S.  So the gcc version is a build input as well (install.sh pins it).
# S2FLAGS: optimisations of gcc -Os switched off for stage 2, because its
# coded image gets smaller without them (greedy search over gcc options on
# the coded size).
# $1 = output prefix, $2 = extra flags for stage 2 (CUT_FILL / CUT_PROBE).
S2BASE=0x10000000
CFL="-nostdlib -static -Oz -ffreestanding -fno-asynchronous-unwind-tables \
     -fno-stack-protector -fcf-protection=none -fno-pic -fno-pie -no-pie \
     -ffunction-sections -fdata-sections -w -I$SRC"
S2FLAGS="-fno-schedule-insns2 -fno-caller-saves -fno-if-conversion -fno-tree-vrp -fno-tree-forwprop -fno-dse \
     -fno-forward-propagate -fira-algorithm=priority -fno-tree-loop-im -fno-ssa-phiopt -fno-move-loop-invariants -fno-expensive-optimizations"
CFL2="-nostdlib -static -Os -ffreestanding -fno-asynchronous-unwind-tables \
     -fno-stack-protector -fcf-protection=none -fno-pic -fno-pie -no-pie \
     -ffunction-sections -fdata-sections -w $S2FLAGS -I$SRC"
build_stub() {
  gcc $CFL2 -fno-inline-functions -fno-jump-tables $2 \
      -Wl,-T,"$SRC/s2.lds" -Wl,--build-id=none -Wl,--gc-sections \
      -e s2_main "$SRC/stub.c" -o "$1.s2.elf"
  "$OBJCOPY_SS" -O binary --only-section=.text --only-section=.rodata \
      --only-section=.data "$1.s2.elf" "$1.s2.bin"
  local len fend end tend mem
  len=$(stat -c%s "$1.s2.bin")
  fend=$(nm "$1.s2.elf" | awk '$3=="s2_filend"{print strtonum("0x"$1)}')
  end=$(nm "$1.s2.elf" | awk '$3=="s2_end"{print strtonum("0x"$1)}')
  tend=$(nm "$1.s2.elf" | awk '$3=="s2_textend"{print strtonum("0x"$1)}')
  [ "$len" -eq $((fend - S2BASE)) ] || { echo "stage 2: flat image $len != $((fend - S2BASE))" >&2; exit 1; }
  mem=$(( (end - S2BASE + 4095) / 4096 * 4096 ))
  S2TEXT=$((tend - S2BASE))
  ./cmcut_tool boot "$1.s2.bin" "$1.s2.cm"
  $CC $CFL -DS2BASE=$S2BASE -DS2MEM=$mem -DS2LEN=$len -DS2CM="\"$PWD/$1.s2.cm\"" \
      -Wl,-T,"$SRC/s1.lds" -Wl,--build-id=none -Wl,--gc-sections \
      -e _start "$SRC/stub_boot.c" -o "$1.elf"
  readelf -lW "$1.elf" | grep -q "There is 1 program header" || { echo "stage 1: not exactly one program header" >&2; exit 1; }
  [ "$(nm "$1.elf" | awk '$3=="cut_hdr"{print $1}')" = "0000000000400078" ] || { echo "stage 1: header not at file offset 120" >&2; exit 1; }
  strip -s "$1.elf" 2>/dev/null || true
  objcopy --remove-section=.comment --remove-section=.note.gnu.property \
          --remove-section=.note.gnu.build-id "$1.elf" 2>/dev/null || true
  sstrip "$1.elf" "$1.ss"
  ./cmcut_tool compress "$N.x" "$1.p" "$1.s2.bin" "$S2TEXT"
}

# Unset, the truncated region is zero-filled; CUT_FILL=0xCC (test builds only)
# fills it with int3.
CUT_FILL_FLAG=""
[ -n "" ] && CUT_FILL_FLAG="-DCUT_FILL=$CUT_FILL"
build_stub cutstub "$CUT_FILL_FLAG"
S2TEXT_SHIP=$S2TEXT
cp cutstub.p "$N.p"

./cmcut_tool pack cutstub.ss "$N.p" "$NF" "$ND" "$N.cut" cutstub.s2.bin "$S2TEXT_SHIP"

# --- Cross-checks ---------------------------------------------------------
# Each check ABORTS the build on a mismatch ("cmp && echo" would not: under
# set -e a failing command inside an && list does not stop the script).
fail() { echo "CUT-GEGENPROBE FEHLGESCHLAGEN: $1" >&2; exit 1; }
# (a) full package unpacks bit-identical to the input (reference unpacker)
./cmcut_tool unpack "$N.cut" "$N.chk" cutstub.s2.bin "$S2TEXT_SHIP" >/dev/null
cmp -s "$N.ss" "$N.chk" || fail "voll"
echo "CUT-Gegenprobe voll: bitidentisch"
# (b) the truncated .decomp_bin yields the prefix [0,Nd) bit-identical
./cmcut_tool cut "$N.cut" "$N.cut.dec"
./cmcut_tool unpack "$N.cut.dec" "$N.chkdec" cutstub.s2.bin "$S2TEXT_SHIP" >/dev/null
cmp -s <(head -c "$ND" "$N.chkdec") <(head -c "$ND" "$N.ss") || fail "dec"
echo "CUT-Gegenprobe dec: Praefix bitidentisch"
# (c) the REAL stub code path: a probe build of the same sources, which
# writes the image to stdout instead of executing it (full package and cut
# .decomp_bin).  The reference unpacker alone once passed while the shipped
# stub did not.  The probe's own code differs, so with the warm-up it gets
# its own stream (same image, warmed up on the probe stub).
build_stub cutprobe "-DCUT_PROBE $CUT_FILL_FLAG"
./cmcut_tool pack cutprobe.ss cutprobe.p "$NF" "$ND" "$N.probe" cutprobe.s2.bin "$S2TEXT" >/dev/null
./cmcut_tool cut "$N.probe" "$N.probe.dec" >/dev/null
chmod +x "$N.probe" "$N.probe.dec"
( ulimit -c 0; "./$N.probe" > "$N.probe.out" ) || fail "Stubprobe voll (rc)"
cmp -s "$N.ss" "$N.probe.out" || fail "Stubprobe voll"
echo "CUT-Stubprobe voll: bitidentisch"
( ulimit -c 0; "./$N.probe.dec" > "$N.probe.dec.out" ) || fail "Stubprobe dec (rc)"
[ "$(stat -c%s "$N.probe.dec.out")" = "$NF" ] || fail "stub probe dec (length)"
cmp -s <(head -c "$ND" "$N.probe.dec.out") <(head -c "$ND" "$N.ss") || fail "Stubprobe dec"
echo "CUT-Stubprobe dec: Praefix bitidentisch"
rm -f "$N.probe" "$N.probe.dec" "$N.probe.out" "$N.probe.dec.out"
# (d) the SHIPPED stub with the SHIPPED stream: package and cut .decomp_bin are
# started for real; an LD_PRELOAD helper copies the image the stub exec'd
# (/proc/self/exe) before anything of it runs, then exits (stub_capture.c).
gcc -O2 -w -shared -fPIC -o stub_capture.so "$SRC/stub_capture.c"
chmod +x "$N.cut" "$N.cut.dec"
rm -f "$N.cap" "$N.cap.dec"
( ulimit -c 0; LD_PRELOAD="$PWD/stub_capture.so" STUB_CAPTURE="$PWD/$N.cap" "./$N.cut" -h ) || fail "Echtstart voll (rc)"
cmp -s <(head -c "$NF" "$N.cap") "$N.ss" || fail "Echtstart voll"
echo "CUT-Echtstart voll: bitidentisch"
( ulimit -c 0; LD_PRELOAD="$PWD/stub_capture.so" STUB_CAPTURE="$PWD/$N.cap.dec" "./$N.cut.dec" -h ) || fail "Echtstart dec (rc)"
cmp -s <(head -c "$ND" "$N.cap.dec") <(head -c "$ND" "$N.ss") || fail "Echtstart dec"
cmp -s <(head -c "$NF" "$N.cap.dec" | tail -c +"$((ND + 1))") <(head -c "$((NF - ND))" /dev/zero) || fail "Echtstart dec (Nullen)"
echo "CUT-Echtstart dec: Praefix bitidentisch, Rest Nullen"
rm -f "$N.cap" "$N.cap.dec" stub_capture.so

cp "$N.cut" "$N.pak"

echo
printf '%-38s %9d\n' "input (raw)"                  "$(stat -c %s "$IN_ABS")"
printf '%-38s %9d\n' "after sstrip (Nf)"               "$NF"
printf '%-38s %9d\n' "  of which encoder block from Nd"    "$SZ"
printf '%-38s %9d\n' "Nd (cut point)"             "$ND"
printf '%-38s %9d\n' "CM raw stream"                    "$(stat -c %s "$N.p")"
printf '%-38s %9d\n' "stage 2 (flat / compressed)"    "$(stat -c %s cutstub.s2.bin)"
printf '%-38s %9d\n' "  compressed"                  "$(stat -c %s cutstub.s2.cm)"
printf '%-38s %9d\n' "Stub (sstripped)"               "$(stat -c %s cutstub.ss)"
printf '%-38s %9d\n' "Package = cmix_orig"              "$(stat -c %s "$N.cut")"
printf '%-38s %9d\n' ".decomp_bin (truncated)"    "$(stat -c %s "$N.cut.dec")"
printf '%-38s %9d\n' "savings per .decomp_bin"       "$(( $(stat -c %s "$N.cut") - $(stat -c %s "$N.cut.dec") ))"
