#!/bin/bash
# build.sh - builds the compressor and writes it into the current working
# directory under the name it is submitted as.
#
# What this script does NOT do -- every line is checkable:
#   * no apt, no network, no downloads: everything comes from install.sh
#   * no root, no sudo
#   * it reads nothing from the environment and exports nothing: the build
#     itself is started with "env -i", so the compiler cannot see an
#     inherited variable either
#   * the build runs in a copy (.build/ in the working directory); the only
#     file it leaves behind is the compressor, in the working directory
#   * it may be started from any directory, the unpacked source directory
#     included
set -eu

SOURCE="$(cd "$(dirname "$0")" && pwd)"      # this package, read-only is fine
TARGET="$(pwd -P)"                           # working directory, writable
WORK="$TARGET/.build"

[ -w "$TARGET" ] || { echo "build.sh: $TARGET is not writable." >&2; exit 1; }

# Check the tools before anything else.  Without this check, "set -eu"
# would abort in the middle of the build and would not say which tool is
# missing.
missing_tools=""
for w in make clang++-18 clang-18 ld.lld-18 llvm-profdata-18 llvm-strip-18 \
         llvm-objcopy-18 objcopy gawk python3 gcc tar; do
  command -v "$w" >/dev/null 2>&1 || missing_tools="$missing_tools $w"
done
RTLIB="$(clang-18 --print-file-name=libclang_rt.profile-x86_64.a 2>/dev/null || true)"
[ -n "$RTLIB" ] && [ -f "$RTLIB" ] || missing_tools="$missing_tools libclang_rt.profile-x86_64.a"
if [ -n "$missing_tools" ]; then
  echo "build.sh: missing tool(s):$missing_tools" >&2
  echo "build.sh: run 'bash install.sh' first (it installs and checks them)." >&2
  exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK" "$WORK/tmp"
# tar, not "cp -a": started from inside the source directory (the way the
# README shows it), .build/ lies inside SOURCE, and cp refuses to copy a
# directory into itself.
(cd "$SOURCE" && tar cf - --exclude=./.build --exclude=./comp9 .) \
  | (cd "$WORK" && tar xf -)
chmod -R u+w "$WORK"
rm -f "$WORK/install.sh" "$WORK/build.sh"    # they have no part in the build

cd "$WORK"
env -i \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME="$WORK" TMPDIR="$WORK/tmp" LC_ALL=C LANG=C TZ=UTC \
    /bin/bash ./build_and_construct_comp.sh

[ -s "$WORK/run/cmix" ] || { echo "build.sh: the build produced no binary." >&2; exit 1; }
cp -f "$WORK/run/cmix" "$TARGET/comp9"
chmod 0755 "$TARGET/comp9"

cd "$TARGET"
rm -rf "$WORK"

printf '%-24s %10d B\n' "comp9" "$(wc -c < "$TARGET/comp9")"
