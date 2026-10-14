#!/bin/bash
# install.sh - installs the system dependencies of the build.  This is the only
# step that needs root and network access; it builds nothing, and build.sh runs
# afterwards offline and unprivileged.
#
# Non-interactive and repeatable: a second run changes nothing.
set -eu
export DEBIAN_FRONTEND=noninteractive

# The tools the build needs.  One line, so that it stays readable for someone
# who only reads this file.
# gawk (the packer script uses strtonum), python3 (the section guard) and the
# compiler-rt profile runtime are listed explicitly; the check at the end
# covers everything build.sh touches.
PACKAGES="build-essential make binutils ca-certificates curl gnupg gawk python3"
# The toolchain is PINNED to 18: the binary is built with profile-guided
# optimisation and LTO, and the linker (lld) does the code generation -- a
# different major version produces a different binary.
# libclang-rt-18-dev carries libclang_rt.profile-x86_64.a, which the
# profile-guided first pass links against; without it lld-18 stops with
# "cannot open .../libclang_rt.profile-x86_64.a".
LLVM_PACKAGES="clang-18 lld-18 llvm-18 libclang-rt-18-dev"
LLVM_VERSION="18"

# Not only the major version matters.  The size of the compressor counts in
# the score, and a later point release of the compiler, the C++ headers, the C
# library or the linker can move its bytes although it behaves the same.  So
# every package comes from the Ubuntu archive as it was at ONE moment: apt
# (version 2.7 or later, as in Ubuntu 24.04) reads all package lists from the
# Ubuntu snapshot service, snapshot.ubuntu.com, at this time (UTC).  The
# submitted compressor was built from exactly this state.
APT_SNAPSHOT="20260927T160000Z"
# The build-relevant packages and their versions in that state, one
# package=version per line.  After the installation each one is compared with
# what dpkg reports; a difference is reported, it does not stop the
# installation.
REFERENCE_VERSIONS="
clang-18=1:18.1.3-1ubuntu1
libclang-cpp18=1:18.1.3-1ubuntu1
libllvm18=1:18.1.3-1ubuntu1
libclang-common-18-dev=1:18.1.3-1ubuntu1
libclang-rt-18-dev=1:18.1.3-1ubuntu1
lld-18=1:18.1.3-1ubuntu1
llvm-18=1:18.1.3-1ubuntu1
llvm-18-linker-tools=1:18.1.3-1ubuntu1
gcc=4:13.2.0-7ubuntu1
g++=4:13.2.0-7ubuntu1
gcc-13=13.3.0-6ubuntu2~24.04.1
gcc-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1
g++-13=13.3.0-6ubuntu2~24.04.1
g++-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1
cpp-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1
libgcc-13-dev=13.3.0-6ubuntu2~24.04.1
libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1
libstdc++6=14.2.0-4ubuntu2~24.04.1
libgcc-s1=14.2.0-4ubuntu2~24.04.1
libc6=2.39-0ubuntu8.9
libc6-dev=2.39-0ubuntu8.9
libc-dev-bin=2.39-0ubuntu8.9
linux-libc-dev=6.8.0-142.142
binutils=2.42-4ubuntu2.10
binutils-x86-64-linux-gnu=2.42-4ubuntu2.10
libbinutils=2.42-4ubuntu2.10
libgmp10=2:6.3.0+dfsg-2ubuntu6.1
libmpfr6=4.2.1-1build1.1
libmpc3=1.3.1-1build1.1
libisl23=0.26-3build1.1
make=4.3-4.1build2
gawk=1:5.2.1-2ubuntu0.1
python3.12-minimal=3.12.3-1ubuntu0.17
"

# The package names above are Debian/Ubuntu names and are installed with apt.
# On another distribution, install the same programs with its own package
# manager; the check at the end of this file lists exactly what has to be
# there afterwards.
if ! command -v apt-get >/dev/null 2>&1; then
  echo "install.sh: this script uses apt (Debian/Ubuntu)." >&2
  echo "  On another distribution install: $PACKAGES $LLVM_PACKAGES" >&2
  echo "  and re-run this script to let it check the result." >&2
fi

SUDO=""
if [ "$(id -u)" != "0" ]; then
  command -v sudo >/dev/null || { echo "install.sh needs root (or sudo)." >&2; exit 1; }
  SUDO="sudo"
fi

# PIN selects the snapshot for apt; empty means the current archive.  If the
# snapshot service cannot be used, the installation falls back to the current
# archive and says so on stderr: it must not fail because one more service is
# down.  The version check below then shows what differs.
PIN=""
unpin() {
  echo "install.sh: WARNING: $1." >&2
  echo "install.sh: WARNING: using the CURRENT Ubuntu archive instead of snapshot $APT_SNAPSHOT; package versions may differ from the reference state (see the version check)." >&2
  PIN=""
}

if command -v apt-get >/dev/null 2>&1; then
  OS_ID="$(. /etc/os-release 2>/dev/null && printf '%s' "${ID:-}")"
  APT_VERSION="$(dpkg-query -W -f='${Version}' apt 2>/dev/null || true)"
  # An apt older than 2.7 (Ubuntu 22.04 has 2.4) accepts --snapshot without
  # error but ignores it, so its version decides, not its exit code.
  if [ "$OS_ID" != "ubuntu" ]; then
    echo "install.sh: WARNING: not Ubuntu; the package state is not pinned to snapshot $APT_SNAPSHOT." >&2
  elif ! dpkg --compare-versions "${APT_VERSION:-0}" ge 2.7 2>/dev/null; then
    echo "install.sh: WARNING: apt ${APT_VERSION:-(unknown)} has no snapshot support (2.7 or later needed); the package state is not pinned to snapshot $APT_SNAPSHOT." >&2
  else
    PIN="--snapshot $APT_SNAPSHOT"
    # The snapshot service answers over https only, and apt needs the CA
    # certificates to verify it.  A bare image has none; then they come from
    # the current archive first (they play no part in the build).
    if [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
      $SUDO apt-get update
      $SUDO apt-get install -y --no-install-recommends ca-certificates || true
    fi
    # --error-on=any: a list that could not be fetched fails the update, so an
    # unreachable snapshot cannot leave apt working from stale lists.  Then
    # apt has to list the snapshot among its sources, or it is not in use.
    # shellcheck disable=SC2086
    if ! $SUDO apt-get update $PIN --error-on=any; then
      unpin "the Ubuntu snapshot service (snapshot.ubuntu.com) could not be reached"
    elif ! apt-cache $PIN policy | grep -qF "/$APT_SNAPSHOT "; then
      unpin "apt does not list snapshot $APT_SNAPSHOT among its sources"
    else
      echo "install.sh: package state pinned to the Ubuntu archive snapshot $APT_SNAPSHOT."
    fi
  fi
  [ -n "$PIN" ] || $SUDO apt-get update
  # shellcheck disable=SC2086
  if ! $SUDO apt-get install -y --no-install-recommends $PIN $PACKAGES; then
    [ -n "$PIN" ] || exit 1
    unpin "installing from snapshot $APT_SNAPSHOT failed"
    $SUDO apt-get update
    # shellcheck disable=SC2086
    $SUDO apt-get install -y --no-install-recommends $PACKAGES
  fi

  # On older distributions (for example Ubuntu 22.04) clang-18 is not in the
  # standard sources; then it comes from apt.llvm.org.  If it is already
  # present, nothing happens here.  apt.llvm.org keeps no snapshots, so on such
  # a system the toolchain is the one apt.llvm.org serves on the day.
  # "apt-cache madison" and not "apt-cache show": on Ubuntu 22.04 clang-18 is
  # a name other packages refer to, and "show" then succeeds without offering
  # any version.
  if ! command -v "clang++-$LLVM_VERSION" >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    if [ -z "$(apt-cache $PIN madison "clang-$LLVM_VERSION" 2>/dev/null)" ]; then
      [ -z "$PIN" ] || unpin "clang-$LLVM_VERSION is not in this Ubuntu release; it comes from apt.llvm.org, which keeps no snapshots"
      CODENAME="$(. /etc/os-release && printf '%s' "$VERSION_CODENAME")"
      curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | $SUDO gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg
      printf 'deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/%s/ llvm-toolchain-%s-%s main\n' \
        "$CODENAME" "$CODENAME" "$LLVM_VERSION" \
        | $SUDO tee /etc/apt/sources.list.d/llvm-toolchain.list > /dev/null
      $SUDO apt-get update
    fi
    # shellcheck disable=SC2086
    if ! $SUDO apt-get install -y --no-install-recommends $PIN $LLVM_PACKAGES; then
      [ -n "$PIN" ] || exit 1
      unpin "installing the toolchain from snapshot $APT_SNAPSHOT failed"
      $SUDO apt-get update
      # shellcheck disable=SC2086
      $SUDO apt-get install -y --no-install-recommends $LLVM_PACKAGES
    fi
  fi
fi

# --- Check: the versions that decide the bytes of the build -----------------
# A difference does not stop anything: the build still works, but the
# compressor it produces may then differ from the submitted one in its bytes.
# The complete package state follows, so that a difference can be traced.
if command -v dpkg-query >/dev/null 2>&1; then
  total=0; differ=0
  while IFS='=' read -r pkg want; do
    [ -n "$pkg" ] || continue
    total=$((total + 1))
    have="$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || true)"
    if [ "$have" = "$want" ]; then
      printf 'VERSION OK       %-30s %s\n' "$pkg" "$have"
    else
      printf 'VERSION DIFFERS  %-30s installed %s, reference %s\n' "$pkg" "${have:-(none)}" "$want"
      differ=$((differ + 1))
    fi
  done <<< "$REFERENCE_VERSIONS"
  if [ "$differ" -eq 0 ]; then
    echo "install.sh: all $total build-relevant packages have the versions of snapshot $APT_SNAPSHOT."
  else
    echo "install.sh: WARNING: $differ of $total build-relevant packages differ from snapshot $APT_SNAPSHOT (lines VERSION DIFFERS); the build still works, but the compressor may not be byte-identical to the submitted one." >&2
  fi
  echo "--- dpkg -l: the complete package state after install.sh"
  COLUMNS=300 dpkg -l
fi

# --- Check: whatever build.sh needs later has to be here NOW ----------------
# Without this loop a missing toolchain would only show up during the build,
# and by then the network is gone.
missing=0
for w in make cc clang++-18 clang-18 ld.lld-18 llvm-profdata-18 llvm-strip-18 llvm-objcopy-18 llvm-nm-18 objcopy nm gawk python3 gcc tar; do
  if command -v "$w" >/dev/null 2>&1; then
    printf 'OK      %-20s %s\n' "$w" "$(command -v "$w")"
  else
    printf 'MISSING %s\n' "$w"; missing=$((missing + 1))
  fi
done
# The profile runtime is a LIBRARY, not a program, so the loop above cannot see
# it; it is checked here on its own.
if command -v clang-18 >/dev/null 2>&1; then
  RTLIB="$(clang-18 --print-file-name=libclang_rt.profile-x86_64.a 2>/dev/null || true)"
  if [ -n "$RTLIB" ] && [ -f "$RTLIB" ]; then
    printf 'OK      %-20s %s\n' "profile runtime" "$RTLIB"
  else
    printf 'MISSING %s\n' "libclang_rt.profile-x86_64.a (package libclang-rt-18-dev)"
    missing=$((missing + 1))
  fi
fi
[ "$missing" -eq 0 ] || { echo "install.sh: $missing tool(s) missing." >&2; exit 1; }
echo "install.sh: all tools present."
