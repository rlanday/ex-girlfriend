# Reproducing Ex-Girlfriend 2.0

This document describes the verified 2.0 artifact. The expected compressor is
4,057,131 bytes, SHA-256
`8d2a5fd03da10e671286726ab620183c0b6072ba68021c03076304e29f80406a`.

## Source

Download the [v2.0 source tarball](https://github.com/rlanday/ex-girlfriend/releases/download/v2.0/ex-girlfriend-2.0-source.tar.gz)
and [SHA256SUMS](https://github.com/rlanday/ex-girlfriend/releases/download/v2.0/SHA256SUMS)
into the same directory, then verify and extract:

```sh
grep '  ex-girlfriend-2.0-source.tar.gz$' SHA256SUMS | sha256sum -c -
tar -xzf ex-girlfriend-2.0-source.tar.gz
cd ex-girlfriend-2.0
```

Or export the tagged tree from a clone, so the build sees no repository
metadata:

```sh
git clone https://github.com/rlanday/ex-girlfriend.git
mkdir ex-girlfriend-2.0
git -C ex-girlfriend archive v2.0 | tar -x -C ex-girlfriend-2.0
cd ex-girlfriend-2.0
```

## Dependencies

The target is Linux x86-64-v3 (AVX2 and FMA). The checked toolchain is
Ubuntu 24.04 with clang/LLVM/lld **18.1.3-1ubuntu1**, GCC **13.3.0**
(`g++-13`, `libstdc++-13-dev`) and GNU binutils **2.42**. The inherited
`install.sh` records package versions and an Ubuntu snapshot. It installs
system packages, so it needs network access and root or sudo; read it before
running it. It can fall back to other versions with a warning, and a
successful installation alone does not make the build byte-identical.

```sh
bash install.sh
```

Required commands: `make`, `clang++-18`, `clang-18`, `ld.lld-18`,
`llvm-profdata-18`, `llvm-strip-18`, `llvm-objcopy-18`, GNU `objcopy`, `gawk`,
`python3`, `gcc` and `tar`, plus the LLVM profile runtime archive.
`build.sh` checks them before it starts.

**Pitfall: a second GCC installation.** clang picks the newest GCC directory
under `/usr/lib/gcc/x86_64-linux-gnu/`. On our build host a
`libgcc-14-dev` package had been installed without `libstdc++-14-dev`, so
clang chose GCC 14 and failed with `fatal error: 'vector' file not found`.
If that happens, remove the stray GCC 14 development packages, or make
`clang-18` and `clang++-18` on `PATH` wrappers that pass
`--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13`. We used the wrapper for
the release rebuild; it changes only where headers and libraries are found,
and the final checksum below is the acceptance test.

## Construct the compressor

Use a new output directory. `build.sh` copies the source into `.build/`
there, runs the constructor under an empty environment with a fixed system
`PATH`, and leaves only `comp9`. It needs no network or root access.

```sh
mkdir ../ex-girlfriend-build
cd ../ex-girlfriend-build
bash ../ex-girlfriend-2.0/build.sh
printf '%s  %s\n' \
  8d2a5fd03da10e671286726ab620183c0b6072ba68021c03076304e29f80406a comp9 \
  | sha256sum -c -
```

Because the `PATH` inside the build is fixed, the tools must be in
`/usr/bin` or `/usr/local/bin`; putting a private toolchain first on your own
`PATH` does not reach the build. Our verification rebuild used a copy of
`build.sh` with the pinned Ubuntu LLVM 18.1.3-1ubuntu1 tools placed first on
that fixed `PATH`, and `make -j2` instead of `make -j` (the job count does not
change the output). The constructor compiles the program, runs helper builds
that code the article-order and dictionary-hint blocks, packs the executable
and checks the cut decoder stub. Allow for parallel compiler memory use.

The build uses no profile-guided optimization: every `-fprofile-use` flag is
removed from the `makefile`, and LTO is kept. Keep `pgo_data/default.profdata`
anyway; its presence skips the constructor's profiling branch. `src/port_cfg.h`
sets `PORT_ENTRY 1`, which compiles the 2.0 recipe in as constants. Changing
compiler flags, line positions or math implementations can change the
executable or the coded stream. Do not treat a different hash as the verified
release, even if the build succeeds.

## Full encode and fresh decode

Obtain enwik9 independently (1,000,000,000 bytes, SHA-256
`159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc`). Provide
at least 32 GiB of RAM and 30 GB of free disk. Each direction took about 95
hours on our machine.

```sh
mkdir ../ex-girlfriend-encode && cd ../ex-girlfriend-encode
cp ../ex-girlfriend-build/comp9 .
ln -s /absolute/path/to/enwik9 enwik9
./comp9                      # no arguments; writes archive9
printf '%s  %s\n' \
  0b65bc79917f84b78f4974a2dc94d4d4c3fc6c0c53234bdcb6c282db802635a3 archive9 \
  | sha256sum -c -
mkdir ../ex-girlfriend-decode
cp archive9 ../ex-girlfriend-decode/
cd ../ex-girlfriend-decode
chmod +x archive9
./archive9                   # writes enwik9_uncompressed
cmp /absolute/path/to/enwik9 enwik9_uncompressed
```

The decoder extracts its embedded assets itself; do not put weights or
dictionaries in the decode directory. The release verification used exactly
this no-argument path for both directions and a full-file comparison.
