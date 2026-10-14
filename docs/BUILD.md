# Reproducing Ex-Girlfriend 1.0

This document describes the original, verified FINAL artifact. Source files and
constructor inputs have been preserved; the expected compressor is 3,995,122
bytes, SHA-256 `952ce43a221b19ea204e0f2dcf8afd86dbf8e85f823eb5f0949dd61ad1b87e67`.

## Source and dependencies

Download the [v1.0 source tarball](https://github.com/rlanday/ex-girlfriend/releases/download/v1.0/ex-girlfriend-1.0-source.tar.gz)
and [SHA256SUMS](https://github.com/rlanday/ex-girlfriend/releases/download/v1.0/SHA256SUMS)
into the same directory, then verify and extract the source:

```sh
grep '  ex-girlfriend-1.0-source.tar.gz$' SHA256SUMS | sha256sum -c -
tar -xzf ex-girlfriend-1.0-source.tar.gz
cd ex-girlfriend-1.0
```

Alternatively, from a checkout of the release source, export the Git tree so
the compiler works without repository metadata:

```sh
mkdir ../ex-girlfriend-1.0
git archive HEAD | tar -x -C ../ex-girlfriend-1.0
cd ../ex-girlfriend-1.0
```

The target is Linux x86-64-v3 (AVX2 and FMA). The checked compiler family is
Ubuntu clang/LLVM/lld 18.1.3-1ubuntu1 with GNU binutils 2.42. The inherited
`install.sh` records package versions and Ubuntu snapshot `20260927T160000Z`.
It installs packages and can fall back to different versions with warnings;
successful installation alone does not establish byte-identical reconstruction.
Use an Ubuntu 24.04 build environment for that package reference, and inspect
the script before running it: this step installs system packages and needs
network access plus root or sudo. From the extracted source directory:

```sh
bash install.sh
```

Read its package-version comparison and any fallback warnings. It does not
force an already-installed newer package to match the reference. For an exact
reconstruction, resolve version differences before building; the final checksum
is the acceptance check. The dependency installer has not been independently
tested in a pristine Ubuntu environment for this release.

Required commands include
`make`, `clang++-18`, `clang-18`, `ld.lld-18`, `llvm-profdata-18`, `llvm-strip-18`,
`llvm-objcopy-18`, GNU `objcopy`, `gawk`, `python3`, `gcc` and `tar`, plus the
LLVM profile runtime archive.

`environment.cmix-neif.txt` is upstream's environment record, including its
original artifact hashes. Those hashes are not Ex-Girlfriend's hashes.

## Construct the compressor

Use a new output directory. `build.sh` clears `.build/` there, copies its source
and runs the constructor with an empty environment and a fixed system PATH.
It requires no network or root access once dependencies are installed.
The matching tools must be available on its fixed PATH (normally `/usr/bin` or
`/usr/local/bin`); exporting a custom toolchain PATH alone does not override it.
The constructor runs native helper programs and unrestricted `make -j`, so
use an executable filesystem and allow for parallel compiler memory use.

```sh
mkdir ../ex-girlfriend-build
cd ../ex-girlfriend-build
bash ../ex-girlfriend-1.0/build.sh
printf '%s  %s\n' \
  952ce43a221b19ea204e0f2dcf8afd86dbf8e85f823eb5f0949dd61ad1b87e67 comp9 \
  | sha256sum -c -
```

The build includes native compilation, constructor helper runs for hint/order
blocks, packing and full/cut self-extractor checks. It is not just compiling a
single source file. The historical local reconstruction used the same pinned
LLVM tools installed privately and placed first in PATH by a wrapper; the
retained [build log](../verification/build.log) and
[source-identity receipt](../verification/source_identity.json) document that
scope. The runtime, constructor and asset files in this repository match that
checked source tree byte for byte; the publication changes are documentation
and evidence. This demonstrates local byte-for-byte reproduction, not a test
of every distribution or compiler combination. A new build on another system
must satisfy the compressor checksum. Do not treat a different hash as the
verified release, even if compilation succeeds.

The make target is named `prof_use`, but FINAL removes all `-fprofile-use`
flags and keeps LTO. Retain `pgo_data/default.profdata`: its presence avoids the
constructor's profiling branch. Keep the dictionary, hint source, article-order
residual, weight container, encoder linker script, export map and packer source.
Changing compiler flags, source line positions or math implementations can change
the executable or coded stream. The fixed assets have their own manifest.

## Full encode and fresh decode

Obtain enwik9 independently. Its required SHA-256 is
`159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc`
and its size is exactly 1,000,000,000 bytes. Provide at least 25 GB of free
working space, writable temporary storage and memory above the ~10 GiB process
RSS measured on our host. PPM uses a sparse paging file, so disk availability
and cache pressure matter independently of RSS.

```sh
./comp9 -e /absolute/path/to/enwik9 archive9
printf '%s  %s\n' \
  a85629443ad1afabf36c88fb1f2dc1f3bcdaf8f19ce3f31d0948e93bb0128a6c archive9 \
  | sha256sum -c -
mkdir ../ex-girlfriend-decode
cp archive9 ../ex-girlfriend-decode/
cd ../ex-girlfriend-decode
chmod +x archive9
./archive9
cmp /absolute/path/to/enwik9 enwik9_uncompressed
sha256sum enwik9_uncompressed
```

Decoder startup extracts the embedded assets itself; do not prepopulate the
fresh directory with weights or dictionaries. The artifact is CPU-only and
does not need the training environment. The release verification used this
no-argument SFX path and exact full-file comparison, not a shortened test.
