# Reproducing Ex-Girlfriend 1.0

This document describes the original, verified FINAL artifact. Source files and
constructor inputs have been preserved; the expected compressor is 3,995,122
bytes, SHA-256 `952ce43a221b19ea204e0f2dcf8afd86dbf8e85f823eb5f0949dd61ad1b87e67`.

## Source and dependencies

Use the source tarball attached to the private draft release, or export the Git
tree so the compiler works from source without repository metadata:

```sh
mkdir ../ex-girlfriend-source
git archive HEAD | tar -x -C ../ex-girlfriend-source
```

The target is Linux x86-64-v3 (AVX2 and FMA). The checked compiler family is
Ubuntu clang/LLVM/lld 18.1.3-1ubuntu1 with GNU binutils 2.42. The inherited
`install.sh` records package versions and Ubuntu snapshot `20260927T160000Z`.
It installs packages and can fall back to different versions with warnings;
successful installation alone does not establish byte-identical reconstruction.
Inspect it before running it on a build machine. Required commands include
`make`, `clang++-18`, `clang-18`, `ld.lld-18`, `llvm-profdata-18`, `llvm-strip-18`,
`llvm-objcopy-18`, GNU `objcopy`, `gawk`, `python3`, `gcc` and `tar`, plus the
LLVM profile runtime archive.

`environment.cmix-neif.txt` is upstream's environment record, including its
original artifact hashes. Those hashes are not Ex-Girlfriend's hashes.

## Construct the compressor

Use a new output directory. `build.sh` clears `.build/` there, copies its source
and runs the constructor with an empty environment and a fixed system PATH.
It requires no network or root access once dependencies are installed.

```sh
mkdir ../ex-girlfriend-build
cd ../ex-girlfriend-build
bash ../ex-girlfriend-source/build.sh
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
scope. A new build on another system must satisfy the compressor checksum.

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
