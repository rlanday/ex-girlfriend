# Ex-Girlfriend 2.0

The project is named after **“Ex-Girlfriend,” the No Doubt song**.

Ex-Girlfriend 2.0 is Ryan Landay's modification of **cmix-neif 2.0**, by
Marcus Wölk, for lossless compression of the 1,000,000,000-byte **enwik9**
benchmark. The verified self-extracting archive is **88,983,124 bytes**,
including the decoder, transformer weights, dictionary and every other asset
it needs. It is **363,268 bytes smaller (0.407%)** than cmix-neif 2.0's
89,346,392-byte archive, and 2,795,113 bytes smaller than Ex-Girlfriend 1.0
(91,778,237 bytes).

The [LTCB table](https://mattmahoney.net/dc/text.html) checked on November 17
lists cmix-neif 2.0 first. Our archive is smaller, pending the benchmark
maintainer's own check. This release is for the Large Text Compression
Benchmark. It is **not** a Hutter Prize entry: it uses
about 24.3 GiB of memory and about 95 hours in each direction on our machine,
outside that contest's limits.

How 2.0 relates to 1.0: version 1.0 modified cmix-neif 1.0.
Marcus Wölk then released cmix-neif 2.0, which adopted three of 1.0's changes
(the fast-weight head and two metadata forecasts) and added much else,
including a semantic retrieval cache (SEMCACHE) and a retrained transformer.
Version 2.0 starts again from cmix-neif 2.0. It makes that cache search harder,
adds two more caches with their own keys, spreads the online fast-weight
updates over three times as many transformer blocks, and stops resetting the
BITLSTM correction model's recurrent state every 1,024 bits. The
[1.0 changes are documented separately](docs/EX-GIRLFRIEND-1.0.md).

## Build and run

Download the verified [**archive9** (88,983,124 bytes)](https://github.com/rlanday/ex-girlfriend/releases/download/v2.0/archive9),
[compressor `comp9`](https://github.com/rlanday/ex-girlfriend/releases/download/v2.0/comp9),
[source tarball](https://github.com/rlanday/ex-girlfriend/releases/download/v2.0/ex-girlfriend-2.0-source.tar.gz)
and [SHA256SUMS](https://github.com/rlanday/ex-girlfriend/releases/download/v2.0/SHA256SUMS)
from the [v2.0 release](https://github.com/rlanday/ex-girlfriend/releases/tag/v2.0).
Binary artifacts are kept out of Git history.

Use **x86-64 Linux with AVX2/FMA (x86-64-v3)**. Allow at least **32 GiB of
RAM** (the measured peak resident set is 24.3 GiB in both directions) and
**30 GB of free disk** in the working directory. PPMd also keeps a sparse
paging file, `ppm.page`, in the working directory; its apparent size is
14.68 GB, of which only a small part is allocated. Compression and
decompression each use one CPU core and no GPU. Each direction took about
95 hours on our machine; see the measured times below before starting.

### Decode the supplied archive

Save `archive9` in a new, writable directory. It is self-extracting and
contains its decoder, weights and dictionary; no build, model files or network
access are needed. From that directory:

```sh
printf '%s  %s\n' \
  0b65bc79917f84b78f4974a2dc94d4d4c3fc6c0c53234bdcb6c282db802635a3 archive9 \
  | sha256sum -c -
chmod +x archive9
./archive9
printf '%s  %s\n' \
  159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc enwik9_uncompressed \
  | sha256sum -c -
```

The output is `enwik9_uncompressed`, exactly 1,000,000,000 bytes. If you have
the original dataset, also run `cmp /absolute/path/to/enwik9 enwik9_uncompressed`.

### Encode with the supplied compressor

The compressor takes **no arguments**. Put `comp9` and the original enwik9,
under the name `enwik9`, in a new writable directory, and run it there:

```sh
printf '%s  %s\n' \
  8d2a5fd03da10e671286726ab620183c0b6072ba68021c03076304e29f80406a comp9 \
  | sha256sum -c -
chmod +x comp9
ln -s /absolute/path/to/enwik9 enwik9     # or copy it
./comp9
printf '%s  %s\n' \
  0b65bc79917f84b78f4974a2dc94d4d4c3fc6c0c53234bdcb6c282db802635a3 archive9 \
  | sha256sum -c -
```

It writes `archive9`. Decode that in a different fresh directory as above and
compare the restored file. `comp9.args` in this repository is empty.

### Build the compressor from source

The source tree rebuilds to a **byte-identical `comp9`** with Ubuntu
clang/LLVM/lld **18.1.3-1ubuntu1**, GCC **13.3.0** support files and GNU
binutils **2.42**, on Ubuntu 24.04. Matching the toolchain matters; a
successful compile with other versions does not reproduce the binary.
[BUILD.md](docs/BUILD.md) gives the dependencies, a pitfall we hit (a stray
GCC 14 installation) and the complete check. In short:

```sh
tar -xzf ex-girlfriend-2.0-source.tar.gz
mkdir ex-girlfriend-build
cd ex-girlfriend-build
bash ../ex-girlfriend-2.0/build.sh
printf '%s  %s\n' \
  8d2a5fd03da10e671286726ab620183c0b6072ba68021c03076304e29f80406a comp9 \
  | sha256sum -c -
```

## Verified result and computational cost

| Measurement | Compression | Fresh decompression |
|---|---:|---:|
| Wall time | 340,598 s / 94.61 h | 342,467.3 s / 95.13 h |
| User CPU time | not recorded | 329,853.4 s |
| System CPU time | not recorded | 12,486.2 s |
| Peak RSS | 25,473,304 KiB / 24.29 GiB | 25,449,428 KiB / 24.27 GiB |
| Exit status | 0 | 0 |

Measured on an **AMD Ryzen Threadripper 3970X (Zen 2), Linux x86-64, Ubuntu
24.04.5 LTS**, on a shared server whose host reports about 252 GiB of RAM.
Each codec run was pinned to one logical CPU, while the server also ran other
workloads; competition for memory bandwidth and disk affects wall time. These
are observations of one run each, not controlled estimates of the cost of each
feature or a prediction for another machine. No GPU is used.

The encode ran from November 10 to November 14, 2026. The independent fresh
decode started from a separate copy of the archive in an empty directory, with
no arguments and an empty environment, and finished on **November 18 at
08:33:55 UTC**. Its output was exactly 1,000,000,000 bytes, `cmp` against the
original enwik9 exited 0, and both files have the SHA-256 below. See the
[verification summary](verification/result.json), the
[encode record](verification/encode.log) and the
[decode receipt](verification/decode_receipt.json).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `archive9` | 88,983,124 | `0b65bc79917f84b78f4974a2dc94d4d4c3fc6c0c53234bdcb6c282db802635a3` |
| `comp9` | 4,057,131 | `8d2a5fd03da10e671286726ab620183c0b6072ba68021c03076304e29f80406a` |
| Original/restored `enwik9` | 1,000,000,000 | `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc` |

The archive is the decoder stub (198,577 bytes), the transformer weight
container (3,744,629 bytes, identical to cmix-neif 2.0's), the coded payload
(85,039,910 bytes) and an 8-byte trailer. Against cmix-neif 2.0's archive,
whose stub is 183,885 bytes, the payload is **377,960 bytes smaller** and the
stub **14,692 bytes larger**: net 363,268 bytes. For comparison, cmix-neif 2.0
reports 49.18 h compression and 9.77 GiB peak on a different, faster machine;
our changes buy their bytes with memory and time.

## Provenance: what we inherited

The immediate base is **cmix-neif 2.0 by Marcus Wölk** (Hutter Prize
submission of 30 October 2026, <https://marwoelk.de/cmix-neif-2.0/>). We
imported its source package `cmix-neif-2.0.tar.gz` (SHA-256
`62da32b97f9909ce5f2d4988a84e1b9cfbff4c9947538cbc6659a2915b024874`) unchanged
as our first commit and verified the import against the tarball file by file.
Upstream's own README is kept as [README.cmix-neif.md](README.cmix-neif.md)
and its environment record as `environment.cmix-neif.txt`; their hashes and
results are upstream's, not ours.

This is a derivative compressor, not a new implementation. We retain
cmix-neif 2.0's preprocessing, dictionary, article order, metadata and census
transforms, its transformer architecture and its **trained transformer and
BITLSTM weights unchanged**, PPMd, the classical context-mixing models, the
semantic cache itself, the mixers, the arithmetic coder and the custom
executable packer. No model was trained for this release.

The main lines of descent, abbreviated from upstream's [NOTICE.md](NOTICE.md):

| Component | Attribution and relationship |
|---|---|
| cmix, cmix-hp, fast-cmix, fx-cmix, fx2-cmix, cmix-lex | Byron Knoll, Saurabh Kumar, Kaido Orav and Ibrahim Marcouch: the coder, context models, mixer, data structures and compressor base |
| fxcm text model, PAQ and PPMd lineage | Kaido Orav, and the contributors named in `NOTICE.md`, including Matt Mahoney, Alexander Rhatushnyak, Przemysław Skibiński, Márcio Pais, Zoltán Gotthardt, Dmitry Shkarin and Eugene Shelwien |
| cmix-neif 1.0 and 2.0 | Marcus Wölk: the previous base and the current one, including the semantic cache (SEMCACHE) that 2.0 tunes and multiplies, the census transform, the article and dictionary orders, the weight containers and the training of the shipped weights |
| Transformer architecture and recipe | cmix-neif-beehive by Vladimir Malinovskii (attention residuals after Kimi Team's *Attention Residuals*, one shared key/value head, value embeddings, quantization-aware training), on Vladimir Ivanov's fx2-cmix-transformer inference engine and training tree; Kimi Delta Attention from Kimi Linear and flash-linear-attention |
| Shipped transformer and BITLSTM weights | Trained by Marcus Wölk for cmix-neif 2.0, reused unchanged here |
| BITLSTM correction model | David Freelan's cmix-obias, as integrated and trained by neif |
| Adaptive mixer depth, end-to-end mixer gradient, word-label model, indel recovery | James Byrne's zmix 1.0, ported into neif |
| Grammar-match model | Halvor Yttredal's fx-deepmix |
| Article-order and preprocessing ancestry | STARLIT by Artemiy Margaritov, phda9 by Alexander Rhatushnyak, and the fx2-cmix / cmix-lex / neif work after them |
| Retrieval methods behind SEMCACHE | Khandelwal et al. (kNN-LM, 2020) and Grave, Joulin and Usunier (continuous cache, 2017), as credited upstream; implemented by neif without adopted code |
| Executable packer | neif's context-mixing packer, drawing on paq8px, paq8f, the LZMA SDK's BCJ idea and Crinkler / oneKpaq |

This table does not replace the full [inherited notices](NOTICE.md),
[training attribution](training/ATTRIBUTION.md),
[third-party licenses](third_party/README.md) or
[upstream README](README.cmix-neif.md), which distinguish copied code,
rewritten implementations and ideas.

**Ex-Girlfriend 1.0 inside the base.** cmix-neif 2.0 already contains three
changes from Ex-Girlfriend 1.0, adapted by Marcus Wölk and credited in his
`NOTICE.md`, `CHANGES.md` and README: the fast-weight head on transformer
blocks 12–15, the revision-id forecast from the page id below revision id
17,000,000, and the timestamp forecast from the slope of the last records.
They are part of the base here, not new in 2.0. 1.0's larger text-model tables
were not among the adopted changes. What 1.0 changed, and how it
measured, is in [docs/EX-GIRLFRIEND-1.0.md](docs/EX-GIRLFRIEND-1.0.md) (the
1.0 README, with links to the [v1.0 tag](https://github.com/rlanday/ex-girlfriend/tree/v1.0)).

**kNN over transformer states, and SEMCACHE.** From 7 October 2026 we were
developing a kNN-LM-style store of transformer hidden states inside the
context-mixing coder. Its variants (NEXT, L1 and a RAM-bounded "kNN-lite")
used confidence-gated lookup, store gating and continuation following. We
did not release that work, so its dates rest on our own commit history, not
on a public record. Marcus Wölk independently developed SEMCACHE, which does
the same job, and released it first in cmix-neif 2.0. This release builds on
SEMCACHE as published: it changes SEMCACHE's search constants and adds two
more SEMCACHE instances with other keys, and it does not ship our earlier kNN
code. Both designs build on earlier work: the
[continuous](https://arxiv.org/abs/1612.04426) and
[unbounded](https://arxiv.org/abs/1711.02604) caches of Grave et al. (2017),
[kNN-LM](https://arxiv.org/abs/1911.00172) (Khandelwal et al., 2020) and
[RetoMaton](https://arxiv.org/abs/2201.12431)'s pointer following (Alon et
al., 2022).

The 2.0 changes were developed by **Ryan Landay with Claude (Anthropic) and
Codex (OpenAI)** between October 31 and November 10, 2026. We did not invent
retrieval caches, locality-sensitive hashing, fast weights or LSTM state
carrying; the contribution is finding these settings and extra caches,
wiring them into this compressor and verifying the full result.

## What changed in 2.0

2.0 changes four things in the model, freezes them as compile-time constants,
and adjusts one check in the packer. Everything else codes as cmix-neif 2.0
does.

| Change | Measured on 50 MB of the 501 MB coder stream, against a cmix-neif 2.0 equivalent build | Cost |
|---|---|---|
| Wider semantic-cache search: 63 entries per bucket (was 15), 15 neighbouring buckets probed per table (was 3), key ring of 2^25 entries (2^24) and history ring of 2^26 bytes (2^25), 512 candidates re-ranked (128) | −5,573 bytes | about +2.2 GiB; more lookup time |
| A second cache keyed on transformer layer 12 | −5,323 bytes alone; −10,675 bytes together with the wider search | about +6 GiB with the wider geometry |
| A third cache on the release key with its own random projection and hash seed | −508 bytes on top of the two above, −1,123 over the warm part of the screen: it costs while empty and gains as it fills | about +6 GiB |
| Online fast-weight updates on blocks 4–15 instead of 12–15 | −2,355 bytes | about 3 h per direction at cmix-neif 2.0's speed (estimated) |
| BITLSTM recurrent state carried across its 1,024-bit windows | −1,218 bytes | none measurable |
| **All of the above, full enwik9** | **−377,960 bytes of payload; −363,268 bytes net of the larger stub** | 24.3 GiB peak; about 95 h per direction |

How to read the table. The component rows are encoder-only screens on the
first 50 MB of the 501,062,281-byte coder stream (one run each; the
run-to-run perturbation floor is about ±200 bytes). Much of the gain comes
late: the caches pay more as they fill, so 50 MB screens under-read them, and
the rows were never meant to add up to the full result. **Do not add the rows
to predict full-corpus savings.** The bottom row is the only combined, full
measurement. The memory figures are per-arm peaks from the screens; the
attribution of the full 24.3 GiB to the three caches is an inference.

### 1. A wider semantic-cache search

SEMCACHE, neif's retrieval model, looks up earlier stream positions where the
transformer was in a similar internal state and reads which byte followed
there. Its screens showed the cache was limited by search effort rather than by
storage: more candidates per bucket, more probed buckets and a deeper exact
re-rank each helped, and larger rings keep more history findable. 2.0 fixes
six constants at the measured optimum; under `PORT_ENTRY 0` the tree keeps the
release values.

Source: [`src/semantic_cache.h:59–63`](src/semantic_cache.h#L59-L63) (one of the six; the others follow the same form at lines 64–74, 93–97, 100–104 and 115–119).

```c++
#if PORT_ENTRY
constexpr int kM{63};   // entry recipe A (release 15)
#else
constexpr int kM = 15;                 // newest entries kept per bucket
#endif
```

`kFlip` 15 is the plateau of a bracket (17 is the structural maximum and
gained nothing more); `kPre` 512 is the largest value the code's static
assertion allows, and needs `kDedupLog` 10 for its candidate set.

### 2. Two more caches with their own keys

The release cache keys on the last transformer layer's MLP input. 2.0 adds two
independent instances, each with its own running key mean, random ±1
projection, SimHash tables, sketch hyperplanes, key ring and learned end
stage. One keys on the depth-mixed input of **layer 12**: a layer bracket of 4,
8 and 12 read +751, −1,609 and −5,323 bytes, so the middle of the network finds
different neighbours from the top. The other keys on the **same release key
with a different seed**, so its hash tables partition the space differently
and surface other neighbours. Each extra cache adds one input to the first
mixer layer, one column to the final mixer and one end stage behind the
release cache's.

Source: [`src/cmx_meas.h:543–569`](src/cmx_meas.h#L543-L569), abbreviated.

```c++
constexpr int kEntrySkN = 2;                       // CMX_SEMKEYS=R,12
constexpr int kEntrySkLayer[kSkMax] = {kSkRel, 12, -1};
constexpr int kEntrySk2Use = 3;                    // CMX_SEMKEY2_USE: 1 layer-0 input + end-mixer column, 2 end stages
constexpr int kEntryFwFirst = 4;                   // CMX_FW_BLOCKS=4
constexpr long long kEntryBitlSeq = 0;             // BITL_SEQ=0
__attribute__((noinline)) inline void EntryApply() {
  // the recipe
  g_nsk = kEntrySkN;
  ...
  g_fw_first = kEntryFwFirst;
  g_bitl_seq = kEntryBitlSeq;
  // every other coding knob: the release default
  ...
```

The caches are created in the predictor, after the release cache, with seed
`0x5EC0CAC4E2 ^ k * 0x9E3779B97F4A7C15` for cache *k*
([`src/predictor.cpp:254–277`](src/predictor.cpp#L254-L277)). The layer-12 key
is copied from the transformer's forward pass for read rows only
([`src/tf1/opt/model_opt.cpp:842–843`](src/tf1/opt/model_opt.cpp#L842-L843)).
Both run in the encoder and the decoder from coded bytes only.

### 3. Fast weights on twelve blocks

cmix-neif 2.0 contains 1.0's fast-weight head: a second transformer
distribution from small weight matrices learned online, while coding, on the
outputs of MLP blocks 12–15. 2.0 extends the online updates to blocks 4–15.
The bracket read −1,985 bytes for 8 blocks, −2,355 for 12 and −2,385 for 16;
12 is kept because 16 is within noise and slower.

Source: [`src/tf1/opt/model_opt.cpp:515–517`](src/tf1/opt/model_opt.cpp#L515-L517).

```c++
#if PORT_CMX_MEAS
    FWK = NL - cmx_meas::g_fw_first;   // research CMX_FW_BLOCKS: blocks g_fw_first .. NL-1
#endif
```

### 4. BITLSTM state across windows

The BITLSTM correction model zeroes its recurrent state at the start of every
window of 1,024 bits, as in its training. 2.0 keeps the state across those
windows and resets it only when the head switches between its weight sets.
The weights are not retrained.

Source: [`src/bit_head.cpp:976–984`](src/bit_head.cpp#L976-L984).

```c++
  bool cmx_switch = false;
  if (s.trig != 0 && s.nbuf == 0) { s.NextSlot(); cmx_switch = true; }
  const long long cmx_seq = cmx_meas::g_bitl_seq;
  const bool cmx_reset = cmx_seq == 1024
      ? (s.seq > 0 && (s.tbptt ? (s.nbuf == 0) : ((s.i % (unsigned long long)s.seq) == 0)))
      : (cmx_switch || (cmx_seq > 0 && (s.i % (unsigned long long)cmx_seq) == 0));
  if (cmx_reset) {
    for (int r = 0; r < s.cells; ++r) { s.h[r] = 0.0f; s.c[r] = 0.0f; }
  }
```

### 5. The recipe as constants, and what else the source contains

The four changes were found with a research build whose switches are read
from environment variables (`CMX_*`, `BITL_SEQ`). The shipping build sets
`PORT_ENTRY 1` in [`src/port_cfg.h`](src/port_cfg.h): `main()` calls
`EntryApply()` once, before any model exists, in every mode that codes or
decodes the main stream, and that function writes every knob the model reads
(the recipe, and the release default of every other knob). The shipping image
does not read the environment: it imports `getenv` zero times and contains no
`CMX_` or `BITL_SEQ` strings.

Source: [`src/runner.cpp:1485–1505`](src/runner.cpp#L1485-L1505).

```c++
#if PORT_ENTRY
#if !AUX_BUILD
  {
    const bool helper_block = argc > 1 && argv[1][0] == '-' && argv[1][1] == 'D';
#if PORT_ENTRY_TEST
    if (!helper_block && argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'e' || argv[1][1] == 'n') && argv[1][2] == '\0')
      cmx_meas::Read();
#endif
    if (!helper_block) cmx_meas::EntryApply();
  }
#endif
#elif PORT_CMX_MEAS
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'e' || argv[1][1] == 'n') && argv[1][2] == '\0') cmx_meas::Read();
#endif
```

The two helper blocks (article order and dictionary hints) were coded at
build time with release defaults, so `-D` helper decodes keep them.

The source also contains the **research code** used to find the recipe:
about 4,000 added lines (mostly in `src/cmx_meas.h`, `src/predictor.cpp`,
`src/runner.cpp`, `src/models/fxcmv1.cpp` and `src/mixer/mixer.cpp`) for
measurement switches and screened alternatives that are not part of the
recipe. They are compiled in, because `PORT_ENTRY` implies `PORT_CMX_MEAS`, but
`EntryApply()` sets each of them to its release default. They are kept so the
source is exactly what was measured and shipped. The release claim rests only
on the full measured run, not on an argument that this code is inert. The
decoder stub is 14,692 bytes larger than upstream's (the recipe and the
compiled-in research code, partly offset by building without PGO); that cost
is included in the net result. The shipping binary still
prints two `CMXTEL semkey2 init` lines to stderr at start-up; they do not
affect the output.

### 6. The packer's cut check

neif's packer cuts encoder-only code out of the decoder stub and checks that
nothing on the decode path reaches the cut section. 2.0 changes that check,
[`tools_src/packer/cut_dpath.py`](tools_src/packer/cut_dpath.py), in two ways:
it fixes a `NameError` on one error path, and it treats an immediate operand
that points inside a function body where no symbol sits as a number rather
than a code pointer (rule 4a). In this build six such immediates are memset
lengths; none points into the cut section. The unmodified check also passes
on the shipping build; rule 4a was needed for an intermediate research build.

## Build recipe used for the measured release

The release tree is the certified source with the two build switches applied
that the measured build applied to its copy, so `build.sh` needs no edits:

* [`src/port_cfg.h:74`](src/port_cfg.h#L74) sets `PORT_ENTRY 1`.
* The `makefile` has every `-fprofile-use=$(ROOT_DIR)/pgo_data` removed from its
  [`prof_use`](makefile#L80-L90) and [`cmix_aux`](makefile#L180-L184) targets, so the shipped binary is built with LTO and **without
  profile-guided optimization**. The upstream profile no longer matches 44
  changed functions, and a PGO build is a different coder that would need its
  own verification. `pgo_data/` is still required: its presence skips the
  constructor's profiling branch.

Everything else is upstream's build: `build.sh` runs
`build_and_construct_comp.sh` in a copy under an empty environment. The
measured build ran on November 9, 2026 with clang 18.1.3, lld 18 and GCC
13.3.0 support files. [BUILD.md](docs/BUILD.md) has the details and
[verification/source_identity.json](verification/source_identity.json) the
rebuild record.

## Headroom and scope

The margin is **363,268 bytes (0.407%)** over cmix-neif 2.0's published LTCB
size, from one full encode and one independent fresh decode. The component
screens explain why these settings were chosen; they are not a measurement of
how much each contributes at full size. This release adds **no training** and
reuses neif's shipped weights; that training effort is upstream's. We have no
audited total of the CPU time spent on all experiments, so none is claimed.

The 0.4% margin, 24.3 GiB of memory and 95 hours per direction are far from
the Hutter Prize's 1% improvement requirement and its memory and time limits.
This release claims no Hutter Prize result.

## License and modification notice

The combined program and our modifications are distributed under
**GNU GPL version 3 only (`GPL-3.0-only`)**. See [LICENSE](LICENSE).
Copyright and license notices of inherited components remain in force; MIT,
BSD, LLVM-exception, zlib and other notices are preserved in `third_party/`
and in the source files. Wikipedia-derived data keeps its separate documented
license.

Modification dates: **October 31–November 10, 2026** for the 2.0 changes,
**November 18, 2026** for this release documentation and evidence. See
[EX-GIRLFRIEND.md](EX-GIRLFRIEND.md) for the changed-file inventory.

**Inherited provenance.** The native compressed-weight loader includes
Vladimir Ivanov's bit-exact transcription of the CUDA libdevice single-precision
sine and cosine, inherited unchanged through cmix-neif. Upstream's
[training/ATTRIBUTION.md](training/ATTRIBUTION.md) documents the open
licensing question about that transcription and other inherited provenance
gaps; this release preserves those disclosures and the original notices.
