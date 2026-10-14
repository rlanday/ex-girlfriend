# The four non-neural artefacts: dictionary, hints, article order, PGO profile

The compressor is a **concatenation** of a packed ELF image and appended blocks,
and so is the archive:

```
comp9     = packed image + comp_hints + comp_order + weight container + header (16 B)
archive9  = .decomp_bin + .dict.comp + weight container + out (payload) + header (16 B)
```

In the builds of this entry `comp_hints` is 57,504 B and `comp_order` 43,780 B.
Both count **once** in S, because only the compressor carries them; the
decompressor inside `archive9` needs neither. The weight container counts twice.

The build of the development tree bound all four artefacts by checksum and
aborted if one did not match (the form of the invocation with
`tools/build_variant.sh`; the paths are those of the same files in this
package, the checksums those of this entry):

```bash
export CACHE=/…/<profile-dir>                     # PGO    md5 6671d8d1 (section 4)
export B6_HINTS_FILE=dictionary/english.hints     # hints  md5 e43bae14
DICT=dictionary/english.dic                       # dict   md5 3f9224b7
ORDER=src/readalike_prepr/data/new_article_order  # order  md5 3beb7ee1 (section 3)
export SIZEOPT_TUNE=-mtune=k8
export SIZEOPT_LDFLAGS="-no-pie -Wl,--icf=all -Wl,-z,norelro -Wl,-T,$LDS"
bash training/data/tools/build_variant.sh "$VAR" "$FEAT" --release --reuse-pgo --jobs 8 \
     --packer stub --src "$TREE" --dict "$DICT" --order "$ORDER" --out "$OUT"
```

Of the four, exactly **one** (the dictionary) is *bit-exactly derived from
`enwik9`* inside the coder. The other three are **files, not recipes**; their
construction is documented, but they are not bit-reproducible from `enwik9`
alone. Section 5 lists this per artefact.

---

## 1. Dictionary: `dictionary/english.dic`

### Shipped and derived, at two different places

* **Inside the compressor binary: no.** The hint package is fixed in the package
  source and replaces the `comp_dict` block by `comp_hints`. The `-e` path derives the
  dictionary from **raw `enwik9` plus the hint package**
  (`runner.cpp`: `dict_derive::BuildDictFromInput(input_path, ".hints",
  ".dict_derived")`). `comp_dict` is **0 B** in this entry.
* **Inside the archive: yes.** The coder compresses the derived dictionary again
  and appends `.dict.comp`; the *decompressor* cannot derive it, because it has
  no `enwik9`. Its size depends on the binary that codes it; in this entry it is
  134,912 B.

### Recipe A: the derivation the entry performs (deterministic)

```bash
g++ -O2 -std=c++17 -DDICT_DERIVE_MAIN src/preprocess/dict_derive.cpp -o dd
./dd enwik9 dictionary/english.hints out.dic
cmp out.dic dictionary/english.dic            # must be identical
```

The decision path is pure integer arithmetic (fixed-point `log2`, no `double`),
so the result is **bit-identical across machines**, verified on two machines and
additionally through the binary path (fresh empty cwd, absolute path).

**Cost:** 94.2 s wall, peak RSS 1.04 GiB, plus ~100 s to produce `.dict.comp`.

### Recipe B: how the file originally came to exist (NOT bit-reproducible)

| stage | tool | result |
|---|---|---|
| inherited cmix-lex dictionary | - | `english.dic`, 44,515 words, 411,996 B |
| +365 words | `make_w4.py` | `cb_cbow_w4e.dic`, 44,880 words, md5 `cc6d0317` |
| +~10,000 words | cost map `(count-2)*(len-3)`, `combine.py --order gain` | 54,857 lines, the entry count of the source |
| band-4 ordering | `ppmi_sparse.py` + k-means on ½ CBOW + ½ PPMI | the band-4 ordered list, md5 `29c0004e` |
| **band-3 ordering** | `order_b3.py`, K=512 clusters, seven fixed sub-bands | **`dictionary/english.dic`** |

The ordering stages use pre-trained sentence embeddings (`intfloat/e5-large-v2`,
`BAAI/bge-reranker-base`), k-means and randomized SVD. **No script rebuilds this
file bit-identically from `enwik9`**, seeds and model versions were not recorded.
The **file is the artefact**. Its measured value against the inherited record
dictionary is approximately **-304,600 B** (sum of measured components).

### Result

| | |
|---|---|
| size | 520,804 B, 54,857 lines |
| sha256 | `0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535` |
| md5 | `3f9224b71a655d5c9cb90e1de4ae9edc` |
| wired constants | the entry count 54,857 is a constant of the source (`src/models/fxcmv1.cpp`) |

---

## 2. Hint package: `dictionary/english.hints` → `comp_hints`

### How the coder uses it

Instead of the dictionary, the hint package is compressed with the build helper
`cmix_aux` (a separate, never-shipped binary built with `-DAUX_BUILD=1`) and
concatenated into the package as the `comp_hints` block. The `-e` run unpacks it
as `.hints` and builds the dictionary from it plus `enwik9` (section 1). The
decompressor does not need the hints; it reads the aux mode from bit 0x40 of the
archive header.

### Recipe

Current 8-block format (`H1 H2 h3exc H4 H5 head seg clu`, fixed in the package
source):

```bash
python3 mkpkg.py <older-hint-package> dictionary/english.dic \
                 <resdir> dictionary/english.hints
# <resdir> = head.bin / seg.bin / clu.bin from the segment residual coder resid.cpp
# built-in cross-check: H3 of the old package must be exactly the band assignment
#                       of the dictionary in alphabetical order
```

`mkpkg.py` carries H1/H2/H4/H5 over **bit-identically** from the older package
(those blocks were measured to be irreducible) and replaces H3/H6 by a rule plus
residual hints; the delivered `h3exc` is 138 B for 48 exceptions.

The older 6-block format is documented by two tools in `tools/`, namely
`build_hints4.py --ref <dict>.dic --cand 50000 --out hints.bin` and
`count_words.py`, which writes the word-count table over the first 100 MB of
`enwik9` that `build_hints4.py` needs.
Both tools carry their paths as `${DATA}/…` and `${WORKDIR}/…` placeholders in
the source, to be replaced by hand; `count_words.py` writes `counts_100m.tsv`
into `s9/counts/`, `build_hints4.py` reads it from `b6hints/`.

**Coupling rule: fix the dictionary first, then build the hints.** Any dictionary
change forces a new hint package.

### Result

| | |
|---|---|
| size | 205,067 B |
| sha256 | `7c16cf40569ad5907585ae79fff48ffdf4b534c9286a3e9f9d54f29f2ac4613c` |
| md5 | `e43bae1402fddfc146612eda4d9917b7` |
| `comp_hints` | 57,504 B in `comp9` of this entry |

A compression run of this entry decodes `comp_hints` 57,504 → 205,067 B and
`comp_order` 43,780 → 115,870 B (about 35 s and 25 s).

Measured worth: the hint package in place of the dictionary block **-46,688 B in
S**; the 8-block format a further **-5,861 B in
`comp_hints`** at zero stream cost.

---

## 3. Article order

### What costs 0 B

**"0 B of side information" applies to the two *rules* below, not to the order
as a whole.** The order *is* shipped, as a residual file (`.new_article_order`,
115,870 B) in the block `comp_order` (43,780 B in this package), counting once
in S. The *step* from the base order to the redirect-front order costs 0 B:

* The rule moves the block of the 71,148 unlisted pages (71,110
  redirects + 38 excluded) to the **front** when writing `.main_reordered`.
  Implemented as a rotating index in the write loop
  (`positions[i < n_rest ? n_listed+i : i-n_rest]`) instead of a re-sort, for
  **+21 B** of code.
* Which page is a redirect is decided by a prefix test **from `enwik9`**, so the
  residual file and `comp_order` are unchanged. That is the 0 B property.
* **The decompressor does not need the order at all.** `reorder()` runs only in
  the `-e` branch, and the reverse direction is a sort by page id recovered
  from the text. The order as a rule plus residual costs the `comp_order`
  block; the *same* order **as an explicit list** costs about 161 KB against a
  residual block of about 47 KB, **+114,686 B in S** (measured with an earlier
  residual).

The second rule is the **calendar block**. The 3,417 pages whose title is a
year, decade, century, millennium, day of the year, month, month and year, or
"Deaths/Births in <year>" leave the ranked part of the order and are written
as one block, sorted by title (years, decades, centuries and millennia
chronologically with BC first, then the days from 1 January to 31 December,
then months, month-year titles and the "Deaths/Births in" pages). The set and
the inner order follow from the titles alone; the residual only stores where
the block goes (one header field of 3 B). Taking those pages out of the ranked
part makes the residual 3,549 B smaller.

The residual itself is coded with a retuned rule (same order, cheaper code),
with a lead window of 600 instead of 4,000 characters for the content words, a
cap on the document frequency of a context word, a fill normalisation of the
scores and a bonus for the clusters of the two previous pages. A residual of this
form ends with a form tag (the symbol `0x1FFFFF`); a decoder of the other form
fails on the symbol count instead of reading a wrong order.

### Recipe

```bash
# (a) build the residual file (offline, with the residual encoder of the
#     development tree; it is NOT in the binary, but it shares extractor and
#     vote with the decoder side, so the round-trip is consistent by construction)
<residual-encoder> encode3    <.main> <base-order> new_article_order 113018
<residual-encoder> roundtrip3 <.main> <base-order> new_article_order 113018
#  -> roundtrip3 reports IDENTICAL over 172,277 positions; the residual file
#     of this entry is the 115,870 B file of the table below

# (b) the redirect-front and calendar-block rules are fixed in the package
#     source (no build option)

# (c) mandatory acceptance, before and after the run
bash check_order_run.sh pre  calendar_tune_c8 <variant-dir>
bash check_order_run.sh post calendar_tune_c8 <run-dir>
```

`n_split=113018` is **mandatory**, the base order has two blocks in the middle of
the normal part, and without the split the encoder aborts with what looks like a
data error.

The acceptance targets of this entry: residual md5 `3beb7ee1…`, `comp_order`
43,780 B, `.main_phda9prepr`
888,085,252 B, `.ready4cmix` 888,096,401 B,
**`.main_reordered` md5 `28ca879dcfdafe7833b042f58cfee517`**,
`.new_article_order` 115,870 B. The `.main_reordered` md5 is the only check that
tests the *order* itself instead of its size; it takes ~3 s on 1 GB.

### Result

| | |
|---|---|
| residual file | 115,870 B, sha256 `c9f8b1001bfbc041a087eab65e5caa609eb87bb8ecf297cfcdbcdaffa7a1dd19`, md5 `3beb7ee15ebeabdb13b2c87d2cf44919` |
| order identity | md5(`.main_reordered`) `28ca879dcfdafe7833b042f58cfee517` |
| base | the order of cmix-lex (the fx2-cmix order in the revision of the `fxcm_v26` archive, in which Kaido Orav gathered the `Wikipedia:` pages in one block at its end), 172,277 lines |
| rules | the `Wikipedia:` block moved from the end to a place behind the census and image pages and in front of the disambiguation pages (`n_split`); redirect pages in front (+21 B of code); calendar block and the retuned residual coding (about -2.1 KB of `S` together against the residual without them, package form) |

### Origin of the base order

The *base* order is the article order of **fx2-cmix** (Kaido Orav and Byron
Knoll), inherited through cmix-lex (its `new_article_order`). **Byron Knoll**
built it, in fx2-cmix commits "article order v5" to "v7" (`0ff7a6f`, `c59d90d`,
`7c015b1`, 3 to 12 July 2024), with pipeline description and notebooks in commit
`b2b6156` (16 July 2024). The steps of the pipeline are articles split at
`<page>`, templates stripped, title plus the first 4,000 characters, a
**commercial embedding API** (`voyage-large-2-instruct`), 1-D t-SNE, k-means on
the 1-D coordinate, reverse and manual sort. The file of cmix-lex comes from **Kaido Orav's** `fxcm_v26`
archive (cmix-lex README, "uses the article-order file included with the
`fxcm_v26` archive") and is a revision of that order, with 99.7 % of its
neighbouring pairs also neighbours in the fx2-cmix order. The
4,000-character lead window of the residual coder's extractor in earlier
generations followed that pipeline; this entry uses 600 characters. The order
was produced offline and only `comp_order` counts in S; for reproduction,
**the order file is an artefact; there is no recipe for it.**

---

## 4. PGO profile

### The shipped profile

`pgo_data/default.profdata`, 189,056 B, sha256
`cc5733f77664c3214773846ce1d3fc7808953ad355cfde984a3781c697931ac6`. It is the
collected profile (189,296 B, md5 `6671d8d1bae63893ff8b253abd8fea2d`, sha256
`93e87692209560cb7e2dfe60beddaa604805b5040b962bcba815547578d97b88`) with the
function names of 212 of its 808 records rewritten to the names in this
package; the counts are unchanged. `build.sh` reuses it and runs no profiling
pass.

### Recipe

The profile was collected with the instrumented build of the submitted model
configuration (development tree commit `99724e79`, clang 18,
`make prof_gen` with `-DSEED=923 -DUPDATE_LIMIT=3000`), with one change. The
pool of the PPMd paging was limited to 384 MiB instead of 4,096 MiB, so that the
paging code is exercised on a short input. This change leaves every function
hash of the profile as it is (all 808 compared). The weights of the correction
model during profiling were random weights of the same shape; the profile is
valid for the shipped weights, the compression results of the profiling runs
are not.

```bash
# bed: the first 50,000,065 B of the preprocessed stream (.ready4cmix of a "-e"
# run of this entry, 888,096,401 B), cut at a page end
./cmix -C dictionary/english.dic dx.arch           # the dictionary compression of "-e"
./cmix -c dictionary/english.dic bed50.bin c50.arch # compression of the bed
./cmix -d c50.arch d50.out                          # decompression, byte-exact
llvm-profdata-18 merge -output=default.profdata <the three .profraw>   # weight 1 each
```

`-e` itself cannot be profiled on an excerpt, because its preprocessing cuts at fixed
line numbers, the reorder step walks all articles of the order, and the
metadata unit expects all 243,425 records. `-c` runs the same compression call
as `-e` (preprocessing, dictionary pre-training, metadata form, coder), so the
coder sees the same stream. The coder stream of the bed is a byte-exact prefix
of the full coder stream except for the length field in bytes 1 to 5.

| | |
|---|---|
| bed | 50,000,065 B, sha256 `25a54b414008c0a5cd4b4fc66811c08b723126175be1a3b90882db1437b849f4`: metadata block, the 71,148 redirect pages, then 5,832 articles (76,980 pages) |
| coverage | 599 functions with counts, 209 without (the `-e` prelude: census, split, comment index, reorder, dictionary derivation, and paths outside the runs) |
| runtime | about 4.5 h for each of the compression and the decompression |

On the profile:

* **A PGO profile is an output-determining build input, like the dictionary.**
  Two *fresh* profiles produce the same archive; the *stale* cached profile
  produces a different one. Profiles must therefore be pinned, not "whatever is
  in the cache".
* **The profile must be collected with the configuration being measured.** A
  profile collected with a component switched off leaves that code cold.

---

## 5. What can be rebuilt from `enwik9` alone

| artefact | rebuildable from `enwik9`? |
|---|---|
| dictionary | **yes, bit-exactly**, but only with the hint package as input; the hint package itself is a file |
| hint package | **no.** The construction is documented (`mkpkg.py` + residual coder), but it consumes an older hint package and a word-count table that are themselves artefacts |
| article order (residual file) | **no.** The residual coder is reproducible; the *base order* it encodes is the fx2-cmix order (Byron Knoll's pipeline with a commercial embedding API, in the revision of Kaido Orav's `fxcm_v26` archive) and is an artefact |
| PGO profile | **no**, in practice: the recipe and the bed are documented (the bed is a prefix of the preprocessed stream of a `-e` run), but the instrumented runs need the development tree, and profiles of different compiler builds differ |

All four files are shipped with the entry. They were checksum-bound in the
development build; here the published `comp9` sha256 covers them. They are
legitimate under the rules (offline-produced inputs whose bytes count in S);
what they are *not* is regenerable byte-for-byte by a third party.

## 6. Not in this package

* The tools that produced the dictionary and the current hint package
  (`make_w4.py`, `ppmi_sparse.py`, `order_b3.py`, `mkpkg.py`, the segment
  residual coder `resid.cpp` and its outputs `head/seg/clu.bin`);
  `tools/build_hints4.py` covers only the older 6-block hint format.
* The order residual encoder, and the tool that tuned the residual coding.
* The 50 MB bed of the shipped profile (a prefix of the `.ready4cmix` of a
  `-e` run, see section 4) and the development tree it was collected with.

`tools/check_order_run.sh` knows the order of this entry as `calendar_tune_c8`
and checks `comp_order` within its tolerance of 300 B.
