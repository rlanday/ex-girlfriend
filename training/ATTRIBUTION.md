# Attribution and licences

This `training/` directory is distributed under the same terms as the rest of
the package, **GPL-3** and not "or later", because no inherited component
grants the later option for the work as a whole (the licence section of
`README.md` gives the reasoning). The full licence text is in `LICENSE` at the
top level of this package. Exceptions are the third-party files in
`tf/tree_upstream/src/ds/`, the altered `avx_mathfun.h` routines in
`tf/tree_upstream/cpp_infer/` and in `bitl/trainer/bit_head.cpp` (section 2,
`third_party/README.md`) and the Wikipedia excerpts in
`tf/tree_upstream/prof_input/` (section 5).

This compressor is the latest link in a long chain of GPL-3 work, and almost
every component below has an author other than me. The training-side Python in
this directory imports and subclasses upstream code directly, so `training/` is
itself a **derivative work** of fx2-cmix-transformer; it is no independent tool
that happens to sit next to it.

Where a file in this package is a fork of an upstream file, a provenance header
or docstring says so; the export tools in `tf/4_export/` also carry an SPDX
line.

---

## 1. The descent of the compressor

| component | author | licence | origin | where | changes in cmix-neif |
|---|---|---|---|---|---|
| **cmix**, predictor stack: coder, contexts, states, mixer, the bracket / byte / direct / indirect / match models | Byron Knoll | GPL-3 | `https://github.com/byronknoll/cmix` | the body of `src/` | model and mixer layer substantially extended; geometry, memory policy |
| **fxcm / fxcm_v26**, text model | Kaido Orav; its integration into the cmix predictor for cmix-lex Ibrahim Marcouch | GPL-2-or-later (header), therefore GPL-3 here | `https://github.com/kaitz/fxcm`; **v26 itself is unpublished** and reached cmix-neif through cmix-lex | `src/models/fxcmv1.{cpp,h}` | state-table constants and curve form, compact buckets and smaller maps, number, date, link and UTF-8 contexts, extra contexts of the first APM stage, deeper recency protection in the buckets, prefetch, the byte-model feed, page-reset handling, and the items from other entries listed in section 3 |
| **cmix-hp** | Byron Knoll | GPL-3 | `https://github.com/byronknoll/cmix-hp` (a fork of STARLIT) | inherited via fast-cmix | - |
| **fast-cmix**, a faster implementation of cmix-hp | Saurabh Kumar | GPL-3 | `https://github.com/saurabhk/fast-cmix`, commit `b63c1cfb634c` (2023; Hutter Prize 2023) | inherited via fx-cmix, fx2-cmix and cmix-lex; the source of the hash maps and of `src/ds/` (the PyTorch v2.0.0 `c10` forms of `SmallVector` and `AlignOf`) | - |
| **fx-cmix** | Kaido Orav | GPL-3 | `https://github.com/kaitz/fx-cmix` (2024 Hutter Prize winner) | inherited via cmix-lex | - |
| **fx2-cmix** | Kaido Orav, Byron Knoll | GPL-3 | `https://github.com/kaitz/fx2-cmix`, commit `04c5806f99b9b0fa8572be8c8063b4324ec405de` (2024-10-11; Hutter Prize 2024) | `src/ds/`, `readalike_prepr/`, build form | reference basis |
| **cmix-lex**, **the immediate fork base**, derived from fx2-cmix (Kaido Orav, Byron Knoll) | Ibrahim Marcouch | GPL-3 | `https://github.com/blahem/cmix-lex`, commit `370e698f7ea62168cc64326ff97950c3dc212691`; Hutter Prize 2026, awarded to Ibrahim Marcouch and Kaido Orav | `src/r1_reorder_transform.*` and `src/r1_reorder_impl.inc` (its tail transform; in the package, not called), payload layout | consolidation into the current tree; the disk-backed PPMd mapping it carried (from fx2-cmix) is replaced by the paging of this entry |
| **fx2-cmix-transformer**, the transformer inference engine and **the entire training pipeline this directory documents** | Vladimir Ivanov (GitHub `astOwOlfo`) | GPL-3 | see the note in section 7. The source is public as `github.com/astOwOlfo/fx2-cmix-transformer-v1` (GPL-3.0, since 2 September 2026); its commit `83f2603f` holds the 2026-08-21 revision byte for byte (all 189 source files; the repository adds model checkpoints). The LTCB entry for fx2-cmix-transformer links the distributed source. The inference engine is ported from the 2026-08-21 revision of `cpp_infer/` (with his exact-division fix in `opt/vec_math.h`); the training tree is the 2026-07-24 release. The recomputation of the rotary tables in his container code is his bit-exact port of `__nv_sinf` and `__nv_cosf` of NVIDIA's CUDA libdevice (section 7), and his range coder follows the form of the LZMA range coder (section 2) | 25 files in `src/tf1/`, each with a provenance banner, and, in the coder-side driver `tf1_engine.{h,cpp}` written after his `src/predictor.{h,cpp}`, his float16 helpers (verbatim), error helper, float16 file writer, loss report and the per-byte step around the article separator (separator window, cut into pieces, PPMd distribution on the first token of a piece, float16 rounding and floor of each row), beside 10 further files of that directory that are this entry's own; `pysrc/` and `training_recipes/` in the training tree | vocabulary 205 → 208 (25 rewrites in 7 files); layer count and attention window as build constants; the side-channel inputs (two preceding tokens, match byte and length class, sentinel rows); an integer kernel for the attention values with a pair-interleaved value cache; a last layer that computes only K and V for rows whose output is not read; the MLP width per layer, for the removal of dead neurons; a read-only view of the final hidden state for BITLSTM; unused pool tails given back; the decoder of container formats v3 and v4; provenance banners. **His weights, his article-separator constant and his preprocessor are NOT used.** |
| **STARLIT**, the idea of reordering articles and the code that applies it | Artemiy Margaritov | GPL-3 | `https://github.com/amargaritov/starlit`, commit `14d68047` | inherited via fx2-cmix and cmix-lex: `src/readalike_prepr/article_reorder.h` (reordering, sort back by page id), `misc.h` (input split), `self_extract.h` (dictionary and order embedded in the executable), and the merge of the phda9 transforms into `phda9_preprocess.h` | a new order encoding, the redirect-front and calendar-block rules, a flat line store |
| **the base article order**, the article order of fx2-cmix (title and first 4,000 characters of each article, embedded with a commercial embedding service, arranged on one axis by t-SNE and clustered by k-means) | **Byron Knoll** (the embedding pipeline and the order: fx2-cmix commits `0ff7a6f`, `c59d90d`, `7c015b1`, "article order v5" to "v7", 3 to 12 July 2024; description and notebooks in `b2b6156`, 16 July 2024); **Kaido Orav** (the revision in his `fxcm_v26` archive, which cmix-lex inherited) | GPL-3 | `https://github.com/kaitz/fx2-cmix`, `src/article_order/` and `src/readalike_prepr/data/new_article_order`; the file `new_article_order` of cmix-lex, which its README describes as "the article-order file included with the `fxcm_v26` archive"; 99.7 % of its neighbouring pairs are neighbours in the fx2-cmix order as well | its modification is encoded in `src/readalike_prepr/data/new_article_order` | modified order (the `Wikipedia:` block of the base order moved from its end to a place behind the census and image pages and in front of the disambiguation pages; the redirect pages in front; the pages with a calendar title as one block), encoded as rules plus a residual; the rule recognises the page classes of the manual sort step (`src/article_order/manual-sort.ipynb`) with its strings, transcribed to C++ in `src/readalike_prepr/order_codec.h` |
| **phda9**, the Wikipedia transforms of the 2017 entry, including the metadata block | Alexander Rhatushnyak | GPL-3 as distributed in STARLIT (no licence notice of its own) | 2017 Hutter Prize winner; the open-sourced transforms were merged into one file by STARLIT and rewritten as a single pass in fx2-cmix; inherited via cmix-lex | `src/readalike_prepr/phda9_preprocess.h` | metadata block at the head with the field forms of README section 4.9, page form, `&sup2;` as a control byte, language links folded in, comment index, census transform |
| **paq8l / paq8hp12any / paq8px**, WRT, state tables | Matt Mahoney, Alexander Rhatushnyak, Przemysław Skibiński and others; in paq8px the English stemmer by Márcio Pais and the match model by Márcio Pais and Zoltán Gotthardt | GPL | inherited via the cmix line | `src/preprocess/`, `src/states/`; in `src/models/fxcmv1.cpp` the stemmer and the second match model (based on paq8px v208) | dictionary bands, frozen byte set |

**Note on four inherited files.** `phda9_preprocess.h`, `article_reorder.h`,
`self_extract.h` and `misc.h` carry **no copyright header**, because none was
present in the inherited code. I have not added one (I cannot speak for their
authors) and have not removed anything. `article_reorder.h`, `misc.h` and
`self_extract.h` begin with a line that names STARLIT as their origin; the
provenance of `phda9_preprocess.h` is recorded here.

**Name spellings.** Spellings vary across sources (Rhatushnyak, the spelling of
the Hutter Prize and, since 2012, of the Large Text Compression Benchmark;
Ratushnyak in older paq8 headers; L. Lakhani / Layth Lakhani / IntrepidEnki;
ktprime / Huang Yuanbing; Vladimir Ivanov (some third-party notices write
Vladimer), GitHub `astOwOlfo`). This document uses one spelling per person and
notes the handle where the handle is how the work is found.

## 2. Embedded third-party libraries and other borrowed code (C++)

The libraries are inherited from fx2-cmix (which has them from fast-cmix; the
`c10` forms are those of PyTorch v2.0.0) with unchanged licence notices; the
rows after them name code and methods from other sources. Modified here are
`SmallVector.h` (see the notice at its top) and `emhash_map.hpp` (its
allocation calls go through two macros that resolve to `malloc` and `free` in
this build, and one include); `emhash_set.hpp` differs by one trailing blank.
Every one of these licences is GPL-3 compatible.

| component | author | licence | where |
|---|---|---|---|
| `emhash6::HashMap` v1.7.1 | Huang Yuanbing (`ktprime`) | **MIT** | `src/ds/emhash_map.hpp` |
| `emhash` set v1.4.3 | Huang Yuanbing | **MIT** | `src/ds/emhash_set.hpp` |
| LLVM `SmallVector` (via PyTorch `c10`) | LLVM Project / Meta | **Apache-2.0 WITH LLVM-exception**; the PyTorch changes **BSD-3-Clause** | `src/ds/SmallVector.{h,hpp}` |
| LLVM `AlignOf.h` (via PyTorch `c10`) | LLVM Project / Meta | **NCSA / University of Illinois OSL**; the PyTorch changes **BSD-3-Clause** | `src/ds/AlignOf.h` |
| PPMd variant H / `mod_ppmd_v2` | Dmitry Shkarin (PPMd), Eugene Shelwien (mod_ppmd) | PPMd: public domain / GPL as distributed; `mod_ppmd`: GPL | `src/models/ppmd.{cpp,h}` |
| SSE stage | Eugene Shelwien | GPL | `src/mixer/sse.{cpp,h}` |
| form of the adaptive binary range coder (11-bit probabilities, shift 5) | Igor Pavlov (LZMA SDK) | **Public Domain** | Ivanov's container code, `src/tf1/weights_io_compressed.cpp` and `tf/4_export/weights_compress_v3.py` (his code in that form) |
| squash function and table (33 points), inverse stretch table, binary arithmetic coder, APM form | Matt Mahoney (`paq7`) | **GPL** | executable packer stub, `tools_src/packer/cm_core.h`, `tools_src/packer/boot_cm.h` |
| the idea of the x86 BCJ filter (relative call and jump targets made absolute before coding), extended here to an exact address transform | Igor Pavlov (LZMA SDK) | **Public Domain**; only the idea is used | executable packer stub, `tools_src/packer/x86t.h` |
| the principle of a context-mixing decoder in a small executable stub (Crinkler) | Rune L. H. Stubbe, Aske Simon Christensen | only the idea is used | executable packer stub, `tools_src/packer/` |
| `days_from_civil` / `civil_from_days` | Howard Hinnant | **Public Domain** | `src/timestamp_algebra.h` |
| magic-number division | T. Granlund and P. L. Montgomery (1994), in the form of `magicu` from Henry S. Warren, Jr., *Hacker's Delight*, section 10-9 | method from the literature; the code of the book's website may be used freely | `src/fast_div.h` |
| one xor-shift and multiply step of the MurmurHash3 finalizer (its multiplier `0x85ebca6b`) | Austin Appleby | **Public Domain** | `src/context_map_compact.h` (second stage of the check-value hash) |
| xxHash32 primes and avalanche step | Yann Collet | **BSD-2-Clause** (text in `third_party/xxHash-LICENSE.txt`, xxHash v0.8.4, commit `c87183a7`) | called: the primes and the first three avalanche steps in `cm_hashn`, the context hash of the packer (`tools_src/packer/cm_core.h`), and the primes with the first avalanche steps in `BigCol` and `TriCol` of `src/lstm_aux_contexts.h`, the pair and triple hashes of the small byte LSTM (called from `src/mixer/lstm.hpp`); defined but not called: the full avalanche step `XXH32_avalanche` as `Mix32` in `src/invariants.h` and two primes in `UniByte` and `UniProb` of `src/lstm_fx_link.h` |
| polynomial forms of `expf`, `logf` and `tanhf` | Stephen L. Moshier (Cephes Mathematical Library) | free use, as stated by the Cephes distribution | `src/tf1/opt/vec_math.h` and `src/tf1/opt/kda_math.h` (Ivanov's engine), `src/bit_head.cpp` and `bitl/trainer/bit_head.cpp` (`Exp8`, through `cmix-obias`), `src/fast_math_vec.cpp` (`Tanh8`); `Exp8`, `exp256_ps`, `log256_ps` and `exp_full8` are altered versions of the avx_mathfun routines of the next row |
| `exp256_ps` and `log256_ps` of `avx_mathfun.h`, based on `sse_mathfun.h` | Giovanni Garberoglio; Julien Pommier | **zlib** (notices in `third_party/mathfun-LICENSE.txt`) | altered versions: `src/bit_head.cpp` and `bitl/trainer/bit_head.cpp` (`Exp8`, through `cmix-obias`), `src/tf1/opt/vec_math.h` (`exp256_ps`, `log256_ps`) and `src/tf1/opt/kda_math.h` (`exp_full8`, through `fx2-cmix-transformer`); upstream copies in `tf/tree_upstream/cpp_infer/` (`src/opt/vec_math.h`, `src/opt/kda_math.h`, `bench/misc.cpp`) |
| the FNV-1a hash scheme with the FNV primes | Glenn Fowler, Landon Curt Noll, Kiem-Phong Vo | **Public Domain** | `src/alias_table.h`, `src/string_pool.h`, `src/metablock.h`, `src/readalike_prepr/order_codec.h`, `src/readalike_prepr/phda9_preprocess.h`, `src/preprocess/dict_derive.cpp`, `src/preprocess/dictionary.h`, `src/ppmd_prewarm.h`, `src/tf1/tf1_engine.cpp`, `src/tf1/tf1_z3.cpp`, `tf/tree_overlay/zctx/z3_extract.py` |
| splitmix64 | Sebastiano Vigna (after G. L. Steele, D. Lea, C. H. Flood) | **Public Domain** | `src/bit_head.cpp` (`Rng`) and `bitl/trainer/bit_head.cpp` (random projections and initial weights of BITLSTM); two of its constants in `src/word_label.cpp` (through `zmix`) |
| the split of ln 2 (`kLn2Hi`, `kLn2Lo` = `ln2HI`, `ln2LO`), 1/ln 2 (`kLog2e` = `invln2`) and ln 2 (`kLn2` = `lg2`), values and decimal spelling | Sun Microsystems (fdlibm, `e_exp.c`, `e_pow.c`) | numerical constants only; no fdlibm code is used (the `exp`, `log` and `tanh` of the file are series of this entry), so the fdlibm notice is not reproduced | `src/detmath.cpp` |

**MIT, Apache-2.0-with-LLVM-exception, BSD-3-Clause, BSD-2-Clause, NCSA and
zlib require the licence text or notice to travel with the code, in addition to
the header.** Those texts ship in the `third_party/` directory of this package
(`emhash-LICENSE.txt`, `LLVM-LICENSE.txt`, `PyTorch-LICENSE.txt`,
`mathfun-LICENSE.txt`, `xxHash-LICENSE.txt`).

## 3. Harvested from sibling GPL-3 forks

| component | author | licence | origin | where | kind |
|---|---|---|---|---|---|
| **the BITLSTM, the LSTM correction model behind the mixer** (name and design from his `bitlstm32`): the AVX2 kernels (`Exp8`, `Sigmoid8`, `Tanh8`, `DotPadded`), the model's form (in-projection → SiLU, computed as `z * Sigmoid8(z)` → LSTM cell → linear, correcting additively in logit space), several of its input features, and the approach of training it offline on a trace of a compression run and shipping the frozen weights | **David Freelan** | GPL-3 | `https://huggingface.co/dfreelan/cmix-obias`, revision `51488a0c1228dbeab7c1be837fc90ceaed351728`; inner source snapshot `60c499e28702f707b8935087c2fb43890e9e09bd` | `src/bit_head.cpp`; the hand-over of the stage-1 row, the raw final-mixer output and the override flag in `src/predictor.{h,cpp}` and `src/coder/{encoder,decoder}.cpp`; the trainer in `bitl/trainer/` | **code and ideas**. `Exp8` is an altered version of `exp256_ps` of `avx_mathfun.h` (section 2) over the Cephes `expf` polynomial. New in cmix-neif are the training step (backward pass, optimizer) that the replay trainer runs, the trace and replay tooling, the input selection for this configuration and the training recipe. |
| **the PPMd prior on the output logits of the byte LSTM** (`g · ln p_PPMd` added before the softmax, so that the LSTM learns a correction to the PPMd prediction) | **David Freelan** | GPL-3 | as above, `src/models/obias-prior.*` there, where the weight is learned or a constant | `src/ppm_logit_prior.h`, called from `src/predictor.cpp`, hook in `src/mixer/byte-mixer.{h,cpp}` and `src/mixer/lstm.h`, applied in `src/mixer/lstm.hpp` | **code and idea**: the bias row and the output-bias hook (`SetOutputBias`, applied before the softmax, PPMd input kept) follow his constant-weight build; here a fixed factor of 0.55. Active in the submitted build. |
| the branch-free evaluation of the context-map states (zero-state and run-length masking, a presence mask for the duplicate check; value-identical) | David Freelan | GPL-3 | his speed work on `cmix-obias` (`src/models/fxcmv1.cpp` there) | `src/models/fxcmv1.cpp` (`ContextMap3/4::mix3`) | **code** (the masking lines of `ContextMap3/4::mix3`) and **idea** (run-length masking); the presence-mask duplicate check is new in cmix-neif |
| **GrammarMatch**, a second match channel | **Halvor Yttredal** | GPL-3 | `https://mattmahoney.net/dc/fx-deepmix-source.zip`, sha256 `10a50f65bbe30ab1f8fcdafca3338bccb624a50d7f8a0bff1fc9548e52bed295` | `src/models/grammar-match.{h,cpp}` | **line-by-line port** |
| the feed of the byte-mixer prediction into the internal mixers of the text model | Halvor Yttredal (form); original wiring Kaido Orav | GPL-3 | as above | `src/models/fxcmv1.cpp`, described in `src/lstm_fx_link.h` | two lines in the form of his code; the wiring is Orav's |
| the final stage 0.1 of the learning-rate schedule of the cmix mixers (stock value 0.2) | Halvor Yttredal | GPL-3 | as above, `MIXER_DECAY_FINAL` in `src/mixer/mixer.cpp` there, set to 0.1 in its build script `b72_200_p3.sh` | `src/mixer_decay.h` | **parameter**, measured here |
| index retyping of several hash tables and a larger shared map | **Layth Lakhani** (`IntrepidEnki`) | GPL-3; his changes GPL-3.0-or-later | `https://github.com/IntrepidEnki/fx3-cmix-submission`, release `v1.0` | `src/context-manager.{cpp,h}` | idea; the code and the map size (500,000 rows) are new in cmix-neif |
| the question of the widths of the direct state maps of the text model (made smaller there; the larger widths here were measured for this entry) | **Seth Lupo** | GPL-3 | `https://github.com/Seth-Lupo/fx3-cmix` (linked on encode.su) | `src/models/fxcmv1.cpp` | **idea**; the values and the measurement are those of cmix-neif |
| an SSE2 search for check values in the context maps (`simdscan::FindChk`); the reuse of the last context lookup in the mixer (`Mixer::GetContextData`); header inlining of `Sigmoid::Logit` and `MixerInput::Set*` | **Rohit Dwivedula** | GPL-3 | `https://github.com/rohitdwivedula/fx2-cmix`, commit `468e92b9649eaa54b42364a6ce4d41c862cd9173` (2026-06-19) | `src/simd_scan.h` (called from `src/models/fxcmv1.cpp`; the same idea in `src/context_map_compact.h`), `src/mixer/mixer.{h,cpp}`, `src/mixer/sigmoid.h`, `src/mixer/mixer-input.h` | search and reuse: rewritten and measured (the compare of the search in the form of that commit, extended to 16 slots; the reuse in the structure of that commit, with names and validity check of this entry); `Sigmoid::Logit` inline with a body of this entry; the four inline `MixerInput::Set*` **taken over** in the form of the commit (the `fx2-cmix` bodies moved into the header) |
| the stretch curve ln(p^k / (1 - p^z)), here with k = 0.75 and mirrored at p = 0.5, in the stretch table of the context maps | the encode.su contributor **mpais** | - | encode.su thread 4534, "Research ideas", post 88828 (26 August 2026) | `src/models/fxcmv1.cpp` (`mpais_sym`) | **idea**, reimplemented and measured here |
| the adaptive-depth stage (an online logistic regression in front of the mixer cascade that skips the final mixers, the SSE stage and the BITLSTM on confident bits); the end-to-end gradient of the mixer cascade; the log-bilinear word-label model with a hierarchical softmax over the code-word tree (after Mnih and Hinton 2007 and Morin and Bengio 2005, as cited there); the indel recovery of the match model; a context-map time constant (18); the flag history of the secondary estimation stage at 4 bits; a narrower flag field in its second stage (his axis, the width of this entry); the first mixing stage of the check-sum hash of the compact context maps | **James Byrne** | GPL-3-or-later | `zmix 1.0`, submitted 14 September 2026 (source `zmix.frostbyrne.io/v1.0/zmix-src.tar.gz`, description `zmix.frostbyrne.io/v1.0/README.md`); the build options named in the banners | `src/adaptive_depth.{h,cpp}`, `src/mixer_chain.h` with the backward recursion in `src/predictor.cpp` and the start row in `src/mixer/mixer.cpp`, `src/word_label.{h,cpp}`, `src/mixer/sse.cpp`, `src/models/fxcmv1.cpp` and the match and context-map code named in the banners | adaptive depth: **mechanism**, rewritten in C++ from his Zig source (its state structure follows that source) and adapted to this entry's mixer row; the end-to-end gradient: **mechanism**, its backward recursion rewritten line by line in C++; the word-label model (function by function) and the indel recovery (line by line): **ported** from the Zig source, the word-label model extended to this entry's fourth code-word band; the constants: **parameters**. All active in the submitted build, each item with a banner in the file that defines it. |
| the block-diagonal recurrence of the small byte LSTM (each cell reads only the previous state of its own block of 16 cells; the other blocks stay zero) | **Xeloraa** | - | `https://github.com/Xeloraa/Hutter-compression-research` (`lstm-layer-bd.cpp`, research log entries E27 and E28) | `src/mixer/lstm-layer.{h,hpp}`, `src/lstm_blockdiag.h` | **idea / form**, implemented independently and measured |

## 4. The training side: what is in this directory

In `training/` the code of cmix-neif and upstream code sit side by side.

| component | author | licence | where | changes in cmix-neif |
|---|---|---|---|---|
| **The whole upstream training tree**, the model (`model.py`), trainer, data loader, optimizer, quantization, weight export, container formats v1 and v2, logging, attention and KDA reference implementations; plus `training_recipes/`, `cpp_infer/`, `src/`, `dictionary/`, `prof_input/` and the dependency lock file | **Vladimir Ivanov** | GPL-3 | **`tf/tree_upstream/`, shipped verbatim in this package**, 189 files, about 4.3 MB, `LICENSE` and every copyright header untouched. Snapshot 2026-07-24. | **none in the tree itself.** The extensions of cmix-neif live in `tf/tree_overlay/` and subclass or wrap the upstream modules. No upstream file is written, because `tf/2_package/patch_paket11.py` applies 19 idempotent, individually marked edits to the overlay files (`zctx/`, `tft/`) and copies three helpers into the working tree; the stage-1 loader is not touched at all. |
| **`tf/4_export/weights_compress_v3.py`** | Ivanov (base) + cmix-neif (formats v3 and v4) | GPL-3 | this package | a fork of his `pysrc/weights_compress.py`. Container formats v1 and v2, the range coder, the bit trees and the RoPE recomputation are his; new in cmix-neif are the added formats v3 (counting trees) and v4 (counting trees on the low byte planes, a flag per all-zero int4 row, names and metadata from a template) and `zero_collapsed_rows`. The shipped container is format v4 (`FX2TFWC4`). The file carries an SPDX line and a provenance header. |
| **`tf/4_export/wpack2_bf16.py`** | cmix-neif | GPL-3 | this package | a tool of cmix-neif, but it reads and writes his reference weight format and imports his `pysrc.export_weights`. Header added. |
| **`tf/4_export/prune_dead_neurons.py`**, **`convert_container_v4.py`**, `export.sh` | cmix-neif | GPL-3 | this package | tools of cmix-neif. They read and write his container format through `weights_compress_v3.py`; the deadness test evaluates the closed form of his quantization arithmetic (`cpp_infer/src/opt/qmat.h`, `qgemv_relu2q` in `cpp_infer/src/opt/qmat_dense.cpp`). Provenance headers and SPDX lines. |
| **`tf/tree_overlay/tft/`**, geometry registry, dataset checks, runner, PoC entry points, export with manifest | cmix-neif, derived from `training_recipes/*` and `pysrc/{data,model,export_weights}.py` | GPL-3 | this package | provenance is stated in each module's docstring |
| **`tf/tree_overlay/zctx/`**, the training form: gate zones, prompts, sentinels, the predecessor and match channels, the int64 packing, the loaders | cmix-neif, derived from `pysrc/data.py` and `pysrc/model.py` | GPL-3 | this package | provenance stated in each module's docstring |
| **`tf/tree_overlay/run/`**, shell drivers of the GPU host (environment set-up, smoke test, night run, export chain) | cmix-neif | GPL-3 | this package | new |
| **`tf/2_package/`**, dump reader, package builder, dump loader, the steps of the package build, acceptance tests | cmix-neif; `zpdump.py` derived from `pysrc/data.py` | GPL-3 | this package | new, except `zpdump.py`, `dumpv2.py` and `warmstart.py`, which are copies of the overlay modules `zctx/zpdump.py`, `zctx/dumpv2.py` and `paket11_warmstart.py` |
| **`tf/1_dump/`**, **`tf/3_train/`**: the patch with the coder-side modes (stream, dump, priors) and its build script, the build lines of the original coder-side runs, the run scripts of the training chain, the repair of subnormal rows | cmix-neif | GPL-3 | this package | new |
| **`bitl/`**: the replay trainer of the BITLSTM and its tools (trace cut, training driver, bfloat16 conversion) | cmix-neif; the AVX2 kernels and the model form in the trainer sources are David Freelan's (section 3) | GPL-3 | this package | the trainer sources are the correction-model code of the compressor plus a replay driver, reduced to the trainer's configuration |

### 4.1 Runtime dependencies of the trainer

These are **installed, not redistributed**; naming them is sufficient. Versions
are those pinned by the upstream lock file.

| library | version | used for | licence |
|---|---|---|---|
| `torch` | 2.12.1 | model, autograd, distributed training | BSD-3-Clause |
| `numpy` | 2.4.6 | container I/O, requantization | BSD-3-Clause |
| `flash-linear-attention` (`fla`) | 0.5.1 | KDA kernels, fused gated RMS norm, causal conv | MIT |
| `pytorch-optimizer` | 3.10.1 | the SOAP optimizer (imported by the upstream optimizer module; not used for the shipped weights) | Apache-2.0 |
| `jaxtyping` | 0.3.11 | tensor type annotations | MIT |
| `tqdm` | 4.68.4 | progress display | MPL-2.0 and MIT (dual) |
| `python-dotenv` | 1.2.2 | recipe environment files | BSD-3-Clause |
| `wandb` | 0.28.0 | optional logging sink | MIT |
| `flash-attn` | 2.8.3 | optional attention path (guarded import); the training hosts installed a prebuilt wheel for the CUDA 13 / torch 2.12 combination | BSD-3-Clause |
| `triton`, `einops`, `safetensors`, `transformers`, `huggingface-hub` | transitive | - | MIT / MIT / Apache-2.0 / Apache-2.0 / Apache-2.0 |

The licence texts of these packages are **not** included here, because the
packages are not shipped with this entry; the licences in the table are as
stated in their package metadata. `tqdm` is dual-licensed, and the split is
stated only in its own licence file.

**No pre-trained third-party model and no third-party tokenizer is used
anywhere in the transformer chain.** The vocabulary is byte-based (208 values
derived from the preprocessing of cmix-neif); nothing is loaded from a model
hub. The ordering of the dictionary used two pre-trained embedding models
offline (section 5).

### 4.2 Ideas the upstream trainer itself builds on

Cited for completeness of the chain. These papers and public techniques
contributed no copied code; they are modded-nanogpt (Keller Jordan, MIT), the Muon optimizer (Keller
Jordan and co-authors, 2024; the shipped weights were trained with
`torch.optim.Muon` for the weight matrices and AdamW for the other parameters),
Kimi Linear (arXiv 2510.26692), whose kernels in `flash-linear-attention` (MIT)
the C++ KDA of the engine reproduces, rotary position embeddings
(arXiv 2104.09864), query/key normalisation (arXiv 2010.04245) and ParetoQ
(arXiv 2502.02631).

## 5. Data artefacts

| artefact | author | licence | note |
|---|---|---|---|
| the inherited word list, `tf/tree_upstream/dictionary/english.dic` (411,996 B, 44,515 words) | word set from the paq8hp / DRT line (Rhatushnyak, Mahoney, Skibiński); **ordering by Kaido Orav** (2024, word2vec) | GPL | byte-identical to the file in cmix v21, fx-cmix, fx2-cmix and cmix-lex. The common attribution of the *ordering* to Rhatushnyak is incorrect, because the macro bands are the paq8hp inheritance and the 510 fine clusters are Orav's. |
| `dictionary/english.dic` (520,804 B, 54,857 entries) | cmix-neif, on the inherited word list | GPL-3 | the inherited list with about 10,300 further entries, an extra code band and reordered bands 3 and 4; see `data/README.md` |
| `src/readalike_prepr/data/new_article_order` | cmix-neif, on the base article order of section 1 (Byron Knoll, Kaido Orav) | GPL-3 | the order residual of this entry (115,870 B); it encodes a modification of the base article order, which the file of that name in cmix-lex holds as a list |
| offline tools of the dictionary ordering, not shipped: the pre-trained embedding models `intfloat/e5-large-v2` and `BAAI/bge-reranker-base` | their authors (Hugging Face model cards) | MIT | used once, with k-means, to order bands 3 and 4 of `dictionary/english.dic` (`data/README.md` section 1); nothing of the models ships with the entry |
| `prof_input/`, the PGO training text | **Wikipedia contributors** | **GFDL / CC-BY-SA, not GPL** | an excerpt of `enwik9`, inherited from upstream; the build reads it only when no profile is shipped. The modification notice of cmix-neif does not extend to it. |
| the dictionary, hint package, order residual and PGO profile of cmix-neif | cmix-neif | GPL-3 | see `data/README.md` |
| the trained transformer weights and the BITLSTM weights | cmix-neif | GPL-3 | produced by the pipeline documented here from `enwik9` alone |

## 6. Obligations under GPL-3, and how this package meets them

1. **One copyleft chain.** fx2-cmix, cmix-lex, fx2-cmix-transformer (both
   snapshots), Rohit Dwivedula's fx2-cmix fork and fx-deepmix ship a
   byte-identical GPL-3 text (md5 `1ebbd3e34237af26da5dc08a4e440464`, the same
   as `LICENSE` of this package); cmix-obias carries the same text with different
   line endings. zmix is GPL-3-or-later, which is compatible. The single GPL-2
   header in the chain (fxcm) says "or any later version" and therefore merges
   into GPL-3.
2. **All upstream copyright and licence notices are left unchanged.** The
   redaction step that prepares the submission source removes the conditional
   branches and the internal notes of this entry; it keeps inherited copyright,
   licence and origin notes and the comments of the ported transformer files,
   and only replaces typographic dashes in comments.
3. **Modification notice** (GPL-3 §5a): `CHANGES.md` at the top level of this
   package, together with sections 3 and 4 of `README.md`, states what was
   changed relative to upstream.
4. **Third-party licence texts.** The GPL-3 text ships with the entry. The MIT,
   Apache-2.0-with-LLVM-exception, BSD-3-Clause (PyTorch), BSD-2-Clause
   (xxHash) and NCSA texts and the zlib notices of `avx_mathfun.h` and
   `sse_mathfun.h` required by section 2 ship alongside it in `third_party/`.
5. **The upstream training tree is redistributed here, verbatim.** It sits in
   `tf/tree_upstream/` with its own `LICENSE` (GPL-3, md5
   `1ebbd3e34237af26da5dc08a4e440464`) and every copyright header untouched; not
   one of its 189 files was edited to build this package. The files of cmix-neif
   sit beside it in `tf/tree_overlay/`, never merged into it, so the boundary
   between Ivanov's work and that of cmix-neif stays visible file by file. The
   upstream trained checkpoints (`models/`, 71 MB) are **not** shipped, because
   this entry does not use them and they are not needed to reproduce anything
   here.
6. **No copyright claim is made over inherited code.** The claim of
   Marcus Wölk (Marcus Woelk) covers the 2026 modifications only.

## 7. Open points

1. **No pinned version for three old ancestors.** For cmix (Knoll), cmix-hp
   and phda9 I have no tag and no commit hash, only project URLs and release
   dates.
2. **`fxcm_v26` is unpublished.** It is the actual text-model core, but there is
   no URL, no commit and no separate licence statement from its author for that
   specific code; it is citable only through cmix-lex.
3. **The files of fx2-cmix-transformer name no author.** The attribution to
   Vladimir Ivanov rests on the Hutter Prize page, which lists him as the
   winner for `fx2-cmix-T` (24 July 2026), on the LTCB entry and on the GitHub
   account `astOwOlfo` that publishes the source (GPL-3.0, since 2 September
   2026) and carries his name.
4. **The Python packages the training uses are not shipped with this entry;**
   their licences are as stated in their package metadata.
5. **Four inherited source files carry no copyright header** (section 1).
6. **NVIDIA libdevice.** `sincosf_cuda` and `trig_slowpath` in
   `src/tf1/weights_io_compressed.cpp`, and their twin in
   `tf/4_export/weights_compress_v3.py`, recompute the rotary tables of the
   container exactly as `torch.sin` and `torch.cos` produced them on CUDA. They
   are Vladimir Ivanov's transcription of `__nv_sinf` and `__nv_cosf` from the
   LLVM IR of `libdevice.10.bc` of CUDA 13.0, as his comment states, published
   by him under GPL-3.0 with his engine and taken over unchanged. Replacing
   them would change the decoded weights and the archive. Facts, as of
   2 October 2026:
   * NVIDIA distributes `libdevice.10.bc` under the CUDA Toolkit End User
     License Agreement (`docs.nvidia.com/cuda/eula/`, versions 13.0 and 13.4).
     Attachment A lists it ("NVIDIA Common Device Math Functions Library")
     among the files that may be distributed with an application, in object
     code form (section 1.1.1). Section 1.2 states: "You may not reverse
     engineer, decompile or disassemble ... any portion of the SDK" and
     "Except as expressly provided in this Agreement, you may not ... modify,
     or create derivative works of any portion of the SDK."
   * No licence of libdevice other than this agreement was found (NVIDIA's
     Python wheel `nvidia-nvvm` 13.0.48: "LicenseRef-NVIDIA-Proprietary";
     conda-forge `cuda-nvvm`: the agreement). The libdevice User's Guide
     states none.
   * The transcription contains no NVIDIA source text. The Payne-Hanek
     reduction (1983) and the Cody-Waite reduction (1980) are published
     methods; the bits of 2/pi also stand in fdlibm (`k_rem_pio2.c`), and the
     2/pi table and the three-part split of pi/2 in the header
     `math_functions.h` that NVIDIA shipped in source form with the CUDA
     Toolkit of 2012. The seven polynomial coefficients were found only in
     NVIDIA compiler output (PTX) and in other ports of libdevice.
   * Whether the agreement covers such a transcription is not settled here.
