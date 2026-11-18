# Origin and changes

This program is a derivative work and is distributed under the GNU General
Public License (see `LICENSE`).  The copyright and licence notices in the
individual source files are unchanged and remain in force.

## Descent of the compressor

| component | author | licence |
|---|---|---|
| `cmix`: coder, contexts, states, mixer, byte/direct/indirect/match models | Byron Knoll | GPL-3 |
| `cmix-hp`: changes to cmix on top of STARLIT | Byron Knoll | GPL-3 |
| `fast-cmix`: a faster implementation of `cmix-hp`, from which `fx-cmix` was developed; it brought the hash maps and the containers of `src/ds/` | Saurabh Kumar | GPL-3 |
| `fxcm` / `fxcm_v26`: text model, in `cmix-lex` integrated into the cmix predictor (`src/models/fxcmv1.cpp`) | Kaido Orav; the integration Ibrahim Marcouch | GPL-2-or-later |
| `fx-cmix` | Kaido Orav | GPL-3 |
| `fx2-cmix` | Kaido Orav, Byron Knoll | GPL-3 |
| `cmix-lex`: the fork base, derived from `fx2-cmix` (Kaido Orav, Byron Knoll), with Kaido Orav's text model `fxcm_v26`; Hutter Prize 2026, awarded to Ibrahim Marcouch and Kaido Orav | Ibrahim Marcouch | GPL-3 |
| **`cmix-neif-1.0`: the previous version of this entry**, on which this version is built (`github.com/marwoelk/cmix-neif-1.0`, commit `1c9ea5ed`); its changes against `cmix-lex` are listed in its `CHANGES.md` | Marcus Wölk | GPL-3 |
| **`cmix-neif-beehive`** (`github.com/galqiwi/cmix-neif-beehive`, commit `92c07c0b`; a fork of `cmix-neif-1.0`): the architecture and training recipe of the transformer (attention residuals, one key and value head shared by all query heads, value embeddings, MLP width 1024, prior-logit head, weight rows of 2, 3 or 4 bits, quantization-aware training with a size term as the cool-down); the engine parts that implement them, adapted (`src/tf1/opt/attnres_mix.h`, the attention-residual reads, the depth mixture and the expansion of the shared key and value head with the value embeddings in `src/tf1/opt/model_opt.cpp`, the loader parts in `src/tf1/opt/arena_build.cpp`, the kernel of the shared key and value head in `src/tf1/opt/attn.cpp`, the prior-logit head in `src/tf1/opt/glue.cpp`); the prefetch depth of the dense kernels (`src/tf1/opt/qmat_dense.cpp`); the confidence of the transformer as context of a layer-0 mixer (`src/predictor.cpp`); the training scripts on which the trainer of this entry is built (`training/tf/3_train/tree/`, the export tools in `training/tf/3_train/export/`, each file naming its source file); the weights from which the training of the shipped transformer weights started | Vladimir Malinovskii | GPL-3 |
| attention residuals (a softmax-weighted mixture of earlier sublayer outputs in place of the residual sum), *Attention Residuals*, arXiv 2603.15031, through `cmix-neif-beehive` | Kimi Team | - (a method) |
| the online normalizer of the softmax, by which the engine evaluates the depth mixture of the attention residuals (`src/tf1/opt/model_opt.cpp`, `src/tf1/opt/attnres_mix.h`), through `cmix-neif-beehive` | M. Milakov, N. Gimelshein, *Online normalizer calculation for softmax* (2018, arXiv 1805.02867) | - (a method) |
| one key and value head for all query heads (multi-query attention), through `cmix-neif-beehive` | Noam Shazeer (2019) | - (a method) |
| value embeddings (a learned embedding of the token mixed into the values of an attention layer), through `fx2-cmix-transformer` and `cmix-neif-beehive` | Braden Koszarsky, NanoGPT speedrun record of 4 December 2024 in Modded-NanoGPT (`github.com/KellerJordan/modded-nanogpt`, Keller Jordan and contributors), inspired by Zhanchao Zhou et al., *Value Residual Learning* (2024, arXiv 2410.17897) | - (a method) |
| **`ex-girlfriend 1.0`** (`github.com/rlanday/ex-girlfriend`, commit `e923faec`; a modification of `cmix-neif-1.0`): the forecast of the revision id from the page id below 17,000,000 (`src/metablock.h`, `src/metablock_form.h`, `src/id_exclude.h`), the timestamp forecast from the slope of the last records (`src/ts_progress.h`), and the fast-weight head on the MLP blocks 12 to 15 (`src/tf1/opt/model_opt.*`, `src/tf1/opt/arena_build.h`, `src/tf1/tf1_engine.*`, `src/predictor.cpp`), code adapted, each with a note naming its origin | Ryan Landay | GPL-3.0-only |
| **`fx2-cmix-transformer`: the transformer inference engine** (`github.com/astOwOlfo/fx2-cmix-transformer-v1`; 25 files under `src/tf1/`, each with a provenance banner, and, in the coder-side driver written after his `src/predictor.cpp`, his float16 helpers, error helper, float16 file writer, loss report and the per-byte step around the article separator; beside the driver, 12 further files of that directory are not his), including the layer pattern of linear-attention and windowed attention layers and the container formats v1 and v2; and the training tree on which the training of the transformer runs, shipped verbatim in `training/tf/tree_upstream/`, with the options for one key and value head, value embeddings, the MLP width and the prior-logit head | Vladimir Ivanov | GPL-3 |
| the single-precision sine and cosine of NVIDIA's CUDA libdevice (`__nv_sinf`, `__nv_cosf`), ported bit-exactly by Vladimir Ivanov in his container code to recompute the rotary tables (`src/tf1/weights_io_compressed.cpp`, `training/tf/3_train/export/export4/weights_compress_v3.py`) | NVIDIA, transcribed by Vladimir Ivanov | libdevice: CUDA Toolkit End User License Agreement, no licence of its own; the transcription contains no NVIDIA source text and was published by Ivanov under GPL-3.0; see section 7 of `training/ATTRIBUTION.md` of `cmix-neif-1.0` |
| **the base article order** that the order of this entry modifies: the article order of `fx2-cmix` (title and first 4,000 characters of each article, embedded, arranged on one axis and clustered; a manual sort step moves census, image and disambiguation pages to the end; `fx2-cmix` commits "article order v5" to "v7", July 2024), inherited through `cmix-lex` in the revision of the `fxcm_v26` archive; the order rule of this entry recognises those page classes with the strings of that manual sort step; in that revision Kaido Orav gathered the `Wikipedia:` pages in one block at the end | Byron Knoll, Kaido Orav | GPL-3 |
| the 510 fine clusters of the inherited dictionary order | Kaido Orav | GPL-3 |
| STARLIT: the idea of reordering articles, and the inherited code in `src/readalike_prepr/` that reorders the articles, sorts them back by page id, splits the input and embeds dictionary and article order in the executable | Artemiy Margaritov | GPL-3 |
| phda9: Wikipedia preprocessing and the metadata block | Alexander Rhatushnyak | GPL-3 as distributed in STARLIT (the transforms carry no licence notice of their own) |
| paq8l / paq8hp12any: WRT, state tables | Matt Mahoney, Alexander Rhatushnyak, Przemysław Skibiński and others | GPL |
| paq8px: in the text model (`src/models/fxcmv1.cpp`) the English stemmer, after the Porter2 algorithm of Martin Porter, and the second match model, based on paq8px v208 | Márcio Pais (stemmer); Márcio Pais, Zoltán Gotthardt (match model) | GPL |
| PPMd: model and allocator | Dmitry Shkarin | public domain / GPL as distributed |
| `mod_ppmd` and the SSE stage | Eugene Shelwien | GPL |
| from `fx2-cmix` (`github.com/rohitdwivedula/fx2-cmix`, commit `468e92b9`): an SSE2 search for check values in the context maps (its compare in the form of that commit, extended to 16 slots) and the reuse of the last context lookup in the mixer (in the structure of that commit, with names and a validity flag of this entry), both **rewritten** here; the header inlining of `Sigmoid::Logit` (with a body of this entry) and of `MixerInput::Set*` (**taken over** in the form of the commit) | Rohit Dwivedula | GPL-3 |
| **grammar-match model**: line-for-line port from `fx-deepmix` (`mattmahoney.net/dc/fx-deepmix-source.zip`) | Halvor Yttredal | GPL-3 |
| the feed of the byte-model prediction into the internal mixers of the text model (two lines in the form of `fx-deepmix`; the wiring is Kaido Orav's from `fx2-cmix`), which carries a constant since the byte LSTM is removed, and the value 0.1 of the last step of the mixer learning-rate schedule | Halvor Yttredal | GPL-3 |
| **BITLSTM, the LSTM correction model behind the mixer** (published in `cmix-obias`, `huggingface.co/dfreelan/cmix-obias`, revision `51488a0c`, as `bitlstm32`): its name, its form, its AVX2 kernels, several of its input features and its offline training on a trace with trained weights shipped | David Freelan | GPL-3 |
| the branch-free evaluation of the context-map states: the zero-state masking taken over in the form of `cmix-obias`, the run-length masking after it; the presence-mask duplicate check is this entry's own | David Freelan | GPL-3 |
| **from `zmix 1.0`** (`zmix.frostbyrne.io/v1.0/zmix-src.tar.gz`): adaptive depth (a logistic model in front of the mixer cascade that skips the cascade on confident bits), indel recovery in the match model, the end-to-end gradient of the mixer cascade, the word-label model with its hyper-parameters (after Mnih and Hinton 2007 and Morin and Bengio 2005, as cited there), the time constant of one context-map family, the width of one flag field of the secondary estimation and the option of narrowing a second one, and the first mixing stage of a check-value hash; rewritten in C++ for this entry (the word-label model and the backward recursion of the gradient function by function, the indel recovery line by line, from the Zig source), each item with a banner naming its origin in the file that defines it | James Byrne | GPL-3-or-later |
| narrower index types of several hash tables and a larger shared map: ideas from Layth Lakhani's experiments on `fx3-cmix` (the program is Kaido Orav's), implemented independently | Layth Lakhani | GPL-3; his changes GPL-3.0-or-later (only ideas are used) |
| the question of the widths of the direct state maps of the text model: from the public `fx3-cmix` experiments (`github.com/Seth-Lupo/fx3-cmix`), where they were made smaller; the larger widths were measured for this entry | Seth Lupo | GPL-3 (only the idea is used) |
| the stretch curve ln(p^k / (1 - p^z)), here with k = 0.75 and mirrored at p = 0.5, in the stretch table of the context maps (`mpais_sym` in `src/models/fxcmv1.cpp`), after a public encode.su post (thread 4534, "Research ideas", post 88828, 26 August 2026), reimplemented and measured | mpais | - |
| the type of the linear-attention layers of the transformer, Kimi Delta Attention from Kimi Linear (arXiv 2510.26692), used through the design of `fx2-cmix-transformer`, whose C++ form reproduces the semantics of the kernels of `flash-linear-attention` | Kimi Team; Songlin Yang, Yu Zhang, Zhiyuan Li (`flash-linear-attention`) | -; `flash-linear-attention`: MIT |
| methods of SEMCACHE (`src/semantic_cache.h`, `src/predictor.cpp`), implemented without adopted code: prediction from stored states and the tokens that followed them (the nearest-neighbour language model; the continuous cache and its unbounded form with approximate nearest-neighbour search), random-hyperplane hashing (SimHash), multi-probe hashing, AdaGrad in the end stage | U. Khandelwal et al. (2020); E. Grave, A. Joulin, N. Usunier (2017); E. Grave, M. Cissé, A. Joulin (2017); M. Charikar (2002); Q. Lv et al. (2007); J. Duchi, E. Hazan, Y. Singer (2011) | - (methods) |
| the fixed share of the Bayesian term of the final mixer (`src/final_mix.h`) | M. Herbster, M. K. Warmuth, *Tracking the best expert* (1995) | - (a method) |
| the ridge regression with which the number coder of the census transform estimates a value from the features of its record, per kind of place and field (`src/readalike_prepr/censblock_zahl.h`), implemented without adopted code | A. E. Hoerl, R. W. Kennard, *Ridge Regression: Biased Estimation for Nonorthogonal Problems*, Technometrics 12(1), 55-67 (1970) | - (a method) |
| the generator xorshift64* for the fixed random projection, the table hyperplanes and the sketch hyperplanes of SEMCACHE (`SemRng` in `src/predictor.cpp`), and its multiplier in the context hash of the first stage of the packer (`tools_src/packer/boot_cm.h`, `tools_src/packer/stub_boot.c`) | Sebastiano Vigna (xorshift64*, 2016), on George Marsaglia's xorshift (2003), with the multiplier 2685821657736338717 from the tables of Pierre L'Ecuyer (1999) | public domain |
| the constant 0x9E3779B97F4A7C15 (2^64 divided by the golden ratio) as a hash constant in SEMCACHE (`src/predictor.cpp`), the census transform (`src/readalike_prepr/censblock_zahl.h`), the numbering of repeated edit comments (`src/readalike_prepr/komm_transform.h`) and the first stage of the packer (`tools_src/packer/boot_cm.h`, `tools_src/packer/stub_boot.c`), as in multiplicative hashing with the golden ratio; the same value is the increment of splitmix64 (below) | D. E. Knuth, *The Art of Computer Programming*, volume 3, section 6.4 | - (a numerical constant) |
| the container formats NP and NP2 of the transformer weights (`src/tf1/weights_io_compressed.cpp`): the order of interchangeable groups of rows written as a permutation in the factorial number system (the Lehmer code), and the bits-back argument for the free choice of that order | C.-A. Laisant (1888); G. E. Hinton, D. van Camp (1993) | - (methods) |
| `log1p` in `src/final_mix.cpp`, after fdlibm `s_log1p.c` 5.1 with the polynomial evaluation for pipelined processors of Naohiko Shimizu (1997), implementation adapted from glibc 2.39 (`sysdeps/ieee754/dbl-64/s_log1p.c` and its x86-64 FMA variant); the notice of Sun Microsystems is kept in the source | Sun Microsystems (SunPro); Naohiko Shimizu (Tokai University); the GNU C Library contributors | fdlibm notice (kept); glibc: LGPL-2.1-or-later, distributed here under the GPL as its section 3 permits |
| `emhash` hash map / hash set | Huang Yuanbing (`ktprime`) | MIT, full text in `third_party/` |
| `SmallVector` (via PyTorch `c10`, version 2.0.0, through `fast-cmix`) | LLVM Project / Meta | Apache-2.0 WITH LLVM-exception; the PyTorch changes BSD-3-Clause; full texts in `third_party/` |
| `AlignOf` (via PyTorch `c10`, version 2.0.0, through `fast-cmix`) | LLVM Project (University of Illinois) / Meta | NCSA; the PyTorch changes BSD-3-Clause; full texts in `third_party/` |
| the form of the adaptive binary range coder of the transformer weight container (11-bit probabilities, shift 5), which Ivanov's container code follows | Igor Pavlov (LZMA SDK) | public domain |
| squash function with its 33-point table, inverse stretch table, binary arithmetic coder and APM form of the self-extractor's decoder (`paq7`) | Matt Mahoney | GPL |
| the coder of the dictionary hints and the article-order residual (`src/preprocess/hd_cm.h`, `src/preprocess/dict_derive.cpp`): squash with interpolation, stretch as its inverse, the binary arithmetic coder and the counter rate of `lpaq1`, adapted; the same squash table and stretch function in the sign rule of the census region (`src/invariants.h`) | Matt Mahoney | GPL-2.0-or-later (`lpaq1`) |
| the run map and byte history per context of the second stage of the packer (`tools_src/packer/cm_core.h`), after `ContextMap2` of paq8px v217, written anew; the index of the byte-history input in the form of paq8px | the paq8px authors (the run map as a state map: Zoltán Gotthardt) | GPL-2.0-or-later (form and index only) |
| the indirect contexts over the instruction in the second stage of the packer (`tools_src/packer/cm_core.h`, the two bytes that last followed the same role, field index, opcode and ModRM byte), like the indirect context model of paq8f, written anew | Matt Mahoney (paq8f, 2006) | - (only the idea is used) |
| the idea of the x86 BCJ filter of the LZMA SDK (relative call and jump targets made absolute before coding), which the exact address transform of the self-extractor's stub extends (`tools_src/packer/x86t.h`) | Igor Pavlov | public domain (only the idea is used) |
| a context-mixing decoder in a small executable stub, the principle of Crinkler, followed by the self-extractor's stub (`tools_src/packer/`); its first stage uses context models over byte masks of the last bytes in the form of Crinkler, like in oneKpaq (Teemu Suutari), written anew (`tools_src/packer/boot_cm.h`) | Rune L. H. Stubbe, Aske Simon Christensen (Crinkler); Teemu Suutari (oneKpaq) | - (only the idea is used) |
| civil-date conversion of the timestamps (`days_from_civil`, `civil_from_days` in `src/timestamp_algebra.h`) | Howard Hinnant | public domain |
| division by run-time constants through a magic multiplier (`src/fast_div.h`): the method of Granlund and Montgomery (1994) in the 64-bit form of `magicu` from *Hacker's Delight*, section 10-9 | T. Granlund, P. L. Montgomery; Henry S. Warren, Jr. | - (a method; the code of the book's website may be used freely) |
| one xor-shift and multiply step with the multiplier `0x85ebca6b` of the MurmurHash3 finalizer, in the check-value hash of the compact context maps (`src/context_map_compact.h`); the same multiplier in the context hash of the coder of the dictionary hints (`src/preprocess/hd_cm.h`) | Austin Appleby | public domain |
| the primes and the first three avalanche steps of xxHash32 in the context hash of the packer (`cm_hashn` in `tools_src/packer/cm_core.h`); in a helper that the program does not call, its full avalanche step (`Mix32` in `src/invariants.h`) | Yann Collet | BSD-2-Clause, text in `third_party/` (xxHash v0.8.4) |
| the split of ln 2 into `kLn2Hi` and `kLn2Lo`, and `kLog2e` and `kLn2`, in `src/detmath.cpp`: the values and decimal spelling of `ln2HI`, `ln2LO` and `invln2` of `e_exp.c` and `lg2` of `e_pow.c`; no fdlibm code is used there, the functions are series of this entry | Sun Microsystems (fdlibm) | - (numerical constants) |
| polynomial forms of `expf` and `logf`: the vector `exp` and `log` of the transformer engine (`src/tf1/opt/vec_math.h`, from `fx2-cmix-transformer`) and `Exp8` of BITLSTM (`src/bit_head.cpp`, from `cmix-obias`) | Stephen L. Moshier (Cephes Mathematical Library) | free use, as stated by the Cephes distribution |
| altered versions of `exp256_ps` and `log256_ps` of `avx_mathfun.h` (based on `sse_mathfun.h`): `Exp8` of BITLSTM (`src/bit_head.cpp`, through `cmix-obias`), `exp256_ps`, `log256_ps` and `exp_full8` of the transformer engine (`src/tf1/opt/vec_math.h`, `src/tf1/opt/kda_math.h`, through `fx2-cmix-transformer`) | Giovanni Garberoglio; Julien Pommier | zlib, notices in `third_party/` |
| the FNV-1a hash scheme with the FNV primes, in `src/alias_table.h`, `src/metablock.h`, `src/ppmd_prewarm.h`, `src/string_pool.h`, `src/text_start_pack.h`, `src/preprocess/dictionary.h`, `src/preprocess/dict_derive.cpp`, `src/readalike_prepr/censblock_seite.h`, `src/readalike_prepr/komm_transform.h`, `src/readalike_prepr/order_codec.h`, `src/readalike_prepr/phda9_preprocess.h`, `src/tf1/tf1_z3.cpp`, `src/tf1/weights_io.h`, `src/tf1/weights_io_compressed.cpp`, `training/tf/tree_overlay/zctx/z3_extract.py` and `training/bitl/tools/weights_header.py`; the FNV primes alone as multipliers in the context hashes of `src/preprocess/hd_cm.h`, `tools_src/packer/boot_cm.h` and `tools_src/packer/stub_boot.c` | Glenn Fowler, Landon Curt Noll, Kiem-Phong Vo | public domain |
| splitmix64, for the fixed random projection of BITLSTM (`src/bit_head.cpp`) and the initial weights in its trainer (`training/bitl/tools/bg_kern.py`); two of its constants also in the hash of the word-label model (`src/word_label.cpp`, through `zmix`) | Sebastiano Vigna (after G. L. Steele, D. Lea, C. H. Flood) | public domain |
| the Muon optimizer for the weight matrices in the training of the transformer, in the batched form of `cmix-neif-beehive` (`training/tf/3_train/tree/fastmuon.py`, the algorithm of `torch.optim.Muon`) | Keller Jordan and co-authors | - (a method) |

The PPMd prior on the output logits of the small byte LSTM of
`cmix-neif-1.0` (David Freelan, `cmix-obias`), the block-diagonal recurrence of
that LSTM (Xeloraa), its eight-wide `tanh` with the Cephes polynomial and the
xxHash32 steps of its pair and triple hashes were removed with it and are no
longer in the program.

The pre-trained weights shipped with this entry, those of the transformer and
those of BITLSTM, were trained on `enwik9` alone. The transformer weights were
trained for this entry, starting from the weights that Vladimir Malinovskii
trained for `cmix-neif-beehive` from a random initialisation on the training
package of `cmix-neif-1.0`. The weights of BITLSTM were trained for this entry.
Neither Ivanov's weights, nor his article-separator constant, nor his
preprocessor are used, and no weights of `cmix-obias` are used. How the weights
were produced is described in section 5 of `README.md` and in
`training/README.md`. The borrowed parts of the training are named in
`training/ATTRIBUTION.md`.

## Data shipped with the source

`prof_input/` holds two short excerpts of the English Wikipedia, inherited with
the fork base (they are taken from `enwik8`, which is itself the first 100 MB of
`enwik9`); `training/tf/tree_upstream/prof_input/` is an identical copy.  The
build reads `input` only to train a compiler profile when `pgo_data/` holds
none; with the shipped profile neither file is read, and not one of their
bytes ends up in the compressed archive.  Wikipedia text is published under
the **GNU Free Documentation License** and the **Creative Commons
Attribution-ShareAlike** licence; the authors are the contributors to the
articles concerned, listed in the page histories of `en.wikipedia.org`.  The
same is true for the two excerpts `enwik8`/`enwik9` themselves, which the
Hutter Prize distributes.

`dictionary/english.dic`, `dictionary/english.hints` and the article-order
residual were derived from `enwik9` and from the inherited word list and article
order (section 8 of `training/README.md` of `cmix-neif-1.0`); the dictionary
and the order are those of `cmix-neif-1.0`, and only the coding of the hints and
of the residual is new.  They contain word forms and numbers, not text.  Bands
3 and 4 of the dictionary were ordered offline with the pre-trained embedding
models `intfloat/e5-large-v2` and `BAAI/bge-reranker-base` (MIT), of which
nothing ships.

## Changes

This version was changed in 2026 by Marcus Wölk (Marcus Woelk), first as
`cmix-neif-1.0` and then as this version.  The changes of this version concern
the semantic cache SEMCACHE, the census pages and their metadata block, the
start of the text and reversible rules on the page text, the pre-trained
transformer and BITLSTM, the mixer and the PPMd model, and the packer of the
archive.  A build of the source published here reproduces the submitted
compressor byte for byte.
