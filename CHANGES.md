> **Ex-Girlfriend 1.0 (2026-10-10).** This is a modified version of cmix-neif 1.0 (Marcus Wölk). The text below is
> cmix-neif's own document, unchanged. The changes made after cmix-neif 1.0, their dates and their credits are in
> `EX-GIRLFRIEND.md`; the measured result of this version is there too.

# Changes against `cmix-lex`

The reference point is `cmix-lex` by Ibrahim Marcouch, derived from `fx2-cmix`
(Kaido Orav, Byron Knoll), with Kaido Orav's text model `fxcm_v26`; Hutter
Prize 2026, awarded to Ibrahim Marcouch and Kaido Orav; the published fork
base (`archive9` 109,190,109 B).
This entry keeps its architecture, XML-aware preprocessing, a word-replacing
dictionary transform, a large bank of context models under a two-stage neural
mixer, an arithmetic coder and a self-extracting archive, and changes it in the
nine groups below.

Besides the groups below, the code carries numerous small changes (renamings,
simplifications, error handling and console messages, removal of unused code,
and small run-time and memory improvements) that are not listed one by one.

This file is the modification notice required by section 5a of the GNU General
Public License. Sections 3 and 4 of `README.md` carry the same material with
the measurements, the individual items and the reasons; `NOTICE.md` and
`training/ATTRIBUTION.md` name the borrowed parts and their authors. Section 6
of `README.md` lists further optimisation opportunities that are not part of
this entry.

## 1. The byte model

* The online byte LSTM (170 cells in `cmix-lex`, 208 in this entry before the
  transformer) was the largest single model of `cmix-lex` and also
  its largest single time item. It is **replaced by a byte-level transformer**
  that is trained offline on this entry's own coder stream and shipped frozen
  inside both executables. It has 16 layers, of which every fourth is attention
  over a window of 2048 positions and the others are linear-attention layers,
  about 7.8 M parameters, 4-bit weights and 8-bit activations. The inference
  engine is adopted from Vladimir Ivanov's `fx2-cmix-transformer` and ported
  file by file. This gave 25 ported files and a coder-side driver written after
  his `src/predictor.cpp` that takes over his float16 helpers, his error helper,
  float16 file writer, loss report and per-byte step around the article
  separator, beside 10 further files of that directory that are this entry's
  own. The weights, the alphabet of 208 byte values, the layer count and
  attention window as build constants, the bfloat16 rounding of the unquantized
  tensors, two further container formats and the removal of dead neurons are
  this entry's own; Ivanov's weights, his article-separator constant and his
  preprocessor are not used.
* A **small online byte LSTM of 80 cells**, with a block-diagonal recurrence, is
  kept beside the transformer and feeds the mixer and BITLSTM. It uses part of
  the run time that the transformer leaves within the limit. Besides the
  PPMd distribution of the next byte it reads hashed pairs and triples of the
  last bytes and the byte predicted by the longest match as sparse inputs, and
  the coding cost of the bits of the previous byte as dense inputs. Its output
  logits receive a PPMd prior with a fixed weight, taken from David Freelan's
  `cmix-obias`.
* **BITLSTM, a small LSTM correction model of 16 cells,** sits behind the mixer
  and corrects the final probability additively in logit space. Its form, its
  AVX2 kernels, several of its input features and the approach of
  training it offline on a trace of a compression run are adopted from David
  Freelan's `cmix-obias`, where it was published as `bitlstm32`. The replay
  trainer, which runs the coder's own code and learning step, the input set
  with fixed sketches of the transformer state and of the byte-LSTM state, and
  the measurements behind them are this entry's own. Its weights (7,386 B) are
  compiled into the program. They were trained on the first 20 % of the stream
  only, because a trace of the whole stream and the longer training would have
  cost about three more days; the model runs frozen up to the end of that range
  and learns online from there, at a learning rate of 0.0003.

## 2. Stream and preprocessing

* **Page form.** Most of the XML in `enwik9` is the same on every page. The
  fork base coded it and relied on the models to learn that it is predictable;
  this entry does not code it at all. A page is the title on the first line,
  the body, the edit comment on its own line and, where the revision was
  marked minor, `minor` as a plain word in the last line, with a single
  separator per page.
* **Metadata block.** The per-page metadata are collected in a block at the
  head of the stream, so that every model knows the contributor and the date
  of a page before it codes the page. Its second half is sorted by revision id,
  which bundles the runs of automated edits, and the two halves are paired
  again by rank without any side information. Revision ids and timestamps are
  written as digit pairs in base 100, timestamps above a fixed rank as a
  difference to the predecessor, and a forecast of every timestamp is fed to
  the models as known bytes. Addresses of anonymous edits are packed, and a
  user id is coded only at the first mention of the name. An exclusion model
  over the page and revision identifiers uses the fact that an identifier
  already seen cannot appear again.
* In front of the comment line of every page, the contributor of the page is
  fed to all models as known bytes.
* Deterministically recomputable substitutions for structured number fields:
  where a field follows from other fields of the same record, the stream
  carries only a control byte and the decompressor recomputes it.
* A section reference at the start of an edit comment becomes the index of the
  matching heading of the same page; the tails of the stream, language links
  and similar trailers, are folded into the main stream instead of being kept
  apart. In the front block of redirect pages the usual redirect mark is
  dropped and its three rarer spellings become one code character.
* The tail transform of `cmix-lex` (a reordering of the metadata tail with side
  data at the end of the stream) and its separate stream of ids are not used;
  the metadata block at the head replaces them. Their source
  (`src/r1_reorder_transform.cpp`, `src/r1_reorder_impl.inc`) is still in the
  package and is not called by the compressor.
* The **byte alphabet is frozen at 208 values** and stamped at build time, so
  the models can be laid out for exactly those values.

## 3. Prompt injection and gate

* **Prompt injection.** The transformer is given context that the decompressor
  can derive on its own and that therefore costs zero archive bytes. It
  consists of page number, page id, contributor and date in front of the title, a mark for the
  pages of the redirect block, the redirect synonyms of a page, the incoming
  link context, earlier link texts for the same target, the article title with
  the running heading number at every section heading, the contributor in front
  of the comment line, and length classes for runs of digits. Prompt lines are
  fed to the model but never predicted; the title is fed and predicted.
* **Gate.** The metadata block is withheld from the transformer, in training
  and in coding alike. The gate is driven by bytes the decompressor has already
  seen, so both sides agree.

## 4. Article order

The articles are reordered before coding so that similar articles lie together,
an idea from STARLIT (Artemiy Margaritov), whose reordering code this entry
inherits. The order coded here is a modification of the article order of
`fx2-cmix` (Kaido Orav and Byron Knoll; Byron Knoll built its embedding
pipeline and its manual sort step), in the revision of Kaido Orav's
`fxcm_v26` archive that `cmix-lex` inherited. The pages of the five classes
that its manual sort step places at the end are sorted by descending page id
inside each class, the `Wikipedia:` pages, which that revision gathers in one
block at its end, move to a place behind the census and image pages and in
front of the disambiguation pages, the redirect pages move to the front, and the calendar pages (years,
decades, centuries, days, months, lists of deaths) are sorted into one block by
their titles. The order rule
recognises the page classes of the manual sort step with its strings. The order is not shipped as an explicit
list but as a rule plus a small residual file, which saves about 100 KB of `S`
against an explicit list.

## 5. Dictionary

A larger word-replacement dictionary with an additional code band for frequent
character sequences that were left unworded before. `S` does not carry a
packed dictionary, because the compressor derives the dictionary at run time from the
input plus a compact hint package embedded in the binary, so what is paid for is
a hint block of about 57 KB. The decompressor has no input to derive it from, so
the compressor appends the compressed dictionary to the archive.

## 6. Context models and mixer

* From `zmix 1.0` by James Byrne: indel recovery in the match model of the text
  model, an end-to-end gradient for the two-stage mixer, a log-bilinear
  word-label model over the code words of the dictionary, a time constant of
  one context-map family, a narrower flag field in one SSE stage and the option
  of narrowing a second, and the first mixing stage of a check-value hash, each
  ported from its Zig source or rewritten in C++ for this predictor (the indel
  recovery line by line).
* From Halvor Yttredal's `fx-deepmix`: a second match channel anchored in the
  parser state (ported), the form in which the feed of the byte-model
  prediction into the text model (Kaido Orav's wiring from `fx2-cmix`) is
  restored, and the value 0.1 of the last step of the mixer learning-rate
  schedule.
* Mixer learning rates per stage, and a restart of the learning-rate schedule
  after the metadata block.
* A more compact bucket form for one family of context maps, which holds 16
  instead of 14 entries in the same space, with smaller maps.
* Two double-indirect models removed, two stationary maps halved.
* PPMd at order 25 with an order limit that depends on the classes of the last
  two bytes, order 14 inside the metadata block, and a restart after the block
  with a warm start from the stems of the dictionary words.
* Own additions to the text model: number and date contexts, UTF-8 contexts,
  extra contexts of its first APM stage (coding cost, link targets already
  seen, word-prefix recency, article tail), protection of the four most recently
  used slots in the context-map buckets, a wider dead zone of its internal
  mixers, match positions in 3 bytes, and the XML-tag contexts switched off.
* The final prediction of the text model fills its last input column, which
  held a constant in `cmix-lex`; a 24th layer-0 mixer selects its weights by
  whether the most probable byte of the byte LSTM can still occur after the bits
  seen so far, and by its probability; after the dictionary word "isbn" and nine
  digits the final probability follows the ISBN-10 check digit.

## 7. Speed

Time decides how much model fits, and the time saved here was spent on model
capacity. The group contains adaptive depth, taken from `zmix 1.0`, which codes
a bit with a small logistic model and skips the mixer cascade where that model
is sure; vectorisation of the hot loops, a bfloat16 copy of the byte LSTM's
gate weights for its forward pass, an output layer of the byte LSTM that is not
copied at every byte, the layer-0 dot products in one pass
over the input vector, a logit table indexed from the float exponent, a prefetch
package with one measured prefetch site per place, an automaton that marks bit
positions whose value the structure already determines, a bump allocator that
requests transparent huge pages and asks the kernel to collapse dense regions,
build constants over runtime values that are invariant on this corpus, removal
of code that cannot execute on this corpus, an integer kernel for the attention
values of the transformer, and a profile collected on this entry's own stream
for the inherited profile-guided build with link-time optimisation; the
profile ships with the package. Positions of the transformer whose output is
never read skip the output part of the last layer.

## 8. Memory

The PPMd model needs more memory than the rules allow, and its data live in a
file. The program pages the model itself, with its data in a pool of 4 KiB
frames that is paged in and out with `pread` and `pwrite`, written back only
where it was written, and sized at run time from `/proc/self/statm` in steps of
64 MiB, up to 4 GiB, for a margin of 256 MiB below the limit; section 4.8 of
`README.md` describes it. The memory for a
large pool comes from the compact context-map buckets and smaller maps (about
2 GiB), the removed models and narrower tables (about 0.7 GiB), one compact
table each for the metadata block and for redirects and aliases, and memory
released before helper processes and before the inverse transform. Tables
that are filled lazily, unused pool tails given back and heap trims after the
model is built and after the metadata block save about 150 MiB more. Index
compaction on the frozen alphabet saves a further 7 MiB or so and is
bit-identical only because the byte set is frozen.

## 9. Packaging

* The executables are packed with a two-stage self-extracting stub contained in
  this package instead of with an external packer. It holds a context-mixing
  decoder that models x86-64 code instruction by instruction, after a transform that
  makes RIP-relative displacements and branch targets absolute. The stub finds
  its own image through `/proc/self/exe`, so it is independent of the name it
  is started under and of the working directory. The squash function and its
  table, the inverse stretch table, the arithmetic coder and the APM form of
  the decoder follow Matt Mahoney's `paq7` (GPL).
* Code that runs once per program run is compiled for size, and code the
  submitted configuration can never reach is not part of this source.
* Encoder-only code is placed in its own section, so that it is carried once,
  in the compressor, instead of twice through the embedded decompressor.
* Dictionary hints, article order residual and the transformer weight container
  are appended to the executables as compressed blocks; the weights of
  BITLSTM are compiled into the program. Apart from `enwik9`, its own
  image and two entries under `/proc`, neither program reads a file it did not
  write itself; each one unpacks from its own image what it needs.
* Both programs print a short console report with each step and its duration,
  the position in the stream and the progress.

## Result

| | bytes |
|---|---:|
| `archive9` of this entry, self-extracting | 91,913,343 |
| `comp9` of this entry, the compressor | 4,000,689 |
| `comp9.args` of this entry, the command-line options of the compression run (`-e enwik9 archive9`, one per line) | 19 |
| **`S` = `comp9` + `archive9` + `comp9.args`** | **95,914,051** |
| for comparison, the previous submission `cmix-neif-pre3f` | 97,262,944 |
| for comparison, the published predecessor `cmix-lex`, archive only | 109,190,109 |

## What was adopted from other entries

Named here so that it is visible without reading the register; `NOTICE.md`
lists the items with licences, `training/ATTRIBUTION.md` file by file.

From the 2026 round:

* from **Vladimir Ivanov** (`fx2-cmix-transformer`): the transformer inference
  engine and the training code, with his per-byte step around the article
  separator and, in his container code, his bit-exact port of the sine and
  cosine of NVIDIA's CUDA libdevice and a range coder in the form of the LZMA
  SDK (**Igor Pavlov**, public domain);
* from **David Freelan** (`cmix-obias`): BITLSTM, the correction model behind
  the mixer, with its offline training on a trace, the PPMd prior of the byte
  LSTM (code of its constant-weight build) and the branch-free evaluation of
  the context-map states (the zero-state masking line for line);
* from **James Byrne** (`zmix 1.0`): adaptive depth, indel recovery in the
  match model, the end-to-end mixer gradient, the word-label model, a
  context-map time constant, the width of one SSE flag field and the option of
  narrowing a second, and the first stage of a check-value hash, all in C++
  (the word-label model and the backward recursion of the gradient function by
  function, the indel recovery line by line, from the Zig source);
* from **Halvor Yttredal** (`fx-deepmix`): one ported model, the form of the
  restored feed of the byte-model prediction into the text model (two lines;
  the wiring is Kaido Orav's) and the value 0.1 of the last step of the mixer
  learning-rate schedule;
* from **Rohit Dwivedula's** `fx2-cmix` fork (commit `468e92b9`): an SSE2
  search for check values in the context maps and the reuse of the last
  context lookup in the mixer, both rewritten (the compare in the form of that
  commit, the reuse in its structure), and the header inlining of
  `Sigmoid::Logit` and `MixerInput::Set*`, the latter taken over in the form of
  that commit;
* from **Layth Lakhani's** experiments on Kaido Orav's `fx3-cmix`: two memory
  ideas (narrower index types, a larger shared map);
* from **Seth Lupo's** `fx3-cmix` experiments: the question of the widths of
  the direct state maps.

From earlier work:

* the base article order, built by **Byron Knoll** for `fx2-cmix` with an
  embedding pipeline and inherited through `cmix-lex` in the revision of
  **Kaido Orav's** `fxcm_v26` archive, which gathers the `Wikipedia:` pages in
  one block at its end, and the page classes of his manual sort step, which the
  order rule recognises with the same strings;
* the article reordering code of **STARLIT** (**Artemiy Margaritov**);
* a stretch curve posted on encode.su by **mpais**;
* the squash function, the inverse stretch table, the arithmetic coder and the
  APM form of the self-extractor's decoder from **Matt Mahoney's** `paq7`
  (GPL);
* the hash maps of **Huang Yuanbing** (`emhash`, MIT) and LLVM's `SmallVector`
  and `AlignOf` (**LLVM Project**, in PyTorch's `c10` form) in `src/ds/`, which
  reached the line through **Saurabh Kumar's** `fast-cmix`.

From outside the program family:

* the block-diagonal recurrence of the byte LSTM in the form published by
  **Xeloraa**;
* the Kimi Delta Attention layer type of Kimi Linear (**Kimi Team**), through
  Ivanov's design, whose C++ form follows the kernels of
  `flash-linear-attention` (**Songlin Yang, Yu Zhang, Zhiyuan Li**);
* polynomials of the Cephes library (**Stephen L. Moshier**), and the `exp` and
  `log` routines of `avx_mathfun.h` (**Giovanni Garberoglio**) and
  `sse_mathfun.h` (**Julien Pommier**), zlib, in altered form;
* the civil-date conversion of **Howard Hinnant** (public domain);
* the division by run-time constants of **T. Granlund and P. L. Montgomery** in
  the form of *Hacker's Delight* by **Henry S. Warren, Jr.**;
* a multiply step of the MurmurHash3 finalizer (**Austin Appleby**, public
  domain), and the primes and the first avalanche steps of xxHash32 (**Yann
  Collet**, BSD-2-Clause) in the context hash of the packer and in the pair and
  triple hashes of the byte LSTM;
* the hash scheme FNV-1a (**Glenn Fowler, Landon Curt Noll, Kiem-Phong Vo**,
  public domain);
* the generator splitmix64 (**Sebastiano Vigna**, public domain) in BITLSTM and
  its trainer;
* four constants of fdlibm (Sun Microsystems) in the deterministic `exp` and
  `log`;
* for the self-extracting stub, the idea of the x86 BCJ filter of the LZMA SDK
  (**Igor Pavlov**) and the principle of Crinkler (**Rune L. H. Stubbe, Aske
  Simon Christensen**);
* for the training of the transformer, the Muon optimizer (**Keller Jordan**
  and co-authors) as used by Ivanov's trainer, and for ordering two bands of
  the dictionary offline the pre-trained embedding models
  `intfloat/e5-large-v2` and `BAAI/bge-reranker-base`.
