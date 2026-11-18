# Stage (b): the coder-side runs

Everything the trainer consumes is produced *by the compressor itself*, from
`enwik9`. The tools are plain builds of the source of this entry with the patch
`trainmodes.patch` of this directory, which restores the run modes of the
development tree as `TRAIN_MODE` 1 to 5. Without `TRAIN_MODE` the patched
source builds the compressor of the entry unchanged.
`../2_package/build_package.sh` runs the whole chain; this page describes the
runs one by one.

The stream of the text form: the metadata block and behind it the census
metadata block (both gated, zone 3), then the text start (the dictionary
article, the contributor article and the census pseudo article), then page 0,
whose title line is a short zone-3 run of its own (the gate runs on over it),
then the pages.

| file | what it is |
|---|---|
| `trainmodes.patch` | the run modes on the source of the entry (section 0) |
| `build_dump_tools.sh` | builds the tools of the table in section 0 from the source of the entry with the patch; `--tree` builds the first four from the development tree with the build lines below |
| `features_stream.txt`, `features_dump.txt`, `features_priors.txt` | the build lines with which the development tree built the tools of the shipped training package |
| `features_end.txt` | a coder line of the development tree, reference of the line check of `--tree` |
| `dumpleser.py` | reference reader of the dump records |

## 0. Build

```
bash build_dump_tools.sh <entry directory> <tools directory> [stream dump priors bed trace]
```

The script copies `src/`, `makefile`, `encoder_split.lds` and
`runtime_exports.map` of the unpacked source package, applies the patch and
runs `make cmix TRAIN_MODE=<n> CC="clang++-18 -DTRAIN_MODE=<n>"` (no profile,
no LTO, no packing; clang 18 and lld 18 as in `install.sh`), about 20 s per
tool with 8 make jobs (`JOBS`).

| `TRAIN_MODE` | tool | run | writes |
|---:|---|---|---|
| 1 | `cmix_stream` | `-e enwik9` | the coder stream (`out.cmix.temp`), stops before the coder |
| 2 | `cmix_dump` | `-n` on the bed | the fed rows, the zone mask, the match channel (section 2) |
| 3 | `cmix_priors` | `-n` on the bed | the PPMd priors, their byte tokens and origin (section 3) |
| 4 | `cmix_bed` | `-e enwik9` | the canonical stream (the bed of the `-n` runs), the census pseudo article of the text start (`.vs3`) and the coder stream, stops before the coder |
| 5 | `cmix_trace` | `-e enwik9` | the inputs of BITLSTM for every bit (`../../bitl/README.md`, section 1) |

**How the patch was made.** The package generator writes the source of the
entry from the development tree and resolves every build switch to the value
of the submitted build. It was run once with the build line of the entry and
once per mode with that line plus the switches of the mode; the patch merges
the five mode sources into the source of the entry. Where a mode differs, a
region reads

```
#if TRAIN_MODE == 2 || TRAIN_MODE == 3
  (text of these modes)
#else
  (text of the entry)
#endif
```

(in the `makefile`: `ifneq (,$(filter $(TRAIN_MODE),2 3))`). Regions never
cross a conditional or a continued line of the file, and `unifdef
-DTRAIN_MODE=0` of every patched file gives back the file of the entry; for
each mode `n`, `unifdef -DTRAIN_MODE=n` gives back the generated source of that
mode (with the English texts below). Switches of the development tree added to
the line of the entry:

| mode | switches |
|---|---|
| 1, 4 | `PREPARE_ONLY=1 REUSE_EXTRACTED=1`; 4 also `CENSBLOCK_PRUEF=1` |
| 2, 3 | `PROGRESS=1 TF1_SLOT=0 B5C_SLIM=0 TF1_ENGINE_FIXED=0 TF1_SELBSTAUSKUNFT=1 TF1_PAKET_DUMP=1` and `TF1_DUMPONLY=1` (2) or `PPMD_ONLY=1` (3); without the semantic cache, the embedded container, the transformer state as an input of BITLSTM and the cuts of the release form that need the fixed engine |
| 5 | `BITL_MESS=1 BITL_TRACE=1 REUSE_EXTRACTED=1` |

In all five modes BITLSTM runs without the three stamped weight sets (its
online form, as in the build lines of the development tree); none of the
outputs depends on its weights. In mode 5 it also runs in the metadata blocks,
so the trace covers the whole stream. `PROGRESS=1` gives modes 2 and 3 the
stderr lines the recipe reads (the slot sizes of the text start, the PPMd
restart and warm-ups) and `progress.log`. The run modes keep the names of the
development tree, and `NEIF_*` environment variables name their output files.
The messages that only these modes print are given English texts before the
merge (a table of 40 replacements, none of them in the source of the entry).

## 1. The bed run

```
mkdir bed && cd bed
cp <dictionary> .dict                 # 520,804 B, sha256 0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535
cp <order v4> .new_article_order      # 40,694 B, sha256 0eec59a305774b5a46b17b83a31274aafd5081ba387a3536def3372a442ba59d
<tools>/cmix_bed -e <enwik9> out
```

The dictionary is `dictionary/english.dic` of the entry, the order file is
`src/readalike_prepr/data/new_article_order`.

| output | what it is | bytes | sha256 |
|---|---|---:|---|
| `censpruef_kanon.bin` | canonical stream: the bed of the `-n` runs | 508,139,578 | `c10b5f3d…` |
| `censpruef_vs3.bin` | census pseudo article of the text start, `.vs3` of the `-n` runs | 82,445 | `ed68bb4becbaae3e…` |
| `censpruef_strom.bin` | coder stream | 501,062,281 | `c35d3073eeba70f2…` |
| `.main_phda9prepr` | stream before the article order | 797,117,307 | `aec42ab7821e9f7a…` |

`cmix_stream -e` in a second directory with the same inputs writes the same
coder stream as `out.cmix.temp` (cross-check). About 6 min on an EPYC 7502P,
2.1 GB maximum resident set, about 5 GB of temporary files in the run
directory.

## 2. The dump run

```
mkdir dump && cd dump
cp <dictionary> .dict
cp ../bed/censpruef_vs3.bin .vs3
ln -s /dev/null aus
NEIF_TF1_WEIGHTS=none NEIF_TF1_PAKET_DUMP=./ NEIF_TF1_GATE_DUMP=$PWD/zonenmaske.u8 \
NEIF_TF1_MATCH_DUMP=./ NEIF_TF1_SAVE_PRIORS=/dev/null OMP_NUM_THREADS=1 \
  <tools>/cmix_dump -n ../bed/censpruef_kanon.bin ./aus
```

`NEIF_TF1_WEIGHTS` only switches the transformer path on; the dump build never
reads a container. **`.vs3` must lie in the run directory.** Without it the
census pseudo article is packed empty and nothing complains (stderr
`slot content a / b / 0 B`).

The run is complete when the exit code is 0, `zonenmaske.u8`,
`match-bytes.uint8` and `match-len.uint8` have one byte per coded byte (coder
stream + 6, 501,062,287 B), the last stream row of `paket-zeilen.bin` is the
last coded byte and every stream row carries the token of the coder stream
(`../2_package/dump_check.py <run dir> <coder stream>`). The stderr line
`text start pack: … (slot content a / b / c B)` names c = size of `.vs3`.

| output | what it is |
|---|---|
| `paket-zeilen.bin` | fed rows, 12-B records with a header record (section 4) |
| `zonenmaske.u8` | zone id per coded byte: 3 on the metadata blocks (first run, from the stream head) and on the title line of page 0, 0 elsewhere |
| `match-bytes.uint8`, `match-len.uint8` | per coded byte the token index of the predicted byte (208 = no match) and the match length |

## 3. The prior run

```
mkdir priors && cd priors
cp <dictionary> .dict
cp ../bed/censpruef_vs3.bin .vs3
ln -s /dev/null aus
NEIF_TF1_WEIGHTS=none NEIF_TF1_SAVE_PRIORS=<dir>/ppmd-probs.float16 \
NEIF_PPMD_ONLY_BYTES=<dir>/ppmd-bytes.uint8 NEIF_PPMD_ONLY_QUELLE=<dir>/ppmd-quelle.u8 OMP_NUM_THREADS=1 \
  <tools>/cmix_priors -n ../bed/censpruef_kanon.bin ./aus
```

PPMd-only mode runs the PPMd model of the coder on every byte the compressor
feeds it, the coded bytes and the bytes injected through all models (the
contributor names, the timestamps, the census values, the coordinates of census
pages and the heads of the text-start articles), and writes per byte the PPMd
distribution over the 208 byte values as float16 (416 B per row; row *i* is
the distribution for byte *i+1*), the token of the byte and its origin (0
coded, 1 injected). The stderr lines `ppmd: metablock end at model byte …` and
`ppmd warm-up …` give the restarts and the warm-ups; the warm-up at the end of
the dictionary article is the same in the dump run.

The run is complete when the exit code is 0, the rows of origin 0 are exactly
the coded bytes (coder stream + 6) with their tokens and the priors have 416 B
per row (`../2_package/dump_check.py <run dir> <coder stream> --prioren <dir>
--kopf-laenge <size of the bed>`). The 6-byte segment header of a `-n` run
carries the length of its input (the bed) in place of the length of the coder
stream; it stands in front of the PPMd restart at the end of the metadata
blocks and in front of the cut. The package keeps the rows of coded bytes
behind the cut (`trim_quelle.py`).

The dump run and the prior run read the same bed and run in parallel.

## 4. Record format of the dump (version 2), read by `dumpleser.py` and `../2_package/dumpv2.py`

```
record 0 (header):  int32 -2 | 'P' 'K' 'D' | uint8 version 2 | uint8 record size 12 |
                    uint8 number of used flag bits (8) | uint16 0
records (12 B):     int32 src (-1 = inserted) | uint8 tok prev1 prev2 match_b match_cls flags |
                    uint16 reserved
flags:              bit0 inserted . bit1 MASKED . bit2 document start .
                    bit3 redirect page . bit4 prior0 row (no stream target) .
                    bit5 reinjected alias or contributor byte . bit6 contributor byte .
                    bit7 metadata prompt row
```

`src` is the position in the coded bytes (coder stream + 6-byte segment
header).
A dump written by an `-e` run (a coder build with the dump switches) counts
`src`, the zone mask and the match files from the first byte of the coder
stream, without the 6 header bytes; add 6 before comparing it with a `-n`
dump.

## 5. Probe bed C40

The first 40,000,000 B of the canonical stream, cut back to the end of the last
page anchor in them (stream stages look ahead inside a page; the rule stages
start at page 71,361, byte 21,628,867). It holds the metadata blocks, the text
start and about 22.5 MB of whole pages. `build_package.sh` with
`PREFIX=40000000`. The dump writes no row for the last byte of a page anchor,
so the last stream row of a probe dump is the coded byte in front of it.
Values of the stream form of this entry:

| quantity | value |
|---|---|
| probe bed | 39,998,324 B |
| coded bytes (zone mask) | 37,565,854 |
| cut (`token_start`), `n_tokens` | 13,023,920, 24,541,934 |
| further zone-3 run (title line of page 0) | 15,282,101 to 15,282,107 (7 B) |
| `slot content` | 490,360 / 1,685,370 / 82,445 B |
| dump records | 29,546,663 (24,537,085 stream, 5,009,578 inserted) |
| prior raster | 46,621,131 rows (37,565,854 coded, 9,055,277 injected), `ppmd-bytes` of the package sha256 `f045e25fff19b7de…` |
| package documents / longest | 4,844 / 131,038 rows |
| positions masked by the coder (forced name bytes of the contributor article) | 694,429 |

## 6. The tools of the patch against those of the development tree

The shipped training package was built with the tools of the development tree
(`--tree`, clang 17). The tools of the patch (clang 18, EPYC 7502P) were run
against them on the same inputs; every output was the same, byte for byte:

| tool | input | outputs (bytes, sha256) |
|---|---|---|
| `cmix_bed` | `enwik9` | the four outputs of section 1 (`c10b5f3d…`, `ed68bb4b…`, `c35d3073…`, `aec42ab7…`) |
| `cmix_stream` | `enwik9` | `out.cmix.temp` 501,062,281 `c35d3073…`, `.main_phda9prepr` `aec42ab7…` |
| `cmix_dump` | probe bed C40 | `paket-zeilen.bin` 354,559,956 `0b124db7…`, `zonenmaske.u8` `dc8b1597…`, `match-bytes.uint8` `3b6e8d55…`, `match-len.uint8` `bec3f847…` (37,565,854 each) |
| `cmix_priors` | probe bed C40 | `ppmd-probs.float16` 19,394,390,496 `eebb3812…`, `ppmd-bytes.uint8` `2b56a78b…`, `ppmd-quelle.u8` `49729226…` (46,621,131 each) |
| `../2_package/build_package.sh`, `ENTRY` form | `PREFIX=40000000` | the probe package of section 5 (`zeilen.dump` `e861d051…`, `ppmd-bytes.uint8` `f045e25f…`, prior0 `97de4610…`, trimmed priors `fb615d84…`, `seiten.tsv` `9808f122…`), acceptance passed |
| `cmix_trace` | `enwik9`, container `729694cb…` | the first 40,000,000 records of `spur.bin` (6,560,000,020 B) `c918c795…` and the first 700,000,000 B of `kanal.bin` `5faeb21b…`, against the trace build of the development tree with `BITL_META_OFF=0` |

The full-stream dump and priors were not rerun with the tools of the patch;
the probe bed C40 holds the metadata blocks, the text start and about 22.5 MB
of whole pages (section 5).
