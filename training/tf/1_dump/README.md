# Stage (b): the coder-side runs

Everything the trainer consumes is produced *by the compressor itself*, from
`enwik9`. The source of the entry is reduced to the submitted configuration and
has no mode that writes these files. `trainmodes.patch` adds three modes to a
copy of `src/`; `build_dump_tools.sh` builds one tool per mode. The source of
the entry stays unchanged.

| file | what it is |
|---|---|
| `trainmodes.patch` | adds the stream, dump and prior modes to a copy of `src/` (9 files, `TRAIN_MODE` blocks and conditions; built without the define, the patched copy gives the same binary as the unpatched one) |
| `build_dump_tools.sh` | copies `src/`, applies the patch, builds `cmix_stream`, `cmix_dump`, `cmix_priors` |
| `dumpleser.py` | reference reader of the dump records (record format below) |
| `features_*.txt` | record: build lines of the development tree (commit `786964b1`) that ran stage (b) for the shipped transformer; the tools give the same outputs (section 4) |

## 0. Build

```
bash build_dump_tools.sh <entry directory> <tools directory>
```

Toolchain of `build.sh` (clang++-18, ld.lld-18, make) plus `patch`. Plain
builds, no profile, no LTO, no packing.

| tool | build | mode |
|---|---|---|
| `cmix_stream` | `-DTRAIN_MODE=1` | `-e`: reads `./.dict` and `./.new_article_order` instead of the blocks of the binary, writes the coder stream before and after the pack steps, stops after the preprocessing |
| `cmix_dump` | `-DTRAIN_MODE=2` | `-n`: codes the unpacked stream with a constant prediction; the match models, the gate, the prompt automaton, the link context and the metadata unit run, the transformer forward step and all other models do not; writes the fed-row dump, the zone mask and the match channel |
| `cmix_priors` | `-DTRAIN_MODE=3` | `-n`: as `cmix_dump`, with PPMd as the only model; writes in addition the PPMd distribution and the token of every byte fed to PPMd, and the PPMd restart line on stderr |

The archives of `cmix_dump` and `cmix_priors` are not decodable. All outputs go
to the working directory, so each run gets its own directory.

## 1. The coder stream

```
mkdir stream && cd stream
cp <entry>/dictionary/english.dic .dict
cp <entry>/src/readalike_prepr/data/new_article_order .new_article_order
<tools directory>/cmix_stream -e <enwik9> archive
```

| output | bytes | sha256 |
|---|---:|---|
| `archive.cmix.temp`, the coder stream (packed form) | 552,750,198 | `8e0b264a777cb56a6be48c58002450c4711b1cebf3ffcb0129a32ef34273c3ba` |
| `archive.cmix.temp.vorpack`, the same stream before the metadata block is packed | 555,771,398 | `e556d3f837c66c69e0a4f3a1e972b05bdb034633f6a8cadd6d91ef9c34135111` |

Input files, both shipped with the entry: `.dict` 520,804 B, sha256
`0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535`;
`.new_article_order` 115,870 B, sha256
`c9f8b1001bfbc041a087eab65e5caa609eb87bb8ecf297cfcdbcdaffa7a1dd19`.

About 1 min, 1.4 GB RAM. The packed stream is the one the compressor of this
entry codes. The unpacked form is the input of the two `-n` runs: `-n` applies
the pack steps itself, so the coder sees the stream of a `-e` run behind a
6-byte segment header (552,750,204 B in total).

## 2. The dump run

```
mkdir dump && cd dump
cp <entry>/dictionary/english.dic .dict      # dictionary pretraining of the match models
ln -s /dev/null aus
<tools directory>/cmix_dump -n ../stream/archive.cmix.temp.vorpack ./aus
```

The run is complete when its last stderr line `coded stream bytes N` gives the
stream size + 6 (552,750,204).

| output | value |
|---|---|
| `paket-zeilen.bin` | 7,117,383,324 B, sha256 `58fc6c74c09400192c28cf3608e6eeb0387ba4f40b4255d39dbc8ac5e0bf66a0`, 593,115,277 records including the header |
| `zonenmaske.u8` | 552,750,204 B, one zone id per stream byte; zone 3 (metadata block) on bytes 1 to 5,164,202, zone 0 elsewhere |
| `match-bytes.uint8`, `match-len.uint8` | 552,750,204 B each, per stream byte the token index of the predicted byte (208 = no match) and the match length (capped at 255) |
| first byte after the metadata block | 5,164,203 (the training stream starts here) |
| runtime | 22 to 25 min on one core (i9-13900), 6.5 GB maximum resident set |

### Record format of the dump (version 2), read by `dumpleser.py` and `../2_package/dumpv2.py`

```
record 0 (header):  int32 -2 | 'P' 'K' 'D' | uint8 version 2 | uint8 record size 12 |
                    uint8 number of used flag bits (8) | uint16 0
records (12 B):     int32 src (-1 = inserted) | uint8 tok prev1 prev2 match_b match_cls flags |
                    uint16 reserved
flags:              bit0 inserted . bit1 MASKED . bit2 document start .
                    bit3 redirect page . bit4 prior0 row (no stream target) .
                    bit5 reinjected alias or author byte . bit6 author byte .
                    bit7 metadata prompt row
```

Author rows run through all models and carry bit5 and bit6; alias rows reach
only the transformer and carry bit5 alone (`../2_package/prior_mask.py`).

## 3. The prior run

```
mkdir priors && cd priors
cp <entry>/dictionary/english.dic .dict
ln -s /dev/null aus
<tools directory>/cmix_priors -n ../stream/archive.cmix.temp.vorpack ./aus
```

To write the priors to another disk, create `ppmd-probs.float16` and
`ppmd-bytes.uint8` as symlinks into it before the run. The run also writes the
four files of section 2 (match channel: 208 and 0 throughout); `prior_mask.py` reads its
`paket-zeilen.bin`.

PPMd-only mode runs the PPMd model of the entry (an order limit per context
class, order 14 inside the metadata block, a restart after it, a warm start
from the word stems of the dictionary) on every byte the compressor feeds it and
writes, per byte, the PPMd distribution over the 208 byte values as float16
(416 B per row). Row *i* is the distribution for byte *i+1*, written after the
model was updated with byte *i*. The stderr line
`ppmd: metablock end at model byte N, order 25 -> 25, ...` gives the restart
position, where the trimming cuts.

The raw raster has 555,385,644 rows, the stream bytes plus the bytes the
coder injects through all models (author names of the metadata unit, 1,561,159,
and the timestamp feed before the restart, 1,074,281). The training package
keeps the rows of stream bytes behind the metadata block, 547,586,001 rows
(`../2_package`, trimming step); the row sums of 4,000 samples lie in
0.99967 ... 1.00030.

| output | value |
|---|---|
| `ppmd-probs.float16` after trimming | 227,795,776,416 B, sha256 `3ee22a2618de7d9ac458ada84d92247a58676e32571df4c9dc52494de5768194` |
| `ppmd-bytes.uint8` after trimming | 547,586,001 B, sha256 `bc419ee2f7bd4dddc6004851ab0c87623c6e59ef3fccb778a4d650e94c1ad9f9` |
| raw `ppmd-probs.float16` | about 231 GB |
| runtime | 1 h 2 min to 1 h 12 min on one core (i9-13900), 8.6 GB maximum resident set; trimming and probes (`../2_package`) 12 to 14 min |

## 4. Tools against the original runs

The original runs used the development tree with the build lines in
`features_*.txt` (clang 17, no profile; the stream build had one extra file
copy, which the stream mode contains). Byte comparison; on the bed both sides
ran on the same input, on the full stream the tools are compared with the
outputs of the original runs (`../../MANIFEST.tsv`):

| run | input | outputs | result |
|---|---|---|---|
| stream | `enwik9` | `archive.cmix.temp`, `.vorpack`; md5 of `.main_reordered` | equal (`8e0b264a…`, `e556d3f8…`, `28ca879d…`) |
| dump | 8-MiB bed | `paket-zeilen.bin`, `zonenmaske.u8`, `match-bytes.uint8`, `match-len.uint8` | equal (`f0222bc6…`, `265a235c…`, `0eeb53da…`, `bdd5ed97…`) |
| priors | 8-MiB bed | `ppmd-probs.float16`, `ppmd-bytes.uint8` and the four dump files; restart line (model byte 6,238,484, order 25 -> 25) | equal (`ecd93e33…`, `fae22290…`) |
| dump | full unpacked stream | `paket-zeilen.bin` | equal (`58fc6c74…`); `zonenmaske.u8` behind the cut: zone 0 only (547,586,001 B) |
| priors | full unpacked stream | trimmed `ppmd-probs.float16` and `ppmd-bytes.uint8`, `z3-prior0.float16` (`../2_package/build_package.sh`, sections 0, 1, 4 to 6); restart line | equal (`3ee22a26…`, `bc419ee2…`, `59121815…`); model byte 6,238,484, order 25 -> 25, as in the original run |

8-MiB bed: the first 8,388,608 B of `archive.cmix.temp.vorpack`, sha256
`0e570e2b9c3fd752c0e67956ecee25861eb6bc94388dc8bdd9bf78f30d2af8d9`. It holds
the metadata block and the first 4,307 pages, all from the redirect block, no
article text; the two full-stream rows cover the text.
The patched copy built without `-DTRAIN_MODE` and the unpatched copy give the
same binary.

## Not in this package

The outputs above, 235 GB in total; `../../MANIFEST.tsv` lists them as
references with their checksums.
