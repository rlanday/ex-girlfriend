# Ex-Girlfriend 2.0 modification notice

Modified by Ryan Landay, developed with Claude (Anthropic) and Codex (OpenAI).
Changes: October 31–November 10, 2026. Release documentation and verification
evidence: November 18, 2026. Our modifications use GPL-3.0-only; all
inherited notices remain in force.

Ex-Girlfriend 2.0 is a modified version of Marcus Wölk's cmix-neif 2.0. Its
archive is 88,983,124 bytes. We checked it by decoding a fresh copy and
comparing the output with the original enwik9, byte for byte. The [README](README.md) gives the result, provenance, changes,
measurements and limits. Ex-Girlfriend 1.0, a modification of cmix-neif 1.0,
is documented in [docs/EX-GIRLFRIEND-1.0.md](docs/EX-GIRLFRIEND-1.0.md).

The base is upstream's source package `cmix-neif-2.0.tar.gz`, SHA-256
`62da32b97f9909ce5f2d4988a84e1b9cfbff4c9947538cbc6659a2915b024874`,
imported unchanged. The complete source difference from it is
[verification/upstream_diff.patch](verification/upstream_diff.patch); the
research commits it condenses are listed in
[verification/port_history.txt](verification/port_history.txt).

## Files changed from cmix-neif 2.0

| Files | Change |
|---|---|
| `src/port_cfg.h` | Build switches `PORT_CMX_MEAS`, `PORT_ENTRY` (set to 1 in this release) and `PORT_ENTRY_TEST` (0; certification observers, never shipped) |
| `src/cmx_meas.h` (new code) | The research knobs, their environment reader (not called in the shipping build) and `EntryApply()`, which fixes the 2.0 recipe and sets every other knob to its release default |
| `src/semantic_cache.h` | The six cache geometry constants of the recipe under `PORT_ENTRY` |
| `src/predictor.cpp`, `src/predictor.h` | The two extra semantic caches (construction, per-bit prediction, end stages, learning, mixer inputs); research alternatives that the recipe leaves at release defaults |
| `src/tf1/opt/model_opt.cpp` | First fast-weight block from `g_fw_first` (4 in the recipe); copy of the layer-12 key for the extra cache |
| `src/bit_head.cpp` | BITLSTM state reset rule from `g_bitl_seq` (0 in the recipe: reset only at weight-set switches) |
| `src/mixer/mixer.cpp`, `src/mixer/mixer.h` | Mixer input counts for the extra caches; research switches left at release defaults |
| `src/runner.cpp` | Calls `EntryApply()` in every mode that codes or decodes the main stream; research modes (`-n` coder-only runs, recode, telemetry) used for the screens |
| `src/coder/encoder.cpp`, `encoder.h`, `src/coder/decoder.cpp` | Research telemetry hooks; the shipping build keeps only the final flush |
| `src/models/fxcmv1.cpp`, `fxcmv1.h`, `src/models/match.h`, `src/models/ppmd.cpp`, `src/word_label.cpp`, `word_label.h`, `src/census_block.h`, `src/readalike_prepr/censblock_seite.h`, `censblock_zahl.h` | Research switches and telemetry for screened alternatives, left at release defaults by `EntryApply()`; not part of the recipe |
| `tools_src/packer/cut_dpath.py` | Cut check: a `NameError` fix and rule 4a (immediates inside function bodies are numbers) |
| `makefile` | Every `-fprofile-use` flag removed (no PGO; LTO kept) |
| `README.md` (new), `README.cmix-neif.md`, `environment.cmix-neif.txt` | Upstream's README and environment record renamed without content changes; new README for 2.0 |
| `EX-GIRLFRIEND.md`, `docs/`, `verification/`, `.gitattributes`, `.gitignore` | Release documentation and evidence |

`build.sh`, `build_and_construct_comp.sh`, `install.sh`, the dictionary, the
article order, the hint source, the transformer weight container, `pgo_data/`,
`training/`, `third_party/`, `LICENSE`, `NOTICE.md` and `CHANGES.md` are
upstream's, unchanged.
