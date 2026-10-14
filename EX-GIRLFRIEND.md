# Ex-Girlfriend 1.0 modification notice

Modified by Ryan Landay, developed with Claude (Anthropic) and Codex (OpenAI).
Project techniques: September 24–October 10, 2026. Neif port: October 6–10.
Release documentation and verification receipt: October 14, 2026.
Our modifications use GPL-3.0-only; all inherited notices remain in force.

This derivative of Marcus Wölk's cmix-neif 1.0 is the original FINAL candidate,
91,778,237 bytes, verified by a fresh full decode and exact comparison. The
[README](README.md) gives results, provenance, measurements and limitations.
[FAST_WEIGHTS.md](docs/FAST_WEIGHTS.md) describes the implementation, and
[ABLATIONS.md](docs/ABLATIONS.md) separates component screens from the full result.

The following files differ from the pinned upstream source. Source text is
preserved from the measured release tree; this notice identifies the changes
without inserting new lines into arithmetic/build inputs.

| Files | Change |
|---|---|
| `src/port_cfg.h` (new) | Compile-time switches: active geometry, fast weights with four blocks, larger tables; other experiments disabled |
| `src/neif_levers.h` (new) | E1/E3b metadata switches |
| `src/b1_interp.h` (new) | Integer page-id/revision-id forecast |
| `src/metablock.h`, `src/metablock_form.h`, `src/id_exclude.h` | Reversible E1 row form and associated exclusion handling |
| `src/ts_progress.h` | E3b post-migration timestamp forecast |
| `src/models/fxcmv1.cpp` | W3 table capacities; inactive experimental branches also retained |
| `src/tf1/opt/model_opt.cpp`, `model_opt.h` | Four-block auxiliary readout and online updates |
| `src/tf1/opt/arena_build.cpp`, `arena_build.h` | Float unembedding reconstruction for the surrogate update |
| `src/tf1/tf1_engine.cpp`, `tf1_engine.h` | Auxiliary output slot and causal target pairing |
| `src/predictor.cpp`, `predictor.h`, `src/mixer/mixer.cpp`, `mixer.h` | Expanded mixer inputs and wiring; inactive rate branches retained |
| `src/context-manager.cpp`, `src/word_label.cpp` | Inactive restoration/rate experiment branches; not release features |
| `makefile` | Remove profile-use flags, retain LTO and numeric flags |
| `CHANGES.md`, `NOTICE.md`, `README.cmix-neif.md` | Dated notices over preserved upstream documentation |
| `README.md`, `EX-GIRLFRIEND.md`, `docs/`, `verification/`, `.gitignore` | Release description, provenance, evidence and repository hygiene |

The original upstream README is retained as `README.cmix-neif.md` and its
environment record as `environment.cmix-neif.txt`. The preserved upstream
`CHANGES.md` describes neif's history, not new Ex-Girlfriend work. The source
patch against the pinned base is in `verification/upstream_diff.patch`.

All model weights, dictionary/hint inputs, article-order residual, packer,
training tooling and third-party licenses are inherited unchanged.
