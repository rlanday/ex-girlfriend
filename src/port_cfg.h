// Port switches of our integration tree (h3_neif2_port_20261031/merge2, Claude Opus 5.5, 2026-10-31).
// Every switch defaults to 0, and with every switch 0 the tree compiles to exactly the upstream program
// (cmix-neif-2.0): the release recipe reproduces the published comp9 byte for byte.
//
//   PORT_CMX_MEAS  research measurement knobs (src/cmx_meas.h), for cheap prefix A/Bs of `comp9 -e enwik9 archive`.
//                  The environment is read in a top-level -e run only (the -D children of the self-extraction
//                  inherit it and ignore it); the decoder never reads it.  Knobs unset = the coding is unchanged.
//                  Observers (never change a coded bit):
//                    CMX_LOG_EVERY=N   a line "CMX <coder_pos> <coded_bytes> <code_bits> <wall_s>" on stderr every N
//                                      coder bytes, at CMX_STOP_AT and at the end of the stream; code_bits is the
//                                      exact code length of the coder interval (8 * coded bytes + 32 - log2 range).
//                                      Also one "CMXMARK <name> <coder_pos> <coded_bytes> <code_bits>" line at each
//                                      region boundary of the stream (meta_end, census_end, slot1/2/4_end,
//                                      text_start = the first byte of page 0 (the front redirect block),
//                                      first_article = the first byte of page 71,148, behind the redirect block).
//                    CMX_AT=p1,p2,...  extra CMX lines at these coder positions (checkpoints, e.g. 20000000 for a
//                                      Delta(50M - 20M) differencing A/B with CMX_STOP_AT=50000000).
//                    CMX_PAGES=k,...   a "CMXMARK page<k>" line where page k of the article order begins.
//                    CMX_BYTECOST=1    per-coder-byte code length (bits, float32) to ./bytecost.f32 at CMX_STOP_AT or
//                                      at the end of the stream.
//                    CMX_STOP_AT=N     stop after N coder bytes: the coder is flushed (the output file is a prefix
//                                      of the archive payload, 5 header bytes included), then _exit(0).  Preprocessing
//                                      runs in full as in the release; only the coding stops early.
//                  Ablations (change the coding; encoder-side measurement only, such an output is NOT decodable):
//                    CMX_TFOFF=1       the transformer row never reaches the models: every byte takes the gated
//                                      form of Perceive (neutral row in the transformer slot, fast-weight head
//                                      neutral), the form the release itself uses inside the metadata blocks (gate
//                                      zone HDRBLOCK), in title masks and on fed rows; so the coding of the metadata
//                                      blocks is unchanged.  SEMCACHE and the top-1 path stage abstain with it (they
//                                      key on a live row).  The engine still steps (its prompt automaton, forced
//                                      bytes, link context and gate stay in sync with the stream), so there is no
//                                      time saving, and BITLSTM still reads the engine's state.
//                    CMX_TFOFF=2       as 1, and BITLSTM's transformer-state input reads zeros (from byte 0).
//                    CMX_SEMCACHE_OFF=1  SEMCACHE takes its no-live-row path at every byte: no lookup, no store, its
//                                      mixer inputs, BITLSTM columns and end stage see "no neighbours", as on every
//                                      byte without a live transformer row; the top-1 path stage keeps running.
//                  Lever switches of the 10-31 batch (encoder-side size screening; full list in src/cmx_meas.h):
//                    CMX_MIXROW=1|2|3, CMX_MIXTEL=1 (layer-0 mixer row cap / re-key at page 1 / fallback telemetry),
//                    CMX_FX_DECCODE, CMX_FX_VERB, CMX_FX_APM_OFF (fxcm stream-position constants),
//                    CMX_MIXSKIP_L0, CMX_MIXSKIP_L1 (x/4096 mixer error-limit skip), CMX_Z3LR (census-block mixer LR),
//                    CMX_ZVAR (census number-coder variants V2 / RK / RF), CMX_SEMTEL=1 (SEMCACHE telemetry).
//                  Research modes: -n STREAM OUT (code a finished coder stream, no preprocessing) and -r IN OUT V
//                  (recode the census line of a coder stream with number-coder variant V).
//
//   PORT_ENTRY     the entry form of recipe "candidate A" (branch entry, Claude Fable 5.2, 2026-11-09): the research
//                  code is compiled in (PORT_ENTRY 1 implies PORT_CMX_MEAS 1) and the recipe is a set of constants that
//                  main() forces into the knobs (cmx_meas::EntryApply, src/cmx_meas.h) in EVERY mode that codes or
//                  decodes the main stream: -e, a top-level -d, -n, and both no-argument forms (comp9 = -e enwik9
//                  archive9, archive9 = the self-extracting decoder).  So the encoder and the decoder run the same
//                  model.  The helper blocks (-D children of an -e run, coded at build time by cmix_aux -c) keep the
//                  release defaults: the mode is decided on argv before the -D -> -d rewrite, and the build helper
//                  (AUX_BUILD) never forces.  The recipe:
//                    CMX_SEMKEYS=R,12 (two extra caches: the release key with its own seed, the mixer read of layer
//                    12; CMX_SEMKEY2_USE=3: layer-0 input + end-mixer column, and the serial end stages),
//                    CMX_FW_BLOCKS=4 (fast-weight head on blocks 4..15), BITL_SEQ=0 (no periodic reset of the BITLSTM
//                    state; it is zeroed at a weight-set switch only),
//                    and the SEMCACHE geometry kM 63, kFlip 15, kNLog 25, kWhLog 26, kPre 512, kDedupLog 10
//                    (src/semantic_cache.h, constants under PORT_ENTRY).
//                  The shipping form (PORT_ENTRY 1, PORT_ENTRY_TEST 0) never reads the environment: cmx_meas::Read()
//                  is not called, the research mode -r is not compiled, no observer runs (cmx_meas::g_hook stays
//                  false, so Compress() runs the model calls of Decompress() and nothing else).  -n stays (it codes a
//                  finished coder stream with the entry recipe).
//   PORT_ENTRY_TEST  with PORT_ENTRY 1 only: a top-level -e or -n run reads the environment as PORT_CMX_MEAS does,
//                  then EntryApply() puts every coding knob back (recipe values and release defaults), so only the
//                  observers CMX_LOG_EVERY, CMX_STOP_AT, CMX_AT, CMX_BYTECOST(+_PATH) and CMX_SEMTEL stay in effect.
//                  For byte-equality certification against the research build; never the form that ships.
#ifndef PORT_CFG_H_
#define PORT_CFG_H_

#ifndef PORT_CMX_MEAS
#define PORT_CMX_MEAS 0   /* 1: CMX_LOG_EVERY / CMX_BYTECOST / CMX_STOP_AT / CMX_TFOFF / CMX_SEMCACHE_OFF (-e only) */
#endif
#ifndef PORT_ENTRY
#define PORT_ENTRY 1   /* 1: recipe "candidate A" as constants in the encoder and the decoder (implies PORT_CMX_MEAS 1) */
#endif
#ifndef PORT_ENTRY_TEST
#define PORT_ENTRY_TEST 0   /* 1 (with PORT_ENTRY 1): observer environment for certification runs; never ships */
#endif
#if PORT_ENTRY
#undef PORT_CMX_MEAS
#define PORT_CMX_MEAS 1
#elif PORT_ENTRY_TEST
#error "PORT_ENTRY_TEST 1 needs PORT_ENTRY 1"
#endif
#if !PORT_CMX_MEAS
// The lap macros of the research CMX_CM_TICKS (src/cmx_meas.h with PORT_CMX_MEAS 1) expand to no token here.
#define CMXK_START(t)
#define CMXK_LAP(sec, t)
#define CMXK_LAP2(sec, t)
#endif

#endif
