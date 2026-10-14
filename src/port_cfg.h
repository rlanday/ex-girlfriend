#ifndef PORT_CFG_H_
#define PORT_CFG_H_

// Lane C A/B arm switches (work/h3_neif_port_20261006, 2026-10-06).  Each arm
// tree carries its own copy of this header; the code is the same in every arm.
//
//   PORT_GEOM  layer-0 row 588 -> 589 and final mixer 27 -> 28.  The new
//              columns are 0 (neutral) unless PORT_FW wires them.  Adaptive
//              depth gathers the old 588 columns only and BITLSTM keeps its
//              28-wide row (final-mixer cols 0..26 + byte-LSTM logit).
//   PORT_FW    FW-DUAL/FWRH: fast weights on the last MLP of the transformer
//              (dual head, FW-ROW per-feature RMS step, DIC kappa 6e-3 ->
//              1.5e-3, tau 8000), its distribution as layer-0 col 588 and
//              (FW1) as final-mixer col 27.  All FW math is in the precise TU
//              src/tf1/opt/model_opt.cpp.
//   PORT_PRL   per-row mixer decay: max(global step schedule, row schedule
//              1.0/0.7/0.3/floor at row updates < 1k/5k/25k/else).
//   PORT_ETA0  word-label eta for byte-0 trie nodes = 0.0225 (WL2 eta0).
//
// Lane C second wave (2026-10-07), LTCB-only capacity knobs, each on top of B
// (GEOM+FW) and switchable so they can be stacked:
//   PORT_CASCADE  (W1) the mixer cascade (layer-0 mixers, final mixer, SSE)
//              runs on EVERY bit.  Adaptive depth still computes its skip
//              decision and trains on every bit as before; the decision now
//              only gates BITLSTM (Adjust/Observe in coder/encoder.cpp and
//              decoder.cpp, unchanged), so BITLSTM's bit set and position
//              counter are the same as with adaptive depth.  On a would-be-
//              skipped bit the coder takes the cascade p instead of p_pre.
//   PORT_RESTORE  (W2) undo neif's Hutter-RAM cuts (README 3.3 "two models
//              removed, two maps halved" + shared map):
//                PORT_RESTORE_IND    double-indirect models ind2/ind3 with
//                  their tables (16M + 32M uint32), update rule and position
//                  (between ind1 and ind5) as in fx2-cmix-transformer
//                  (training/tf/tree_upstream); layer-0 row +2 (589 -> 591);
//                  the two columns sit inside the adaptive-depth gather, as
//                  before the removal (AD: 33 -> 35 features).
//                PORT_RESTORE_MAPS16 text-model stationary maps maps1/maps2
//                  at 16 context bits (neif: 15).
//                PORT_RESTORE_SHARED shared indirect map 1,000,000 rows
//                  (neif: 500,000).
//   PORT_BIGTAB   (W3) bigger text-model tables:
//                PORT_BIGTAB_CM3 every ContextMap3 table x2 (m = m1*4,
//                  neif m1*2; fxcmv1.cpp ContextMap3::Init).
//                PORT_BIGTAB_DSM the five direct state maps +1 bit each
//                  (28/29/24/27/26 -> 29/30/25/28/27).
#define PORT_GEOM 1
#define PORT_FW 1
#define PORT_PRL 0
#define PORT_ETA0 0
#define PORT_FW_LAYERS 4   /* F4: FW on the last 4 MLP blocks (see F4/src port_cfg.h) */
#define PORT_CASCADE 0
#define PORT_RESTORE 0
#define PORT_BIGTAB 1

#ifndef PORT_RESTORE_IND
#define PORT_RESTORE_IND PORT_RESTORE
#endif
#ifndef PORT_RESTORE_MAPS16
#define PORT_RESTORE_MAPS16 PORT_RESTORE
#endif
#ifndef PORT_RESTORE_SHARED
#define PORT_RESTORE_SHARED PORT_RESTORE
#endif
#ifndef PORT_BIGTAB_CM3
#define PORT_BIGTAB_CM3 PORT_BIGTAB
#endif
#ifndef PORT_BIGTAB_DSM
#define PORT_BIGTAB_DSM PORT_BIGTAB
#endif

#if PORT_FW && !PORT_GEOM
#error "PORT_FW needs PORT_GEOM (the FW columns are the new ones)"
#endif

#if PORT_RESTORE_IND
#define PORT_IND_COLS 2
#else
#define PORT_IND_COLS 0
#endif

#if PORT_GEOM
#define PORT_N0 (589 + PORT_IND_COLS)
#define PORT_N1 28
#else
#define PORT_N0 (588 + PORT_IND_COLS)
#define PORT_N1 27
#endif

#endif
