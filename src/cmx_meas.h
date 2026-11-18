// Research measurement knobs of PORT_CMX_MEAS (src/port_cfg.h; Claude Opus 5.5, 2026-10-31).
// Included only when PORT_CMX_MEAS is 1.  The values are plain globals at their release defaults unless a top-level
// -e or -n run read the environment (Read(), from main()); the encoder loop (runner.cpp, Compress), the predictor,
// the mixers, fxcm and the census number coder read them.  Every default is the release behaviour.
#ifndef CMX_MEAS_H_
#define CMX_MEAS_H_

#include "port_cfg.h"   // PORT_ENTRY / PORT_ENTRY_TEST (every includer has it already; guarded)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>   // __rdtsc (CMX_CM_TICKS, CMX_VALTEL)

namespace cmx_meas {
inline long long g_log_every = 0;   // CMX_LOG_EVERY: CMX line every N coder bytes (0 = off)
inline long long g_stop_at = 0;     // CMX_STOP_AT: stop after N coder bytes (0 = never)
inline int g_bytecost = 0;          // CMX_BYTECOST: per-byte code length dump ./bytecost.f32
inline int g_tfoff = 0;             // CMX_TFOFF: 1 transformer row off, 2 also BITLSTM's transformer state
inline int g_semoff = 0;            // CMX_SEMCACHE_OFF: SEMCACHE abstains at every byte
// CMX_AT=p1,p2,...: extra CMX lines at these coder positions (checkpoints of a differencing A/B, e.g. 20000000)
// CMX_PAGES=k1,k2,...: a CMXMARK line page<k> where page k of the order begins (page 0 = text_start)
constexpr int kMaxList = 64;
inline long long g_at[kMaxList], g_pages[kMaxList];
inline int g_at_n = 0, g_pages_n = 0;

// ---- lever switches (2026-10-31 batch; defaults = release) ----------------------------------------------------
// CMX_MIXROW: layer-0 mixer rows.  0 release (10,000 rows first-come per mixer, then the shared context_base_ row);
// 1 at page 1 (first page behind the text start) every layer-0 mixer releases its rows and re-keys the SAME row
// memory first-come again, rows zeroed; 2 as 1, the re-keyed rows start as a copy of context_base_; 3 limit 65,536
// rows from byte 0 and every new layer-0 row starts as a copy of context_base_.
inline int g_mixrow = 0;
inline unsigned long long g_mix_limit = 10000;
inline int g_mixtel = 0;            // CMX_MIXTEL=1: fallback-row bits per layer-0 mixer (zero-byte telemetry)
// CMX_ROWDECAY (driver 11-01): the layer-0 learning-rate staircase per row instead of per mixer.  1 = the row's own
// clock: global steps since the later of the last restart and the row's creation (old rows as released, a new row
// starts at 1.0 -> 0.7 -> 0.3 -> 0.1 after 16 M / 80 M / 400 M steps of its own); 2 = the row's own update count with
// the same thresholds.  g_rd_epoch counts the restarts (Predictor::MetalrBlockEnde).
inline int g_rowdecay = 0;
inline uint32_t g_rd_epoch = 0;
inline unsigned long long g_fb[4][24], g_fbn[4];   // per region: primed bits on the fallback row, primed bits
// CMX_FX_DECCODE / CMX_FX_VERB: fxcm's stream-position constants (x.blpos = bytes fxcm has seen, injected ones
// included): deccode = lastCW only below DECCODE (release 448,131,719); verb typing only below VERB (451,531,986).
// CMX_FX_APM_OFF: offset added to the APM-rate / sscm thresholds 14*256*1024 and 28*512*1024 (release 0).
inline unsigned long long g_fx_deccode = 448131719ULL;
inline unsigned long long g_fx_verb = 451531986ULL;
inline unsigned long long g_fx_apm_off = 0;
// CMX_MIXSKIP_L0 / CMX_MIXSKIP_L1 (x 1/4096): skip a mixer update when the mixer's own prediction error
// |sigma(p) - bit| < e (paq8px v212/213 error limit); L0 = the 563-input layer-0 mixers, L1 = the final mixers.
inline float g_skip0 = 0.0f, g_skip1 = 0.0f;
inline bool g_skip_on = false;
// CMX_MIXSKIP_TEXT=1: the skip acts only from first_article on (telemetry region >= 2), driver 10-31 text-only arms
inline int g_skip_from = 0;
inline int g_region = 0;            // 0 < text_start <= 1 < first_article <= 2 < 20,000,000 <= 3 (telemetry)
inline unsigned long long g_upd[4][2], g_skp[4][2];   // updates seen / skipped per region and layer
// CMX_Z3LR=x: mixer learning-rate multiplier inside the census metadata block (meta_end .. the next
// MetalrBlockEnde = census_end), both layers (the side agent's C16 = 1.6)
inline float g_z3_mul = 1.0f;
inline bool g_z3_on = false;
// CMX_ZVAR: census number-coder variant bits (census_block::zahl::g_var, the side agent's NEIF_ZVAR):
// 2 V2 county value as a place feature (changes the tokens: the bed must be recoded with -r), 4 RK page metadata in
// front of each census run symbol (model side), 128 RF rounding-boundary class of rule values in the scale byte
// (model side), 256 RF without WAT_MI.
inline int g_zvar = 0;
// BITL_SEQ: BITLSTM state reset period in head bits (release 1024 = every window of the frozen head).  0 = no
// periodic reset, 4096 = every 4,096 bits; in both the state is also zeroed at a weight-set switch, and the switch
// itself stays on the 1,024-bit window grid (NextSlot waits for nbuf == 0 as released).
inline long long g_bitl_seq = 1024;
// CONST-BRACKET (driver 10-31): CMX_FX_KA = fxcm mixer error multiplier in per mille (fx_mixer_lr::kA, release
// 1000; scales err = ((y<<12) - pr) * uperr / 4 of every fxcm Mixer1 exactly by kA/1000); CMX_ADEPTH_TAU = the
// adaptive-depth threshold (release adaptive_depth::kTau = 7); CMX_FWK_MUL = factor on the fast-weight head's
// kappa_t = 1.5e-3 + 4.5e-3 / (1 + t / 8000) (both terms, release 1).
inline int g_fx_ka = 1000;
inline bool g_tau_set = false;
inline float g_tau = 0.0f;
inline double g_fwk_mul = 1.0;
// MATCH-AGE (driver 10-31): CMX_MATCH_AGE=1 adds the bit "the fxcm MatchModel2 source is more than 2,048 buffer bytes
// back" to its length context (ctx[0], StateMap 512 -> 1024 entries); =2 instead keys the layer-0 mixer that
// selects by longest_match_ with the MtcSel length bucket of that match (mixer_selectors.h) x the same bit.
inline int g_match_age = 0;
inline unsigned int g_mtc_far = 0;
inline unsigned long long g_mtc_ctx = 0;
// WL-MIXGRAD (driver 10-31): CMX_WL_MIXGRAD=kappa (set, even 0, = the moved path): the word-label rows train on
// g' = g + kappa * delta_wl, delta_wl = g_final * sum_i gamma_i * W0_i[j_wl] (E2EGrad, pre-update rows), and
// word_label's Perceive runs behind the mixer cascade (delta 0 on bits that skip the cascade).
inline bool g_wl_on = false;
inline float g_wl_kappa = 0.0f;
inline unsigned int g_wl_idx = 0;
inline float g_wl_delta = 0.0f;
// CMX_WL_TEXT=1 (driver 11-07, condition met by wl0.3): the kappa term only from census_end (= text_start_begin) on;
// before it the word-label rows train on their own gradient alone (still on the moved path)
inline bool g_wl_text = false;
inline bool g_wl_live = true;
// FXCM-LAZYRESET (driver 10-31): CMX_FX_LAZY=1 replaces the per-page memset of fxcm's context maps by a per-bucket
// generation tag (a bucket is zeroed at its first access after the reset; bit-identical by construction).
// CMX_FX_TICKS=1 counts the rdtsc cycles of the page-reset block and the lazy zeroings (zero-byte).
inline int g_fx_lazy = 0, g_fx_ticks = 0;
inline unsigned long long g_fx_resets = 0, g_fx_reset_cyc = 0, g_fx_lazy_zero = 0;
// CMX_SEMDUMP=<dir> (driver 10-31, zero-byte observer): one 58-byte record per coder position in
// [CMX_SEMDUMP_FROM, CMX_SEMDUMP_TO) (default 13,020,000 .. 37,600,000) with SEMCACHE's prediction for the byte at
// that position, written to <dir>/semcache_records.bin; <dir>/pages.u32 = (page index, first record) pairs from the
// text start on; <dir>/meta.json = layout and region boundaries.  Symbols are the transformer's 208-way token index
// (transformer::g_b2i, the mapping of its PPMd prior rows).
struct __attribute__((packed)) SemRec {
  uint8_t flags;      // bit0 valid (>= 1 neighbour, fresh or followed), bit1 lookup ran, bit2 cache top-1 ==
                      // transformer top-1, bit3 stored (entry written at the end of this byte)
  uint8_t nfresh;     // fresh neighbours of the lookup (<= 32)
  uint8_t nnb;        // neighbours used, fresh + followed (<= 160)
  uint8_t nsym;       // distinct symbols among them
  int16_t sim;        // highest cosine of the lookup (or of the newest followed lookup) x 32767
  uint16_t hrate;     // SEMCACHE's running top-1 hit rate x 65535
  uint16_t resid;     // weight mass outside the 16 listed symbols x 65535
  uint8_t sym[16];    // the 16 heaviest symbols (token index 0..207), heaviest first, ties to the lower byte
  uint16_t prob[16];  // their weight share x 65535 (exp((s - s_max) / T) weights as SEMCACHE reads them)
};
static_assert(sizeof(SemRec) == 58, "SemRec layout");
inline const char* g_semdump = nullptr;
inline long long g_sd_from = 13020000, g_sd_to = 37600000;
inline SemRec g_semrec;
// F8 (driver 10-31): CMX_FW_BLOCKS = first MLP block of the fast-weight head (release 12 = blocks 12..15; 8 =
// blocks 8..15; 4 = blocks 4..15).  CMX_FW_TICKS=1: rdtsc cycles in fw_apply / fw_head / fw_update (zero-byte).
inline int g_fw_first = 12;
inline int g_fw_ticks = 0;
inline unsigned long long g_fw_cyc_apply = 0, g_fw_cyc_head = 0, g_fw_cyc_update = 0, g_fw_n_update = 0;
// CMX_MATCHDUMP=<dir> (driver 10-31, zero-byte observer): one 8-byte record per coder position in the CMX_SEMDUMP
// span [CMX_SEMDUMP_FROM, CMX_SEMDUMP_TO): the match predictions available before the byte is coded (captured at
// the byte's first bit, before it is coded): cmix's longest of its five match models and fxcm's MatchModel2 best
// candidate.  <dir>/match_records.bin, pages.u32, meta.json.  CMX_BYTECOST_PATH=<file>: bytecost.f32 there.
struct __attribute__((packed)) MatchRec {
  uint8_t flags;      // bit0 cmix match length > 0, bit1 fxcm candidate length > 0, bit2 both present and the same
                      // symbol, bit3 fxcm in delta mode, bits 4..6 index (0..4) of the longest cmix model
  uint8_t src;        // the stronger source: 0 none, 1 cmix (length/8 >= fxcm length), 2 fxcm
  uint8_t cm_sym;     // cmix predicted byte as token index 0..207 (255: none / outside the set)
  uint8_t cm_len;     // its verified length in BITS (Match::match_length_, 0..255)
  uint8_t fx_sym;     // fxcm expected byte as token index (255: no candidate)
  uint8_t fx_len;     // fxcm rebased candidate length (1 = shortest accepted match), capped at 255
  uint16_t cm_conf;   // cmix model's P(the first bit follows the prediction) x 65535 (predictions_[length])
};
static_assert(sizeof(MatchRec) == 8, "MatchRec layout");
inline const char* g_matchdump = nullptr;
inline bool g_md_arm = false;     // armed by the runner before a stream byte's bits; fxcm and Predict capture once
inline MatchRec g_mdrec;
inline const char* g_bytecost_path = nullptr;
// SEMKEY (driver 11-01): CMX_SEM_KCEN0=1 restarts SEMCACHE's running key mean at first_article; CMX_SEM_KCEN = the
// number of counted lookups after which the mean freezes (release kCen = 2^20).
inline int g_sem_kcen0 = 0;
inline uint32_t g_sem_kcen = 0;     // 0 = semantic_cache::kCen
// CMX_SEMTOP (driver 11-07, interp phase 3 idea 2): stages 1 (kUC4) and 3 (kUC6) of SEMCACHE's end stage keyed on the
// vote's top share t = acc[top-1] / wsum (the largest normalized vote weight over all neighbours, fresh and followed;
// SemByteStart, from entries stored >= kGap stream bytes back and followed continuations before this byte) instead of
// / besides the similarity class of s_max (kC1E3 / kC2E3).  The cells keep level 0 (no consistent neighbour for the
// bit: class and agreement dropped) and the agreement bit; the class levels 1..L replace the similarity levels 1..3:
// 1 = share class q4 (t < 0.4, < 0.8, < 0.95, >= 0.95; L 4), 2 = similarity x q4 (L 12), 3 = share class q3 (t < 0.4,
// < 0.8, >= 0.8; L 3, the release geometry).  Stages 1 and 3 then read wide tables on the heap ((L + 1) x 12 and
// (L + 1) x 128 cells per bit position; release 48 / 512).  0 = release.
inline int g_semtop = 0;
inline int g_st_lv = 3;                  // class levels besides level 0 (release: the 3 similarity classes)
inline int g_st_s1 = 48, g_st_s3 = 512;  // cells per bit position of stages 1 / 3
inline unsigned long long g_st_cnt[4][13];   // bytes with neighbours by class 1..L, per region (zero-byte)
inline int SemTopLevels(int m) { return m == 1 ? 4 : (m == 2 ? 12 : 3); }
inline int SemTopShare(double top, double wsum, int m) {   // 0..3 (m 1, 2) or 0..2 (m 3)
  const double t = wsum > 0.0 ? top / wsum : 0.0;
  if (m == 3) return t < 0.4 ? 0 : (t < 0.8 ? 1 : 2);
  return t < 0.4 ? 0 : (t < 0.8 ? 1 : (t < 0.95 ? 2 : 3));
}
// the class 1..L in place of the similarity class cls (1..3) of the cells
inline int SemTopClass(int cls, int q, int m) { return m == 2 ? 1 + (cls - 1) * 4 + q : 1 + q; }
// CMX_ADEPTH_MODE (driver 11-07, interp phase 3 idea 4): the adaptive-depth gate (|z| > tau) split into its coding and
// its learning half.  0 release (a gate bit is coded with sigma(z) and the cascade neither runs nor learns on it).
// 1 learn-skip only: the cascade runs and codes every bit (as tau = inf), but on gate bits its learners stand still as
// in release: no E2EGrad / layer-0 step / EndLearn (Bayes term) / final-mixer step (PerceiveSkip), SSE context only, no
// cvz step, and the BITLSTM is neither applied (no Adjust: the coder takes the cascade's p) nor stepped (no Observe).
// 2 code-skip only: gate bits are coded with sigma(z) as in release, but the cascade runs forward and learns on every
// bit, and the BITLSTM runs Adjust (on the cascade's p, output not coded) and Observe on every bit.
// In both modes the gate trains on every bit as released; the floor stage and the top-1 path stage see "skipped" only
// where the coder takes sigma(z).
inline int g_adm = 0;
inline unsigned long long g_adm_bits[4], g_adm_gate[4];   // non-forced bits, gate bits per region (zero-byte)
// CMX_ADEPTH_TAU_CODE=x (driver 11-08, split tau; ignored while CMX_ADEPTH_MODE is set): a bit is CODED with sigma(z)
// iff |z| > tau_code, and the cascade's learners (and the BITLSTM) stand still iff |z| > tau_learn (= CMX_ADEPTH_TAU,
// release 7).  The band tau_code < |z| <= tau_learn is coded with sigma(z) while the cascade runs and learns (mode 2
// on the band); the band tau_learn < |z| <= tau_code is coded by the cascade, which does not learn (mode 1 on the band).
// The gate trains as released.  Unset = release; tau_code = tau_learn codes as release.
inline bool g_tauc_set = false;
inline float g_tauc = 0.0f;
inline unsigned long long g_adm_cgate[4];   // bits coded with sigma(z) per region (zero-byte)
// CMX_SEMIN (driver 11-08, vote reliability as continuous inputs, no new cells): x1 = t * st(p_sem) and x2 = agree *
// st(p_sem), t = the vote's top share (as CMX_SEMTOP computes it), agree = cache top-1 == transformer top-1 (s.agr),
// st(p_sem) = the cache's log-odds of the bit (end stage: f[1] = SemPredBit's s.rx; final mixer: the cache column);
// both 0 without a vote or without a consistent neighbour.  New weights start at 0.
// 1 = two more inputs of end-stage stage 1 (kUC4 rows); 2 = 1 and two more final-mixer columns (layer-1 row 24, 25);
// 3 = two more inputs of all three end-stage stages (kUC4, kUC5, kUC6 rows), no final-mixer columns.
inline int g_semin = 0;
inline bool g_semx = false;   // CMX_SEMTOP or CMX_SEMIN: the end stage takes the research tail (predictor.cpp)
// CMX_SEMIN_SELFTEST=1 (with CMX_SEMIN set; a test run, not a coding run): at the first stream byte the production
// end-stage tail, its learning step and (CMX_SEMIN=2) the final-mixer columns are driven on a synthetic vote, the
// checks are printed as "CMXTEL semin selftest" lines, and the process ends (exit 0 = all hold, 3 = a failure)
inline int g_semin_st = 0;
// CMX_SEMKEY2=L (driver 11-08, ideas round 3 idea 5): a second SEMCACHE instance beside the release one, keyed on the
// mixer read of transformer layer L (read index q = 2L: the depth-mixed residual-stream input of block L, 1..15;
// 0 = release) instead of the MLP read of the last layer.  Its own running key mean, its own seeded +-1 projection
// 192 -> 128 and SimHash / sketch hyperplanes (seed 0x5EC0CAC4E2 ^ 0x9E3779B97F4A7C15), its own key ring, bucket table,
// followers, hit rate and end stage; release constants and gates (both caches key the same bytes).
// CMX_SEMKEY2_USE (bit mask, default 3): 1 = its per-bit log-odds is one more layer-0 input (row 563 -> 564) and one
// more final-mixer column behind the row (behind CMX_SEMIN=2's two); 2 = its end stage runs behind the release end
// stage (before the top-1 path stage); 0 = shadow (it runs and reports, nothing reads it: the coding is the
// release's).  Independent of CMX_SEMCACHE_OFF, CMX_SEM_KCEN*, CMX_SEMTOP, CMX_SEMIN and CMX_SEMDUMP (release cache).
// CMX_SEMKEYS=L1,L2,... (driver 11-08, generalizes CMX_SEMKEY2): up to kSkMax extra caches, cache k (1..n) keyed on the
// mixer read of layer Lk (1..15; a layer may repeat), each a full independent instance as above: its own key copy,
// State, tables (SEMCONST constants, as the release cache), seed 0x5EC0CAC4E2 ^ k * 0x9E3779B97F4A7C15 (k = 1: the
// CMX_SEMKEY2 seed), layer-0 input 562 + k (row 563 + n) and final-mixer column (behind the row and CMX_SEMIN=2's two,
// in order k) with zero-init weights, and end stage in series k = 1..n behind the release end stage (before the top-1
// path stage).  CMX_SEMKEY2_USE applies to every extra cache.  CMX_SEMKEY2=L is the alias CMX_SEMKEYS=L (bit for bit);
// both set (non-zero) is an error.  "0" or empty = release.
// List element R (driver 11-08): an extra cache keyed on the release key itself (tf1_engine_->LastKey(), the last
// layer's MLP read, q = 31, what the release SEMCACHE keys on) with that cache's own seed (k-th seed rule), so its own
// +-1 projection, SimHash index, sketch planes and end stage; e.g. CMX_SEMKEYS=R,12.  Stored as layer kSkRel.
constexpr int kSkMax = 3;
constexpr int kSkRel = 16;                  // g_sk_layer value of an R element (no layer copy: g_sk_lmask bit unset)
inline int g_semkey2 = 0;                   // CMX_SEMKEY2 as given (CMXCFG)
inline int g_nsk = 0;                       // extra caches (0 = release)
inline int g_sk_layer[kSkMax] = {-1, -1, -1};
inline unsigned int g_sk_lmask = 0;         // bit l: model_opt.cpp copies the mixer read of layer l into g_sk_key[l]
inline char g_sk_list[48] = "-";            // the layers as a list (CMXCFG)
inline char g_sk_name[kSkMax][4];           // per cache: "8" or "R" (CMXTEL)
inline int g_sk2_use = 3;
inline bool g_sk2_mix = false, g_sk2_end = false;
inline float g_sk_key[16][192];             // per layer: the key of the last read row (model_opt.cpp zb_layers; one engine)
inline unsigned long long g_sk2_cnt[kSkMax][4][12];  // per cache and region (zero-byte), Predictor::CmxSem2Report
// CMX_SEMTEL=1: SEMCACHE telemetry (zero-byte)
inline int g_semtel = 0;
constexpr int kAgeBins = 33;
inline unsigned long long g_sem_store[4], g_sem_evict[4][kAgeBins], g_sem_look[4], g_sem_look_nk[4],
    g_sem_look_nb[4], g_sem_nage[4][kAgeBins], g_sem_keyed[4];
inline unsigned long long g_sem_hyp1[16 * 18], g_sem_hypn = 0;
inline double g_sem_ksum[192], g_sem_ksq = 0.0;
inline unsigned long long g_sem_kn = 0;
inline int Log2Bin(unsigned long long a) { int b = 0; while (a > 0 && b < kAgeBins - 1) { a >>= 1; ++b; } return b; }

// ---- CM census, phase 2a (driver 11-08; experiments/plan_cm_time_census_20261108.md) ------------------------------
// CMX_FXLS=1 (lever, plan §5.2 #1c): no fxcm-internal mixer / APM training on adaptive-depth gate bits.  A gate bit is
// a bit on which the cascade's learners stand still (Predictor's adm_loff_: |z| > tau, = adepth_.skip while
// CMX_ADEPTH_MODE / CMX_ADEPTH_TAU_CODE are unset; a vocabulary-forced bit counts by the same flag).  On such a bit
// mxA[0..16], mxA1[0..5] and mxA2[0..3] skip Mixer1::update, and the next predict_half skips the step of the six APMs
// (apmA0..5) and of the mmmO[0..3] chain for it.  Everything else of fxcm learns as released: context maps, state maps,
// the DirectStateMaps with their Mix chains, the elim / fails bookkeeping.  0 = release.
inline int g_fxls = 0;
inline bool g_fxls_off = false;    // the bit fxcm learns from next is a gate bit (Predictor::Perceive, before fxcm)
inline bool g_fxls_pend = false;   // its APM / mmmO steps (taken by the next predict_half) are skipped
inline unsigned long long g_fxls_n[4][2];   // fxcm mixer learning steps taken / skipped per region (zero-byte)
// CMX_FWOFF=1 (lever, plan §5.2 #10): the fast-weight head off.  No fw_apply / fw_head (model_opt.cpp) and no
// fw_update (Tf1Engine never sees a live fast-weight row); on a byte with a live transformer row the fast-weight slot
// takes that row itself, which is exactly what the second head outputs with zero fast weights (dW = 0: x + dW h = x),
// so the layer-0 input, the final-mixer column, the Bayes expert and the frozen BITLSTM's FW column stay inside the
// release value range (a copy of the transformer slot).  Gated / neutral bytes as released.  0 = release.
inline int g_fwoff = 0;
// CMX_CM_TICKS=1|2 (zero-byte observer, plan §3.2): rdtsc cycles per CM-side pipeline section (CkSec) and telemetry
// region, by a lap timer (one rdtsc per section boundary, no lfence); 2 = also every fxcm mix() site and fxcm mixer
// stage (g_fxk) and the PPMd byte split.  The report gives tsc_hz (TSC against CLOCK_MONOTONIC since Read()) and the
// median back-to-back rdtsc delta (ovh) for the overhead correction.  Section VALTEL is the CMX_VALTEL work, measured
// inside E2E (the report prints E2E without it).
inline int g_cm_ticks = 0;
#define CMX_CK_LIST(X) X(LOOP) X(SEMBYTE) X(AFTER) X(FXPRED) X(SEL) X(CMIXP) X(FXCOPY) X(SLOTP) X(GATE) X(PRIME) \
  X(BANK) X(FINAL) X(SSE) X(HEAD) X(ENDST) X(LEARN) X(WLU) X(CMIXU) X(SLOTU) X(E2E) X(VALTEL) X(PRECOND) X(BANKU) \
  X(FINALU) X(GATEU) X(CTX) X(CMIXB) X(PPMDB) X(TF) X(TFROW) X(SLOTB) X(FXUPD) X(PPMD_UPD) X(PPMD_PREP) X(PPMD_FILL)
enum CkSec : int {
#define CMX_CK_ENUM(n) CK_##n,
  CMX_CK_LIST(CMX_CK_ENUM)
#undef CMX_CK_ENUM
  CK_N
};
inline const char* const kCkName[CK_N] = {
#define CMX_CK_STR(n) #n,
  CMX_CK_LIST(CMX_CK_STR)
#undef CMX_CK_STR
};
inline unsigned long long g_ck[4][CK_N], g_ckn[4][CK_N];   // cycles and laps per region and section
inline unsigned long long g_ck_tsc0 = 0, g_ck_ovh = 0;
inline double g_ck_t0 = 0.0;
inline unsigned long long CkNow() { return __builtin_expect(g_cm_ticks != 0, 0) ? __rdtsc() : 0ull; }
// the section s ends now: its cycles since t, and t restarts
inline void CkLap(int s, unsigned long long& t) {
  if (__builtin_expect(g_cm_ticks != 0, 0)) {
    const unsigned long long n = __rdtsc();
    g_ck[g_region][s] += n - t;
    ++g_ckn[g_region][s];
    t = n;
  }
}
inline void CkLap2(int s, unsigned long long& t) {   // level-2 sections only
  if (__builtin_expect(g_cm_ticks >= 2, 0)) CkLap(s, t);
}
#define CMXK_START(t) unsigned long long t = cmx_meas::CkNow();
#define CMXK_LAP(sec, t) cmx_meas::CkLap(cmx_meas::sec, t);
#define CMXK_LAP2(sec, t) cmx_meas::CkLap2(cmx_meas::sec, t);
// fxcm sites (CMX_CM_TICKS=2 cycles, and the input layout for CMX_VALTEL): ids and names come from fxcmv1.cpp (FxSite)
constexpr int kFxN = 96;
inline unsigned long long g_fxk[4][kFxN], g_fxkn[4][kFxN];
inline unsigned g_fxlay[kFxN][4];         // [prediction lo, hi) and [mxInputs1 lo, hi) of the site's last call
inline const char* g_fxname[kFxN];
inline unsigned long long g_fx_trn[4][2];   // mxA Mixer1::update calls, of them with err != 0 (level 2)
inline bool g_fxrec = false;              // fxcm records its sites (CMX_CM_TICKS=2 or CMX_VALTEL)
// CMX_VALTEL=k (zero-byte observer, plan §4): every k-th cascade bit (E2EGrad, pre-update rows) the frozen-weight
// occlusion of every model group: dz_G = -sum_{j in G} x_j g_j, g_j = sum_i gamma_i W0_i[j] (+ the direct final-mixer
// column for fxcm aux / transformer / FW / SEMCACHE), plus the exact change of the Bayes term for its experts;
// occl_bits = sum (l(z + dz_G) - l(z)) x k, l = -log2 P(coded bit).  occl_fx_bits adds the first-order change of the
// mxA[0..16] outputs (fxcm-internal path).  CMX_VALTEL_GROUPS="name=a+b+...,..." adds composite groups.
inline int g_valtel = 0;
inline const char* g_valtel_groups = nullptr;
// back-to-back rdtsc median and the TSC / clock origin (Read(), with CMX_CM_TICKS)
inline void CkCalibrate() {
  unsigned long long d[1001];
  for (int i = 0; i < 1001; ++i) { const unsigned long long a = __rdtsc(); d[i] = __rdtsc() - a; }
  for (int i = 1; i < 1001; ++i) {
    const unsigned long long v = d[i];
    int j = i;
    while (j > 0 && d[j - 1] > v) { d[j] = d[j - 1]; --j; }
    d[j] = v;
  }
  g_ck_ovh = d[500];
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  g_ck_tsc0 = __rdtsc();
  g_ck_t0 = (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
inline double CkTscHz() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const unsigned long long n = __rdtsc();
  const double dt = (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec - g_ck_t0;
  return dt > 0.0 ? (double)(n - g_ck_tsc0) / dt : 0.0;
}

// runner hooks needed (observers, the page-1 re-key, the census LR window, region tracking)
inline bool g_hook = false;

inline long long Env(const char* name) {
  const char* v = getenv(name);
  return v ? atoll(v) : 0;
}
inline double EnvD(const char* name, double dflt) {
  const char* v = getenv(name);
  return v ? atof(v) : dflt;
}
// a comma-separated list of positive numbers, sorted ascending (insertion sort; at most kMaxList)
inline int EnvList(const char* name, long long* out) {
  const char* v = getenv(name);
  int n = 0;
  while (v && *v && n < kMaxList) {
    char* e = nullptr;
    const long long x = strtoll(v, &e, 10);
    if (e == v) break;
    if (x > 0) {
      int j = n++;
      while (j > 0 && out[j - 1] > x) { out[j] = out[j - 1]; --j; }
      out[j] = x;
    }
    v = (*e == ',') ? e + 1 : e;
  }
  return n;
}
// the observers (any of them prints the CMXMARK lines)
inline bool Observe() { return g_log_every > 0 || g_stop_at > 0 || g_bytecost != 0 || g_at_n > 0 || g_pages_n > 0; }

// main(), top-level -e or -n only: read the environment once and print what is set
inline void Read() {
  g_log_every = Env("CMX_LOG_EVERY");
  g_stop_at = Env("CMX_STOP_AT");
  g_bytecost = (int)Env("CMX_BYTECOST");
  g_tfoff = (int)Env("CMX_TFOFF");
  g_semoff = (int)Env("CMX_SEMCACHE_OFF");
  g_at_n = EnvList("CMX_AT", g_at);
  g_pages_n = EnvList("CMX_PAGES", g_pages);
  if (g_log_every < 0) g_log_every = 0;
  if (g_stop_at < 0) g_stop_at = 0;
  g_mixrow = (int)Env("CMX_MIXROW");
  if (g_mixrow < 0 || g_mixrow > 3) g_mixrow = 0;
  if (g_mixrow == 3) g_mix_limit = 65536;
  g_mixtel = (int)Env("CMX_MIXTEL");
  g_rowdecay = (int)Env("CMX_ROWDECAY");
  g_sem_kcen0 = (int)Env("CMX_SEM_KCEN0");
  g_sem_kcen = (uint32_t)Env("CMX_SEM_KCEN");
  if (getenv("CMX_FX_DECCODE")) g_fx_deccode = (unsigned long long)Env("CMX_FX_DECCODE");
  if (getenv("CMX_FX_VERB")) g_fx_verb = (unsigned long long)Env("CMX_FX_VERB");
  g_fx_apm_off = (unsigned long long)Env("CMX_FX_APM_OFF");
  g_skip0 = (float)(EnvD("CMX_MIXSKIP_L0", 0.0) / 4096.0);
  g_skip1 = (float)(EnvD("CMX_MIXSKIP_L1", 0.0) / 4096.0);
  g_skip_on = g_skip0 > 0.0f || g_skip1 > 0.0f;
  g_skip_from = Env("CMX_MIXSKIP_TEXT") ? 2 : 0;
  g_z3_mul = (float)EnvD("CMX_Z3LR", 1.0);
  g_zvar = (int)Env("CMX_ZVAR");
  g_semtel = (int)Env("CMX_SEMTEL");
  g_semtop = (int)Env("CMX_SEMTOP");
  if (g_semtop < 0 || g_semtop > 3) g_semtop = 0;
  if (g_semtop) {
    g_st_lv = SemTopLevels(g_semtop);
    g_st_s1 = (g_st_lv + 1) * 12;    // (levels incl. 0) x agreement x word start x age class
    g_st_s3 = (g_st_lv + 1) * 128;   // (levels incl. 0) x agreement x previous-byte class x run class
  }
  g_adm = (int)Env("CMX_ADEPTH_MODE");
  if (g_adm < 0 || g_adm > 2) g_adm = 0;
  if (getenv("CMX_ADEPTH_TAU_CODE")) { g_tauc_set = true; g_tauc = (float)EnvD("CMX_ADEPTH_TAU_CODE", 7.0); }
  g_semin = (int)Env("CMX_SEMIN");
  if (g_semin < 0 || g_semin > 3) g_semin = 0;
  g_semx = g_semtop != 0 || g_semin != 0;
  g_semin_st = (g_semin && Env("CMX_SEMIN_SELFTEST")) ? 1 : 0;
  g_semkey2 = (int)Env("CMX_SEMKEY2");
  if (getenv("CMX_SEMKEY2_USE")) g_sk2_use = (int)Env("CMX_SEMKEY2_USE");
  if (g_semkey2 < 0 || g_semkey2 > 15 || g_sk2_use < 0 || g_sk2_use > 3) {
    fprintf(stderr, "CMX_SEMKEY2=%d / CMX_SEMKEY2_USE=%d: a layer 1..15 (0 = release) and a mask 0..3\n", g_semkey2,
            g_sk2_use);
    exit(2);
  }
  {   // CMX_SEMKEYS=L1,L2,...: strict list of decimal layers 1..15 or R, at most kSkMax ("0" or empty = release)
    const char* v = getenv("CMX_SEMKEYS");
    int n = 0;
    bool bad = false;
    if (v && *v && strcmp(v, "0") != 0) {
      const char* q = v;
      while (!bad) {
        const char* e = nullptr;
        long x = 0;
        if (*q == 'R') {
          x = kSkRel;
          e = q + 1;
        } else {
          if (*q < '0' || *q > '9') { bad = true; break; }
          char* ee = nullptr;
          x = strtol(q, &ee, 10);
          e = ee;
          if (x < 1 || x > 15) { bad = true; break; }
        }
        if (n >= kSkMax) { bad = true; break; }
        g_sk_layer[n++] = (int)x;
        if (*e == 0) break;
        if (*e != ',') { bad = true; break; }
        q = e + 1;
      }
      if (!bad && g_semkey2 != 0) {
        fprintf(stderr, "CMX_SEMKEY2=%d and CMX_SEMKEYS=%s: set one of them (CMX_SEMKEY2=L is CMX_SEMKEYS=L)\n",
                g_semkey2, v);
        exit(2);
      }
    } else if (g_semkey2 != 0) {
      g_sk_layer[n++] = g_semkey2;   // the alias: one extra cache
    }
    if (bad) {
      fprintf(stderr, "CMX_SEMKEYS=%s: a comma-separated list of 1..%d elements, each a layer 1..15 or R (the release "
              "key) (\"0\" or empty = release)\n", v, kSkMax);
      exit(2);
    }
    g_nsk = n;
    g_sk_lmask = 0;
    int w = 0;
    for (int k = 0; k < n; ++k) {
      if (g_sk_layer[k] == kSkRel) snprintf(g_sk_name[k], sizeof(g_sk_name[k]), "R");
      else {
        g_sk_lmask |= 1u << g_sk_layer[k];
        snprintf(g_sk_name[k], sizeof(g_sk_name[k]), "%d", g_sk_layer[k]);
      }
      w += snprintf(g_sk_list + w, sizeof(g_sk_list) - (size_t)w, k ? ",%s" : "%s", g_sk_name[k]);
    }
  }
  g_sk2_mix = g_nsk != 0 && (g_sk2_use & 1) != 0;
  g_sk2_end = g_nsk != 0 && (g_sk2_use & 2) != 0;
  if (getenv("BITL_SEQ")) g_bitl_seq = Env("BITL_SEQ");
  if (getenv("CMX_FX_KA")) g_fx_ka = (int)Env("CMX_FX_KA");
  if (g_fx_ka <= 0) g_fx_ka = 1000;
  if (getenv("CMX_ADEPTH_TAU")) { g_tau_set = true; g_tau = (float)EnvD("CMX_ADEPTH_TAU", 7.0); }
  g_fwk_mul = EnvD("CMX_FWK_MUL", 1.0);
  g_match_age = (int)Env("CMX_MATCH_AGE");
  if (getenv("CMX_WL_MIXGRAD")) { g_wl_on = true; g_wl_kappa = (float)EnvD("CMX_WL_MIXGRAD", 0.0); }
  g_wl_text = Env("CMX_WL_TEXT") != 0;
  if (g_wl_text) g_wl_live = false;   // set at census_end (runner.cpp AfterByte)
  g_fx_lazy = (int)Env("CMX_FX_LAZY");
  if (getenv("CMX_FW_BLOCKS")) g_fw_first = (int)Env("CMX_FW_BLOCKS");
  if (g_fw_first < 4 || g_fw_first > 12) g_fw_first = 12;
  g_fw_ticks = (int)Env("CMX_FW_TICKS");
  g_matchdump = getenv("CMX_MATCHDUMP");
  if (g_matchdump && !*g_matchdump) g_matchdump = nullptr;
  g_bytecost_path = getenv("CMX_BYTECOST_PATH");
  g_semdump = getenv("CMX_SEMDUMP");
  if (g_semdump && !*g_semdump) g_semdump = nullptr;
  if (getenv("CMX_SEMDUMP_FROM")) g_sd_from = Env("CMX_SEMDUMP_FROM");
  if (getenv("CMX_SEMDUMP_TO")) g_sd_to = Env("CMX_SEMDUMP_TO");
  g_fx_ticks = (int)Env("CMX_FX_TICKS");
  g_fxls = (int)Env("CMX_FXLS");
  g_fwoff = (int)Env("CMX_FWOFF");
  g_cm_ticks = (int)Env("CMX_CM_TICKS");
  g_valtel = (int)Env("CMX_VALTEL");
  g_valtel_groups = getenv("CMX_VALTEL_GROUPS");
  if (g_valtel_groups && !*g_valtel_groups) g_valtel_groups = nullptr;
  if (g_fxls < 0 || g_fxls > 1 || g_fwoff < 0 || g_fwoff > 1 || g_cm_ticks < 0 || g_cm_ticks > 2 || g_valtel < 0) {
    fprintf(stderr, "CMX_FXLS=%d / CMX_FWOFF=%d (0 or 1), CMX_CM_TICKS=%d (0..2), CMX_VALTEL=%d (>= 0)\n", g_fxls, g_fwoff,
            g_cm_ticks, g_valtel);
    exit(2);
  }
  if (g_valtel_groups && !g_valtel) {
    fprintf(stderr, "CMX_VALTEL_GROUPS needs CMX_VALTEL\n");
    exit(2);
  }
  g_fxrec = g_cm_ticks >= 2 || g_valtel > 0;
  if (g_cm_ticks) CkCalibrate();
  if (g_bitl_seq < 0) g_bitl_seq = 1024;
  g_hook = Observe() || g_mixrow == 1 || g_mixrow == 2 || g_z3_mul != 1.0f || g_mixtel || g_semtel || g_skip_on ||
           g_sem_kcen0 || g_wl_text || g_semtop || g_adm || g_tauc_set || g_semin || g_nsk ||
           g_semdump != nullptr || g_matchdump != nullptr || g_fxls || g_cm_ticks || g_valtel;
  if (Observe() || g_tfoff || g_semoff || g_mixrow || g_mixtel || g_fx_deccode != 448131719ULL ||
      g_fx_verb != 451531986ULL || g_fx_apm_off || g_skip_on || g_z3_mul != 1.0f || g_zvar || g_semtel ||
      g_bitl_seq != 1024 || g_fx_ka != 1000 || g_tau_set || g_fwk_mul != 1.0 || g_match_age || g_wl_on ||
      g_fx_lazy || g_fx_ticks || g_semdump || g_fw_first != 12 || g_fw_ticks || g_matchdump || g_rowdecay ||
      g_sem_kcen0 || g_sem_kcen || g_wl_text || g_semtop || g_adm || g_tauc_set || g_semin || g_nsk || g_fxls ||
      g_fwoff || g_cm_ticks || g_valtel) {
    fprintf(stderr, "CMXCFG log_every=%lld stop_at=%lld bytecost=%d tfoff=%d semcache_off=%d at=%d pages=%d"
            " mixrow=%d limit=%llu mixtel=%d fx_deccode=%llu fx_verb=%llu fx_apm_off=%llu skip0=%.9g skip1=%.9g"
            " z3lr=%.9g zvar=%d semtel=%d bitl_seq=%lld fx_ka=%d tau=%s%.9g fwk_mul=%.9g match_age=%d"
            " wl_mixgrad=%s%.9g fx_lazy=%d fx_ticks=%d skip_from_region=%d semdump=%s[%lld,%lld) fw_first_block=%d"
            " fw_ticks=%d matchdump=%s bytecost_path=%s rowdecay=%d sem_kcen0=%d sem_kcen=%u wl_text=%d"
            " semtop=%d semtop_levels=%d semtop_cells=%d/%d adepth_mode=%d tau_code=%s%.9g semin=%d"
            " semin_selftest=%d semkey2=%d semkey2_use=%d semkeys=%s fxls=%d fwoff=%d cm_ticks=%d valtel=%d valtel_groups=%s\n",
            g_log_every, g_stop_at, g_bytecost, g_tfoff, g_semoff, g_at_n, g_pages_n, g_mixrow, g_mix_limit, g_mixtel,
            g_fx_deccode, g_fx_verb, g_fx_apm_off, (double)g_skip0, (double)g_skip1, (double)g_z3_mul, g_zvar,
            g_semtel, g_bitl_seq, g_fx_ka, g_tau_set ? "" : "release:", (double)g_tau, g_fwk_mul, g_match_age,
            g_wl_on ? "" : "off:", (double)g_wl_kappa, g_fx_lazy, g_fx_ticks, g_skip_from,
            g_semdump ? g_semdump : "-", g_sd_from, g_sd_to, g_fw_first, g_fw_ticks, g_matchdump ? g_matchdump : "-",
            g_bytecost_path ? g_bytecost_path : "-", g_rowdecay, g_sem_kcen0, (unsigned)g_sem_kcen, (int)g_wl_text,
            g_semtop, g_st_lv, g_st_s1, g_st_s3, g_adm, g_tauc_set ? (g_adm ? "ignored:" : "") : "off:",
            (double)g_tauc, g_semin, g_semin_st, g_semkey2, g_sk2_use, g_sk_list, g_fxls, g_fwoff, g_cm_ticks, g_valtel,
            g_valtel_groups ? g_valtel_groups : "-");
    fflush(stderr);
  }
}

#if PORT_ENTRY
// ---- PORT_ENTRY (src/port_cfg.h; Claude Fable 5.2, 2026-11-09): recipe "candidate A" as constants ----------------
// main() calls EntryApply() once, before any model object exists, in every mode that codes or decodes the main
// stream (never for a -D helper block, never in the build helper).  It writes EVERY knob the model code reads: the
// recipe values, and the release default of every other coding knob.  In the shipping form nothing else ever writes
// a knob (Read() is not called), so the second group restates the static initializers; in the test form
// (PORT_ENTRY_TEST) it undoes whatever Read() took from the environment.  The state it leaves is the one Read()
// leaves for CMX_SEMKEYS=R,12 CMX_FW_BLOCKS=4 BITL_SEQ=0 (CMX_SEMKEY2_USE unset = 3).
constexpr int kEntrySkN = 2;                       // CMX_SEMKEYS=R,12
constexpr int kEntrySkLayer[kSkMax] = {kSkRel, 12, -1};
constexpr int kEntrySk2Use = 3;                    // CMX_SEMKEY2_USE: 1 layer-0 input + end-mixer column, 2 end stages
constexpr int kEntryFwFirst = 4;                   // CMX_FW_BLOCKS=4
constexpr long long kEntryBitlSeq = 0;             // BITL_SEQ=0
__attribute__((noinline)) inline void EntryApply() {
  // the recipe
  g_nsk = kEntrySkN;
  g_sk_lmask = 0;
  int w = 0;
  for (int k = 0; k < kSkMax; ++k) {
    g_sk_layer[k] = kEntrySkLayer[k];
    g_sk_name[k][0] = 0;
    if (k >= kEntrySkN) continue;
    if (g_sk_layer[k] == kSkRel) snprintf(g_sk_name[k], sizeof(g_sk_name[k]), "R");
    else {
      g_sk_lmask |= 1u << g_sk_layer[k];
      snprintf(g_sk_name[k], sizeof(g_sk_name[k]), "%d", g_sk_layer[k]);
    }
    w += snprintf(g_sk_list + w, sizeof(g_sk_list) - (size_t)w, k ? ",%s" : "%s", g_sk_name[k]);
  }
  g_semkey2 = 0;
  g_sk2_use = kEntrySk2Use;
  g_sk2_mix = (kEntrySk2Use & 1) != 0;
  g_sk2_end = (kEntrySk2Use & 2) != 0;
  g_fw_first = kEntryFwFirst;
  g_bitl_seq = kEntryBitlSeq;
  // every other coding knob: the release default
  g_tfoff = 0;
  g_semoff = 0;
  g_mixrow = 0;
  g_mix_limit = 10000;
  g_rowdecay = 0;
  g_sem_kcen0 = 0;
  g_sem_kcen = 0;
  g_fx_deccode = 448131719ULL;
  g_fx_verb = 451531986ULL;
  g_fx_apm_off = 0;
  g_skip0 = g_skip1 = 0.0f;
  g_skip_on = false;
  g_skip_from = 0;
  g_z3_mul = 1.0f;
  g_z3_on = false;
  g_zvar = 0;
  g_semtop = 0;
  g_st_lv = 3;
  g_st_s1 = 48;
  g_st_s3 = 512;
  g_adm = 0;
  g_tauc_set = false;
  g_tauc = 0.0f;
  g_semin = 0;
  g_semx = false;
  g_semin_st = 0;
  g_fx_ka = 1000;
  g_tau_set = false;
  g_tau = 0.0f;
  g_fwk_mul = 1.0;
  g_match_age = 0;
  g_wl_on = false;
  g_wl_kappa = 0.0f;
  g_wl_text = false;
  g_wl_live = true;
  g_fx_lazy = 0;
  g_fxls = 0;
  g_fwoff = 0;
  // research telemetry and dumps that are not on the test form's observer list: off
  g_pages_n = 0;
  g_mixtel = 0;
  g_fx_ticks = 0;
  g_fw_ticks = 0;
  g_cm_ticks = 0;
  g_valtel = 0;
  g_valtel_groups = nullptr;
  g_fxrec = false;
  g_semdump = nullptr;
  g_matchdump = nullptr;
#if PORT_ENTRY_TEST
  // the observers CMX_LOG_EVERY, CMX_STOP_AT, CMX_AT, CMX_BYTECOST(+_PATH), CMX_SEMTEL stay as Read() set them
  g_hook = Observe() || g_semtel != 0;
  fprintf(stderr, "CMXENTRY test form: recipe A forced (semkeys=%s semkey2_use=%d fw_first_block=%d bitl_seq=%lld), every "
          "other coding knob at its release default; observers log_every=%lld stop_at=%lld at=%d bytecost=%d semtel=%d "
          "hook=%d\n", g_sk_list, g_sk2_use, g_fw_first, g_bitl_seq, g_log_every, g_stop_at, g_at_n, g_bytecost,
          g_semtel, (int)g_hook);
  fflush(stderr);
#else
  // the shipping form: no observer, no hook (Compress() makes the model calls of Decompress() and nothing else)
  g_log_every = 0;
  g_stop_at = 0;
  g_bytecost = 0;
  g_bytecost_path = nullptr;
  g_at_n = 0;
  g_semtel = 0;
  g_hook = false;
#endif
}
#endif  // PORT_ENTRY
}  // namespace cmx_meas

#endif
