#include "meld.h"
#include "predictor.h"
#include "attributes.h"
#include "cold_size.h"
#include "invariants.h"
#include "mixer_selectors.h"
#include "mixer_weights.h"
#include "logit_table.h"
#include "mixer_bank.h"
#include "mixer_lr.h"
#include "char_rate.h"
#include "vocab_frozen.h"
#include "mix_select.h"
#include "final_mix.h"
#include <cmath>
#include "port_cfg.h"
#if PORT_CMX_MEAS
#include "cmx_meas.h"   // CMX_TFOFF / CMX_SEMCACHE_OFF (research ablations, encoder side only)
#include <cstring>
#include <string>       // CMX_VALTEL group names
#endif

#include "transformer_gate_ctx.h"
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>

 Predictor::Predictor(const std::vector<bool>& vocab) : manager_(),
    sigmoid_(100001), vocab_(vocab) {
  wordlbl_ = word_label::Create();
  char_rate::Init(vocab);
  fxcm_neutral_input_ = sigmoid_.LogitBase(0.5f);
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)  // one-time table: scalar code is smaller
  for (int raw = -2047; raw <= 2047; ++raw) {
    float p = fxcm_model_.RawPredictionProbability(static_cast<short>(raw));
    if (p < 1.0e-4f) p = 1.0e-4f;
    else if (p > 1.0f - 1.0e-4f) p = 1.0f - 1.0e-4f;
    fxcm_stretched_inputs_[raw + 2047] = sigmoid_.LogitBase(p);
  }
  fxcm_stretched_inputs_[4095] = fxcm_neutral_input_;
  for (int q = 0; q <= 4095; ++q) {
    float p = q * (1.0f / 4095.0f);
    if (p < 1.0e-4f) p = 1.0e-4f;
    else if (p > 1.0f - 1.0e-4f) p = 1.0f - 1.0e-4f;
    aux_pr_stretched_[q] = sigmoid_.LogitBase(p);
  }
  AddBracket();
  AddPPMD();
  AddWord();
  AddMatch();
  AddDoubleIndirect();
  AddGrammarMatch();
  AddMixers();
  sigmoid_.S52Activate();
}

void Predictor::FreeFxcmMemory() {
  fxcm_model_.FreeMemory();
}

void Predictor::FreeFxcmRest() {
  fxcm_model_.FreeRest();
}

unsigned long long Predictor::GetNumModels() {
  unsigned long long num = 0;

  // models
  num += bracket_model_->NumOutputs(); // bracket
  num += fxcm_model_.NumOutputs();
  num += direct_models_.size();
  num += match_models_.size() - 0;
  num += 1;
  num += indirect_ns_models_.size();
  num += indirect_r_models_.size();
  num += byte_model_->NumOutputs();
  num += tf1_slot_->NumOutputs();
  num += 1;
  num += 1;
  num += 1;
#if PORT_CMX_MEAS
  // research CMX_SEMKEYS / CMX_SEMKEY2: the extra caches' layer-0 inputs, one per cache
  if (__builtin_expect(cmx_meas::g_sk2_mix, 0)) num += (unsigned long long)cmx_meas::g_nsk;
#endif
  return num;
}

 void Predictor::AddMixer(int layer, const unsigned long long& context,
    float learning_rate) {
  mixer_lr::g_layer = layer;
  if (layer == 0) {
    mixer_0_.emplace_back(
        layers_[layer].Inputs(), layers_[layer].ExtraInputs(), context,
      learning_rate, mixer_0_.size());
  } else {
    mixer_1_.emplace_back(
        layers_[layer].Inputs(), layers_[layer].ExtraInputs(), context,
      learning_rate, mixer_1_.size());
  }
}

 void Predictor::AddBracket() {
  bracket_model_.emplace(manager_.bit_context_, 200, 10, 100000, vocab_);
  const Context& context = manager_.AddBracketContext(manager_.bit_context_, 256, 15);
  direct_models_.emplace_back(context.GetContext(), manager_.bit_context_, 30, 0,
      context.Size());
  indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
      manager_.bit_context_, 300, manager_.shared_map_);
}

#include "ppmd_meta.h"
 void Predictor::PpmdWaermen() { byte_model_->Waermen(); }
 void Predictor::AddPPMD() {
  byte_model_.emplace(25, 14000, manager_.bit_context_, vocab_);
}

// The word orders of the sparse contexts, one row each: {count, orders...}.
// Static rows instead of nested vectors: same contexts in the same order.
COLD_SIZE_ATTR  void Predictor::AddWord() {
  float delta = 200;
  static const unsigned int kInd[10][4] = {
      {1, 0}, {2, 0, 1}, {1, 1}, {2, 1, 2}, {2, 1, 3},
      {2, 2, 3}, {2, 3, 4}, {3, 1, 2, 4}, {3, 2, 3, 4}, {1, 2}};
  for (const auto& r : kInd) {
    const Context& context = manager_.AddSparseContext(manager_.words_, r + 1, r[0]);
    indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
        manager_.bit_context_, delta, manager_.shared_map_);
  }
  static const unsigned int kMat[5][4] = {{1, 0}, {1, 1}, {2, 1, 3}, {3, 1, 2, 3}, {2, 7, 2}};
  for (const auto& r : kMat) {
    const Context& context = manager_.AddSparseContext(manager_.words_, r + 1, r[0]);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, 200, 0.5, 2000000, &(manager_.longest_match_));
    if (r[0] == 1 && r[1] == 1) {
      indirect_r_models_.emplace_back(manager_.run_map_, context.GetContext(),
          manager_.bit_context_, delta, manager_.shared_map_);
    }
  }
}

COLD_SIZE_ATTR  void Predictor::AddMatch() {
  float delta = 0.5;
  int limit = 200;
  unsigned long long max_size = 2000000;
  // {order, hash bits}: a static table, no vector (its range constructor
  // vector<int>(const int*, const int*) lies in the encoder block).
  static const int model_params[5][2] = {{0, 8}, {1, 8}, {7, 4}, {11, 3}, {13, 2}};

  // The longest of these matches feeds the transformer's match channel.
  tf1_match_base_ = match_models_.size();
  for (const auto& params : model_params) {
    const Context& context = manager_.AddContextHashContext(manager_.bit_context_,params[0], params[1]);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, limit, delta, std::min(max_size, context.Size()),
        &(manager_.longest_match_));
  }
  tf1_match_count_ = match_models_.size() - tf1_match_base_;
}

 void Predictor::AddGrammarMatch() {
  grammar_match_.emplace(manager_.bit_context_, 200, 0.5);
}

unsigned long long PredictorLayoutTagFast() {
  return ((unsigned long long)sizeof(Predictor) * 4ULL + (unsigned long long)(1) * 2ULL + (unsigned long long)(0));
}

 void Predictor::AddDoubleIndirect() {
  float delta = 400;
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind1,  manager_.bit_context_, delta, manager_.shared_map_);
  (void)rand(); (void)rand();
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind5,  manager_.bit_context_, delta, manager_.shared_map_);
}

unsigned int Discretize(float p) {
  return 1 + 4094 * p;
}
 void Predictor::AddMixers() {
  // There is no byte LSTM and no byte mixer: the transformer has its own
  // slot (Tf1Slot) and takes the place the byte LSTM had in the mixer
  // rows.
  if (transformer::Running()) {
    tf1_engine_.reset(
        new transformer::Tf1Engine(vocab_, transformer::WeightsPath()));
    bitl_hid_ = tf1_engine_->LastHidden();
    bitl_n_hid_ = bitl_hid_ ? tf1_engine_->HiddenSize() : 0;
    bitl_n_tf_ = 4;
  }
  if (!transformer::Active()) {
    // not const: zeros in .bss instead of 768 zero bytes in .rodata; nobody writes it
    static float kKeinTfZustand[192];
    bitl_hid_ = kKeinTfZustand;
    bitl_n_hid_ = 192;
  } else if (bitl_n_hid_ != 192) {
    msg_note(112)

;
    std::fflush(stderr);
    std::abort();
  }
  tf1_slot_.emplace(vocab_);
  fw_slot_.emplace(vocab_);

  // The gate zone of the byte about to be coded.  A POINTER: the
  // automaton writes it once per byte (on already decoded tokens only), the
  // head reads it live, and encoder and decoder read the same value.
  bitl_zone_ = &tf_gate::ctx.zone;

  for (int i = 0; i < 2; ++i) {
    layers_.emplace_back(sigmoid_,
        (1.0e-6f));   // clamp of the layer-0 inputs (checked in MixerInput)
  }

  unsigned long long input_size = GetNumModels();
  layers_[0].SetNumModels(input_size);

  AddMixer(0, manager_.mx9, 0.005);
  AddMixer(0, manager_.mx10, 0.0005);
  AddMixer(0, manager_.recent_bytes_[2], 0.002);
  AddMixer(0, manager_.line_break_, 0.0007);
#if PORT_CMX_MEAS
  // research CMX_MATCH_AGE=2: this mixer selects by the MtcSel bucket of fxcm's match x the far bit instead
  AddMixer(0, cmx_meas::g_match_age == 2 ? cmx_meas::g_mtc_ctx : manager_.longest_match_, 0.0005);
#else
  AddMixer(0, manager_.longest_match_, 0.0005);
#endif
  AddMixer(0, manager_.mx19cxt, 0.002);
  AddMixer(0, manager_.auxiliary_context_, 0.0005);
  AddMixer(0, manager_.mx18, 0.001);
  AddMixer(0,manager_.mx7, 0.001);
  AddMixer(0, manager_.wordscxt, 0.005);
  AddMixer(0, manager_.b2streamcxt, 0.001);
  AddMixer(0, manager_.mx5, 0.001);
  AddMixer(0, manager_.mx6, 0.005);
  AddMixer(0, manager_.b3streamcxt, 0.001);
  AddMixer(0, manager_.mx8, 0.001);
  AddMixer(0, manager_.mx17, 0.005);
  AddMixer(0, manager_.mx16, 0.005);
  AddMixer(0, manager_.mx14, 0.005);
  AddMixer(0, manager_.mx15, 0.005);
  AddMixer(0, manager_.lx1_agree, 0.005);

  input_size = mixer_0_.size() + auxiliary_size_;
  ++input_size;
  ++input_size;
  ++input_size;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semin == 2, 0)) {   // research CMX_SEMIN=2: two final-mixer columns behind the row
    cmx_si_col_ = (unsigned int)input_size;
    input_size += 2;
  }
  // research CMX_SEMKEYS / CMX_SEMKEY2: the extra caches (tables at byte 0), their final-mixer columns the last ones,
  // in cache order
  static_assert(cmx_meas::kSkMax == 3, "Predictor::cmx_sk_ arrays");
  if (__builtin_expect(cmx_meas::g_nsk != 0, 0)) {
    const int n = cmx_meas::g_nsk;
    for (int k = 0; k < n; ++k) {
      cmx_sk_[k] = new semantic_cache::State();
      if (cmx_meas::g_sk2_mix) cmx_sk_col_[k] = (unsigned int)input_size + (unsigned int)k;
    }
    if (cmx_meas::g_sk2_mix) input_size += (unsigned long long)n;
    for (int k = 0; k < n; ++k) {
      char kn[24];   // the key: the mixer read q 2L of layer L, or (element R) the release key q 31
      if (cmx_meas::g_sk_layer[k] == cmx_meas::kSkRel) snprintf(kn, sizeof(kn), "release_q31");
      else snprintf(kn, sizeof(kn), "mixer_read_q%d", 2 * cmx_meas::g_sk_layer[k]);
      fprintf(stderr, "CMXTEL semkey2 init cache=%d/%d layer=%s use=%d key=%s seed=0x%016llx "
              "layer0_inputs=%zu layer0_col=%d final_mixer_cols=%llu sem2_col=%d state_bytes=%zu\n", k + 1, n,
              cmx_meas::g_sk_name[k], cmx_meas::g_sk2_use, kn,
              (unsigned long long)(0x5EC0CAC4E2ULL ^ (uint64_t)(k + 1) * 0x9E3779B97F4A7C15ULL),
              layers_[0].Inputs().size(), cmx_meas::g_sk2_mix ? (int)layers_[0].Inputs().size() - n + k : -1, input_size,
              cmx_meas::g_sk2_mix ? (int)cmx_sk_col_[k] : -1, sizeof(semantic_cache::State));
    }
    fflush(stderr);
  }
#endif
  layers_[1].SetNumModels(input_size);

  // Final mixer with the transformer's gate zone as its context: zone 0 (not
  // gated) selects the same weight row as a context-free mixer, every gated
  // zone gets a row of its own.  This costs rows in the hash map, no
  // geometry.
  AddMixer(1, transformer::Active() ? manager_.tf1_gate_ctx1
                                 : manager_.zero_context_, 0.0003);
  // the add rows of the final mixer on the same stage-1 row, each keyed by its
  // selector; their weights start at zero and add to the gate-zone row.  No
  // extra inputs (a layer-1 mixer has no cascade row).
  mixer_lr::g_layer = 1;
  mixer_1_.emplace_back(layers_[1].Inputs(), layers_[1].ExtraInputs(),
      mix_select::ctxa, 0.0003f * (float)mix_select::kAddLr * 0.001f, 0u);

  layers_[0].SetExtraInputSize(mixer_0_.size());

  mixer_1_[0].E2ESeedHead(
      (unsigned int)mixer_0_.size(),
      (float)(mixer_chain::kSeed) * 0.001f / (float)mixer_0_.size());

}

int lstmpr=2048;

namespace dictreload {
unsigned int mtc_len = 0;
int ord_x = 0, ord_w = 0;
}

float tf1_slot_output=0.0f;
float fw_slot_output=0.0f;

// fast weights after ex-girlfriend 1.0 (Ryan Landay,
// GPL-3.0-only): every step of the fast-weight slot in Predict() and
// Perceive() is one of these calls.  noinline, so the two hot functions get
// no block of their own from it and keep their PGO profile entries.
__attribute__((noinline)) void Predictor::FwSetInput(unsigned int i) {
  fw_index_ = i;
  layers_[0].SetInput(i, fw_slot_output);
}
__attribute__((noinline)) void Predictor::FwSetStage1() {
  layers_[1].SetStretchedInput(mixer_0_.size() + auxiliary_size_ + 1
                               , layers_[0].Inputs()[fw_index_]);
}
__attribute__((noinline)) void Predictor::FwPerceive(int bit) {
  fw_slot_->Perceive(bit);
}
__attribute__((noinline)) void Predictor::FwSetNeutral() {
  fw_slot_->SetProbsNeutral();
  tf1_engine_->FwNeutral();
}
__attribute__((noinline)) void Predictor::FwSetRow() {
  fw_slot_->SetProbs(tf1_engine_->FwRow());
}
__attribute__((noinline)) void Predictor::FwSetOff() {
  fw_slot_->SetProbsOff();
  if (tf1_engine_) tf1_engine_->FwNeutral();
}
__attribute__((noinline)) void Predictor::FwPredictBit() {
  fw_slot_output = fw_slot_->Predict()[0];
}
// The selector of the layer-0 mixer lx1_agree with a transformer slot: the
// mass class of the favourite of the transformer row while that byte is still
// possible after the bits already coded (1..8, class limits below; 0 = no
// live row or favourite ruled out), a gated zone selects 9 + zone; times 256
// plus the bit context.  Favourite and class are taken at the first bit of a
// byte; a row of zeros (slot off) or the uniform gate-neutral row is no live
// row.  Implementation adapted from cmix-neif-beehive (Vladimir Malinovskii,
// 2026, commit 92c07c0b).  noinline: Predict() keeps a single call here and
// its PGO profile entry; minsize: the byte part runs once per byte.
namespace {
struct Lx1Tf { int ex = -1; unsigned int q = 0; };
Lx1Tf g_lx1tf;   // favourite of this byte (-1 = no live row) and its class
const float kLx1TfLim[7] = {0.3f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f};
}
__attribute__((noinline, minsize)) void Predictor::Lx1TfConf() {
  const int bot = tf1_slot_->Lx1Bot(), top = tf1_slot_->Lx1Top();
  if (bot == 0 && top == 255) {
    const std::valarray<float>& pr = tf1_slot_->BytePredict();
    int ex = 0;
    for (int i = 1; i < 256; ++i)
      if (pr[i] > pr[ex]) ex = i;
    const float b = pr[ex];
    unsigned int q = 1;   // the limits rise: count them while they hold
    while (q < 8 && b >= kLx1TfLim[q - 1]) ++q;
    g_lx1tf.q = q;
    g_lx1tf.ex = b > 1.0f / (float)transformer::kSetSize ? ex : -1;
  }
  unsigned int q = (g_lx1tf.ex >= bot && g_lx1tf.ex <= top) ? g_lx1tf.q : 0u;
  if (tf_gate::ctx.zone) q = 9u + tf_gate::ctx.zone;
  manager_.lx1_agree = (unsigned long long)q * 256 + manager_.long_bit_context_;
}

namespace mix_select { unsigned long long ctxa = 0, ctxb = 0; }
namespace tf_path { St st; }
__attribute__((noinline)) void Predictor::KtxSelect() {
  // on a byte with a live transformer row the add rows select their row by
  // their form; gated zones and bytes without a row keep the zone row.
  const bool live = manager_.tf1_gate_ctx1 == 0 && tf1_slot_->KtxLive();
  // selector: path state x confidence class
  mix_select::ctxa = live ? mix_select::kBase + (unsigned int)tf_path::Sel24()
                           : manager_.tf1_gate_ctx1;
}
__attribute__((noinline)) void Predictor::KtxPrime() {
  mixer_1_[mix_select::kIa].Prime();
}
__attribute__((noinline)) float Predictor::KtxAddMix() {
  return mixer_1_[mix_select::kIa].Mix();
}
__attribute__((noinline)) void Predictor::KtxW1Sum(float* w1s, unsigned int n) {
  // the chain factor of the direct path into the end mixer is the summed row
  const mixw_t* const wa = mixer_1_[0].E2EWRow();
  const mixw_t* const wb = mixer_1_[mix_select::kIa].E2EWRow();
  for (unsigned int k = 0; k < n; ++k) w1s[k] = wa[k] + wb[k];
}
// Path state of the transformer slot for the
// next bit, from the row at the byte boundary and the slot's interval after
// the bit just coded.  Per byte: favourite, confidence class and top-16 list
// of the row (a row of zeros or a flat row is no live row); per bit: path
// class ps (favourite inside the interval = 0, else the rank class of the best
// top-16 byte inside it) and the orientation d of the best remaining byte.
__attribute__((noinline)) void Predictor::MlPath(bool byte_update) {
  using tf_path::st;
  const std::valarray<float>& pr = tf1_slot_->BytePredict();
  if (byte_update) {
    // favourite, confidence class and top list of the new row; a row of zeros
    // (slot off) or a flat row (gate-neutral) is no live transformer row.
    int ex = 0;
    float best = pr[0], sum = pr[0], mn = pr[0] > 0.0f ? pr[0] : 2.0f;
    for (int i = 1; i < 256; ++i) {
      const float v = pr[i];
      sum += v;
      if (v > best) { best = v; ex = i; }
      if (v > 0.0f && v < mn) mn = v;
    }
    st.live = (sum > 0.0f && best > mn) ? 1 : 0;
    st.fav = ex;
    st.conf = best < 0.5f * sum ? 0 : best < 0.9f * sum ? 1 : best < 0.99f * sum ? 2 : 3;
    int nt = 0;
    for (int i = 0; i < 256; ++i) {
      const float v = pr[i];
      if (v <= 0.0f) continue;
      int p = nt;
      while (p > 0 && pr[st.top[p - 1]] < v) --p;
      if (p < tf_path::kTop) {
        for (int q = (nt < tf_path::kTop ? nt : tf_path::kTop - 1); q > p; --q) st.top[q] = st.top[q - 1];
        st.top[p] = i;
        if (nt < tf_path::kTop) ++nt;
      }
    }
    st.ntop = nt;
  }
  st.ps = 0;
  st.d = 0;
  if (!st.live) return;
  const int bot = tf1_slot_->Lx1Bot(), top = tf1_slot_->Lx1Top();
  const int width = top - bot + 1;
  if (width < 2) return;
  const int k = 8 - __builtin_ctz((unsigned)width);   // bits of the byte already coded
  int b2 = -1, r = 99;
  for (int q = 0; q < st.ntop; ++q) {
    const int b = st.top[q];
    if (b >= bot && b <= top) { b2 = b; r = q + 1; break; }
  }
  if (b2 < 0) {   // no top-list byte left: the best byte of the interval
    b2 = bot;
    float bb = pr[bot];
    for (int i = bot + 1; i <= top; ++i) if (pr[i] > bb) { bb = pr[i]; b2 = i; }
  }
  st.ps = (st.fav >= bot && st.fav <= top) ? 0
        : r <= 2 ? 1 : r == 3 ? 2 : r <= 5 ? 3 : r <= 10 ? 4 : 5;
  st.d = ((b2 >> (7 - k)) & 1) ? 1 : -1;
}
static_assert(13 == 13, "MlAppend() writes 13 columns, HEAD_X_COLS must say so");

__attribute__((noinline)) void Predictor::MlAppend() {
  static_assert(22 + 13 <= (int)(sizeof(bitl_s1buf_) / sizeof(bitl_s1buf_[0])),
                "the stage-1 row and the columns of MlAppend() must fit bitl_s1buf_");
  using tf_path::st;
  const int n = bitl_n_stage1_;
  for (int j = 0; j < n; ++j) bitl_s1buf_[j] = bitl_stage1_[j];
  bitl_stage1_ = bitl_s1buf_;
  float* o = bitl_s1buf_ + n;
  o[0] = layers_[1].Inputs()[mixer_0_.size() + auxiliary_size_ + 1];   // FwSetStage1's column
  o[1] = (float)(31 - __builtin_clz(manager_.bit_context_));
  for (int j = 2; j < 13; ++j) o[j] = 0.0f;
  if (st.live) {
    o[2 + st.ps] = 1.0f;
    o[8 + st.conf] = 1.0f;
    o[12] = (float)st.d;
  }
  bitl_n_stage1_ = n + 13;
}
static_assert(563 <= mixer_precond::kMaxN0,
              "the layer-0 row must fit the column state");
static_assert(100 > 0 && 100 < 10000, "is the fixed-share alpha in units of 1e-4");

static final_mix::Bay g_mxs_bay;
__attribute__((noinline)) float Predictor::MxsBayMix(float zrow) {
  const unsigned n0 = (unsigned)mixer_0_.size();
  const float* s1 = &layers_[1].Inputs()[0];
  // stage-1 row: fxcm at n0, transformer at n0 + 1, fast-weight row at n0 + 2
  const float s4[final_mix::kExperts] = {s1[n0 + 1], s1[n0 + 2], layers_[0].Inputs()[mxs_ppmd_ix_], s1[n0]};
  return final_mix::BayMix(g_mxs_bay, zrow, s4);
}
static_assert(400 > 0 && 400 < 1000, "is a per-mille weight below 1000");

__attribute__((noinline)) void Predictor::EndLearn(int bit) {
  // the Bayes weight w steps on the final mixer's staircase (Mixer::Perceive)
  // with the stock thresholds (the scaled thresholds apply to the mixer rows
  // only)
  const unsigned long long st = mixer_1_[0].Steps();
  float decay = mixdecay::kFinal;
  if (st < 25000000ULL) { decay = 0.3f; if (st < 5000000ULL) { decay = 0.7f; if (st < 1000000ULL) decay = 1.0f; } }
  const float lr0 = 0.0003f * decay * mixer_lr::Gain(1);
  const float e = final_mix::BayLearn(g_mxs_bay, bit, lr0);
  mixer_1_[0].E2ESetErr(e);
  mixer_1_[mix_select::kIa].E2ESetErr(e);
}
 float Predictor::Predict() {
  CMXK_START(cmx_ck)
  fxcm_model_.PredictHalf();
  CMXK_LAP(CK_FXPRED, cmx_ck)
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_md_arm, 0) && manager_.bit_context_ == 1) {   // research CMX_MATCHDUMP
    cmx_meas::MatchRec& r = cmx_meas::g_mdrec;
    unsigned int best = tf1_match_base_;
    unsigned char bl = match_models_[best].MatchLength();
    for (unsigned int q = 1; q < tf1_match_count_; ++q) {
      const unsigned char l = match_models_[tf1_match_base_ + q].MatchLength();
      if (l > bl) { bl = l; best = tf1_match_base_ + q; }
    }
    if (r.flags & 2u) {   // fxcm left the raw byte
      const unsigned char fb = r.fx_sym;
      r.fx_sym = transformer::InSet(fb) ? (uint8_t)transformer::g_b2i[fb] : 255u;
    }
    const unsigned char cb = match_models_[best].MatchByte();
    r.cm_len = bl;
    r.cm_sym = transformer::InSet(cb) ? (uint8_t)transformer::g_b2i[cb] : 255u;
    const float cf = match_models_[best].MatchConf();
    r.cm_conf = (uint16_t)lrintf((cf < 0.0f ? 0.0f : (cf > 1.0f ? 1.0f : cf)) * 65535.0f);
    if (bl > 0) r.flags |= 1u;
    r.flags |= (uint8_t)(((best - tf1_match_base_) & 7u) << 4);
    if ((r.flags & 3u) == 3u && r.cm_sym == r.fx_sym) r.flags |= 4u;
    const unsigned int fl = (r.flags & 2u) ? r.fx_len : 0u;
    r.src = (r.flags & 3u) == 0 ? 0 : ((r.flags & 1u) && (unsigned int)bl / 8u >= fl ? 1 : ((r.flags & 2u) ? 2 : 1));
    cmx_meas::g_md_arm = false;
  }
#endif
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_match_age == 2, 0))   // research CMX_MATCH_AGE=2 (fxcm set mtc_len just now)
    cmx_meas::g_mtc_ctx = ((unsigned long long)dictreload::MtcSel() * 2ull + cmx_meas::g_mtc_far) * 256ull +
                          manager_.long_bit_context_;
#endif
  Lx1TfConf();   // lx1_agree selector: transformer favourite class
  manager_.UpdateTf1Gate();   // gate zone as mixer context
  KtxSelect();   // contexts of the add rows
  SemPredBit();
  CMXK_LAP(CK_SEL, cmx_ck)
  unsigned int input_index = 0;
  auto bracket_model_output = bracket_model_->Predict()[0];
  layers_[0].SetInput(input_index++, bracket_model_output);
  CMXK_LAP(CK_CMIXP, cmx_ck)

  const unsigned int fxcm_model_outputs = fxcm_model_.NumOutputs();
  const short* fxcm_raw_outputs = fxcm_model_.RawPredictions();
  unsigned int fxcm_active_outputs = fxcm_model_.ActivePredictions();
  if (fxcm_active_outputs > fxcm_model_outputs) fxcm_active_outputs = fxcm_model_outputs;
  // fxcmv1 emits outputs densely from zero; carrying an active count avoids
  // a hot per-bit mask scan and the old unused float prediction mirror.
  // The lookup table is already clamped to MixerInput's stretched range, so
  // these 560-ish assignments can skip duplicate bounds checks.
  for (unsigned int j = 0; j < fxcm_active_outputs; ++j) {
    layers_[0].SetStretchedInputUnchecked(
        input_index, fxcm_stretched_inputs_[fxcm_raw_outputs[j] + 2047]);
    ++input_index;
  }
  for (unsigned int j = fxcm_active_outputs; j < fxcm_model_outputs; ++j) {
    layers_[0].SetStretchedInputUnchecked(input_index, fxcm_neutral_input_);
    ++input_index;
  }
  const unsigned int adepth_lo = input_index;
  auto fxcm_model_index = input_index - 1;
  float aux_fxcm_input = fxcm_neutral_input_;
  const bool aux_is_filler = (fxcm_active_outputs < fxcm_model_outputs);
  if (aux_is_filler) {
    int aux_pr = fxcm_model_.FinalPrediction();
    if (aux_pr < 0) aux_pr = 0;
    else if (aux_pr > 4095) aux_pr = 4095;
    aux_fxcm_input = aux_pr_stretched_[aux_pr];
    layers_[0].SetStretchedInputUnchecked(fxcm_model_index, aux_fxcm_input);
  } else {
    aux_fxcm_input = layers_[0].Inputs()[fxcm_model_index];
  }
  {
    const unsigned int kNeed = (6 == 7) ? 3u : 2u;
    if (fxcm_active_outputs + kNeed > fxcm_model_outputs) {
      msg_note(113)
;
      fflush(stderr);
      exit(3);
    }
    const unsigned int c0_sh = manager_.bit_context_;
    const int bp_sh = 31 - __builtin_clz(c0_sh);
    layers_[0].SetStretchedInputUnchecked(fxcm_model_index - 1,
        fxcm_neutral_input_ + id_exclude::MixDelta(c0_sh, bp_sh));
  }
  CMXK_LAP(CK_FXCOPY, cmx_ck)

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    const std::valarray<float>& outputs = direct_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }

  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    const std::valarray<float>& outputs = match_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }

  if (grammar_match_->HaveExpectation()) {
    layers_[0].SetInput(input_index, grammar_match_->Predict()[0]);
  } else {
    layers_[0].SetZero(input_index);
  }
  ++input_index;

  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    const std::valarray<float>& outputs = indirect_ns_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }

  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    const std::valarray<float>& outputs = indirect_r_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
  CMXK_LAP(CK_CMIXP, cmx_ck)
  layers_[0].SetInput(input_index++, byte_model_->Predict()[0]);

  mxs_ppmd_ix_ = (unsigned)(input_index - 1);
  float byte_mixer_override = -1;
  {
    const int chr_forced = char_rate::Forced(manager_.bit_context_);
    if (chr_forced) byte_mixer_override = chr_forced > 0 ? 1.0f : 0.0f;
  }
  layers_[0].SetInput(input_index++, tf1_slot_output);
  auto tf1_slot_index = input_index - 1;

#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_wl_on, 0)) cmx_meas::g_wl_idx = input_index;   // research WL-MIXGRAD: j_wl
#endif
  layers_[0].SetStretchedInput(input_index, word_label::Forward(wordlbl_));
  ++input_index;
  const float aux_aux_in = aux_fxcm_input;
  float auxiliary_average = Sigmoid::Logistic(aux_aux_in)
      ;
  auxiliary_average /= auxiliary_size_;
  manager_.auxiliary_context_ =auxiliary_average * 15;
  {
    const float reload_d = 0.0f;
    manager_.reload_aux = (unsigned long long)dictreload::Q8diff(reload_d) * 256
                       + manager_.long_bit_context_;
  }
  CMXK_LAP(CK_SLOTP, cmx_ck)

  {
    adepth_.Gather(&layers_[0].Inputs()[0], adepth_lo, input_index,
                   layers_[0].Inputs()[0], aux_fxcm_input);
    const float adepth_z = adaptive_depth::Gate(adepth_);
    bb_z_ = adepth_z;
    const float adepth_az = adepth_z < 0.0f ? -adepth_z : adepth_z;
#if PORT_CMX_MEAS
    adepth_.skip = (adepth_az > (__builtin_expect(cmx_meas::g_tau_set, 0) ? cmx_meas::g_tau : adaptive_depth::kTau)) &&
#else
    adepth_.skip = (adepth_az > adaptive_depth::kTau) &&
#endif
                   (adepth_.coded >= (unsigned long long)0) &&
                   !adaptive_depth::MetaOff();
  }
#if PORT_CMX_MEAS
  // research CMX_ADEPTH_MODE (cmx_meas.h): the gate decision, split; unset: learning stands still exactly where the
  // coder takes sigma(z), as released
  adm_loff_ = adepth_.skip;
  adm_run_ = false;
  if (__builtin_expect(cmx_meas::g_adm != 0, 0)) {
    if (byte_mixer_override < 0) {
      ++cmx_meas::g_adm_bits[cmx_meas::g_region];
      if (adepth_.skip) ++cmx_meas::g_adm_gate[cmx_meas::g_region];
    }
    if (cmx_meas::g_adm == 1) {
      adepth_.skip = false;   // learn-skip only: the cascade codes every bit, its learners skip the gate bits
    } else {
      adm_loff_ = false;      // code-skip only: the cascade runs and learns on every bit,
      adm_run_ = adepth_.skip && byte_mixer_override < 0;   // the coder keeps sigma(z) on the gate bits
    }
  } else if (__builtin_expect(cmx_meas::g_tauc_set, 0)) {
    // research CMX_ADEPTH_TAU_CODE: adepth_.skip above is the learning decision |z| > tau_learn; the coding decision
    // is |z| > tau_code (same form)
    const float az = bb_z_ < 0.0f ? -bb_z_ : bb_z_;
    const bool code = (az > cmx_meas::g_tauc) && (adepth_.coded >= (unsigned long long)0) && !adaptive_depth::MetaOff();
    if (byte_mixer_override < 0) {
      ++cmx_meas::g_adm_bits[cmx_meas::g_region];
      if (adm_loff_) ++cmx_meas::g_adm_gate[cmx_meas::g_region];
      if (code) ++cmx_meas::g_adm_cgate[cmx_meas::g_region];
    }
    adepth_.skip = code;                                         // coded with sigma(z)
    adm_run_ = code && !adm_loff_ && byte_mixer_override < 0;    // ... while the cascade runs and learns
  }
#endif
  // after ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only), the
  // fast-weight row, the LAST layer-0 input -- appended after the
  // adaptive-depth gather, so its feature set and its skip decisions (hence
  // the BITLSTM's bit set) do not see it.  GetNumModels() counts it.  All
  // fast-weight steps of this hot function are calls without a branch of
  // their own, so its PGO profile entry stays valid.
  CMXK_LAP(CK_GATE, cmx_ck)
  FwSetInput(input_index++);
  SemSetInput(input_index++);
  CMXK_LAP(CK_SLOTP, cmx_ck)
  const bool chr_skip_mix = (byte_mixer_override >= 0);
  float p;
  if (chr_skip_mix) {
    p = 0.5f;
#if PORT_CMX_MEAS
  } else if (adepth_.skip && !adm_run_) {   // research CMX_ADEPTH_MODE=2: a gate bit runs the cascade (below)
#else
  } else if (adepth_.skip) {
#endif
    p = adepth_.p;
  } else {
  for (unsigned int i = 0; i < mixer_0_.size(); ++i) mixer_0_[i].Prime();
  mixer_1_[0].Prime();
  KtxPrime();
  CMXK_LAP(CK_PRIME, cmx_ck)
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_mixtel, 0)) CmxMixTel();   // research CMX_MIXTEL (zero-byte)
#endif
  const unsigned int bank_n = mixer_0_.size();
  if (bank_n <= 64) {
    float bank_p[64];
    MixBankMain(&mixer_0_[0], bank_n, bank_p);
    for (unsigned int i = 0; i < bank_n; ++i) {
      float p = mixer_0_[i].BankFinish(bank_p[i]);
      layers_[0].SetExtraInput(i, p);
      layers_[1].SetStretchedInput(i, p);
    }
  } else {
    for (unsigned int i = 0; i < bank_n; ++i) {
      float p = mixer_0_[i].Mix();
      layers_[0].SetExtraInput(i, p);
      layers_[1].SetStretchedInput(i, p);
    }
  }
  CMXK_LAP(CK_BANK, cmx_ck)
  layers_[1].SetStretchedInput(mixer_0_.size(), aux_aux_in);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1, layers_[0].Inputs()[tf1_slot_index]);
  // The fast-weight row, the last end-mixer column (FW1 of ex-girlfriend
  // 1.0); bitl_n_stage1_ below does not count it.
  FwSetStage1();
  SemSetStage1();
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semin == 2, 0)) CmxSemInCols();   // research CMX_SEMIN=2: the two final-mixer columns
#endif

  const float m1raw = MxsBayMix(mixer_1_[0].Mix() + KtxAddMix());
  end_z_ = m1raw;
  bitl_stage1_ = &layers_[1].Inputs()[0];
  bitl_n_stage1_ = (int)(mixer_0_.size() + auxiliary_size_);
  ++bitl_n_stage1_;
  MlAppend();   // the 13 columns behind the row (FW logit, bit position, path state)
  SemAppend();   // the semantic-cache columns behind the row
  CMXK_LAP(CK_FINAL, cmx_ck)
  bitl_m1raw_ = m1raw;
  p = Sigmoid::Logistic(m1raw);
  p = sse_.Predict(p);
  p = final_mix::SseMix(p, m1raw);
  {
    const float kEpsIsbn = 1.0f / 32.0f;
    if (invariants::g_isbn_rule) {
      const float r = (invariants::g_isbn_rule == 2) ? 1.0f : 0.0f;
      p = (1.0f - kEpsIsbn) * r + kEpsIsbn * p;
    }
    p = invariants::cvz.End(p, manager_.bit_context_);
    if (p < 1.0f / 65535.0f) p = 1.0f / 65535.0f;
    if (p > 65534.0f / 65535.0f) p = 65534.0f / 65535.0f;
  }
  {
    const unsigned int c0_hd = manager_.bit_context_;
    p = id_exclude::Hard(p, c0_hd, 31 - __builtin_clz(c0_hd));
  }
  CMXK_LAP(CK_SSE, cmx_ck)
#if PORT_CMX_MEAS
  // research CMX_ADEPTH_MODE=2, a gate bit: the cascade ran forward (it learns in Perceive, the BITLSTM reads its p);
  // the coder takes sigma(z) as released
  if (__builtin_expect(adm_run_, 0)) {
    adm_pcas_ = p;
    p = adepth_.p;
  }
#endif
  }
  if (byte_mixer_override >= 0) p = byte_mixer_override;
  bitl_override_ = (byte_mixer_override >= 0) ? 1 : 0;
  last_p16_ = (unsigned int)(1 + 65534 * p);
  return p;
}

 void Predictor::MetalrBlockEnde() {
#if PORT_CMX_MEAS
  cmx_meas::g_z3_on = false;   // research CMX_Z3LR: the census LR window ends with the block
  ++cmx_meas::g_rd_epoch;      // research CMX_ROWDECAY: every row clock restarts with the mixers' steps
#endif
  for (auto& m : mixer_0_) m.MetalrSteps0();
  for (auto& m : mixer_1_) m.MetalrSteps0();
}

#include <sys/mman.h>
#include "page_registry.h"
namespace ppmd_page { int raum(unsigned long long bytes); }   // models/ppmd_page.cpp
namespace {
static inline uint32_t SemRng(uint64_t& s) {
  s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
  return (uint32_t)((s * 0x2545F4914F6CDD1DULL) >> 32);
}
static inline double SemClamp(double x, double a) { return x > a ? a : (x < -a ? -a : x); }
// out[r] = sum_c rows[r][c] * x[c] for N-byte rows of +-1 and |x[c]| <= 127,
// nr a multiple of 4.  Exact: vpsignb gives +-x, vpmaddubsw adds byte pairs
// into int16 (N / 32 pair sums of |.| <= 254 per lane), vpmaddwd widens.
template <int N>
__attribute__((noinline)) void SemDotPm1(const int8_t* rows, int nr, const int8_t* x, int32_t* out) {
  static_assert(N % 32 == 0 && N <= 256, "row length");
  constexpr int kV = N / 32;
  const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
  __m256i xv[kV];
  for (int v = 0; v < kV; ++v) xv[v] = _mm256_loadu_si256((const __m256i*)(x + 32 * v));
  for (int r = 0; r < nr; r += 4) {
    __m256i acc[4];
    for (int i = 0; i < 4; ++i) {
      const int8_t* h = rows + (size_t)(r + i) * N;
      __m256i a = _mm256_maddubs_epi16(one8, _mm256_sign_epi8(xv[0], _mm256_loadu_si256((const __m256i*)h)));
      for (int v = 1; v < kV; ++v)
        a = _mm256_add_epi16(a, _mm256_maddubs_epi16(one8, _mm256_sign_epi8(xv[v], _mm256_loadu_si256((const __m256i*)(h + 32 * v)))));
      acc[i] = _mm256_madd_epi16(a, one16);
    }
    const __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(acc[0], acc[1]), _mm256_hadd_epi32(acc[2], acc[3]));
    _mm_storeu_si128((__m128i*)(out + r), _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1)));
  }
}
// every cache line of one bucket (kM 16-byte slots: 4 or 5 lines)
static inline void SemPfBucket(const semantic_cache::Slot* b) {
  const uintptr_t e = (uintptr_t)(b + semantic_cache::kM) - 1;
  for (uintptr_t a = (uintptr_t)b & ~(uintptr_t)63; a <= e; a += 64) __builtin_prefetch((const void*)a);
}
// every cache line of one row of an end stage (20 doubles: 3 or 4 lines)
static inline void SemPfRow(const double* r) {
  static_assert(semantic_cache::kUIn == 20, "four prefetches cover a row of 160 B");
  __builtin_prefetch(r);
  __builtin_prefetch(r + 8);
  __builtin_prefetch(r + 16);
  __builtin_prefetch(r + 19);
}
// int4 ring key, max |k| -> 7, low nibble first
// (k * f rounded to nearest as lrintf, 32 components per step; |k| <= 127)
static inline void SemPut4(uint8_t* dst, const int8_t* k) {
  constexpr int n = semantic_cache::kD;
  static_assert(n % 32 == 0, "key length");
  __m256i am = _mm256_setzero_si256();
  for (int a = 0; a < n; a += 32) am = _mm256_max_epu8(am, _mm256_abs_epi8(_mm256_loadu_si256((const __m256i*)(k + a))));
  __m128i m8 = _mm_max_epu8(_mm256_castsi256_si128(am), _mm256_extracti128_si256(am, 1));
  m8 = _mm_max_epu8(m8, _mm_srli_si128(m8, 8));
  m8 = _mm_max_epu8(m8, _mm_srli_si128(m8, 4));
  m8 = _mm_max_epu8(m8, _mm_srli_si128(m8, 2));
  m8 = _mm_max_epu8(m8, _mm_srli_si128(m8, 1));
  const int m = _mm_cvtsi128_si32(m8) & 0xFF;
  const float f = m > 0 ? 7.0f / (float)m : 0.0f;
  const __m256 fv = _mm256_set1_ps(f);
  const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m256i lo4 = _mm256_set1_epi16(0x000F), hi4 = _mm256_set1_epi16(0x00F0);
  for (int a = 0; a < n; a += 32) {
    __m256i r[4];
    for (int i = 0; i < 4; ++i)
      r[i] = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(k + a + 8 * i)))), fv));
    // components a..a+31 as int8 in order, then byte j = (c[2j] & 15) | (c[2j+1] & 15) << 4
    const __m256i b = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(_mm256_packs_epi32(r[0], r[1]), _mm256_packs_epi32(r[2], r[3])), perm);
    const __m256i w = _mm256_or_si256(_mm256_and_si256(b, lo4), _mm256_and_si256(_mm256_srli_epi16(b, 4), hi4));
    const __m256i o = _mm256_permute4x64_epi64(_mm256_packus_epi16(w, w), 0x08);
    _mm_storeu_si128((__m128i*)(dst + (a >> 1)), _mm256_castsi256_si128(o));
  }
}
// first slot of a bucket with the smallest entry counter (unsigned), as the
// scan "um = 0; if (e1[u] < e1[um]) um = u"; reads 16 slots, the 16th (the next
// bucket or the table's padding) does not count
static inline int SemOldest(const semantic_cache::Slot* sl) {
#if PORT_CMX_MEAS
  // research SEMCACHE capacity builds (driver 11-07, SEMCONST kM != 15): the same rule as the scan in the comment
  if (semantic_cache::kM != 15) {
    int um = 0;
    for (int u = 1; u < semantic_cache::kM; ++u) if (sl[u].e1 < sl[um].e1) um = u;
    return um;
  }
#else
  static_assert(semantic_cache::kM == 15, "15 slots per bucket");
#endif
  const __m256i* v = (const __m256i*)sl;
  const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  // counters of slots 0..7 and 8..15 in slot order
  const __m256i e0 = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi64(
      _mm256_unpacklo_epi32(_mm256_loadu_si256(v), _mm256_loadu_si256(v + 1)),
      _mm256_unpacklo_epi32(_mm256_loadu_si256(v + 2), _mm256_loadu_si256(v + 3))), perm);
  __m256i e1 = _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi64(
      _mm256_unpacklo_epi32(_mm256_loadu_si256(v + 4), _mm256_loadu_si256(v + 5)),
      _mm256_unpacklo_epi32(_mm256_loadu_si256(v + 6), _mm256_loadu_si256(v + 7))), perm);
  e1 = _mm256_blend_epi32(e1, _mm256_set1_epi32(-1), 0x80);
  __m256i mn = _mm256_min_epu32(e0, e1);
  mn = _mm256_min_epu32(mn, _mm256_permute2x128_si256(mn, mn, 1));
  mn = _mm256_min_epu32(mn, _mm256_shuffle_epi32(mn, 0x4E));
  mn = _mm256_min_epu32(mn, _mm256_shuffle_epi32(mn, 0xB1));
  const unsigned k = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(e0, mn))) |
                     ((unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(e1, mn))) << 8);
  return __builtin_ctz(k);
}
// log-odds of a readout from its masses by bit value (part sums of the
// neighbours consistent with the prefix, SemPredBit) and its weight sum
static inline double SemUtReadM(const double* m, double ws) {
  const double m0 = m[0], m1 = m[1];
  if (!((m0 + m1) > 0.0) || !(ws > 0.0)) return 0.0;
  return log((m1 / ws + 0.01) / (m0 / ws + 0.01));
}
// the same with equal weights, from the counts by bit value
static inline double SemUtReadC(int c0, int c1, int n) {
  const double m0 = (double)c0, m1 = (double)c1;
  const double ws = (double)n;
  return log((m1 / ws + 0.01) / (m0 / ws + 0.01));
}
static inline double SemUtLogitP(double p) {
  if (!(p > 0.0)) return -40.0;
  if (!(p < 1.0)) return 40.0;
  return log(p / (1.0 - p));
}
// e = exp(-z) of the stage output z
static inline void SemUtStepE(double* w, double* g2, const double* f, int n, double e, int bit, double lr) {
  const double err = 1.0 / (1.0 + e) - (double)bit;
  for (int i = 0; i < n; ++i) {
    const double g = err * f[i];
    g2[i] += g * g;
    w[i] -= lr * g / (sqrt(g2[i]) + 1e-6);
  }
}
static inline void SemUtStep(double* w, double* g2, const double* f, int n, double z, int bit, double lr) {
  SemUtStepE(w, g2, f, n, exp(-z), bit, lr);
}
}  // namespace

// large zero-initialised table, transparent huge pages requested (2-MiB
// aligned).  reg: a table that is read before it is written (the buckets)
// gets the zero page on its first read and stays in 4-KiB pages after the
// first write; the periodic MADV_COLLAPSE of the allocation registry
// (huge_pages.h) turns its densely resident blocks into huge pages.
static void* SemTable(size_t bytes, bool reg) {
  const size_t kHuge = (size_t)2 << 20;
  const size_t len = (bytes + 2 * kHuge - 1) & ~(kHuge - 1);
  void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  uintptr_t a = ((uintptr_t)p + kHuge - 1) & ~(uintptr_t)(kHuge - 1);
  madvise((void*)a, len - (a - (uintptr_t)p), MADV_HUGEPAGE);
  if (reg) PAGE_REGISTRY_ADD((void*)a, bytes);
  return (void*)a;
}

__attribute__((noinline)) COLD_SIZE_ATTR void Predictor::SemInit() {
  using namespace semantic_cache;
  semantic_cache::State& s = sem_;
  s.key = (uint8_t*)SemTable((size_t)kN * kKB, false);
  s.val = (uint8_t*)SemTable((size_t)kN, false);
  s.slot = (Slot*)SemTable(((size_t)kL << kB) * kM * sizeof(Slot), true);
  s.epos = (uint32_t*)SemTable((size_t)kN * 4, false);
  s.hist = (uint8_t*)SemTable((size_t)kHist, false);
  if (!s.key || !s.val || !s.slot || !s.epos || !s.hist) {
    msg_note(553);
    exit(1);
  }
  // Resident now, at the first stream byte, behind room made in the PPMd
  // pool.  Lookups and entries start at the text; the bucket table is read
  // before it is written, and on a kernel that answers a write to the huge
  // zero page with a huge page its first few hundred entries made all 960 MiB
  // resident within 100 ms, after the pool had grown.  All of
  // it is resident at the end anyway.  Zero-filled, so no value changes.
  {
    const size_t n[5] = {(size_t)kN * kKB, (size_t)kN, ((size_t)kL << kB) * kM * sizeof(Slot),
                         (size_t)kN * 4, (size_t)kHist};
    void* const t[5] = {s.key, s.val, s.slot, s.epos, s.hist};
    unsigned long long sum = 0;
    for (int i = 0; i < 5; ++i) sum += n[i];
    if (ppmd_page::raum(sum))
      for (int i = 0; i < 5; ++i) memset(t[i], 0, n[i]);
  }
  // one generator, in this order: projection, table hyperplanes, sketch hyperplanes
  uint64_t r = 0x5EC0CAC4E2ULL;
  for (int a = 0; a < kD; ++a)
    for (int c = 0; c < 192; ++c) s.proj[a][c] = (SemRng(r) >> 31) ? 1 : -1;
  for (int h = 0; h < kL * kB; ++h)
    for (int a = 0; a < kD; ++a) s.hyp[h][a] = (SemRng(r) >> 31) ? 1 : -1;
  for (int h = 0; h < kSkGen; ++h)
    for (int a = 0; a < kD; ++a) s.skp[h][a] = (SemRng(r) >> 31) ? 1 : -1;
  // every stage of the end stage starts at "coder unchanged" (weight 1 on
  // the coder log-odds), the final stage at the mean of the three; the rows
  // are zero from the State constructor and written only after this point
  for (int c = 0; c < kUC4; ++c) s.u4w[c][0] = 1.0;
  for (int c = 0; c < kUC5; ++c) s.u5w[c][0] = 1.0;
  for (int c = 0; c < kUC6; ++c) s.u6w[c][0] = 1.0;
  // f[10] of the end stage (one neighbour of weight 1): log((m1 + 0.01) /
  // (m0 + 0.01)); the asm keeps the arguments opaque, so log runs at run time
  {
    double a0 = (0.0 + 0.01) / (1.0 + 0.01), a1 = (1.0 + 0.01) / (0.0 + 0.01);
    asm volatile("" : "+x"(a0), "+x"(a1));
    s.l10[0] = log(a0);
    s.l10[1] = log(a1);
  }
  for (int c = 0; c < 8; ++c) {
    s.ufw[c][0] = s.ufw[c][1] = s.ufw[c][2] = 1.0 / 3.0;
    s.ufw[c][3] = 0.0;
    for (int i = 0; i < 4; ++i) s.ufg[c][i] = 0.0;
  }
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semtop != 0, 0)) CmxSemTopInit();   // research CMX_SEMTOP: the wide tables
  if (__builtin_expect(cmx_meas::g_semin != 0, 0)) CmxSemInInit();     // research CMX_SEMIN: the extra weights
  if (__builtin_expect(cmx_meas::g_semin_st != 0, 0)) CmxSemInSelfTest();   // research CMX_SEMIN_SELFTEST: ends here
#endif
}

#if PORT_CMX_MEAS
// research CMX_SEMDUMP: the record of this byte from SEMCACHE's main readout (observer only)
__attribute__((noinline)) static void CmxSemCapture(const int* used, int nu, const double* acc, double wsum,
                                                    double smax0, int nb, int nk, int agr, double hrate) {
  cmx_meas::SemRec& r = cmx_meas::g_semrec;
  r.flags |= (uint8_t)(1u | (agr ? 4u : 0u));
  r.nfresh = (uint8_t)(nk < 255 ? nk : 255);
  r.nnb = (uint8_t)(nb < 255 ? nb : 255);
  r.nsym = (uint8_t)(nu < 255 ? nu : 255);
  const double sm = smax0 > 1.0 ? 1.0 : (smax0 < -1.0 ? -1.0 : smax0);
  r.sim = (int16_t)lrint(sm * 32767.0);
  const double hr = hrate > 1.0 ? 1.0 : (hrate < 0.0 ? 0.0 : hrate);
  r.hrate = (uint16_t)lrint(hr * 65535.0);
  int ord[256];
  for (int u = 0; u < nu; ++u) ord[u] = used[u];
  const int m = nu < 16 ? nu : 16;
  double top = 0.0;
  for (int k = 0; k < m; ++k) {   // partial selection: the heaviest first, ties to the lower byte
    int b = k;
    for (int u = k + 1; u < nu; ++u)
      if (acc[ord[u]] > acc[ord[b]] || (acc[ord[u]] == acc[ord[b]] && ord[u] < ord[b])) b = u;
    const int t = ord[k]; ord[k] = ord[b]; ord[b] = t;
    const int c = ord[k];
    r.sym[k] = (uint8_t)transformer::g_b2i[c];
    const double pv = wsum > 0.0 ? acc[c] / wsum : 0.0;
    r.prob[k] = (uint16_t)lrint((pv > 1.0 ? 1.0 : pv) * 65535.0);
    top += acc[c];
  }
  const double rest = wsum > 0.0 ? (wsum - top) / wsum : 0.0;
  r.resid = (uint16_t)lrint((rest > 1.0 ? 1.0 : (rest < 0.0 ? 0.0 : rest)) * 65535.0);
}
#endif

__attribute__((noinline)) void Predictor::SemByteStart(int strom_pb) {
  using namespace semantic_cache;
  semantic_cache::State& s = sem_;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semdump != nullptr, 0)) memset(&cmx_meas::g_semrec, 0, sizeof cmx_meas::g_semrec);
#endif
  strom_pb &= 0xFF;   // the previous stream byte; recent_bytes_[0] is the last injected one behind an injection
  TreffByteStart(strom_pb);   // the top-1 path stage reads nothing of the cache
  if (!s.key) SemInit();
  s.ufresh = 1;   // layer-0 input: the next byte predicted is this stream byte
  // 1. the previous byte is complete: its entry goes into the memory and
  // replaces the oldest entry (or the first empty slot) of each own bucket
  if (s.pend) {
    const uint32_t ri = (uint32_t)(s.nins & (kN - 1));
    SemPut4(s.key + (size_t)ri * kKB, s.ck);
    s.val[ri] = (uint8_t)strom_pb;
    for (int t = 0; t < kL; ++t) {
      Slot* sl = s.slot + (((size_t)t << kB) + s.chb[t]) * kM;
      const int um = SemOldest(sl);
#if PORT_CMX_MEAS
      if (__builtin_expect(cmx_meas::g_semtel, 0) && sl[um].e1) {   // research CMX_SEMTEL: age of the evicted entry (stream bytes)
        const uint32_t ev = sl[um].e1 - 1u;
        ++cmx_meas::g_sem_evict[cmx_meas::g_region][cmx_meas::Log2Bin((s.spos - 1) - (uint64_t)s.epos[ev & (kN - 1)])];
      }
#endif
      sl[um].e1 = (uint32_t)s.nins + 1u;
      sl[um].s2 = s.c2;
      sl[um].s01 = s.c01;
    }
    s.epos[ri] = (uint32_t)(s.spos - 1);
    ++s.nins;
    s.pend = false;
#if PORT_CMX_MEAS
    if (__builtin_expect(cmx_meas::g_semtel, 0)) ++cmx_meas::g_sem_store[cmx_meas::g_region];
#endif
  }
  // 2. hit of the cache top-1 of the previous byte (if it had neighbours)
  // for the running hit rate; word start, previous byte and letter run of
  // this byte
  if (s.htopb >= 0) {
    const double hit = (s.htopb == strom_pb) ? 1.0 : 0.0;
    s.hrate += (hit - s.hrate) / 64.0;
    s.htopb = -1;
  }
  const uint64_t spos = s.spos++;   // SemByteStart runs once per stream byte in Compress and Decompress
  const uint32_t ut_pos = (uint32_t)spos;
  {
    const int pb = strom_pb;
    s.ws =(pb == 0x20 || pb == 0x0A || pb == '@') ? 1 : 0;
    s.pb = pb;
    // letters and bytes >= 0x80 extend the run
    const bool al = (pb >= 'a' && pb <= 'z') || (pb >= 'A' && pb <= 'Z') || pb >= 0x80;
    s.wlen = al ? (s.wlen < 15 ? s.wlen + 1 : 15) : 0;
    const int pc = (pb >= 'a' && pb <= 'z') ? 0 : (pb >= 'A' && pb <= 'Z') ? 1 : (pb >= '0' && pb <= '9') ? 2 :
                   pb == ' ' ? 3 : pb == 10 ? 4 : pb >= 0x80 ? 5 : (pb == '[' || pb == ']' || pb == '|') ? 6 : 7;
    const int w = s.wlen;
    const int wl = w == 0 ? 0 : w < 3 ? 1 : w < 5 ? 2 : w < 7 ? 3 : w < 9 ? 4 : w < 11 ? 5 : w < 14 ? 6 : 7;
    s.ecl = pc * 8 + wl;
  }
  // 3. the previous byte goes into the history; every followed neighbour
  // whose continuation differs from it is dead
  if (spos > 0) {
    const uint8_t pb = (uint8_t)strom_pb;
    s.hist[(spos - 1) & (kHist - 1)] = pb;
    for (int g = 0; g <= kG; ++g) {
      if (s.fn[g] <= 0) continue;
      const uint64_t t = spos - s.fq[g];
      if (t > (uint64_t)kWtMax) { s.fn[g] = 0; continue; }
      for (int j = 0; j < s.fn[g]; ++j)
        if (s.fal[g][j] && s.hist[(s.fvp[g][j] + t - 1) & (kHist - 1)] != pb) s.fal[g][j] = 0;
    }
  }
  s.nh[spos & 4095] = (uint32_t)s.nins;   // entries with a stream position <= spos - 1
  s.nn = 0;
  s.cell = -1;
#if PORT_CMX_MEAS
  // research CMX_SEMCACHE_OFF: the path of a byte without a live transformer row at every byte (no lookup, no store)
  if (cmx_meas::g_semoff || !(manager_.tf1_gate_ctx1 == 0 && tf1_slot_->KtxLive())) {
#else
  if (!(manager_.tf1_gate_ctx1 == 0 && tf1_slot_->KtxLive())) {
#endif
    for (int g = 0; g <= kG; ++g) s.fn[g] = 0;
    return;
  }
  // 4. top-1 of the transformer row (that of the top-1 path stage) and its
  // probability
  const float* t = &tf1_slot_->BytePredict()[0];
  const int t1 = treff_.a1;   // top-1 of this live row (TreffByteStart, above)
  double sum = 0.0, rest = 0.0;
  for (int i = 0; i < 256; ++i) { sum += (double)t[i]; if (i != t1) rest += (double)t[i]; }
  if (!(sum > 0.0)) return;
  const float cmiss = rest > 0.0 ? (float)(-semantic_cache::Log2(rest / sum)) : 64.0f;
  const double p1 = 1.0 - exp2(-(double)cmiss);
  const bool q_ok = p1 < (double)kQpE3 / 1000.0;   // looked up
  const bool s_ok = p1 < (double)kSpE3 / 1000.0;   // stored
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semdump != nullptr, 0) && s_ok) cmx_meas::g_semrec.flags |= 8u;
#endif
  if (!q_ok && !s_ok) return;
  // 5. key of this byte: the residual behind layer NL-2 minus its running
  // mean (frozen after kCen counted lookups), int8, +-1 projection, int8
  // again; table hashes (moved by the previous byte, with the order of the
  // hash bits by certainty) and sketch
  const float* x = tf1_engine_->LastKey();
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semtel, 0)) {   // research CMX_SEMTEL: keyed bytes; raw key moments on article text
    ++cmx_meas::g_sem_keyed[cmx_meas::g_region];
    if (q_ok && cmx_meas::g_region >= 2) {
      for (int c = 0; c < 192; ++c) { cmx_meas::g_sem_ksum[c] += (double)x[c]; cmx_meas::g_sem_ksq += (double)x[c] * (double)x[c]; }
      ++cmx_meas::g_sem_kn;
    }
  }
#endif
  float xc[192];
  {
    const bool kc_ok = p1 < (double)kCalqE3 / 1000.0;
#if PORT_CMX_MEAS
    const uint32_t cmx_kcen = cmx_meas::g_sem_kcen ? cmx_meas::g_sem_kcen : kCen;   // research CMX_SEM_KCEN
    if (kc_ok && s.kcnt < cmx_kcen) {
#else
    if (kc_ok && s.kcnt < kCen) {
#endif
      for (int c = 0; c < 192; ++c) s.kmu[c] += (double)x[c];
      ++s.kcnt;
    }
    const double inv = s.kcnt ? 1.0 / (double)s.kcnt : 0.0;
#if PORT_CMX_MEAS
    if (s.kcnt < cmx_kcen) {
#else
    if (s.kcnt < kCen) {
#endif
      for (int c = 0; c < 192; ++c) xc[c] = x[c] - (float)(s.kmu[c] * inv);
    } else {   // frozen: the float mean once
      if (!s.kmf) {
        for (int c = 0; c < 192; ++c) s.kmuf[c] = (float)(s.kmu[c] * inv);
        s.kmf = 1;
      }
      for (int c = 0; c < 192; ++c) xc[c] = x[c] - s.kmuf[c];
    }
    x = xc;
  }
  alignas(32) int8_t q[192];
  {
    float m = 0.0f;
    for (int c = 0; c < 192; ++c) { const float a = fabsf(x[c]); if (a > m) m = a; }
    // x / (m / 127) as the compiler evaluates it: x * (127 / m); clamped to
    // +-127, rounded to nearest (as lrintf), eight lanes at a time
    const __m256 rv = _mm256_set1_ps(m > 0.0f ? 127.0f / m : 1.0f);
    const __m256 hi = _mm256_set1_ps(127.0f), lo = _mm256_set1_ps(-127.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int c = 0; c < 192; c += 32) {
      __m256i v[4];
      for (int i = 0; i < 4; ++i)
        v[i] = _mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(x + c + 8 * i), rv), hi), lo));
      const __m256i b = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
      _mm256_store_si256((__m256i*)(q + c), _mm256_permutevar8x32_epi32(b, perm));
    }
  }
  int32_t y[kD];
  SemDotPm1<192>(&s.proj[0][0], kD, q, y);
  int32_t ym = 0;
  for (int a = 0; a < kD; ++a) {
    const int32_t av = y[a] < 0 ? -y[a] : y[a];
    if (av > ym) ym = av;
  }
  uint32_t n2 = 0;
  {
    // (y * 127) / ym truncated toward zero, as a multiplication with
    // ceil(2^37 / ym): exact for |y| * 127 < 2^22 and ym < 2^15
    static_assert(192 * 127 * 127 < (1 << 22) && 192 * 127 < (1 << 15), "key range");
    const uint64_t mr = ym > 0 ? ((((uint64_t)1) << 37) + (uint64_t)ym - 1) / (uint64_t)ym : 0;
    for (int a = 0; a < kD; ++a) {
      const uint32_t u = (uint32_t)(y[a] < 0 ? -y[a] : y[a]) * 127u;
      const int32_t m = (int32_t)(((uint64_t)u * mr) >> 37);
      const int32_t v = y[a] < 0 ? -m : m;
      s.ck[a] = (int8_t)v;
      n2 += (uint32_t)(v * v);
    }
  }
  size_t bix[kL * kProbe];
  {
    const uint32_t part = (uint32_t)((((uint64_t)strom_pb + 1u) * 0x9E3779B97F4A7C15ull) >> (64 - kB));
    int32_t dh[kL * kB];
    SemDotPm1<kD>(&s.hyp[0][0], kL * kB, s.ck, dh);
    for (int tb = 0; tb < kL; ++tb) {
      uint32_t h = 0;
      int32_t ad[kB];   // |dot| per bit position (position kB-1-j for hyperplane j)
      for (int j = 0; j < kB; ++j) {
        const int32_t d = dh[tb * kB + j];
        h = (h << 1) | (d > 0 ? 1u : 0u);
#if PORT_CMX_MEAS
        if (__builtin_expect(cmx_meas::g_semtel, 0)) cmx_meas::g_sem_hyp1[tb * kB + j] += d > 0 ? 1u : 0u;   // research CMX_SEMTEL
#endif
        ad[kB - 1 - j] = d < 0 ? -d : d;
      }
      h ^= part;
      s.chb[tb] = h;
      // the kFlip least certain positions, ties to the lower position
      int ord[kB];
      for (int pp = 0; pp < kB; ++pp) {
        int j = pp;
        while (j > 0 && ad[ord[j - 1]] > ad[pp]) { ord[j] = ord[j - 1]; --j; }
        ord[j] = pp;
      }
      bix[tb * kProbe] = ((size_t)tb << kB) + h;
      for (int fl = 1; fl < kProbe; ++fl) bix[tb * kProbe + fl] = ((size_t)tb << kB) + (h ^ (1u << ord[fl - 1]));
    }
  }
  // the first buckets of the scan load while the sketch is computed
  if (q_ok)
    for (int tp = 0; tp < kPf; ++tp) SemPfBucket(s.slot + bix[tp] * kM);
  uint64_t q01 = 0;
  uint32_t q2 = 0;
  {
    int32_t dk[kSk];
    SemDotPm1<kD>(&s.skp[0][0], kSk, s.ck, dk);
    for (int j = 0; j < kSk; ++j) {
      if (dk[j] > 0) {
        if (j < 64) q01 |= 1ull << j; else q2 |= 1u << (j - 64);
      }
    }
  }
  s.c01 = q01;
  s.c2 = q2;
  s.pend = s_ok;
  if (!q_ok) return;   // stored, not looked up
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semdump != nullptr, 0)) cmx_meas::g_semrec.flags |= 2u;
#endif
  // 6. query: the buckets in table order (own, then the flipped ones),
  // sketch distances of the valid entries at least kGap stream bytes back;
  // counting sort by distance keeps the visiting order on ties, an entry met
  // again is skipped; the kPre first are re-ranked by the int4 cosine
  constexpr int kNB = kL * kProbe;
  const uint64_t oldest = s.nins > kN ? s.nins - kN : 0;
  const uint64_t enear = spos + 1 >= (uint64_t)kGap ? s.nh[(spos + 1 - kGap) & 4095] : 0;
  int nc = 0;
  int cnt[kSk + 2];
  for (int h = 0; h < kSk + 2; ++h) cnt[h] = 0;
  for (int tp = 0; tp < kNB; ++tp) {
    if (tp + kPf < kNB) SemPfBucket(s.slot + bix[tp + kPf] * kM);
    const Slot* sl = s.slot + bix[tp] * kM;
    for (int u = 0; u < kM; ++u) {
      const uint32_t e1 = sl[u].e1;
      if (e1 == 0u || (uint64_t)(e1 - 1u) < oldest) continue;
      if ((uint64_t)(e1 - 1u) >= enear) continue;
      const int hm = __builtin_popcountll(q01 ^ sl[u].s01) + __builtin_popcount(q2 ^ sl[u].s2);
      s.cand_hm[nc] = (uint8_t)hm;
      s.cand_e[nc] = e1 - 1u;
      ++cnt[hm + 1];
      ++nc;
    }
  }
  for (int h = 1; h < kSk + 2; ++h) cnt[h] += cnt[h - 1];   // cnt[h] = start of distance h
  for (int i = 0; i < nc; ++i) s.ord_e[cnt[s.cand_hm[i]]++] = s.cand_e[i];
  // after the placement cnt[h] is the end of distance h
  // an entry met again comes from another table with the same sketch, so it
  // has the same distance: the selected entries are a set (open addressing,
  // slots of other lookups carry an older stamp)
  int npre = 0;
  uint32_t pid[kPre];
  {
    if (++s.dstamp == 0) {   // wrap: forget all stamps
      for (int x = 0; x < kDedup; ++x) s.dst[x] = 0;
      s.dstamp = 1;
    }
    const uint32_t st = s.dstamp;
    const int nall = cnt[kSk];   // the end of the last distance = all candidates
    for (int i = 0; i < nall && npre < kPre; ++i) {
      const uint32_t e = s.ord_e[i];
      for (uint32_t x = (e * 0x9E3779B1u) >> (32 - kDedupLog);; x = (x + 1) & (kDedup - 1)) {
        if (s.dst[x] != st) {
          s.dst[x] = st;
          s.dke[x] = e;
          pid[npre++] = e;
          __builtin_prefetch(s.key + (size_t)(e & (kN - 1)) * kKB);   // its ring key, for the re-rank
          break;
        }
        if (s.dke[x] == e) break;
      }
    }
  }
  int nk = 0;
  uint32_t kid[kK];
  double ks[kK];
  const double qn = n2 > 0 ? 1.0 / sqrt((double)n2) : 0.0;
  // int4 cosine, exact in integers: the even key components meet the low
  // nibbles, the odd ones the high nibbles; vpmaddubsw takes |component| and
  // the nibble with the component's sign (|pair| <= 2 x 127 x 8)
  static_assert(kKB == 64, "two 32-byte halves per ring key");
  __m256i cev[2], cov[2], aev[2], aov[2];
  {
    alignas(32) int8_t ce[kKB], co[kKB];
    for (int c = 0; c < kKB; ++c) { ce[c] = s.ck[2 * c]; co[c] = s.ck[2 * c + 1]; }
    for (int v = 0; v < 2; ++v) {
      cev[v] = _mm256_load_si256((const __m256i*)(ce + 32 * v));
      cov[v] = _mm256_load_si256((const __m256i*)(co + 32 * v));
      aev[v] = _mm256_abs_epi8(cev[v]);
      aov[v] = _mm256_abs_epi8(cov[v]);
    }
  }
  const __m256i m0f = _mm256_set1_epi8(0x0F), m08 = _mm256_set1_epi8(8), one16 = _mm256_set1_epi16(1);
  for (int a = 0; a < npre; ++a) {
    const uint32_t ri = pid[a] & (kN - 1);
    const uint8_t* kk = s.key + (size_t)ri * kKB;
    __m256i da = _mm256_setzero_si256(), na = _mm256_setzero_si256();
    for (int v = 0; v < 2; ++v) {
      const __m256i x = _mm256_loadu_si256((const __m256i*)(kk + 32 * v));
      const __m256i lo = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(x, m0f), m08), m08);
      const __m256i hi = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(x, 4), m0f), m08), m08);
      da = _mm256_add_epi16(da, _mm256_maddubs_epi16(aev[v], _mm256_sign_epi8(lo, cev[v])));
      da = _mm256_add_epi16(da, _mm256_maddubs_epi16(aov[v], _mm256_sign_epi8(hi, cov[v])));
      const __m256i al = _mm256_abs_epi8(lo), ah = _mm256_abs_epi8(hi);
      na = _mm256_add_epi16(na, _mm256_maddubs_epi16(al, al));
      na = _mm256_add_epi16(na, _mm256_maddubs_epi16(ah, ah));
    }
    const __m256i t = _mm256_hadd_epi32(_mm256_madd_epi16(da, one16), _mm256_madd_epi16(na, one16));
    const __m128i u = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
    const __m128i w = _mm_hadd_epi32(u, u);
    const int32_t d = _mm_cvtsi128_si32(w);
    const uint32_t kn2 = (uint32_t)_mm_extract_epi32(w, 1);
    const double sim = kn2 > 0 ? d * qn / sqrt((double)kn2) : 0.0;
    if (nk < kK || sim > ks[nk - 1]) {
      int j = nk < kK ? nk++ : kK - 1;
      while (j > 0 && ks[j - 1] < sim) { ks[j] = ks[j - 1]; kid[j] = kid[j - 1]; --j; }
      ks[j] = sim;
      kid[j] = ri;
    }
  }
  // 7. followers: a lookup below kGq starts a generation (generation 0 = the
  // fresh neighbours, the last one moves to the newest earlier generation);
  // a lookup above reads the followers as if it started one, but the
  // generations stay as they are
  const bool ngen = !(p1 < (double)kGqE3 / 1000.0);
  if (!ngen) {
    if (s.fn[0] > 0) {
      s.ghead = s.ghead % kG + 1;   // 1..kG
      const int g = s.ghead;
      s.fn[g] = s.fn[0];
      s.fq[g] = s.fq[0];
      for (int j = 0; j < s.fn[0]; ++j) { s.fvp[g][j] = s.fvp[0][j]; s.fsim[g][j] = s.fsim[0][j]; s.fal[g][j] = s.fal[0][j]; }
    }
    s.fn[0] = nk;
    s.fq[0] = spos;
    // a continuation outside the history ring cannot be followed (the value
    // at this lookup stays)
    for (int j = 0; j < nk; ++j) {
      s.fvp[0][j] = (uint64_t)s.epos[kid[j]];
      s.fsim[0][j] = ks[j];
      s.fal[0][j] = (spos - s.fvp[0][j]) + (uint64_t)kWtMax < kHist ? 1 : 0;
    }
  }
  // followed generations of this lookup, newest first (kG of them): with a
  // starting lookup 1..kG; otherwise generation 0 first (if any), as pushed
  int gl[kG], ngl = 0;
  if (ngen && s.fn[0] > 0) gl[ngl++] = 0;
  for (int gi = 0; ngl < kG; ++gi) gl[ngl++] = (s.ghead - 1 - gi + kG) % kG + 1;
  bool anyf = false;
  for (int gi = 0; gi < kG; ++gi) if (s.fn[gl[gi]] > 0) anyf = true;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semtel, 0)) {   // research CMX_SEMTEL: lookups, with >= 1 fresh neighbour
    ++cmx_meas::g_sem_look[cmx_meas::g_region];
    if (nk > 0) ++cmx_meas::g_sem_look_nk[cmx_meas::g_region];
  }
#endif
  if (nk == 0 && !anyf) return;
  // 8. the main readout: fresh neighbours weigh exp((s - s_max) / T), the
  // followed ones the byte behind their continuation with exp((s - s_max of
  // their lookup) / T); the end stage reads every neighbour with its
  // similarity relative to the s_max of its own lookup and its age
  const double tinv = 1000.0 / (double)kTE3;
  double acc[256];
  int used[kNb], nu = 0;
  uint8_t has[256] = {};   // bytes in used[]
  s.wsum = 0.0;
  int nb = 0;
  for (int j = 0; j < nk; ++j) {
    const int c = s.val[kid[j]];
    const double w = exp((ks[j] - ks[0]) * tinv);
    s.nbyte[nb] = (uint8_t)c;
    s.nw[nb] = w;
    s.asim[nb] = ks[j] - ks[0];
    s.aage[nb] = ut_pos - s.epos[kid[j]];
#if PORT_CMX_MEAS
    if (__builtin_expect(cmx_meas::g_semtel, 0)) ++cmx_meas::g_sem_nage[cmx_meas::g_region][cmx_meas::Log2Bin(s.aage[nb])];
#endif
    ++nb;
    s.wsum += w;
    if (has[c]) acc[c] += w;
    else { has[c] = 1; used[nu++] = c; acc[c] = w; }
  }
  double smax0 = nk > 0 ? ks[0] : 0.0;
  bool smset = nk > 0;
  s.nfr = nb;
  for (int gi = 0; gi < kG; ++gi) {
    const int g = gl[gi];
    if (s.fn[g] <= 0) continue;
    if (!smset) { smax0 = s.fsim[g][0]; smset = true; }   // no fresh neighbour: the newest followed lookup
    const uint64_t t = spos - s.fq[g];
    for (int j = 0; j < s.fn[g]; ++j) {
      if (!s.fal[g][j] || s.fvp[g][j] + t >= spos) continue;
      const int c = s.hist[(s.fvp[g][j] + t) & (kHist - 1)];
      const double w = exp((s.fsim[g][j] - s.fsim[g][0]) * tinv);
      s.nbyte[nb] = (uint8_t)c;
      s.nw[nb] = w;
      s.asim[nb] = s.fsim[g][j] - s.fsim[g][0];
      s.aage[nb] = (uint32_t)(s.fq[g] - s.fvp[g][j]);
      ++nb;
      s.wsum += w;
      if (has[c]) acc[c] += w;
      else { has[c] = 1; used[nu++] = c; acc[c] = w; }
    }
  }
  if (nb == 0) return;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semtel, 0)) ++cmx_meas::g_sem_look_nb[cmx_meas::g_region];   // research CMX_SEMTEL: >= 1 neighbour incl. followed
#endif
  s.nn = nb;
  int topb = used[0];
  for (int u = 1; u < nu; ++u) {
    const int c = used[u];
    if (acc[c] > acc[topb] || (acc[c] == acc[topb] && c < topb)) topb = c;
  }
  // s_max of the lookup (smax0) gives the similarity class
  s.cls = smax0 < (double)kC1E3 / 1000.0 ? 1 : (smax0 < (double)kC2E3 / 1000.0 ? 2 : 3);
  s.agr = (topb == t1) ? 1 : 0;
#if PORT_CMX_MEAS
  // research CMX_SEMTOP: the share class of the vote's top share acc[top-1] / wsum (all neighbours of this byte; the
  // same value as SemRec.prob[0] of CMX_SEMDUMP); SemUtByte puts it into the cells
  if (__builtin_expect(cmx_meas::g_semtop != 0, 0)) cmx_stq_ = cmx_meas::SemTopShare(acc[topb], s.wsum, cmx_meas::g_semtop);
  // research CMX_SEMIN: the top share itself (the inputs x1 = t * st(p_sem) of this byte's bits)
  if (__builtin_expect(cmx_meas::g_semin != 0, 0)) cmx_tsh_ = s.wsum > 0.0 ? acc[topb] / s.wsum : 0.0;
  if (__builtin_expect(cmx_meas::g_semdump != nullptr, 0))
    CmxSemCapture(used, nu, acc, s.wsum, smax0, nb, nk, s.agr, s.hrate);
#endif
  // BITLSTM column: the similarity of the lookup (s_max)
  {
    const double sm = (smax0 - 0.9) * 10.0;
    s.b_smax = (float)(sm > 1.0 ? 1.0 : (sm < -1.0 ? -1.0 : sm));
  }
  SemUtByte(topb);
}

// byte level of the end stage: cache top-1, age class of the nearest
// neighbour, agreement of the match byte, the fast-weight top-1 and the PPMd
// top-1 with the cache top-1, weights of the other readouts
__attribute__((noinline)) void Predictor::SemUtByte(int topb) {
  semantic_cache::State& s = sem_;
  s.topb = topb;
  s.htopb = topb;
  s.wfg = 0;
  const uint32_t a0 = s.aage[0];
  s.gcl = a0 < 2048u ? 0 : (a0 < 1048576u ? 1 : 2);
  // match byte and the fast-weight and PPMd top-1 as the top-1 path stage
  // found them at this byte start (a byte with neighbours has a live row
  // with a positive top-1)
  const cache_hit::State& tr = treff_;
  s.mtop = (tr.mlen > 0 && tr.mbyte == topb) ? 1 : 0;
  s.ftop = (tr.fa == topb) ? 1 : 0;
  s.ptop = (tr.pa == topb) ? 1 : 0;
  const double tinv = 1000.0 / (double)semantic_cache::kTE3;
  s.hws = 0.0;
  for (int j = 0; j < s.nn; ++j) { s.hw[j] = exp(s.asim[j] * 4.0 * tinv); s.hws += s.hw[j]; }
  // the readouts at temperature T weigh exp(asim / T), which is nw (same
  // product, same exp); counts and weight sums of the near and far neighbours
  s.n_n = s.n_f = 0;
  s.ws_n = s.ws_f = 0.0;
  for (int j = 0; j < s.nn; ++j) {
    if (s.aage[j] < semantic_cache::kUAge) { ++s.n_n; s.ws_n += s.nw[j]; }
    else { ++s.n_f; s.ws_f += s.nw[j]; }
  }
  // cells of the three stages: the byte part (without / with a consistent
  // neighbour; similarity class and agreement count only with one)
  s.uo1[0] = s.ws * 3 + s.gcl;
  s.uo1[1] = (s.cls * 2 + s.agr) * 6 + s.ws * 3 + s.gcl;
  s.uo3[0] = s.ecl;
  s.uo3[1] = (s.cls * 2 + s.agr) * 64 + s.ecl;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semtop != 0, 0)) CmxSemTopCells();   // research CMX_SEMTOP: the class part by the vote
#endif
  // part sums: the neighbours grouped by byte value (groups in byte order),
  // per group the masses of the seven readouts; the readouts of a bit add the
  // groups consistent with its prefix (SemPredBit) instead of looping over all
  // neighbours per readout
  {
    uint64_t bm[4] = {0, 0, 0, 0};
    for (int j = 0; j < s.nn; ++j) bm[s.nbyte[j] >> 6] |= 1ull << (s.nbyte[j] & 63);
    int g = 0;
    for (int w = 0; w < 4; ++w)
      for (uint64_t bb = bm[w]; bb; bb &= bb - 1) {
        const int c = w * 64 + __builtin_ctzll(bb);
        s.gb[g] = (uint8_t)c;
        s.gix[c] = (uint8_t)g;
        ++g;
      }
    s.ngr = g;
    memset(s.gm, 0, sizeof(s.gm[0]) * (size_t)g);
    for (int j = 0; j < s.nn; ++j) {
      double* m = s.gm[s.gix[s.nbyte[j]]];
      m[0] += s.nw[j];
      m[1] += s.hw[j];
      m[j < s.nfr ? 2 : 3] += s.nw[j];
      m[s.aage[j] < semantic_cache::kUAge ? 4 : 5] += s.nw[j];
      m[6] += 1.0;
    }
  }
}

// the end stage of one coded bit: 20 inputs (coder log-odds; the cache
// log-odds of the main readout, bias, the same times the consistent mass;
// transformer, fast-weight, PPMd and fxcm bit log-odds; the cache log-odds
// sharp (T / 4), with equal weights, of the nearest neighbour, at
// temperature T, of the near and of the far neighbours; vote of the cache
// top-1 x match / fast-weight / PPMd top-1 agreeing; the cache log-odds x
// running hit rate; the readout of the fresh and of the followed neighbours)
// in three logistic stages (bit x similarity x agreement x word start x age
// class | bit x previous byte | bit x similarity x agreement x previous-byte
// class x letter-run class), mixed per bit position
__attribute__((noinline)) uint32_t Predictor::SemUtCorrect(uint32_t p32) {
  using namespace semantic_cache;
  semantic_cache::State& s = sem_;
  const unsigned int ctx = manager_.bit_context_;
  const int k = 31 - __builtin_clz(ctx | 1u);
  double* f = s.uf;
  const double conf = s.rc;   // the main readout of this bit (SemPredBit)
  const double xk = s.rx;
  const bool info = conf > 0.0;
  f[0] = SemClamp(log((double)p32 / (4294967296.0 - (double)p32)), 30.0);
  f[1] = xk;
  f[2] = 1.0;
  f[3] = xk * conf;
  f[4] = SemClamp(SemUtLogitP((double)tf1_slot_output), 30.0);
  f[5] = SemClamp(SemUtLogitP((double)fw_slot_output), 30.0);
  f[6] = SemClamp((double)layers_[0].Inputs()[mxs_ppmd_ix_], 30.0);
  f[7] = SemClamp((double)adepth_.f[2], 30.0);
  {   // readouts from the part sums of this bit (SemPredBit)
    f[8] = SemUtReadM(s.rm[1], s.hws);
    f[9] = SemUtReadC((int)s.rm[6][0], (int)s.rm[6][1], s.nn);
    {   // the nearest neighbour alone: 0 or one of two log-odds (SemInit)
      const unsigned int c = 0x100u | s.nbyte[0];
      f[10] = ((c >> (8 - k)) != ctx) ? 0.0 : s.l10[(c >> (7 - k)) & 1u];
    }
    f[11] = xk;   // temperature T: the weights and their sum are nw and wsum
    f[12] = s.n_n ? SemUtReadM(s.rm[4], s.ws_n) : 0.0;
    f[13] = s.n_f ? SemUtReadM(s.rm[5], s.ws_f) : 0.0;
  }
  {
    const unsigned int tb = (unsigned int)s.topb;
    const double vote = ((((0x100u | tb) >> (8 - k))) == ctx) ? (((tb >> (7 - k)) & 1u) ? 1.0 : -1.0) : 0.0;
    f[14] = vote * (double)s.mtop;
    f[15] = vote * (double)s.ftop;
    f[16] = vote * (double)s.ptop;
  }
  f[17] = info ? xk * (s.hrate - 0.4) : 0.0;
  {
    const int nf = s.nfr, ng = s.nn - s.nfr;
    if (!s.wfg) {   // once per byte
      double wf = 0.0, wg = 0.0;
      for (int j = 0; j < nf; ++j) wf += s.nw[j];
      for (int j = nf; j < s.nn; ++j) wg += s.nw[j];
      s.wf = wf;
      s.wg = wg;
      s.wfg = 1;
    }
    f[18] = nf > 0 ? SemUtReadM(s.rm[2], s.wf) : 0.0;
    f[19] = ng > 0 ? SemUtReadM(s.rm[3], s.wg) : 0.0;
  }
#if PORT_CMX_MEAS
  // research CMX_SEMTOP (stages 1 and 3 from the wide tables) / CMX_SEMIN (two more inputs): the research tail
  if (__builtin_expect(cmx_meas::g_semx, 0)) return CmxSemTail(k, info ? 1 : 0);
#endif
  // cell = ((bit x 4 + similarity class) x 2 + agreement) x ... (class and
  // agreement only with a consistent neighbour); the byte part from SemUtByte
  const int c1 = k * 48 + s.uo1[info], c2 = k * 256 + s.pb, c3 = k * 512 + s.uo3[info];
  double z1 = 0.0, z2 = 0.0, z3 = 0.0;
  for (int i = 0; i < kUIn; ++i) { z1 += s.u4w[c1][i] * f[i]; z2 += s.u5w[c2][i] * f[i]; z3 += s.u6w[c3][i] * f[i]; }
  s.uz[0] = SemClamp(z1, 30.0);
  s.uz[1] = SemClamp(z2, 30.0);
  s.uz[2] = SemClamp(z3, 30.0);
  s.uc[0] = c1;
  s.uc[1] = c2;
  s.uc[2] = c3;
  const double* fw = s.ufw[k];
  s.ufz = SemClamp(fw[0] * s.uz[0] + fw[1] * s.uz[1] + fw[2] * s.uz[2] + fw[3], 30.0);
  s.cell = k;
  s.ufe = exp(-s.ufz);   // the learning step of the final stage reads it
  double v = floor(1.0 / (1.0 + s.ufe) * 4294967296.0 + 0.5);
  if (v < 1.0) v = 1.0;
  if (v > 4294967295.0) v = 4294967295.0;
  return (uint32_t)v;
}

__attribute__((noinline)) void Predictor::SemUtLearn(int bit) {
  using namespace semantic_cache;
  semantic_cache::State& s = sem_;
  if (s.cell < 0) return;
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_semx, 0)) { CmxSemLearn(bit); return; }   // research CMX_SEMTOP / CMX_SEMIN
#endif
  const double lr = (double)kLrE6 * 1e-6;
  SemUtStep(s.u4w[s.uc[0]], s.u4g[s.uc[0]], s.uf, kUIn, s.uz[0], bit, lr);
  SemUtStep(s.u5w[s.uc[1]], s.u5g[s.uc[1]], s.uf, kUIn, s.uz[1], bit, lr);
  SemUtStep(s.u6w[s.uc[2]], s.u6g[s.uc[2]], s.uf, kUIn, s.uz[2], bit, lr);
  {
    const double ff[4] = {s.uz[0], s.uz[1], s.uz[2], 1.0};
    SemUtStepE(s.ufw[s.cell], s.ufg[s.cell], ff, 4, s.ufe, bit, lr * 0.2);
  }
  s.cell = -1;
}

#if PORT_CMX_MEAS
// ---- research CMX_SEMTOP (src/cmx_meas.h; Claude Opus 5.5, 2026-11-07) ---------------------------------------------
// The wide tables of stages 1 and 3: 8 x g_st_s1 and 8 x g_st_s3 rows of kUIn weights plus their AdaGrad sums; every
// row starts at "coder unchanged" (weight 1 on the coder log-odds), as SemInit sets the release rows.
__attribute__((noinline)) void Predictor::CmxSemTopInit() {
  using semantic_cache::kUIn;
  const size_t r4 = (size_t)8 * (size_t)cmx_meas::g_st_s1, r6 = (size_t)8 * (size_t)cmx_meas::g_st_s3;
  double* t = (double*)calloc((r4 + r6) * 2 * (size_t)kUIn, sizeof(double));
  if (!t) {
    fprintf(stderr, "CMX semtop: no memory for the wide end-stage tables\n");
    exit(1);
  }
  cmx_st_w4_ = t;
  cmx_st_g4_ = cmx_st_w4_ + r4 * kUIn;
  cmx_st_w6_ = cmx_st_g4_ + r4 * kUIn;
  cmx_st_g6_ = cmx_st_w6_ + r6 * kUIn;
  for (size_t c = 0; c < r4; ++c) cmx_st_w4_[c * kUIn] = 1.0;
  for (size_t c = 0; c < r6; ++c) cmx_st_w6_[c * kUIn] = 1.0;
  fprintf(stderr, "CMXTEL semtop init mode=%d levels=%d stage1_rows=%zu stage3_rows=%zu table_bytes=%zu\n",
          cmx_meas::g_semtop, cmx_meas::g_st_lv, r4, r6, (r4 + r6) * 2 * (size_t)kUIn * sizeof(double));
  fflush(stderr);
}
// SemUtByte's class part of the cells with a consistent neighbour: the class of the vote in place of the similarity
// class (level 0, the cells without a consistent neighbour, stays as released)
__attribute__((noinline)) void Predictor::CmxSemTopCells() {
  semantic_cache::State& s = sem_;
  const int vc = cmx_meas::SemTopClass(s.cls, cmx_stq_, cmx_meas::g_semtop);
  ++cmx_meas::g_st_cnt[cmx_meas::g_region][vc];
  s.uo1[1] = (vc * 2 + s.agr) * 6 + s.ws * 3 + s.gcl;
  s.uo3[1] = (vc * 2 + s.agr) * 64 + s.ecl;
}
// research CMX_SEMIN: two extra weights (and AdaGrad sums) per cell of every end-stage stage, zero (calloc): stage 1 rows
// [0, 8 x g_st_s1), stage 2 rows [o2, o2 + kUC5), stage 3 rows [o3, o3 + 8 x g_st_s3); two doubles per row
__attribute__((noinline)) void Predictor::CmxSemInInit() {
  const size_t r1 = (size_t)8 * (size_t)cmx_meas::g_st_s1, r2 = (size_t)semantic_cache::kUC5,
               r3 = (size_t)8 * (size_t)cmx_meas::g_st_s3;
  double* t = (double*)calloc((r1 + r2 + r3) * 4, sizeof(double));
  if (!t) {
    fprintf(stderr, "CMX semin: no memory for the extra end-stage weights\n");
    exit(1);
  }
  cmx_si_w_ = t;
  cmx_si_g_ = t + (r1 + r2 + r3) * 2;
  cmx_si_o2_ = r1;
  cmx_si_o3_ = r1 + r2;
  fprintf(stderr, "CMXTEL semin init mode=%d stage_rows=%zu/%zu/%zu final_mixer_cols=%s\n", cmx_meas::g_semin, r1, r2, r3,
          cmx_meas::g_semin == 2 ? "24,25" : "-");
  fflush(stderr);
}
// research CMX_SEMIN=2: the two final-mixer columns of this bit, x1 = t * c and x2 = agree * c with c = the cache's
// log-odds column of the row (SemSetStage1: the layer-0 cache input, 0 without neighbours or on an injected byte)
__attribute__((noinline)) void Predictor::CmxSemInCols() {
  const semantic_cache::State& s = sem_;
  const float c = layers_[0].Inputs()[sem_index_];
  const bool on = s.nn > 0;
  layers_[1].SetStretchedInput(cmx_si_col_, on ? (float)(cmx_tsh_ * (double)c) : 0.0f);
  layers_[1].SetStretchedInput(cmx_si_col_ + 1, (on && s.agr) ? c : 0.0f);
}
// research CMX_SEMIN report (zero-byte): mean |w| and mean w of the extra weights over the touched cells per stage;
// with =2 the two final-mixer weights of the gate-zone row and of the add row the last bit resolved
__attribute__((noinline)) void Predictor::CmxSemInReport(const char* tag) {
  if (!cmx_si_w_) return;
  const size_t lo[3] = {0, cmx_si_o2_, cmx_si_o3_};
  const size_t hi[3] = {cmx_si_o2_, cmx_si_o3_, cmx_si_o3_ + (size_t)8 * (size_t)cmx_meas::g_st_s3};
  for (int st = 0; st < 3; ++st) {
    unsigned long long n = 0;
    double a[2] = {0, 0}, m[2] = {0, 0};
    for (size_t r = lo[st]; r < hi[st]; ++r) {
      if (!(cmx_si_g_[r * 2] > 0.0 || cmx_si_g_[r * 2 + 1] > 0.0)) continue;
      ++n;
      for (int j = 0; j < 2; ++j) { a[j] += fabs(cmx_si_w_[r * 2 + j]); m[j] += cmx_si_w_[r * 2 + j]; }
    }
    if (!n) continue;
    fprintf(stderr, "CMXTEL semin %s stage=%d touched_cells=%llu x1_mean_abs_w=%.6g x1_mean_w=%.6g x2_mean_abs_w=%.6g "
            "x2_mean_w=%.6g\n", tag, st + 1, n, a[0] / n, m[0] / n, a[1] / n, m[1] / n);
  }
  if (cmx_meas::g_semin == 2)
    fprintf(stderr, "CMXTEL semin %s final_mixer zone_row_w=%.6g,%.6g add_row_w=%.6g,%.6g\n", tag,
            (double)mixer_1_[0].E2EWRow()[cmx_si_col_], (double)mixer_1_[0].E2EWRow()[cmx_si_col_ + 1],
            (double)mixer_1_[mix_select::kIa].E2EWRow()[cmx_si_col_],
            (double)mixer_1_[mix_select::kIa].E2EWRow()[cmx_si_col_ + 1]);
  fflush(stderr);
}
// the tail of SemUtCorrect (s.uf filled) under CMX_SEMTOP (stages 1 and 3 in the wide tables) and / or CMX_SEMIN (two
// more inputs: stage 1, or all three stages with =3); stage 2 and the final stage otherwise as released
__attribute__((noinline)) uint32_t Predictor::CmxSemTail(int k, int info) {
  using semantic_cache::kUIn;
  semantic_cache::State& s = sem_;
  const double* f = s.uf;
  const int c1 = k * cmx_meas::g_st_s1 + s.uo1[info], c2 = k * 256 + s.pb, c3 = k * cmx_meas::g_st_s3 + s.uo3[info];
  const double* w1 = cmx_meas::g_semtop ? cmx_st_w4_ + (size_t)c1 * kUIn : s.u4w[c1];
  const double* w3 = cmx_meas::g_semtop ? cmx_st_w6_ + (size_t)c3 * kUIn : s.u6w[c3];
  double z1 = 0.0, z2 = 0.0, z3 = 0.0;
  for (int i = 0; i < kUIn; ++i) { z1 += w1[i] * f[i]; z2 += s.u5w[c2][i] * f[i]; z3 += w3[i] * f[i]; }
  if (cmx_meas::g_semin) {   // x1 = t * f[1], x2 = agree * f[1] (f[1] = s.rx: 0 without a consistent neighbour)
    double* x = cmx_si_x_;
    x[0] = cmx_tsh_ * f[1];
    x[1] = s.agr ? f[1] : 0.0;
    const double* v1 = cmx_si_w_ + (size_t)c1 * 2;
    z1 += v1[0] * x[0] + v1[1] * x[1];
    if (cmx_meas::g_semin == 3) {
      const double* v2 = cmx_si_w_ + (cmx_si_o2_ + (size_t)c2) * 2;
      const double* v3 = cmx_si_w_ + (cmx_si_o3_ + (size_t)c3) * 2;
      z2 += v2[0] * x[0] + v2[1] * x[1];
      z3 += v3[0] * x[0] + v3[1] * x[1];
    }
  }
  s.uz[0] = SemClamp(z1, 30.0);
  s.uz[1] = SemClamp(z2, 30.0);
  s.uz[2] = SemClamp(z3, 30.0);
  s.uc[0] = c1;
  s.uc[1] = c2;
  s.uc[2] = c3;
  const double* fw = s.ufw[k];
  s.ufz = SemClamp(fw[0] * s.uz[0] + fw[1] * s.uz[1] + fw[2] * s.uz[2] + fw[3], 30.0);
  s.cell = k;
  s.ufe = exp(-s.ufz);
  double v = floor(1.0 / (1.0 + s.ufe) * 4294967296.0 + 0.5);
  if (v < 1.0) v = 1.0;
  if (v > 4294967295.0) v = 4294967295.0;
  return (uint32_t)v;
}
// SemUtLearn of the research tail (s.cell >= 0): the same AdaGrad step on the rows CmxSemTail read; the extra weights
// of CMX_SEMIN step on the error of their stage
__attribute__((noinline)) void Predictor::CmxSemLearn(int bit) {
  using semantic_cache::kUIn;
  semantic_cache::State& s = sem_;
  const double lr = (double)semantic_cache::kLrE6 * 1e-6;
  const int c1 = s.uc[0], c2 = s.uc[1], c3 = s.uc[2];
  double* w1 = cmx_meas::g_semtop ? cmx_st_w4_ + (size_t)c1 * kUIn : s.u4w[c1];
  double* g1 = cmx_meas::g_semtop ? cmx_st_g4_ + (size_t)c1 * kUIn : s.u4g[c1];
  double* w3 = cmx_meas::g_semtop ? cmx_st_w6_ + (size_t)c3 * kUIn : s.u6w[c3];
  double* g3 = cmx_meas::g_semtop ? cmx_st_g6_ + (size_t)c3 * kUIn : s.u6g[c3];
  SemUtStep(w1, g1, s.uf, kUIn, s.uz[0], bit, lr);
  SemUtStep(s.u5w[c2], s.u5g[c2], s.uf, kUIn, s.uz[1], bit, lr);
  SemUtStep(w3, g3, s.uf, kUIn, s.uz[2], bit, lr);
  if (cmx_meas::g_semin) {
    SemUtStep(cmx_si_w_ + (size_t)c1 * 2, cmx_si_g_ + (size_t)c1 * 2, cmx_si_x_, 2, s.uz[0], bit, lr);
    if (cmx_meas::g_semin == 3) {
      const size_t r2 = (cmx_si_o2_ + (size_t)c2) * 2, r3 = (cmx_si_o3_ + (size_t)c3) * 2;
      SemUtStep(cmx_si_w_ + r2, cmx_si_g_ + r2, cmx_si_x_, 2, s.uz[1], bit, lr);
      SemUtStep(cmx_si_w_ + r3, cmx_si_g_ + r3, cmx_si_x_, 2, s.uz[2], bit, lr);
    }
  }
  {
    const double ff[4] = {s.uz[0], s.uz[1], s.uz[2], 1.0};
    SemUtStepE(s.ufw[s.cell], s.ufg[s.cell], ff, 4, s.ufe, bit, lr * 0.2);
  }
  s.cell = -1;
}
// research CMX_SEMIN_SELFTEST: the production SemUtCorrect / SemUtLearn / CmxSemInCols on a synthetic vote (one
// neighbour, a consistent readout rx = 1.5 for this bit, share t = 0.6, agreement 1), then the process ends
__attribute__((noinline)) void Predictor::CmxSemInSelfTest() {
  semantic_cache::State& s = sem_;
  const int mode0 = cmx_meas::g_semin;
  int fails = 0, checks = 0;
  auto chk = [&](bool ok, const char* what, double got, double want) {
    ++checks;
    if (!ok) ++fails;
    fprintf(stderr, "CMXTEL semin selftest %s %s got=%.17g want=%.17g\n", ok ? "ok" : "FAIL", what, got, want);
  };
  s.nn = 1;
  s.nfr = 0;
  s.rx = 1.5;
  s.rc = 0.8;
  s.agr = 1;
  s.ws = 1;
  s.gcl = 2;
  s.ecl = 37;
  s.cls = 2;
  s.pb = 'e';
  cmx_tsh_ = 0.6;
  s.uo1[0] = s.ws * 3 + s.gcl;
  s.uo1[1] = (s.cls * 2 + s.agr) * 6 + s.ws * 3 + s.gcl;
  s.uo3[0] = s.ecl;
  s.uo3[1] = (s.cls * 2 + s.agr) * 64 + s.ecl;
  const uint32_t p32 = 2147483648u;
  const int k = 31 - __builtin_clz(manager_.bit_context_ | 1u);
  const int c1 = k * cmx_meas::g_st_s1 + s.uo1[1], c2 = k * 256 + s.pb, c3 = k * cmx_meas::g_st_s3 + s.uo3[1];
  const double x1 = 0.6 * 1.5, x2 = 1.5;
  // 1. zero extra weights: the research tail gives the release tail's stage outputs (same rows, same 20 inputs)
  cmx_meas::g_semx = false;
  cmx_meas::g_semin = 0;
  const uint32_t pr = SemUtCorrect(p32);
  const double r0 = s.uz[0], r1 = s.uz[1], r2 = s.uz[2];
  cmx_meas::g_semx = true;
  cmx_meas::g_semin = 1;
  const uint32_t pt = SemUtCorrect(p32);
  const double d0 = fabs(s.uz[0] - r0) + fabs(s.uz[1] - r1) + fabs(s.uz[2] - r2);
  chk(d0 < 1e-12, d0 == 0.0 ? "zero_weights_eq_release(exact)" : "zero_weights_eq_release(ulp)", s.uz[0], r0);
  chk(pt == pr || (pt > pr ? pt - pr : pr - pt) <= 1u, "zero_weights_p32", (double)pt, (double)pr);
  chk(s.uc[0] == c1 && s.uc[1] == c2 && s.uc[2] == c3 && s.cell == k, "cells", (double)s.uc[0], (double)c1);
  // 2. the inputs: x1 = t * f[1], x2 = agree * f[1]; 0 on disagreement (x2) and without a consistent neighbour
  chk(cmx_si_x_[0] == x1, "x1_share_times_logodds", cmx_si_x_[0], x1);
  chk(cmx_si_x_[1] == x2, "x2_agree_times_logodds", cmx_si_x_[1], x2);
  s.agr = 0;
  SemUtCorrect(p32);
  chk(cmx_si_x_[1] == 0.0 && cmx_si_x_[0] == x1, "x2_zero_on_disagreement", cmx_si_x_[1], 0.0);
  s.agr = 1;
  s.rx = 0.0;
  SemUtCorrect(p32);
  chk(cmx_si_x_[0] == 0.0 && cmx_si_x_[1] == 0.0, "x_zero_without_consistent_neighbour", cmx_si_x_[0], 0.0);
  s.rx = 1.5;
  // 3. mode 1: the stage-1 extra weights move z1 by v . x, stages 2 and 3 untouched (their extra weights ignored)
  double* v1 = cmx_si_w_ + (size_t)c1 * 2;
  double* v2 = cmx_si_w_ + (cmx_si_o2_ + (size_t)c2) * 2;
  double* v3 = cmx_si_w_ + (cmx_si_o3_ + (size_t)c3) * 2;
  v1[0] = 0.5; v1[1] = -0.25; v2[0] = 0.125; v2[1] = 0.0; v3[0] = 0.0; v3[1] = 0.0625;
  SemUtCorrect(p32);
  chk(fabs(s.uz[0] - (r0 + 0.5 * x1 - 0.25 * x2)) < 1e-12, "mode1_stage1_z", s.uz[0], r0 + 0.5 * x1 - 0.25 * x2);
  chk(s.uz[1] == r1 && s.uz[2] == r2, "mode1_stages2_3_unchanged", s.uz[1] + s.uz[2], r1 + r2);
  // 4. mode 3: all three stages
  cmx_meas::g_semin = 3;
  SemUtCorrect(p32);
  chk(fabs(s.uz[0] - (r0 + 0.5 * x1 - 0.25 * x2)) < 1e-12, "mode3_stage1_z", s.uz[0], r0 + 0.5 * x1 - 0.25 * x2);
  chk(fabs(s.uz[1] - (r1 + 0.125 * x1)) < 1e-12, "mode3_stage2_z", s.uz[1], r1 + 0.125 * x1);
  chk(fabs(s.uz[2] - (r2 + 0.0625 * x2)) < 1e-12, "mode3_stage3_z", s.uz[2], r2 + 0.0625 * x2);
  // 5. learning: from zero weights and sums, each extra weight takes SemUtStep on its stage's error (bit 1)
  v1[0] = v1[1] = v2[0] = v2[1] = v3[0] = v3[1] = 0.0;
  SemUtCorrect(p32);
  const double z[3] = {s.uz[0], s.uz[1], s.uz[2]};
  SemUtLearn(1);
  const double lr = (double)semantic_cache::kLrE6 * 1e-6;
  const double* vv[3] = {v1, v2, v3};
  const char* nm[3][2] = {{"learn_stage1_x1", "learn_stage1_x2"}, {"learn_stage2_x1", "learn_stage2_x2"},
                          {"learn_stage3_x1", "learn_stage3_x2"}};
  for (int st = 0; st < 3; ++st)
    for (int j = 0; j < 2; ++j) {
      const double err = 1.0 / (1.0 + exp(-z[st])) - 1.0;
      const double g = err * (j ? x2 : x1);
      const double want = -lr * g / (sqrt(g * g) + 1e-6);
      chk(fabs(vv[st][j] - want) <= 1e-12 * fabs(want) + 1e-18, nm[st][j], vv[st][j], want);
    }
  chk(s.cell == -1, "learn_cell_reset", (double)s.cell, -1.0);
  // 6. CMX_SEMIN=2: a 26-column layer-1 row, columns 24 / 25 = t * c and agree * c of the cache column c, zero
  // without a vote, zero weights in both final mixers
  if (mode0 == 2) {
    chk(layers_[1].Inputs().size() == 26 && cmx_si_col_ == 24, "final_mixer_width_26_cols_24_25",
        (double)layers_[1].Inputs().size(), 26.0);
    sem_index_ = (unsigned int)layers_[0].Inputs().size() - 1;   // SemSetInput's index (the last layer-0 input)
    layers_[0].SetStretchedInput(sem_index_, 2.0f);
    cmx_tsh_ = 0.75;
    s.nn = 1;
    s.agr = 1;
    CmxSemInCols();
    chk(layers_[1].Inputs()[24] == 1.5f, "col24_share_times_c", layers_[1].Inputs()[24], 1.5);
    chk(layers_[1].Inputs()[25] == 2.0f, "col25_agree_times_c", layers_[1].Inputs()[25], 2.0);
    s.agr = 0;
    CmxSemInCols();
    chk(layers_[1].Inputs()[25] == 0.0f, "col25_zero_on_disagreement", layers_[1].Inputs()[25], 0.0);
    s.nn = 0;
    CmxSemInCols();
    chk(layers_[1].Inputs()[24] == 0.0f && layers_[1].Inputs()[25] == 0.0f, "cols_zero_without_vote",
        layers_[1].Inputs()[24], 0.0);
    const mixw_t* wz = mixer_1_[0].E2EWRow();
    const mixw_t* wa = mixer_1_[mix_select::kIa].E2EWRow();
    chk(wz[24] == 0.0f && wz[25] == 0.0f && wa[24] == 0.0f && wa[25] == 0.0f, "final_mixer_weights_start_zero",
        (double)(fabsf(wz[24]) + fabsf(wz[25]) + fabsf(wa[24]) + fabsf(wa[25])), 0.0);
  }
  fprintf(stderr, "CMXTEL semin selftest done mode=%d checks=%d failures=%d\n", mode0, checks, fails);
  fflush(stderr);
  std::_Exit(fails ? 3 : 0);
}
#endif

// behind the floor stage: the end stage (bytes with neighbours), then the top-1 path
// stage; each learns on its own output.  A bit the coder is already sure of
// (P >= kSureE3 / 1000 for one value) keeps its probability, and the end
// stage does not learn on it (cell stays -1); the top-1 path stage runs.
__attribute__((noinline)) uint32_t Predictor::SemCorrect(uint32_t p32) {
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_sk2_end, 0)) return CmxSem2Correct(p32);   // research CMX_SEMKEY2: both end stages
#endif
  semantic_cache::State& s = sem_;
  s.cell = -1;
  constexpr uint32_t kSure = (uint32_t)((1000 - semantic_cache::kSureE3) * 4294967.296);
  const bool sure = p32 <= kSure || p32 >= 4294967295u - kSure;
  const uint32_t pu = (s.nn == 0 || sure) ? p32 : SemUtCorrect(p32);
  return TreffCorrect(pu);
}

__attribute__((noinline)) void Predictor::SemLearn(int bit) {
  SemUtLearn(bit);
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_sk2_end, 0))   // research CMX_SEMKEYS / CMX_SEMKEY2: their end stages
    for (int k = 0; k < cmx_meas::g_nsk; ++k) CmxSem2UtLearn(k, bit);
#endif
  TreffLearn(bit);
}

// The cache's layer-0 input: the cache log-odds of the bit about to be
// predicted (the main readout, same neighbours, same prefix test, same
// formula), computed in Predict() before the layer-0 row; 0 on a byte without
// neighbours, on a byte injected behind the stream byte (observed without
// coding) and without a consistent neighbour
__attribute__((noinline)) void Predictor::SemPredBit() {
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_nsk != 0, 0))   // research CMX_SEMKEYS / CMX_SEMKEY2: their readouts of the bit
    for (int k = 0; k < cmx_meas::g_nsk; ++k) CmxSem2PredBit(k);
#endif
  semantic_cache::State& s = sem_;
  s.ux = 0.0;
  if (manager_.bit_context_ == 1u) {   // first bit of a byte
    s.uact = s.ufresh;
    s.ufresh = 0;
  }
  s.rok = 0;
  s.rx = s.rc = 0.0;
  if (s.nn > 0) {
    const unsigned int ctx = manager_.bit_context_;
    const int k = 31 - __builtin_clz(ctx | 1u);
    // part sums of the groups consistent with this prefix (SemUtByte), by
    // the value of this bit (groups in byte order: value 0 in front of 1);
    // the end stage reads all seven, the main readout the first
    double g[2][8] = {};
    for (int i = 0; i < s.ngr; ++i) {
      const unsigned int c = 0x100u | s.gb[i];
      if ((c >> (8 - k)) != ctx) continue;
      double* t = g[(c >> (7 - k)) & 1u];
      for (int r = 0; r < 8; ++r) t[r] += s.gm[i][r];
    }
    for (int r = 0; r < 8; ++r) {
      s.rm[r][0] = g[0][r];
      s.rm[r][1] = g[1][r];
    }
    const double m0 = g[0][0], m1 = g[1][0];
    if ((m0 + m1) > 0.0) {
      s.rok = 1;
      s.rx = log((m1 / s.wsum + 0.01) / (m0 / s.wsum + 0.01));
      s.rc = (m0 + m1) / s.wsum;
    }
    if (s.uact) {
      s.ux = s.rx;
      // the rows of the end stage this bit reads and learns (SemUtCorrect,
      // SemUtLearn; cells as there): loaded here, long before they are used
      const int info = s.rc > 0.0;
#if PORT_CMX_MEAS
      if (__builtin_expect(cmx_meas::g_semtop != 0, 0)) {   // research CMX_SEMTOP: stages 1 and 3 in the wide tables
        const size_t w1 = (size_t)(k * cmx_meas::g_st_s1 + s.uo1[info]) * semantic_cache::kUIn;
        const size_t w3 = (size_t)(k * cmx_meas::g_st_s3 + s.uo3[info]) * semantic_cache::kUIn;
        SemPfRow(cmx_st_w4_ + w1);
        SemPfRow(cmx_st_g4_ + w1);
        SemPfRow(s.u5w[k * 256 + s.pb]);
        SemPfRow(s.u5g[k * 256 + s.pb]);
        SemPfRow(cmx_st_w6_ + w3);
        SemPfRow(cmx_st_g6_ + w3);
        return;
      }
#endif
      const int c1 = k * 48 + s.uo1[info], c2 = k * 256 + s.pb, c3 = k * 512 + s.uo3[info];
      SemPfRow(s.u4w[c1]);
      SemPfRow(s.u4g[c1]);
      SemPfRow(s.u5w[c2]);
      SemPfRow(s.u5g[c2]);
      SemPfRow(s.u6w[c3]);
      SemPfRow(s.u6g[c3]);
    }
  }
}
// The cache's input: one more layer-0 input (behind the fast-weight row, outside the
// adaptive-depth gather) and the last end-mixer column (outside the BITLSTM
// row); GetNumModels() and AddMixers() count it
__attribute__((noinline)) void Predictor::SemSetInput(unsigned int i) {
  sem_index_ = i;
  layers_[0].SetStretchedInput(i, (float)sem_.ux);
#if PORT_CMX_MEAS
  // research CMX_SEMKEYS / CMX_SEMKEY2 (mixers): the extra caches' log-odds, the layer-0 inputs behind it (row 563 + n)
  if (__builtin_expect(cmx_meas::g_sk2_mix, 0))
    for (int k = 0; k < cmx_meas::g_nsk; ++k) layers_[0].SetStretchedInput(i + 1 + (unsigned int)k, (float)cmx_sk_[k]->ux);
#endif
}
__attribute__((noinline)) void Predictor::SemSetStage1() {
  layers_[1].SetStretchedInput(mixer_0_.size() + auxiliary_size_ + 1 + (4 ? 1 : 0)
                               , layers_[0].Inputs()[sem_index_]);
#if PORT_CMX_MEAS
  // research CMX_SEMKEYS / CMX_SEMKEY2 (mixers): their final-mixer columns, copies of their layer-0 inputs as for the
  // release cache
  if (__builtin_expect(cmx_meas::g_sk2_mix, 0))
    for (int k = 0; k < cmx_meas::g_nsk; ++k)
      layers_[1].SetStretchedInput(cmx_sk_col_[k], layers_[0].Inputs()[sem_index_ + 1 + (unsigned int)k]);
#endif
}

#if PORT_CMX_MEAS
// ---- research CMX_SEMKEYS / CMX_SEMKEY2 (src/cmx_meas.h; Claude Opus 5.5, 2026-11-08) ------------------------------
// Up to three extra semantic caches sk = 0..n-1 (cache sk + 1) on *cmx_sk_[sk], keyed on the mixer read of transformer
// layer g_sk_layer[sk] (cmx_meas::g_sk_key[layer], written by model_opt.cpp for every read row), or (list element R,
// g_sk_layer[sk] == kSkRel) on the release key itself (tf1_engine_->LastKey(), the last layer's MLP read q 31) with
// that cache's own seed.  CmxSem2Init /
// CmxSem2Byte / CmxSem2UtByte / CmxSem2UtCorrect / CmxSem2UtLearn / CmxSem2PredBit are the PORT_CMX_MEAS 0 text of
// SemInit / SemByteStart / SemUtByte / SemUtCorrect / SemUtLearn / SemPredBit (generated by a script from the release
// text) with these edits only: the instance parameter sk, their own State (*cmx_sk_[sk]), the key of its layer, the
// generator seeded 0x5EC0CAC4E2 ^ (sk + 1) * 0x9E3779B97F4A7C15 (sk = 0: the CMX_SEMKEY2 seed), no TreffByteStart (the
// release byte start ran it for this byte; treff_ is read as there), the calls renamed.  None of the release cache's
// research hooks.
__attribute__((noinline)) COLD_SIZE_ATTR void Predictor::CmxSem2Init(int sk) {
  using namespace semantic_cache;
  semantic_cache::State& s = *cmx_sk_[sk];
  s.key = (uint8_t*)SemTable((size_t)kN * kKB, false);
  s.val = (uint8_t*)SemTable((size_t)kN, false);
  s.slot = (Slot*)SemTable(((size_t)kL << kB) * kM * sizeof(Slot), true);
  s.epos = (uint32_t*)SemTable((size_t)kN * 4, false);
  s.hist = (uint8_t*)SemTable((size_t)kHist, false);
  if (!s.key || !s.val || !s.slot || !s.epos || !s.hist) {
    msg_note(553);
    exit(1);
  }
  // Resident now, at the first stream byte, behind room made in the PPMd
  // pool.  Lookups and entries start at the text; the bucket table is read
  // before it is written, and on a kernel that answers a write to the huge
  // zero page with a huge page its first few hundred entries made all 960 MiB
  // resident within 100 ms, after the pool had grown.  All of
  // it is resident at the end anyway.  Zero-filled, so no value changes.
  {
    const size_t n[5] = {(size_t)kN * kKB, (size_t)kN, ((size_t)kL << kB) * kM * sizeof(Slot),
                         (size_t)kN * 4, (size_t)kHist};
    void* const t[5] = {s.key, s.val, s.slot, s.epos, s.hist};
    unsigned long long sum = 0;
    for (int i = 0; i < 5; ++i) sum += n[i];
    if (ppmd_page::raum(sum))
      for (int i = 0; i < 5; ++i) memset(t[i], 0, n[i]);
  }
  // one generator, in this order: projection, table hyperplanes, sketch hyperplanes
  uint64_t r = 0x5EC0CAC4E2ULL ^ (uint64_t)(sk + 1) * 0x9E3779B97F4A7C15ULL;   // its own seed (cache sk + 1)
  for (int a = 0; a < kD; ++a)
    for (int c = 0; c < 192; ++c) s.proj[a][c] = (SemRng(r) >> 31) ? 1 : -1;
  for (int h = 0; h < kL * kB; ++h)
    for (int a = 0; a < kD; ++a) s.hyp[h][a] = (SemRng(r) >> 31) ? 1 : -1;
  for (int h = 0; h < kSkGen; ++h)
    for (int a = 0; a < kD; ++a) s.skp[h][a] = (SemRng(r) >> 31) ? 1 : -1;
  // every stage of the end stage starts at "coder unchanged" (weight 1 on
  // the coder log-odds), the final stage at the mean of the three; the rows
  // are zero from the State constructor and written only after this point
  for (int c = 0; c < kUC4; ++c) s.u4w[c][0] = 1.0;
  for (int c = 0; c < kUC5; ++c) s.u5w[c][0] = 1.0;
  for (int c = 0; c < kUC6; ++c) s.u6w[c][0] = 1.0;
  // f[10] of the end stage (one neighbour of weight 1): log((m1 + 0.01) /
  // (m0 + 0.01)); the asm keeps the arguments opaque, so log runs at run time
  {
    double a0 = (0.0 + 0.01) / (1.0 + 0.01), a1 = (1.0 + 0.01) / (0.0 + 0.01);
    asm volatile("" : "+x"(a0), "+x"(a1));
    s.l10[0] = log(a0);
    s.l10[1] = log(a1);
  }
  for (int c = 0; c < 8; ++c) {
    s.ufw[c][0] = s.ufw[c][1] = s.ufw[c][2] = 1.0 / 3.0;
    s.ufw[c][3] = 0.0;
    for (int i = 0; i < 4; ++i) s.ufg[c][i] = 0.0;
  }
}

__attribute__((noinline)) void Predictor::CmxSem2Byte(int sk, int strom_pb) {
  using namespace semantic_cache;
  semantic_cache::State& s = *cmx_sk_[sk];
  strom_pb &= 0xFF;   // the previous stream byte; recent_bytes_[0] is the last injected one behind an injection
  if (!s.key) CmxSem2Init(sk);
  s.ufresh = 1;   // layer-0 input: the next byte predicted is this stream byte
  // 1. the previous byte is complete: its entry goes into the memory and
  // replaces the oldest entry (or the first empty slot) of each own bucket
  if (s.pend) {
    const uint32_t ri = (uint32_t)(s.nins & (kN - 1));
    SemPut4(s.key + (size_t)ri * kKB, s.ck);
    s.val[ri] = (uint8_t)strom_pb;
    for (int t = 0; t < kL; ++t) {
      Slot* sl = s.slot + (((size_t)t << kB) + s.chb[t]) * kM;
      const int um = SemOldest(sl);
      sl[um].e1 = (uint32_t)s.nins + 1u;
      sl[um].s2 = s.c2;
      sl[um].s01 = s.c01;
    }
    s.epos[ri] = (uint32_t)(s.spos - 1);
    ++s.nins;
    s.pend = false;
  }
  // 2. hit of the cache top-1 of the previous byte (if it had neighbours)
  // for the running hit rate; word start, previous byte and letter run of
  // this byte
  if (s.htopb >= 0) {
    const double hit = (s.htopb == strom_pb) ? 1.0 : 0.0;
    s.hrate += (hit - s.hrate) / 64.0;
    s.htopb = -1;
  }
  const uint64_t spos = s.spos++;   // SemByteStart runs once per stream byte in Compress and Decompress
  const uint32_t ut_pos = (uint32_t)spos;
  {
    const int pb = strom_pb;
    s.ws =(pb == 0x20 || pb == 0x0A || pb == '@') ? 1 : 0;
    s.pb = pb;
    // letters and bytes >= 0x80 extend the run
    const bool al = (pb >= 'a' && pb <= 'z') || (pb >= 'A' && pb <= 'Z') || pb >= 0x80;
    s.wlen = al ? (s.wlen < 15 ? s.wlen + 1 : 15) : 0;
    const int pc = (pb >= 'a' && pb <= 'z') ? 0 : (pb >= 'A' && pb <= 'Z') ? 1 : (pb >= '0' && pb <= '9') ? 2 :
                   pb == ' ' ? 3 : pb == 10 ? 4 : pb >= 0x80 ? 5 : (pb == '[' || pb == ']' || pb == '|') ? 6 : 7;
    const int w = s.wlen;
    const int wl = w == 0 ? 0 : w < 3 ? 1 : w < 5 ? 2 : w < 7 ? 3 : w < 9 ? 4 : w < 11 ? 5 : w < 14 ? 6 : 7;
    s.ecl = pc * 8 + wl;
  }
  // 3. the previous byte goes into the history; every followed neighbour
  // whose continuation differs from it is dead
  if (spos > 0) {
    const uint8_t pb = (uint8_t)strom_pb;
    s.hist[(spos - 1) & (kHist - 1)] = pb;
    for (int g = 0; g <= kG; ++g) {
      if (s.fn[g] <= 0) continue;
      const uint64_t t = spos - s.fq[g];
      if (t > (uint64_t)kWtMax) { s.fn[g] = 0; continue; }
      for (int j = 0; j < s.fn[g]; ++j)
        if (s.fal[g][j] && s.hist[(s.fvp[g][j] + t - 1) & (kHist - 1)] != pb) s.fal[g][j] = 0;
    }
  }
  s.nh[spos & 4095] = (uint32_t)s.nins;   // entries with a stream position <= spos - 1
  s.nn = 0;
  s.cell = -1;
  if (!(manager_.tf1_gate_ctx1 == 0 && tf1_slot_->KtxLive())) {
    for (int g = 0; g <= kG; ++g) s.fn[g] = 0;
    return;
  }
  // 4. top-1 of the transformer row (that of the top-1 path stage) and its
  // probability
  const float* t = &tf1_slot_->BytePredict()[0];
  const int t1 = treff_.a1;   // top-1 of this live row (TreffByteStart of the release byte start)
  double sum = 0.0, rest = 0.0;
  for (int i = 0; i < 256; ++i) { sum += (double)t[i]; if (i != t1) rest += (double)t[i]; }
  if (!(sum > 0.0)) return;
  const float cmiss = rest > 0.0 ? (float)(-semantic_cache::Log2(rest / sum)) : 64.0f;
  const double p1 = 1.0 - exp2(-(double)cmiss);
  const bool q_ok = p1 < (double)kQpE3 / 1000.0;   // looked up
  const bool s_ok = p1 < (double)kSpE3 / 1000.0;   // stored
  if (!q_ok && !s_ok) return;
  // 5. key of this byte: the mixer read of its layer or the release key (CMX_SEMKEYS) minus its running
  // mean (frozen after kCen counted lookups), int8, +-1 projection, int8
  // again; table hashes (moved by the previous byte, with the order of the
  // hash bits by certainty) and sketch
  // the mixer read of its layer, or (element R) the release key itself
  const float* x = cmx_meas::g_sk_layer[sk] == cmx_meas::kSkRel ? tf1_engine_->LastKey()
                                                               : cmx_meas::g_sk_key[cmx_meas::g_sk_layer[sk]];
  float xc[192];
  {
    const bool kc_ok = p1 < (double)kCalqE3 / 1000.0;
    if (kc_ok && s.kcnt < kCen) {
      for (int c = 0; c < 192; ++c) s.kmu[c] += (double)x[c];
      ++s.kcnt;
    }
    const double inv = s.kcnt ? 1.0 / (double)s.kcnt : 0.0;
    if (s.kcnt < kCen) {
      for (int c = 0; c < 192; ++c) xc[c] = x[c] - (float)(s.kmu[c] * inv);
    } else {   // frozen: the float mean once
      if (!s.kmf) {
        for (int c = 0; c < 192; ++c) s.kmuf[c] = (float)(s.kmu[c] * inv);
        s.kmf = 1;
      }
      for (int c = 0; c < 192; ++c) xc[c] = x[c] - s.kmuf[c];
    }
    x = xc;
  }
  alignas(32) int8_t q[192];
  {
    float m = 0.0f;
    for (int c = 0; c < 192; ++c) { const float a = fabsf(x[c]); if (a > m) m = a; }
    // x / (m / 127) as the compiler evaluates it: x * (127 / m); clamped to
    // +-127, rounded to nearest (as lrintf), eight lanes at a time
    const __m256 rv = _mm256_set1_ps(m > 0.0f ? 127.0f / m : 1.0f);
    const __m256 hi = _mm256_set1_ps(127.0f), lo = _mm256_set1_ps(-127.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int c = 0; c < 192; c += 32) {
      __m256i v[4];
      for (int i = 0; i < 4; ++i)
        v[i] = _mm256_cvtps_epi32(_mm256_max_ps(_mm256_min_ps(_mm256_mul_ps(_mm256_loadu_ps(x + c + 8 * i), rv), hi), lo));
      const __m256i b = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
      _mm256_store_si256((__m256i*)(q + c), _mm256_permutevar8x32_epi32(b, perm));
    }
  }
  int32_t y[kD];
  SemDotPm1<192>(&s.proj[0][0], kD, q, y);
  int32_t ym = 0;
  for (int a = 0; a < kD; ++a) {
    const int32_t av = y[a] < 0 ? -y[a] : y[a];
    if (av > ym) ym = av;
  }
  uint32_t n2 = 0;
  {
    // (y * 127) / ym truncated toward zero, as a multiplication with
    // ceil(2^37 / ym): exact for |y| * 127 < 2^22 and ym < 2^15
    static_assert(192 * 127 * 127 < (1 << 22) && 192 * 127 < (1 << 15), "key range");
    const uint64_t mr = ym > 0 ? ((((uint64_t)1) << 37) + (uint64_t)ym - 1) / (uint64_t)ym : 0;
    for (int a = 0; a < kD; ++a) {
      const uint32_t u = (uint32_t)(y[a] < 0 ? -y[a] : y[a]) * 127u;
      const int32_t m = (int32_t)(((uint64_t)u * mr) >> 37);
      const int32_t v = y[a] < 0 ? -m : m;
      s.ck[a] = (int8_t)v;
      n2 += (uint32_t)(v * v);
    }
  }
  size_t bix[kL * kProbe];
  {
    const uint32_t part = (uint32_t)((((uint64_t)strom_pb + 1u) * 0x9E3779B97F4A7C15ull) >> (64 - kB));
    int32_t dh[kL * kB];
    SemDotPm1<kD>(&s.hyp[0][0], kL * kB, s.ck, dh);
    for (int tb = 0; tb < kL; ++tb) {
      uint32_t h = 0;
      int32_t ad[kB];   // |dot| per bit position (position kB-1-j for hyperplane j)
      for (int j = 0; j < kB; ++j) {
        const int32_t d = dh[tb * kB + j];
        h = (h << 1) | (d > 0 ? 1u : 0u);
        ad[kB - 1 - j] = d < 0 ? -d : d;
      }
      h ^= part;
      s.chb[tb] = h;
      // the kFlip least certain positions, ties to the lower position
      int ord[kB];
      for (int pp = 0; pp < kB; ++pp) {
        int j = pp;
        while (j > 0 && ad[ord[j - 1]] > ad[pp]) { ord[j] = ord[j - 1]; --j; }
        ord[j] = pp;
      }
      bix[tb * kProbe] = ((size_t)tb << kB) + h;
      for (int fl = 1; fl < kProbe; ++fl) bix[tb * kProbe + fl] = ((size_t)tb << kB) + (h ^ (1u << ord[fl - 1]));
    }
  }
  // the first buckets of the scan load while the sketch is computed
  if (q_ok)
    for (int tp = 0; tp < kPf; ++tp) SemPfBucket(s.slot + bix[tp] * kM);
  uint64_t q01 = 0;
  uint32_t q2 = 0;
  {
    int32_t dk[kSk];
    SemDotPm1<kD>(&s.skp[0][0], kSk, s.ck, dk);
    for (int j = 0; j < kSk; ++j) {
      if (dk[j] > 0) {
        if (j < 64) q01 |= 1ull << j; else q2 |= 1u << (j - 64);
      }
    }
  }
  s.c01 = q01;
  s.c2 = q2;
  s.pend = s_ok;
  if (!q_ok) return;   // stored, not looked up
  // 6. query: the buckets in table order (own, then the flipped ones),
  // sketch distances of the valid entries at least kGap stream bytes back;
  // counting sort by distance keeps the visiting order on ties, an entry met
  // again is skipped; the kPre first are re-ranked by the int4 cosine
  constexpr int kNB = kL * kProbe;
  const uint64_t oldest = s.nins > kN ? s.nins - kN : 0;
  const uint64_t enear = spos + 1 >= (uint64_t)kGap ? s.nh[(spos + 1 - kGap) & 4095] : 0;
  int nc = 0;
  int cnt[kSk + 2];
  for (int h = 0; h < kSk + 2; ++h) cnt[h] = 0;
  for (int tp = 0; tp < kNB; ++tp) {
    if (tp + kPf < kNB) SemPfBucket(s.slot + bix[tp + kPf] * kM);
    const Slot* sl = s.slot + bix[tp] * kM;
    for (int u = 0; u < kM; ++u) {
      const uint32_t e1 = sl[u].e1;
      if (e1 == 0u || (uint64_t)(e1 - 1u) < oldest) continue;
      if ((uint64_t)(e1 - 1u) >= enear) continue;
      const int hm = __builtin_popcountll(q01 ^ sl[u].s01) + __builtin_popcount(q2 ^ sl[u].s2);
      s.cand_hm[nc] = (uint8_t)hm;
      s.cand_e[nc] = e1 - 1u;
      ++cnt[hm + 1];
      ++nc;
    }
  }
  for (int h = 1; h < kSk + 2; ++h) cnt[h] += cnt[h - 1];   // cnt[h] = start of distance h
  for (int i = 0; i < nc; ++i) s.ord_e[cnt[s.cand_hm[i]]++] = s.cand_e[i];
  // after the placement cnt[h] is the end of distance h
  // an entry met again comes from another table with the same sketch, so it
  // has the same distance: the selected entries are a set (open addressing,
  // slots of other lookups carry an older stamp)
  int npre = 0;
  uint32_t pid[kPre];
  {
    if (++s.dstamp == 0) {   // wrap: forget all stamps
      for (int x = 0; x < kDedup; ++x) s.dst[x] = 0;
      s.dstamp = 1;
    }
    const uint32_t st = s.dstamp;
    const int nall = cnt[kSk];   // the end of the last distance = all candidates
    for (int i = 0; i < nall && npre < kPre; ++i) {
      const uint32_t e = s.ord_e[i];
      for (uint32_t x = (e * 0x9E3779B1u) >> (32 - kDedupLog);; x = (x + 1) & (kDedup - 1)) {
        if (s.dst[x] != st) {
          s.dst[x] = st;
          s.dke[x] = e;
          pid[npre++] = e;
          __builtin_prefetch(s.key + (size_t)(e & (kN - 1)) * kKB);   // its ring key, for the re-rank
          break;
        }
        if (s.dke[x] == e) break;
      }
    }
  }
  int nk = 0;
  uint32_t kid[kK];
  double ks[kK];
  const double qn = n2 > 0 ? 1.0 / sqrt((double)n2) : 0.0;
  // int4 cosine, exact in integers: the even key components meet the low
  // nibbles, the odd ones the high nibbles; vpmaddubsw takes |component| and
  // the nibble with the component's sign (|pair| <= 2 x 127 x 8)
  static_assert(kKB == 64, "two 32-byte halves per ring key");
  __m256i cev[2], cov[2], aev[2], aov[2];
  {
    alignas(32) int8_t ce[kKB], co[kKB];
    for (int c = 0; c < kKB; ++c) { ce[c] = s.ck[2 * c]; co[c] = s.ck[2 * c + 1]; }
    for (int v = 0; v < 2; ++v) {
      cev[v] = _mm256_load_si256((const __m256i*)(ce + 32 * v));
      cov[v] = _mm256_load_si256((const __m256i*)(co + 32 * v));
      aev[v] = _mm256_abs_epi8(cev[v]);
      aov[v] = _mm256_abs_epi8(cov[v]);
    }
  }
  const __m256i m0f = _mm256_set1_epi8(0x0F), m08 = _mm256_set1_epi8(8), one16 = _mm256_set1_epi16(1);
  for (int a = 0; a < npre; ++a) {
    const uint32_t ri = pid[a] & (kN - 1);
    const uint8_t* kk = s.key + (size_t)ri * kKB;
    __m256i da = _mm256_setzero_si256(), na = _mm256_setzero_si256();
    for (int v = 0; v < 2; ++v) {
      const __m256i x = _mm256_loadu_si256((const __m256i*)(kk + 32 * v));
      const __m256i lo = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(x, m0f), m08), m08);
      const __m256i hi = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(x, 4), m0f), m08), m08);
      da = _mm256_add_epi16(da, _mm256_maddubs_epi16(aev[v], _mm256_sign_epi8(lo, cev[v])));
      da = _mm256_add_epi16(da, _mm256_maddubs_epi16(aov[v], _mm256_sign_epi8(hi, cov[v])));
      const __m256i al = _mm256_abs_epi8(lo), ah = _mm256_abs_epi8(hi);
      na = _mm256_add_epi16(na, _mm256_maddubs_epi16(al, al));
      na = _mm256_add_epi16(na, _mm256_maddubs_epi16(ah, ah));
    }
    const __m256i t = _mm256_hadd_epi32(_mm256_madd_epi16(da, one16), _mm256_madd_epi16(na, one16));
    const __m128i u = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
    const __m128i w = _mm_hadd_epi32(u, u);
    const int32_t d = _mm_cvtsi128_si32(w);
    const uint32_t kn2 = (uint32_t)_mm_extract_epi32(w, 1);
    const double sim = kn2 > 0 ? d * qn / sqrt((double)kn2) : 0.0;
    if (nk < kK || sim > ks[nk - 1]) {
      int j = nk < kK ? nk++ : kK - 1;
      while (j > 0 && ks[j - 1] < sim) { ks[j] = ks[j - 1]; kid[j] = kid[j - 1]; --j; }
      ks[j] = sim;
      kid[j] = ri;
    }
  }
  // 7. followers: a lookup below kGq starts a generation (generation 0 = the
  // fresh neighbours, the last one moves to the newest earlier generation);
  // a lookup above reads the followers as if it started one, but the
  // generations stay as they are
  const bool ngen = !(p1 < (double)kGqE3 / 1000.0);
  if (!ngen) {
    if (s.fn[0] > 0) {
      s.ghead = s.ghead % kG + 1;   // 1..kG
      const int g = s.ghead;
      s.fn[g] = s.fn[0];
      s.fq[g] = s.fq[0];
      for (int j = 0; j < s.fn[0]; ++j) { s.fvp[g][j] = s.fvp[0][j]; s.fsim[g][j] = s.fsim[0][j]; s.fal[g][j] = s.fal[0][j]; }
    }
    s.fn[0] = nk;
    s.fq[0] = spos;
    // a continuation outside the history ring cannot be followed (the value
    // at this lookup stays)
    for (int j = 0; j < nk; ++j) {
      s.fvp[0][j] = (uint64_t)s.epos[kid[j]];
      s.fsim[0][j] = ks[j];
      s.fal[0][j] = (spos - s.fvp[0][j]) + (uint64_t)kWtMax < kHist ? 1 : 0;
    }
  }
  // followed generations of this lookup, newest first (kG of them): with a
  // starting lookup 1..kG; otherwise generation 0 first (if any), as pushed
  int gl[kG], ngl = 0;
  if (ngen && s.fn[0] > 0) gl[ngl++] = 0;
  for (int gi = 0; ngl < kG; ++gi) gl[ngl++] = (s.ghead - 1 - gi + kG) % kG + 1;
  bool anyf = false;
  for (int gi = 0; gi < kG; ++gi) if (s.fn[gl[gi]] > 0) anyf = true;
  if (nk == 0 && !anyf) return;
  // 8. the main readout: fresh neighbours weigh exp((s - s_max) / T), the
  // followed ones the byte behind their continuation with exp((s - s_max of
  // their lookup) / T); the end stage reads every neighbour with its
  // similarity relative to the s_max of its own lookup and its age
  const double tinv = 1000.0 / (double)kTE3;
  double acc[256];
  int used[kNb], nu = 0;
  uint8_t has[256] = {};   // bytes in used[]
  s.wsum = 0.0;
  int nb = 0;
  for (int j = 0; j < nk; ++j) {
    const int c = s.val[kid[j]];
    const double w = exp((ks[j] - ks[0]) * tinv);
    s.nbyte[nb] = (uint8_t)c;
    s.nw[nb] = w;
    s.asim[nb] = ks[j] - ks[0];
    s.aage[nb] = ut_pos - s.epos[kid[j]];
    ++nb;
    s.wsum += w;
    if (has[c]) acc[c] += w;
    else { has[c] = 1; used[nu++] = c; acc[c] = w; }
  }
  double smax0 = nk > 0 ? ks[0] : 0.0;
  bool smset = nk > 0;
  s.nfr = nb;
  for (int gi = 0; gi < kG; ++gi) {
    const int g = gl[gi];
    if (s.fn[g] <= 0) continue;
    if (!smset) { smax0 = s.fsim[g][0]; smset = true; }   // no fresh neighbour: the newest followed lookup
    const uint64_t t = spos - s.fq[g];
    for (int j = 0; j < s.fn[g]; ++j) {
      if (!s.fal[g][j] || s.fvp[g][j] + t >= spos) continue;
      const int c = s.hist[(s.fvp[g][j] + t) & (kHist - 1)];
      const double w = exp((s.fsim[g][j] - s.fsim[g][0]) * tinv);
      s.nbyte[nb] = (uint8_t)c;
      s.nw[nb] = w;
      s.asim[nb] = s.fsim[g][j] - s.fsim[g][0];
      s.aage[nb] = (uint32_t)(s.fq[g] - s.fvp[g][j]);
      ++nb;
      s.wsum += w;
      if (has[c]) acc[c] += w;
      else { has[c] = 1; used[nu++] = c; acc[c] = w; }
    }
  }
  if (nb == 0) return;
  s.nn = nb;
  int topb = used[0];
  for (int u = 1; u < nu; ++u) {
    const int c = used[u];
    if (acc[c] > acc[topb] || (acc[c] == acc[topb] && c < topb)) topb = c;
  }
  // s_max of the lookup (smax0) gives the similarity class
  s.cls = smax0 < (double)kC1E3 / 1000.0 ? 1 : (smax0 < (double)kC2E3 / 1000.0 ? 2 : 3);
  s.agr = (topb == t1) ? 1 : 0;
  // BITLSTM column: the similarity of the lookup (s_max)
  {
    const double sm = (smax0 - 0.9) * 10.0;
    s.b_smax = (float)(sm > 1.0 ? 1.0 : (sm < -1.0 ? -1.0 : sm));
  }
  CmxSem2UtByte(sk, topb);
}

__attribute__((noinline)) void Predictor::CmxSem2UtByte(int sk, int topb) {
  semantic_cache::State& s = *cmx_sk_[sk];
  s.topb = topb;
  s.htopb = topb;
  s.wfg = 0;
  const uint32_t a0 = s.aage[0];
  s.gcl = a0 < 2048u ? 0 : (a0 < 1048576u ? 1 : 2);
  // match byte and the fast-weight and PPMd top-1 as the top-1 path stage
  // found them at this byte start (a byte with neighbours has a live row
  // with a positive top-1)
  const cache_hit::State& tr = treff_;
  s.mtop = (tr.mlen > 0 && tr.mbyte == topb) ? 1 : 0;
  s.ftop = (tr.fa == topb) ? 1 : 0;
  s.ptop = (tr.pa == topb) ? 1 : 0;
  const double tinv = 1000.0 / (double)semantic_cache::kTE3;
  s.hws = 0.0;
  for (int j = 0; j < s.nn; ++j) { s.hw[j] = exp(s.asim[j] * 4.0 * tinv); s.hws += s.hw[j]; }
  // the readouts at temperature T weigh exp(asim / T), which is nw (same
  // product, same exp); counts and weight sums of the near and far neighbours
  s.n_n = s.n_f = 0;
  s.ws_n = s.ws_f = 0.0;
  for (int j = 0; j < s.nn; ++j) {
    if (s.aage[j] < semantic_cache::kUAge) { ++s.n_n; s.ws_n += s.nw[j]; }
    else { ++s.n_f; s.ws_f += s.nw[j]; }
  }
  // cells of the three stages: the byte part (without / with a consistent
  // neighbour; similarity class and agreement count only with one)
  s.uo1[0] = s.ws * 3 + s.gcl;
  s.uo1[1] = (s.cls * 2 + s.agr) * 6 + s.ws * 3 + s.gcl;
  s.uo3[0] = s.ecl;
  s.uo3[1] = (s.cls * 2 + s.agr) * 64 + s.ecl;
  // part sums: the neighbours grouped by byte value (groups in byte order),
  // per group the masses of the seven readouts; the readouts of a bit add the
  // groups consistent with its prefix (SemPredBit) instead of looping over all
  // neighbours per readout
  {
    uint64_t bm[4] = {0, 0, 0, 0};
    for (int j = 0; j < s.nn; ++j) bm[s.nbyte[j] >> 6] |= 1ull << (s.nbyte[j] & 63);
    int g = 0;
    for (int w = 0; w < 4; ++w)
      for (uint64_t bb = bm[w]; bb; bb &= bb - 1) {
        const int c = w * 64 + __builtin_ctzll(bb);
        s.gb[g] = (uint8_t)c;
        s.gix[c] = (uint8_t)g;
        ++g;
      }
    s.ngr = g;
    memset(s.gm, 0, sizeof(s.gm[0]) * (size_t)g);
    for (int j = 0; j < s.nn; ++j) {
      double* m = s.gm[s.gix[s.nbyte[j]]];
      m[0] += s.nw[j];
      m[1] += s.hw[j];
      m[j < s.nfr ? 2 : 3] += s.nw[j];
      m[s.aage[j] < semantic_cache::kUAge ? 4 : 5] += s.nw[j];
      m[6] += 1.0;
    }
  }
}

__attribute__((noinline)) uint32_t Predictor::CmxSem2UtCorrect(int sk, uint32_t p32) {
  using namespace semantic_cache;
  semantic_cache::State& s = *cmx_sk_[sk];
  const unsigned int ctx = manager_.bit_context_;
  const int k = 31 - __builtin_clz(ctx | 1u);
  double* f = s.uf;
  const double conf = s.rc;   // the main readout of this bit (SemPredBit)
  const double xk = s.rx;
  const bool info = conf > 0.0;
  f[0] = SemClamp(log((double)p32 / (4294967296.0 - (double)p32)), 30.0);
  f[1] = xk;
  f[2] = 1.0;
  f[3] = xk * conf;
  f[4] = SemClamp(SemUtLogitP((double)tf1_slot_output), 30.0);
  f[5] = SemClamp(SemUtLogitP((double)fw_slot_output), 30.0);
  f[6] = SemClamp((double)layers_[0].Inputs()[mxs_ppmd_ix_], 30.0);
  f[7] = SemClamp((double)adepth_.f[2], 30.0);
  {   // readouts from the part sums of this bit (SemPredBit)
    f[8] = SemUtReadM(s.rm[1], s.hws);
    f[9] = SemUtReadC((int)s.rm[6][0], (int)s.rm[6][1], s.nn);
    {   // the nearest neighbour alone: 0 or one of two log-odds (SemInit)
      const unsigned int c = 0x100u | s.nbyte[0];
      f[10] = ((c >> (8 - k)) != ctx) ? 0.0 : s.l10[(c >> (7 - k)) & 1u];
    }
    f[11] = xk;   // temperature T: the weights and their sum are nw and wsum
    f[12] = s.n_n ? SemUtReadM(s.rm[4], s.ws_n) : 0.0;
    f[13] = s.n_f ? SemUtReadM(s.rm[5], s.ws_f) : 0.0;
  }
  {
    const unsigned int tb = (unsigned int)s.topb;
    const double vote = ((((0x100u | tb) >> (8 - k))) == ctx) ? (((tb >> (7 - k)) & 1u) ? 1.0 : -1.0) : 0.0;
    f[14] = vote * (double)s.mtop;
    f[15] = vote * (double)s.ftop;
    f[16] = vote * (double)s.ptop;
  }
  f[17] = info ? xk * (s.hrate - 0.4) : 0.0;
  {
    const int nf = s.nfr, ng = s.nn - s.nfr;
    if (!s.wfg) {   // once per byte
      double wf = 0.0, wg = 0.0;
      for (int j = 0; j < nf; ++j) wf += s.nw[j];
      for (int j = nf; j < s.nn; ++j) wg += s.nw[j];
      s.wf = wf;
      s.wg = wg;
      s.wfg = 1;
    }
    f[18] = nf > 0 ? SemUtReadM(s.rm[2], s.wf) : 0.0;
    f[19] = ng > 0 ? SemUtReadM(s.rm[3], s.wg) : 0.0;
  }
  // cell = ((bit x 4 + similarity class) x 2 + agreement) x ... (class and
  // agreement only with a consistent neighbour); the byte part from SemUtByte
  const int c1 = k * 48 + s.uo1[info], c2 = k * 256 + s.pb, c3 = k * 512 + s.uo3[info];
  double z1 = 0.0, z2 = 0.0, z3 = 0.0;
  for (int i = 0; i < kUIn; ++i) { z1 += s.u4w[c1][i] * f[i]; z2 += s.u5w[c2][i] * f[i]; z3 += s.u6w[c3][i] * f[i]; }
  s.uz[0] = SemClamp(z1, 30.0);
  s.uz[1] = SemClamp(z2, 30.0);
  s.uz[2] = SemClamp(z3, 30.0);
  s.uc[0] = c1;
  s.uc[1] = c2;
  s.uc[2] = c3;
  const double* fw = s.ufw[k];
  s.ufz = SemClamp(fw[0] * s.uz[0] + fw[1] * s.uz[1] + fw[2] * s.uz[2] + fw[3], 30.0);
  s.cell = k;
  s.ufe = exp(-s.ufz);   // the learning step of the final stage reads it
  double v = floor(1.0 / (1.0 + s.ufe) * 4294967296.0 + 0.5);
  if (v < 1.0) v = 1.0;
  if (v > 4294967295.0) v = 4294967295.0;
  return (uint32_t)v;
}

__attribute__((noinline)) void Predictor::CmxSem2UtLearn(int sk, int bit) {
  using namespace semantic_cache;
  semantic_cache::State& s = *cmx_sk_[sk];
  if (s.cell < 0) return;
  const double lr = (double)kLrE6 * 1e-6;
  SemUtStep(s.u4w[s.uc[0]], s.u4g[s.uc[0]], s.uf, kUIn, s.uz[0], bit, lr);
  SemUtStep(s.u5w[s.uc[1]], s.u5g[s.uc[1]], s.uf, kUIn, s.uz[1], bit, lr);
  SemUtStep(s.u6w[s.uc[2]], s.u6g[s.uc[2]], s.uf, kUIn, s.uz[2], bit, lr);
  {
    const double ff[4] = {s.uz[0], s.uz[1], s.uz[2], 1.0};
    SemUtStepE(s.ufw[s.cell], s.ufg[s.cell], ff, 4, s.ufe, bit, lr * 0.2);
  }
  s.cell = -1;
}

__attribute__((noinline)) void Predictor::CmxSem2PredBit(int sk) {
  semantic_cache::State& s = *cmx_sk_[sk];
  s.ux = 0.0;
  if (manager_.bit_context_ == 1u) {   // first bit of a byte
    s.uact = s.ufresh;
    s.ufresh = 0;
  }
  s.rok = 0;
  s.rx = s.rc = 0.0;
  if (s.nn > 0) {
    const unsigned int ctx = manager_.bit_context_;
    const int k = 31 - __builtin_clz(ctx | 1u);
    // part sums of the groups consistent with this prefix (SemUtByte), by
    // the value of this bit (groups in byte order: value 0 in front of 1);
    // the end stage reads all seven, the main readout the first
    double g[2][8] = {};
    for (int i = 0; i < s.ngr; ++i) {
      const unsigned int c = 0x100u | s.gb[i];
      if ((c >> (8 - k)) != ctx) continue;
      double* t = g[(c >> (7 - k)) & 1u];
      for (int r = 0; r < 8; ++r) t[r] += s.gm[i][r];
    }
    for (int r = 0; r < 8; ++r) {
      s.rm[r][0] = g[0][r];
      s.rm[r][1] = g[1][r];
    }
    const double m0 = g[0][0], m1 = g[1][0];
    if ((m0 + m1) > 0.0) {
      s.rok = 1;
      s.rx = log((m1 / s.wsum + 0.01) / (m0 / s.wsum + 0.01));
      s.rc = (m0 + m1) / s.wsum;
    }
    if (s.uact) {
      s.ux = s.rx;
      // the rows of the end stage this bit reads and learns (SemUtCorrect,
      // SemUtLearn; cells as there): loaded here, long before they are used
      const int info = s.rc > 0.0;
      const int c1 = k * 48 + s.uo1[info], c2 = k * 256 + s.pb, c3 = k * 512 + s.uo3[info];
      SemPfRow(s.u4w[c1]);
      SemPfRow(s.u4g[c1]);
      SemPfRow(s.u5w[c2]);
      SemPfRow(s.u5g[c2]);
      SemPfRow(s.u6w[c3]);
      SemPfRow(s.u6g[c3]);
    }
  }
}

// the extra caches' byte starts, behind the release SemByteStart of the same byte (Compress and Decompress), in cache
// order: per cache the telemetry of the last byte (the release and its top-1 against this stream byte), then its steps
// 1-8
__attribute__((noinline)) void Predictor::CmxSem2ByteStart(int strom_pb) {
  const int pb = strom_pb & 0xFF;
  const int h1 = cmx_sk_h1_;
  for (int sk = 0; sk < cmx_meas::g_nsk; ++sk) {
    {
      unsigned long long* c = cmx_meas::g_sk2_cnt[sk][cmx_meas::g_region];
      const int h2 = cmx_sk_h2_[sk];
      if (h2 >= 0) ++c[0];
      if (h1 >= 0) ++c[1];
      if (h1 >= 0 && h2 >= 0) {
        ++c[2];
        c[3] += h1 == h2;
        c[4] += h1 == pb;
        c[5] += h2 == pb;
        c[6] += h2 == pb && h1 != pb;
        c[7] += h1 == pb && h2 != pb;
      } else if (h2 >= 0) {
        ++c[8];
        c[9] += h2 == pb;
      } else if (h1 >= 0) {
        ++c[11];
      }
    }
    CmxSem2Byte(sk, strom_pb);
    if (cmx_sk_[sk]->pend) ++cmx_meas::g_sk2_cnt[sk][cmx_meas::g_region][10];
    cmx_sk_h2_[sk] = cmx_sk_[sk]->nn > 0 ? cmx_sk_[sk]->topb : -1;
  }
  cmx_sk_h1_ = sem_.nn > 0 ? sem_.topb : -1;
}
// SemCorrect with CMX_SEMKEY2_USE bit 2: the release end stage, then the extra caches' end stages in series (cache 1
// on the release output, cache k + 1 on cache k's; the same "sure" rule on each one's own input), then the top-1 path
// stage; each learns on its own output (SemLearn)
__attribute__((noinline)) uint32_t Predictor::CmxSem2Correct(uint32_t p32) {
  semantic_cache::State& s = sem_;
  s.cell = -1;
  constexpr uint32_t kSure = (uint32_t)((1000 - semantic_cache::kSureE3) * 4294967.296);
  const bool sure = p32 <= kSure || p32 >= 4294967295u - kSure;
  uint32_t pu = (s.nn == 0 || sure) ? p32 : SemUtCorrect(p32);
  for (int sk = 0; sk < cmx_meas::g_nsk; ++sk) {
    semantic_cache::State& t = *cmx_sk_[sk];
    t.cell = -1;
    const bool sure2 = pu <= kSure || pu >= 4294967295u - kSure;
    pu = (t.nn == 0 || sure2) ? pu : CmxSem2UtCorrect(sk, pu);
  }
  return TreffCorrect(pu);
}
// zero-byte report, per extra cache: per telemetry region the bytes with a vote of that cache / the release cache, with
// both, how often their top-1 bytes agree and which one is right, that cache alone, its stored entries
__attribute__((noinline)) void Predictor::CmxSem2Report(const char* tag) {
  for (int sk = 0; sk < cmx_meas::g_nsk; ++sk) {
    const semantic_cache::State* t = cmx_sk_[sk];
    if (!t) continue;
    for (int r = 0; r < 4; ++r) {
      const unsigned long long* c = cmx_meas::g_sk2_cnt[sk][r];
      if (!(c[0] | c[1] | c[10])) continue;
      fprintf(stderr, "CMXTEL semkey2 %s cache=%d layer=%s use=%d region=%d stored2=%llu voted2=%llu voted1=%llu "
              "both=%llu both_same_top1=%llu both_hit1=%llu both_hit2=%llu both_only2_right=%llu both_only1_right=%llu "
              "alone2=%llu alone2_hit=%llu alone1=%llu\n", tag, sk + 1, cmx_meas::g_sk_name[sk], cmx_meas::g_sk2_use,
              r, c[10], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[11]);
    }
    fprintf(stderr, "CMXTEL semkey2 %s cache=%d layer=%s entries2=%llu key_mean_lookups2=%u hit_rate2=%.4f "
            "hit_rate1=%.4f\n", tag, sk + 1, cmx_meas::g_sk_name[sk], (unsigned long long)t->nins, t->kcnt, t->hrate,
            sem_.hrate);
  }
  fflush(stderr);
}
#endif

namespace {
static inline double TreffClamp(double x, double a) { return x > a ? a : (x < -a ? -a : x); }
static inline double TreffLogit(double p) {
  if (!(p > 1e-300)) return -40.0;
  if (!(p < 1.0)) return 40.0;
  return log(p) - semantic_cache::Log1p(-p);
}
// log-odds of byte b in a row (vocabulary entries only, as the slots read it)
static inline double TreffRowLogit(const float* t, int b) {
  double s = 0.0;
  for (int i = 0; i < 256; ++i) s += (double)t[i];
  if (!(s > 0.0)) return 0.0;
  const double rest = s - (double)t[b];
  if (!(t[b] > 0.0f)) return -40.0;
  if (!(rest > 0.0)) return 40.0;
  return log((double)t[b]) - log(rest);
}
// index of the first largest entry of a 256-entry row (rows hold no NaN):
// the same as the scan "m = 0; if (t[i] > t[m]) m = i", i.e. ties to the
// lower index (+0 and -0 compare equal in both)
__attribute__((noinline)) int TreffArgmax(const float* t) {
  __m256 m = _mm256_loadu_ps(t);
  for (int i = 8; i < 256; i += 8) m = _mm256_max_ps(m, _mm256_loadu_ps(t + i));
  m = _mm256_max_ps(m, _mm256_permute2f128_ps(m, m, 1));
  m = _mm256_max_ps(m, _mm256_shuffle_ps(m, m, 0x4E));
  m = _mm256_max_ps(m, _mm256_shuffle_ps(m, m, 0xB1));
  for (int i = 0; i < 256; i += 8) {
    const int k = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(t + i), m, _CMP_EQ_OQ));
    if (k) return i + __builtin_ctz(k);
  }
  return 0;
}
}  // namespace

__attribute__((noinline)) void Predictor::TreffByteStart(int last) {
  cache_hit::State& s = treff_;
  s.cell = -1;
  const double miss = (s.prev_t1 >= 0 && s.prev_t1 != last) ? 1.0 : 0.0;
  s.t1 = -1;
  s.prev_t1 = -1;
  if (!tf1_slot_->KtxLive()) return;   // no live transformer row for this byte
  const float* t = &tf1_slot_->BytePredict()[0];
  const int t1 = TreffArgmax(t);
  s.a1 = t1;
  if (!(t[t1] > 0.0f)) return;
  s.t1 = t1;
  s.prev_t1 = t1;
  double* fb = s.fb;
  for (int i = 0; i < cache_hit::kNF; ++i) fb[i] = 0.0;
  s.mbyte = -1;
  s.mlen = 0;
  if (tf1_match_count_ > 0) {
    unsigned int best = tf1_match_base_;
    unsigned char best_len = match_models_[tf1_match_base_].MatchLength();
    for (unsigned int q = 1; q < tf1_match_count_; ++q) {
      const unsigned char l = match_models_[tf1_match_base_ + q].MatchLength();
      if (l > best_len) { best_len = l; best = tf1_match_base_ + q; }
    }
    s.mbyte = match_models_[best].MatchByte();
    s.mlen = best_len;
  }
  if (s.mlen > 0) {
    if (s.mbyte == t1) { fb[3] = 1.0; fb[5] = semantic_cache::Log2(1.0 + s.mlen) / 6.0; }
    else fb[4] = 1.0;
  }
  {
    const float* q = &byte_model_->BytePredict()[0];
    fb[6] = TreffClamp(TreffRowLogit(q, t1), 16.0) / 8.0;
    s.pa = TreffArgmax(q);
    fb[7] = (s.pa == t1) ? 1.0 : 0.0;
  }
  {
    const float* q = &fw_slot_->BytePredict()[0];
    s.fa = TreffArgmax(q);
    fb[8] = (s.fa == t1) ? 1.0 : 0.0;
    fb[9] = TreffClamp(TreffRowLogit(q, t1), 16.0) / 8.0;
  }
  fb[13] = (last == 0x20) ? 1.0 : 0.0;
  fb[14] = miss;
}

__attribute__((noinline)) uint32_t Predictor::TreffCorrect(uint32_t p32) {
  cache_hit::State& s = treff_;
  s.cell = -1;
  if (s.t1 < 0) return p32;
  const unsigned int ctx = manager_.bit_context_;
  const int k = 31 - __builtin_clz(ctx | 1u);
  if (ctx != ((0x100u | (unsigned int)s.t1) >> (8 - k))) return p32;   // left the top-1 path
  const int d = (s.t1 >> (7 - k)) & 1;
  const double sg = d ? 1.0 : -1.0;
  double* f = s.f;
  for (int i = 0; i < cache_hit::kNF; ++i) f[i] = s.fb[i];
  f[0] = TreffClamp(sg * log((double)p32 / (4294967296.0 - (double)p32)), 30.0);
  f[1] = TreffClamp(sg * TreffLogit((double)tf1_slot_output), 30.0) / 8.0;
  f[2] = 1.0;
  f[10] = TreffClamp(sg * (double)adepth_.f[2], 12.0) / 8.0;
  f[11] = TreffClamp(sg * (double)layers_[0].Inputs()[mxs_ppmd_ix_], 12.0) / 8.0;
  f[12] = TreffClamp(sg * TreffLogit((double)fw_slot_output), 16.0) / 8.0;
  f[15] = adepth_.skip ? 1.0 : 0.0;
  double z = 0.0;
  for (int i = 0; i < cache_hit::kNF; ++i) z += s.w[i] * f[i];
  z = TreffClamp(z, 30.0);
  s.cell = 0;
  s.d = d;
  s.z = z;
  const double q = 1.0 / (1.0 + exp(-z));            // P(top-1 bit)
  const double p1 = d ? q : 1.0 - q;                   // P(bit = 1)
  double v = floor(p1 * 4294967296.0 + 0.5);
  if (v < 1.0) v = 1.0;
  if (v > 4294967295.0) v = 4294967295.0;
  return (uint32_t)v;
}

__attribute__((noinline)) void Predictor::TreffLearn(int bit) {
  cache_hit::State& s = treff_;
  if (s.cell < 0) return;
  const double y = (bit == s.d) ? 1.0 : 0.0;
  const double err = 1.0 / (1.0 + exp(-s.z)) - y;
  const double lr = (double)cache_hit::kLrE6 * 1e-6;
  double* w = s.w;
  double* g2 = s.g2;
  for (int i = 0; i < cache_hit::kNF; ++i) {
    const double g = err * s.f[i];
    g2[i] += g * g;
    w[i] -= lr * g / (sqrt(g2[i]) + 1e-6);
  }
  s.cell = -1;
}

// The column count is the one the head adds to its row (bit_head.h).
namespace semantic_head_cols { constexpr int kCols = 4; }
static_assert(semantic_head_cols::kCols == 4, "SemAppend() writes four columns, HEAD_SEM_COLS must say so");

__attribute__((noinline)) void Predictor::SemAppend() {
  static_assert(22 + 13 + semantic_head_cols::kCols <= (int)(sizeof(sem_s1buf_) / sizeof(sem_s1buf_[0])),
                "the stage-1 row, the columns of MlAppend() and the cache columns must fit sem_s1buf_");
  semantic_cache::State& s = sem_;
  const int n = bitl_n_stage1_;
  float* row = sem_s1buf_;   // the row plus the cache columns
  for (int j = 0; j < n; ++j) row[j] = bitl_stage1_[j];
  bitl_stage1_ = row;
  float* o = row + n;
  for (int j = 0; j < semantic_head_cols::kCols; ++j) o[j] = 0.0f;
  if (s.rok) {   // the main readout of this bit (SemPredBit)
    o[0] = (float)s.rx;
    o[1] = (float)s.rc;
    o[2] = s.b_smax;
    o[3] = s.agr ? 1.0f : -1.0f;
  }
  s.b_x0 = o[0];
  bitl_n_stage1_ = n + semantic_head_cols::kCols;
}

// Exact end-to-end gradient of the mixer cascade: gamma_i = dz/dh_i over the
// layer-0 cascade, exact because the stack is affine in h.  The backward
// recursion follows src/predictor_lex.zig of zmix 1.0 by James Byrne
// (GPL-3.0-or-later), rewritten in C++.  It runs while every row of this bit
// is still resolved (Perceive() consumes primed_data_), so it stands BEFORE
// the update loop and reads the PRE-update weights, which is what the
// gradient is defined on.
void Predictor::E2EGrad(int bit) {
  const unsigned int n = (unsigned int)mixer_0_.size();
  if (n == 0 || n > mixer_chain::kMaxMixers) return;   // stock rule stays
  // (sigma(z) - y) on the RAW layer-1 output.  That is the head zmix
  // differentiates through as well; the SSE chain and the BITLSTM sit behind
  // it and are not part of this gradient.
  // the final-mixer output of this bit (all rows plus the Bayes term)
  const float g_final = Sigmoid::Logistic(end_z_) - (float)bit;
  float gamma[mixer_chain::kMaxMixers];
  const mixw_t* xw[mixer_chain::kMaxMixers];
  for (unsigned int k = 0; k < n; ++k) xw[k] = mixer_0_[k].E2EXRow();
  // the chain factor of the direct path is the summed row
  float w1s[mixer_chain::kMaxMixers];
  KtxW1Sum(w1s, n);
  const float* const w1 = w1s;
  unsigned int i = n;
  while (i > 0) {
    --i;
    // direct path into the layer-1 mixer ...
    float s = MIXER_CHAIN_W(w1[i]);
    // ... plus every later cascade stage, which reads clamp(h_i) as its extra
    // input number i (mixer k has exactly k extra inputs).
    for (unsigned int k = i + 1; k < n; ++k) s += gamma[k] * MIXER_CHAIN_W(xw[k][i]);
    const float h = mixer_0_[i].E2EP();
    // clamp gate: d clamp(h)/dh is 0 inside the saturated region, so a mixer
    // sitting in the clamp has no influence on z and gets no update.
    gamma[i] = (h > (0x1.868cfep+3f) || h < (-0x1.86983p+3f)) ? 0.0f : s;
  }
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_valtel != 0, 0)) CmxValTel(bit, gamma, n);   // research CMX_VALTEL (zero-byte)
#endif
  float scale = 1.0f;
  for (unsigned int j = 0; j < n; ++j)
    mixer_0_[j].E2ESetErr(g_final * gamma[j] * scale);
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_wl_on, 0)) {   // research WL-MIXGRAD: delta_wl on the PRE-update rows
    float d = 0.0f;
    for (unsigned int k = 0; k < n; ++k) d += gamma[k] * mixer_0_[k].E2EWRow()[cmx_meas::g_wl_idx];
    cmx_meas::g_wl_delta = g_final * d;
  }
#endif
}

 void Predictor::Perceive(int bit) {
  CMXK_START(cmx_ck)
#if PORT_CMX_MEAS
  if (__builtin_expect(cmx_meas::g_wl_on, 0)) cmx_meas::g_wl_delta = 0.0f;   // set by E2EGrad if the cascade runs
  else
#endif
  word_label::Perceive(wordlbl_, bit);
  CMXK_LAP(CK_WLU, cmx_ck)
  pagectx::surprise.AddBit(last_p16_, bit, manager_.bit_context_ >= 128);
  bracket_model_->Perceive(bit);

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }
  grammar_match_->Perceive(bit);
  {}
  CMXK_LAP(CK_CMIXU, cmx_ck)
  byte_model_->Perceive(bit);
  tf1_slot_->Perceive(bit);
  FwPerceive(bit);
  CMXK_LAP(CK_SLOTU, cmx_ck)

  if (char_rate::Forced(manager_.bit_context_)) {
    sse_.PerceiveContextOnly(bit);
    for (auto& mixer: mixer_0_) mixer.PerceiveSkip();
    for (auto& mixer: mixer_1_) mixer.PerceiveSkip();
#if PORT_CMX_MEAS
  } else if (adm_loff_) {   // research CMX_ADEPTH_MODE: = adepth_.skip with the knob unset (Predict)
    // mode 1 / tau_code > tau_learn: cvz.End may have run in the cascade's forward pass on this bit; no step on it, as
    // released (where pctx is already -1 here)
    if (__builtin_expect(cmx_meas::g_adm == 1 || cmx_meas::g_tauc_set, 0)) invariants::cvz.pctx = -1;
#else
  } else if (adepth_.skip) {
#endif
    sse_.PerceiveContextOnly(bit);
    for (auto& mixer: mixer_0_) mixer.PerceiveSkip();
    for (auto& mixer: mixer_1_) mixer.PerceiveSkip();
  } else {
  E2EGrad(bit);
#if PORT_CMX_MEAS
  CMXK_LAP(CK_E2E, cmx_ck)
  {   // research CMX_CM_TICKS: the preconditioned row first (the release argument), then the step on it
    const float* const cmx_xt = mixer_precond::L0ColScale(mxp_l0col_, &layers_[0].Inputs()[0],
                                                          (int)layers_[0].Inputs().size());
    CMXK_LAP(CK_PRECOND, cmx_ck)
    PerceiveBankX(&mixer_0_[0], mixer_0_.size(), bit, cmx_xt);
  }
  CMXK_LAP(CK_BANKU, cmx_ck)
#else
  PerceiveBankX(&mixer_0_[0], mixer_0_.size(), bit,
                mixer_precond::L0ColScale(mxp_l0col_, &layers_[0].Inputs()[0],
                                     (int)layers_[0].Inputs().size()));
#endif
  EndLearn(bit);   // every final-mixer row on the error of z
  for (auto& mixer: mixer_1_) {
    mixer.Perceive(bit);
  }

  sse_.Perceive(bit);
  }
  CMXK_LAP(CK_FINALU, cmx_ck)
#if PORT_CMX_MEAS
  // research WL-MIXGRAD: the word-label rows behind the cascade, with kappa * delta_wl (0 if the cascade was skipped)
  if (__builtin_expect(cmx_meas::g_wl_on, 0))
    word_label::PerceiveG(wordlbl_, bit, cmx_meas::g_wl_live ? cmx_meas::g_wl_kappa * cmx_meas::g_wl_delta : 0.0f);
  CMXK_LAP(CK_WLU, cmx_ck)
#endif
  if (!char_rate::Forced(manager_.bit_context_)) {
    adaptive_depth::TrainStep(adepth_, bit);
    ++adepth_.coded;
  }
  CMXK_LAP(CK_GATEU, cmx_ck)

  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;

  if (byte_update)
    id_exclude::Byte(((manager_.bit_context_ << 1) | (unsigned int)bit) & 0xFFu);
  invariants::cvz.Perceive(bit, manager_.bit_context_);
  manager_.UpdateContexts(bit);
  CMXK_LAP(CK_CTX, cmx_ck)
  if (byte_update) {
    // The completed byte (bit_context_ holds it here) into the
    // ISBN state; the rule of the next byte reads it.
    invariants::isbn2.Byte(manager_.bit_context_ & 0xFFu);
    invariants::umr.Byte(manager_.bit_context_ & 0xFFu);   // Conversion rule
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }

    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
    grammar_match_->ByteUpdate();
    CMXK_LAP(CK_CMIXB, cmx_ck)
    byte_model_->ByteUpdate();
    CMXK_LAP(CK_PPMDB, cmx_ck)
    if (tf1_prior_writer_) {
      const std::valarray<float>& tf1_pp = byte_model_->BytePredict();
      for (int i = 0; i < transformer::kSetSize; ++i) {
        tf1_prior_row_[i] = tf1_pp[tf1_vocab_bytes_[i]];
      }
      tf1_prior_writer_->Write(tf1_prior_row_.data(), transformer::kSetSize);
    }

    if (tf1_engine_ && tf1_match_count_ > 0) {
      unsigned int best = tf1_match_base_;
      unsigned char best_len = match_models_[tf1_match_base_].MatchLength();
      for (unsigned int q = 1; q < tf1_match_count_; ++q) {
        unsigned char l = match_models_[tf1_match_base_ + q].MatchLength();
        if (l > best_len) { best_len = l; best = tf1_match_base_ + q; }
      }
      tf1_engine_->SetMatch(match_models_[best].MatchByte(), (int)best_len);
    }
    const float* tf1_row = tf1_engine_
        ? tf1_engine_->Step(byte_model_->BytePredict(), manager_.bit_context_)
        : nullptr;
    CMXK_LAP(CK_TF, cmx_ck)
    if (tf1_row && transformer::Active()) {
      if (tf1_engine_->GateMasked()
          || alias_prompt::Laeuft()   // alias rows are being fed
          || timestamp_skip::Laeuft()
#if PORT_CMX_MEAS
          // research CMX_TFOFF: every byte takes the gated form (neutral row, as inside the metadata blocks)
          || cmx_meas::g_tfoff
#endif
         ) {
        tf1_slot_->SetProbsNeutral();
        FwSetNeutral();
      } else
      {
        tf1_slot_->SetProbs(tf1_row);
#if PORT_CMX_MEAS
        if (__builtin_expect(cmx_meas::g_fwoff != 0, 0)) CmxFwOffRow(tf1_row);   // research CMX_FWOFF
        else
#endif
        FwSetRow();
      }
    } else {
      // no transformer: abstention row, arithmetically no slot at all.
      tf1_slot_->SetProbsOff();
      FwSetOff();
    }
    CMXK_LAP(CK_TFROW, cmx_ck)
  }
  tf1_slot_output = tf1_slot_->Predict()[0];
  FwPredictBit();
  MlPath(byte_update);   // path state of the next bit (selector of the add rows)
  CMXK_LAP(CK_SLOTB, cmx_ck)
#if PORT_CMX_MEAS
  // research CMX_FXLS: fxcm learns from a gate bit without its mixer / APM steps (= adepth_.skip with the knob unset)
  if (__builtin_expect(cmx_meas::g_fxls != 0, 0)) cmx_meas::g_fxls_off = adm_loff_;
#endif
  fxcm_model_.Perceive(bit);
  CMXK_LAP(CK_FXUPD, cmx_ck)
  if (byte_update)manager_.bit_context_ = 1;
}

 void Predictor::ObserveKnownBit(int bit) {
  Predict();
  Perceive(bit);
}

void Predictor::ObserveTfOnlyByte(unsigned int byte) {
  if (tf1_engine_) tf1_engine_->Step(byte_model_->BytePredict(), byte);
}

void Predictor::Pretrain(int bit) {
  bracket_model_->Predict();
  fxcm_model_.PredictHalf();
  fxcm_model_.Predict();

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Predict();
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Predict();
  }
  grammar_match_->Predict();

  bracket_model_->Perceive(bit);
#if PORT_CMX_MEAS
  cmx_meas::g_fxls_off = false;   // research CMX_FXLS: a pretraining bit is no gate bit
#endif
  fxcm_model_.Perceive(bit);

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }
  grammar_match_->Perceive(bit);

  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;
  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
    grammar_match_->ByteUpdate();
    manager_.bit_context_ = 1;
  }
}


#if PORT_CMX_MEAS
// ---- research telemetry and the page-1 re-key (PORT_CMX_MEAS, src/cmx_meas.h; Claude Opus 5.5, 2026-10-31) -----
// CMX_MIXROW 1/2: at page 1 every layer-0 mixer whose map is full (the release cap) releases its rows to its pool and
// keys them again first-come; mixers below the cap keep their rows (conservative reading: no fallback there).
__attribute__((noinline)) void Predictor::CmxMixRekey() {
  fprintf(stderr, "CMXTEL mixrow rekey mode=%d limit=%llu rows:", cmx_meas::g_mixrow, cmx_meas::g_mix_limit);
  for (unsigned int i = 0; i < mixer_0_.size(); ++i) {
    const size_t r = mixer_0_[i].CmxRows();
    const bool full = r >= cmx_meas::g_mix_limit;
    fprintf(stderr, " %u:%zu%s", i, r, full ? "*" : "");
    if (full) mixer_0_[i].CmxRekey();
  }
  fprintf(stderr, "\n");
  fflush(stderr);
}
// research CMX_SEM_KCEN0: the running key mean starts again (runner, at first_article)
__attribute__((noinline)) void Predictor::CmxSemKeyReset() {
  semantic_cache::State& s = sem_;
  fprintf(stderr, "CMXTEL semkey reset: kcnt was %u frozen=%d\n", (unsigned)s.kcnt, (int)s.kmf);
  for (int c = 0; c < 192; ++c) s.kmu[c] = 0.0;
  s.kcnt = 0;
  s.kmf = 0;
}
__attribute__((noinline)) void Predictor::CmxMixTel() {
  const int r = cmx_meas::g_region;
  ++cmx_meas::g_fbn[r];
  const unsigned int n = mixer_0_.size() < 24 ? (unsigned int)mixer_0_.size() : 24u;
  for (unsigned int i = 0; i < n; ++i) cmx_meas::g_fb[r][i] += mixer_0_[i].CmxOnBase() ? 1u : 0u;
}
// the mixers in AddMixers() order: mx9 mx10 recent_bytes2 line_break longest_match mx19cxt auxiliary mx18 mx7
// wordscxt b2streamcxt mx5 mx6 b3streamcxt mx8 mx17 mx16 mx14 mx15 lx1_agree
__attribute__((noinline)) void Predictor::CmxMixReport(const char* tag) {
  for (int r = 0; r < 4; ++r) {
    if (!cmx_meas::g_fbn[r]) continue;
    fprintf(stderr, "CMXTEL mixfb %s region=%d primed_bits=%llu fallback_frac:", tag, r, cmx_meas::g_fbn[r]);
    for (unsigned int i = 0; i < mixer_0_.size() && i < 24; ++i)
      fprintf(stderr, " %.4f", (double)cmx_meas::g_fb[r][i] / (double)cmx_meas::g_fbn[r]);
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "CMXTEL mixrows %s:", tag);
  for (unsigned int i = 0; i < mixer_0_.size(); ++i) fprintf(stderr, " %zu", mixer_0_[i].CmxRows());
  fprintf(stderr, "\n");
  fflush(stderr);
}
// CMX_SEMTEL report: per region stores, keyed bytes, lookups, lookups with >= 1 fresh / any neighbour, log2 histograms
// of the eviction age and of the fresh-neighbour age (stream bytes); hyperplane balance; frozen vs article key mean
__attribute__((noinline)) void Predictor::CmxSemReport(const char* tag) {
  using namespace cmx_meas;
  for (int r = 0; r < 4; ++r) {
    if (!g_sem_keyed[r] && !g_sem_store[r]) continue;
    fprintf(stderr, "CMXTEL sem %s region=%d keyed=%llu stores=%llu lookups=%llu with_fresh=%llu with_any=%llu\n", tag, r,
            g_sem_keyed[r], g_sem_store[r], g_sem_look[r], g_sem_look_nk[r], g_sem_look_nb[r]);
    fprintf(stderr, "CMXTEL sem_evict_age_log2 %s region=%d:", tag, r);
    for (int b = 0; b < kAgeBins; ++b) fprintf(stderr, " %llu", g_sem_evict[r][b]);
    fprintf(stderr, "\nCMXTEL sem_nbr_age_log2 %s region=%d:", tag, r);
    for (int b = 0; b < kAgeBins; ++b) fprintf(stderr, " %llu", g_sem_nage[r][b]);
    fprintf(stderr, "\n");
  }
  unsigned long long kn = 0;
  for (int r = 0; r < 4; ++r) kn += g_sem_keyed[r];
  if (kn) {
    double mn = 1.0, mx = 0.0, dev = 0.0;
    for (int i = 0; i < semantic_cache::kL * semantic_cache::kB; ++i) {
      const double f = (double)g_sem_hyp1[i] / (double)kn;
      if (f < mn) mn = f;
      if (f > mx) mx = f;
      dev += (f - 0.5) * (f - 0.5);
    }
    fprintf(stderr, "CMXTEL sem_hyp %s keyed=%llu p1_min=%.4f p1_max=%.4f rms_dev_from_half=%.4f p1:", tag, kn, mn, mx,
            sqrt(dev / (semantic_cache::kL * semantic_cache::kB)));
    for (int i = 0; i < semantic_cache::kL * semantic_cache::kB; ++i) fprintf(stderr, " %.3f", (double)g_sem_hyp1[i] / (double)kn);
    fprintf(stderr, "\n");
  }
  const semantic_cache::State& st = sem_;
  if (g_sem_kn) {
    double mu_art[192], fr[192], d2 = 0.0, ma2 = 0.0, fr2 = 0.0;
    const double inv = st.kcnt ? 1.0 / (double)st.kcnt : 0.0;
    for (int c = 0; c < 192; ++c) {
      mu_art[c] = g_sem_ksum[c] / (double)g_sem_kn;
      fr[c] = st.kmf ? (double)st.kmuf[c] : st.kmu[c] * inv;
      d2 += (fr[c] - mu_art[c]) * (fr[c] - mu_art[c]);
      ma2 += mu_art[c] * mu_art[c];
      fr2 += fr[c] * fr[c];
    }
    const double ex2 = g_sem_ksq / (double)g_sem_kn;   // E ||x||^2 over article lookups
    const double var = ex2 - ma2;                     // E ||x - mu_art||^2
    fprintf(stderr, "CMXTEL sem_keymean %s frozen=%d kcnt=%llu art_lookups=%llu |frozen-art|=%.6g key_rms=%.6g "
            "spread_rms=%.6g rel_to_key_rms=%.4f rel_to_spread=%.4f |frozen|=%.6g |art|=%.6g\n", tag, (int)st.kmf,
            (unsigned long long)st.kcnt, g_sem_kn, sqrt(d2), sqrt(ex2), sqrt(var > 0 ? var : 0), sqrt(d2) / sqrt(ex2),
            var > 0 ? sqrt(d2) / sqrt(var) : 0.0, sqrt(fr2), sqrt(ma2));
  }
  fflush(stderr);
}

// ---- research CMX_FWOFF / CMX_VALTEL (CM census phase 2a, src/cmx_meas.h; Claude Opus 5.5, 2026-11-08) ------------
// CMX_FWOFF: on a byte with a live transformer row the fast-weight slot takes that row, which is what the second head
// outputs with zero fast weights; no live fast-weight row, so Tf1Engine takes no fw_update for the byte.
__attribute__((noinline)) void Predictor::CmxFwOffRow(const float* row) {
  fw_slot_->SetProbs(row);
  tf1_engine_->FwNeutral();
}

namespace {
// CMX_VALTEL groups: layer-0 index ranges [clo, chi), fxcm mxInputs1 ranges [flo, fhi) and the Bayes experts they hold
// (bit k = expert k of MxsBayMix: 0 transformer, 1 fast-weight head, 2 PPMd, 3 fxcm aux)
struct CmxVG {
  std::string name;
  std::vector<unsigned> clo, chi, flo, fhi;
  unsigned ex = 0;
};
std::vector<CmxVG> g_vg;
enum { kVaN = 0, kVaLive, kVaMabs, kVaOcc, kVaOccFx, kVaX2, kVaWx, kVaK };
std::vector<double> g_vacc[4];
double g_vmix[4][mixer_chain::kMaxMixers][3];   // per bank mixer: |gamma|, occlusion bits, samples
double g_vbase[4][2];                           // sampled cascade bits, their code length at z (bits)
unsigned long long g_vcall = 0;
bool g_vbuilt = false;
unsigned g_vn = 0, g_vmxlo = 0;                 // bank mixers; layer-0 index of the mxA[0] output
std::vector<double> g_vg_j, g_vwx, g_vpc, g_vpx2, g_vpwx, g_vpnz;
// -log2 P(bit) for P(1) = sigma(z)
inline double CmxBits(double z, int bit) {
  const double s = bit ? z : -z;
  return ((s < 0.0 ? -s : 0.0) + log1p(exp(-fabs(s)))) * 1.4426950408889634;
}
int CmxVgFind(const std::string& nm) {
  for (size_t i = 0; i < g_vg.size(); ++i) if (g_vg[i].name == nm) return (int)i;
  return -1;
}
// a composite group: the union of its members (members are disjoint input sets); false if a member is unknown
bool CmxVgComp(const std::string& nm, const std::vector<std::string>& mem) {
  CmxVG g;
  g.name = nm;
  for (const std::string& m : mem) {
    const int i = CmxVgFind(m);
    if (i < 0) {
      fprintf(stderr, "CMXTEL valtel group %s: no member %s (group dropped)\n", nm.c_str(), m.c_str());
      return false;
    }
    const CmxVG& s = g_vg[(size_t)i];
    g.clo.insert(g.clo.end(), s.clo.begin(), s.clo.end());
    g.chi.insert(g.chi.end(), s.chi.begin(), s.chi.end());
    g.flo.insert(g.flo.end(), s.flo.begin(), s.flo.end());
    g.fhi.insert(g.fhi.end(), s.fhi.begin(), s.fhi.end());
    g.ex |= s.ex;
  }
  g_vg.push_back(g);
  return true;
}
std::vector<std::string> CmxVgPrefix(const char* pre) {
  std::vector<std::string> v;
  const size_t n = strlen(pre);
  for (const CmxVG& g : g_vg) if (g.name.compare(0, n, pre) == 0) v.push_back(g.name);
  return v;
}
std::vector<std::string> CmxVgSeq(const char* pre, int a, int b) {
  std::vector<std::string> v;
  for (int i = a; i <= b; ++i) v.push_back(std::string(pre) + std::to_string(i));
  return v;
}
}  // namespace

// The groups, once fxcm's layout is complete: every cmix model, every fxcm site (fxcmv1.cpp FxSite) with inputs, the
// two fxcm filler slots, the composites of the census plan and those of CMX_VALTEL_GROUPS.
__attribute__((noinline)) COLD_SIZE_ATTR bool Predictor::CmxValBuild(unsigned int n) {
  const unsigned nfx = fxcm_model_.NumOutputs();
  if (fxcm_model_.ActivePredictions() + 2u != nfx) return false;   // fxcm has not emitted its full row yet
  auto add = [](const std::string& nm, unsigned lo, unsigned hi, unsigned flo, unsigned fhi, unsigned ex) {
    CmxVG g;
    g.name = nm;
    if (hi > lo) { g.clo.push_back(lo); g.chi.push_back(hi); }
    if (fhi > flo) { g.flo.push_back(flo); g.fhi.push_back(fhi); }
    g.ex = ex;
    g_vg.push_back(g);
  };
  add("bracket", 0, 1, 0, 0, 0);
  for (int id = 0; id < cmx_meas::kFxN; ++id) {
    if (!cmx_meas::g_fxname[id]) continue;
    const unsigned* l = cmx_meas::g_fxlay[id];
    if (l[1] <= l[0] && l[3] <= l[2]) continue;
    add(std::string("fx_") + cmx_meas::g_fxname[id], 1 + l[0], 1 + l[1], l[2], l[3], 0);
    if (!strcmp(cmx_meas::g_fxname[id], "mxA")) g_vmxlo = 1 + l[0];
  }
  add("fx_idx", nfx - 1, nfx, 0, 0, 0);   // id_exclude::MixDelta in the fxcm filler slot
  add("fx_aux", nfx, nfx + 1, 0, 0, 8u);  // fxcm's final prediction (also final-mixer column and Bayes expert)
  unsigned k = 1 + nfx;
  for (unsigned i = 0; i < direct_models_.size(); ++i, ++k) add("cm_direct" + std::to_string(i), k, k + 1, 0, 0, 0);
  for (unsigned i = 0; i < match_models_.size(); ++i, ++k)
    add(i < tf1_match_base_ ? "cm_match_w" + std::to_string(i) : "cm_match_h" + std::to_string(i - tf1_match_base_),
        k, k + 1, 0, 0, 0);
  add("grammar", k, k + 1, 0, 0, 0);
  ++k;
  for (unsigned i = 0; i < indirect_ns_models_.size(); ++i, ++k) {
    std::string nm = "cm_ind_ns" + std::to_string(i);   // AddBracket, AddWord x10, AddDoubleIndirect x2
    if (indirect_ns_models_.size() == 13) nm = i == 0 ? "cm_ind_bracket" : (i <= 10 ? "cm_ind_w" + std::to_string(i - 1)
                                                                                  : "cm_ind_d" + std::to_string(i - 11));
    add(nm, k, k + 1, 0, 0, 0);
  }
  for (unsigned i = 0; i < indirect_r_models_.size(); ++i, ++k) add("cm_ind_run" + std::to_string(i), k, k + 1, 0, 0, 0);
  add("ppmd", mxs_ppmd_ix_, mxs_ppmd_ix_ + 1, 0, 0, 4u);
  add("tf1", mxs_ppmd_ix_ + 1, mxs_ppmd_ix_ + 2, 0, 0, 1u);
  add("word_label", mxs_ppmd_ix_ + 2, mxs_ppmd_ix_ + 3, 0, 0, 0);
  add("fw", fw_index_, fw_index_ + 1, 0, 0, 2u);
  add("sem", sem_index_, sem_index_ + 1, 0, 0, 0);
  // the census plan's composites (plan_cm_time_census_20261108.md §6)
  CmxVgComp("cmcr18", CmxVgSeq("fx_cmcr_", 1, 8));
  CmxVgComp("cmcr2", CmxVgSeq("fx_cmcr2_", 0, 3));
  CmxVgComp("dsm", {"fx_dsmset", "fx_dcsm", "fx_dcsm0", "fx_dcsm1", "fx_dcsm2", "fx_dcsmN"});
  CmxVgComp("cmC4pl3", {"fx_cmC4_4", "fx_cmC4_7", "fx_cmC4_8"});
  CmxVgComp("cmC4pl", {"fx_cmC4_1", "fx_cmC4_2", "fx_cmC4_4", "fx_cmC4_6", "fx_cmC4_7", "fx_cmC4_8", "fx_cmC_0"});
  CmxVgComp("fxsmall", {"fx_cmC4_0", "fx_cmC4_3", "fx_cmC_2", "fx_cmC_3", "fx_cmCR_0", "fx_cmCR_1", "fx_cmCR_2",
                        "fx_cmC2_6", "fx_cmC2_15", "fx_cmLink"});
  CmxVgComp("fxmixstack", {"fx_mxA", "fx_mxA1", "fx_apm", "fx_aux"});
  CmxVgComp("cm_ind_word", CmxVgSeq("cm_ind_w", 0, 9));
  CmxVgComp("cm_match_word", CmxVgSeq("cm_match_w", 0, 4));
  CmxVgComp("cm_match_hash", CmxVgSeq("cm_match_h", 0, 4));
  {
    std::vector<std::string> m = CmxVgPrefix("cm_");
    m.push_back("grammar");
    CmxVgComp("cmix_small", m);
  }
  CmxVgComp("fxcm_all", CmxVgPrefix("fx_"));
  // CMX_VALTEL_GROUPS="name=a+b+c,name2=d+e" (members: any group above)
  for (const char* q = cmx_meas::g_valtel_groups; q && *q;) {
    const char* e = strchr(q, ',');
    const std::string item = e ? std::string(q, (size_t)(e - q)) : std::string(q);
    q = e ? e + 1 : nullptr;
    const size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0) { fprintf(stderr, "CMXTEL valtel bad group spec '%s'\n", item.c_str()); continue; }
    std::vector<std::string> mem;
    for (size_t a = eq + 1; a <= item.size();) {
      size_t b = item.find('+', a);
      if (b == std::string::npos) b = item.size();
      if (b > a) mem.push_back(item.substr(a, b - a));
      a = b + 1;
    }
    CmxVgComp(item.substr(0, eq), mem);
  }
  g_vn = n;
  for (int r = 0; r < 4; ++r) g_vacc[r].assign(g_vg.size() * kVaK, 0.0);
  const size_t N = layers_[0].Inputs().size();
  g_vg_j.assign(N, 0.0); g_vwx.assign(N, 0.0);
  g_vpc.assign(N + 1, 0.0); g_vpx2.assign(N + 1, 0.0); g_vpwx.assign(N + 1, 0.0); g_vpnz.assign(N + 1, 0.0);
  fprintf(stderr, "CMXTEL valtel init k=%d groups=%zu layer0=%zu mixers=%u mxA_out=%u\n", cmx_meas::g_valtel,
          g_vg.size(), N, n, g_vmxlo);
  fflush(stderr);
  return true;
}

// Every CMX_VALTEL-th cascade bit, from E2EGrad (pre-update rows, gamma of this bit, the bit known; nothing is
// written but the accumulators): the frozen-weight occlusion of every group and of every bank mixer's output.
__attribute__((noinline)) void Predictor::CmxValTel(int bit, const float* gamma, unsigned int n) {
  if ((++g_vcall % (unsigned long long)cmx_meas::g_valtel) != 0) return;
  const unsigned long long t0 = __rdtsc();
  if (!g_vbuilt) {
    if (!CmxValBuild(n)) return;
    g_vbuilt = true;
  }
  if (n != g_vn) return;
  const int r = cmx_meas::g_region;
  const float* const x = &layers_[0].Inputs()[0];
  const unsigned N = (unsigned)g_vg_j.size();
  for (unsigned j = 0; j < N; ++j) { g_vg_j[j] = 0.0; g_vwx[j] = 0.0; }
  for (unsigned i = 0; i < n; ++i) {
    const mixw_t* const w = mixer_0_[i].E2EWRow();
    const double gi = gamma[i];
    for (unsigned j = 0; j < N; ++j) {
      g_vg_j[j] += gi * (double)w[j];
      g_vwx[j] += fabs((double)w[j] * (double)x[j]);
    }
  }
  // the direct final-mixer columns behind the row: fxcm aux, transformer, fast-weight head, SEMCACHE
  float w1s[mixer_chain::kMaxMixers];
  KtxW1Sum(w1s, n + 4);
  g_vg_j[fxcm_model_.NumOutputs()] += w1s[n];
  g_vg_j[mxs_ppmd_ix_ + 1] += w1s[n + 1];
  g_vg_j[fw_index_] += w1s[n + 2];
  g_vg_j[sem_index_] += w1s[n + 3];
  for (unsigned j = 0; j < N; ++j) {
    const double xj = x[j];
    g_vpc[j + 1] = g_vpc[j] + xj * g_vg_j[j];
    g_vpx2[j + 1] = g_vpx2[j] + xj * xj;
    g_vpwx[j + 1] = g_vpwx[j] + g_vwx[j];
    g_vpnz[j + 1] = g_vpnz[j] + (xj != 0.0 ? 1.0 : 0.0);
  }
  const double z = end_z_;
  const double l0 = CmxBits(z, bit);
  g_vbase[r][0] += 1.0;
  g_vbase[r][1] += l0;
  const float* const s1 = &layers_[1].Inputs()[0];
  const float s4[final_mix::kExperts] = {s1[n + 1], s1[n + 2], x[mxs_ppmd_ix_], s1[n]};
  final_mix::Bay b0 = g_mxs_bay;   // copies: the release state is not touched
  const double ws0 = final_mix::BayMix(b0, 0.0f, s4);
  double* const acc = g_vacc[r].data();
  for (size_t gi = 0; gi < g_vg.size(); ++gi) {
    const CmxVG& G = g_vg[gi];
    double sc = 0.0, sx2 = 0.0, swx = 0.0, snz = 0.0;
    for (size_t q = 0; q < G.clo.size(); ++q) {
      const unsigned lo = G.clo[q], hi = G.chi[q] < N ? G.chi[q] : N;
      if (hi <= lo) continue;
      sc += g_vpc[hi] - g_vpc[lo];
      sx2 += g_vpx2[hi] - g_vpx2[lo];
      swx += g_vpwx[hi] - g_vpwx[lo];
      snz += g_vpnz[hi] - g_vpnz[lo];
    }
    double dz = -sc;
    if (G.ex) {
      float s4b[final_mix::kExperts];
      for (int e = 0; e < final_mix::kExperts; ++e) s4b[e] = (G.ex >> e) & 1u ? 0.0f : s4[e];
      final_mix::Bay b1 = g_mxs_bay;
      dz += (double)final_mix::BayMix(b1, 0.0f, s4b) - ws0;
    }
    const double l1 = CmxBits(z + dz, bit);
    double lfx = l1;
    if (!G.flo.empty() && g_vmxlo) {   // the mxA[0..16] outputs without the group's fxcm-internal inputs
      short o[17], nw[17];
      fxcm_model_.CmxValFx(G.flo.data(), G.fhi.data(), (int)G.flo.size(), o, nw);
      double dfx = 0.0;
      for (int i = 0; i < 17; ++i)
        dfx += g_vg_j[g_vmxlo + (unsigned)i] *
               ((double)fxcm_stretched_inputs_[nw[i] + 2047] - (double)fxcm_stretched_inputs_[o[i] + 2047]);
      lfx = CmxBits(z + dz + dfx, bit);
    }
    double* const a = acc + gi * kVaK;
    a[kVaN] += 1.0;
    a[kVaLive] += snz > 0.0 ? 1.0 : 0.0;
    a[kVaMabs] += fabs(dz);
    a[kVaOcc] += l1 - l0;
    a[kVaOccFx] += lfx - l0;
    a[kVaX2] += sx2;
    a[kVaWx] += swx;
  }
  for (unsigned i = 0; i < n && i < mixer_chain::kMaxMixers; ++i) {
    const double dz = -(double)gamma[i] * (double)s1[i];
    g_vmix[r][i][0] += fabs((double)gamma[i]);
    g_vmix[r][i][1] += CmxBits(z + dz, bit) - l0;
    g_vmix[r][i][2] += 1.0;
  }
  if (cmx_meas::g_cm_ticks) {
    cmx_meas::g_ck[r][cmx_meas::CK_VALTEL] += __rdtsc() - t0;
    ++cmx_meas::g_ckn[r][cmx_meas::CK_VALTEL];
  }
}

// CMXTEL val lines: per region and group the sampled bits, the share with a live input, the mean |dz|, the occlusion
// bits scaled by k (an estimate for every cascade bit of the region), the same with the fxcm-internal path, the input
// rms and the mean sum_i |W0_i[j] x_j|; then per bank mixer the mean |gamma| and its output's occlusion bits.
__attribute__((noinline)) COLD_SIZE_ATTR void Predictor::CmxValReport(const char* tag) {
  const double k = (double)cmx_meas::g_valtel;
  for (int r = 0; r < 4; ++r) {
    if (!(g_vbase[r][0] > 0.0)) continue;
    fprintf(stderr, "CMXTEL valtel %s region=%d sampled=%.0f k=%d code_bits_scaled=%.1f groups=%zu\n", tag, r,
            g_vbase[r][0], cmx_meas::g_valtel, g_vbase[r][1] * k, g_vg.size());
    for (size_t gi = 0; gi < g_vg.size(); ++gi) {
      const CmxVG& G = g_vg[gi];
      const double* a = g_vacc[r].data() + gi * kVaK;
      if (!(a[kVaN] > 0.0)) continue;
      unsigned width = 0;
      for (size_t q = 0; q < G.clo.size(); ++q) width += G.chi[q] - G.clo[q];
      fprintf(stderr, "CMXTEL val %s region=%d group=%s n=%.0f live=%.4f mabs=%.5f occl_bits=%.1f occl_fx_bits=%.1f "
              "x_rms=%.4f wx=%.5f ex=%u cm=", tag, r, G.name.c_str(), a[kVaN], a[kVaLive] / a[kVaN],
              a[kVaMabs] / a[kVaN], a[kVaOcc] * k, a[kVaOccFx] * k,
              width ? sqrt(a[kVaX2] / (a[kVaN] * (double)width)) : 0.0, a[kVaWx] / a[kVaN], G.ex);
      for (size_t q = 0; q < G.clo.size(); ++q) fprintf(stderr, "%s%u-%u", q ? "+" : "", G.clo[q], G.chi[q]);
      fprintf(stderr, " fx=");
      if (G.flo.empty()) fprintf(stderr, "-");
      for (size_t q = 0; q < G.flo.size(); ++q) fprintf(stderr, "%s%u-%u", q ? "+" : "", G.flo[q], G.fhi[q]);
      fprintf(stderr, "\n");
    }
    for (unsigned i = 0; i < g_vn && i < mixer_chain::kMaxMixers; ++i) {
      if (!(g_vmix[r][i][2] > 0.0)) continue;
      fprintf(stderr, "CMXTEL valmix %s region=%d mixer=%u gamma_abs=%.5f occl_bits=%.1f\n", tag, r, i,
              g_vmix[r][i][0] / g_vmix[r][i][2], g_vmix[r][i][1] * k);
    }
  }
  fflush(stderr);
}
#endif
