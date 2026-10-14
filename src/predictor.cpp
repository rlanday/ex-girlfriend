#include "predictor.h"
#include "attributes.h"
#include "invariants.h"
#include "lstm_sharpen.h"
#include "mixer_selectors.h"
#include "mixer_weights.h"
#include "logit_table.h"
#include "mixer_bank.h"
#include "mixer_lr.h"
#include "lstm_fx_link.h"
#include "char_rate.h"
#include "ppm_logit_prior.h"
#include "vocab_frozen.h"
#include <cmath>

#include "transformer_gate_ctx.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>

 Predictor::Predictor(const std::vector<bool>& vocab) : manager_(),
    sigmoid_(100001), vocab_(vocab) {
  wordlbl_ = word_label::Create();
  char_rate::Init(vocab);
  fxcm_neutral_input_ = sigmoid_.LogitBase(0.5f);
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
  auxiliary_size_ = 2;
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
  num += byte_mixer_->NumOutputs();
  num += tf1_slot_->NumOutputs();
  num += 1;
#if PORT_GEOM
  num += 1;   // col 588: FW-DUAL row (0 when not wired)
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
 void Predictor::AddPPMD() {
  byte_model_.emplace(25, 14000, manager_.bit_context_, vocab_);
}

 void Predictor::AddWord() {
  float delta = 200;
  std::vector<std::vector<unsigned int>> model_params = {
  {0},
   {0, 1},
      {1},
      {1, 2},
      {1, 3},
       {2, 3},
       {3, 4},
      {1, 2, 4},
      {2, 3, 4},
       {2}
      };
  for (const auto& params : model_params) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
        manager_.bit_context_, delta, manager_.shared_map_);
  }

  std::vector<std::vector<unsigned int>> model_params2 = {
  {0},
  {1},
      {1, 3},
       {1, 2, 3},
       {7, 2}};
  for (const auto& params : model_params2) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, 200, 0.5, 2000000, &(manager_.longest_match_));
    if (params[0] == 1 && params.size() == 1) {
      indirect_r_models_.emplace_back(manager_.run_map_, context.GetContext(),
          manager_.bit_context_, delta, manager_.shared_map_);
    }
  }
}

 void Predictor::AddMatch() {
  float delta = 0.5;
  int limit = 200;
  unsigned long long max_size = 2000000;
  std::vector<std::vector<int>> model_params = {
  {0, 8},
  {1, 8},
  {7, 4},
      {11, 3},
      {13, 2},
  };

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
#if PORT_RESTORE_IND
  // W2: the two double-indirect models neif removed, at their original place
  // (fx2-cmix-transformer AddDoubleIndirect); they consume the two rand()
  // calls the placeholder below kept for them.
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind2,  manager_.bit_context_, delta, manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind3,  manager_.bit_context_, delta, manager_.shared_map_);
#else
  (void)rand(); (void)rand();
#endif
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind5,  manager_.bit_context_, delta, manager_.shared_map_);
}

unsigned int Discretize(float p) {
  return 1 + 4094 * p;
}
 void Predictor::AddMixers() {
  constexpr unsigned int vocab_size = 208;
  lstm_mixer_inputs::st.Geometry(vocab_size + 1u + (unsigned int)80);
  byte_mixer_.emplace(1, manager_.bit_context_, vocab_, vocab_size,
      new Lstm(vocab_size
                + lstm_mixer_inputs::st.slots
                , vocab_size, 80,
                1, 128,
                0.03, 10));
  if (transformer::Running()) {
    tf1_engine_.reset(
        new transformer::Tf1Engine(vocab_, transformer::WeightsPath()));
    bitl_hid_ = tf1_engine_->LastHidden();
    bitl_n_hid_ = bitl_hid_ ? tf1_engine_->HiddenSize() : 0;
    bitl_n_tf_ = 4;
  }
  if (!transformer::Active()) {
    static const float kKeinTfZustand[192] = {0.0f};
    bitl_hid_ = kKeinTfZustand;
    bitl_n_hid_ = 192;
  } else if (bitl_n_hid_ != 192) {
    std::fprintf(stderr,
        "\ncorrection model: compiled transformer state width %d, transformer reports %d\n",
        (int)192, bitl_n_hid_);
    std::fflush(stderr);
    std::abort();
  }
  bitl_lhid_ = byte_mixer_->LstmHidden(&bitl_n_lhid_);
  tf1_slot_.emplace(vocab_);
#if PORT_FW
  fw_slot_.emplace(vocab_);
#endif

  // The gate zone of the byte about to be coded.  A POINTER: the
  // automaton writes it once per byte (on already decoded tokens only), the
  // head reads it live, and encoder and decoder read the same value.
  bitl_zone_ = &tf_gate::ctx.zone;

  for (int i = 0; i < 2; ++i) {
    layers_.emplace_back(sigmoid_,
        1.0e-4);
  }

  unsigned long long input_size = GetNumModels();
  layers_[0].SetNumModels(input_size);

  AddMixer(0, manager_.mx9, 0.005);
  AddMixer(0, manager_.mx10, 0.0005);
  AddMixer(0, manager_.mx11, 0.005);
  AddMixer(0, manager_.mx12, 0.0005);
  AddMixer(0, manager_.mx13, 0.005);
  AddMixer(0, manager_.mxx, 0.001);
  AddMixer(0, manager_.recent_bytes_[2], 0.002);
  AddMixer(0, manager_.line_break_, 0.0007);
  AddMixer(0, manager_.longest_match_, 0.0005);
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
#if PORT_GEOM
  ++input_size;   // col 27: FW1 (0 when not wired); BITLSTM still reads cols 0..26
#endif
  layers_[1].SetNumModels(input_size);

  // Final mixer with the transformer's gate zone as its context: zone 0 (not
  // gated) selects the same weight row as a context-free mixer, every gated
  // zone gets a row of its own.  This costs rows in the hash map, no
  // geometry.
  AddMixer(1, transformer::Active() ? manager_.tf1_gate_ctx1
                                 : manager_.zero_context_, 0.0003);

  layers_[0].SetExtraInputSize(mixer_0_.size());

  mixer_1_[0].E2ESeedHead(
      (unsigned int)mixer_0_.size(),
      (float)(mixer_chain::kSeed) * 0.001f / (float)mixer_0_.size());

}
int lstmpr=0, lstmex=0;

namespace dictreload {
unsigned int mtc_len = 0;
int ord_x = 0, ord_w = 0;
}
namespace lstm_sharpen { State state; }
float byte_mixer_output=0.0f;

float tf1_slot_output=0.0f;
#if PORT_FW
float fw_slot_output=0.0f;
#endif
 float Predictor::Predict() {
  fxcm_model_.PredictHalf();
  manager_.UpdateLx1();
  manager_.UpdateTf1Gate();   // gate zone as mixer context
  unsigned int input_index = 0;
  auto bracket_model_output = bracket_model_->Predict()[0];
  layers_[0].SetInput(input_index++, bracket_model_output);

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
      fprintf(stderr, "id model: fxcm uses %u of %u columns, %u free ones needed at the end\n",
              fxcm_active_outputs, fxcm_model_outputs, kNeed);
      fflush(stderr);
      exit(3);
    }
    const unsigned int c0_sh = manager_.bit_context_;
    const int bp_sh = 31 - __builtin_clz(c0_sh);
    layers_[0].SetStretchedInputUnchecked(fxcm_model_index - 1,
        fxcm_neutral_input_ + id_exclude::MixDelta(c0_sh, bp_sh));
  }

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
  layers_[0].SetInput(input_index++, byte_model_->Predict()[0]);

  float byte_mixer_override = -1;

  if (byte_mixer_output == 0 || byte_mixer_output == 1) byte_mixer_override = byte_mixer_output;
  layers_[0].SetInput(input_index++, byte_mixer_output);
  auto byte_mixer_index = input_index - 1;
  layers_[0].SetInput(input_index++, tf1_slot_output);
  auto tf1_slot_index = input_index - 1;

  layers_[0].SetStretchedInput(input_index, word_label::Forward(wordlbl_));
  ++input_index;
  const float aux_aux_in = aux_fxcm_input;
  float auxiliary_average = Sigmoid::Logistic(aux_aux_in)
      + Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index])
      ;
  auxiliary_average /= auxiliary_size_;
  manager_.auxiliary_context_ =auxiliary_average * 15;
  {
    const float reload_d = Sigmoid::Logistic(aux_aux_in)
                        - Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]);
    manager_.reload_aux = (unsigned long long)dictreload::Q8diff(reload_d) * 256
                       + manager_.long_bit_context_;
  }

  {
    adepth_.Gather(&layers_[0].Inputs()[0], adepth_lo, input_index,
                   layers_[0].Inputs()[0], aux_fxcm_input);
    const float adepth_z = adaptive_depth::Gate(adepth_);
    const float adepth_az = adepth_z < 0.0f ? -adepth_z : adepth_z;
    adepth_.skip = (adepth_az > adaptive_depth::kTau) &&
                   (adepth_.coded >= (unsigned long long)0) &&
                   !adaptive_depth::MetaOff();
  }
#if PORT_GEOM
  // Col 588, appended AFTER the adaptive-depth gather: the AD feature set and
  // its skip decisions (hence BITLSTM's bit set and position counter) do not
  // see it.
  const unsigned int fw_index = input_index++;
#if PORT_FW
  layers_[0].SetInput(fw_index, fw_slot_output);
#else
  layers_[0].SetZero(fw_index);
#endif
#endif
  const bool chr_skip_mix =
      (byte_mixer_override >= 0) && char_rate::Forced(manager_.bit_context_);
  float p;
  if (chr_skip_mix) {
    p = 0.5f;
#if PORT_CASCADE
  // W1: no cascade skip.  adepth_.skip (computed above, unchanged) only gates
  // BITLSTM in the coder; the cascade, SSE and the final rules run on every bit.
  } else {
#else
  } else if (adepth_.skip) {
    p = adepth_.p;
  } else {
#endif
  for (unsigned int i = 0; i < mixer_0_.size(); ++i) mixer_0_[i].Prime();
  mixer_1_[0].Prime();
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
  layers_[1].SetStretchedInput(mixer_0_.size(), aux_aux_in);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1, layers_[0].Inputs()[byte_mixer_index]);
  layers_[1].SetStretchedInput(mixer_0_.size() + 2, layers_[0].Inputs()[tf1_slot_index]);
#if PORT_GEOM
#if PORT_FW
  // FW1: the FW row's stretched input also feeds the final mixer.
  layers_[1].SetStretchedInput(mixer_0_.size() + 3, layers_[0].Inputs()[fw_index]);
#else
  layers_[1].SetZero(mixer_0_.size() + 3);
#endif
#endif

  const float m1raw = mixer_1_[0].Mix();
  bitl_stage1_ = &layers_[1].Inputs()[0];
  bitl_n_stage1_ = (int)(mixer_0_.size() + auxiliary_size_);
  ++bitl_n_stage1_;
  {
    const int lmix_n = bitl_n_stage1_;
    for (int j = 0; j < lmix_n; ++j) bitl_s1buf_[j] = bitl_stage1_[j];
    float lmix_p = byte_mixer_->Predict()[0];
    if (lmix_p < 1.0e-4f) lmix_p = 1.0e-4f;
    else if (lmix_p > 1.0f - 1.0e-4f) lmix_p = 1.0f - 1.0e-4f;
    bitl_s1buf_[lmix_n] = sigmoid_.Logit(lmix_p);
    bitl_stage1_ = bitl_s1buf_;
    bitl_n_stage1_ = lmix_n + 1;
  }
  bitl_m1raw_ = m1raw;
  p = Sigmoid::Logistic(m1raw);
  p = sse_.Predict(p);
  {
    const float kEpsIsbn = 1.0f / 128.0f;
    if (invariants::g_isbn_rule) {
      const float r = (invariants::g_isbn_rule == 2) ? 1.0f : 0.0f;
      p = (1.0f - kEpsIsbn) * r + kEpsIsbn * p;
    }
    if (p < 1.0f / 65535.0f) p = 1.0f / 65535.0f;
    if (p > 65534.0f / 65535.0f) p = 65534.0f / 65535.0f;
  }
  {
    const unsigned int c0_hd = manager_.bit_context_;
    p = id_exclude::Hard(p, c0_hd, 31 - __builtin_clz(c0_hd));
  }
  }
  if (byte_mixer_override >= 0) p = byte_mixer_override;
  bitl_override_ = (byte_mixer_override >= 0) ? 1 : 0;
  last_p16_ = (unsigned int)(1 + 65534 * p);
  return p;
}

 void Predictor::MetalrBlockEnde() {
  for (auto& m : mixer_0_) m.MetalrSteps0();
  for (auto& m : mixer_1_) m.MetalrSteps0();
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
  const float g_final = Sigmoid::Logistic(mixer_1_[0].E2EP()) - (float)bit;
  float gamma[mixer_chain::kMaxMixers];
  const mixw_t* xw[mixer_chain::kMaxMixers];
  for (unsigned int k = 0; k < n; ++k) xw[k] = mixer_0_[k].E2EXRow();
  const mixw_t* const w1 = mixer_1_[0].E2EWRow();
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
  float scale = 1.0f;
  for (unsigned int j = 0; j < n; ++j)
    mixer_0_[j].E2ESetErr(g_final * gamma[j] * scale);
}

 void Predictor::Perceive(int bit) {
  word_label::Perceive(wordlbl_, bit);
  pagectx::surprise.AddBit(last_p16_, bit, manager_.bit_context_ >= 128);
  lstm_mixer_inputs::st.AddBit(bit, last_p16_);
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
  byte_model_->Perceive(bit);
  byte_mixer_->Perceive(bit);
  tf1_slot_->Perceive(bit);
#if PORT_FW
  fw_slot_->Perceive(bit);
#endif

  if (char_rate::Forced(manager_.bit_context_)) {
    sse_.PerceiveContextOnly(bit);
    for (auto& mixer: mixer_0_) mixer.PerceiveSkip();
    for (auto& mixer: mixer_1_) mixer.PerceiveSkip();
#if !PORT_CASCADE
  } else if (adepth_.skip) {
    sse_.PerceiveContextOnly(bit);
    for (auto& mixer: mixer_0_) mixer.PerceiveSkip();
    for (auto& mixer: mixer_1_) mixer.PerceiveSkip();
#endif
  } else {
  E2EGrad(bit);
  PerceiveBank(&mixer_0_[0], mixer_0_.size(), bit);
  for (auto& mixer: mixer_1_) {
    mixer.Perceive(bit);
  }

  sse_.Perceive(bit);
  }
  if (!char_rate::Forced(manager_.bit_context_)) {
    adaptive_depth::TrainStep(adepth_, bit);
    ++adepth_.coded;
  }

  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;

  if (byte_update)
    id_exclude::Byte(((manager_.bit_context_ << 1) | (unsigned int)bit) & 0xFFu);
  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    {
      unsigned int best = 0, bl = 0;
      for (unsigned int i = 0; i < match_models_.size(); ++i) {
        const unsigned int l = match_models_[i].MatchLenSlot();
        if (l > bl) { bl = l; best = i; }
      }
      lstm_match_channel::st.Set(bl, match_models_[best].MatchByteSlot());
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }

    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
    grammar_match_->ByteUpdate();
    byte_model_->ByteUpdate();
    if (tf1_prior_writer_) {
      const std::valarray<float>& tf1_pp = byte_model_->BytePredict();
      for (int i = 0; i < transformer::kSetSize; ++i) {
        tf1_prior_row_[i] = tf1_pp[tf1_vocab_bytes_[i]];
      }
      tf1_prior_writer_->Write(tf1_prior_row_.data(), transformer::kSetSize);
    }

    lstm_mixer_inputs::st.ByteEnd();
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
    if (tf1_row && transformer::Active()) {
      if (tf1_engine_->GateMasked()
          || alias_prompt::Laeuft()   // alias rows are being fed
          || timestamp_skip::Laeuft()
         ) {
        tf1_slot_->SetProbsNeutral();
#if PORT_FW
        fw_slot_->SetProbsNeutral();
        tf1_engine_->FwNeutral();
#endif
      } else
#if PORT_FW
      {
      tf1_slot_->SetProbs(tf1_row);
      fw_slot_->SetProbs(tf1_engine_->FwRow());
      }
#else
      tf1_slot_->SetProbs(tf1_row);
#endif
    } else {
      // no transformer: abstention row, arithmetically no slot at all.
      tf1_slot_->SetProbsOff();
#if PORT_FW
      fw_slot_->SetProbsOff();
      if (tf1_engine_) tf1_engine_->FwNeutral();
#endif
    }
    {
    const std::valarray<float>& p = byte_model_->BytePredict();
    for (unsigned int j = 0; j < 256; ++j) {
      byte_mixer_->SetInput(j,p[j]);
    }
    {
      float obias_row[256];
      ppm_logit_prior::Row(&p[0], obias_row);
      byte_mixer_->SetOutputBias(obias_row);
    }
    byte_mixer_->ByteUpdate();
    }

  }
  byte_mixer_output = byte_mixer_->Predict()[0];
  lstmpr=Discretize(byte_mixer_output);
  lstmex=byte_mixer_->ex;
  lstm_fx_link::Push(lstmpr, lstmex);
  tf1_slot_output = tf1_slot_->Predict()[0];
#if PORT_FW
  fw_slot_output = fw_slot_->Predict()[0];
#endif
  lstm_sharpen::state.full_alive =
      (lstm_sharpen::state.full_ex >= byte_mixer_->Lx1Bot() &&
       lstm_sharpen::state.full_ex <= byte_mixer_->Lx1Top()) ? 1 : 0;
  fxcm_model_.Perceive(bit);
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
    {
      unsigned int best = 0, bl = 0;
      for (unsigned int i = 0; i < match_models_.size(); ++i) {
        const unsigned int l = match_models_[i].MatchLenSlot();
        if (l > bl) { bl = l; best = i; }
      }
      lstm_match_channel::st.Set(bl, match_models_[best].MatchByteSlot());
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
