#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "mixer/sigmoid.h"
#include "lstm_match_channel.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/byte-mixer.h"
#include "mixer/sse.h"
#include "models/model.h"
#include "models/byte-model.h"
#include "context-manager.h"
#include "models/direct.h"
#include "models/direct-hash.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/grammar-match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
#include "mixer/lstm.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/sparse.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/bit-context.h"
#include "contexts/combined-context.h"

#include "id_exclude.h"
#include "page_contexts.h"
#include "fast_math.h"
#include "transformer.h"
#include "tf1/tf1_engine.h"
#include "char_rate.h"
#include "run_mode.h"
#include "bit_head.h"
#include "adaptive_depth.h"
#include "word_label.h"
#include "lstm_mixer_inputs.h"
#include "port_cfg.h"

#include "ds/SmallVector.h"
#include "ds/emhash_set.hpp"

#include <vector>
#include <set>
#include <memory>
#include <optional>
#include <array>

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab);
  float Predict();
  void Perceive(int bit);
  void Pretrain(int bit);
  int ChrForced() const { return char_rate::Forced(manager_.bit_context_); }
  void FreeFxcmMemory();
  void FreeFxcmRest();
  void ObserveKnownBit(int bit);
  void ObserveTfOnlyByte(unsigned int byte);

  const float* bitl_hid_ = nullptr;
  int bitl_n_hid_ = 0;
  float bitl_tf_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int bitl_n_tf_ = 0;
  const float* bitl_lhid_ = nullptr;
  int bitl_n_lhid_ = 0;
  const unsigned int* bitl_zone_ = nullptr;
  const float* bitl_stage1_ = nullptr;   // layers_[1].Inputs(); this hand-over follows bitlstm32 of cmix-obias (David Freelan, GPL-3)
  int bitl_n_stage1_ = 0;
  float bitl_m1raw_ = 0.0f;              // raw mixer_1_[0].Mix()
  int bitl_override_ = 0;
  bool AdepthSkipped() const { return adepth_.skip; }
  void WordlblBlockEnde() {
    word_label::Destroy(wordlbl_);
    wordlbl_ = word_label::Create();
  }
  void MetalrBlockEnde();

 private:
  adaptive_depth::State adepth_;
  unsigned long long GetNumModels();
  void AddMixer(int layer, const unsigned long long& context,
      float learning_rate);
  void AddAuxiliary();
  void AddPPMD();
  void AddBracket();
  void AddWord();
  void AddDirect();
  void AddMatch();
  void AddGrammarMatch();
  void AddDoubleIndirect();
  void AddMixers();
  void E2EGrad(int bit);

  std::unique_ptr<transformer::Tf1Engine> tf1_engine_;
  std::unique_ptr<transformer::HalfFileWriter> tf1_prior_writer_;
  std::vector<float> tf1_prior_row_;
  std::vector<int> tf1_vocab_bytes_;
  llvm::SmallVector<Indirect<Nonstationary>, 32> indirect_ns_models_; // non-stationary
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_; // run map
  llvm::SmallVector<Direct, 4> direct_models_;
  llvm::SmallVector<Match, 10> match_models_;
  unsigned int tf1_match_base_ = 0;
  unsigned int tf1_match_count_ = 0;
  std::optional<GrammarMatch> grammar_match_;

  std::optional<Bracket> bracket_model_;
  size_t auxiliary_size_ = 2; // 0 -> fxcm, 1 -> byte_mixer
  SSE sse_;
  llvm::SmallVector<MixerInput,2> layers_;
  llvm::SmallVector<Mixer, 24> mixer_0_;
  llvm::SmallVector<Mixer, 1> mixer_1_;
  std::vector<unsigned int> auxiliary_;
  ContextManager manager_;
  Sigmoid sigmoid_;
  std::array<float, 4096> fxcm_stretched_inputs_;
  float fxcm_neutral_input_ = 0.0f;
  std::array<float, 4096> aux_pr_stretched_;
  std::optional<PPMD::PPMD> byte_model_;
  std::optional<ByteMixer> byte_mixer_;
  float bitl_s1buf_[128] = {0.0f};
  std::optional<Tf1Slot> tf1_slot_;
#if PORT_FW
  std::optional<Tf1Slot> fw_slot_;   // FW-DUAL row of the transformer
#endif
  word_label::State* wordlbl_ = nullptr;
  std::vector<bool> vocab_;
   FXCM fxcm_model_;

  friend unsigned long long PredictorLayoutTagFast();
  unsigned int last_p16_ = 32768;
};

unsigned long long PredictorLayoutTagFast();

#endif
