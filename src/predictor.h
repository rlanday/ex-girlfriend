#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "mixer/sigmoid.h"
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
#include "transformer.h"
#include "tf1/tf1_engine.h"
#include "char_rate.h"
#include "run_mode.h"
#include "bit_head.h"
#include "adaptive_depth.h"
#include "semantic_cache.h"
#include "word_label.h"
#include "mixer_precond.h"

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
  const unsigned int* bitl_zone_ = nullptr;
  const float* bitl_stage1_ = nullptr;   // layers_[1].Inputs(); this hand-over follows bitlstm32 of cmix-obias (David Freelan, GPL-3)
  int bitl_n_stage1_ = 0;
  float bitl_m1raw_ = 0.0f;
  float bb_z_ = 0.0f, bb_ppre_ = 0.5f;
  unsigned int BbBitCtx() const { return manager_.bit_context_; }
  int bitl_override_ = 0;
  bool AdepthSkipped() const { return adepth_.skip; }
  int Tf1ZwangByte() const { return tf1_engine_ ? tf1_engine_->ZwangByte() : -1; }
  void WordlblBlockEnde() { wordlbl_ = word_label::Neu(wordlbl_); }
  void MetalrBlockEnde();
  void PpmdWaermen();
  void SemByteStart(int pb);
  uint32_t SemCorrect(uint32_t p32);
  void SemLearn(int bit);
  // the semantic-cache columns of the BITLSTM row (Predict(), every bit)
  void SemAppend();
#if PORT_CMX_MEAS
  // research CMX_MIXROW / CMX_MIXTEL (src/cmx_meas.h)
  void CmxMixRekey();
  void CmxMixReport(const char* tag);
  void CmxMixTel();
  void CmxSemReport(const char* tag);
  void CmxSemKeyReset();
  // research CMX_ADEPTH_MODE (src/cmx_meas.h): AdepthLearnOff() = the cascade's learners and the BITLSTM stand still on
  // this bit (equals AdepthSkipped() with the knob unset); AdmRun() = mode 2 gate bit (the cascade ran, the coder takes
  // sigma(z)); AdmCascadeP() = the cascade's p of that bit (the BITLSTM's input)
  bool AdepthLearnOff() const { return adm_loff_; }
  bool AdmRun() const { return adm_run_; }
  float AdmCascadeP() const { return adm_pcas_; }
#endif

 private:
#if PORT_CMX_MEAS
  bool adm_loff_ = false, adm_run_ = false;
  float adm_pcas_ = 0.5f;
  // research CMX_SEMTOP: share class of this byte's vote; stages 1 and 3 of the end stage in wide tables (w, AdaGrad g)
  int cmx_stq_ = 0;
  double *cmx_st_w4_ = nullptr, *cmx_st_g4_ = nullptr, *cmx_st_w6_ = nullptr, *cmx_st_g6_ = nullptr;
  void CmxSemTopInit();
  void CmxSemTopCells();
  // the end stage's tail and learning step under CMX_SEMTOP and/or CMX_SEMIN
  uint32_t CmxSemTail(int k, int info);
  void CmxSemLearn(int bit);
  // research CMX_SEMIN: the vote's top share (byte start), two extra weights (and AdaGrad sums) per end-stage cell
  // (rows: stage 1 at 0, stage 2 at cmx_si_o2_, stage 3 at cmx_si_o3_), the two inputs of this bit, and the first of
  // the two final-mixer columns (CMX_SEMIN=2)
  double cmx_tsh_ = 0.0;
  double *cmx_si_w_ = nullptr, *cmx_si_g_ = nullptr;
  size_t cmx_si_o2_ = 0, cmx_si_o3_ = 0;
  double cmx_si_x_[2] = {0.0, 0.0};
  unsigned int cmx_si_col_ = 0;
  void CmxSemInInit();
  void CmxSemInCols();
  void CmxSemInSelfTest();
 public:
  void CmxSemInReport(const char* tag);
 private:
  // research CMX_SEMKEYS / CMX_SEMKEY2: the extra semantic caches sk = 0..n-1 (cache sk + 1; States on the heap from
  // AddMixers, tables at their first byte start; 3 = cmx_meas::kSkMax), their final-mixer columns, and the top-1 bytes
  // of the last byte of the release cache and of each extra cache (telemetry, judged at the next)
  semantic_cache::State* cmx_sk_[3] = {nullptr, nullptr, nullptr};
  unsigned int cmx_sk_col_[3] = {0, 0, 0};
  int cmx_sk_h1_ = -1, cmx_sk_h2_[3] = {-1, -1, -1};
  void CmxSem2Init(int sk);
  void CmxSem2Byte(int sk, int strom_pb);
  void CmxSem2UtByte(int sk, int topb);
  void CmxSem2PredBit(int sk);
  uint32_t CmxSem2UtCorrect(int sk, uint32_t p32);
  void CmxSem2UtLearn(int sk, int bit);
  uint32_t CmxSem2Correct(uint32_t p32);
 public:
  void CmxSem2ByteStart(int strom_pb);
  void CmxSem2Report(const char* tag);
 private:
  // research CMX_FWOFF / CMX_VALTEL (src/cmx_meas.h, CM census phase 2a): the fast-weight slot on the transformer
  // row; the frozen-weight occlusion telemetry (E2EGrad), its groups and its report
  void CmxFwOffRow(const float* row);
  void CmxValTel(int bit, const float* gamma, unsigned int n);
  bool CmxValBuild(unsigned int n);
 public:
  void CmxValReport(const char* tag);
 private:
#endif
  void SemInit();
  void SemUtByte(int topb);
  uint32_t SemUtCorrect(uint32_t p32);
  void SemUtLearn(int bit);
  // the cache's readout of the bit about to be predicted (Predict(), before
  // the layer-0 row), its layer-0 input and its end-mixer column
  void SemPredBit();
  void SemSetInput(unsigned int i);
  void SemSetStage1();
  unsigned int sem_index_ = 0;
  void TreffByteStart(int last);
  uint32_t TreffCorrect(uint32_t p32);
  void TreffLearn(int bit);
  semantic_cache::State sem_;
  cache_hit::State treff_;
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
  // fxcm is the one "auxiliary" input (the byte LSTM was the second).
  size_t auxiliary_size_ = 1; // 0 -> fxcm
  SSE sse_;
  llvm::SmallVector<MixerInput,2> layers_;
  // Inline capacities cover every line (emplace_back stops the program
  // instead of growing): 23 layer-0 mixers, 2 final mixers (the second one
  // holds the add rows).
  llvm::SmallVector<Mixer, 23> mixer_0_;
  llvm::SmallVector<Mixer, 2> mixer_1_;
  std::vector<unsigned int> auxiliary_;
  ContextManager manager_;
  Sigmoid sigmoid_;
  std::array<float, 4096> fxcm_stretched_inputs_;
  float fxcm_neutral_input_ = 0.0f;
  std::array<float, 4096> aux_pr_stretched_;
  std::optional<PPMD::PPMD> byte_model_;
  std::optional<Tf1Slot> tf1_slot_;
  // selector of the lx1_agree mixer from the transformer row (predictor.cpp)
  void Lx1TfConf();
  // The slot of the transformer's second (fast-weight) head and
  // its steps (predictor.cpp, noinline: no new block in Predict/Perceive).
  std::optional<Tf1Slot> fw_slot_;
  unsigned int fw_index_ = 0;   // its layer-0 input
  void FwSetInput(unsigned int i);
  void FwSetStage1();
  void FwPerceive(int bit);
  void FwSetNeutral();
  void FwSetRow();
  void FwSetOff();
  void FwPredictBit();
  // the add rows of the final mixer; every step is a noinline call in
  // predictor.cpp
  void KtxSelect();
  void KtxPrime();
  float KtxAddMix();
  void KtxW1Sum(float* w1s, unsigned int n);
  void MlPath(bool byte_update);
  // the Bayes term of the final mixer
  float MxsBayMix(float zrow);
  unsigned int mxs_ppmd_ix_ = 0;   // layer-0 index of the PPMd input
  // the final-mixer output z of this bit (all rows plus the Bayes term); every
  // final-mixer row learns on the error of z
  float end_z_ = 0.0f;
  void EndLearn(int bit);
  float sem_s1buf_[128] = {0.0f};   // the row plus the semantic-cache columns
  // the 13 columns of the BITLSTM row behind the stage-1 row (FW logit, true
  // bit position, transformer path state of MlPath); predictor.cpp, noinline.
  void MlAppend();
  float bitl_s1buf_[128] = {0.0f};   // the row plus these columns
  mixer_precond::L0Col mxp_l0col_;   // layer-0 column powers

  word_label::State* wordlbl_ = nullptr;
  // A reference: the caller (RunCompression/RunDecompression) keeps vocab
  // alive longer than the Predictor, so no copy of it is made.
  const std::vector<bool>& vocab_;
   FXCM fxcm_model_;

  friend unsigned long long PredictorLayoutTagFast();
  unsigned int last_p16_ = 32768;
};

unsigned long long PredictorLayoutTagFast();

#endif
