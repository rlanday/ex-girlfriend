#ifndef TF1_ENGINE_H_
#define TF1_ENGINE_H_

// Coder-side driver of the transformer inference (src/tf1/opt = Ivanov's
// TransformerOpt, src/tf1/model.* = his fp32 reference path).  Everything
// cmix-side lives here so that predictor.cpp keeps a two-line hook.
//
// The float16 helpers below are taken verbatim from
// fx2-cmix-transformer/src/predictor.cpp (Vladimir Ivanov, GPL-3), INCLUDING
// the documented subnormal quirk of the scalar decode path: it is part of the
// probability-rounding contract between the training pipeline and the coder,
// and changing it would change the coded probabilities.
//
// Also after his predictor.cpp: Fail(), HalfFileWriter, the loss report
// (ReportLoss) and the per-byte step of Tf1Engine::Step around the article
// separator (separator window, cut into pieces, PPMd distribution on the
// first token of a piece, float16 rounding and the 1e-6 floor of each row).

#include "../transformer.h"
#include "../port_cfg.h"

#include <stdint.h>
#include <stdio.h>

#include <string>
#include <valarray>
#include <vector>

namespace transformer {

[[noreturn]] void Fail(const char* fmt, ...);

// ---- float16 (contract with the training pipeline) ----
uint16_t FloatToHalf(float f);
float HalfToFloat(uint16_t h);
void FloatsToHalves(const float* src, uint16_t* dst, size_t n);
void HalvesToFloats(const uint16_t* src, float* dst, size_t n);

// buffered float16 file sink (used only by the training-data dumps)
class HalfFileWriter {
 public:
  explicit HalfFileWriter(const std::string& path);
  ~HalfFileWriter();
  void Write(const float* values, size_t n);
  void WriteHalves(const uint16_t* values, size_t n);
  void Flush();

 private:
  std::string path_;
  FILE* file_ = nullptr;
  std::vector<uint16_t> buffer_;
  size_t used_ = 0;
};

// Checks the runtime vocabulary against the frozen 208-byte set element by
// element (not by size) and fails if the stream carries a byte outside it.
void CheckVocabSubset(const std::vector<bool>& vocab);

// The dictionary-encoded article separator in token index space, stamped from
// this entry's own dictionary (the upstream constant belongs to a different
// dictionary and would be a different byte sequence).
const unsigned char* SeparatorIndices();
int SeparatorLength();

// ---- the engine ----
class Tf1Engine {
 public:
  // vocab: the runtime vocabulary of the stream (checked against the set)
  Tf1Engine(const std::vector<bool>& vocab, const char* weights_path);
  ~Tf1Engine();

  // Called once per completed byte, in Predictor::Perceive's byte-update
  // block.  `ppmd` is the PPMd byte distribution over 256 values, `byte` the
  // byte that just completed.  Returns V floats in set order: the
  // distribution over the NEXT byte, >= 1e-6 guarded, ready for
  // ByteMixer::SetProbs().
  const float* Step(const std::valarray<float>& ppmd, unsigned int byte);

  // Read-only access to the
  // transformer's own per-BYTE state, for the BITL head.  `LastHidden` is the
  // d_model vector the output head reads (nullptr on the reference path,
  // which has no accessor).  Valid until the next Step().
  const float* LastHidden() const;
  int HiddenSize() const;

  // loss report; the coder accumulates no loss (see (1) in Step())
  void ReportLoss() const;

  // Is the row that Step() just returned masked, i.e. is the byte about to be
  // coded in a gated zone?  The compressor asks this instead of looking at the
  // row, so that "masked" stays a decision of the automaton, never a property
  // of the numbers.
  bool GateMasked() const;

  void SetMatch(int hot_byte, int raw_len_bits);

  // End-of-run check: stops if a document exceeded the rotary-table length.
  void ReportChannels() const;

#if PORT_FW
  // FW-DUAL row for the slot fw_slot_, to be called right after Step() when
  // the transformer row goes to the slot: the FW-head distribution of the
  // same read row the Step() row came from (float16-rounded and 1e-6-floored
  // like that row), or a copy of the Step() row when it came from the PPMd
  // prior (piece start, title/sentinel prior).  Marks the next stream token
  // as a target of the FW head iff the row came from the FW head.
  const float* FwRow();
  // The slot is neutral or off for the next stream token: no FW target.
  void FwNeutral();
#endif

 private:
  struct Impl;
  Impl* impl_;
};

}

#endif
