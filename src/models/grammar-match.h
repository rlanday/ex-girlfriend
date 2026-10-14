#ifndef GRAMMAR_MATCH_H
#define GRAMMAR_MATCH_H

// ---------------------------------------------------------------------------
// Line-by-line port from fx-deepmix (Halvor Yttredal), GPL-3.0.  Changes
// against the original: title capture follows this entry's page form (it
// starts at the page separator and ends at the newline after the title); the
// ECHO arm threshold defaults to the original 2, and the timestamp family and
// the stderr census of the original are off by default.
//
// GrammarMatch is a second match channel.  match.cpp predicts continuations of
// previously seen bytes found by HASHING; GrammarMatch predicts continuations
// found by PARSING the post-WRT modeled stream.  It adds no alphabet symbol
// and performs no stream transformation, so it costs no new symbol in the
// coder stream.  Families:
//
//   PIPED - piped links image as '[' <target> 'Q' <label> ']' post-WRT
//           ('|' -> 'Q' via the encode_text remap, verified on our stream).
//           When the label's byte image literally extends the target's image
//           (cap flag 0x40 stripped), the label bytes are derivable; when the
//           first label byte diverges (e.g. codeword lead), the channel goes
//           idle and the learned confidence prices the rest.
//   ECHO  - in-article recurrences of the page title, whose post-WRT image is
//           byte-identical at every recurrence.  Title anchors in fx-deepmix:
//             <title>  ->  'L' 0xDF 0x9B 'N'      (0x4C 0xDF 0x9B 0x4E)
//             </title> ->  'L' '/' 0xDF 0x9B 'N'  (0x4C 0x2F 0xDF 0x9B 0x4E)
//           This port anchors the title at the page start instead (kG1Anchor).
//
// The byte->bit bridge, the count-limited learned confidence table and the
// ByteUpdate hook mirror Match line for line; the state index is
// (family x position-bucket) instead of match_length_.  Idle emits exactly
// 0.5, which the predictor wiring maps to an exact stretched 0 via
// MixerInput::SetZero (Logit(0.5) itself is not reliably 0 under fast-math),
// so an idle channel contributes nothing to any mixer dot product and nothing
// to any weight update.
// All state is bounded and scalar; the parser consumes only completed bytes
// (bit_context at ByteUpdate time), so encode/decode symmetry is Match's.
// ---------------------------------------------------------------------------

#include "model.h"

#include <array>

class GrammarMatch : public Model {
 public:
  GrammarMatch(const unsigned int& bit_context, int limit, float delta);
  ~GrammarMatch();
  const std::valarray<float>& Predict() const;
  void Perceive(int bit);
  void ByteUpdate();

  // Introspection accessors; never called on the hot path.
  bool HaveExpectation() const { return have_expectation_; }
  unsigned char ExpectedByte() const { return expected_byte_; }
  int StateIndex() const { return state_; }

 private:
  enum Family { kFamilyPiped = 0, kFamilyPipedPostCopy = 1, kFamilyEcho = 2 };
  enum BracketState { kBrIdle, kBrTarget, kBrLabel, kBrPostCopy, kBrScan };

  void ParseByte(unsigned char c);
  void BracketAdvance(unsigned char c);
  void EchoAdvance(unsigned char c);
  void FinishTitle();
  void ExportExpectation();

  static bool AbortClass(unsigned char c) {
    return c <= 0x05 || c == 0x08 || c == '\n' || (c >= 0x0E && c <= 0x10);
  }

  static const int kTitleCap = 256;   // raw cap was 200 cps
  static const int kTargetCap = 512;  // raw caps: 120 cps target
  static const int kEchoArm = 2;  // k>=ARM before ECHO exports

  static const unsigned long long kG1Anchor = 0x20204C2FDF984E0AULL;

  const unsigned int& bit_context_;

  // byte->bit bridge (Match's pattern)
  unsigned char expected_byte_ = 0, bit_pos_ = 128;
  bool have_expectation_ = false;
  int state_ = 0;
  int limit_;
  float delta_, divisor_;
  std::array<float, 256> predictions_;
  std::array<int, 256> counts_;

  // title tracking (ECHO)
  unsigned long long recent_ = 0;  // last 8 bytes, newest in the low byte
  bool in_title_ = false;
  int tbuf_len_ = 0;
  int title_len_ = 0;
  int echo_k_ = 0;  // KMP prefix-match length against title_
  std::array<unsigned char, kTitleCap> tbuf_;
  std::array<unsigned char, kTitleCap> title_;
  std::array<int, kTitleCap> fail_;

  // bracket tracking (PIPED)
  int br_state_ = kBrIdle;
  int tgt_len_ = 0;
  int label_j_ = 0;    // next target byte the label is expected to copy
  int label_pos_ = 0;  // bytes seen since 'Q'
  std::array<unsigned char, kTargetCap> tgt_;

};

#endif  // GRAMMAR_MATCH_H
