// Line-by-line port from fx-deepmix (Halvor Yttredal), GPL-3.0.
// See grammar-match.h for the design notes and the changes.
#include "grammar-match.h"

GrammarMatch::GrammarMatch(const unsigned int& bit_context, int limit,
    float delta) : bit_context_(bit_context), limit_(limit), delta_(delta),
    divisor_(1.0 / (limit + delta)) {
  // Flat 0.5 init: the channel is born silent everywhere and only speaks once
  // a state's confidence has moved off neutral (unlike Match's ramped init,
  // which encodes a match-length prior we do not have).
  predictions_.fill(0.5f);
  counts_.fill(0);
  tbuf_.fill(0);
  title_.fill(0);
  fail_.fill(0);
  tgt_.fill(0);
}

GrammarMatch::~GrammarMatch() {
}

const std::valarray<float>& GrammarMatch::Predict() const {
  if (!have_expectation_) {
    outputs_[0] = 0.5f;  // exact: the predictor wiring turns this into SetZero
    return outputs_;
  }
  if (expected_byte_ & bit_pos_) outputs_[0] = predictions_[state_];
  else outputs_[0] = 1 - predictions_[state_];
  return outputs_;
}

void GrammarMatch::Perceive(int bit) {
  if (!have_expectation_) return;  // idle is silence: no table update either
  int match = 0;
  if (bit == ((expected_byte_ & bit_pos_) != 0)) match = 1;
  bit_pos_ /= 2;

  float divisor = divisor_;
  if (counts_[state_] < limit_) {
    ++counts_[state_];
    divisor = 1.0 / (counts_[state_] + delta_);
  }
  predictions_[state_] += (match - predictions_[state_]) * divisor;

  if (!match) {
    // The grammar's byte is wrong at this site: go soft (silent) for the rest
    // of the byte; ByteUpdate sees the actual byte and re-derives the
    // byte-level state (divergence -> scan/idle) at the boundary.
    have_expectation_ = false;
  }
}

void GrammarMatch::ByteUpdate() {
  ParseByte(static_cast<unsigned char>(bit_context_));
  bit_pos_ = 128;
}

void GrammarMatch::ParseByte(unsigned char c) {
  recent_ = (recent_ << 8) | c;

  // --- title capture -----------------------------------------------------
  if (!in_title_) {
    if (recent_ == kG1Anchor) {
      // A new page begins: the previous title (and its echoes) are stale.
      in_title_ = true;
      tbuf_len_ = 0;
      title_len_ = 0;
      echo_k_ = 0;
    }
  } else {
    if (c == '\n') {
      FinishTitle();
    } else if (tbuf_len_ >= kTitleCap) {
      in_title_ = false;  // oversize title: no echo family this article
      tbuf_len_ = 0;
    } else {
      tbuf_[tbuf_len_++] = c;
    }
  }

  BracketAdvance(c);
  if (!in_title_ && title_len_ > 0) EchoAdvance(c);
  ExportExpectation();
}

void GrammarMatch::BracketAdvance(unsigned char c) {
  if (c == '[') {  // a fresh bracket restarts target capture in every state
    br_state_ = kBrTarget;
    tgt_len_ = 0;
    return;
  }
  switch (br_state_) {
    case kBrIdle:
      break;
    case kBrTarget:
      if (c == 'Q') {  // 'Q' is '|': literal uppercase never survives WRT
        if (tgt_len_ > 0) {
          br_state_ = kBrLabel;
          label_j_ = (tgt_[0] == 0x40 && tgt_len_ > 1) ? 1 : 0;  // CAP-strip
          label_pos_ = 0;
        } else {
          br_state_ = kBrIdle;
        }
      } else if (c == ']' || AbortClass(c)) {
        br_state_ = kBrIdle;  // unpiped link / class violation
      } else if (tgt_len_ >= kTargetCap) {
        br_state_ = kBrIdle;  // overflow
      } else {
        tgt_[tgt_len_++] = c;
      }
      break;
    case kBrLabel:
      if (c == ']') {
        br_state_ = kBrIdle;  // label ended (early ']' = PREFIX site over)
      } else if (c == 'Q') {
        br_state_ = kBrScan;  // multi-parameter ([image:...Q...Q...])
      } else if (AbortClass(c)) {
        br_state_ = kBrIdle;
      } else if (label_j_ < tgt_len_ && c == tgt_[label_j_]) {
        ++label_j_;
        if (label_j_ >= tgt_len_) br_state_ = kBrPostCopy;
      } else if (label_pos_ == 0 && label_j_ == 1 && c == 0x40) {
        // Label carries its own cap flag ([@xyz Q @xyz...]): the first label
        // byte repeats 0x40; keep comparing from tgt_[1].
      } else {
        br_state_ = kBrScan;  // diverged (e.g. codeword lead in the label)
      }
      ++label_pos_;
      break;
    case kBrPostCopy:
      if (c == ']') {
        br_state_ = kBrIdle;  // full-copy site completed
      } else if (AbortClass(c)) {
        br_state_ = kBrIdle;
      } else {
        br_state_ = kBrScan;  // EXTENDS suffix running; idle until ']'
      }
      ++label_pos_;
      break;
    case kBrScan:
      if (c == ']' || AbortClass(c)) {
        br_state_ = kBrIdle;
      } else if (++label_pos_ > kTargetCap) {
        br_state_ = kBrIdle;  // runaway label region
      }
      break;
  }
}

void GrammarMatch::EchoAdvance(unsigned char c) {
  while (echo_k_ > 0 && c != title_[echo_k_]) echo_k_ = fail_[echo_k_ - 1];
  if (c == title_[echo_k_]) ++echo_k_;
  if (echo_k_ >= title_len_) echo_k_ = fail_[title_len_ - 1];  // full echo
}

void GrammarMatch::FinishTitle() {
  in_title_ = false;
  title_len_ = tbuf_len_;
  echo_k_ = 0;
  if (title_len_ <= 0) return;
  for (int i = 0; i < title_len_; ++i) title_[i] = tbuf_[i];
  fail_[0] = 0;
  for (int i = 1; i < title_len_; ++i) {
    int k = fail_[i - 1];
    while (k > 0 && title_[i] != title_[k]) k = fail_[k - 1];
    if (title_[i] == title_[k]) ++k;
    fail_[i] = k;
  }
}

void GrammarMatch::ExportExpectation() {
  have_expectation_ = false;
  // Priority: bracket family > echo (bracket knowledge is sharper); exactly
  // one expected byte is exported per byte.
  if (br_state_ == kBrLabel && label_j_ < tgt_len_) {
    expected_byte_ = tgt_[label_j_];
    state_ = kFamilyPiped * 32 + (label_j_ < 31 ? label_j_ : 31);
    have_expectation_ = true;
  } else if (br_state_ == kBrPostCopy) {
    expected_byte_ = ']';
    state_ = kFamilyPipedPostCopy * 32;
    have_expectation_ = true;
  } else if (!in_title_ && echo_k_ >= kEchoArm && echo_k_ < title_len_) {
    expected_byte_ = title_[echo_k_];
    state_ = kFamilyEcho * 32 + (echo_k_ < 31 ? echo_k_ : 31);
    have_expectation_ = true;
  }

}

