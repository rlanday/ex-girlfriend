#include "context-manager.h"
#include "attributes.h"
#include "page_registry.h"
#include "cold_size.h"

extern unsigned long long wrtcxt;

static unsigned char bl_digits[256];

extern const unsigned char wrt_2b[256];
extern const unsigned char wrt_3b[256];
const unsigned char wrt_4b[256]={
 6, 0,12,15,12,15,14,14, 5, 3,14, 0,15,13, 8,13,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
13, 5,15,11,10,12, 6,12, 0,11,14, 1, 1,10, 9, 8,
 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9,11, 6, 1, 0, 4,
 9,10,10, 4, 5, 1, 4, 2,11, 8, 4, 1, 0,10,10, 5,
 4, 7,15, 4, 5,13, 0, 1, 4,12, 0, 1, 3, 3, 3,11,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 8, 0,11, 7,

 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
 };
 const unsigned char wrt_5b[256]={
 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
  };
#define COLON         'J' // :
#define SEMICOLON     'K' // ;
#define LESSTHAN      'L' // <
#define EQUALS        'M' // =
#define GREATERTHAN   'N' // >
#define QUESTION      'O' // ?
#define FIRSTUPPER     64 // @ - wrt first char in word is in upper case
#define SQUAREOPEN     91 // [
#define BACKSLASH      92 // '\'
#define SQUARECLOSE    93 // ]
#define CURLYOPENING  'P' // {
#define VERTICALBAR   'Q' // |
#define CURLYCLOSE    'R' // }

// All vectors use the fill constructor, which is instantiated anyway for
// words_/recent_bytes_; a resize() would pull in
// std::vector<unsigned long long>::_M_fill_insert (818 B).
// shared_map_ has 500000 rows in the release configuration (inherited size
// 400000; a larger map is an idea from L. Lakhani's fx3-cmix experiments,
// GPL-3.0).  The initializer list follows the declaration order in
// context-manager.h (clang warns otherwise); the order has no effect.
COLD_SIZE_ATTR  ContextManager::ContextManager() : history_(60000000u, 0),
    shared_map_(256*(std::size_t)500000, 0),
    words_(8, 0), recent_bytes_(8, 0),
    hashes_ind1((std::size_t)(dense_index::kN * dense_index::kN * dense_index::kN), 0),
    hashes_ind4(0x100, 0),
    hashes_ind2(0, 0), hashes_ind3(0, 0),
    hashes_ind5(0x100, 0) {
  PAGE_REGISTRY_ADD(history_.data(), history_.size());
  PAGE_REGISTRY_ADD(shared_map_.data(), shared_map_.size());
  PAGE_REGISTRY_ADD(hashes_ind1.data(), hashes_ind1.size());
  PAGE_REGISTRY_ADD(hashes_ind2.data(), hashes_ind2.size() * sizeof(std::uint32_t));
  PAGE_REGISTRY_ADD(hashes_ind3.data(), hashes_ind3.size() * sizeof(std::uint32_t));
  {
    unsigned int rank = 1;
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
    for (int i = 0; i < 256; ++i)
      bl_digits[i] = vocab_frozen::InSet(i) ? (unsigned char)(rank++) : (unsigned char)0;
  }
}

void ContextManager::UpdateLineState(unsigned char c) {
  if (c == '\n') {
    line_class_ = 0;
    line_prefix_hash_ = 0;
    return;
  }
  if (line_break_ <= 8) {
    line_prefix_hash_ = (line_prefix_hash_ * 131U + c) & 0xffU;
  }
  if (line_class_ != 0 || c == ' ' || c == '\t' || c == '\r') return;
  if (c == '#') line_class_ = 1;
  else if (c == '@') line_class_ = 2;
  else if ((c >= '0' && c <= '9') || c == 'N' || c == '-') line_class_ = 3;
  else if (c >= 0x80) line_class_ = 4;
  else if (c == '[') line_class_ = 5;
  else if (c == '*' || c == 'P' || c == 'Q' || c == 'R' || c == 'L' ||
      c == 'M' || c == '|') line_class_ = 6;
  else line_class_ = 7;
}

void ContextManager::UpdateHistory() {
  history_[history_pos_] = bit_context_;
  ++history_pos_;
  if (history_pos_ == 60000000u) history_pos_ = 0;
}

void ContextManager::UpdateWords() {
  unsigned char c = bit_context_;
  if (c==CURLYCLOSE || c==CURLYOPENING ||c==SQUARECLOSE)    b3stream= (b3stream&0xfffffff8)+3;
  else if (c==EQUALS)  b3stream=(b3stream&0xfffffff8)+4;
  n2bState=wrt_2b[c];
  b2stream=b2stream*4+n2bState;
  n3bState=wrt_3b[c];
  b3stream=b3stream*8+n3bState;
  if (o3bState!=n3bState){
      stream3bR=(stream3bR<<3)+n3bState;
      o3bState=n3bState;
  }
  if (c==10 || c==')') b3stream=b3stream<<6;
  if (c==VERTICALBAR)  b3stream=b3stream*8+wrt_3b[c];
  b2streamcxt=b2stream&0x3ff;// 2^10 bits
  b3streamcxt=b3stream&0x1ff;// 2^9 bits

  if (o2bState!=n2bState){
      stream2bR=(stream2bR<<2)+n2bState;
      o2bState=n2bState;
  }
  b4stream=b4stream*16+wrt_4b[c];
  mx18cxt=mx18cxt*16+wrt_5b[c];
  mx18=mx18cxt&0xff;

  words=words*2;

  if ((c >= 'a' && c <= 'z') || c >= 0x80) {
    words_[7] = words_[7] * 997*16 + c;
    if (recent_bytes_[0]!=12) words=words+1;

  } else {
    words_[7] = 0;
  }
  if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == 8 || c == 6 ||
      c >= 0x80) {
    words_[0] = words_[0] * 997*16 + c;
    words_[0] &= 0xfffffff;
    words_[1] = words_[1] * 263*32 + c;
  } else {
    for (int i = 6; i >= 2; --i) {
      words_[i] = words_[i-1];
    }
    words_[1] = 0;
  }
   if (c==10  )     words=0xfffc;
   else if (c=='.')                 words= words|0xffc;
   else if (c==',')                 words=words|0xffc;
   mx5=b2stream&0xffff;//2^16 bits
   mx7=    b4stream&0xff;
   mx8=(mx8*(1 << 2)+(b3stream&0x3f))& 0x3FFF;// o1, mask 2^(7*2) , n3 should be 8 bits, use 7 for now
   mx9cxt=(mx9cxt * (1 << 4) + c) &0xff;

   mx10cxt=c;

   mx14 = c*256+recent_bytes_[0];
   mx15 = recent_bytes_[0]*256+recent_bytes_[1];

   hashes_ind1[context1_ind] = (ind1 * (1 << 8) + c) & (0x100-1);
   bd1_d2_ = bd1_d1_; bd1_d1_ = bd1_d0_; bd1_d0_ = bl_digits[c];
   context1_ind = (bd1_d2_ * dense_index::kN + bd1_d1_) * dense_index::kN + bd1_d0_;
   ind1 = hashes_ind1[context1_ind];

  hashes_ind5[context1_ind5] = (ind5 * (1 << 6) + c) & (0x40000000-1);
  context1_ind5 = c;
  ind5 = hashes_ind5[context1_ind5];
}

void ContextManager::UpdateRecentBytes() {
  for (int i = 7; i >= 1; --i) {
    recent_bytes_[i] = recent_bytes_[i-1];
  }
  recent_bytes_[0] = bit_context_;
}

void ContextManager::UpdateWRTContext() {
  if (bit_context_ < 0x80) {
    wrt_state_ = 0;
  } else {
    if (wrt_state_ == 0) wrt_context_ = 0;
    wrt_state_ = 1;
    wrt_context_ <<= 8;
    wrt_context_ += bit_context_;
    if (wrt_context_ > 0xFFEFCF) wrt_context_ = 0;
  }
}

void ContextManager::UpdatePage() {
  page_sur = (unsigned long long)pagectx::surprise.cxtmix * 256 + long_bit_context_;
  page_link = (unsigned long long)(pagectx::link ? pagectx::link->state : 0u)
             * 256 + long_bit_context_;
}

void ContextManager::UpdateLx1() {
  {
    unsigned int q = 0;
    // The gate zone as a LAYER-0 CONTEXT, without adding a mixer or an
    // input (see Predictor::AddMixers): a weight row per gate zone, 5 + zone.
    // The zone is written by the gate automaton on already decoded tokens only.
    if (tf_gate::ctx.zone) q = 5u + tf_gate::ctx.zone;
    lx1_agree = (unsigned long long)q * 256 + long_bit_context_;
  }
}

// The gate bit as a mixer context.  Called like UpdatePage() /
// UpdateLx1() / UpdateReload() at the start of Predictor::Predict(), i.e. after
// UpdateContexts() has set long_bit_context_ to the bit now due.
// tf_gate::ctx.zone was written at the end of the PREVIOUS byte by the
// gate automaton, which sees already decoded tokens only, so encoder and
// decoder read the same value.
//
// The end mixer is the one place where the bit is a context of its own; the
// layer-0 side is folded into the selector lx1_agree (see UpdateLx1 above and
// the note in Predictor::AddMixers on why an extra layer-0 mixer is not
// buildable here).
void ContextManager::UpdateTf1Gate() {
  tf1_gate_ctx1 = (unsigned long long)tf_gate::ctx.zone;
}

void ContextManager::UpdateReload() {
  // Not called in this configuration; the line class from UpdateLineState(),
  // which is not called either, stays 0.
  reload_line = (unsigned long long)(line_class_ & 7u) * 256 + long_bit_context_;
  reload_mtc = (unsigned long long)dictreload::MtcSel() * 256 + long_bit_context_;
  reload_ord = (unsigned long long)dictreload::OrdSel() * 256 + long_bit_context_;
}

void ContextManager::UpdateContexts(int bit) {
  bit_context_ += bit_context_ + bit;
  long_bit_context_ = bit_context_;
  if (bit_context_ >= 256) {
    bit_context_ -= 256;
    long_bit_context_ = 1;
    longest_match_ = 0;

    if (bit_context_ == '\n') {
      line_break_ = 0;
    } else if (line_break_ < 99) {
      ++line_break_;
    }

    UpdateHistory();
    UpdateWords();
    UpdateRecentBytes();
    UpdateWRTContext();

    for (auto& context : context_hash_contexts_) {
      context.Update();
    }
    for (auto& context : sparse_contexts_) {
      context.Update();
    }
    for (auto& context : bracket_contexts_) {
      context.Update();
    }
  }
  wordscxt=(words&0x7F)*256+long_bit_context_;

  bpos=(bpos+1)&7;

  mx6=(stream2bR&0xff)*256+long_bit_context_;

  mx19cxt=    wrtcxt;
  mx9=(mx9cxt)*256+long_bit_context_;
  mx10=(mx10cxt)*256+long_bit_context_;
  mx16=(recent_bytes_[1])*256+long_bit_context_;
  mx17=(b3stream&0x3f)*256+long_bit_context_;// 7f or 3f
}
