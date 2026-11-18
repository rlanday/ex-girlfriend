#include "../meld.h"
#include "tf1_gate.h"

#include "../cold_size.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

// Package form: the engine's stop (tf1_engine.cpp, linked anyway) prints the
// same lines as G1Fail below (newline, code, newline) and exits with 1.
namespace transformer {
[[noreturn]] void Fail(const char* fmt, ...);
}

namespace tf_gate {

CtxState ctx;

// ---- the stamped grammar -------------------------------------------------
static const unsigned char kAnker[8] =
    { 0x0b, 0x0b, 0x28, 0x1a, 0xaf, 0x68, 0x2a, 0x08 };

// A loud stop of its own: `transformer::Fail` lives in tf1_engine.cpp, and the
// tool build (`tf1_gate`) links this translation unit alone; pulling the
// engine in for one error path would be the wrong trade in a binary that
// counts double.  Same behaviour: message to stderr, exit 1, no return.
#define G1Fail ::transformer::Fail

const char* ZoneName(int z) {
  switch (z) {
    case kZoneKopfrest: return "KOPFREST";
    case kZoneCmtidx: return "CMTIDX";
    case kZoneHdrblock: return "HDRBLOCK";
    case kZoneDict0: return "DICT0";
    case kZoneTitelmask: return "TITELMASK";
    default: return "-";
  }
}

COLD_SIZE_ATTR CoderGate::CoderGate(const Params& p)
    : p_(p), cmtidx_form_(0), st_state_(kIdle), zone_(kZoneNone),
      g1_pure_(false), g1_dig_(0)
      , mf_auf_(false), mf_raus_(false), mf_tab_(false)
      , mb_nach_(false)
      {
  for (int i = 0; i < kZoneCount; ++i) arm_[i] = false;
  arm_[kZoneHdrblock] = true;
  r_anker_.Init(kAnker, 8);
}

COLD_SIZE_ATTR CoderGate::~CoderGate() {
}

void CoderGate::SetZone(int z) {
  if (z != kZoneNone && !arm_[z]) z = kZoneNone;
  if (z != kZoneNone && zone_ == kZoneNone) {}
  zone_ = z;
  ctx.zone = (unsigned int)z;
  ctx.masked = (z != kZoneNone) ? 1u : 0u;
}

void CoderGate::Observe(unsigned char token) {
  // book the token that was just processed
  ++st_.tokens;
  if (zone_ != kZoneNone) {}
  G1Observe(token);
}

void CoderGate::G1Observe(unsigned char token) {
  if (!mf_raus_) {
    static const unsigned char kTabTok = transformer::kByteToIndex[0x09];
    bool mf_ausgang = false;
    if (!mf_auf_) { mf_auf_ = true; st_.mf_start = st_.tokens; }
    {
      static_assert(transformer::kByteToIndex[0x0B] >= 0, "the end mark 0x0B is in the byte set");
      static const unsigned char kEndeTok = transformer::kByteToIndex[0x0B];
      if (metablock::mk.ImTextanfang()) {
        ++st_.ta_tokens;
        const bool letzte = token == kEndeTok && metablock::mk.TaSchliesstLetzten(0x0B);
        SetZone(letzte && arm_[kZoneHdrblock] ? kZoneHdrblock : kZoneNone);
        return;
      }
      if (token == 0x08 && metablock::mk.SchliesstMit(0x0A)) {
        SetZone(kZoneNone);
        return;
      }
    }
    if (metablock::mk.Fertig()) {
      if (!mb_nach_) {
        mb_nach_ = true;
        if (token == 0x08)
          G1Fail("E214 %llu", st_.tokens);
      } else if (token == 0x08) {
        mf_ausgang = true;
      }
    }
    (void)kTabTok;
    if (mf_ausgang) {
      mf_raus_ = true;
        st_.mf_ende = st_.tokens;        // line feed of the first title line
        st_state_ = kBody;               // its title is spent -- body follows
        g1_pure_ = false;
        g1_dig_ = 0;
        r_anker_.Reset();
        SetZone(kZoneHdrblock);
        return;
    }
    if (!metablock::mk.Offen() &&
        st_.tokens > (unsigned long long)4096)
      G1Fail("E215 %llu", st_.tokens);
    SetZone(mf_auf_ && arm_[kZoneHdrblock] ? kZoneHdrblock : kZoneNone);
    return;
  }

  // (1) Inside the metadata block every token is gated.  The anchor
  //     recognizer keeps running: an anchor in here would prove that the zone
  //     was entered too early, the one failure this rule can have, so it
  //     stops the program.
  if (st_state_ == kHdr) {
    if (r_anker_.Push(token)) {
      G1Fail("E216 %llu %llu", st_.tokens, st_.hdr_start);
    }
    SetZone(kZoneHdrblock);
    return;
  }

  // (2) The anchor closes a page and opens the next one.  Checked first and
  //     in every state: it is the only page mark, and a page boundary
  //     overrides whatever line the automaton thought it was in.  It ends on
  //     the line feed, so the title begins with the very next token.
  if (r_anker_.Push(token)) {
    ++ctx.pages;   // page count behind the metablock (zone switch of the BITLSTM)
    st_state_ = kTitle;
    g1_pure_ = true;
    g1_dig_ = 0;
    SetZone(kZoneTitelmask);
    return;
  }

  // (3) the title line.
  if (st_state_ == kTitle) {
    if (token == 0x08) {
      // The line is over.  A pure digit line arms the metadata-block test; it
      // is not the decision yet.  The line feed itself stays inside the title
      // mask, because the automaton only sees it after the fact.
      if (g1_pure_ && g1_dig_ > 0) {
        st_state_ = kG1Cand2;
        g1_pure_ = true;
        g1_dig_ = 0;
      } else {
        st_state_ = kBody;
      }
      SetZone(kZoneNone);
      return;
    }
    if (token < 0x1b || token > 0x24) {
      g1_pure_ = false;
    } else {
      ++g1_dig_;
    }
    SetZone(kZoneTitelmask);
    return;
  }

  // (4) Rule of the earlier stream form with the metadata block at the
  //     tail: two pure digit lines in a row would open the block.  With the
  //     block at the head it never fires on enwik9; zone 3 is set from the
  //     line count.
  if (st_state_ == kG1Cand2) {
    if (token == 0x08) {
      if (g1_pure_ && g1_dig_ > 0 && arm_[kZoneHdrblock]) {
        st_state_ = kHdr;
        st_.hdr_start = st_.tokens;   // the first gated token, 0-based
        r_anker_.Reset();
        SetZone(kZoneHdrblock);
      } else {
        st_state_ = kBody;
        SetZone(kZoneNone);
      }
      return;
    }
    if (token < 0x1b || token > 0x24) {
      g1_pure_ = false;
    } else {
      ++g1_dig_;
    }
    SetZone(kZoneNone);
    return;
  }

  if (st_state_ == kIdle) {
    if (token == 0x08) {
      st_state_ = kTitle;
      g1_pure_ = true;
      g1_dig_ = 0;
      SetZone(kZoneTitelmask);
      return;
    }
    SetZone(kZoneNone);
    return;
  }

  SetZone(kZoneNone);
}

void CoderGate::SetArms(const bool* arms, int cmtidx_form) {
  for (int i = 0; i < kZoneCount; ++i) arm_[i] = arms[i];
  cmtidx_form_ = cmtidx_form;
  if (!mf_raus_ && arm_[kZoneHdrblock]) {
    zone_ = kZoneHdrblock;
  }
}

void CoderGate::Report() const {
}

}  // namespace tf_gate

