#ifndef TF1_GATE_H_
#define TF1_GATE_H_

#include "../alias_title_subst.h"

#include "../metablock.h"

#include "../transformer.h"

#include <stdint.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "../transformer_gate_ctx.h"

namespace tf_gate {

// ---- zones ---------------------------------------------------------------
// Kopfrest = page head without title, Titelmask = title line (fed, not
// predicted), Hdrblock = metadata block.  The constructor arms the zones of
// the build.
enum Zone {
  kZoneNone = 0,
  kZoneKopfrest = 1,
  kZoneCmtidx = 2,
  kZoneHdrblock = 3,
  kZoneDict0 = 4,
  kZoneTitelmask = 5,
  kZoneCount = 6
};

static_assert(kZoneHdrblock == 3,
              "bit_head::MetaOff() compares against 3");

inline bool ZoneFeeds(int z) {
  return z == kZoneNone || z == kZoneTitelmask;
}
const char* ZoneName(int z);

// ---- parameters ----------------------------------------------------------
struct Params {
  int max_head = 4096;   // safety valve: this many tokens per page head
  int max_cidx = 64;
};

// ---- counters ------------------------------------------------------------
struct Stats {
  unsigned long long tokens = 0;
  unsigned long long zones[kZoneCount] = {0, 0, 0, 0, 0, 0};
  unsigned long long bytes[kZoneCount] = {0, 0, 0, 0, 0, 0};
  unsigned long long heads = 0;
  unsigned long long notaus = 0;      // "emergency stop": safety valve fired
  unsigned long long planbruch = 0;   // "plan break": schedule aborted
  unsigned long long cidx_index = 0;  // pure index form
  unsigned long long cidx_esc = 0;    // escape form "\x02\x03<SEC>\n"
  unsigned long long steps_saved = 0;
  unsigned long long mf_start = 0;
  unsigned long long mf_ende = 0;
  unsigned long long mf_falsch = 0;
  unsigned long long hdr_start = 0;
  // how often a page reached the second line of the digit test (2,099 times
  // over the full stream)
  unsigned long long hdr_kandidat = 0;
};

// ---- the automaton -------------------------------------------------------
// `Zone()` is the zone of the NEXT token, decided from the tokens seen so
// far.  `Observe(token)` takes in one coded token; after it `Zone()` holds
// the answer for the token after that.
class CoderGate {
 public:
  explicit CoderGate(const Params& p);
  ~CoderGate();

  void Observe(unsigned char token);

  int Zone() const { return zone_; }
  bool Gated() const { return zone_ != kZoneNone; }
  bool Feeds() const { return ZoneFeeds(zone_); }

  void CountStepSaved() { ++st_.steps_saved; }

  void Report() const;
  const Stats& GetStats() const { return st_; }

  // Runtime selection of the zones for the standalone tool; the compressor
  // uses the fixed selection set in the constructor.
  void SetArms(const bool* arms, int cmtidx_form);

 private:
  enum {
    kIdle = 0, kPlan, kTitle, kHead, kCmt, kBody, kRev, kCidx, kPend, kHdr,
    kG1Cand2
  };

  // A rolling comparison against a fixed token sequence.  The buffer holds
  // the last `n` tokens; `Push` returns true when they equal the pattern.
  struct Roll {
    const unsigned char* pat = nullptr;
    int n = 0;
    unsigned char buf[16] = {0};
    void Init(const unsigned char* p, int len) { pat = p; n = len; Reset(); }
    void Reset() { for (int i = 0; i < 16; ++i) buf[i] = 0; }
    bool Push(unsigned char t) {
      for (int i = 0; i + 1 < n; ++i) buf[i] = buf[i + 1];
      buf[n - 1] = t;
      for (int i = 0; i < n; ++i) if (buf[i] != pat[i]) return false;
      return true;
    }
  };

  void SetZone(int z);
  void G1Observe(unsigned char token);

  Params p_;
  bool arm_[kZoneCount];
  int cmtidx_form_;        // 1 = both forms, 2 = only the pure index form
  int st_state_;
  int zone_;
  Roll r_anker_;
  bool g1_pure_;
  int g1_dig_;
  bool mf_auf_, mf_raus_, mf_tab_;
  bool mb_nach_;
  Stats st_;
  FILE* dump_ = nullptr;   // one byte per token: the zone id
  std::vector<unsigned char> dbuf_;
};

// Tool mode: run the same class over a stream file.
int RunTool(int argc, char** argv);

}

#endif
