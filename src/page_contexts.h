
#ifndef PAGE_CONTEXTS_H_
#define PAGE_CONTEXTS_H_

// Width of the extra APM axis (bits), per switch.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace pagectx {

// ------------------------------------ surprise: recent coding cost as context --

// log2(1+f/32) * 16, rounded.  This lets log2(n) for n in [1,65535] be formed
// in integers with 1/16 bit resolution from msb + 5 fractional bits.
static const unsigned char kLog2Frac5[32] = {
   0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
   9,10,10,11,11,12,12,13,13,13,14,14,15,15,15,16
};

// Cost of the actually coded bit in 1/16 bit (0..256).
// p16 = P(bit==1)*65536 as used by the coder (1..65535).
inline int BitCostQ4(unsigned int p16, int bit) {
  unsigned int n = bit ? p16 : (65536u - p16);   // 1..65535
  const int msb = 31 - __builtin_clz(n);
  const unsigned int frac = (msb >= 5) ? ((n >> (msb - 5)) & 31u)
                                       : ((n << (5 - msb)) & 31u);
  const int log2q4 = (msb << 4) + kLog2Frac5[frac];
  return 256 - log2q4;                            // 16 bit == 256 sixteenths
}

// thresholds in 1/16 bit
enum { kQ2 = 32, kQ6 = 96, kQ10 = 160 };

struct SurpriseState {
  unsigned short hist[17];   // ring buffer of the last 17 byte costs (1/16 bit)
  int pos;                   // next write index
  unsigned int acc;          // cost accumulator of the current byte
  int nbits;                 // bits in the current byte
  int since;                 // bytes since the last expensive byte (> 6 bit)
  int s1, s4, s16, sinceb;   // quantised features
  unsigned int cxt3;         // 3 bits for the APM axis
  unsigned int cxtmix;       // 0..23 for the mixer

  void Reset() {
    memset(hist, 0, sizeof(hist));
    pos = 0; acc = 0; nbits = 0; since = 1000;
    s1 = s4 = s16 = sinceb = 0; cxt3 = 0; cxtmix = 0;
  }

  // Book one coded/decoded bit.  byte_done == true on the 8th bit.
  void AddBit(unsigned int p16, int bit, bool byte_done) {
    acc += (unsigned int)BitCostQ4(p16, bit);
    ++nbits;
    if (!byte_done) return;
    unsigned int c = acc;
    if (c > 65000u) c = 65000u;
    acc = 0; nbits = 0;
    hist[pos] = (unsigned short)c;
    pos = (pos + 1) % 17;

    // s1: cost of the last byte  (<2 / 2-6 / 6-10 / >10 bit)
    s1 = c < kQ2 ? 0 : (c < kQ6 ? 1 : (c < kQ10 ? 2 : 3));
    // s4: sum of the last 4 bytes  (<8 bit / <20 bit / above)
    unsigned int sum4 = 0;
    for (int i = 1; i <= 4; ++i) sum4 += hist[(pos - i + 17) % 17];
    s4 = sum4 < 128u ? 0 : (sum4 < 320u ? 1 : 2);
    // s16: number of expensive bytes (> 6 bit) among the 16 *before*
    int n = 0;
    for (int i = 2; i <= 17; ++i) {
      if (hist[(pos - i + 17) % 17] > (unsigned short)kQ6) { ++n; if (n >= 2) break; }
    }
    s16 = n;
    // bytes since the last expensive byte (0-3 / 4-15 / >= 16)
    if (c > (unsigned int)kQ6) since = 0; else if (since < 1000) ++since;
    sinceb = since <= 3 ? 0 : (since < 16 ? 1 : 2);

    cxt3   = (unsigned int)(s1 * 2 + (s16 >= 1 ? 1 : 0));          // 0..7
    cxtmix = (unsigned int)((s1 * 3 + s4) * 2 + (sinceb == 0));    // 0..23
  }
};

inline SurpriseState surprise = { {0}, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0 };

// ------------------------- link targets already seen in the current article --

enum { kLinkBits = 16, kLinkSize = 1u << kLinkBits, kLinkMask = kLinkSize - 1,
       kMaxPrefix = 48 };

struct LinkState {
  unsigned short epoch_tab[kLinkSize];  // article epoch per slot
  unsigned char  flag_tab[kLinkSize];   // bit0 prefix, bit1 complete
  unsigned short epoch;
  unsigned int   pref[kMaxPrefix];      // prefix hashes of the current target
  int   npref;
  int   inlink;                         // 1 = in target (after [[ , before | ]])
  int   done;                           // target already committed
  unsigned int hash;
  unsigned int state;                   // 0..3 -> APM/Mixer
  unsigned long long n_targets, n_repeat, n_prefix, n_bytes;

  void Reset() {
    memset(epoch_tab, 0, sizeof(epoch_tab));
    memset(flag_tab, 0, sizeof(flag_tab));
    epoch = 1; npref = 0; inlink = 0; done = 0; hash = 0; state = 0;
    n_targets = n_repeat = n_prefix = n_bytes = 0;
  }
  unsigned char Get(unsigned int h) const {
    const unsigned int k = (h * 2654435761u) >> (32 - kLinkBits);
    return epoch_tab[k] == epoch ? flag_tab[k] : 0;
  }
  void Put(unsigned int h, unsigned char f) {
    const unsigned int k = (h * 2654435761u) >> (32 - kLinkBits);
    if (epoch_tab[k] != epoch) { epoch_tab[k] = epoch; flag_tab[k] = 0; }
    flag_tab[k] |= f;
  }
  void NewArticle() {                    // reset at </page>
    ++epoch;
    if (epoch == 0) { memset(epoch_tab, 0, sizeof(epoch_tab)); epoch = 1; }
    Commit();
    inlink = 0; done = 0; npref = 0; hash = 0; state = 0;
  }
  void Enter() { inlink = 1; done = 0; npref = 0; hash = 0; state = 1; }
  // One character of the current link target.
  void Char(unsigned int j) {
    if (!inlink || done) return;
    hash = hash * 2104u + j;
    if (npref < kMaxPrefix) pref[npref++] = hash;
    const unsigned char f = Get(hash);
    state = (f & 2) ? 3u : ((f & 1) ? 2u : 1u);
    ++n_bytes;
    if (f & 1) ++n_prefix;
  }
  void Commit() {
    if (!inlink || done) return;
    done = 1;
    if (npref > 0) {
      ++n_targets;
      if (Get(pref[npref - 1]) & 2) ++n_repeat;
      for (int i = 2; i < npref - 1; ++i) Put(pref[i], 1);
      Put(pref[npref - 1], 3);
    }
  }
  void Leave() { Commit(); inlink = 0; done = 0; npref = 0; hash = 0; state = 0; }
};

inline LinkState* LinkAlloc() {
  LinkState* s = (LinkState*)calloc(1, sizeof(LinkState));
  if (s) s->Reset();
  return s;
}
inline LinkState* link = LinkAlloc();

inline unsigned int ApmExtra() {
  unsigned int v = 0;
  v = (v << 3) | (surprise.cxt3 & 7u);
  v = (v << 2) | (link ? (link->state & 3u) : 0u);
  return v;
}

}

#endif
