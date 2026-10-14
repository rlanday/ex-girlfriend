#ifndef LSTM_MIXER_INPUTS_H_
#define LSTM_MIXER_INPUTS_H_

#include <stdint.h>
#include <string.h>

#include "lstm_row_geometry.h"
#include <stdio.h>
#include <stdlib.h>

// Dense extra inputs of the byte LSTM: for each bit position of the last
// byte, the coded cost of that bit minus its running average (EMA), in bits.
namespace lstm_mixer_inputs {

static const unsigned char kLog2Frac5[32] = {
   0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
   9,10,10,11,11,12,12,13,13,13,14,14,15,15,15,16
};

// Cost of the bit actually coded, in 1/16 bit (0..256).
inline int BitCostQ4(unsigned int p16, int bit) {
  unsigned int n = bit ? p16 : (65536u - p16);   // 1..65535
  const int msb = 31 - __builtin_clz(n);
  const unsigned int frac = (msb >= 5) ? ((n >> (msb - 5)) & 31u)
                                       : ((n << (5 - msb)) & 31u);
  const int log2q4 = (msb << 4) + kLog2Frac5[frac];
  return 256 - log2q4;
}

// ------------------------------------------------------------- state -----
struct State {
  static constexpr unsigned int slots = kSlots;   // payload + fill
  static constexpr unsigned int row   = kRow;     // vocab+1+cells+slots
  float slot[kMaxSlot] = {};     // what the ByteMixer feeds in per byte

  // running byte accumulation (1/16 bit)
  int sys_q4[8] = {};
  int bpos = 0;

  // EMA states, integer in 1/(16*256) bit
  int ema3[8] = {};
  bool warm = false;             // first bytes: set the EMA to the first value
  int nbyte = 0;

  // ring buffer (not read in the release configuration)
  float ring[64][kPay > 0 ? kPay : 1] = {};
  int rpos = 0;

  void Reset() {
    memset(sys_q4, 0, sizeof(sys_q4));
    memset(ema3, 0, sizeof(ema3));
    memset(slot, 0, sizeof(slot));
    memset(ring, 0, sizeof(ring));
    bpos = 0; rpos = 0; nbyte = 0; warm = false;
  }

  void Geometry(unsigned int base) {
    if (base != kBase) {
      fprintf(stderr,
          "\nabort: the dense row reports base=%u, "
          "compiled in is %u (frozen byte set + 1 + cells"
          ").  Changing the byte set or the cell count changes "
          "this number too.\n", base, kBase);
      fflush(stderr);
      abort();
    }
  }

  // ---- account one coded/decoded bit (from Predictor::Perceive) ----------
  void AddBit(int bit, unsigned int p16_sys) {
    if (bpos < 8) sys_q4[bpos] = BitCostQ4(p16_sys, bit);
    ++bpos;
  }

  // ---- byte finished: build the slots for the NEXT byte ------------------
  void ByteEnd() {
    float v[kPay > 0 ? kPay : 1];
    ++nbyte;
    for (int i = 0; i < 8; ++i) {
      const int raw = sys_q4[i];
      // Cold start: the EMA starts ON the first value, not on 0 - otherwise
      // the first byte produces an artificial huge excursion.
      if (!warm) ema3[i] = raw << 8;
      v[i] = (float)(raw - (ema3[i] >> 8)) * (1.0f / (16.0f * 4));
      ema3[i] += ((raw << 8) - ema3[i]) >> 6;
      sys_q4[i] = 0;
    }
    warm = true;
    bpos = 0;
    for (int k = 0; k < kPay; ++k) slot[k] = v[k];
  }
};

// ONE instance for all translation units (predictor.cpp sets it up,
// byte-mixer.cpp reads the slots) - hence inline, not static.
inline State st;

}

#endif
