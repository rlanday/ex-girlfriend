#ifndef LSTM_ROW_GEOMETRY_H_
#define LSTM_ROW_GEOMETRY_H_


namespace lstm_mixer_inputs {

enum {
  kPay3   = 8,
  kPay    = kPay3,
  kPayMax = 38,
  kMaxSlot = 80                              // payload + fill, upper bound
};

constexpr unsigned int kBase = (unsigned int)(289);   // vocab + 1 + cells

// Payload slots, padded so that the row length is a multiple of 32.
constexpr unsigned int CalcSlots() {
  unsigned int s = 0 ? (unsigned int)kPayMax
                                      : (unsigned int)kPay;
  while ((kBase + s) % (unsigned int)32) ++s;
  if (s > (unsigned int)kMaxSlot) s = (unsigned int)kMaxSlot;
  return s;
}

constexpr unsigned int kSlots = CalcSlots();      // payload + fill
constexpr unsigned int kRow   = kBase + kSlots;   // vocab+1+cells+slots

}

#endif
