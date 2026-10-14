// Feed of the byte-mixer prediction into the fxcm text model.
//
// fx2-cmix (Kaido Orav) fed the byte-mixer prediction `lstmpr` into the fxcm
// INTERNAL mixers; fxcm v26 / cmix-lex kept only the producer side
// (predictor.cpp stores `lstmpr = Discretize(byte_mixer_output)`).  The feed
// is restored here in the form of fx-deepmix (Halvor Yttredal, GPL-3.0):
//
//     x.mxInputs1.add_internal(stretch(lstmpr))
//     x.mxInputs2.add_internal(stretch(lstmpr)/2)
//
// in fxcmv1.cpp.  `add_internal` adds a mixer input only; `Inputs::add` would
// also consume a prediction slot.  `lstmex` (the argmax selector context of
// fx2-cmix) is stored but not read by default.
//
// Determinism: `lstmpr` is computed from already-coded bits only; encoder and
// decoder run the identical code path.

#ifndef LSTM_FX_LINK_H_
#define LSTM_FX_LINK_H_

namespace lstm_fx_link {

// Called once per bit where lstmpr/lstmex are stored; empty by default.
inline void Push(int, int) {}

// Information-free substitutes of the same range (control arms, off by default):
// multiplicative hashes of the byte position with the xxHash32 primes 1 and 2
// (prime 1 is Knuth's multiplier too), identical on both sides, constant in a byte.
inline unsigned int UniByte(unsigned int blpos) {
  return (unsigned int)((blpos * 2654435761u) >> 13) & 255u;
}
inline unsigned int UniProb(unsigned int blpos) {
  // 1..4094 - the range Discretize() produces, so stretch() sees the same
  // domain as it does for the real lstmpr.
  return 1u + (((blpos * 2246822519u) >> 11) % 4094u);
}

}

#endif
