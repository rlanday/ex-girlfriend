#ifndef LSTM_MATCH_CHANNEL_H_
#define LSTM_MATCH_CHANNEL_H_

namespace lstm_match_channel {

struct State {
  unsigned char byte_ = 0;   // predicted next byte of the longest match
  unsigned char ok_   = 0;   // is there at least one FULL BYTE of match?
  unsigned int  sym   = 0;   // column inside the thin block (0 .. vocab)

  // Called from Predictor::Perceive, AFTER the match models' ByteUpdate.
  // len is match_length_ in BITS, so a full byte is 8.
  void Set(unsigned int len, unsigned int predicted_byte) {
    byte_ = (unsigned char)predicted_byte;
    ok_   = (unsigned char)(len >= 8u ? 1u : 0u);
  }
};

// ONE instance for all translation units (predictor.cpp fills byte_/ok_,
// byte-mixer.cpp turns them into sym, lstm.hpp reads sym) - hence inline,
// not static.
inline State st;

}
#endif
