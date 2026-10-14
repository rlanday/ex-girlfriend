#ifndef DECODER_H
#define DECODER_H

#include <fstream>

#include "../predictor.h"
#include "../char_rate.h"
#include "../run_mode.h"

class Decoder {
 public:
  Decoder(std::ifstream* is, Predictor* p);
  int Decode();
  int DecodeRawBit(unsigned int p = 32768);
  inline void ObserveKnownBit(int bit) { p_->ObserveKnownBit(bit); }
  void ObserveKnownByte(unsigned int byte);
  inline void ObserveTfOnlyByte(unsigned int byte) {
    p_->ObserveTfOnlyByte(byte);
  }

 private:
  int ReadByte();
  unsigned int Discretize(float p);

  std::ifstream* is_;
  unsigned int x1_, x2_, x_;
  Predictor* p_;
};

#endif
