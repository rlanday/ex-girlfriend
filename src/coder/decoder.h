#ifndef DECODER_H
#define DECODER_H

#include <stdio.h>

#include "../predictor.h"
#include "../char_rate.h"
#include "../run_mode.h"
#include "../floor_stage.h"

class Decoder {
 public:
  Decoder(FILE* is, Predictor* p);
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

  FILE* is_;
  unsigned int x1_, x2_, x_;
  Predictor* p_;
  floor_stage::Stage floor_;
};

#endif
