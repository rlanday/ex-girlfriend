#ifndef ENCODER_H
#define ENCODER_H

#include <stdio.h>
#include <vector>

#include "../predictor.h"
#include "../char_rate.h"
#include "../run_mode.h"
#include "../floor_stage.h"
#include "../port_cfg.h"

class Encoder {
 public:
  Encoder(FILE* os, Predictor* p);
  void Encode(int bit);
  void EncodeRawBit(int bit, unsigned int p = 32768);
  inline void ObserveKnownBit(int bit) { p_->ObserveKnownBit(bit); }
  void ObserveKnownByte(unsigned int byte);
  inline void ObserveTfOnlyByte(unsigned int byte) {
    p_->ObserveTfOnlyByte(byte);
  }
  void BeginTraceByte(unsigned long long offset, unsigned int actual_byte,
      unsigned int prev4);
  void EndTraceByte();
  void Flush();
  size_t OutputSize() { return total_out_ + out_n_;}
#if PORT_CMX_MEAS
  // the coder interval [x1, x2] (research code length, src/cmx_meas.h): 8 * OutputSize() + 32 - log2(x2 - x1 + 1)
  unsigned int CmxX1() const { return x1_; }
  unsigned int CmxX2() const { return x2_; }
#endif
 private:
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  static inline char out_[(size_t)1024 * 1024];
  size_t out_n_ = 0;
  size_t total_out_ = 0;
  FILE* os_;
  unsigned int x1_, x2_;
  Predictor* p_;
  floor_stage::Stage floor_;
};
#endif
