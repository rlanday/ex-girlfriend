#ifndef ENCODER_H
#define ENCODER_H

#include <fstream>
#include <vector>

#include "../predictor.h"
#include "../char_rate.h"
#include "../run_mode.h"

class Encoder {
 public:
  Encoder(std::ofstream* os, Predictor* p);
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
  size_t OutputSize() { return total_out_ + out_.size();}
 private:
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  std::vector<char> out_;
  size_t total_out_ = 0;
  std::ofstream* os_;
  unsigned int x1_, x2_;
  Predictor* p_;
};
#endif
