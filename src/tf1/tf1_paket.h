#ifndef TF1_PAKET_H_
#define TF1_PAKET_H_


namespace fx2 {

// Side channels of one fed row ("Paket" = the row form of the training
// package).  prev1/prev2 hold a "no predecessor" mark (216 in the submitted
// build), match_byte = V (208) means "no match".
struct PaketChannels {
  int prev1 = 216;   // token index of the previous fed row
  int prev2 = 216;   // the one before that
  int match_byte = 208;
  int match_class = 0;           // 0..15
};

}  // namespace fx2

#endif
