#ifndef MIX_SELECT_H_
#define MIX_SELECT_H_


// Add rows of the final mixer.  An add row is one more final mixer behind
// mixer_1_[0] on the same stage-1 row, with a context of its own: on a byte
// with a live transformer row (the slot got SetProbs, the gate zone is 0) the
// context is kBase + selector, otherwise the gate zone.  Its weights start at
// zero; the final-mixer output is the sum of all rows, every row learns on the
// error of that sum, and the gradient of the layer-0 mixers takes the summed
// row as its chain factor.  Learning rate kAddLr per mille of the final
// mixer's 0.0003.  Selector: path state of the transformer slot in
// 6 classes (its favourite still inside the slot's bit interval; else the rank
// of the best remaining top-16 candidate: 2, 3, 4-5, 6-10, > 10) x confidence
// class of p(top-1) of the row (< 0.5, 0.5-0.9, 0.9-0.99, >= 0.99) = 24 rows.
// Encoder and decoder see the same selector: the row comes from SetProbs at
// the byte boundary, the path state from the slot's bit interval after the bit
// just coded.

namespace mix_select {

// End-mixer contexts of a selector start here; gate zones are 0..5.
constexpr unsigned long long kBase = 16;
// learning rate of an add row, per mille of the end mixer's 0.0003
constexpr int kAddLr = 1000;
// index of the add rows in mixer_1_ (row A first, if it exists)
constexpr unsigned int kIa = 1;
constexpr unsigned int kIb = 2;

extern unsigned long long ctxa, ctxb;   // contexts of the add rows

}  // namespace mix_select

// Path state of the transformer slot for the next bit (form 16), filled by
// Predictor::MlPath once per bit.
namespace tf_path {
constexpr int kTop = 16;
struct St {
  int live, fav, conf, ntop, top[kTop];
  int ps, d;   // path class and orientation of the next bit
};
extern St st;
// path state (6) x confidence class (4) of the transformer row, 0..23
inline int Sel24() { return st.ps * 4 + st.conf; }
}  // namespace tf_path

#endif
