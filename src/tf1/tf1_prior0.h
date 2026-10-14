#ifndef TF1_PRIOR0_H_
#define TF1_PRIOR0_H_

#include "../transformer.h"

#include <stdint.h>

#include "tf1_prior0_set_v9.h"

namespace transformer {

// prior0 is the replacement prior of inserted rows.  The stamped table must be
// the one of the training package the weights were trained on; a table from
// another package would not fail, it would only predict worse.

static_assert(208 == 208,
    "the stamped prior0 table must have exactly V columns "
    "(z3-prior0.float16 of the training package).");
inline const uint16_t* Prior0Stamped() {
  static const uint16_t kTable[208] = PRIOR0_INIT;
  return kTable;
}
inline const char* Prior0Sha() { return PRIOR0_SHA256; }
// text naming the prior0 source (sha256 of the stamped table)
inline const char* Prior0Label(bool given) {
  return Prior0Stamped() ? "stamped:" PRIOR0_SHA256
                         : (given ? "supplied" : "uniform");
}

}

#endif
