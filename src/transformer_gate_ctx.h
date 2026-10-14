#ifndef TRANSFORMER_GATE_CTX_H_
#define TRANSFORMER_GATE_CTX_H_

namespace tf_gate {

struct CtxState {
  // zone id of the byte about to be coded, 0 = not gated
  unsigned int zone = 0;
  // the transformer's row is masked for this byte (not predicted)
  unsigned int masked = 0;
};

extern CtxState ctx;

}

#endif
