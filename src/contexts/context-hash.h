#ifndef CONTEXT_HASH_H
#define CONTEXT_HASH_H

#include "context.h"
#include "../fast_div.h"

class ContextHash : public Context {
 public:
  ContextHash(const unsigned int& bit_context, unsigned int order,
      unsigned int hash_size);
  void Update();
  bool IsEqual(const ContextHash* p) const;

 private:
  const unsigned int& byte_;
  unsigned int hash_size_;
  unsigned long long size_mask_ = 0;
};

#endif

