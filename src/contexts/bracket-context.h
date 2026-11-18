#ifndef BRACKET_CONTEXT_H
#define BRACKET_CONTEXT_H

#include "context.h"

#include "../attributes.h"
#include <vector>

class BracketContext : public Context {
 public:
  // Keeps its vtable: its object layout is part of the profiled hot path
  // (Update); the other contexts have none.
  virtual ~BracketContext() {}
  BracketContext(const unsigned int& bit_context, int distance_limit,
      int stack_limit);
  void Update();
  bool IsEqual(const BracketContext* p) const;

 private:
  const unsigned int& byte_;
  unsigned char brackets_[256];
  unsigned int distance_limit_, stack_limit_;
  std::vector<unsigned int> active_, distance_;
};

#endif

