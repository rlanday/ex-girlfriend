#ifndef BRACKET_CONTEXT_H
#define BRACKET_CONTEXT_H

#include "context.h"

#include "../attributes.h"
#include <vector>

class BracketContext : public Context {
 public:
  static constexpr int kKind = Context::kBracketContext;
  int Kind() const override { return kKind; }
  BracketContext(const unsigned int& bit_context, int distance_limit,
      int stack_limit);
  void Update();
  bool IsEqual(Context* c) const;

 private:
  const unsigned int& byte_;
  unsigned char brackets_[256];
  unsigned int distance_limit_, stack_limit_;
  std::vector<unsigned int> active_, distance_;
};

#endif

