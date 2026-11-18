#ifndef SPARSE_H
#define SPARSE_H

#include "context.h"

#include <vector>

class Sparse : public Context {
 public:
  // At most kMaxOrders word orders (the rows of Predictor::AddWord carry 1-3).
  static constexpr unsigned int kMaxOrders = 3;
  Sparse(const std::vector<unsigned long long>& recent_contexts,
      const unsigned int* orders, unsigned int n);
  void Update();
  bool IsEqual(const Sparse* p) const;

 private:
  // The orders and the constant factors live in the object resp. in a table
  // instead of two heap vectors: the class is trivially copyable and
  // destructible (no copy, fill or destroy code for it).
  const std::vector<unsigned long long>& recent_contexts_;
  unsigned int n_;
  unsigned int orders_[kMaxOrders];
};

#endif

