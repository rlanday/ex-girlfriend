#include "sparse.h"

#include <limits.h>

// factors of order position i (the first kMaxOrders of 1, 256, 29*31, 29*31*37,
// 29*31*37*41, 29*31*37*41*43)
static const unsigned int kSparseFactors[Sparse::kMaxOrders] = {1, 256, 29 * 31};

Sparse::Sparse(const std::vector<unsigned long long>& recent_contexts,
    const unsigned int* orders, unsigned int n) :
    recent_contexts_(recent_contexts), n_(n) {
  if (n > kMaxOrders) __builtin_trap();
  for (unsigned int i = 0; i < n; ++i) orders_[i] = orders[i];
  context_ = 0;
  size_ = ULLONG_MAX;
}

void Sparse::Update() {
  context_ = recent_contexts_[orders_[0]];
  for (unsigned int i = 1; i < n_; ++i) {
    context_ += kSparseFactors[i] * recent_contexts_[orders_[i]];
  }
}

bool Sparse::IsEqual(const Sparse* p) const {
  if (!p) return false;
  if (&recent_contexts_ != &(p->recent_contexts_)) return false;
  if (n_ != p->n_) return false;
  for (unsigned int i = 0; i < n_; ++i) {
    if (orders_[i] != p->orders_[i]) return false;
  }
  return true;
}

