#include "direct.h"
#include "../page_registry.h"

#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
static void* DirectMap(unsigned long long n) {
  void* p = mmap(nullptr, (size_t)n, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("direct mmap"); abort(); }
  return p;
}
Direct::Direct(const unsigned long long& byte_context,
    const unsigned int& bit_context, int limit, float delta, int size) :
    byte_context_(byte_context), bit_context_(bit_context), limit_(limit),
    delta_(delta), divisor_(1.0 / (limit + delta)),
    rows_((unsigned long long)size) {
  predictions_ = (std::array<float, 256>*)DirectMap(rows_ * sizeof(predictions_[0]));
  counts_ = (std::array<unsigned char, 256>*)DirectMap(rows_ * sizeof(counts_[0]));
  ready_ = (unsigned char*)DirectMap(rows_);
}
Direct::Direct(Direct&& o) noexcept : Model(o),
    byte_context_(o.byte_context_), bit_context_(o.bit_context_),
    limit_(o.limit_), delta_(o.delta_), divisor_(o.divisor_),
    predictions_(o.predictions_), counts_(o.counts_), ready_(o.ready_),
    rows_(o.rows_) {
  o.predictions_ = nullptr; o.counts_ = nullptr; o.ready_ = nullptr; o.rows_ = 0;
}
Direct::~Direct() {
  if (predictions_) munmap(predictions_, (size_t)(rows_ * sizeof(predictions_[0])));
  if (counts_) munmap(counts_, (size_t)(rows_ * sizeof(counts_[0])));
  if (ready_) munmap(ready_, (size_t)rows_);
}

const std::valarray<float>& Direct::Predict() const {
  Ready(byte_context_);
  outputs_[0] = predictions_[byte_context_][bit_context_];
  return outputs_;
}

void Direct::Perceive(int bit) {
  Ready(byte_context_);
  float divisor = divisor_;
  if (counts_[byte_context_][bit_context_] < limit_) {
    ++counts_[byte_context_][bit_context_];
    divisor = 1.0 / (counts_[byte_context_][bit_context_] + delta_);
  }
  predictions_[byte_context_][bit_context_] +=
      (bit - predictions_[byte_context_][bit_context_]) * divisor;
}

