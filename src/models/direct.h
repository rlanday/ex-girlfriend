#ifndef DIRECT_H
#define DIRECT_H

#include "model.h"

#include <vector>
#include <array>

class Direct : public Model {
 public:
  Direct(const unsigned long long& byte_context,
      const unsigned int& bit_context, int limit, float delta, int size);
  const std::valarray<float>& Predict() const;
  void Perceive(int bit);
  void ByteUpdate() {};

 private:
  const unsigned long long& byte_context_;
  const unsigned int& bit_context_;
  int limit_;
  float delta_, divisor_;
 public:
  Direct(const Direct&) = delete;
  Direct& operator=(const Direct&) = delete;
  Direct(Direct&& o) noexcept;
  ~Direct();
 private:
  void Ready(unsigned long long r) const {
    if (__builtin_expect(!ready_[r], 0)) { predictions_[r].fill(0.5); ready_[r] = 1; }
  }
  std::array<float, 256>* predictions_ = nullptr;
  std::array<unsigned char, 256>* counts_ = nullptr;
  unsigned char* ready_ = nullptr;
  unsigned long long rows_ = 0;
};

#endif

