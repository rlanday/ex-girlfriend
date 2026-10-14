#ifndef PPMD_H
#define PPMD_H

#include "byte-model.h"

#include <array>
#include <memory>
#include <vector>

namespace PPMD {

struct ppmd_Model;

class PPMD : public ByteModel {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab);
  ~PPMD();
  std::valarray<float>& Predict();
  void Perceive(int bit);
  void ByteUpdate();
 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::valarray<int> byte_map_;
  std::array<unsigned int, 256> tree_zero_;
  std::array<unsigned int, 256> tree_total_;
  std::vector<unsigned char> disabled_bytes_;
  unsigned int tree_context_ = 1;
  bool vocab_full_ = false;
};

} // namespace PPMD

#endif
