#ifndef STRING_POOL_H_
#define STRING_POOL_H_

#include <stdint.h>
#include <string.h>

#include <vector>


namespace string_pool {

// An arena buffer plus an open hash table.  No std::string, no std::map: on
// the full stream the table carries about a million entries, and allocating
// every entry separately would be the bottleneck.
class StrPool {
 public:
  // The length is not stored: the pool only grows at the end, so `off_[id+1]`
  // is the end of `id` (a length column would save nothing and cost 4 B per
  // entry).
  StrPool() {
    off_.push_back(0);            // id 0: placeholder, length 0
    off_.push_back(0);            // also the start of the first entry
    next_.push_back(0);
    head_.assign(1 << 12, 0);
    mask_ = head_.size() - 1;
  }
  // Id of the string; creates it when it is new.  0 is reserved.
  uint32_t Intern(const unsigned char* p, int len) {
    if (uint32_t id = Find(p, len)) return id;
    if (Count() + 1 > head_.size()) Grow();
    const uint32_t id = (uint32_t)off_.size() - 1;
    bytes_.insert(bytes_.end(), p, p + len);
    off_.push_back((uint32_t)bytes_.size());
    const size_t b = (size_t)(HashOf(p, len) & mask_);
    next_.push_back(head_[b]);
    head_[b] = id;
    return id;
  }
  // Id of the string, or 0 when unknown (creates nothing).
  uint32_t Find(const unsigned char* p, int len) const {
    const size_t b = (size_t)(HashOf(p, len) & mask_);
    for (uint32_t id = head_[b]; id; id = next_[id]) {
      if (Len(id) == len && memcmp(Data(id), p, (size_t)len) == 0) {
        return id;
      }
    }
    return 0;
  }
  const unsigned char* Data(uint32_t id) const {
    return bytes_.data() + off_[id];
  }
  int Len(uint32_t id) const { return (int)(off_[id + 1] - off_[id]); }
  size_t Count() const { return off_.size() - 2; }
  size_t Bytes() const {
    return bytes_.capacity() + off_.capacity() * 4
         + head_.capacity() * 4 + next_.capacity() * 4;
  }

 private:
  std::vector<unsigned char> bytes_;
  std::vector<uint32_t> off_;     // off_[id] .. off_[id+1] are the bytes of id
  std::vector<uint32_t> head_;    // bucket -> id, 0 = empty
  std::vector<uint32_t> next_;
  size_t mask_ = 0;
  uint64_t HashOf(const unsigned char* p, int len) const {
    uint64_t h = 1469598103934665603ULL;                  // FNV-1a
    for (int i = 0; i < len; ++i) {
      h ^= (uint64_t)p[i];
      h *= 1099511628211ULL;
    }
    return h;
  }
  void Grow() {
    head_.assign(head_.size() * 2, 0);
    mask_ = head_.size() - 1;
    for (uint32_t id = 1; id <= (uint32_t)Count(); ++id) {
      const size_t b = (size_t)(HashOf(Data(id), Len(id)) & mask_);
      next_[id] = head_[b];
      head_[b] = id;
    }
  }
};

}

#endif
