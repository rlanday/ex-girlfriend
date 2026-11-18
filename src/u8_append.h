#ifndef U8_APPEND_H_
#define U8_APPEND_H_
// Append / assign / construct a vector<unsigned char>
// from a byte range through one out-of-line helper instead of the range
// templates of libstdc++ (_M_range_insert, _M_assign_aux, _M_range_initialize,
// one copy per iterator type).  The source never lies in the target vector
// (resize may move its storage).
#include <cstring>
#include <vector>

namespace u8_append {

__attribute__((noinline)) inline void AnhK(std::vector<unsigned char>* v,
                                           const unsigned char* p, size_t n) {
  const size_t o = v->size();
  v->resize(o + n);
  if (n) memcpy(v->data() + o, p, n);
}
inline void Anh(std::vector<unsigned char>& v, const unsigned char* a, const unsigned char* b) {
  AnhK(&v, a, (size_t)(b - a));
}
inline void Anh(std::vector<unsigned char>& v, std::vector<unsigned char>::const_iterator a,
                std::vector<unsigned char>::const_iterator b) {
  AnhK(&v, a.base(), (size_t)(b - a));
}
template <class I> inline void Setze(std::vector<unsigned char>& v, I a, I b) {
  v.clear();
  Anh(v, a, b);
}

}  // namespace u8_append

#endif
