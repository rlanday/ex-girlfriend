#ifndef PPMD_PAGE_PTR_H_
#define PPMD_PAGE_PTR_H_

#include "ppmd_page.h"
#include <type_traits>
#include <cstddef>

namespace ppmd_page {

// hzg<T> is a proxy pointer into the paged PPMd heap: it stores the canonical
// address and translates it to the current frame on every dereference.
// FREMD = 1 only for BLK_NODE and MEM_BLK, whose pointers may leave the heap
// (BList in the model object, s0 on the stack).
// HEISS = 1 inlines the fast path at this use (the hot functions only).
template <class T, int FREMD = 0, int HEISS = 0> struct hzg {
  unsigned char* a;                               // CANONICAL address

  hzg() : a(0) {}
  hzg(T* p) : a((unsigned char*)p) {}
  hzg(const hzg& o) : a(o.a) {}
  explicit hzg(void* p) : a((unsigned char*)p) {}
  // derived -> base (MEM_BLK -> BLK_NODE), as with raw pointers
  template <class U, int G, int H, class = typename std::enable_if<
              std::is_convertible<U*, T*>::value && !std::is_same<U,T>::value>::type>
  hzg(const hzg<U,G,H>& o) : a(o.a) {}
  // same type, other FREMD/HEISS level: implicit (only the code form differs)
  template <int G, int H> hzg(const hzg<T,G,H>& o) : a(o.a) {}
  // any other reinterpretation only explicitly (like a C cast)
  template <class U, int G, int H> explicit hzg(const hzg<U,G,H>& o, int = 0) : a(o.a) {}

  // read access -> translated, read-only
  const T* roh() const { return (const T*)xl_t<FREMD,HEISS,0>(a, sizeof(T)); }
  const T* operator->() const { return roh(); }
  const T& operator*()  const { return *roh(); }
  const T& operator[](long i) const {
    return *(const T*)xl_t<FREMD,HEISS,0>(a + i * (long)sizeof(T), sizeof(T));
  }
  // write access -> translated, marks the frame(s) dirty
  T* w() const { return (T*)xl_t<FREMD,HEISS,1>(a, sizeof(T)); }
  T& w(long i) const {
    return *(T*)xl_t<FREMD,HEISS,1>(a + i * (long)sizeof(T), sizeof(T));
  }

  // arithmetic and comparison -> canonical, unchanged against the canon
  hzg  operator+(long n) const { return hzg((unsigned char*)(a + n*(long)sizeof(T))); }
  hzg  operator-(long n) const { return hzg((unsigned char*)(a - n*(long)sizeof(T))); }
  long operator-(const hzg& o) const { return (long)((a - o.a) / (long)sizeof(T)); }
  hzg& operator++()    { a += sizeof(T); return *this; }
  hzg  operator++(int) { hzg t(*this); a += sizeof(T); return t; }
  hzg& operator--()    { a -= sizeof(T); return *this; }
  hzg  operator--(int) { hzg t(*this); a -= sizeof(T); return t; }
  hzg& operator+=(long n) { a += n*(long)sizeof(T); return *this; }
  hzg& operator-=(long n) { a -= n*(long)sizeof(T); return *this; }

  bool operator==(const hzg& o) const { return a == o.a; }
  bool operator!=(const hzg& o) const { return a != o.a; }
  bool operator< (const hzg& o) const { return a <  o.a; }
  bool operator> (const hzg& o) const { return a >  o.a; }
  bool operator<=(const hzg& o) const { return a <= o.a; }
  bool operator>=(const hzg& o) const { return a >= o.a; }

  // conversions yield the canonical address, never a frame address
  explicit operator unsigned char*() const { return a; }
  operator void*() const { return (void*)a; }
  // explicit bool: `if(p)` would otherwise be ambiguous
  explicit operator bool() const { return a != 0; }
  bool operator!() const { return a == 0; }
};

// Block copy between two heap objects: the only case where two translated
// references must live at once.  Page-wise through a bounce buffer, so source
// and destination can never be the same frame; the destination is marked dirty.
void kopiere(unsigned char* ziel, const unsigned char* quelle, unsigned long long n);

}
#endif
