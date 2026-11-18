// glibc_compat.cpp - keeps the binary on symbol versions every glibc >= 2.34
// provides.
//
// glibc 2.38 headers redirect strtol, strtoll and sscanf to __isoc23_* under
// _GNU_SOURCE (which C++ always defines); the redirected symbols exist only
// from glibc 2.38 on, so a binary built on Ubuntu 24.04 did not start on
// Ubuntu 22.04 ("GLIBC_2.38 not found").  The C23 variants differ only in
// accepting a 0b/0B prefix for base 0 or 2 and %b in scanf; every call in
// this program uses base 8 or 10 and the formats %ld, %lx and %s, so the
// classic entry points give identical results.  This unit defines the
// __isoc23_* names and forwards them to those entry points.  It includes no
// C library header on purpose: the headers would apply the redirect here too.
#include <stdarg.h>

extern "C" {
long compat_strtol(const char* s, char** end, int base) __asm__("strtol");
long long compat_strtoll(const char* s, char** end, int base) __asm__("strtoll");
int compat_vsscanf(const char* s, const char* fmt, va_list ap) __asm__("vsscanf");

long __isoc23_strtol(const char* s, char** end, int base) {
  return compat_strtol(s, end, base);
}
long long __isoc23_strtoll(const char* s, char** end, int base) {
  return compat_strtoll(s, end, base);
}
int __isoc23_sscanf(const char* s, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int r = compat_vsscanf(s, fmt, ap);
  va_end(ap);
  return r;
}
}

// The five std::__throw_* the image calls end in a
// trap here instead of being imported from libstdc++.  They are reached only
// on a broken invariant (length beyond max_size, failed allocation, .at()
// out of range); the build has no exceptions, so libstdc++ would end in
// abort as well.  The version script keeps them local (no export).
namespace std {
[[noreturn]] void __throw_length_error(const char*) { __builtin_trap(); }
[[noreturn]] void __throw_bad_array_new_length() { __builtin_trap(); }
[[noreturn]] void __throw_bad_alloc() { __builtin_trap(); }
[[noreturn]] void __throw_logic_error(const char*) { __builtin_trap(); }
[[noreturn]] void __throw_out_of_range_fmt(const char*, ...) { __builtin_trap(); }
}  // namespace std

// Plain new/delete straight to malloc/free,
// one indirection and two imports less.  Not in builds whose allocator object
// defines them itself.
extern "C" void* malloc(decltype(sizeof 0));
extern "C" void free(void*);
void* operator new(decltype(sizeof 0) n) { return malloc(n); }
void operator delete(void* p) noexcept { free(p); }
