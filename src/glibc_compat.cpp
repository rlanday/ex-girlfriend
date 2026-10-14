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
