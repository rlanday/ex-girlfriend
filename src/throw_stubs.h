#ifndef THROW_STUBS_H_
#define THROW_STUBS_H_
// Pre-included (-include) in the
// translation units without a PGO profile (runner/r1, dict_derive, the
// transformer units).  The std::__throw_* calls of the container templates (length
// beyond max_size, failed allocation) become a trap in place: the argument
// set-up and the message texts go.  These paths never run; the build has no
// exceptions, so libstdc++ would end in abort as well.  The profiled units
// keep their calls (the trap would change their pre-inlining and so their
// PGO hash); glibc_compat.cpp defines the functions for them, locally.
#include <bits/functexcept.h>

namespace std {
__attribute__((noreturn, always_inline)) inline void __throw_length_error(const char*) { __builtin_trap(); }
__attribute__((noreturn, always_inline)) inline void __throw_bad_array_new_length() { __builtin_trap(); }
__attribute__((noreturn, always_inline)) inline void __throw_bad_alloc() { __builtin_trap(); }
__attribute__((noreturn, always_inline)) inline void __throw_logic_error(const char*) { __builtin_trap(); }
}  // namespace std

#endif
