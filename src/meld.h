#ifndef MELD_H_
#define MELD_H_
// Messages of the package form: the code E<n> on stderr (ERRORS.md).
#include <cstdio>
#include <cstdlib>
[[noreturn]] __attribute__((noinline, cold)) inline void msg_stop(int n) {
  fprintf(stderr, "E%d\n", n);
  exit(1);
}
__attribute__((noinline, cold)) inline void msg_note(int n) {
  fprintf(stderr, "E%d\n", n);
}
#endif  // MELD_H_
