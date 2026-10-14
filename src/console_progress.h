#ifndef CONSOLE_PROGRESS_H_
#define CONSOLE_PROGRESS_H_

#include <stdio.h>
#include <time.h>
#include <string>
#include <unistd.h>
#include <sys/syscall.h>

namespace console_progress {

// Monotonic wall clock in seconds (clock_gettime through syscall()).
inline double Now() {
  timespec ts;
  syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

inline unsigned long long next_pos = 0;   // stream position of the next line

// Embedded blocks of the self-extraction and the build helper print nothing.
inline bool Quiet() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

// Called from the progress tick of Compress() and Decompress().
__attribute__((noinline)) inline void Tick(unsigned long long pos,
                                           unsigned long long total) {
  if (pos < next_pos || Quiet()) return;
  next_pos = pos + 500000;
  fprintf(stderr, "progress %.2f %%\n", 100.0 * pos / total);
}

// Console names of the steps (wording shared with the verbose build of the
// earlier submission).  An embedded block names itself when it starts, by its
// output file ("decoding article order: ", stdout, flushed), and its end line
// completes that line.  The long silent steps of the main run print
// "<step> ... " on stderr when they start and "<t> s." when they end, before
// anything else reaches the console, so no line is torn, neither in a
// terminal nor in separate stdout/stderr files.  The end line of the main run
// names the main stream.  All seconds are wall clock seconds from the start
// of main().  The build helper keeps its old end line.
inline bool Block() {
  return transformer::BlobForm();
}
inline double t_start = 0;         // start of main(), for the end line
inline bool phase_open = false;    // a step line waits for its duration
inline double phase_t0 = 0;
// First statement of main().
inline void Begin() { t_start = Now(); }
// Ends the open step line, if any.
inline void PhaseEnd() {
  if (!phase_open) return;
  phase_open = false;
  fprintf(stderr, "%.2f s.\n", Now() - phase_t0);
}
// Starts a step line (and ends the previous one).  Pending stdout goes out
// first, so the two streams keep their order.
inline void Phase(const char* what) {
  if (Quiet()) return;
  PhaseEnd();
  fflush(nullptr);
  fprintf(stderr, "%s ... ", what);
  fflush(stderr);
  phase_open = true;
  phase_t0 = Now();
}
// A section of the stream begins (or an event happens) at stream byte `at`.
inline void Mark(const char* what, unsigned long long at,
                 unsigned long long total) {
  if (!Quiet()) fprintf(stderr, "%s at %.2f %%\n", what, 100.0 * at / total);
}
// Start of a run, after the arguments.  enc: the run encodes; out: its output
// file.  An embedded block opens its line here.
inline void Start(bool enc, const std::string& out) {
  if (!Block() || out.size() < 2) return;
  printf("%scoding %s: ", enc ? "en" : "de",
         out[1] == 'n' ? "article order"
         : out[1] == 'h' ? "dictionary hints" : "dictionary");
  fflush(stdout);
}
// Right before the end line; returns the seconds of the run.
inline double End(bool enc) {
  PhaseEnd();   // the last step of a decompression ends here
  if (!Quiet()) printf("%scoding preprocessed enwik9: ", enc ? "en" : "de");
  return Now() - t_start;
}

}
#endif
