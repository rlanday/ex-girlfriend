// L3 co-residency thrasher: continuously sweeps a buffer of a given footprint
// from a chosen core, to measure how much CCX-L3 another program can use before
// the transformer (running on a sibling core of the same CCX) slows down.
//
// Usage: l3_thrasher <mb> <core> [mode]
//   mb    - footprint in MiB (fractional ok, e.g. 0.5)
//   core  - cpu to pin to (pick a core in the same CCX as the transformer,
//           e.g. transformer on 42 -> thrasher on 43 or SMT sibling 106)
//   mode  - "read" (default): streaming reads; "rw": 50/50 read+write
// Runs until killed. Prints its own sweep rate once per second to stderr.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <sched.h>
#include <sys/mman.h>
#include <immintrin.h>

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <mb> <core> [read|rw]\n", argv[0]); return 1; }
    double mb = atof(argv[1]);
    int core = atoi(argv[2]);
    bool rw = argc > 3 && !strcmp(argv[3], "rw");

    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set)) { perror("affinity"); return 1; }

    size_t bytes = (size_t)(mb * 1024.0 * 1024.0);
    bytes = (bytes + 4095) & ~(size_t)4095;
    if (bytes < 4096) bytes = 4096;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(p, bytes, MADV_HUGEPAGE);
    memset(p, 1, bytes);

    volatile uint64_t sink = 0;
    __m256i acc = _mm256_setzero_si256();
    double t0 = now_s(); uint64_t sweeps = 0;
    for (;;) {
        char* b = (char*)p;
        if (!rw) {
            for (size_t i = 0; i + 128 <= bytes; i += 128) {
                acc = _mm256_or_si256(acc, _mm256_load_si256((__m256i*)(b + i)));
                acc = _mm256_or_si256(acc, _mm256_load_si256((__m256i*)(b + i + 64)));
            }
        } else {
            for (size_t i = 0; i + 128 <= bytes; i += 128) {
                __m256i v = _mm256_load_si256((__m256i*)(b + i));
                _mm256_store_si256((__m256i*)(b + i + 32), v);
                acc = _mm256_or_si256(acc, v);
            }
        }
        sweeps++;
        double t = now_s();
        if (t - t0 > 1.0) {
            sink += (uint64_t)_mm256_extract_epi64(acc, 0);
            fprintf(stderr, "[thrasher %.2f MiB core %d %s] %.2f GB/s\n",
                    bytes / 1048576.0, core, rw ? "rw" : "read",
                    sweeps * (double)bytes / (t - t0) / 1e9);
            t0 = t; sweeps = 0;
        }
    }
    return (int)(sink & 1);
}
