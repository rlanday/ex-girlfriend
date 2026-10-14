#ifndef PPMD_PAGE_H_
#define PPMD_PAGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
static const unsigned long long DYN_STUFE_MIB =
    (0) ? (0) : 64ULL;
static const unsigned long long DYN_HALTE =
    (0) ? (0) : (8ULL << 20);
// Room that must stay free on top of a grown pool: it absorbs what the rest of
// the process adds between two polls, so a growth is not undone by the next
// poll.  A constant, not a switch.
static const unsigned long long DYN_RESERVE_MIB = 32ULL;
// The jury clamp is max(ru_maxrss, process-tree sum) <= 10 GiB.
static const unsigned long long DYN_KLEMME_KIB =
    (0) ? (0) : 10485760ULL;
// Fixed safety margin below the clamp (the constant below).  It absorbs
// whatever the anonymous part adds between two polls; measured worst case with
// a coarse poll was 104 MiB.
static const unsigned long long DYN_ABSTAND_KIB =
    (262144) ? (262144) : 196608ULL;
// Poll cadence in model bytes, not in misses: a large pool produces few misses,
// so a miss-driven poll would go quiet exactly when it is needed.
static const unsigned long long DYN_TAKT =
    (0) ? (0) : 256ULL;
static const unsigned long long DYN_MIN_MIB =
    (0) ? (0) : 64ULL;

namespace ppmd_page {

enum { SEITE = 4096 };

struct Vorrat {
  // One-entry translation cache first: it is read at every dereference.
  // Fixed addresses let the compiler schedule both loads; indexed multi-entry
  // caches measured slower.
  unsigned long long letzte;   // page number of the cached page (~0 = none)
  unsigned char* letzter;      // its frame in the pool
  // Write cache: the last page translated for WRITING, already marked dirty.
  // A write through it needs no store at all - a store through a byte
  // pointer at every write would force the compiler to reload everything.
  unsigned long long letzte_w; // page number (~0 = none)
  unsigned char* letzter_w;    // its frame
  unsigned char* heap;        // canonical address space (PROT_NONE, untouched)
  unsigned long long heapgr;  // its size
  unsigned char* puffer;      // frame pool (N frames + mirror)
  unsigned long long rahmen;  // N, the frame count (any N >= 4)
  unsigned long long maske;   // N-1: written, not read (the mapping is a real modulo)
  unsigned long long rez;     // reciprocal for page mod N: floor(2^schieb / N) + 1
  unsigned int       schieb;  // 32 + ceil(log2 N)
  // Zone layout.  Inactive (= plain direct mapping): z_text 0, z_basis 0,
  // z_ring N, z_grenze ~0, z_versatz 0.
  unsigned long long z_text;     // pages below it form the text area (0 = none)
  unsigned long long z_tgroesse; // S, frames of the text area
  unsigned long long t_rez;      // Barrett for S
  unsigned int       t_schieb;
  unsigned long long z_basis;    // first frame of the units ring (= S)
  unsigned long long z_ring;     // R = N - S, frames of the units ring
  unsigned long long z_grenze;   // first page of the upper units band
  unsigned long long z_versatz;  // its offset in the ring (frames)
  unsigned long long z_ursprung; // page of UnitsStart (LoUnit origin)
  int                z_aus;      // 1 = switched back to direct for good
  int                z_mf;       // memfd of the pool (mirror remapping)
  unsigned int*  marke;       // marke[slot] = resident page number or LEER
  int fd;                     // backing file
  int mf;                     // memfd of the pool: kept open so it can grow
  int statmfd;                // /proc/self/statm, opened once
  int dyn_aus;                // 1 = sensor dead, size stays fixed
  unsigned long long dyn_rest;        // model bytes until the next poll
  unsigned long long dyn_wechsel;     // number of size changes
  unsigned long long dyn_max_rahmen, dyn_min_rahmen;
  unsigned long long dyn_rss_spitze_kib;
  unsigned long long dyn_umzug_bytes; // bytes written back by resizes
  unsigned long long dyn_uhr;         // model bytes seen by the polls
  unsigned long long dyn_halt_bytes;  // how long the room for growth lasted
  unsigned long long dyn_halt_min;    // smallest target during that time
  unsigned long long dyn_wachs, dyn_schrumpf;   // changes by direction
  // Oscillation counter: a growth whose previous change was a shrink that
  // itself followed a growth, all within one hold time.  MUST stay 0.
  unsigned long long dyn_pendel;
  int                dyn_letzt_r[2];            // directions of the last two changes
  unsigned long long dyn_letzt_t[2];            // and their model-byte clock
  // counters (self-report only)
  unsigned long long fehlgriff, rueckschreibung, uebersetzung, schnell;
  unsigned char* schmutz;          // dirty byte per frame
  unsigned long long verdraengt;   // evictions of occupied frames
  unsigned long long gespart;      // write-backs skipped (frame clean)
};

static const unsigned int LEER = 0xFFFFFFFFu;
extern Vorrat V;

// ---- setup -----------------------------------------------------------------
// Returns 0 on success.  Every failure is loud: there is no silent fallback.
int start(unsigned long long heapgroesse, unsigned long long pufferbytes,
          const char* file_name);
void schluss(void);
void bericht(unsigned long long modellbytes);

void freigeben(void);
// Called ONCE PER MODEL BYTE: one decrement and one compare.  Against 1.714
// translations per model byte that is not measurable.
void dyn_tick(void);
// Called from InitSubAllocator with the byte offset of UnitsStart: places the
// upper band.  Called between two model bytes (ppmd_PrepareByte) when a front
// comes near the boundary: write back, switch to direct mapping for good.
void zonen_setze(unsigned long long unitsstart_off);
void zonen_aus(void);
// 1 = this process is a helper (-c/-d/-C/-D, set in main()): only then does
// the dynamic pool stay below its miss count.
extern int hilfs;
enum { ZONEN_RAND = 1 << 20 };   // guard distance of the fronts, bytes

// ---- slow path (tag table, page-in, page-spanning objects) ------------------
// Takes the CANONICAL address; addresses outside the heap come back unchanged
// (BList lives in the model object, s0 on the stack).  langsam_w also marks
// every frame the object touches as dirty and fills the write cache.
unsigned char* langsam(unsigned char* a, unsigned long long len);
unsigned char* langsam_w(unsigned char* a, unsigned long long len);

template <int FREMD, int SCHREIB>
static inline __attribute__((always_inline))
unsigned char* xl_kern(unsigned char* a, unsigned long long len) {
  const unsigned long long off = (unsigned long long)(a - V.heap);
  // Out-of-heap exit stays inline: moving it into the slow path measured
  // 2.2 s slower per 1 M model bytes.
  if (FREMD && __builtin_expect(off >= V.heapgr, 0)) {
    return a;
  }
  const unsigned long long page_index   = off >> 12;
  const unsigned long long inseite = off & 4095;
  if (SCHREIB) {
    if (__builtin_expect(page_index == V.letzte_w && inseite + len <= 4096, 1)) {
      return V.letzter_w + inseite;
    }
    return langsam_w(a, len);
  }
  // inseite < 4096, so the sum cannot overflow.
  if (__builtin_expect(page_index == V.letzte && inseite + len <= 4096, 1)) {
    return V.letzter + inseite;
  }
  return langsam(a, len);
}

// out-of-line translation: one call instead of the full fast path per site
unsigned char* xl_ruf0(unsigned char* a, unsigned long long len);
unsigned char* xl_ruf1(unsigned char* a, unsigned long long len);
unsigned char* xl_ruf0w(unsigned char* a, unsigned long long len);
unsigned char* xl_ruf1w(unsigned char* a, unsigned long long len);
// HEISS = 1 inlines the fast path (used in the hot functions only).
// FREMD = 1: the address may lie outside the heap.  SCHREIB = 1: write access.
template <int FREMD, int HEISS, int SCHREIB>
static inline unsigned char* xl_t(unsigned char* a, unsigned long long len) {
  if (HEISS) return xl_kern<FREMD, SCHREIB>(a, len);
  if (SCHREIB) return FREMD ? xl_ruf1w(a, len) : xl_ruf0w(a, len);
  return FREMD ? xl_ruf1(a, len) : xl_ruf0(a, len);
}

// Raw translations for the few sites that reinterpret heap bytes (casts,
// block copies).  The address may lie outside the heap.  xl_w marks dirty.
static inline const unsigned char* xl_a(const unsigned char* a,
                                        unsigned long long len) {
  return xl_t<1,0,0>((unsigned char*)a, len);
}
static inline unsigned char* xl_w(unsigned char* a, unsigned long long len) {
  return xl_t<1,0,1>(a, len);
}

}
#endif
