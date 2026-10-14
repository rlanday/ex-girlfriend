#include "ppmd_page.h"
#include <cstring>
#include <cerrno>
#include <ctime>

namespace ppmd_page {

Vorrat V;
static void rez_setze(unsigned long long n);
static void zonen_rechne(void);
int hilfs = 0;

// The mirror covers the wrap N-1 -> 0.  PPMd touches at most 128 units =
// 1,536 B at once, so two pages leave ample margin.
enum { SPIEGEL = 2 };

static int memfd(const char* name) {
#ifdef SYS_memfd_create
  return (int)syscall(SYS_memfd_create, name, 0u);
#else
  (void)name; errno = ENOSYS; return -1;
#endif
}

int start(unsigned long long heapgroesse, unsigned long long pufferbytes,
          const char* file_name) {
  memset(&V, 0, sizeof(V)); V.fd = -1;

  if (pufferbytes % SEITE) { fprintf(stderr,
      "ppmd page: pool %llu is not a multiple of 4096\n", pufferbytes);
    return 1; }
  const unsigned long long N = pufferbytes / SEITE;
  if (N < 4) { fprintf(stderr,
      "ppmd page: frame count %llu is smaller than 4\n", N); return 2; }

  // 1) canonical address space: reserved, PROT_NONE, NEVER touched.  Every
  //    missed access site faults here loudly.
  V.heap = (unsigned char*)mmap(NULL, heapgroesse, PROT_NONE,
             MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
  if (V.heap == MAP_FAILED) { perror("ppmd page: mmap address space"); return 3; }
  V.heapgr = heapgroesse;

  // 2) backing file (sparse; a never-written hole reads as zeros - exactly
  //    what a fresh MAP_SHARED mapping delivers).
  V.fd = open(file_name, O_RDWR|O_CREAT|O_TRUNC|O_NOATIME, 0600);
  if (V.fd < 0) V.fd = open(file_name, O_RDWR|O_CREAT|O_TRUNC, 0600);
  if (V.fd < 0) { perror("ppmd page: open backing file"); return 4; }
  if (ftruncate(V.fd, (off_t)heapgroesse) != 0) {
    perror("ppmd page: ftruncate"); return 5; }

  // 3) frame pool in a memfd, so its first pages can be mapped a second
  //    time behind the pool (ring mirror against the wrap).
  const int mf = memfd("ppmd_page");
  if (mf < 0) { perror("ppmd page: memfd_create"); return 6; }
  if (ftruncate(mf, (off_t)pufferbytes) != 0) {
    perror("ppmd page: ftruncate memfd"); return 7; }
  const unsigned long long gesamt = pufferbytes + (unsigned long long)SPIEGEL*SEITE;
  unsigned char* b = (unsigned char*)mmap(NULL, gesamt, PROT_NONE,
                       MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
  if (b == MAP_FAILED) { perror("ppmd page: mmap pool range"); return 8; }
  if (mmap(b, pufferbytes, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FIXED, mf, 0)
      == MAP_FAILED) { perror("ppmd page: mmap pool"); return 9; }
  if (mmap(b + pufferbytes, (size_t)SPIEGEL*SEITE, PROT_READ|PROT_WRITE,
           MAP_SHARED|MAP_FIXED, mf, 0) == MAP_FAILED) {
    perror("ppmd page: mmap mirror"); return 10; }
  V.mf = mf;                 // stays open: the pool has to be able to grow
  V.puffer = b; V.rahmen = N; V.maske = N - 1;
  // Placed by zonen_setze() from InitSubAllocator; until then plain direct.
  // The memfd stays open: the mirror is remapped when the ring moves.
  V.z_mf = mf;
  V.z_text = 0; V.z_tgroesse = 0; V.t_rez = 0; V.t_schieb = 0;
  V.z_basis = 0; V.z_ring = N; V.z_grenze = ~0ULL; V.z_versatz = 0;
  V.z_ursprung = ~0ULL; V.z_aus = 0;
  rez_setze(N);
  V.dyn_aus = 0; V.dyn_rest = DYN_TAKT; V.dyn_wechsel = 0;
  V.dyn_max_rahmen = N; V.dyn_min_rahmen = N;
  V.dyn_rss_spitze_kib = 0; V.dyn_umzug_bytes = 0;
  V.dyn_uhr = 0; V.dyn_halt_bytes = 0; V.dyn_halt_min = ~0ULL;
  V.dyn_wachs = 0; V.dyn_schrumpf = 0; V.dyn_pendel = 0;
  V.dyn_letzt_r[0] = V.dyn_letzt_r[1] = 0;
  V.dyn_letzt_t[0] = V.dyn_letzt_t[1] = 0;
  V.statmfd = open("/proc/self/statm", O_RDONLY|O_CLOEXEC);
  if (V.statmfd < 0) {
    V.dyn_aus = 1;
    fprintf(stderr, "ppmd page: /proc/self/statm unreadable, pool fixed at %llu MiB\n",
            (N*SEITE)>>20);
  }

  // 4) tables.  The pool is populated up front: its RSS is set from the
  //    first second on and changes only at a resize (dyn_setze).
  V.marke = (unsigned int*)malloc((size_t)N * sizeof(unsigned int));
  if (!V.marke) { fprintf(stderr, "ppmd page: no memory for the tag table\n");
    return 11; }
  for (unsigned long long i = 0; i < N; ++i) V.marke[i] = LEER;
  V.schmutz = (unsigned char*)calloc((size_t)N, 1);
  if (!V.schmutz) { fprintf(stderr, "ppmd page: no memory for the dirty table\n");
    return 11; }
  V.letzte_w = ~(unsigned long long)0;
  V.letzter_w = 0;
  V.letzte = ~(unsigned long long)0;   // empty
  V.letzter = 0;
  memset(V.puffer, 0, (size_t)pufferbytes);      // populate

  // 5) mirror check: the same physical frame under two addresses.  Without
  //    it objects at the wrap would span silently wrong.  VOLATILE is
  //    mandatory: both addresses have a constant distance in one object, and
  //    under PGO+LTO the compiler removed the second load.
  {
    volatile unsigned char* a0 = V.puffer;
    volatile unsigned char* a1 = V.puffer + pufferbytes;
    *a0 = 0xA5;
    if (*a1 != 0xA5) { fprintf(stderr, "ppmd page: ring mirror not mapped\n");
      return 12; }
    *a1 = 0x5A;
    if (*a0 != 0x5A) { fprintf(stderr, "ppmd page: ring mirror not coherent\n");
      return 13; }
    *a0 = 0;
  }
  return 0;
}

// Write back every DIRTY frame, re-map the pool at the new size, clear the
// bookkeeping.  The model cannot see this: the frames are only a cache of the
// backing file, and the file is complete afterwards.
static int dyn_setze(unsigned long long N_neu) {
  const unsigned long long alt_bytes = V.rahmen * SEITE;
  const unsigned long long alt_gesamt =
      alt_bytes + (unsigned long long)SPIEGEL * SEITE;
  for (unsigned long long i = 0; i < V.rahmen; ++i) {
    const unsigned int m = V.marke[i];
    if (m == LEER) continue;
    int schreiben = V.schmutz[i];
    if (!schreiben) { ++V.gespart; continue; }
    if (pwrite(V.fd, V.puffer + (i << 12), SEITE,
               (off_t)((unsigned long long)m << 12)) != (ssize_t)SEITE) {
      perror("ppmd page dyn: pwrite"); return -1; }
    ++V.rueckschreibung;
    V.dyn_umzug_bytes += SEITE;
  }
  const unsigned long long neu_bytes = N_neu * SEITE;
  const unsigned long long neu_gesamt =
      neu_bytes + (unsigned long long)SPIEGEL * SEITE;
  munmap(V.puffer, (size_t)alt_gesamt);
  if (ftruncate(V.mf, (off_t)neu_bytes) != 0) {
    perror("ppmd page dyn: ftruncate"); return -2; }
  unsigned char* b = (unsigned char*)mmap(NULL, neu_gesamt, PROT_NONE,
                       MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
  if (b == MAP_FAILED) { perror("ppmd page dyn: mmap space"); return -3; }
  if (mmap(b, neu_bytes, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FIXED, V.mf, 0)
      == MAP_FAILED) { perror("ppmd page dyn: mmap pool"); return -4; }
  if (mmap(b + neu_bytes, (size_t)SPIEGEL*SEITE, PROT_READ|PROT_WRITE,
           MAP_SHARED|MAP_FIXED, V.mf, 0) == MAP_FAILED) {
    perror("ppmd page dyn: mmap mirror"); return -5; }
  unsigned int* mk = (unsigned int*)realloc(V.marke,
                        (size_t)N_neu * sizeof(unsigned int));
  if (!mk) { fprintf(stderr, "ppmd page dyn: out of memory (tags)\n"); return -6; }
  V.marke = mk;
  for (unsigned long long i = 0; i < N_neu; ++i) V.marke[i] = LEER;
  unsigned char* sm = (unsigned char*)realloc(V.schmutz, (size_t)N_neu);
  if (!sm) { fprintf(stderr, "ppmd page dyn: out of memory (dirty)\n"); return -7; }
  V.schmutz = sm;
  memset(V.schmutz, 0, (size_t)N_neu);
  memset(b, 0, (size_t)neu_bytes);     // populate up front, as at setup
  V.puffer = b; V.rahmen = N_neu; V.maske = N_neu - 1;
  zonen_rechne();              // the pool is empty; sets the reciprocals too
  V.letzte = ~(unsigned long long)0; V.letzter = 0;
  V.letzte_w = ~(unsigned long long)0; V.letzter_w = 0;
  if (N_neu > V.dyn_max_rahmen) V.dyn_max_rahmen = N_neu;
  if (N_neu < V.dyn_min_rahmen) V.dyn_min_rahmen = N_neu;
  ++V.dyn_wechsel;
  return 0;
}

// Choose the pool size: how much room does the rest of the process (fremd_kib)
// leave under the upper bound minus the safety margin (deckel)?
static void dyn_pruefe(void) {
  char b[128];
  const ssize_t n = pread(V.statmfd, b, sizeof(b)-1, 0);
  if (n <= 0) { V.dyn_aus = 1;
    fprintf(stderr, "ppmd page: /proc/self/statm unreadable, pool fixed at %llu MiB\n",
            (V.rahmen*SEITE)>>20); return; }
  b[n] = 0;
  unsigned long long ges = 0, res = 0;
  if (sscanf(b, "%llu %llu", &ges, &res) != 2) return;
  const unsigned long long rss_kib = res * 4ULL;
  if (rss_kib > V.dyn_rss_spitze_kib) V.dyn_rss_spitze_kib = rss_kib;
  const unsigned long long eigen_kib = ((V.rahmen + SPIEGEL) * SEITE) >> 10;
  const unsigned long long fremd_kib =
      rss_kib > eigen_kib ? rss_kib - eigen_kib : 0ULL;
  const unsigned long long deckel = DYN_KLEMME_KIB - DYN_ABSTAND_KIB;
  unsigned long long frei_kib = deckel > fremd_kib ? deckel - fremd_kib : 0ULL;
  // Helper cap: never more frames than there have been
  // MISSES so far.  More frames cannot have saved anything up to here, so the
  // bound is exact, not a guess; it rises with every miss.  dyn_tick() sees
  // only its OWN process, the jury the sum of all of them: without the cap a
  // helper grew its pool while the parent was alive (measured: 9.662 GiB).
  // The main process passes 512 Ki misses (2 GiB) after about 820,000 model
  // bytes; a helper with a 520 KB payload stays at about 1,200 frames.  (The
  // HIGHEST touched page is no bound: PPMd fills its text area from the top.)
  // The cap holds for HELPER processes only (-c/-d/-C/-D, flag set in
  // main()): the main process keeps its start pool (it shrinks only when the
  // pool does not fit) instead of dropping to 64 MiB at the first poll and
  // growing back with its misses (measured: +0.8 M misses and six size
  // changes up to 8 % of the stream, +1.04 % wall clock at 9 %).
  if (hilfs)
  {
    const unsigned long long ber_kib = (V.fehlgriff * SEITE) >> 10;
    if (frei_kib > ber_kib) frei_kib = ber_kib;
  }
  const unsigned long long max_kib = (unsigned long long)4096 << 10;
  const unsigned long long min_kib =
      DYN_MIN_MIB << 10;
  // The upper bound limits the POOL (in frames), not the room; the pool is
  // compared with the room together with its two mirror pages.
  // The decision is NOT "what is the ideal size" -- recomputing that every
  // poll makes the control flutter, because our own pool is part of the RSS it
  // is computed from.  It is "does the pool we HAVE still fit, and is there
  // room for a larger one".
  unsigned long long N_ziel = V.rahmen;
  // Under the zone layout the miss curve is monotonic in N, so the
  // size moves in fine steps.  Shrink AT ONCE to the largest multiple of the
  // step that fits (the safe direction).  Grow only when the room for the next
  // step plus DYN_RESERVE_MIB has lasted DYN_HALTE model bytes without a
  // break, and then only to the largest step that fit with that reserve during
  // the whole wait (the smallest such target seen).  Every shrink restarts the
  // wait, so a growth can never follow a shrink within one hold time (counted
  // in dyn_pendel, MUST be 0).
  {
    V.dyn_uhr += DYN_TAKT;
    const unsigned long long stage = DYN_STUFE_MIB << 8;         // frames
    const unsigned long long reserve = DYN_RESERVE_MIB << 8;     // frames
    const unsigned long long n_min = min_kib >> 2;
    const unsigned long long obergrenze = max_kib >> 2;
    unsigned long long n_raum = frei_kib >> 2;                   // room, frames
    n_raum = n_raum > (unsigned long long)SPIEGEL ? n_raum - SPIEGEL : 0ULL;
    unsigned long long n_max = n_raum > obergrenze ? obergrenze : n_raum;
    unsigned long long z = n_max / stage * stage;                // fits now
    if (z < n_min) z = n_min;
    unsigned long long zw = n_raum > reserve ? n_raum - reserve : 0ULL;
    if (zw > obergrenze) zw = obergrenze;
    zw = zw / stage * stage;                          // fits with the reserve
    if (eigen_kib > frei_kib || V.rahmen > obergrenze) {
      N_ziel = z;                                     // does not fit: at once
      V.dyn_halt_bytes = 0; V.dyn_halt_min = ~0ULL;
    } else if (zw >= V.rahmen + stage) {              // room to grow: wait
      if (zw < V.dyn_halt_min) V.dyn_halt_min = zw;
      V.dyn_halt_bytes += DYN_TAKT;
      if (V.dyn_halt_bytes >= DYN_HALTE) {
        N_ziel = V.dyn_halt_min;
        V.dyn_halt_bytes = 0; V.dyn_halt_min = ~0ULL;
      }
    } else {
      V.dyn_halt_bytes = 0; V.dyn_halt_min = ~0ULL;
    }
    if (N_ziel != V.rahmen) {
      const int r = N_ziel > V.rahmen ? 1 : -1;
      if (r > 0 && V.dyn_letzt_r[0] < 0 && V.dyn_letzt_r[1] > 0
          && V.dyn_uhr - V.dyn_letzt_t[1] < DYN_HALTE) ++V.dyn_pendel;
      V.dyn_letzt_r[1] = V.dyn_letzt_r[0]; V.dyn_letzt_t[1] = V.dyn_letzt_t[0];
      V.dyn_letzt_r[0] = r;                V.dyn_letzt_t[0] = V.dyn_uhr;
      if (r > 0) ++V.dyn_wachs; else ++V.dyn_schrumpf;
    }
  }
  if (N_ziel == V.rahmen) return;
  const unsigned long long vorher = V.rahmen;
  if (dyn_setze(N_ziel) != 0) { V.dyn_aus = 1; return; }
  (void)vorher; (void)rss_kib; (void)fremd_kib;
}

void dyn_tick(void) {
  if (V.dyn_aus) return;
  if (--V.dyn_rest) return;
  V.dyn_rest = DYN_TAKT;
  dyn_pruefe();
}

// Page-in of ONE frame.  The previous resident is written back unless the
// dirty bit says it was never written.
static unsigned char* nachladen(unsigned long long page_index, unsigned long long slot) {
  unsigned char* rahmen = V.puffer + (slot << 12);
  const unsigned int alt = V.marke[slot];
  if (alt != LEER) {
    ++V.verdraengt;
    int schreiben = V.schmutz[slot];
    if (schreiben) {
      if (pwrite(V.fd, rahmen, SEITE, (off_t)((unsigned long long)alt << 12))
          != (ssize_t)SEITE) {
        perror("ppmd page: pwrite"); abort(); }
      ++V.rueckschreibung;
    } else {
      ++V.gespart;
    }
  }
  const ssize_t n = pread(V.fd, rahmen, SEITE, (off_t)(page_index << 12));
  if (n < 0) { perror("ppmd page: pread"); abort(); }
  if (n < (ssize_t)SEITE) memset(rahmen + n, 0, (size_t)(SEITE - n)); // hole
  V.schmutz[slot] = 0;
  // A cache entry must go if it pointed to THIS frame - it now carries
  // another page.  This is the only place where an entry can go stale.
  if (V.letzter == rahmen) V.letzte = ~(unsigned long long)0;
  if (V.letzter_w == rahmen) V.letzte_w = ~(unsigned long long)0;
  V.marke[slot] = (unsigned int)page_index;
  ++V.fehlgriff;
  return rahmen;
}

// Out-of-line translation: a call instead of the full fast path at access
// sites that are not hot.
__attribute__((noinline)) unsigned char* xl_ruf0(unsigned char* a, unsigned long long len) {
  return xl_kern<0,0>(a, len);
}
__attribute__((noinline)) unsigned char* xl_ruf1(unsigned char* a, unsigned long long len) {
  return xl_kern<1,0>(a, len);
}
__attribute__((noinline)) unsigned char* xl_ruf0w(unsigned char* a, unsigned long long len) {
  return xl_kern<0,1>(a, len);
}
__attribute__((noinline)) unsigned char* xl_ruf1w(unsigned char* a, unsigned long long len) {
  return xl_kern<1,1>(a, len);
}

static unsigned int leer_zelle = LEER;
static int freigegeben = 0;

void freigeben(void) {
  if (freigegeben) {
    fprintf(stderr, "ppmd page: release called twice - ignored\n");
    return;
  }
  freigegeben = 1;
  // Pool AND mirror are MAP_SHARED views of the same memfd inside one range.
  // Replacing the whole range by a PROT_NONE reservation drops every view in
  // one call and keeps the addresses reserved.
  const unsigned long long gesamt =
      (V.rahmen + (unsigned long long)SPIEGEL) * (unsigned long long)SEITE;
  if (mmap(V.puffer, (size_t)gesamt, PROT_NONE,
           MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE|MAP_FIXED, -1, 0) == MAP_FAILED)
    perror("ppmd page: release pool");
  // The dynamic pool keeps its memfd open so it can be resized; as long as a
  // descriptor is open the shmem pages survive the unmapping.  Closing it is
  // what actually returns them.  The statm sensor goes too.
  if (V.mf >= 0) { close(V.mf); V.mf = -1; }
  if (V.statmfd >= 0) { close(V.statmfd); V.statmfd = -1; }
  V.dyn_aus = 1;
  // Backing file: the caller removes the name ("ppm.page"); with the last
  // descriptor closed its page cache goes with it.
  if (V.fd >= 0) { close(V.fd); V.fd = -1; }
  free(V.marke);
  V.marke = &leer_zelle; V.maske = 0;
  // The slot is a real modulo, not `& maske`.  One frame with the
  // matching reciprocal maps every page to slot 0, i.e. to the single empty
  // cell.
  V.rahmen = 1; V.rez = (1ULL << 32) + 1ULL; V.schieb = 32u;
  // The same for the zone layout: no text area, a ring of one frame.
  V.z_text = 0; V.z_basis = 0; V.z_ring = 1; V.z_grenze = ~0ULL; V.z_versatz = 0;
  V.z_mf = -1;
  free(V.schmutz); V.schmutz = 0;
  V.letzte_w = ~(unsigned long long)0; V.letzter_w = 0;
  V.letzte = ~(unsigned long long)0; V.letzter = 0;
}

// Frame = page mod N for any N, without a division (multiply by rez, shift).
static inline unsigned long long rahmen_von(unsigned long long page_index) {
  const unsigned long long q = (page_index * V.rez) >> V.schieb;
  return page_index - q * V.z_ring;         // the units ring (= N while inactive)
}
// Set the reciprocal AND check the formula against the real modulo for every
// page of the heap.  A wrong mapping would change the model output; that is
// the one error that must not happen, so it is excluded, not assumed.
static void rez_setze(unsigned long long n) {
  unsigned int k = 0; while ((1ULL << k) < n) ++k;   // ceil(log2 n)
  V.z_ring = n;
  V.schieb = 32u + k;
  V.rez = (unsigned long long)((((unsigned __int128)1) << V.schieb) / n) + 1ULL;
  // the upper band adds its offset (< n) before the modulo
  const unsigned long long seiten = (V.heapgr >> 12) + n;
  for (unsigned long long s = 0; s < seiten; ++s) {
    if (rahmen_von(s) != s % n) {
      fprintf(stderr, "ppmd page: frame mapping wrong at page %llu"
              " (N=%llu, reciprocal=%llu, shift=%u)\n",
              s, n, V.rez, V.schieb);
      abort();
    }
  }
}
static inline unsigned long long slot_von(unsigned long long page_index) {
  if (page_index < V.z_text)                            // text: its own area [0, S)
    return page_index - ((page_index * V.t_rez) >> V.t_schieb) * V.z_tgroesse;
  if (page_index >= V.z_grenze) page_index += V.z_versatz;   // upper units band
  return V.z_basis + rahmen_von(page_index);            // units ring [S, N)
}

static unsigned long long zonen_textrahmen(unsigned long long N) {
  return (N / 16) / 16384 * 16384;
}
// Barrett for the text area, checked for every text page like rez_setze().
static void t_setze(unsigned long long S) {
  unsigned int k = 0; while ((1ULL << k) < S) ++k;
  V.t_schieb = 32u + k;
  V.t_rez = (unsigned long long)((((unsigned __int128)1) << V.t_schieb) / S) + 1ULL;
  V.z_tgroesse = S;
  for (unsigned long long s = 0; s < V.z_ursprung; ++s)
    if (s - ((s * V.t_rez) >> V.t_schieb) * S != s % S) {
      fprintf(stderr, "ppmd page: text mapping wrong at page %llu\n", s);
      abort();
    }
}
// The ring mirror: the two frames behind the pool show frames basis and
// basis+1, so an object at the wrap of the units ring stays contiguous.
static void spiegel_setze(unsigned long long basis) {
  unsigned char* hinter = V.puffer + V.rahmen * SEITE;
  if (mmap(hinter, (size_t)SPIEGEL * SEITE, PROT_READ|PROT_WRITE,
           MAP_SHARED|MAP_FIXED, V.z_mf, (off_t)(basis * SEITE)) == MAP_FAILED) {
    perror("ppmd page: mmap mirror"); abort(); }
  volatile unsigned char* a0 = V.puffer + basis * SEITE;
  volatile unsigned char* a1 = hinter;
  const unsigned char alt = *a0;
  *a0 = 0xA5;
  if (*a1 != 0xA5) { fprintf(stderr, "ppmd page: mirror not mapped\n"); abort(); }
  *a1 = alt;
  if (*a0 != alt) { fprintf(stderr, "ppmd page: mirror not coherent\n"); abort(); }
}
// Layout for the CURRENT N.  Must run with an empty pool.  Text area [0, S);
// units ring [S, N) of R frames; the upper band is shifted so that the last
// heap page lands directly below the LoUnit origin: ring position of E-1 =
// ring position of U minus one.  The boundary is the middle of the units
// area, far from both fronts.
static void zonen_rechne(void) {
  const unsigned long long N = V.rahmen;
  unsigned long long S = 0;
  if (!V.z_aus && V.z_ursprung != ~0ULL) S = zonen_textrahmen(N);
  const unsigned long long basis_alt = V.z_basis;
  if (V.z_aus || V.z_ursprung == ~0ULL) {        // plain direct mapping
    V.z_text = 0; V.z_basis = 0; V.z_grenze = ~0ULL; V.z_versatz = 0;
    rez_setze(N);
  } else {
    const unsigned long long E = (V.heapgr + 4095) >> 12, R = N - S;
    V.z_basis = S;
    V.z_text = S ? V.z_ursprung : 0;
    if (S) t_setze(S);
    rez_setze(R);
    V.z_versatz = (R - (E - V.z_ursprung) % R) % R;
    V.z_grenze = (V.z_ursprung + E) / 2;
  }
  if (V.z_basis != basis_alt || V.z_basis) spiegel_setze(V.z_basis);
}
// Write back every dirty frame and empty the pool (the frames are only a
// cache of the backing file).  Must run between two model bytes: no
// translated address may be alive.
static void umordnen(void) {
  for (unsigned long long i = 0; i < V.rahmen; ++i) {
    const unsigned int m = V.marke[i];
    if (m == LEER) continue;
    int schreiben = V.schmutz[i];
    V.schmutz[i] = 0;
    if (!schreiben) { ++V.gespart; V.marke[i] = LEER; continue; }
    if (pwrite(V.fd, V.puffer + (i << 12), SEITE,
               (off_t)((unsigned long long)m << 12)) != (ssize_t)SEITE) {
      perror("ppmd page: pwrite"); abort(); }
    ++V.rueckschreibung;
    V.marke[i] = LEER;
  }
  V.letzte = ~(unsigned long long)0; V.letzter = 0;
  V.letzte_w = ~(unsigned long long)0; V.letzter_w = 0;
}
void zonen_setze(unsigned long long unitsstart_off) {
  const unsigned long long u = unitsstart_off >> 12;
  if (u == V.z_ursprung) return;               // a restart: nothing moves
  umordnen();
  V.z_ursprung = u;
  zonen_rechne();
}
void zonen_aus(void) {
  if (V.z_versatz || V.z_basis) umordnen();
  V.z_aus = 1;
  zonen_rechne();
}

// Slow path.  Takes the CANONICAL address.
//   * outside the heap -> unchanged (BList in the model object, s0 on the
//     stack are handled by the same functions as heap blocks)
//   * otherwise: consult the tag table, page in if needed, and for an object
//     that spans a page boundary make the following page resident too - by
//     direct mapping it lands in the NEIGHBOURING frame, so the object is
//     contiguous; the mirror covers the wrap.
unsigned char* langsam(unsigned char* a, unsigned long long len) {
  const unsigned long long off = (unsigned long long)(a - V.heap);
  if (off >= V.heapgr) return a;      // normally caught inline already
  const unsigned long long page_index = off >> 12;
  const unsigned long long slot  = slot_von(page_index);
  unsigned char* rahmen = (V.marke[slot] == (unsigned int)page_index)
                        ? V.puffer + (slot << 12)
                        : nachladen(page_index, slot);
  const unsigned long long letzte = (off + len - 1) >> 12;
  // Cannot happen: text objects are single bytes, units lie above UnitsStart
  // until the units area is full, and the upper boundary lies in the gap
  // between the fronts (ppmd_PrepareByte leaves the zones 1 MiB before).  The
  // two halves would not be contiguous, so this must never be served.
  if (letzte != page_index
      && (page_index < V.z_text
          || (V.z_versatz && page_index < V.z_grenze && letzte >= V.z_grenze))) {
    fprintf(stderr, "ppmd page: object spans a band boundary\n");
    abort();
  }
  for (unsigned long long t = page_index + 1; t <= letzte; ++t) {
    const unsigned long long s2 = slot_von(t);
    if (V.marke[s2] != (unsigned int)t) nachladen(t, s2);
  }
  // Refill the cache - but NOT for a spanning object: the fast path cannot
  // re-check the following page.
  if (letzte == page_index) { V.letzte = page_index; V.letzter = rahmen; }
  return rahmen + (off & 4095);
}

// Slow path for a write: as above, then mark every frame the object touches
// and remember a single-page object's page in the write cache.
unsigned char* langsam_w(unsigned char* a, unsigned long long len) {
  const unsigned long long off = (unsigned long long)(a - V.heap);
  if (off >= V.heapgr) return a;
  const unsigned long long page_index  = off >> 12;
  const unsigned long long letzte = (off + len - 1) >> 12;
  unsigned char* r = langsam(a, len);
  for (unsigned long long t = page_index; t <= letzte; ++t)
    V.schmutz[slot_von(t)] = 1;
  if (letzte == page_index) { V.letzte_w = page_index; V.letzter_w = r - (off & 4095); }
  return r;
}

void schluss(void) {
  // Already released by freigeben(): nothing is left to close, and the
  // tripwire state (empty tag cell, closed descriptor) must stay.
  if (freigegeben) return;
  if (V.fd >= 0) close(V.fd);
  free(V.marke);
  free(V.schmutz);
  memset(&V, 0, sizeof(V)); V.fd = -1;
}

}
#include "ppmd_page_ptr.h"

namespace ppmd_page {

// Page-wise copy through a bounce buffer: source and destination may map to
// the same frame (frame = page mod N); without the buffer that would corrupt
// silently.
void kopiere(unsigned char* ziel, const unsigned char* quelle, unsigned long long n) {
  unsigned char zwischen[4096];
  while (n) {
    unsigned long long st = 4096 - ((unsigned long long)(quelle - V.heap) & 4095);
    if (st > n) st = n;
    unsigned long long zt = 4096 - ((unsigned long long)(ziel - V.heap) & 4095);
    if (st > zt) st = zt;
    memcpy(zwischen, xl_a(quelle, st), (size_t)st);
    memcpy(xl_w(ziel, st), zwischen, (size_t)st);
    quelle += st; ziel += st; n -= st;
  }
}

}
