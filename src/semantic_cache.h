#ifndef SEMANTIC_CACHE_H_
#define SEMANTIC_CACHE_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <type_traits>

#include "port_cfg.h"   // PORT_ENTRY: the geometry constants of the entry recipe

// Semantic cache.  The match model and the context maps find earlier text by
// exact byte contexts.  The cache adds a memory over the whole stream that is
// addressed by the internal state of the transformer: for a byte whose
// transformer prediction is uncertain it finds earlier stream positions where
// the transformer was in a similar state and reads which byte followed there.
// Coder and decoder build the same memory from coded bytes.
//
// When: only where the transformer row is live (not in the metablock, the
// census metablock or a gated zone).  A byte is looked up when the transformer
// gives its top-1 byte less than kQpE3 / 1000, and stored when it is complete
// and that probability was below kSpE3 / 1000.
//
// Key: the depth-mixed input of the transformer's last MLP (model_opt.cpp
// copies it for read rows), minus its running mean over the first kCen lookups
// below kCalqE3 / 1000, as int8, a fixed +-1 projection to kD dimensions, int8
// again; stored as int4 in a ring of 2^kNLog entries.
//
// Index: kL SimHash tables of 2^kB buckets, the bucket hash moved by the
// previous byte; a bucket keeps its kM newest entries with a 96-bit sketch.
// Lookup: per table the own bucket and kFlip buckets with one of the least
// certain hash bits flipped; candidates at least kGap stream bytes back,
// ordered by sketch distance, the kPre first re-ranked by the int4 cosine s;
// the kK best weigh exp((s - s_max) / T) with T = kTE3 / 1000.
//
// Followers: the neighbours of the last kG lookups below kGqE3 / 1000 go on
// like a match model (up to kWtMax bytes, while their continuation is in the
// 2^kWhLog-byte history) and join the fresh neighbours.
//
// Use: the cache log-odds of the bit is one more layer-0 input and the last
// end-mixer column; four columns of the BITLSTM row (log-odds, consistent
// mass, similarity, agreement with the transformer's top-1 byte); behind the
// floor stage an end stage (three logistic stages over 20 inputs, mixed per
// bit position) and the top-1 path stage (below) correct
// the probability the coder splits with.  Integer arithmetic from the key to
// the candidate order, doubles in the readouts and the end stages.  Memory:
// about 2.2 GB (key ring 1 GiB, bucket table 960 MiB, positions, bytes and
// history 112 MiB).

namespace semantic_cache {

// geometry (index and key ring of about 2.2 GB)
// PORT_ENTRY 1 (src/port_cfg.h) fixes the six constants of recipe "candidate A" here: kM 63, kFlip 15, kNLog 25,
// kWhLog 26, kPre 512, kDedupLog 10 (the research build resm_b25rp512 set the same six through the SEMCONST hook of
// its build script).  The release lines keep their form, so that hook still works on a PORT_ENTRY 0 tree.
constexpr int kD = 128;                // key dimensions
constexpr int kL = 16;                 // SimHash tables
constexpr int kB = 18;                 // bits per table
#if PORT_ENTRY
constexpr int kM{63};   // entry recipe A (release 15)
#else
constexpr int kM = 15;                 // newest entries kept per bucket
#endif
#if PORT_ENTRY
constexpr int kFlip{15};   // entry recipe A (release 3)
#else
constexpr int kFlip = 3;               // extra buckets per table
#endif
constexpr int kProbe = kFlip + 1;      // buckets read per table
#if PORT_ENTRY
constexpr int kNLog{25};   // entry recipe A (release 24)
#else
constexpr int kNLog = 24;              // key ring of 2^kNLog entries
#endif
constexpr uint32_t kN = 1u << kNLog;
constexpr int kK = 32;                 // neighbours of a lookup
constexpr int kSk = 96;                // sketch bits held in a slot
constexpr int kSkGen = 128;            // sketch hyperplanes drawn (generator order)
constexpr int kKB = kD / 2;            // key bytes per ring entry (int4 pairs)
constexpr int kPf = 8;                 // buckets prefetched ahead of the scan
constexpr int kTE3 = 20;               // temperature on the cosine x 1000
// thresholds on p(transformer top-1), x 1000
constexpr int kQpE3 = 990;             // look up below
constexpr int kSpE3 = 950;             // store below
constexpr int kGqE3 = 900;             // a lookup below starts a follower generation
constexpr int kC1E3 = 833;             // similarity classes: s_max < C1, < C2, >= C2
constexpr int kC2E3 = 927;
// content
constexpr uint32_t kCen = 1048576;     // lookups of the running key mean
constexpr int kGap = 1024;             // neighbours lie at least kGap stream bytes back
constexpr int kG = 4;                  // follower generations
constexpr int kWtMax = 24;             // bytes a neighbour is followed
#if PORT_ENTRY
constexpr int kWhLog{26};   // entry recipe A (release 25)
#else
constexpr int kWhLog = 25;             // stream history ring of 2^kWhLog bytes
#endif
constexpr uint64_t kHist = (uint64_t)1 << kWhLog;
// re-ranking depth, end-stage learning rate and key mean
#if PORT_ENTRY
constexpr int kPre{512};   // entry recipe A (release 128)
#else
constexpr int kPre = 128;              // candidates re-ranked by the cosine
#endif
constexpr int kLrE6 = 20000;           // AdaGrad base rate of the end stage in 1e-6
constexpr int kCalqE3 = 900;           // the running key mean counts lookups below this only
// end stage
constexpr int kUIn = 20;               // inputs of a stage
constexpr int kUC4 = 8 * 4 * 2 * 2 * 3;  // bit x similarity x agreement x word start x age class
constexpr int kUC5 = 8 * 256;            // bit position x previous byte
constexpr int kUC6 = 8 * 4 * 2 * 64;     // bit x similarity x agreement x previous-byte class x run class
constexpr uint32_t kUAge = 65536;        // near / far neighbours
constexpr int kSureE3 = 990;             // a bit with P >= this / 1000 for one value skips the end stage
constexpr int kCandMax = kL * kProbe * kM;
#if PORT_ENTRY
constexpr int kDedupLog{10};   // entry recipe A (release 9)
#else
constexpr int kDedupLog = 9;           // set of the kPre selected candidates
#endif
constexpr int kDedup = 1 << kDedupLog;
constexpr int kNb = kK * (1 + kG);       // neighbours read for one byte (fresh + followed)
static_assert(kDedup >= 2 * kPre, "candidate set");
static_assert(kNb <= 256, "part sums: group index in one byte");
static_assert(kKB * 2 == kD && kPre >= kK && kPre <= 512 && kK <= 32 && kGap < 4096 && kCalqE3 <= kQpE3 &&
              kSpE3 <= kQpE3, "form");

// one bucket slot: entry counter + 1 (0 = empty) and its 96-bit sketch
struct Slot {
  uint32_t e1;
  uint32_t s2;    // sketch bits 64..95
  uint64_t s01;   // sketch bits 0..63
};
static_assert(sizeof(Slot) == 16, "slot layout");

struct State {
  uint8_t* key;     // kN x kKB (ring of int4 keys)
  uint8_t* val;     // the byte that followed the key
  Slot* slot;       // kL x 2^kB x kM
  uint32_t* epos;   // kN: stream position of the entry's byte (the stream is shorter than 2^32)
  uint8_t* hist;    // kHist: stream bytes by position
  uint64_t nins;
  int8_t proj[kD][192];
  int8_t hyp[kL * kB][kD];
  int8_t skp[kSkGen][kD];
  // the entry of the current byte, inserted when the byte is complete
  bool pend;
  int8_t ck[kD];
  uint32_t c2;
  uint64_t c01;
  uint32_t chb[kL];
  // candidate buffers of one lookup
  uint8_t cand_hm[kCandMax];
  uint32_t cand_e[kCandMax];
  uint32_t ord_e[kCandMax];
  uint32_t dke[kDedup], dst[kDedup];
  uint32_t dstamp;
  // running key mean
  double kmu[192];
  uint32_t kcnt;
  float kmuf[192];   // the mean as float once kcnt == kCen (kmf: set)
  int kmf;
  // stream position of the byte about to be coded; entries stored before
  // stream position p (ring by p)
  uint64_t spos;
  uint32_t nh[4096];
  // follower generations: 0 = the neighbours of the last starting lookup,
  // 1..kG the earlier ones (ring, ghead = newest); position of the
  // neighbour's byte, its lookup position, similarity and alive flag
  int fn[1 + kG];
  uint64_t fq[1 + kG];
  uint64_t fvp[1 + kG][kK];
  double fsim[1 + kG][kK];
  uint8_t fal[1 + kG][kK];
  int ghead;
  // neighbours of the current byte (fresh, then followed): byte and weight
  // of the main readout (read by the layer-0 input and the end stage), their sum, the
  // similarity class of s_max and the agreement of the cache top-1 with the
  // transformer top-1
  int nn;
  uint8_t nbyte[kNb];
  double nw[kNb];
  double wsum;
  int cls, agr;
  // the main readout of the bit being predicted (SemPredBit; the layer-0
  // input, the BITLSTM columns and the end stage read it): consistent
  // neighbours, log-odds, consistent mass
  int rok;
  double rx, rc;
  double l10[2];   // f[10] for a consistent nearest neighbour with bit 0 / 1
  // BITLSTM columns: the similarity column of this byte and the first column
  // of the last Predict()
  float b_smax;
  float b_x0;
  // end stage: similarity relative to the s_max of the neighbour's own
  // lookup, age in stream bytes, nfr fresh ones in front
  int cell;   // -1 at the start
  double asim[kNb];
  uint32_t aage[kNb];
  int nfr;
  int topb, ws, gcl, mtop, ftop, ptop, pb;   // topb -1 at the start
  int wlen, ecl;            // letter run before this byte (cap 15); previous-byte class x run class
  // byte-level weights of the other readouts (one exp per neighbour and
  // byte); the readouts at temperature T use nw
  double hw[kNb], hws;        // sharp (T / 4)
  double ws_n, ws_f;    // weight sums of the near / far neighbours
  int n_n, n_f;
  // weight sums of the fresh and of the followed neighbours (wfg: computed)
  double wf, wg;
  int wfg;
  double uf[kUIn];
  int uc[3];   // -1 at the start
  double uz[3], ufz, ufe;   // ufe = exp(-ufz), 1 at the start
  int uo1[2], uo3[2];   // byte part of the cells c1 / c3 without / with a consistent neighbour
  // part sums (SemUtByte, once per byte): the neighbours grouped by byte
  // value (groups in byte order), the masses of each group (main, sharp,
  // fresh, followed, near, far, count); rm: the masses of the current bit by
  // its value (SemPredBit), read by the end stage
  uint8_t gb[kNb], gix[256];
  int ngr;
  double gm[kNb][8];
  double rm[8][2];
  double ufw[8][4], ufg[8][4];
  double u4w[kUC4][kUIn], u4g[kUC4][kUIn];
  double u5w[kUC5][kUIn], u5g[kUC5][kUIn];
  double u6w[kUC6][kUIn], u6g[kUC6][kUIn];
  // running hit rate of the cache top-1 and the top-1 of the last byte with
  // neighbours, judged when the next stream byte starts
  double hrate;   // 0.4 at the start
  int htopb;      // -1 at the start
  // layer-0 input: readout of the bit about to be predicted; ufresh = a
  // stream byte started, uact = the bits predicted now belong to it (a byte
  // injected behind it, observed without coding, gets no readout)
  double ux;
  int ufresh, uact;
  // Every field zero except the six set below: one memset instead of a
  // store per field (the state is 2.2 MB on the stack of the coder).
  State() {
    static_assert(std::is_trivially_copyable<State>::value, "State is set by memset");
    memset((void*)this, 0, sizeof(*this));
    cell = -1; topb = -1; htopb = -1;
    uc[0] = uc[1] = uc[2] = -1;
    ufe = 1.0; hrate = 0.4;
  }
};

}  // namespace semantic_cache

namespace cache_hit {

// Top-1 path stage of the semantic cache: a correction of the coder
// probability on the transformer's top-1 path, behind the end stage of the
// cache.  While the bits of the current byte follow the top-1 byte of the
// transformer row, the probability the coder splits with is replaced by
// sigma(w . f), oriented to the top-1 bit; f holds
// the coder's own log-odds, the transformer, fast-weight, fxcm and PPMd bit
// log-odds and the byte-level agreement of the match model, the PPMd row and
// the fast-weight row with the top-1 byte (list in predictor.cpp).  One weight
// row, learnt online after every corrected bit (AdaGrad on the log loss),
// starting at "coder unchanged".  Once the bits leave the top-1 path the
// coder runs unchanged.  Nothing upstream reads the output.

constexpr int kNF = 16;
constexpr int kLrE6 = 10000;   // AdaGrad base rate in 1e-6

struct State {
  int t1 = -1;          // top-1 byte of this byte's live transformer row, -1 = none
  int prev_t1 = -1;     // top-1 byte of the previous stream byte
  // of a live row (read by the semantic cache as well): its top-1, and with a
  // positive top-1 the match byte and length and the PPMd and fast-weight top-1
  int a1 = 0, mbyte = -1, mlen = 0, pa = -1, fa = -1;
  double fb[kNF];       // byte-level features of this byte
  int cell = -1;        // 0 = this bit was corrected, -1 = not
  int d = 0;            // top-1 bit of the corrected bit
  double z = 0.0;
  // double, not float: in float the many tiny steps on hits are rounded away while
  // the large steps on misses survive, and the weights drift to under-confidence
  // (on the first 2.9 % of the stream: +65 B with float state, -143 B with double)
  double f[kNF];
  double w[kNF];
  double g2[kNF];
  State() {
    for (int i = 0; i < kNF; ++i) { fb[i] = 0.0; f[i] = 0.0; w[i] = (i == 0) ? 1.0 : 0.0; g2[i] = 0.0; }
  }
};

}  // namespace cache_hit

namespace semantic_cache {
// Defined in the strict floating-point unit of the end mixer: log1p of glibc
// 2.39 (fdlibm, with the fused operations of its x86-64 FMA variant,
// bit-identical on every float input in (-1, 0) and (0, 1)) and log2 as
// log / ln 2 (detmath's log), so the cache does not depend on the maths
// library of the host.
double Log1p(double x);
double Log2(double x);
}  // namespace semantic_cache

#endif
