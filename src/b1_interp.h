#ifndef B1_INTERP_H_
#define B1_INTERP_H_

// E1 (work/h3_neif_meta_20261007/e1_spec.py): forecast of a pre-migration
// revid from its page id.  The MediaWiki 1.5 conversion assigned the revids
// below 17,000,000 in page-id order, so a B1 revid is predicted by linear
// interpolation between the already-decoded B1 pages nearest in page id
// (78 % exact, 92 % within +-1 on enwik9).  Shared by metablock_form::Pack /
// Unpack and metablock::Einheit::ZeileA; all three feed it in stream order,
// so encoder and decoder see the same set at every line.  Integer only.
//   q1/q2 = nearest page ids below/above p in the set (r1/r2 their revids)
//   both:   r1 + Fdiv((p - q1) * (r2 - r1), q2 - q1)
//   only q1: r1 + (p - q1);  only q2: r2 - (q2 - p);  none: 15898943

#include <stdio.h>
#include <stdlib.h>

#include <iterator>
#include <map>

#include "timestamp_algebra.h"

namespace b1_interp {

static const long long kB1Max = 17000000LL;      // B1 = revid below this
static const long long kLeer = 15898943LL;       // forecast of an empty set
static const unsigned char kMark = 0x2b;         // '+': ends the page id delta of a B1 line

static_assert(kMark != 0x09 && kMark != 0x0a && kMark != 0x2d &&
              (kMark < 0x30 || kMark > 0x39) && kMark < 0x80,
    "'+' must be neither TAB, LF, '-', a digit nor a base-100 byte");

class Interp {
 public:
  long long Prognose(long long p) const {
    std::map<long long, long long>::const_iterator hi = m_.lower_bound(p);
    const bool h2 = hi != m_.end();
    const bool h1 = hi != m_.begin();
    if (h1 && h2) {
      std::map<long long, long long>::const_iterator lo = std::prev(hi);
      const long long q1 = lo->first, r1 = lo->second;
      const long long q2 = hi->first, r2 = hi->second;
      return r1 + ts_algebra::Fdiv((p - q1) * (r2 - r1), q2 - q1);
    }
    if (h1) {
      std::map<long long, long long>::const_iterator lo = std::prev(hi);
      return lo->second + (p - lo->first);
    }
    if (h2) return hi->second - (hi->first - p);
    return kLeer;
  }
  void Add(long long p, long long r) {
    if (!m_.emplace(p, r).second) {
      fprintf(stderr, "\nb1 interp: page id %lld twice among the B1 lines\n", p);
      fflush(stderr);
      exit(1);
    }
  }
  size_t Size() const { return m_.size(); }
  void Clear() { std::map<long long, long long>().swap(m_); }

 private:
  std::map<long long, long long> m_;   // page id -> revid
};

}

#endif
