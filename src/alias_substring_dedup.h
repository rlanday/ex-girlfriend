#ifndef ALIAS_SUBSTRING_DEDUP_H_
#define ALIAS_SUBSTRING_DEDUP_H_

namespace alias_substring_dedup {

// Teilstring (substring): is `n` (needle) a substring of `h` (haystack)?  Both
// already normalised by the caller.  Naive search on purpose: needles are at
// most a title length and the call happens once per entry per article, not
// per byte.
inline bool Teilstring(const unsigned char* n, int nn,
                       const unsigned char* h, int hn) {
  if (nn <= 0) return true;          // an empty key says nothing
  if (nn > hn) return false;
  for (int i = 0; i + nn <= hn; ++i) {
    int k = 0;
    while (k < nn && n[k] == h[i + k]) ++k;
    if (k == nn) return true;
  }
  return false;
}
}

namespace alias_substring_dedup {

// Struck entries per way, for the self-report.
inline unsigned long long g_syn = 0;
inline unsigned long long g_p1 = 0;
inline unsigned long long g_p2 = 0;
// Candidates seen per way, so the report can print a share and not just a
// count.
inline unsigned long long g_syn_ges = 0;
inline unsigned long long g_p1_ges = 0;
inline unsigned long long g_p2_ges = 0;

}

#endif
