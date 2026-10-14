
#ifndef CENSUS_TRANSFORM_H_
#define CENSUS_TRANSFORM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__clang__)
#pragma float_control(precise, on, push)
#endif

// Census transform.  In the machine-generated place articles, numbers that
// follow from other numbers of the same page (population repeated in the
// intro, km2 -> mi2 conversions, land/water areas, densities, DMS coordinates,
// the last percentage of a sum of 100) are replaced by a control byte
// 0x01..0x05, and the decoder recomputes them.  Rules M1..M9 name these
// cases.  "The reference" is the Python implementation the rules were written
// in; the C code reproduces its float parsing, rounding and formatting exactly.

// ---------------------------------------------------------------- Basics --

#define CEN_EQ   ((char)0x01)   /* M1/M2/M3/M4/M5/M7/M8 (+ M9B) */
#define CEN_PCT  ((char)0x02)   /* M6 */
#define CEN_DENS ((char)0x03)   /* M9 */
#define CEN_PCTP ((char)0x04)   /* formula value + 1 ULP */
#define CEN_PCTM ((char)0x05)   /* formula value - 1 ULP */

#define CEN_D(c)  ((unsigned char)(c) >= '0' && (unsigned char)(c) <= '9')
#define CEN_WS(c) ((c) == ' ' || (c) == '\t' || (c) == '\n' || (c) == '\r' || (c) == '\f' || (c) == '\v')
#define CEN_DC(c) (CEN_D(c) || (c) == '.' || (c) == ',')   /* [\d.,] */
#define CEN_DK(c) (CEN_D(c) || (c) == ',')                 /* [\d,]  */
#define CEN_DP(c) (CEN_D(c) || (c) == '.')                 /* [\d.]  */

// Literal at position i? -> position after it, else -1.
#define CEN_AT(h, n, i, lit) (((i) >= 0 && (i) + (long)(sizeof(lit) - 1) <= (n) && memcmp((h) + (i), (lit), sizeof(lit) - 1) == 0) ? (i) + (long)(sizeof(lit) - 1) : -1L)

#define CEN_MAXG 10
typedef struct { long s[CEN_MAXG], e[CEN_MAXG]; } CenM;

static void cen_clear(CenM* m) {
  for (int i = 0; i < CEN_MAXG; ++i) { m->s[i] = -1; m->e[i] = -1; }
}

static long cen_find(const char* h, long n, long from, const char* nd, long m) {
  if (from < 0) from = 0;
  if (m <= 0) return from <= n ? from : -1;
  while (from + m <= n) {
    const char* p = (const char*)memchr(h + from, nd[0], (size_t)(n - from - m + 1));
    if (!p) return -1;
    long i = (long)(p - h);
    if (memcmp(h + i, nd, (size_t)m) == 0) return i;
    from = i + 1;
  }
  return -1;
}

static long cen_run_dc(const char* p, long n, long i) { while (i < n && CEN_DC(p[i])) ++i; return i; }
static long cen_run_dk(const char* p, long n, long i) { while (i < n && CEN_DK(p[i])) ++i; return i; }
static long cen_run_dp(const char* p, long n, long i) { while (i < n && CEN_DP(p[i])) ++i; return i; }
static long cen_run_d (const char* p, long n, long i) { while (i < n && CEN_D(p[i]))  ++i; return i; }
static long cen_run_ws(const char* p, long n, long i) { while (i < n && CEN_WS(p[i])) ++i; return i; }
// [\d,\x01]+ (decoder variant of 'phf')
static long cen_run_dk1(const char* p, long n, long i) {
  while (i < n && (CEN_DK(p[i]) || p[i] == CEN_EQ)) ++i;
  return i;
}

// =========================================================================
// Computation
// =========================================================================

// 1 square mile in square kilometers. volatile: the compiler must not turn
// it into a precomputed reciprocal (-freciprocal-math from -ffp-model=fast).
static const volatile double kCenKm2 = 2.589988110336;

static double cen_div(double a, double b) { return a / b; }

// float(bytes) of the reference: drop commas, then parse correctly rounded.
// returns 0 == Python None (ValueError).
static int cen_parse(const char* p, long n, double* out) {
  char buf[512];
  long m = 0, i = 0;
  int dots = 0, digs = 0;
  if (n <= 0) return 0;
  if (p[0] == '-' || p[0] == '+') { buf[m++] = p[0]; i = 1; }
  for (; i < n; ++i) {
    char c = p[i];
    if (c == ',') continue;
    if (c == '.') { if (++dots > 1) return 0; }
    else if (CEN_D(c)) ++digs;
    else return 0;
    if (m >= (long)sizeof(buf) - 1) return 0;
    buf[m++] = c;
  }
  if (digs == 0) return 0;
  buf[m] = 0;
  *out = strtod(buf, 0);
  return 1;
}

// '%.<nd>f' % x  (glibc and CPython both round correctly, ties-to-even)
static long cen_fmt_nd(double x, int nd, char* out) {
  // Length cap 40: census articles only contain small numbers, and cen_fmt1
  // still appends thousands commas (40 + 12 + 1 < 64 = buffer size of all
  // callers). Overlong values yield "" -> no replacement (harmless, the
  // encoder compares byte-wise anyway).
  int k = snprintf(out, 40, "%.*f", nd, x);
  if (k < 0 || k >= 40) { out[0] = 0; return 0; }
  return (long)k;
}

// Python round(x, nd) == strtod(dtoa(x, fixed, nd)) - exactly this.
static double cen_pyround(double x, int nd) {
  char buf[64];
  cen_fmt_nd(x, nd, buf);
  return strtod(buf, 0);
}

// Python round(x) (without ndigits): ties-to-even, as CPython float.__round__.
static double cen_pyround0(double x) {
  double r = round(x);                    /* half away from zero */
  if (fabs(x - r) == 0.5) r = 2.0 * round(x / 2.0);
  return r;
}

// fmt1: Rambot format - one decimal place, thousands commas.
static long cen_fmt1(double x, char* out) {
  char buf[64];
  long k = cen_fmt_nd(x, 1, buf);
  long dot = -1;
  for (long i = 0; i < k; ++i) if (buf[i] == '.') { dot = i; break; }
  if (dot < 0) { memcpy(out, buf, (size_t)k); out[k] = 0; return k; }   /* inf/nan */
  long o = 0, ips = 0;
  int neg = (buf[0] == '-');
  if (neg) { out[o++] = '-'; ips = 1; }
  long ipl = dot - ips;
  long first = ipl % 3; if (first == 0) first = 3;
  if (first > ipl) first = ipl;
  memcpy(out + o, buf + ips, (size_t)first); o += first;
  for (long i = ips + first; i < dot; i += 3) {
    out[o++] = ',';
    memcpy(out + o, buf + i, 3); o += 3;
  }
  memcpy(out + o, buf + dot, (size_t)(k - dot)); o += k - dot;
  out[o] = 0;
  return o;
}

// dms_from_dec of the reference
static void cen_dms(double dec, long* dd, long* mm, long* ss) {
  double t = fabs(dec);
  long d = (long)t;
  double f = t - (double)d;
  long m = (long)(f * 60.0);
  long s = (long)cen_pyround0(f * 3600.0 - (double)m * 60.0);
  if (s == 60) s = 59;
  *dd = d; *mm = m; *ss = s;
}

// sum(list_of_floats) as in CPython >= 3.12: the built-in sum() has summed
// floats since 3.12 with the compensated Neumaier variant (Improved
// Kahan-Babuska). Without it the last digit drifts and M6 replaces
// values other than the reference (e.g. 94.90+3.06+0.00+2.04 sums naively to
// 100.00000000000001, compensated to exactly 100.0). The start value is
// the int 0, which CPython adds exactly to the first element - so the
// compensation only begins at the second value.
static double cen_sum(const double* v, long k) {
  if (k <= 0) return 0.0;
  double f = v[0], c = 0.0;
  for (long i = 1; i < k; ++i) {
    double x = v[i];
    double t = f + x;
    if (fabs(f) >= fabs(x)) c += (f - t) + x;
    else                    c += (x - t) + f;
    f = t;
  }
  return f + c;
}

// M6: 100 - sum(...)
static double cen_rest100(const double* v, long k) { return 100.0 - cen_sum(v, k); }
// M9: back-calc residual -> value  and  value -> residual
static double cen_dens_val(double pred, long res) { return pred + (double)res / 10.0; }
static double cen_dens_res(double v, double pred) { return (v - pred) * 10.0; }
// M8: water = total - land
static double cen_minus(double a, double b) { return a - b; }

static double cen_wdens(double pdens, double hu, double pop) {
  return pdens * hu / pop;
}

static double cen_pct_val(double rest, long r, double scale) {
  return rest + (double)r / scale;
}

static long cen_fmt1_pm(double base, long r, char* out) {
  return cen_fmt1(cen_dens_val(cen_pyround(base, 1), r), out);
}
// Decoder: control byte -> r
static int cen_pm1_of(char c, long* r) {
  if (c == CEN_PCTP) { *r =  1; return 1; }
  if (c == CEN_PCTM) { *r = -1; return 1; }
  return 0;
}

// =========================================================================
// Patterns (encoder)
// =========================================================================
// In general: all character classes of the reference ([\d.,], [\d,], [\d.],
// \d) are disjoint from the literal characters that follow them. A greedy "+"
// therefore always runs to the end of the maximal run and cannot usefully be
// backed off - backtracking is impossible. A match can thus only start at the
// beginning of such a run, or at the position of a fixed literal; that is
// exactly what the searches below hang on.

// is_census(): "As of the [[census]]({{GR|n}})? of [[2000]]"  or
//              "As of the [[2000]] census"
static int cen_is_census(const char* p, long n) {
  long q = 0;
  while ((q = cen_find(p, n, q, "As of the [[census]]", 20)) >= 0) {
    long i = q + 20;
    long j = CEN_AT(p, n, i, "{{GR|");            /* optional group, greedy */
    if (j >= 0) {
      long d = cen_run_d(p, n, j);
      if (d > j) {
        long k = CEN_AT(p, n, d, "}}");
        if (k >= 0 && CEN_AT(p, n, k, " of [[2000]]") >= 0) return 1;
      }
    }
    if (CEN_AT(p, n, i, " of [[2000]]") >= 0) return 1;   /* Backtracking */
    ++q;
  }
  if (cen_find(p, n, 0, "As of the [[2000]] census", 25) >= 0) return 1;
  return 0;
}

// there (are|were) ([\d,]+) people, ([\d,]+) households, and ([\d,]+) families residing
static int cen_m_phf(const char* p, long n, CenM* m) {
  long q = 0;
  while ((q = cen_find(p, n, q, "there ", 6)) >= 0) {
    long i = q + 6;
    long j = CEN_AT(p, n, i, "are ");
    if (j < 0) j = CEN_AT(p, n, i, "were ");
    if (j >= 0) {
      long a1 = j, b1 = cen_run_dk(p, n, a1);
      if (b1 > a1) {
        long k1 = CEN_AT(p, n, b1, " people, ");
        if (k1 >= 0) {
          long a2 = k1, b2 = cen_run_dk(p, n, a2);
          if (b2 > a2) {
            long k2 = CEN_AT(p, n, b2, " households, and ");
            if (k2 >= 0) {
              long a3 = k2, b3 = cen_run_dk(p, n, a3);
              if (b3 > a3) {
                long k3 = CEN_AT(p, n, b3, " families residing");
                if (k3 >= 0) {
                  cen_clear(m);
                  m->s[0] = q;  m->e[0] = k3;
                  m->s[1] = i;  m->e[1] = j - 1;
                  m->s[2] = a1; m->e[2] = b1;
                  m->s[3] = a2; m->e[3] = b2;
                  m->s[4] = a3; m->e[4] = b3;
                  return 1;
                }
              }
            }
          }
        }
      }
    }
    ++q;
  }
  return 0;
}

// Decoder variant: ([\d,\x01]+) at group 2
static int cen_m_phf_d(const char* p, long n, CenM* m) {
  long q = 0;
  while ((q = cen_find(p, n, q, "there ", 6)) >= 0) {
    long i = q + 6;
    long j = CEN_AT(p, n, i, "are ");
    if (j < 0) j = CEN_AT(p, n, i, "were ");
    if (j >= 0) {
      long a1 = j, b1 = cen_run_dk1(p, n, a1);
      if (b1 > a1) {
        long k1 = CEN_AT(p, n, b1, " people, ");
        if (k1 >= 0) {
          long a2 = k1, b2 = cen_run_dk(p, n, a2);
          if (b2 > a2) {
            long k2 = CEN_AT(p, n, b2, " households, and ");
            if (k2 >= 0) {
              long a3 = k2, b3 = cen_run_dk(p, n, a3);
              if (b3 > a3) {
                long k3 = CEN_AT(p, n, b3, " families residing");
                if (k3 >= 0) {
                  cen_clear(m);
                  m->s[0] = q;  m->e[0] = k3;
                  m->s[1] = i;  m->e[1] = j - 1;
                  m->s[2] = a1; m->e[2] = b1;
                  m->s[3] = a2; m->e[3] = b2;
                  m->s[4] = a3; m->e[4] = b3;
                  return 1;
                }
              }
            }
          }
        }
      }
    }
    ++q;
  }
  return 0;
}

// total population of ([\d,]+)
static int cen_m_intro(const char* p, long n, CenM* m) {
  long q = 0;
  while ((q = cen_find(p, n, q, "total population of ", 20)) >= 0) {
    long a = q + 20, b = cen_run_dk(p, n, a);
    if (b > a) {
      cen_clear(m);
      m->s[0] = q; m->e[0] = b;
      m->s[1] = a; m->e[1] = b;
      return 1;
    }
    ++q;
  }
  return 0;
}

static int cen_m_hhrep(const char* p, long n, CenM* m, int marker) {
  long q = 0;
  while ((q = cen_find(p, n, q, "There ", 6)) >= 0) {
    long i = q + 6;
    long j = CEN_AT(p, n, i, "are ");
    if (j < 0) j = CEN_AT(p, n, i, "were ");
    if (j >= 0) {
      long a = j, b;
      if (marker) b = (a < n && p[a] == CEN_EQ) ? a + 1 : a;
      else        b = cen_run_dk(p, n, a);
      if (b > a) {
        long k = CEN_AT(p, n, b, " households out of which");
        if (k >= 0) {
          cen_clear(m);
          m->s[0] = q; m->e[0] = k;
          m->s[1] = i; m->e[1] = j - 1;
          m->s[2] = a; m->e[2] = b;
          return 1;
        }
      }
    }
    ++q;
  }
  return 0;
}

// \((-?[\d.]+), (-?[\d.]+)\)
static int cen_m_geo(const char* p, long n, CenM* m) {
  long q = 0;
  while (q < n) {
    const char* c = (const char*)memchr(p + q, '(', (size_t)(n - q));
    if (!c) return 0;
    q = (long)(c - p);
    long a1 = q + 1, i = a1;
    if (i < n && p[i] == '-') ++i;
    long b1 = cen_run_dp(p, n, i);
    if (b1 > i) {
      long k = CEN_AT(p, n, b1, ", ");
      if (k >= 0) {
        long a2 = k, i2 = a2;
        if (i2 < n && p[i2] == '-') ++i2;
        long b2 = cen_run_dp(p, n, i2);
        if (b2 > i2 && b2 < n && p[b2] == ')') {
          cen_clear(m);
          m->s[0] = q;  m->e[0] = b2 + 1;
          m->s[1] = a1; m->e[1] = b1;
          m->s[2] = a2; m->e[2] = b2;
          return 1;
        }
      }
    }
    ++q;
  }
  return 0;
}

// \{\{Mapit-US-cityscale\|(-?[\d.]+)\|(-?[\d.]+)\}\}                (enc)
// \{\{Mapit-US-cityscale\|(\x01|-?[\d.]+)\|(\x01|-?[\d.]+)\}\}      (dec)
static int cen_m_mapit(const char* p, long n, CenM* m, int marker) {
  long q = 0;
  while ((q = cen_find(p, n, q, "{{Mapit-US-cityscale|", 21)) >= 0) {
    long a1 = q + 21, b1;
    if (marker && a1 < n && p[a1] == CEN_EQ) {
      b1 = a1 + 1;
    } else {
      long i = a1;
      if (i < n && p[i] == '-') ++i;
      b1 = cen_run_dp(p, n, i);
      if (b1 <= i) b1 = -1;
    }
    if (b1 > 0 && b1 < n && p[b1] == '|') {
      long a2 = b1 + 1, b2;
      if (marker && a2 < n && p[a2] == CEN_EQ) {
        b2 = a2 + 1;
      } else {
        long i2 = a2;
        if (i2 < n && p[i2] == '-') ++i2;
        b2 = cen_run_dp(p, n, i2);
        if (b2 <= i2) b2 = -1;
      }
      if (b2 > 0) {
        long k = CEN_AT(p, n, b2, "}}");
        if (k >= 0) {
          cen_clear(m);
          m->s[0] = q;  m->e[0] = k;
          m->s[1] = a1; m->e[1] = b1;
          m->s[2] = a2; m->e[2] = b2;
          return 1;
        }
      }
    }
    ++q;
  }
  return 0;
}

static int cen_m_dms(const char* p, long n, CenM* m, int marker) {
  long q = 0;
  const char* anchor = "&amp;deg;";
  while ((q = cen_find(p, n, q, anchor, 9)) >= 0) {
    long g[7][2];
    long a = q;
    if (marker) { if (q >= 1 && p[q-1] == CEN_EQ) a = q - 1; else a = q; }
    else        { while (a > 0 && CEN_D(p[a-1])) --a; }
    if (a < q) {
      g[1][0] = a; g[1][1] = q;
      long i = q + 9;
      long ok = 1, cur = i;
      for (int gi = 2; gi <= 6 && ok; ++gi) {
        long as = cur, be;
        if (marker) be = (as < n && p[as] == CEN_EQ) ? as + 1 : as;
        else        be = cen_run_d(p, n, as);
        if (be <= as) { ok = 0; break; }
        g[gi][0] = as; g[gi][1] = be;
        long k;
        switch (gi) {
          case 2: k = CEN_AT(p, n, be, "'"); break;
          case 3: k = CEN_AT(p, n, be, "&quot; North, "); break;
          case 4: k = CEN_AT(p, n, be, "&amp;deg;"); break;
          case 5: k = CEN_AT(p, n, be, "'"); break;
          default: k = CEN_AT(p, n, be, "&quot; West ("); break;
        }
        if (k < 0) { ok = 0; break; }
        cur = k;
      }
      if (ok) {
        long a7 = cur, i7 = a7;
        if (i7 < n && p[i7] == '-') ++i7;
        long b7 = cen_run_dp(p, n, i7);
        if (b7 > i7) {
          long k7 = CEN_AT(p, n, b7, ", ");
          if (k7 >= 0) {
            long a8 = k7, i8 = a8;
            if (i8 < n && p[i8] == '-') ++i8;
            long b8 = cen_run_dp(p, n, i8);
            if (b8 > i8 && b8 < n && p[b8] == ')') {
              cen_clear(m);
              m->s[0] = a; m->e[0] = b8 + 1;
              for (int gi = 1; gi <= 6; ++gi) { m->s[gi] = g[gi][0]; m->e[gi] = g[gi][1]; }
              m->s[7] = a7; m->e[7] = b7;
              m->s[8] = a8; m->e[8] = b8;
              return 1;
            }
          }
        }
      }
    }
    ++q;
  }
  return 0;
}

// (<head>[^=]*?<tail>)  with re.S - lazy, must not skip a '='.
static int cen_m_span(const char* p, long n, const char* head, long hl,
                      const char* tail, long tl, CenM* m) {
  long q = 0;
  while ((q = cen_find(p, n, q, head, hl)) >= 0) {
    long j = q + hl;
    for (;;) {
      if (j + tl <= n && memcmp(p + j, tail, (size_t)tl) == 0) {
        cen_clear(m);
        m->s[0] = q; m->e[0] = j + tl;
        m->s[1] = q; m->e[1] = j + tl;
        return 1;
      }
      if (j >= n || p[j] == '=') break;
      ++j;
    }
    ++q;
  }
  return 0;
}

static int cen_m_racial(const char* p, long n, CenM* m) {
  return cen_m_span(p, n, "The racial makeup of the ", 25, "more races.", 11, m);
}
static int cen_m_age(const char* p, long n, CenM* m) {
  return cen_m_span(p, n, "the population is spread out with", 33, "age or older.", 13, m);
}

static long cen_grp_dc(const char* p, long n, long a, int marker) {
  if (marker && a < n && p[a] == CEN_EQ) return a + 1;
  if (marker && a < n && (p[a] == CEN_PCTP || p[a] == CEN_PCTM)) return a + 1;
  long b = cen_run_dc(p, n, a);
  return b > a ? b : -1;
}

static long cen_grp_dens(const char* p, long n, long a, int marker) {
  if (marker && a < n && p[a] == CEN_EQ) return a + 1;
  if (marker && a < n && (p[a] == CEN_PCTP || p[a] == CEN_PCTM)) return a + 1;
  if (marker && a < n && p[a] == CEN_DENS) {
    long i = a + 1;
    if (i < n && p[i] == '-') ++i;
    long d = cen_run_d(p, n, i);
    if (d > i && d < n && p[d] == ';') return d + 1;
    return -1;
  }
  long b = cen_run_dc(p, n, a);
  return b > a ? b : -1;
}

// total area of ([\d.,]+) (km2-unit) \(([\d.,]+) (mi2-unit)\)\.\s+
// ([\d.,]+) km&amp;sup2; \(([\d.,]+) mi&amp;sup2;\) of it (is|was) land and
// (?:([\d.,]+)\s*km&amp;sup2; \(([\d.,]+) mi&amp;sup2;\) of it (?:is|was)
//    (?:covered by )?water | none of it (?:is|was) covered by water)
static int cen_m_landw(const char* p, long n, CenM* m, int marker) {
  long q = 0;
  while ((q = cen_find(p, n, q, "total area of ", 14)) >= 0) {
    long a1 = q + 14, b1 = cen_run_dc(p, n, a1);
    if (b1 > a1 && b1 < n && p[b1] == ' ') {
      long a2 = b1 + 1, b2 = CEN_AT(p, n, a2, "[[square kilometer|km&amp;sup2;]]");
      if (b2 < 0) b2 = CEN_AT(p, n, a2, "km&amp;sup2;");
      if (b2 > 0) {
        long k = CEN_AT(p, n, b2, " (");
        if (k >= 0) {
          long a3 = k, b3 = cen_grp_dc(p, n, a3, marker);
          if (b3 > 0 && b3 < n && p[b3] == ' ') {
            long a4 = b3 + 1, b4 = CEN_AT(p, n, a4, "[[square mile|mi&amp;sup2;]]");
            if (b4 < 0) b4 = CEN_AT(p, n, a4, "mi&amp;sup2;");
            if (b4 > 0) {
              long k2 = CEN_AT(p, n, b4, ").");
              if (k2 >= 0) {
                long w = cen_run_ws(p, n, k2);
                if (w > k2) {
                  long a5 = w, b5 = cen_grp_dc(p, n, a5, marker);
                  if (b5 > 0) {
                    long k3 = CEN_AT(p, n, b5, " km&amp;sup2; (");
                    if (k3 >= 0) {
                      long a6 = k3, b6 = cen_grp_dc(p, n, a6, marker);
                      if (b6 > 0) {
                        long k4 = CEN_AT(p, n, b6, " mi&amp;sup2;) of it ");
                        if (k4 >= 0) {
                          long a7 = k4, b7 = CEN_AT(p, n, a7, "is");
                          if (b7 < 0) b7 = CEN_AT(p, n, a7, "was");
                          if (b7 > 0) {
                            long k5 = CEN_AT(p, n, b7, " land and ");
                            if (k5 >= 0) {
                              /* alternative 1: water area named */
                              long a8 = k5, b8 = cen_grp_dc(p, n, a8, marker);
                              long done = -1, e8s = -1, e8e = -1, e9s = -1, e9e = -1;
                              if (b8 > 0) {
                                long w2 = cen_run_ws(p, n, b8);
                                long k6 = CEN_AT(p, n, w2, "km&amp;sup2; (");
                                if (k6 >= 0) {
                                  long a9 = k6, b9 = cen_grp_dc(p, n, a9, marker);
                                  if (b9 > 0) {
                                    long k7 = CEN_AT(p, n, b9, " mi&amp;sup2;) of it ");
                                    if (k7 >= 0) {
                                      long k8 = CEN_AT(p, n, k7, "is ");
                                      if (k8 < 0) k8 = CEN_AT(p, n, k7, "was ");
                                      if (k8 > 0) {
                                        long k9 = CEN_AT(p, n, k8, "covered by ");
                                        if (k9 < 0) k9 = k8;
                                        long k10 = CEN_AT(p, n, k9, "water");
                                        if (k10 >= 0) {
                                          done = k10;
                                          e8s = a8; e8e = b8; e9s = a9; e9e = b9;
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                              if (done < 0) {
                                /* Alternative 2: "none of it is covered by water" */
                                long t = CEN_AT(p, n, k5, "none of it ");
                                if (t >= 0) {
                                  long t2 = CEN_AT(p, n, t, "is");
                                  if (t2 < 0) t2 = CEN_AT(p, n, t, "was");
                                  if (t2 > 0) {
                                    long t3 = CEN_AT(p, n, t2, " covered by water");
                                    if (t3 >= 0) done = t3;
                                  }
                                }
                              }
                              if (done >= 0) {
                                cen_clear(m);
                                m->s[0] = q;  m->e[0] = done;
                                m->s[1] = a1; m->e[1] = b1;
                                m->s[2] = a2; m->e[2] = b2;
                                m->s[3] = a3; m->e[3] = b3;
                                m->s[4] = a4; m->e[4] = b4;
                                m->s[5] = a5; m->e[5] = b5;
                                m->s[6] = a6; m->e[6] = b6;
                                m->s[7] = a7; m->e[7] = b7;
                                m->s[8] = e8s; m->e[8] = e8e;
                                m->s[9] = e9s; m->e[9] = e9e;
                                return 1;
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    ++q;
  }
  return 0;
}

// population density\]\] (is|was) (dens)/km&amp;sup2; \((dens)/mi&amp;sup2;\)
static int cen_m_popdens(const char* p, long n, CenM* m, int marker) {
  long q = 0;
  while ((q = cen_find(p, n, q, "population density]] ", 21)) >= 0) {
    long a1 = q + 21, b1 = CEN_AT(p, n, a1, "is ");
    if (b1 < 0) b1 = CEN_AT(p, n, a1, "was ");
    if (b1 > 0) {
      long a2 = b1, b2 = cen_grp_dens(p, n, a2, marker);
      if (b2 > 0) {
        long k = CEN_AT(p, n, b2, "/km&amp;sup2; (");
        if (k >= 0) {
          long a3 = k, b3 = cen_grp_dens(p, n, a3, marker);
          if (b3 > 0) {
            long k2 = CEN_AT(p, n, b3, "/mi&amp;sup2;)");
            if (k2 >= 0) {
              cen_clear(m);
              m->s[0] = q;  m->e[0] = k2;
              m->s[1] = a1; m->e[1] = b1 - 1;
              m->s[2] = a2; m->e[2] = b2;
              m->s[3] = a3; m->e[3] = b3;
              return 1;
            }
          }
        }
      }
    }
    ++q;
  }
  return 0;
}

// ([\d,]+) housing units at an average density of (dens)/km&amp;sup2; \((dens)/mi&amp;sup2;\)
static int cen_m_housing(const char* p, long n, CenM* m, int marker) {
  const char* an = "housing units at an average density of ";
  long al = 39, q = 0;
  while ((q = cen_find(p, n, q, an, al)) >= 0) {
    if (q >= 2 && p[q-1] == ' ' && CEN_DK(p[q-2])) {
      long a1 = q - 1;
      while (a1 > 0 && CEN_DK(p[a1-1])) --a1;
      long b1 = q - 1;
      long a2 = q + al, b2 = cen_grp_dens(p, n, a2, marker);
      if (b2 > 0) {
        long k = CEN_AT(p, n, b2, "/km&amp;sup2; (");
        if (k >= 0) {
          long a3 = k, b3 = cen_grp_dens(p, n, a3, marker);
          if (b3 > 0) {
            long k2 = CEN_AT(p, n, b3, "/mi&amp;sup2;)");
            if (k2 >= 0) {
              cen_clear(m);
              m->s[0] = a1; m->e[0] = k2;
              m->s[1] = a1; m->e[1] = b1;
              m->s[2] = a2; m->e[2] = b2;
              m->s[3] = a3; m->e[3] = b3;
              return 1;
            }
          }
        }
      }
    }
    ++q;
  }
  return 0;
}

// ([\d.,]+) (km2) \(([\d.,]+)(/?)( ?)(mi2)          (enc, finditer)
// ([\d.,]+) (km2) \(\x01(/?)( ?)(mi2)               (dec, finditer)
// returns: start position of the match or -1; *end = m.end().
static long cen_m_kmmi(const char* p, long n, long pos, CenM* m, int marker) {
  for (long i = pos; i < n; ++i) {
    if (!CEN_DC(p[i])) continue;
    if (i > pos && CEN_DC(p[i-1])) continue;    /* run starts only */
    long b1 = cen_run_dc(p, n, i);
    if (b1 >= n || p[b1] != ' ') continue;
    long a2 = b1 + 1, b2 = CEN_AT(p, n, a2, "km&amp;sup2;");
    if (b2 < 0) b2 = CEN_AT(p, n, a2, "[[square kilometer|km&amp;sup2;]]");
    if (b2 < 0) continue;
    long k = CEN_AT(p, n, b2, " (");
    if (k < 0) continue;
    long a3 = k, b3;
    if (marker) { if (k < n && (p[k] == CEN_EQ || p[k] == CEN_PCTP || p[k] == CEN_PCTM)) b3 = k + 1; else continue; }
    else { b3 = cen_run_dc(p, n, a3); if (b3 <= a3) continue; }
    long a4 = b3, b4 = (b3 < n && p[b3] == '/') ? b3 + 1 : b3;
    long a5 = b4, b5 = (b4 < n && p[b4] == ' ') ? b4 + 1 : b4;
    long a6 = b5, b6 = CEN_AT(p, n, a6, "mi&amp;sup2;");
    if (b6 < 0) b6 = CEN_AT(p, n, a6, "[[square mile|mi&amp;sup2;]]");
    if (b6 < 0) continue;
    cen_clear(m);
    m->s[0] = i;  m->e[0] = b6;
    m->s[1] = i;  m->e[1] = b1;
    m->s[2] = a2; m->e[2] = b2;
    m->s[3] = a3; m->e[3] = b3;
    m->s[4] = a4; m->e[4] = b4;
    m->s[5] = a5; m->e[5] = b5;
    m->s[6] = a6; m->e[6] = b6;
    return i;
  }
  return -1;
}

static int cen_is_pctmark(char c) {
  return c == CEN_PCT || c == CEN_PCTP || c == CEN_PCTM;
}
static long cen_m_pct(const char* p, long from, long endp, long* ge, int marker) {
  long i = from;
  while (i < endp) {
    if (marker && cen_is_pctmark(p[i])) {
      if (i + 1 < endp && p[i+1] == '%') { *ge = i + 1; return i; }
      ++i; continue;
    }
    if (!CEN_D(p[i])) { ++i; continue; }
    long r1 = cen_run_d(p, endp, i);
    if (r1 < endp && p[r1] == '.' && r1 + 1 < endp && CEN_D(p[r1+1])) {
      long r2 = cen_run_d(p, endp, r1 + 1);
      if (r2 < endp && p[r2] == '%') { *ge = r2; return i; }
    }
    i = r1;                     /* whole digit run fails together */
  }
  return -1;
}

// =========================================================================
// Buffers
// =========================================================================
typedef struct { long s, e; int nl; char nb[80]; } CenSub;

static unsigned char* cen_used = 0;   static long cen_usedcap = 0;
static CenSub*        cen_subs = 0;   static long cen_subcap = 0;
static char*          cen_ob   = 0;   static long cen_obcap = 0;

static int cen_grow(void** p, long* cap, long need, long elem) {
  if (*cap >= need) return 1;
  long nc = *cap ? *cap : 1024;
  while (nc < need) nc *= 2;
  void* q = realloc(*p, (size_t)(nc * elem));
  if (!q) return 0;
  *p = q; *cap = nc;
  return 1;
}

static int cen_subcmp(const void* a, const void* b) {
  const CenSub* x = (const CenSub*)a;
  const CenSub* y = (const CenSub*)b;
  if (x->s != y->s) return x->s < y->s ? -1 : 1;
  if (x->e != y->e) return x->e < y->e ? -1 : 1;
  return 0;
}

// =========================================================================
// Encoder
// =========================================================================
static long cen_nsub = 0;

static int cen_try_sub(long s, long e, const char* nb, int nl) {
  if (cen_used[s] || cen_used[e-1]) return 0;
  if (!cen_grow((void**)&cen_subs, &cen_subcap, cen_nsub + 1, (long)sizeof(CenSub))) return 0;
  cen_subs[cen_nsub].s = s;
  cen_subs[cen_nsub].e = e;
  cen_subs[cen_nsub].nl = nl;
  memcpy(cen_subs[cen_nsub].nb, nb, (size_t)nl);
  ++cen_nsub;
  for (long i = s; i < e; ++i) cen_used[i] = 1;
  return 1;
}

static int cen_pm1_sub(const char* p, long ms, long me, double base) {
  if (me - ms < 2) return 0;
  if (!(base > -1e15 && base < 1e15)) return 0;      /* NaN/inf/absurd */
  char b[80];
  for (long r = 1; r >= -1; r -= 2) {
    long bl = cen_fmt1_pm(base, r, b);
    if (bl == me - ms && memcmp(b, p + ms, (size_t)bl) == 0) {
      char mk[1]; mk[0] = (r == 1) ? CEN_PCTP : CEN_PCTM;
      return cen_try_sub(ms, me, mk, 1);
    }
  }
  return 0;
}

static int cen_wdens_sub(const char* p, long ms, long me,
                         double pdens, double hu, double pop) {
  if (!(pdens > 0.0) || !(hu > 0.0) || !(pop > 0.0)) return 0;
  if (me - ms < 2) return 0;
  char b[64];
  long bl = cen_fmt1(cen_wdens(pdens, hu, pop), b);
  if (bl != me - ms || memcmp(b, p + ms, (size_t)bl) != 0) {
    if (cen_pm1_sub(p, ms, me, cen_wdens(pdens, hu, pop))) {return 1; }
    return 0;
  }
  char one[1]; one[0] = CEN_EQ;
  if (!cen_try_sub(ms, me, one, 1)) return 0;
  return 1;
}

// M9 helper function (dens_sub of the reference)
static void cen_dens_sub(const char* p, long ms, long me, double base) {
  if (!(base > 0.0)) return;
  double v;
  if (!cen_parse(p + ms, me - ms, &v)) return;
  double pred = cen_pyround(base, 1);
  double rr = cen_pyround0(cen_dens_res(v, pred));
  if (!(rr >= -2147483000.0 && rr <= 2147483000.0)) return;
  long res = (long)rr;
  if (res > 5000 || res < -5000) return;
  char enc[72];
  int nl = 0;
  enc[nl++] = CEN_DENS;
  nl += snprintf(enc + nl, 32, "%ld", res);
  enc[nl++] = ';';
  if (nl >= (int)(me - ms)) return;
  char chk[64];
  long cl = cen_fmt1(cen_dens_val(pred, res), chk);
  if (cl != me - ms || memcmp(chk, p + ms, (size_t)cl) != 0) return;
  cen_try_sub(ms, me, enc, nl);
}

static void cen_dens_ladder(const char* p, long ms, long me, double base) {
  if (!(base > 0.0)) return;
  char b[80];
  long bl = cen_fmt1(base, b);
  if (bl == me - ms && memcmp(b, p + ms, (size_t)bl) == 0) {
    if (me - ms >= 2) { char one[1]; one[0] = CEN_EQ; cen_try_sub(ms, me, one, 1); return; }
  }
  if (cen_pm1_sub(p, ms, me, base)) return;
  cen_dens_sub(p, ms, me, base);
}

// Returns the transformed page in cen_ob (length *outn). 0 == unchanged.
static int cen_encode_page(const char* p, long n, long* outn) {
  CenM phf, in_, hh, lw, kmmi, pd, ho, geo, mp, dms, rac, age;
  int has_phf, has_in, has_hh, has_lw, has_pd, has_ho, has_geo, has_mp, has_dms;
  char one[1]; one[0] = CEN_EQ;
  char two[1]; two[0] = CEN_PCT;

  if (!cen_grow((void**)&cen_used, &cen_usedcap, n + 1, 1)) return 0;
  memset(cen_used, 0, (size_t)n + 1);
  cen_nsub = 0;

  has_phf = cen_m_phf(p, n, &phf);
  has_in  = cen_m_intro(p, n, &in_);
  has_hh  = cen_m_hhrep(p, n, &hh, 0);
  has_lw  = cen_m_landw(p, n, &lw, 0);

  /* M3: people == intro population */
  if (has_phf && has_in && in_.s[0] < phf.s[0] &&
      in_.e[1] - in_.s[1] == phf.e[2] - phf.s[2] &&
      memcmp(p + in_.s[1], p + phf.s[2], (size_t)(in_.e[1] - in_.s[1])) == 0) {
    cen_try_sub(phf.s[2], phf.e[2], one, 1);
  }
  /* M2: households repetition */
  if (has_phf && has_hh && hh.s[0] > phf.e[0] &&
      hh.e[2] - hh.s[2] == phf.e[3] - phf.s[3] &&
      memcmp(p + hh.s[2], p + phf.s[3], (size_t)(hh.e[2] - hh.s[2])) == 0) {
    cen_try_sub(hh.s[2], hh.e[2], one, 1);
  }
  /* M7/M8: land/water from the total area */
  if (has_lw) {
    if (lw.s[8] < 0) {
      if (lw.e[5] - lw.s[5] == lw.e[1] - lw.s[1] &&
          memcmp(p + lw.s[5], p + lw.s[1], (size_t)(lw.e[5] - lw.s[5])) == 0) {
        cen_try_sub(lw.s[5], lw.e[5], one, 1);
      }
    } else {
      double tk, lk, wk;
      if (cen_parse(p + lw.s[1], lw.e[1] - lw.s[1], &tk) &&
          cen_parse(p + lw.s[5], lw.e[5] - lw.s[5], &lk) &&
          cen_parse(p + lw.s[8], lw.e[8] - lw.s[8], &wk)) {
        char b[64];
        long bl = cen_fmt1(cen_minus(tk, lk), b);
        if (bl == lw.e[8] - lw.s[8] && memcmp(b, p + lw.s[8], (size_t)bl) == 0) {
          cen_try_sub(lw.s[8], lw.e[8], one, 1);
        }
        else if (cen_pm1_sub(p, lw.s[8], lw.e[8], cen_minus(tk, lk))) {}
      }
    }
  }
  /* M1: all km->mi pairs */
  {
    long pos = 0, st;
    while ((st = cen_m_kmmi(p, n, pos, &kmmi, 0)) >= 0) {
      pos = kmmi.e[0];
      double km;
      if (cen_parse(p + kmmi.s[1], kmmi.e[1] - kmmi.s[1], &km) &&
          kmmi.e[4] == kmmi.s[4]) {                    /* group 4 != '/' */
        char b[64];
        long bl = cen_fmt1(cen_div(km, kCenKm2), b);
        if (bl == kmmi.e[3] - kmmi.s[3] &&
            memcmp(b, p + kmmi.s[3], (size_t)bl) == 0) {
          cen_try_sub(kmmi.s[3], kmmi.e[3], one, 1);
        }
        else if (cen_pm1_sub(p, kmmi.s[3], kmmi.e[3], cen_div(km, kCenKm2))) {}
      }
    }
  }
  /* M9: densities as residual */
  {
    double land_km = 0, land_mi = 0, pop = 0, hu = 0;
    int hlk = 0, hlm = 0, hp = 0;
    if (has_lw) {
      hlk = cen_parse(p + lw.s[5], lw.e[5] - lw.s[5], &land_km) && land_km != 0.0;
      hlm = cen_parse(p + lw.s[6], lw.e[6] - lw.s[6], &land_mi) && land_mi != 0.0;
    }
    if (has_phf) hp = cen_parse(p + phf.s[2], phf.e[2] - phf.s[2], &pop) && pop != 0.0;
    has_pd = cen_m_popdens(p, n, &pd, 0);
    if (has_pd && hp && hlk) cen_dens_ladder(p, pd.s[2], pd.e[2], cen_div(pop, land_km));
    if (has_pd) {
      double dkl;
      if (cen_parse(p + pd.s[2], pd.e[2] - pd.s[2], &dkl) && dkl > 0.0)
        cen_dens_ladder(p, pd.s[3], pd.e[3], dkl * kCenKm2);
    }
    has_ho = cen_m_housing(p, n, &ho, 0);
    if (has_ho && cen_parse(p + ho.s[1], ho.e[1] - ho.s[1], &hu) && hu != 0.0) {
      int done2 = 0, done3 = 0;
      if (has_pd && hp) {
        double pdk, pdm;
        if (cen_parse(p + pd.s[2], pd.e[2] - pd.s[2], &pdk))
          done2 = cen_wdens_sub(p, ho.s[2], ho.e[2], pdk, hu, pop);
        if (cen_parse(p + pd.s[3], pd.e[3] - pd.s[3], &pdm))
          done3 = cen_wdens_sub(p, ho.s[3], ho.e[3], pdm, hu, pop);
      }
      if (hlk) {
        if (!done2) cen_dens_sub(p, ho.s[2], ho.e[2], cen_div(hu, land_km));
        if (!done3 && hlm) cen_dens_sub(p, ho.s[3], ho.e[3], cen_div(hu, land_mi));
      }
    }
  }
  /* M4: Mapit == Geo */
  has_geo = cen_m_geo(p, n, &geo);
  has_mp  = cen_m_mapit(p, n, &mp, 0);
  if (has_geo && has_mp && mp.s[0] > geo.e[0]) {
    for (int gi = 1; gi <= 2; ++gi) {
      if (geo.e[gi] - geo.s[gi] == mp.e[gi] - mp.s[gi] &&
          memcmp(p + geo.s[gi], p + mp.s[gi], (size_t)(geo.e[gi] - geo.s[gi])) == 0) {
        cen_try_sub(mp.s[gi], mp.e[gi], one, 1);
      }
    }
  }
  /* M5: DMS from the decimal coordinate */
  has_dms = cen_m_dms(p, n, &dms, 0);
  if (has_dms) {
    double lat, lon;
    if (cen_parse(p + dms.s[7], dms.e[7] - dms.s[7], &lat) &&
        cen_parse(p + dms.s[8], dms.e[8] - dms.s[8], &lon)) {
      long v[6];
      cen_dms(lat, &v[0], &v[1], &v[2]);
      cen_dms(lon, &v[3], &v[4], &v[5]);
      int all = 1;
      char b[32];
      for (int gi = 0; gi < 6; ++gi) {
        int bl = snprintf(b, sizeof b, "%ld", v[gi]);
        if (bl != (int)(dms.e[gi+1] - dms.s[gi+1]) ||
            memcmp(b, p + dms.s[gi+1], (size_t)bl) != 0) { all = 0; break; }
      }
      if (all) for (int gi = 1; gi <= 6; ++gi) cen_try_sub(dms.s[gi], dms.e[gi], one, 1);
    }
  }
  /* M6: last percentage from sum=100 */
  for (int key = 0; key < 2; ++key) {
    CenM* mm = key ? &age : &rac;
    int ndec = key ? 1 : 2;
    int have = key ? cen_m_age(p, n, mm) : cen_m_racial(p, n, mm);
    if (!have) continue;
    long ps[512][2];
    long cnt = 0, i = mm->s[1], endp = mm->e[1], ge;
    while (cnt < 512) {
      long gs = cen_m_pct(p, i, endp, &ge, 0);
      if (gs < 0) break;
      ps[cnt][0] = gs; ps[cnt][1] = ge; ++cnt;
      i = ge + 1;                          /* continue past the '%' (non-overlapping) */
    }
    if (cnt < 4) continue;
    double vals[512];
    int bad = 0;
    for (long t = 0; t < cnt; ++t) {
      if (!cen_parse(p + ps[t][0], ps[t][1] - ps[t][0], &vals[t])) { bad = 1; break; }
    }
    if (bad) continue;
    double rest = cen_pyround(cen_rest100(vals, cnt - 1), ndec);
    char want[64];
    long wl = cen_fmt_nd(rest, ndec, want);
    if (wl == ps[cnt-1][1] - ps[cnt-1][0] &&
        memcmp(want, p + ps[cnt-1][0], (size_t)wl) == 0) {
      cen_try_sub(ps[cnt-1][0], ps[cnt-1][1], two, 1);
    }
    else {
      double scale = ndec == 1 ? 10.0 : 100.0;
      double vlast = vals[cnt-1];
      double rr = cen_pyround0((vlast - rest) * scale);
      if (rr == 1.0 || rr == -1.0) {
        long r = (long)rr;
        char b[64];
        long bl = cen_fmt_nd(cen_pct_val(rest, r, scale), ndec, b);
        if (bl == ps[cnt-1][1] - ps[cnt-1][0] &&
            memcmp(b, p + ps[cnt-1][0], (size_t)bl) == 0 && bl > 1) {
          char mk[1]; mk[0] = (r == 1) ? CEN_PCTP : CEN_PCTM;
          if (cen_try_sub(ps[cnt-1][0], ps[cnt-1][1], mk, 1))
            {}
        }
      } else if (rr != 0.0) {
      }
    }
  }

  if (cen_nsub == 0) return 0;
  qsort(cen_subs, (size_t)cen_nsub, sizeof(CenSub), cen_subcmp);
  if (!cen_grow((void**)&cen_ob, &cen_obcap, n + 16, 1)) return 0;
  long o = 0, prev = 0;
  for (long t = 0; t < cen_nsub; ++t) {
    long s = cen_subs[t].s;
    if (s > prev) { memcpy(cen_ob + o, p + prev, (size_t)(s - prev)); o += s - prev; }
    memcpy(cen_ob + o, cen_subs[t].nb, (size_t)cen_subs[t].nl); o += cen_subs[t].nl;
    prev = cen_subs[t].e;
  }
  if (n > prev) { memcpy(cen_ob + o, p + prev, (size_t)(n - prev)); o += n - prev; }
  *outn = o;
  return 1;
}

// =========================================================================
// Decoder
// =========================================================================
static int cen_rep(long s, long e, const char* nb, long nl) {
  for (long t = 0; t < cen_nsub; ++t) if (cen_subs[t].s == s) return 0;
  if (nl >= (long)sizeof(cen_subs[0].nb)) return 0;
  if (!cen_grow((void**)&cen_subs, &cen_subcap, cen_nsub + 1, (long)sizeof(CenSub))) return 0;
  cen_subs[cen_nsub].s = s;
  cen_subs[cen_nsub].e = e;
  cen_subs[cen_nsub].nl = (int)nl;
  memcpy(cen_subs[cen_nsub].nb, nb, (size_t)nl);
  ++cen_nsub;
  return 1;
}

static int cen_is_mark(const char* p, long s, long e, char mk) {
  return (e - s == 1 && p[s] == mk);
}

static int cen_decode_page(const char* p, long n, long* outn) {
  CenM lw, phf, in_, hh, pd, ho, geo, mp, dms, rac, age, kmmi;
  int has_lw, has_phf, has_in, has_hh, has_pd, has_ho, has_geo, has_mp, has_dms;
  char b[64];
  long bl;
#define CEN_MARKSPAN 5u                    /* 0x01..0x05 */
  int seen_marker = 0;
  for (long i = 0; i < n; ++i) {
    if ((unsigned char)((unsigned char)p[i] - 1u) < CEN_MARKSPAN) {
      seen_marker = 1; break;
    }
  }
  if (!seen_marker) return 0;
  cen_nsub = 0;

  double tot_km = 0, land_km = 0, land_mi = 0, pop = 0;
  int h_tot = 0, h_lkm = 0, h_lmi = 0, h_pop = 0;
  long hh_s = -1, hh_e = -1;

  /* M7/M8 + M1 inside landw */
  has_lw = cen_m_landw(p, n, &lw, 1);
  if (has_lw) {
    long rpm;
    h_tot = cen_parse(p + lw.s[1], lw.e[1] - lw.s[1], &tot_km);
    if (cen_is_mark(p, lw.s[3], lw.e[3], CEN_EQ) && h_tot) {
      bl = cen_fmt1(cen_div(tot_km, kCenKm2), b);
      cen_rep(lw.s[3], lw.e[3], b, bl);
    }
    else if (lw.e[3] - lw.s[3] == 1 && cen_pm1_of(p[lw.s[3]], &rpm) && h_tot) {
      bl = cen_fmt1_pm(cen_div(tot_km, kCenKm2), rpm, b);
      cen_rep(lw.s[3], lw.e[3], b, bl);
    }
    if (cen_is_mark(p, lw.s[5], lw.e[5], CEN_EQ)) {
      land_km = tot_km; h_lkm = h_tot;
      cen_rep(lw.s[5], lw.e[5], p + lw.s[1], lw.e[1] - lw.s[1]);
    } else {
      h_lkm = cen_parse(p + lw.s[5], lw.e[5] - lw.s[5], &land_km);
    }
    if (cen_is_mark(p, lw.s[6], lw.e[6], CEN_EQ)) {
      if (h_lkm) {
        bl = cen_fmt1(cen_div(land_km, kCenKm2), b);
        h_lmi = cen_parse(b, bl, &land_mi);
        cen_rep(lw.s[6], lw.e[6], b, bl);
      }
    }
    else if (lw.e[6] - lw.s[6] == 1 && cen_pm1_of(p[lw.s[6]], &rpm)) {
      if (h_lkm) {
        bl = cen_fmt1_pm(cen_div(land_km, kCenKm2), rpm, b);
        h_lmi = cen_parse(b, bl, &land_mi);
        cen_rep(lw.s[6], lw.e[6], b, bl);
      }
    }
    else {
      h_lmi = cen_parse(p + lw.s[6], lw.e[6] - lw.s[6], &land_mi);
    }
    if (lw.s[8] >= 0) {
      double wk = 0; int h_wk = 0;
      if (cen_is_mark(p, lw.s[8], lw.e[8], CEN_EQ)) {
        if (h_tot && h_lkm) {
          bl = cen_fmt1(cen_minus(tot_km, land_km), b);
          h_wk = cen_parse(b, bl, &wk);
          cen_rep(lw.s[8], lw.e[8], b, bl);
        }
      }
      else if (lw.e[8] - lw.s[8] == 1 && cen_pm1_of(p[lw.s[8]], &rpm)) {
        if (h_tot && h_lkm) {
          bl = cen_fmt1_pm(cen_minus(tot_km, land_km), rpm, b);
          h_wk = cen_parse(b, bl, &wk);
          cen_rep(lw.s[8], lw.e[8], b, bl);
        }
      }
      else {
        h_wk = cen_parse(p + lw.s[8], lw.e[8] - lw.s[8], &wk);
      }
      if (cen_is_mark(p, lw.s[9], lw.e[9], CEN_EQ) && h_wk) {
        bl = cen_fmt1(cen_div(wk, kCenKm2), b);
        cen_rep(lw.s[9], lw.e[9], b, bl);
      }
      else if (lw.e[9] - lw.s[9] == 1 && cen_pm1_of(p[lw.s[9]], &rpm) && h_wk) {
        bl = cen_fmt1_pm(cen_div(wk, kCenKm2), rpm, b);
        cen_rep(lw.s[9], lw.e[9], b, bl);
      }
    }
  }
  /* remaining km->mi pairs */
  {
    long pos = 0, st;
    while ((st = cen_m_kmmi(p, n, pos, &kmmi, 1)) >= 0) {
      pos = kmmi.e[0];
      double km;
      if (cen_parse(p + kmmi.s[1], kmmi.e[1] - kmmi.s[1], &km)) {
        long s = kmmi.s[0] + (kmmi.e[1] - kmmi.s[1]) + 1 + (kmmi.e[2] - kmmi.s[2]) + 2;
        long rpm;
        if (cen_pm1_of(p[s], &rpm)) bl = cen_fmt1_pm(cen_div(km, kCenKm2), rpm, b);
        else                        bl = cen_fmt1(cen_div(km, kCenKm2), b);
        cen_rep(s, s + 1, b, bl);
      }
    }
  }
  /* M3: intro population */
  has_in  = cen_m_intro(p, n, &in_);
  has_phf = cen_m_phf_d(p, n, &phf);
  if (has_phf) {
    if (cen_is_mark(p, phf.s[2], phf.e[2], CEN_EQ)) {
      if (has_in) {
        cen_rep(phf.s[2], phf.e[2], p + in_.s[1], in_.e[1] - in_.s[1]);
        h_pop = cen_parse(p + in_.s[1], in_.e[1] - in_.s[1], &pop);
      }
    } else {
      h_pop = cen_parse(p + phf.s[2], phf.e[2] - phf.s[2], &pop);
    }
    hh_s = phf.s[3]; hh_e = phf.e[3];
  }
  /* M2: Households */
  has_hh = cen_m_hhrep(p, n, &hh, 1);
  if (has_hh && hh_s >= 0) cen_rep(hh.s[2], hh.e[2], p + hh_s, hh_e - hh_s);
  /* M9: densities */
  double pdens_km = 0, pdens_mi = 0;
  int h_pdk = 0, h_pdm = 0;
  has_pd = cen_m_popdens(p, n, &pd, 1);
  if (has_pd) {
    long rr2 = 0, rr3 = 0;
    int m2 = (pd.e[2] - pd.s[2] > 0 && p[pd.s[2]] == CEN_DENS);
    int m3 = (pd.e[3] - pd.s[3] > 0 && p[pd.s[3]] == CEN_DENS);
    int e2 = (pd.e[2] - pd.s[2] == 1 && p[pd.s[2]] == CEN_EQ);
    int e3 = (pd.e[3] - pd.s[3] == 1 && p[pd.s[3]] == CEN_EQ);
    int q2 = (pd.e[2] - pd.s[2] == 1 && cen_pm1_of(p[pd.s[2]], &rr2));
    int q3 = (pd.e[3] - pd.s[3] == 1 && cen_pm1_of(p[pd.s[3]], &rr3));
    if ((m2 || e2 || q2) && h_pop && h_lkm) {
      double base = cen_div(pop, land_km);
      if (m2)      bl = cen_fmt1(cen_dens_val(cen_pyround(base, 1), strtol(p + pd.s[2] + 1, 0, 10)), b);
      else if (e2) bl = cen_fmt1(base, b);
      else         bl = cen_fmt1_pm(base, rr2, b);
      cen_rep(pd.s[2], pd.e[2], b, bl);
      h_pdk = cen_parse(b, bl, &pdens_km);
    } else if (!(m2 || e2 || q2)) {
      h_pdk = cen_parse(p + pd.s[2], pd.e[2] - pd.s[2], &pdens_km);
    }
    if ((m3 || e3 || q3) && h_pdk && pdens_km > 0.0) {
      double base = pdens_km * kCenKm2;
      if (m3)      bl = cen_fmt1(cen_dens_val(cen_pyround(base, 1), strtol(p + pd.s[3] + 1, 0, 10)), b);
      else if (e3) bl = cen_fmt1(base, b);
      else         bl = cen_fmt1_pm(base, rr3, b);
      cen_rep(pd.s[3], pd.e[3], b, bl);
      h_pdm = cen_parse(b, bl, &pdens_mi);
    } else if (!(m3 || e3 || q3)) {
      h_pdm = cen_parse(p + pd.s[3], pd.e[3] - pd.s[3], &pdens_mi);
    }
  }
  has_ho = cen_m_housing(p, n, &ho, 1);
  if (has_ho) {
    int m2 = (ho.e[2] - ho.s[2] > 0 && p[ho.s[2]] == CEN_DENS);
    int m3 = (ho.e[3] - ho.s[3] > 0 && p[ho.s[3]] == CEN_DENS);
    int e2 = (ho.e[2] - ho.s[2] == 1 && p[ho.s[2]] == CEN_EQ);
    int e3 = (ho.e[3] - ho.s[3] == 1 && p[ho.s[3]] == CEN_EQ);
    long r2 = 0, r3 = 0;
    int q2 = (ho.e[2] - ho.s[2] == 1 && cen_pm1_of(p[ho.s[2]], &r2));
    int q3 = (ho.e[3] - ho.s[3] == 1 && cen_pm1_of(p[ho.s[3]], &r3));
#define CEN_HO_ANY (m2 || m3 || e2 || e3 || q2 || q3)
    double hu;
    if (CEN_HO_ANY && cen_parse(p + ho.s[1], ho.e[1] - ho.s[1], &hu)) {
      if (e2 && h_pdk && h_pop) {
        bl = cen_fmt1(cen_wdens(pdens_km, hu, pop), b);
        cen_rep(ho.s[2], ho.e[2], b, bl);
      }
      if (e3 && h_pdm && h_pop) {
        bl = cen_fmt1(cen_wdens(pdens_mi, hu, pop), b);
        cen_rep(ho.s[3], ho.e[3], b, bl);
      }
      if (q2 && h_pdk && h_pop) {
        bl = cen_fmt1_pm(cen_wdens(pdens_km, hu, pop), r2, b);
        cen_rep(ho.s[2], ho.e[2], b, bl);
      }
      if (q3 && h_pdm && h_pop) {
        bl = cen_fmt1_pm(cen_wdens(pdens_mi, hu, pop), r3, b);
        cen_rep(ho.s[3], ho.e[3], b, bl);
      }
      if (m2 && h_lkm) {
        long res = strtol(p + ho.s[2] + 1, 0, 10);
        bl = cen_fmt1(cen_dens_val(cen_pyround(cen_div(hu, land_km), 1), res), b);
        cen_rep(ho.s[2], ho.e[2], b, bl);
      }
      if (m3 && h_lmi) {
        long res = strtol(p + ho.s[3] + 1, 0, 10);
        bl = cen_fmt1(cen_dens_val(cen_pyround(cen_div(hu, land_mi), 1), res), b);
        cen_rep(ho.s[3], ho.e[3], b, bl);
      }
    }
  }
#undef CEN_HO_ANY
  /* M4: Mapit */
  has_geo = cen_m_geo(p, n, &geo);
  has_mp  = cen_m_mapit(p, n, &mp, 1);
  if (has_mp && has_geo) {
    for (int gi = 1; gi <= 2; ++gi) {
      if (cen_is_mark(p, mp.s[gi], mp.e[gi], CEN_EQ)) {
        cen_rep(mp.s[gi], mp.e[gi], p + geo.s[gi], geo.e[gi] - geo.s[gi]);
      }
    }
  }
  /* M5: DMS */
  has_dms = cen_m_dms(p, n, &dms, 1);
  if (has_dms) {
    double lat, lon;
    if (cen_parse(p + dms.s[7], dms.e[7] - dms.s[7], &lat) &&
        cen_parse(p + dms.s[8], dms.e[8] - dms.s[8], &lon)) {
      long v[6];
      cen_dms(lat, &v[0], &v[1], &v[2]);
      cen_dms(lon, &v[3], &v[4], &v[5]);
      for (int gi = 1; gi <= 6; ++gi) {
        int k = snprintf(b, sizeof b, "%ld", v[gi-1]);
        cen_rep(dms.s[gi], dms.e[gi], b, k);
      }
    }
  }
  /* M6: last percentages */
  for (int key = 0; key < 2; ++key) {
    CenM* mm = key ? &age : &rac;
    int ndec = key ? 1 : 2;
    int have = key ? cen_m_age(p, n, mm) : cen_m_racial(p, n, mm);
    if (!have) continue;
    long ps[512][2];
    long cnt = 0, i = mm->s[1], endp = mm->e[1], ge;
    while (cnt < 512) {
      long gs = cen_m_pct(p, i, endp, &ge, 1);
      if (gs < 0) break;
      ps[cnt][0] = gs; ps[cnt][1] = ge; ++cnt;
      i = ge + 1;
    }
    if (cnt == 0) continue;
    long ls = ps[cnt-1][0], le = ps[cnt-1][1];
    if (le - ls != 1) continue;
    long r6;
    if      (p[ls] == CEN_PCT)  r6 = 0;
    else if (p[ls] == CEN_PCTP) r6 = 1;
    else if (p[ls] == CEN_PCTM) r6 = -1;
    else continue;
    double vals[512];
    int bad = 0;
    for (long t = 0; t < cnt - 1; ++t) {
      if (!cen_parse(p + ps[t][0], ps[t][1] - ps[t][0], &vals[t])) { bad = 1; break; }
    }
    if (bad) continue;
    double rest = cen_pyround(cen_rest100(vals, cnt - 1), ndec);
    long wl;
    if (r6 != 0) wl = cen_fmt_nd(cen_pct_val(rest, r6, ndec == 1 ? 10.0 : 100.0), ndec, b);
    else         wl = cen_fmt_nd(rest, ndec, b);
    cen_rep(ls, le, b, wl);
  }

  if (cen_nsub == 0) return 0;
  qsort(cen_subs, (size_t)cen_nsub, sizeof(CenSub), cen_subcmp);
  long need = n + 16;
  for (long t = 0; t < cen_nsub; ++t) need += cen_subs[t].nl;
  if (!cen_grow((void**)&cen_ob, &cen_obcap, need, 1)) return 0;
  long o = 0, prev = 0;
  for (long t = 0; t < cen_nsub; ++t) {
    long s = cen_subs[t].s;
    if (s > prev) { memcpy(cen_ob + o, p + prev, (size_t)(s - prev)); o += s - prev; }
    memcpy(cen_ob + o, cen_subs[t].nb, (size_t)cen_subs[t].nl); o += cen_subs[t].nl;
    prev = cen_subs[t].e;
  }
  if (n > prev) { memcpy(cen_ob + o, p + prev, (size_t)(n - prev)); o += n - prev; }
  *outn = o;
  return 1;
}

static int cen_gate(const char* p, long n) {
  if (cen_is_census(p, n)) return 1;
  if (cen_find(p, n, 0, "population density]] ", 21) >= 0) return 1;
  if (cen_find(p, n, 0, " households, and ", 17) >= 0) return 1;
  return 0;
}

static int census_stream(FILE* in, FILE* out, int enc) {
  const long CH = 1 << 20;
  long cap = 4 * CH, n = 0, pos;
  char* buf = (char*)malloc((size_t)cap);
  if (!buf) return 0;
  int inside = 0, eof = 0;
  while (!eof) {
    if (n + CH > cap) {
      long nc = (n + CH) * 2;
      char* q = (char*)realloc(buf, (size_t)nc);
      if (!q) { free(buf); return 0; }
      buf = q; cap = nc;
    }
    size_t got = fread(buf + n, 1, (size_t)CH, in);
    if (got == 0) eof = 1;
    n += (long)got;
    pos = 0;
    for (;;) {
      if (!inside) {
        long s = cen_find(buf, n, pos, "<page>", 6);
        if (s < 0) {
          long keep = eof ? 0 : 5;
          long upto = n - keep;
          if (upto < pos) upto = pos;
          if (upto > pos) fwrite(buf + pos, 1, (size_t)(upto - pos), out);
          pos = upto;
          break;
        }
        if (s > pos) fwrite(buf + pos, 1, (size_t)(s - pos), out);
        pos = s;
        inside = 1;
      } else {
        long e = cen_find(buf, n, pos, "</page>", 7);
        if (e < 0) break;
        e += 7;
        long pl = e - pos, ol = 0;
        int changed;
        if (enc) changed = cen_gate(buf + pos, pl) ? cen_encode_page(buf + pos, pl, &ol) : 0;
        else     changed = cen_decode_page(buf + pos, pl, &ol);
        if (changed) fwrite(cen_ob, 1, (size_t)ol, out);
        else         fwrite(buf + pos, 1, (size_t)pl, out);
        pos = e;
        inside = 0;
      }
    }
    if (pos > 0) { memmove(buf, buf + pos, (size_t)(n - pos)); n -= pos; }
  }
  if (n > 0) fwrite(buf, 1, (size_t)n, out);
  free(buf);
  return 1;
}

static int census_file(const char* inpath, const char* outpath, int enc) {
  FILE* in = fopen(inpath, "rb");
  if (!in) { fprintf(stderr, "census: cannot read %s\n", inpath); return 0; }
  FILE* out = fopen(outpath, "wb");
  if (!out) { fclose(in); fprintf(stderr, "census: cannot write %s\n", outpath); return 0; }
  int ok = census_stream(in, out, enc);
  fclose(in);
  if (fclose(out) != 0) ok = 0;
  return ok;
}

#if defined(__clang__)
#pragma float_control(pop)
#endif

#endif
