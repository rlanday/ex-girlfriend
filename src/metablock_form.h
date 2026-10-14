#ifndef METABLOCK_FORM_H_
#define METABLOCK_FORM_H_

#include "metablock.h"

#include "neif_levers.h"
#include "ts_progress.h"
#if NEIF_E1
#include "b1_interp.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

namespace metablock_form {

static const unsigned char kNL = 0x0a;
static const unsigned char kTab = 0x09;
static const unsigned char kDopp = (unsigned char)0x4a;
static const int kA0 = (int)(unsigned)0x80;

static_assert(0x80 >= 0x80 && 0x80 + 99 <= 0xE3,
    "the 100-value alphabet must lie in 0x80..0xE3 - "
    "outside it is not part of the frozen 208-byte set of the coder stream");

static_assert(kDopp != kTab && kDopp != kNL,
    "collides with the field or line separator");
static_assert(kDopp < 0x30 || kDopp > 0x39,
    "is a digit - the month field would not end");
static_assert((int)kDopp < kA0 || (int)kDopp > kA0 + 99,
    "lies in the base-100 alphabet of the fixed-width forms");

static const int kRevDig = 8;

static inline bool Oben(size_t k) {
  return (long long)k >= (long long)87813;
}

static inline void Fail(const char* what) {
  fprintf(stderr, "\nmetablock codec: %s\n", what);
  fflush(stderr);
  exit(1);
}

static inline void Fest(std::vector<unsigned char>* o, long long v, int n) {
  unsigned char z[8];
  for (int i = n - 1; i >= 0; --i) {
    z[i] = (unsigned char)(kA0 + (int)(v % 100));
    v /= 100;
  }
  if (v) Fail("value exceeds the fixed base-100 width");
  for (int i = 0; i < n; ++i) o->push_back(z[i]);
}
static inline long long LiesFest(const unsigned char* p, int n) {
  long long v = 0;
  for (int i = 0; i < n; ++i) {
    const int x = (int)p[i] - kA0;
    if (x < 0 || x > 99) Fail("base-100 digit outside the alphabet");
    v = v * 100 + x;
  }
  return v;
}

static inline void Dezimal(std::vector<unsigned char>* o, long long v) {
  char b[24];
  const int n = ts_progress::DezimalRoh(b, v);
  for (int i = 0; i < n; ++i) o->push_back((unsigned char)b[i]);
}
static inline bool Ziffer(unsigned char c) { return c >= 0x30 && c <= 0x39; }

struct Ts {
  int yy;
  long long mt, sek;
};
static inline Ts LiesTs(const unsigned char* p, int n) {
  Ts r;
  if (n < 4) Fail("timestamp too short");
  if (!Ziffer(p[0]) || !Ziffer(p[1])) Fail("timestamp without two year digits");
  r.yy = (p[0] - 0x30) * 10 + (p[1] - 0x30);
  int i = 2;
  r.mt = 0;
  for (; i < n && Ziffer(p[i]); ++i) r.mt = r.mt * 10 + (p[i] - 0x30);
  if (i == 2) Fail("timestamp without month field");
  if (i >= n || p[i] != kDopp)
    Fail("timestamp separator byte not as expected");
  ++i;
  if (i >= n) Fail("timestamp without seconds field");
  r.sek = 0;
  for (; i < n; ++i) {
    if (!Ziffer(p[i])) Fail("seconds field not numeric");
    r.sek = r.sek * 10 + (p[i] - 0x30);
  }
  if (r.sek > 86399) Fail("seconds field over 86399");
  return r;
}

static inline void SchreibTs(std::vector<unsigned char>* o, const Ts& t) {
  if (t.yy < 0 || t.yy > 99) Fail("year outside 2001..2100");
  o->push_back((unsigned char)(0x30 + t.yy / 10));
  o->push_back((unsigned char)(0x30 + t.yy % 10));
  Dezimal(o, t.mt);
  o->push_back(kDopp);
  Dezimal(o, t.sek);
}

 static inline bool LiesDatei(const std::string& pfad, std::vector<unsigned char>* v) {
  FILE* f = fopen(pfad.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0L, SEEK_END);
  long n = ftell(f);
  fseek(f, 0L, SEEK_SET);
  v->resize((size_t)(n > 0 ? n : 0));
  bool ok = v->empty() || fread(&(*v)[0], 1, v->size(), f) == v->size();
  fclose(f);
  return ok;
}
 static inline bool SchreibDatei(const std::string& pfad,
                                const std::vector<unsigned char>& v) {
  FILE* f = fopen(pfad.c_str(), "wb");
  if (!f) return false;
  bool ok = v.empty() || fwrite(&v[0], 1, v.size(), f) == v.size();
  fclose(f);
  return ok;
}

 static inline bool Grenzen(const std::vector<unsigned char>& d,
                           size_t* ab, size_t* bis) {
  metablock::Grenze gr;
  const size_t n = d.size();
  for (size_t p = 0; p < n; ++p) {
    gr.Byte(d[p]);
    if (gr.Endet()) { *ab = (size_t)gr.Ab(); *bis = (size_t)gr.Bis(); return true; }
  }
  return false;
}

 static inline void Zeilen(const std::vector<unsigned char>& d, size_t ab,
                          size_t bis, std::vector<size_t>* z) {
  const size_t nn = (size_t)243425;
  z->resize(2 * nn + 1);
  size_t p = ab;
  for (size_t i = 0; i < 2 * nn; ++i) {
    (*z)[i] = p;
    if (p >= bis) Fail("metablock ends before 2N lines");
    const void* e = memchr(&d[p], kNL, bis - p);
    if (!e) Fail("metablock line without newline");
    p = (size_t)((const unsigned char*)e - &d[0]) + 1;
  }
  (*z)[2 * nn] = p;
  if (p != bis) Fail("data after line 2N of the metablock");
}

static inline long long ALeseRev(const unsigned char* p, int n, bool s4) {
  const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
  if (s4) {
    const unsigned char* e = t ? t : p + n;
    if (e - p < 4) Fail("A line without four revid digits");
    return LiesFest(e - 4, 4);
  }
  if (!t) Fail("A line without tab");
  long long v = 0;
  int d = 0;
  for (const unsigned char* q = t + 1; q < p + n && *q != kTab; ++q, ++d) {
    if (!Ziffer(*q)) Fail("revid not numeric");
    v = v * 10 + (*q - 0x30);
  }
  if (!d) Fail("A line without revid");
  return v;
}

#if NEIF_E1
// E1: signed decimal page id delta of a part-A line, [p, e) = "[-]digits".
static inline long long LiesDelta(const unsigned char* p, const unsigned char* e) {
  int vz = 1;
  if (p < e && *p == 0x2d) { vz = -1; ++p; }
  if (p == e) Fail("page id delta empty");
  long long d = 0;
  for (; p < e; ++p) {
    if (!Ziffer(*p)) Fail("page id delta not numeric");
    d = d * 10 + (*p - 0x30);
  }
  return vz * d;
}
#endif

static inline void Rangfolge(std::vector<long long>* rev) {
  std::sort(rev->begin(), rev->end());
  for (size_t i = 1; i < rev->size(); ++i)
    if ((*rev)[i] == (*rev)[i - 1])
      Fail("duplicate revid");
}

// ========================================================== ENCODER SIDE ====
 inline bool Pack(const std::string& pfad) {
  std::vector<unsigned char> d;
  if (!LiesDatei(pfad, &d)) return false;
  size_t ab = 0, bis = 0;
  if (!Grenzen(d, &ab, &bis)) return true;       // no complete block: unchanged
  const size_t nn = (size_t)243425;
  std::vector<size_t> z;
  Zeilen(d, ab, bis, &z);

  std::vector<long long> rev(nn);
  for (size_t i = 0; i < nn; ++i)
    rev[i] = ALeseRev(&d[z[i]], (int)(z[i + 1] - z[i] - 1), false);
  Rangfolge(&rev);

  std::vector<unsigned char> out;
  out.reserve(d.size());
  out.insert(out.end(), d.begin(), d.begin() + (long)ab);

  // ---- part A ----
#if NEIF_E1
  b1_interp::Interp b1;
  long long pid = 0;
#endif
  for (size_t i = 0; i < nn; ++i) {
    const unsigned char* p = &d[z[i]];
    const int n = (int)(z[i + 1] - z[i] - 1);
    {
      const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
      if (!t) Fail("A line without tab");
      const unsigned char* e = p + n;
      const unsigned char* t2 =
          (const unsigned char*)memchr(t + 1, kTab, (size_t)(e - t - 1));
      long long rv = 0;
      for (const unsigned char* q = t + 1; q < (t2 ? t2 : e); ++q) {
        if (!Ziffer(*q)) Fail("revid not numeric");
        rv = rv * 10 + (*q - 0x30);
      }
      out.insert(out.end(), p, t);                 // page id delta, decimal
#if NEIF_E1
      pid += LiesDelta(p, t);
      if (rv < b1_interp::kB1Max) {                // B1: '+' residual vs interpolation
        out.push_back(b1_interp::kMark);
        Dezimal(&out, rv - b1.Prognose(pid));
        b1.Add(pid, rv);
      } else {
        Fest(&out, rv, 4);
      }
#else
      Fest(&out, rv, 4);
#endif
      if (t2) out.insert(out.end(), t2, e);        // TAB restrictions
    }
    out.push_back(kNL);
  }

  // ---- part B ----
  unsigned long long n_oben = 0, n_unten = 0;
  long long tvor = 0;
  for (size_t k = 0; k < nn; ++k) {
    const unsigned char* p = &d[z[nn + k]];
    const int n = (int)(z[nn + k + 1] - z[nn + k] - 1);
    const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
    if (!t) Fail("B line without tab");
    const int fn = (int)(t - p);
    const Ts ts = LiesTs(p, fn);
    const long long t_k = ts_progress::TsZuSek(ts.yy, ts.mt, ts.sek);
    if (Oben(k)) {
      Dezimal(&out, t_k - (k ? tvor : 0LL));
      ++n_oben;
    } else {
      Fest(&out, (long long)ts.yy, 1);
      Fest(&out, ts.mt, 2);
      Fest(&out, ts.sek, 3);
      ++n_unten;
    }
    out.insert(out.end(), t, p + n);               // TAB name [TAB uid]
    out.push_back(kNL);
    tvor = t_k;
  }

  out.insert(out.end(), d.begin() + (long)bis, d.end());
  (void)n_oben; (void)n_unten;
  return SchreibDatei(pfad, out);
}

// ========================================================== DECODER SIDE ====
// The same stage backwards, record by record in the same rank order.
 inline bool Unpack(const std::string& pfad) {
  std::vector<unsigned char> d;
  if (!LiesDatei(pfad, &d)) return false;
  size_t ab = 0, bis = 0;
  if (!Grenzen(d, &ab, &bis)) return true;       // no complete block: unchanged
  const size_t nn = (size_t)243425;
  std::vector<size_t> z;
  Zeilen(d, ab, bis, &z);

  std::vector<long long> rev(nn);
#if !NEIF_E1
  for (size_t i = 0; i < nn; ++i)
    rev[i] = ALeseRev(&d[z[i]], (int)(z[i + 1] - z[i] - 1), true);
  Rangfolge(&rev);
#endif

  std::vector<unsigned char> out;
  out.reserve(d.size() + d.size() / 4);
  out.insert(out.end(), d.begin(), d.begin() + (long)ab);

  // ---- part A ----
#if NEIF_E1
  // E1: a B1 revid needs the interpolation over the earlier B1 lines, so the
  // revids are computed here, in stream order, and checked afterwards.
  b1_interp::Interp b1;
  long long pid = 0;
#endif
  for (size_t i = 0; i < nn; ++i) {
    const unsigned char* p = &d[z[i]];
    const int n = (int)(z[i + 1] - z[i] - 1);
    {
      const unsigned char* e = p + n;
      const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
#if NEIF_E1
      const unsigned char* fe = t ? t : e;         // end of field 1
      const unsigned char* dz = p;
      if (dz < fe && *dz == 0x2d) ++dz;
      while (dz < fe && Ziffer(*dz)) ++dz;
      long long v = 0;
      const unsigned char* r = nullptr;            // end of the page id delta
      if (dz < fe && *dz == b1_interp::kMark) {    // B1: delta '+' [-]residual
        r = dz;
        pid += LiesDelta(p, r);
        const unsigned char* q = r + 1;
        long long res = 0;
        int vz = 1;
        if (q < fe && *q == 0x2d) { vz = -1; ++q; }
        if (q == fe) Fail("B1 line without revid residual");
        for (; q < fe; ++q) {
          if (!Ziffer(*q)) Fail("revid residual not numeric");
          res = res * 10 + (*q - 0x30);
        }
        v = b1.Prognose(pid) + vz * res;
        if (v < 0) Fail("B1 revid negative");
        b1.Add(pid, v);
      } else {                                     // B2: delta + 4 base-100 bytes
        r = fe - 4;
        if (r < p) Fail("A line without four revid digits");
        pid += LiesDelta(p, r);
        v = LiesFest(r, 4);
      }
      rev[i] = v;
#else
      const unsigned char* r = (t ? t : e) - 4;
      if (r < p) Fail("A line without four revid digits");
#endif
      out.insert(out.end(), p, r);                 // page id delta, decimal
      out.push_back(kTab);
      {
#if !NEIF_E1
        long long v = LiesFest(r, 4);
#endif
        char b[24];
        for (int q = kRevDig - 1; q >= 0; --q) { b[q] = (char)('0' + (int)(v % 10)); v /= 10; }
        if (v) Fail("revid has too many digits");
        for (int q = 0; q < kRevDig; ++q) out.push_back((unsigned char)b[q]);
      }
      if (t) out.insert(out.end(), t, e);          // TAB restrictions
    }
    out.push_back(kNL);
  }
#if NEIF_E1
  Rangfolge(&rev);
#endif

  // ---- part B ----
  unsigned long long n_oben = 0, n_unten = 0;
  long long tvor = 0;
  for (size_t k = 0; k < nn; ++k) {
    const unsigned char* p = &d[z[nn + k]];
    const int n = (int)(z[nn + k + 1] - z[nn + k] - 1);
    const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
    if (!t) Fail("B line without tab");
    const int fn = (int)(t - p);
    long long t_k = 0;
    if (Oben(k)) {
      char b[64];
      if (fn <= 0 || fn >= (int)sizeof(b)) Fail("delta field of impossible length");
      memcpy(b, p, (size_t)fn);
      t_k = (k ? tvor : 0LL) + ts_progress::ZahlRoh(b, fn);
      ++n_oben;
    } else {
      if (fn != 6) Fail("short form: field 0 is not 6 wide");
      t_k = ts_progress::TsZuSek((int)LiesFest(p, 1), LiesFest(p + 1, 2),
                                 LiesFest(p + 3, 3));
      ++n_unten;
    }
    {
      Ts ts;
      ts_progress::SekZuTs(t_k, &ts.yy, &ts.mt, &ts.sek);
      SchreibTs(&out, ts);
    }
    out.insert(out.end(), t, p + n);
    out.push_back(kNL);
    tvor = t_k;
  }

  out.insert(out.end(), d.begin() + (long)bis, d.end());
  (void)n_oben; (void)n_unten;
  return SchreibDatei(pfad, out);
}

}

#endif
