#ifndef METABLOCK_FORM_H_
#define METABLOCK_FORM_H_
#include "meld.h"
#include "u8_append.h"

#include "metablock.h"

#include "ts_progress.h"

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
  fprintf(stderr, "\n%s\n", what);
  fflush(stderr);
  exit(1);
}

static inline void Fest(std::vector<unsigned char>* o, long long v, int n) {
  unsigned char z[8];
  for (int i = n - 1; i >= 0; --i) {
    z[i] = (unsigned char)(kA0 + (int)(v % 100));
    v /= 100;
  }
  if (v) __builtin_unreachable();
  for (int i = 0; i < n; ++i) o->push_back(z[i]);
}
static inline long long LiesFest(const unsigned char* p, int n) {
  long long v = 0;
  for (int i = 0; i < n; ++i) {
    const int x = (int)p[i] - kA0;
    if (x < 0 || x > 99) __builtin_unreachable();
    v = v * 100 + x;
  }
  return v;
}

static inline void Dezimal(std::vector<unsigned char>* o, long long v) { ts_progress::DezimalAn(o, v); }
static inline bool Ziffer(unsigned char c) { return c >= 0x30 && c <= 0x39; }

struct Ts {
  int yy;
  long long mt, sek;
};
static inline Ts LiesTs(const unsigned char* p, int n) {
  Ts r;
  if (n < 4) __builtin_unreachable();
  if (!Ziffer(p[0]) || !Ziffer(p[1])) __builtin_unreachable();
  r.yy = (p[0] - 0x30) * 10 + (p[1] - 0x30);
  int i = 2;
  r.mt = 0;
  for (; i < n && Ziffer(p[i]); ++i) r.mt = r.mt * 10 + (p[i] - 0x30);
  if (i == 2) __builtin_unreachable();
  if (i >= n || p[i] != kDopp)
    __builtin_unreachable();
  ++i;
  if (i >= n) __builtin_unreachable();
  r.sek = 0;
  for (; i < n; ++i) {
    if (!Ziffer(p[i])) __builtin_unreachable();
    r.sek = r.sek * 10 + (p[i] - 0x30);
  }
  if (r.sek > 86399) __builtin_unreachable();
  return r;
}

static inline void SchreibTs(std::vector<unsigned char>* o, const Ts& t) {
  if (t.yy < 0 || t.yy > 99) __builtin_unreachable();
  o->push_back((unsigned char)(0x30 + t.yy / 10));
  o->push_back((unsigned char)(0x30 + t.yy % 10));
  Dezimal(o, t.mt);
  o->push_back(kDopp);
  Dezimal(o, t.sek);
}

 static inline bool LiesDatei(const char* pfad, std::vector<unsigned char>* v) {
  FILE* f = fopen(pfad, "rb");
  if (!f) return false;
  fseek(f, 0L, SEEK_END);
  long n = ftell(f);
  fseek(f, 0L, SEEK_SET);
  v->resize((size_t)(n > 0 ? n : 0));
  bool ok = v->empty() || fread(&(*v)[0], 1, v->size(), f) == v->size();
  fclose(f);
  return ok;
}
 static inline bool SchreibDatei(const char* pfad, const std::vector<unsigned char>& v) {
  FILE* f = fopen(pfad, "wb");
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
    if (p >= bis) __builtin_unreachable();
    const void* e = memchr(&d[p], kNL, bis - p);
    if (!e) __builtin_unreachable();
    p = (size_t)((const unsigned char*)e - &d[0]) + 1;
  }
  (*z)[2 * nn] = p;
  if (p != bis) __builtin_unreachable();
}

 inline bool Pack(const std::string& pfad) {
  std::vector<unsigned char> d;
  if (!LiesDatei(pfad.c_str(), &d)) return false;
  size_t ab = 0, bis = 0;
  if (!Grenzen(d, &ab, &bis)) return true;       // no complete block: unchanged
  const size_t nn = (size_t)243425;
  std::vector<size_t> z;
  Zeilen(d, ab, bis, &z);

  std::vector<unsigned char> out;
  out.reserve(d.size());
  u8_append::Anh(out, d.begin(), d.begin() + (long)ab);

  // ---- part A ----
  // After E1 of ex-girlfriend 1.0 (Ryan Landay, GPL-3.0-only): a revid
  // below 17,000,000 goes out as `+[-]<residual>` against the interpolation
  // over the forecast pages written so far (stream order).
  metablock::meta_fields::Interp b1;
  long long pid = 0;
  for (size_t i = 0; i < nn; ++i) {
    const unsigned char* p = &d[z[i]];
    const int n = (int)(z[i + 1] - z[i] - 1);
    {
      const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
      if (!t) __builtin_unreachable();
      const unsigned char* e = p + n;
      const unsigned char* t2 =
          (const unsigned char*)memchr(t + 1, kTab, (size_t)(e - t - 1));
      long long rv = 0;
      for (const unsigned char* q = t + 1; q < (t2 ? t2 : e); ++q) {
        if (!Ziffer(*q)) __builtin_unreachable();
        rv = rv * 10 + (*q - 0x30);
      }
      u8_append::Anh(out, p, t);                 // page id delta, decimal
      pid += metablock::meta_fields::Zahl(p, t);
      if (rv > 0 && rv < metablock::meta_fields::kMax) {
        out.push_back(metablock::meta_fields::kMarke);
        Dezimal(&out, b1.Paar(pid, rv, false));   // the residual
      } else
      Fest(&out, rv, 4);
      if (t2) u8_append::Anh(out, t2, e);        // TAB restrictions
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
    if (!t) __builtin_unreachable();
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
    u8_append::Anh(out, t, p + n);               // TAB name [TAB uid]
    out.push_back(kNL);
    tvor = t_k;
  }

  u8_append::Anh(out, d.begin() + (long)bis, d.end());
  (void)n_oben; (void)n_unten;
  return SchreibDatei(pfad.c_str(), out);
}

// ========================================================== DECODER SIDE ====
// The same stage backwards, record by record in the same rank order.
 inline bool Unpack(const std::string& pfad) {
  std::vector<unsigned char> d;
  if (!LiesDatei(pfad.c_str(), &d)) return false;
  size_t ab = 0, bis = 0;
  if (!Grenzen(d, &ab, &bis)) return true;       // no complete block: unchanged
  const size_t nn = (size_t)243425;
  std::vector<size_t> z;
  Zeilen(d, ab, bis, &z);

  std::vector<unsigned char> out;
  out.reserve(d.size() + d.size() / 4);
  u8_append::Anh(out, d.begin(), d.begin() + (long)ab);

  // ---- part A ----
  // a forecast revid needs the interpolation over the earlier forecast lines,
  // in stream order.
  metablock::meta_fields::Interp b1;
  long long pid = 0;
  for (size_t i = 0; i < nn; ++i) {
    const unsigned char* p = &d[z[i]];
    const int n = (int)(z[i + 1] - z[i] - 1);
    {
      const unsigned char* e = p + n;
      const unsigned char* t = (const unsigned char*)memchr(p, kTab, (size_t)n);
      const unsigned char* const fe = t ? t : e;   // end of field 1
      const unsigned char* r = metablock::meta_fields::Marke(p, fe);
      long long v;
      if (r) {
        pid += metablock::meta_fields::Zahl(p, r);
        v = b1.Paar(pid, metablock::meta_fields::Zahl(r + 1, fe), true);
      } else {
        r = fe - 4;
        if (r < p) __builtin_unreachable();
        pid += metablock::meta_fields::Zahl(p, r);
        v = LiesFest(r, 4);
      }
      u8_append::Anh(out, p, r);                 // page id delta, decimal
      out.push_back(kTab);
      {
        char b[24];
        for (int q = kRevDig - 1; q >= 0; --q) { b[q] = (char)('0' + (int)(v % 10)); v /= 10; }
        if (v) __builtin_unreachable();
        for (int q = 0; q < kRevDig; ++q) out.push_back((unsigned char)b[q]);
      }
      if (t) u8_append::Anh(out, t, e);          // TAB restrictions
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
    if (!t) __builtin_unreachable();
    const int fn = (int)(t - p);
    long long t_k = 0;
    if (Oben(k)) {
      if (fn <= 0 || fn >= 64) __builtin_unreachable();
      t_k = (k ? tvor : 0LL) + metablock::meta_fields::Zahl(p, p + fn);
      ++n_oben;
    } else {
      if (fn != 6) __builtin_unreachable();
      t_k = ts_progress::TsZuSek((int)LiesFest(p, 1), LiesFest(p + 1, 2),
                                 LiesFest(p + 3, 3));
      ++n_unten;
    }
    {
      Ts ts;
      ts_progress::SekZuTs(t_k, &ts.yy, &ts.mt, &ts.sek);
      SchreibTs(&out, ts);
    }
    u8_append::Anh(out, t, p + n);
    out.push_back(kNL);
    tvor = t_k;
  }

  u8_append::Anh(out, d.begin() + (long)bis, d.end());
  (void)n_oben; (void)n_unten;
  return SchreibDatei(pfad.c_str(), out);
}

}  // namespace metablock_form

#endif
