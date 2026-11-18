// The file side of the reversible text rules.  The encoder rewrites the stream
// in front of the text start (runner.cpp, behind the dictionary slot file);
// the decoder undoes it behind Unpack and the dictionary of slot 1, as a
// mirror image.  Both read the WRT dictionary from ./.dict (the coder's own;
// on the decoder side written from slot 1 just before).
#ifndef RULE_STAGES_FILE_H_
#define RULE_STAGES_FILE_H_
#if ((7) != 0) && !AUX_BUILD
#include <stdio.h>
#include <string>
#include <vector>
#include "enc_only.h"
#ifndef RS_ENC_ONLY
#define RS_ENC_ONLY ENC_ONLY
#endif
#include "rule_stages.h"
#include "ip_pack.h"   // LiesDatei

namespace rule_stages {

// the start of the rule region: the first byte behind page anchor number
// 71,361 ("  </page>\n" in WRT form); the stream end if it has fewer pages
inline size_t Anfang(const std::vector<unsigned char>& v) {
  static const unsigned char a[8] = {0x20, 0x20, 0x4c, 0x2f, 0xdf, 0x98, 0x4e, 0x0a};
  size_t q = 0;
  for (long k = 0; k < 71361; ++k) {
    size_t e = q;
    for (;;) {
      const void* f = e + 8 <= v.size() ? memchr(v.data() + e, 0x20, v.size() - e - 7) : nullptr;
      if (!f) return v.size();
      e = (size_t)((const unsigned char*)f - v.data());
      if (memcmp(v.data() + e, a, 8) == 0) break;
      ++e;
    }
    q = e + 8;
  }
  return q;
}
// stream, dictionary and start of a file stage
__attribute__((noinline)) inline bool Lies(const std::string& pfad, std::vector<unsigned char>* v,
                                            DictSpeicher* ds, size_t* a) {
  std::vector<unsigned char> d;
  if (!ip_pack::LiesDatei(".dict", &d) || d.empty() || !ds->FromBytes(d.data(), d.size())) return false;
  if (!ip_pack::LiesDatei(pfad.c_str(), v)) return false;
  *a = Anfang(*v);
  return true;
}
__attribute__((noinline)) inline bool Schreib(const std::string& pfad, Buf* o, DictSpeicher* ds, bool ok) {
  ok = ok && !o->err;
  if (ok) {
    FILE* f = fopen(pfad.c_str(), "wb");
    ok = f && fwrite(o->p, 1, o->n, f) == o->n;
    if (f) ok = fclose(f) == 0 && ok;
  }
  o->Free();
  ds->Free();
  return ok;
}
// the coder: the stages on the stream file in place; the decoder stages must
// give the input back byte for byte, else the run stops here
ENC_ONLY inline bool DateiEnc(const std::string& pfad) {
  std::vector<unsigned char> v;
  DictSpeicher ds;
  size_t a = 0;
  if (!Lies(pfad, &v, &ds, &a)) return false;
  Buf o;
  bool ok = Enc(ds.D(), v.data(), v.size(), a, &o) && !o.err;
  if (ok) {
    Buf r;
    ok = Dec(ds.D(), o.p, o.n, a, &r) && !r.err && r.n == v.size() && memcmp(r.p, v.data(), v.size()) == 0;
    r.Free();
  }
  std::vector<unsigned char>().swap(v);
  return Schreib(pfad, &o, &ds, ok);
}
// the decoder: the stages back, on the stream file in place
__attribute__((noinline)) inline bool DateiDec(const std::string& pfad) {
  std::vector<unsigned char> v;
  DictSpeicher ds;
  size_t a = 0;
  if (!Lies(pfad, &v, &ds, &a)) return false;
  Buf o;
  const bool ok = Dec(ds.D(), v.data(), v.size(), a, &o);
  std::vector<unsigned char>().swap(v);
  return Schreib(pfad, &o, &ds, ok);
}

}  // namespace rule_stages
#endif
#if ((7) != 0) && AUX_BUILD
// the build helper (cmix_aux) packs comp_hints/comp_order with -c: no text
// start, no rule stages
namespace rule_stages {
inline bool DateiEnc(const std::string&) { return true; }
inline bool DateiDec(const std::string&) { return true; }
}  // namespace rule_stages
#endif
#endif
