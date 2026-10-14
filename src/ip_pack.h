#ifndef IP_PACK_H_
#define IP_PACK_H_

#include "metablock.h"

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

namespace ip_pack {

static const int kBase = 16;
static const int kLen = 8;
static const unsigned char kHint = (unsigned char)0x04;

static_assert(0x80 >= 0x80 && 0x80 + 15 <= 0xE3,
    "the 16-value alphabet must lie in 0x80..0xE3 - "
    "outside it is not part of the frozen 208-byte set of the coder stream");
static_assert(0x04 == 0 ||
              0x04 < 0x80 ||
              0x04 > 0x80 + 15,
    "the hint byte lies INSIDE the alphabet - decoding would be ambiguous");

 inline const unsigned char* Alpha() {
  static unsigned char a[kBase];
  static bool init = false;
  if (!init) {
    for (int i = 0; i < kBase; ++i)
      a[i] = (unsigned char)(0x80 + i);
    init = true;
  }
  return a;
}

 inline const signed char* Index() {
  static signed char idx[256];
  static bool init = false;
  if (!init) {
    memset(idx, -1, sizeof(idx));
    const unsigned char* a = Alpha();
    for (int i = 0; i < kBase; ++i) idx[a[i]] = (signed char)i;
    init = true;
  }
  return idx;
}

// forward: four octets into kLen characters
inline void Kodiere(const unsigned char* o, unsigned char* out) {
  const unsigned char* a = Alpha();
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = a[o[i] >> 4];
    out[2 * i + 1] = a[o[i] & 15];
  }
}

// backward: kLen characters into the dotted literal; returns its length
inline int Dekodiere(const unsigned char* in, unsigned char* out) {
  const signed char* idx = Index();
  unsigned char o[4];
  for (int i = 0; i < 4; ++i)
    o[i] = (unsigned char)((idx[in[2 * i]] << 4) | idx[in[2 * i + 1]]);
  int n = 0;
  for (int i = 0; i < 4; ++i) {
    if (i) out[n++] = '.';
    unsigned v2 = o[i];
    if (v2 >= 100) out[n++] = (unsigned char)(48 + v2 / 100);
    if (v2 >= 10) out[n++] = (unsigned char)(48 + (v2 / 10) % 10);
    out[n++] = (unsigned char)(48 + v2 % 10);
  }
  return n;
}

inline bool IstZiffer(int c) { return c >= 48 && c <= 57; }

// Parse one IPv4 literal at p; returns its length or 0.  Strict rule: every
// octet 0..255, no leading zero, neighbours neither digit nor dot.
inline int LiesIp(const unsigned char* d, size_t n, size_t p, unsigned char* o) {
  if (p && (IstZiffer(d[p - 1]) || d[p - 1] == '.')) return 0;
  size_t q = p;
  for (int k = 0; k < 4; ++k) {
    if (k) {
      if (q >= n || d[q] != '.') return 0;
      ++q;
    }
    size_t s = q;
    int v = 0;
    while (q < n && IstZiffer(d[q]) && q - s < 3) {
      v = v * 10 + (d[q] - 48);
      ++q;
    }
    if (q == s || v > 255) return 0;
    if (q - s > 1 && d[s] == 48) return 0;
    o[k] = (unsigned char)v;
  }
  if (q < n && (IstZiffer(d[q]) || d[q] == '.')) return 0;
  return (int)(q - p);
}

 inline bool LiesDatei(const std::string& pfad, std::vector<unsigned char>* v) {
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

 inline bool SchreibDatei(const std::string& pfad,
                         const std::vector<unsigned char>& v) {
  FILE* f = fopen(pfad.c_str(), "wb");
  if (!f) return false;
  bool ok = v.empty() || fwrite(&v[0], 1, v.size(), f) == v.size();
  fclose(f);
  return ok;
}

inline void Wacht(bool block_fertig, unsigned long long n_ip,
                  unsigned long long kuerzung) {
  if (!block_fertig) return;
  if (n_ip == (unsigned long long)35618 &&
      kuerzung == (unsigned long long)143576) return;
  fprintf(stderr,
      "ip pack: compiled %llu addresses / %llu B, counted %llu / %llu\n",
      (unsigned long long)35618,
      (unsigned long long)143576, n_ip, kuerzung);
}

// Coder side: replace every IPv4 literal by hint plus kLen characters.
 inline bool Pack(const std::string& pfad) {
  std::vector<unsigned char> d;
  if (!LiesDatei(pfad, &d)) return false;
  std::vector<unsigned char> out;
  out.reserve(d.size());
  unsigned char o[4], blk[kLen];
  const size_t n = d.size();
  unsigned long long n_ip = 0;
  metablock::Grenze gr;
  bool im_block = false;
  for (size_t p = 0; p < n;) {
    im_block = gr.ImBlock();
    int len = IstZiffer(d[p]) ? LiesIp(&d[0], n, p, o) : 0;
    if (len && !im_block) len = 0;          // outside the block unchanged
    if (len) {
      Kodiere(o, blk);
      out.push_back(kHint);
      out.insert(out.end(), blk, blk + kLen);
      for (int q = 0; q < len; ++q) gr.Byte(d[p + (size_t)q]);
      p += (size_t)len;
      ++n_ip;
    } else {
      gr.Byte(d[p]);
      out.push_back(d[p]);
      ++p;
    }
  }
  Wacht(gr.Fertig(), n_ip, (unsigned long long)(n - out.size()));
  return SchreibDatei(pfad, out);
}

// Decoder side: the same stage backwards, the first thing after the coder.
 inline bool Unpack(const std::string& pfad) {
  std::vector<unsigned char> d;
  if (!LiesDatei(pfad, &d)) return false;
  const signed char* idx = Index();
  std::vector<unsigned char> out;
  out.reserve(d.size() + d.size() / 8);
  unsigned char txt[16];
  const size_t n = d.size();
  metablock::Grenze gr;
  for (size_t p = 0; p < n;) {
    const bool im_block = gr.ImBlock();
    if (
        im_block &&
        d[p] == kHint && p + 1 + (size_t)kLen <= n) {
      bool ok = true;
      for (int k = 0; k < kLen && ok; ++k) ok = idx[d[p + 1 + k]] >= 0;
      if (ok) {
        int m = Dekodiere(&d[p + 1], txt);
        out.insert(out.end(), txt, txt + m);
        for (int q = 0; q < m; ++q) gr.Byte(txt[q]);
        p += 1 + (size_t)kLen;
        continue;
      }
    }
    gr.Byte(d[p]);
    out.push_back(d[p]);
    ++p;
  }
  return SchreibDatei(pfad, out);
}

}

#endif
