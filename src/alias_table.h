#ifndef ALIAS_TABLE_H_
#define ALIAS_TABLE_H_

#include "redirect_table.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "alias_context_set.h"   // link bytes
#include "alias_substring_dedup.h"

namespace alias_table {

constexpr unsigned char kKap = 0x40;   // WRT mark: capital first letter
constexpr int kTextMax = 600;          // longer link targets and texts are dropped
constexpr int kTitelMax = 64;          // zgrammar.MAX_TITLE_TOKENS
constexpr int kTitelMin = 3;           // zgrammar.MIN_TITLE_TOKENS
constexpr int kHops = 12;              // up to twelve redirect hops
constexpr long long kRing = 1 << 16;

[[noreturn]] inline void Fail(const char* m) {
  fprintf(stderr, "alias table: %s\n", m);
  fflush(nullptr);
  exit(1);
}

inline int TitleKey(const unsigned char* p, int n, unsigned char* out) {
  int k = 0;
  for (int i = 0; i < n; ++i) {
    if (p[i] == 0x23) break;
    out[k++] = (p[i] == 0x5f) ? 0x20 : p[i];
  }
  return k;
}

inline int KapWeg(const unsigned char* p, int n) {
  return (n > 1 && p[0] == kKap) ? 1 : 0;
}

inline uint8_t Anders(const unsigned char* p, int n) {
  return memchr(p, 0x23, (size_t)n) || memchr(p, 0x5f, (size_t)n);
}

// ---- the hash: FNV-1a, 64 bit (Fowler, Noll, Vo; public domain) ----------
inline uint64_t Hash(const unsigned char* p, int n) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}
// Hash of the key form (cut at '#', '_' read as a blank).
inline uint64_t SchluesselHash(const unsigned char* p, int n) {
  std::vector<unsigned char> k(p, p + n);
  return Hash(k.data(), TitleKey(k.data(), n, k.data()));
}

struct Ref {
  uint64_t h; uint32_t off, idx; uint16_t len, eimer; uint8_t art, anders;
};

struct H64 { uint64_t h; };

struct Wahl {
  int hops;
  int max_len;                         // > 0: longer source titles are dropped with their chain
  const std::vector<uint32_t>* rang;   // article rank per source; nullptr = stream order
  bool aliase;                         // add link texts, drop duplicates
  unsigned long long* sub_ges;         // substring dedup: candidates / struck
  unsigned long long* sub;
};

// ---- the table ------------------------------------------------------------
class Tab {
 public:
  Tab() : ring_(kRing, 0), blk_{std::vector<std::vector<unsigned char> >(kEimer),
                                std::vector<std::vector<unsigned char> >(kEimer)} {
    for (int i = 0; i < 8; ++i) anker_wort_ = anker_wort_ << 8 | redirect_table::kAnker[i];
  }

  void Byte(unsigned char b) {
    ring_[(size_t)(pos_ & (kRing - 1))] = b;
    t_ = t_ << 8 | b;                         // the last eight bytes
    if (!open_.empty() || (unsigned char)(t_ >> 8) == 0x5b) Link(b);
    if (redirect_table::tab.Fertig()) Redirect();
    ++pos_;
    zu_ = false;
    anker_ = t_ == anker_wort_;
    if (anker_) {
      in_titel_ = true; tok_ = false; tn_ = 0;
      return;
    }
    if (in_titel_) {
      if (tn_ < kTitelMax + 1) tb_[tn_] = b;
      ++tn_;
      if (b == 0x0a) {
        in_titel_ = false;
        tok_ = zu_ = tn_ - 1 >= kTitelMin;
        seite_ok_ = false;                    // new page, new candidates
      } else if (tn_ > kTitelMax + 1) {
        in_titel_ = false;
      }
    }
  }
  void Schluss() { if (redirect_table::tab.Fertig()) Redirect(); }

  bool Anker() const { return anker_; }
  int TitelZu(const unsigned char** t) const { *t = tb_; return zu_ ? tn_ - 1 : 0; }

  void Kandidaten(const unsigned char* t, int n, const Wahl& w,
                  std::vector<Ref>* out, int* nq = nullptr, int* nt = nullptr) {
    Seite(t, n);
    if (w.hops > kHops) Fail("hops > 12");
    out->clear();
    std::vector<const Kand*> e;
    for (size_t i = 0; i < kand_.size(); ++i) {
      const Kand& c = kand_[i];
      if (c.tiefe < w.hops && (w.max_len <= 0 || c.maxlen <= w.max_len)) e.push_back(&c);
    }
    for (size_t x = 1; x < e.size(); ++x) {
      const Kand* v = e[x];
      const uint32_t r = w.rang ? (*w.rang)[v->r.idx] : v->r.idx;
      size_t y = x;
      for (; y > 0 && (w.rang ? (*w.rang)[e[y - 1]->r.idx] : e[y - 1]->r.idx) > r; --y) e[y] = e[y - 1];
      e[y] = v;
    }
    int a = 0, b = 0;
    for (size_t x = 0; x < e.size(); ++x) a += Nimm(e[x]->r, w.aliase, out);
    if (w.aliase) {
      for (size_t x = 0; x < ali_.size(); ++x) b += Nimm(ali_[x], true, out);
    }
    if (nq) *nq = a;
    if (nt) *nt = b;
    if (!w.sub) return;
    size_t k = 0;
    for (size_t x = 0; x < out->size(); ++x) {
      ++*w.sub_ges;
      if (Teil((*out)[x], tk_.data(), (int)tk_.size(), th_, true)) { ++*w.sub; continue; }
      (*out)[k++] = (*out)[x];
    }
    out->resize(k);
  }

  const unsigned char* Data(const Ref& r) const { return blk_[r.art][r.eimer].data() + r.off; }
  uint64_t K(const Ref& r) const {                  // hash of the key form
    return r.anders ? SchluesselHash(Data(r), r.len) : r.h;
  }
  bool Teil(const Ref& r, const unsigned char* tk, int tn, uint64_t th, bool drin) const {
    if (tn > 0 && !r.anders && (drin ? r.len >= tn : r.len <= tn)) return r.len == tn && r.h == th;
    std::vector<unsigned char> k(Data(r), Data(r) + r.len);
    k.resize((size_t)TitleKey(k.data(), (int)k.size(), k.data()));
    return drin ? alias_substring_dedup::Teilstring(k.data(), (int)k.size(), tk, tn)
                : alias_substring_dedup::Teilstring(tk, tn, k.data(), (int)k.size());
  }
  // Title key of the page asked last (used to drop the entry that contains
  // the title).
  const std::vector<unsigned char>& TitelSchluessel(uint64_t* th) const { *th = th_; return tk_; }

  unsigned long long links = 0, piped = 0, verworfen = 0, redirects = 0;
  size_t Eintraege() const { return (size_t)redirects + texte_; }
  size_t Bytes() const {
    size_t n = ring_.capacity() + quelle_.capacity() * sizeof(H64);
    for (int a = 0; a < 2; ++a) for (int i = 0; i < kEimer; ++i) n += blk_[a][i].capacity() + sizeof(blk_[a][i]);
    return n;
  }
  size_t Block_() const {
    size_t n = 0;
    for (int a = 0; a < 2; ++a) for (int i = 0; i < kEimer; ++i) n += blk_[a][i].size();
    return n;
  }
  size_t Texte_() const { return texte_; }
  size_t Quellen_() const { return quelle_.size(); }

 private:
  static constexpr int kEimerBits = 8;
  static constexpr int kEimer = 1 << kEimerBits;
  enum { kAlias = 0, kQuelle = 1 };
  struct Kand { Ref r; uint8_t tiefe, maxlen; };
  std::vector<unsigned char> ring_;
  long long pos_ = 0;
  uint64_t t_ = 0, anker_wort_ = 0;
  struct Cand { long long open, pipe; };
  std::vector<Cand> open_;
  bool in_titel_ = false, tok_ = false, zu_ = false, anker_ = false;
  int tn_ = 0;
  unsigned char tb_[kTitelMax + 1];
  std::vector<std::vector<unsigned char> > blk_[2];
  std::vector<H64> quelle_;                 // hash per source title (raw), in stream order
  size_t texte_ = 0;
  bool seite_ok_ = false;
  uint64_t seite_h_ = 0, th_ = 0;
  std::vector<unsigned char> tk_;
  std::vector<Kand> kand_;
  std::vector<Ref> ali_;

  static uint32_t Eimer(uint64_t h) { return (uint32_t)(h >> (64 - kEimerBits)); }
  static int Nimm(const Ref& r, bool einmal, std::vector<Ref>* out) {
    if (einmal) for (size_t i = 0; i < out->size(); ++i) if ((*out)[i].h == r.h) return 0;
    out->push_back(r);
    return 1;
  }
  static uint32_t Lies4(const unsigned char* p) { uint32_t v; memcpy(&v, p, 4); return v; }
  template <class F> void ForEach(int art, uint64_t z, F f) {
    const uint32_t ei = Eimer(z);
    const std::vector<unsigned char>& v = blk_[art][ei];
    const unsigned char* p = v.data();
    const unsigned char* e = p + v.size();
    while (p < e) {
      uint64_t h;
      memcpy(&h, p, 8);
      const unsigned char* t = p + 12;
      const unsigned char* q = (const unsigned char*)memchr(t, 0x0a, (size_t)(e - t));
      if (h == z) {
        const int n = (int)(q - t);
        f(Ref{Hash(t, n), (uint32_t)(t - v.data()), Lies4(p + 8), (uint16_t)n, (uint16_t)ei,
              (uint8_t)art, Anders(t, n)}, (unsigned char*)p + 8);
      }
      p = q + 1;
    }
  }
  void Leg(int art, uint64_t z, uint32_t vier, const unsigned char* t, int n) {   // append
    std::vector<unsigned char>& v = blk_[art][Eimer(z)];
    if (v.size() + 13 + (size_t)n > 0xFFFFFFFFu) Fail("sub-block over 4 GiB");
    v.insert(v.end(), (const unsigned char*)&z, (const unsigned char*)&z + 8);
    v.insert(v.end(), (const unsigned char*)&vier, (const unsigned char*)&vier + 4);
    v.insert(v.end(), t, t + n);
    v.push_back(0x0a);
  }

  void Seite(const unsigned char* t, int n) {
    const uint64_t h = Hash(t, n);
    if (seite_ok_ && seite_h_ == h) return;
    seite_ok_ = true; seite_h_ = h;
    tk_.assign(t, t + n);
    tk_.resize((size_t)TitleKey(tk_.data(), n, tk_.data()));
    th_ = Hash(tk_.data(), (int)tk_.size());
    kand_.clear(); ali_.clear();
    const int kap = KapWeg(t, n);
    const uint64_t k = Hash(t + kap, n - kap);
    std::vector<std::pair<uint64_t, uint8_t> > stage(1, std::make_pair(k, (uint8_t)0)), naechste;
    for (int d = 0; d < kHops && !stage.empty(); ++d) {
      naechste.clear();
      for (size_t i = 0; i < stage.size(); ++i) {
        const uint8_t ml0 = stage[i].second;
        ForEach(kQuelle, stage[i].first, [&](const Ref& r, unsigned char*) {
          if (r.h == h) return;
          const uint8_t ml = r.len > ml0 ? (uint8_t)r.len : ml0;
          kand_.push_back(Kand{r, (uint8_t)d, ml});
          const unsigned char* q = Data(r);
          const int o = KapWeg(q, r.len);
          naechste.push_back(std::make_pair(Hash(q + o, r.len - o), ml));
        });
      }
      stage.swap(naechste);
    }
    std::vector<uint32_t> zahl;
    ForEach(kAlias, k, [&](const Ref& r, unsigned char* vier) {
      ali_.push_back(r); zahl.push_back(Lies4(vier));
    });
    for (size_t x = 1; x < ali_.size(); ++x) {          // most common first, stable
      const Ref v = ali_[x];
      const uint32_t c = zahl[x];
      size_t y = x;
      for (; y > 0 && zahl[y - 1] < c; --y) { ali_[y] = ali_[y - 1]; zahl[y] = zahl[y - 1]; }
      ali_[y] = v; zahl[y] = c;
    }
  }

  // Link scanner (incremental) over the folded link form `[target|text]`.
  void Link(unsigned char b) {
    const long long q = pos_;
    const unsigned char prev = (unsigned char)(t_ >> 8), prev2 = (unsigned char)(t_ >> 16);
    if (prev == 0x5b && b != 0x5b && prev2 != 0x5b) {
      open_.push_back(Cand{q - 1, -1});
    }
    if (open_.empty()) return;
    if (b == 0x0a) { open_.clear(); return; }
    if (b == 0x51) {
      for (size_t i = 0; i < open_.size(); ++i) if (open_[i].pipe < 0) open_[i].pipe = q;
      return;
    }
    if (b != 0x5d) return;
    unsigned char key[kTextMax], txt[kTextMax];
    const long long ring_base = q >= kRing ? q - kRing + 1 : 0;
    for (size_t i = 0; i < open_.size(); ++i) {
      const Cand c = open_[i];
      ++links;
      const long long za = c.open + 1;
      if (za < ring_base) { ++verworfen; continue; }
      if (c.pipe < 0 || c.pipe <= za) continue;
      ++piped;
      int kn = 0;
      bool bad = false;
      for (long long x = za; x < c.pipe; ++x) {
        const unsigned char t = ring_[(size_t)(x & (kRing - 1))];
        if (t == 0x23) { bad = x == za; break; }
        if (kn == kTextMax) { bad = true; break; }
        key[kn++] = t;
      }
      if (bad || kn == 0) continue;
      kn = TitleKey(key, kn, key);
      if (kn == 0 || memchr(key, 0x4a, (size_t)kn)) continue;
      const long long tn = q - c.pipe - 1;          // text up to this `]`
      if (tn <= 0 || tn > kTextMax) continue;
      for (long long x = 0; x < tn; ++x) {
        txt[x] = ring_[(size_t)((c.pipe + 1 + x) & (kRing - 1))];
      }
      if (tn == kn && memcmp(txt, key, (size_t)kn) == 0) continue;
      const int o = KapWeg(key, kn);
      const uint64_t z = Hash(key + o, kn - o);
      bool da = false;
      ForEach(kAlias, z, [&](const Ref& r, unsigned char* vier) {
        if (da || r.len != tn || memcmp(Data(r), txt, (size_t)tn) != 0) return;
        const uint32_t v = Lies4(vier) + 1;
        memcpy(vier, &v, 4);
        da = true;
      });
      if (!da) { Leg(kAlias, z, 1, txt, (int)tn); ++texte_; }
    }
    open_.clear();
  }

  void Redirect() {
    const unsigned char* zp = nullptr;
    int kn = redirect_table::tab.Ziel(&zp);
    if (kn == 0 || !tok_) return;
    unsigned char key[kTextMax];
    if (kn > kTextMax) kn = kTextMax;
    kn = TitleKey(zp, kn, key);
    if (kn == 0) return;
    const int o = KapWeg(key, kn);
    const uint64_t z = Hash(key + o, kn - o);
    const int tlen = tn_ - 1, kap = KapWeg(tb_, tlen);
    if (z == Hash(tb_ + kap, tlen - kap)) return;
    const uint64_t roh = Hash(tb_, tlen);            // per title the first counts
    for (size_t i = 0; i < quelle_.size(); ++i) if (quelle_[i].h == roh) return;
    quelle_.push_back(H64{roh});
    Leg(kQuelle, z, (uint32_t)quelle_.size(), tb_, tlen);
    ++redirects;
  }
};
inline Tab tab;

struct SynMenge {
  uint64_t titel = 0;
  bool gesetzt = false;
  std::vector<H64> ids;               // at most `syn_max` entries
  unsigned long long n_merk = 0, n_ges = 0, n_p2 = 0;
  unsigned long long n_ohne_syn = 0, n_titel_ungleich = 0;

  void Neu(uint64_t t) { ids.clear(); titel = t; gesetzt = true; }
  void Merke(uint64_t h) { ids.push_back(H64{h}); ++n_merk; }
  bool Passt(uint64_t t) {
    if (!gesetzt) { ++n_ohne_syn; return false; }
    if (titel != t) { ++n_titel_ungleich; return false; }
    return true;
  }
  bool Kennt(uint64_t h) const {
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i].h == h) return true;
    return false;
  }
  void Verbraucht() { gesetzt = false; }
};
inline SynMenge& Syn() {
  static SynMenge m;
  return m;
}

}

#endif
