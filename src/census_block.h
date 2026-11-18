// Census block, stream level.
// The machine-written census pages of enwik9 (Rambot articles on US counties,
// cities, towns, villages, CDPs, townships and boroughs) follow a few fixed
// templates per census type.  The encoder cuts each such page into a stub (the
// title without state and type word, the text that deviates from the template,
// the names) and one record (template choices, gap bits, the numbers).  The
// records of all census pages form one line behind the metadata block, and the
// template texts are written once, as a pseudo article in slot 3 of the text
// start.  This puts the numbers together, where a number coder predicts each
// one from the fields coded before, the nearest places and the state, and the
// template text is coded once for all pages.
// This file puts the line into the coder stream behind the last metadata line
// (encoder) and takes it out again (decoder); it also holds the model side that
// reads the line and the contexts it injects.  The insertion point is the
// boundary rule of the metadata block reader (counted line feeds).  Page level
// (stubs, records, line form): readalike_prepr/censblock_seite.h; number coder:
// readalike_prepr/censblock_zahl.h.
#ifndef CENSUS_BLOCK_H_
#define CENSUS_BLOCK_H_
#include "readalike_prepr/censblock_seite.h"
#include "metablock.h"
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "preprocess/dictionary.h"
#include "census_title.h"

namespace census_block {

// The output alphabet mapping of the WRT stage (preprocessor.cpp encode_text /
// dictionary.cpp NextChar): the pseudo article is coded like a page.
inline int WrtTausch(int c) {
  if (c >= '{' && c < 127) c += 'P' - '{';
  else if (c >= 'P' && c < 'T') c -= 'P' - '{';
  else if ((c >= ':' && c <= '?') || (c >= 'J' && c <= 'O')) c ^= 0x70;
  if (c == 'X' || c == '`') c ^= 'X' ^ '`';
  return c;
}
// Through the stream coder of the pages (Encode) on memory
// streams; at most 3 output bytes per input byte (one-letter capital word
// with a 3-byte code and an escape byte), +1 at the end.
ENC_ONLY inline bool PseudoKodiere(FILE* dictionary, const std::string& roh, std::string* o) {
  if (!dictionary) return false;
  preprocessor::Dictionary d(dictionary, true, false);
  const size_t n = roh.size(), cap = 3 * n + 16;
  char* b = (char*)malloc(cap);
  FILE* in = b ? fmemopen((void*)roh.data(), n, "rb") : nullptr;
  FILE* out = in ? fmemopen(b, cap, "wb") : nullptr;
  if (!out) return false;
  d.Encode(in, (int)n, out);
  fflush(out);
  const long k = ftell(out);
  fclose(in);
  fclose(out);
  if (k < 0 || (size_t)k >= cap) return false;
  o->clear();
  for (long i = 0; i < k; ++i) o->push_back((char)WrtTausch((unsigned char)b[i]));
  free(b);
  return true;
}
// decodes the coded pseudo article up to and including its closing empty variant
inline bool PseudoDekodiere(FILE* dictionary, const std::string& kod, std::string* roh) {
  if (!dictionary || kod.empty()) return false;
  FILE* f = fmemopen((void*)kod.data(), kod.size(), "rb");
  if (!f) return false;
  preprocessor::Dictionary d(dictionary, false, true);
  roh->clear();
  bool ok = false;
  while (roh->size() < (1u << 24)) {
    if (ftell(f) > (long)kod.size()) break;
    roh->push_back((char)d.Decode(f));
    const size_t n = roh->size();
    if (n >= 2 && (*roh)[n - 1] == '\x0b' && (*roh)[n - 2] == '\x0b') { ok = true; break; }
  }
  const bool alles = ok && ftell(f) == (long)kod.size();
  fclose(f);
  return alles;
}

// The WRT-coded parts of the stripped titles, per state and type word: the
// suffix the title readers append (" <type word>, <state>") and the prompt line
// in front of the title ("<state><type word>").  Both sides read the WRT forms
// of the states and type words from the word list at the end of the coded
// pseudo article (one segment per word, closed by 0x0B).
struct TitelWrt {
  std::string zusatz[kStaaten][kTypworte];
  std::string line_buf[kStaaten][kTypworte];
  std::string staat[kStaaten], typ[kTypworte], kwort[kKontextWorte], census;
  std::string rech[kRechWorte];   // words of the computed context (anchor, hemispheres), WRT form
  std::string k1[kK1Worte];
  bool da = false;
};
inline TitelWrt titelwrt;
// kod: the coded pseudo article including its closing 0x0B 0x0B
inline bool TitelAusPseudo(const std::string& kod) {
  std::vector<std::string> seg;
  std::string a;
  for (size_t i = 0; i + 1 < kod.size(); ++i) {
    if (kod[i] == '\x0b') { seg.push_back(a); a.clear(); } else a.push_back(kod[i]);
  }
  const size_t nw = (size_t)kStaaten + (size_t)kTypworte - 1 + (size_t)kKontextWorte + 1 + (size_t)kRechWorte +
                    (size_t)kK1Worte;
  if (seg.size() < nw) return false;
  const std::string* st = &seg[seg.size() - nw];
  const std::string* ty = st + kStaaten;
  const std::string* kw = ty + kTypworte - 1;
  for (int si = 0; si < kStaaten; ++si) titelwrt.staat[si] = st[si];
  for (int ti = 1; ti < kTypworte; ++ti) titelwrt.typ[ti] = ty[ti - 1];
  for (int i = 0; i < kKontextWorte; ++i) titelwrt.kwort[i] = kw[i];
  titelwrt.census = kw[kKontextWorte];
  for (int i = 0; i < kK1Worte; ++i) titelwrt.k1[i] = kw[kKontextWorte + 1 + kRechWorte + i];
  for (int i = 0; i < kRechWorte; ++i) titelwrt.rech[i] = kw[kKontextWorte + 1 + i];
  for (int si = 0; si < kStaaten; ++si) {
    for (int ti = 0; ti < kTypworte; ++ti) {
      const std::string t = ti ? ty[ti - 1] : std::string();
      std::string& zu = titelwrt.zusatz[si][ti];   // t + ", " + state, without operator+ temporaries
      zu = t;
      zu += ", ";
      zu += st[si];
      std::string& ze = titelwrt.line_buf[si][ti];    // state + t + line feed
      ze = st[si];
      ze += t;
      ze += '\n';
    }
  }
  titelwrt.da = true;
  return true;
}

// The context row of the transformer in front of a stripped census title:
// "#<census words> <type word> [<title word>] <state>" + line feed, WRT-coded
// (tf1_prompt.cpp CensusZeile).  Type word: census type c t v p s b by its
// word; County and the other county words from the title (without its leading
// space); type x from a word in parentheses of the title; else none.  A place
// whose title ends in a county word ("Elizabeth Township, Allegheny County,
// Pennsylvania") gets that word too, so nothing of the stripped title is lost.
inline std::string KontextZeile(int typ, int tcode, int staat) {
  std::string w;
  if (typ >= 1 && typ <= 6) {
    w = titelwrt.kwort[typ - 1];
  } else if (tcode >= 1 && tcode <= 8) {
    w = titelwrt.typ[tcode];
    if (!w.empty() && w[0] == ' ') w.erase(0, 1);
  } else if (tcode >= 9 && tcode < kTypworte) {
    // (city) (town) (village) (CDP) (township) (borough) (census-designated place) (municipality)
    static const int kKlammer[] = {0, 1, 2, 3, 4, 5, 3, 6};
    w = titelwrt.kwort[kKlammer[tcode - 9]];
  }
  std::string z;
  z.push_back('#');
  z += titelwrt.census;
  z += ' ';
  if (!w.empty()) { z += w; z += ' '; }
  if (typ >= 1 && tcode >= 1 && tcode <= 8) {
    std::string w2 = titelwrt.typ[tcode];
    if (!w2.empty() && w2[0] == ' ') w2.erase(0, 1);
    z += w2;
    z += ' ';
  }
  z += titelwrt.staat[staat];
  z += '\n';
  return z;
}

// -e, RunCompression before the metablock codec: the block behind the last metablock line
ENC_ONLY inline bool Einfuegen(const std::string& pfad, const char* seitenpfad, FILE* dictionary) {
  FILE* bf = fopen(seitenpfad, "rb");
  if (!bf) return false;
  std::string blk;
  { char tb[1 << 16]; size_t g; while ((g = fread(tb, 1, sizeof tb, bf)) > 0) blk.append(tb, g); }
  fclose(bf);
  {
    // the pseudo article (texts, WRT-coded) in front of the census line
    const std::string pp = std::string(seitenpfad, strlen(seitenpfad)) + "p";
    FILE* pf = fopen(pp.c_str(), "rb");
    if (!pf) return false;
    std::string roh, kod;
    { char tb[1 << 16]; size_t g; while ((g = fread(tb, 1, sizeof tb, pf)) > 0) roh.append(tb, g); }
    fclose(pf);
    remove(pp.c_str());
    if (!PseudoKodiere(dictionary, roh, &kod)) return false;
    // every byte in the coder alphabet, and the closing 0x0B 0x0B only at the end
    size_t bad = kod.find("\x0b\x0b");
    for (size_t i = 0; i < kod.size() && bad == kod.size() - 2; ++i)
      if (!ImAlphabet((unsigned char)kod[i])) bad = i;
    if (kod.size() < 2 || bad != kod.size() - 2) {
      __builtin_unreachable();
      return false;
    }
    // slot 3 of the text start: the content without the closing
    // 0x0B 0x0B (the end mark of the slot takes their place) to the hand-over
    // file of the text-start stage, which puts it behind the census line
    if (kod.size() < 2) return false;
    FILE* sf = fopen(".vs3", "wb");
    if (!sf) return false;
    const bool sok = fwrite(kod.data(), 1, kod.size() - 2, sf) == kod.size() - 2;
    if (fclose(sf) != 0 || !sok) return false;
  }
  FILE* in = fopen(pfad.c_str(), "rb");
  if (!in) return false;
  const std::string tmp = pfad + ".cb";
  FILE* out = fopen(tmp.c_str(), "wb");
  if (!out) { fclose(in); return false; }
  metablock::Grenze gr;
  std::vector<char> buf(1 << 20);
  bool drin = false;
  size_t g;
  while ((g = fread(buf.data(), 1, buf.size(), in)) > 0) {
    size_t a = 0;
    if (!drin) {
      for (size_t i = 0; i < g; ++i) {
        gr.Byte((unsigned char)buf[i]);
        if (gr.Fertig()) {
          fwrite(buf.data(), 1, i + 1, out);
          fwrite(blk.data(), 1, blk.size(), out);
          a = i + 1;
          drin = true;
          break;
        }
      }
      if (!drin) { fwrite(buf.data(), 1, g, out); continue; }
    }
    fwrite(buf.data() + a, 1, g - a, out);
  }
  fclose(in);
  if (fclose(out) != 0 || !drin) {
    __builtin_unreachable();
    return false;
  }
  remove(seitenpfad);
  return rename(tmp.c_str(), pfad.c_str()) == 0;
}

// ------------------------------------------------------------ the decoder --
// canonical stream -> block (side file) + stream without the block
inline bool Entnehmen(const std::string& pfad, const char* seitenpfad, FILE* dictionary) {
  FILE* in = fopen(pfad.c_str(), "rb");
  if (!in) return false;
  const std::string tmp = pfad + ".cb";
  FILE* out = fopen(tmp.c_str(), "wb");
  FILE* bf = fopen(seitenpfad, "wb");
  if (!out || !bf) { fclose(in); return false; }
  metablock::Grenze gr;
  std::vector<unsigned char> buf;
  buf.resize(1 << 20);   // as the (n) constructor: zeros, through the resize instance
  int zone_state = 0;   // 0 metablock, 1 census line, 2 rest
  {
    // slot 3 of the text start: the text-start stage wrote the pseudo article
    // back to its hand-over file; the side file starts with it
    std::string pk;
    FILE* sf = fopen(".vs3", "rb");
    if (!sf) {   // slot 3 always carries the pseudo article
      fclose(in); fclose(out); fclose(bf);
      return false;
    }
    char tb[1 << 16];
    size_t n;
    while ((n = fread(tb, 1, sizeof tb, sf)) > 0) pk.append(tb, n);
    fclose(sf);
    remove(".vs3");
    pk.append("\x0b\x0b", 2);
    std::string roh;
    if (!PseudoDekodiere(dictionary, pk, &roh) || fwrite(roh.data(), 1, roh.size(), bf) != roh.size()) {
      fclose(in); fclose(out); fclose(bf);
      return false;
    }
  }
  size_t g;
  while ((g = fread(buf.data(), 1, buf.size(), in)) > 0) {
    size_t a = 0;
    if (zone_state == 0) {
      for (size_t i = 0; i < g; ++i) {
        gr.Byte((unsigned char)buf[i]);
        if (gr.Fertig()) {
          fwrite(buf.data(), 1, i + 1, out);
          a = i + 1;
          zone_state = 1;   // the census line (the pseudo article is slot 3)
          break;
        }
      }
      if (zone_state == 0) { fwrite(buf.data(), 1, g, out); continue; }
    }
    if (zone_state == 1) {
      const unsigned char* nl = (const unsigned char*)memchr(buf.data() + a, '\n', g - a);
      const size_t e = nl ? (size_t)(nl - buf.data()) + 1 : g;
      fwrite(buf.data() + a, 1, e - a, bf);
      a = e;
      if (nl) zone_state = 2;
    }
    fwrite(buf.data() + a, 1, g - a, out);
  }
  fclose(in);
  const bool zu1 = fclose(out) == 0;
  const bool zu2 = fclose(bf) == 0;
  const bool ok = zu1 && zu2;
  if (!ok || zone_state != 2) {
    __builtin_unreachable();
    return false;
  }
  return rename(tmp.c_str(), pfad.c_str()) == 0;
}

// The metablock data of page s that the model side injects before the flags of
// its record: revid / 10^6 and day number / 30 (one base-100 digit each), then
// 0x0B; the author is not part of it.  Returns the number of bytes in b (<= 3).
inline int KontextBytes(size_t s, unsigned char* b) {
  int n = 0;
  if (s < metablock::mk.SatzZahl()) {
    const unsigned rv = metablock::mk.RevidVon(s), tg = metablock::mk.TagnrVon(s);
    b[n++] = (unsigned char)(0x80 + (rv / 1000000) % 100);
    b[n++] = (unsigned char)(0x80 + (tg / 30) % 100);
  }
  b[n++] = 0x0B;
  return n;
}

// Computed context.  Behind the anchor of a rule (WRT form, the last bytes of
// the stream) the census automaton injects "$" + the text the rule computes for
// the running page into cmix and the transformer (observed, not coded).  There
// is one rule (RechDms, the coordinate sentence); an empty anchor means none.
inline std::string rech_anker;
// the DMS pair of the page's record coordinate (rounded seconds), as the stub writes it
inline bool RechDms(long long page_index, std::string* out) {
  size_t n1 = 0, n2 = 0;
  const char* la = CensWert(page_index, zahl::F_LAT, &n1);
  const char* lo = CensWert(page_index, zahl::F_LON, &n2);
  if (!la || !lo) { la = CensWert(page_index, zahl::F_MLAT, &n1); lo = CensWert(page_index, zahl::F_MLON, &n2); }
  if (!la || !lo) return false;
  out->clear();
  for (int h = 0; h < 2; ++h) {
    const std::string w(h ? lo : la, h ? n2 : n1);
    long long v = 0;
    int nk = 0;
    if (!DmsDez(w, 0, &v, &nk) || DmsDez(w, 0, &v, &nk) != w.size()) return false;
    std::string x[3];
    DmsAus(v, nk, true, x);
    if (h) *out += ", ";
    *out += x[0];
    *out += kGradStrom;
    *out += x[1];
    *out += '\'';
    *out += x[2];
    *out += "\" ";
    *out += titelwrt.rech[1 + 2 * h + (w[0] == '-' ? 1 : 0)];
  }
  // the decimal pair of the record as the sentence writes it behind the DMS pair
  out->append(" (").append(la, n1).append(", ").append(lo, n2);
  *out += ')';
  return true;
}
inline void RechRegelnSetzen() {
  rech_anker.clear();
  if (titelwrt.rech[0].empty()) return;
  rech_anker = titelwrt.rech[0];   // "located at " (also behind "@" or a line feed)
  rech_anker += ' ';
}

inline bool K1Text(long long page_index, const std::string& name, std::string* out) {
  const char t = CensTyp(page_index);
  if (!t) return false;
  const std::string* W = titelwrt.k1;
  int ti = 0;
  while (kTypen[ti] && kTypen[ti] != t) ++ti;
  const std::string& T = W[K1_TYP + ti];
  auto v = [&](int f, std::string* x) {
    size_t n = 0;
    const char* p = CensWert(page_index, f, &n);
    // a hand-edited census sentence: the twin of the intro or household sentence
    if ((!p || !n) && f == zahl::F_POP) p = CensWert(page_index, zahl::F_POP_I, &n);
    if ((!p || !n) && f == zahl::F_HH) p = CensWert(page_index, zahl::F_HH2, &n);
    if (!p || !n) return false;
    x->assign(p, n);
    return true;
  };
  std::string& o = *out;
  o.clear();
  // the units " km² (" and " mi²)": built once, appended like the words
  std::string KM = W[K1_KM], MI = W[K1_MI];
  KM += kQuadratStrom;
  KM += W[K1_KLA];
  MI += kQuadratStrom;
  MI += W[K1_ZU];
  std::string a, b, c, d;
  if (v(zahl::F_POP, &a)) {
    if (t == 'k') {
      o += W[K1_POPK];
    } else {
      o += W[K1_POPO1];
      o += T;
      o += W[K1_POPO2];
    }
    o += a;
    std::string teil[3];
    int nt = 0;
    static const int kTf[3] = {zahl::F_HH, zahl::F_FAM, zahl::F_HU};
    static const int kTw[3] = {K1_HH, K1_FAM, K1_HU};
    for (int j = 0; j < 3; ++j)
      if (v(kTf[j], &teil[nt])) teil[nt++] += W[kTw[j]];
    for (int i = 0; i < nt; ++i) {
      o += i == 0 ? W[K1_WITH] : (i + 1 == nt ? W[K1_AND] : W[K1_KOMMA]);
      o += teil[i];
    }
    o += W[K1_PUNKT];
  }
  // "<x> km² (<y> mi²)"
  auto paar = [&](const std::string& x, const std::string& y) { o += x; o += KM; o += y; o += MI; };
  if (v(zahl::F_TOT_KM, &a) && v(zahl::F_TOT_MI, &b)) {
    o += W[K1_THE];
    o += T;
    o += W[K1_AREA];
    paar(a, b);
    if (v(zahl::F_LAND_KM, &c) && v(zahl::F_LAND_MI, &d)) {
      o += W[K1_DP];
      paar(c, d);
      o += W[K1_LAND];
      if (v(zahl::F_WAT_KM, &a) && v(zahl::F_WAT_MI, &b)) {
        o += W[K1_AND];
        paar(a, b);
        o += W[K1_WATER];
        if (v(zahl::F_WAT_PCT, &a)) {
          o += W[K1_KOMMA];
          o += a;
          o += W[K1_PCT];
        }
      }
    }
    o += W[K1_PUNKT];
  }
  if (v(zahl::F_DENS_KM, &a) && v(zahl::F_DENS_MI, &b)) {
    o += W[K1_DENS];
    o += a;
    o += W[K1_KMD];
    o += kQuadratStrom;
    o += W[K1_KLA];
    o += b;
    o += W[K1_MID];
    o += kQuadratStrom;
    o += W[K1_KLZ];
  }
  if ((v(zahl::F_LAT, &a) && v(zahl::F_LON, &b)) || (v(zahl::F_MLAT, &a) && v(zahl::F_MLON, &b))) {
    long long la = 0, lo = 0;
    int nla = 0, nlo = 0;
    if (DmsDez(a, 0, &la, &nla) == a.size() && DmsDez(b, 0, &lo, &nlo) == b.size()) {
      std::string x[3], y[3];
      DmsAus(la, nla, true, x);
      DmsAus(lo, nlo, true, y);
      // "<d>°<m>'<s>" <hemisphere>"
      auto dms = [&](const std::string* z, int hw) {
        o += z[0];
        o += kGradStrom;
        o += z[1];
        o += '\'';
        o += z[2];
        o += "\" ";
        o += W[hw];
      };
      o += ' ';
      o += name;
      o += W[K1_LOC];
      dms(x, a[0] == '-' ? K1_SOUTH : K1_NORTH);
      o += W[K1_KOMMA];
      dms(y, b[0] == '-' ? K1_WEST : K1_EAST);
      o += W[K1_KLA];
      o += a;
      o += W[K1_KOMMA];
      o += b;
      o += W[K1_KLZ];
    }
  }
  if (v(zahl::F_INC_HH, &a)) {
    o += W[K1_INC1];
    o += a;
    if (v(zahl::F_INC_FAM, &b)) { o += W[K1_INC2]; o += b; }
    if (v(zahl::F_INC_PC, &c)) { o += W[K1_INC3]; o += c; }
    o += W[K1_PUNKT];
  }
  if (!o.empty() && o[0] == ' ') o.erase(0, 1);
  return !o.empty();
}

class Aut {
 public:
  __attribute__((noinline)) void Observe(unsigned char c) {
    okb_n_ = 0;
    tf_ = false;
    zeile_endet_ = false;
    if (z_ == kAus) {
      if (metablock::mk.MetaEndet()) {
        z_ = kZMagic;
        n_ = 0;
      }
      return;
    }
    if (z_ == kFertig) {
      // the pages behind the census line: a page anchor starts the next page
      anker_ = anker_ << 8 | c;
      if (anker_ == kAnkerWort) SeiteStart(seite_akt_ + 1);
      Rechne(c);
      return;
    }
    if (saetze_an_) saetze_.push_back((char)c);
    Schritt(c);
  }
  int TakeBlock(const unsigned char** p) { *p = okb_; return okb_n_; }
  // The pseudo article is slot 3 of the text start, behind the census line; at
  // its end mark (p, n = the slot content) the word list sets the title tables
  // and the computed-context rules, and page 0 starts (the next stream byte is
  // its first).
  void Slot3(const unsigned char* p, size_t n) {
    if (z_ != kFertig) Bruch(0x0B);   // the census line is read in front of slot 3
    std::string kod((const char*)p, n);
    kod.append("\x0b\x0b", 2);      // the closing pair the slot's end mark replaced
    if (!TitelAusPseudo(kod)) Bruch(0x0B);   // the word list of the titles
    RechRegelnSetzen();
    TitelFertig();
  }
  // computed context: behind the anchor of a rule its text, "$" in front, into cmix and the transformer
  void Rechne(unsigned char c) {
    ring_.push_back((char)c);
    if (ring_.size() > 32) ring_.erase(0, ring_.size() - 32);
    const std::string& a = rech_anker;
    if (a.empty() || ring_.size() < a.size() ||
        memcmp(ring_.data() + ring_.size() - a.size(), a.data(), a.size()) != 0)
      return;
    std::string t;
    if (!RechDms(seite_akt_, &t)) return;
    t += '$';                                    // the anchor once more: the copy keeps its context
    t += a;
    if (t.size() > sizeof(okb_)) return;
    kende_ = okb_n_ + (int)t.size() - (1 + (int)a.size());
    Inject((const unsigned char*)t.data(), (int)t.size());
    tf_ = true;
  }
  long long SeiteAkt() const { return seite_akt_; }
  int EndeNach(int n) {
    const int k = kende_;
    kende_ = 0;
    return k > 0 ? n - k + 1 : 0;
  }
  int kende_ = 0;   // bytes of the block up to and including the computed text
  bool TfSieht() const { return tf_; }
  // the line feed that ends the census line (= the end of the whole metablock)
  bool ZeileEndet() const { return zeile_endet_; }
  // the statistics of the census automaton (the release build prints none)
  void Report(const char* page_index) const {
    (void)page_index;
  }

 private:
  enum { kAus, kZMagic, kNPos, kNVar, kLab, kStaat, kFlag, kAnzahl, kVarNr, kLuecke, kWert, kSatzEnde,
         kNLab, kJahr, kTCode, kLaufKopf, kLauf, kZeilenEnde, kFertig };
  static constexpr unsigned long long kAnkerWort = 0x20204c2fdf984e0aull;   // page anchor
  // a page begins: the suffix the title readers append (empty on a page
  // without a stripped title)
  void SeiteStart(long long k) {
    seite_akt_ = k;
    const auto it = std::lower_bound(census_title::seiten.begin(), census_title::seiten.end(), k);
    if (it == census_title::seiten.end() || *it != k) {
      census_title::zusatz.clear();
      census_title::zusatz_i = -1;
      return;
    }
    const size_t i = (size_t)(it - census_title::seiten.begin());
    census_title::zusatz = census_title::zusaetze[i];
    census_title::zusatz_i = (long long)i;
  }
  // the census line is read: the stripped titles sorted by page
  void TitelFertig() {
    qsort(trec_.data(), trec_.size(), sizeof(trec_[0]), [](const void* x, const void* y) {
      const long long* a = (const long long*)x;
      const long long* b = (const long long*)y;
      for (int k = 0; k < 4; ++k)
        if (a[k] != b[k]) return a[k] < b[k] ? -1 : 1;
      return 0;
    });
    census_title::seiten.clear();
    census_title::line_buf.clear();
    census_title::kontext.clear();
    census_title::zusaetze.clear();
    for (const auto& r : trec_) {
      census_title::seiten.push_back(r[0]);
      census_title::line_buf.push_back(titelwrt.line_buf[r[1]][r[2]]);
      const std::string kz = KontextZeile((int)r[3], (int)r[2], (int)r[1]);
      census_title::kontext.push_back(kz);
      census_title::zusaetze.push_back(titelwrt.zusatz[r[1]][r[2]]);
    }
    trec_.clear();
    trec_.shrink_to_fit();
    SeiteStart((long long)0);   // the first page follows the census line directly
  }
  [[noreturn]] void Bruch(unsigned char c) {
    __builtin_unreachable();
    exit(3);
  }
  // digit of a base-100 field; true when the field of w digits is complete
  bool Ziffer100(unsigned char c, int w) {
    if (c < 0x80 || c > 0xE3) Bruch(c);
    acc_ = acc_ * 100 + (c - 0x80);
    return ++n_ == w;
  }
  void Inject(const unsigned char* b, int n) {
    for (int i = 0; i < n && okb_n_ < (int)sizeof(okb_); ++i) okb_[okb_n_++] = b[i];
  }
  void Feldkennung(size_t i) {
#if PORT_CMX_MEAS
    zahl::g_cur_page = seite_;   // research V2 (censblock_zahl.h)
#endif
    const unsigned char b = (unsigned char)(0x80 + felder_[i]);
    Inject(&b, 1);
    {   // the scale byte: half-octave class of the estimate the number coder uses for this token
      zahl::SkalaEst sk;
      std::vector<std::string> w;
      std::vector<char> bek;
      Zahlen(spiegel_[(size_t)rt_], rt_ == 0, staat_, slots_, &w, &toks_, false, nullptr, (long)i, nullptr, &sk, &bek);
      const int sb = zahl::SkalaByte(sk);
      if (sb >= 0) {
        const unsigned char x = (unsigned char)sb;
        Inject(&x, 1);
      }
      {   // X fields: learn from the X token in front, inject the predicted field value in front of this one
        std::vector<unsigned char> lab;
        for (const Slot& q : slots_) lab.push_back(q.lab);
        if (i == 0) xfi_.vor = zahl::XfiTafel::kStart;
        if (i > 0 && felder_[i - 1] == zahl::F_X) xfi_.Lerne(rt_, zahl::XfiTreffer(lab, w, bek, toks_[i - 1]));
        if (felder_[i] == zahl::F_X) {
          const int pf = xfi_.Vorhersage(rt_);
          const std::string* v = pf >= 0 ? zahl::XfiWert(lab, w, bek, pf) : nullptr;
          if (v) {
            const int nv = (int)(v->size() < 40 ? v->size() : 40);
            Inject((const unsigned char*)v->data(), nv);
            const unsigned char e = 0x0B;
            Inject(&e, 1);
          }
        }
      }
    }
  }
  void Kontext() {   // metablock data of page seite_ before its flags
    unsigned char b[48];
    Inject(b, KontextBytes((size_t)seite_, b));
  }
  // the next record: type and page from the run (read first, behind the forms)
  void SatzStart() {
    if (!lauf_da_) { z_ = kLaufKopf; n_ = 0; acc_ = 0; return; }
    if (lk_ >= lauf_.size()) { z_ = kZeilenEnde; return; }
    rt_ = lauf_[lk_].first;
    seite_ = lauf_[lk_].second;
    ++lk_;
    z_ = kStaat; n_ = 0; acc_ = 0;
  }
  // the head of a record is read (type, page, state, type word): its metablock
  // context and the flags of its first position
  void KopfFertig() {
    // as the decoder of the line: a County record joins the county list of its state
    if (rt_ == 0) zahl::kreisliste[(size_t)staat_].push_back(seite_);
    trec_.push_back({(long long)seite_, (long long)staat_, (long long)tcode_, (long long)rt_});
    slots_.clear(); inst_ = 0; pi_ = 0;
    Kontext();
    if (form_[(size_t)rt_].empty()) { NachFlags(); return; }
    FlagStart();
  }
  void FlagStart() {   // before the flag of position pi_
    z_ = kFlag; n_ = 0; acc_ = 0;
    unsigned char b[2] = {(unsigned char)(0x80 + pi_), 0x80};
    Inject(b, 1);
  }
  void NachFlags() {
    nluecke_ = (int)((inst_ + 1 + 5) / 6);
    z_ = kLuecke; n_ = 0; acc_ = 0;
  }
  void NachLuecken() {
    // the order of the value tokens: a dry run of the number coder
    felder_.clear();
    KreisZusatz(kTypen[rt_], &slots_);   // the county of a place, when no unit carries it
    std::vector<std::string> w, t;
    if (!Zahlen(trocken_, rt_ == 0, staat_, slots_, &w, &t, false, &felder_) || felder_.size() != slots_.size())
      Bruch(0);
    wi_ = 0;
    toks_.clear();
    tokcur_.clear();
    Anl4Setzen(rt_, seite_, staat_);   // the counties of the records of earlier types (as LiesSaetze)
    if (felder_.empty()) { z_ = kSatzEnde; return; }
    z_ = kWert;
    Feldkennung(0);
  }
  void SatzFertig() {   // the mirror of the coder model learns the record
#if PORT_CMX_MEAS
    zahl::g_cur_page = seite_;   // research V2 (censblock_zahl.h)
#endif
    std::vector<std::string> w;
    if (!Zahlen(spiegel_[(size_t)rt_], rt_ == 0, staat_, slots_, &w, &toks_, false)) Bruch(0x0B);
    if (!felder_.empty() && felder_.back() == zahl::F_X && w.size() == slots_.size()) {   // the last X token
      std::vector<unsigned char> lab;
      for (const Slot& q : slots_) lab.push_back(q.lab);
      const std::vector<char> alle(slots_.size(), 1);
      xfi_.Lerne(rt_, zahl::XfiTreffer(lab, w, alle, toks_.back()));
    }
    long kr = -1;   // as LiesSaetze: the county of the record, for the records of later types
    for (size_t q = 0; q < slots_.size() && q < w.size(); ++q)
      if (slots_[q].lab == zahl::F_KREIS) { kr = strtol(w[q].c_str(), nullptr, 10); break; }
    Anl4Merken(rt_, seite_, staat_, rt_ == 0 ? seite_ : kr);
  }
  void FlagWeiter() {   // next position of the record, or the gaps
    ++pi_;
    if (pi_ >= form_[(size_t)rt_].size()) NachFlags();
    else FlagStart();
  }
  void Instanz(int v) {
    const auto& vs = form_[(size_t)rt_][pi_];
    if (v < 0 || (size_t)v >= vs.size()) Bruch((unsigned char)v);
    const VarD& vd = vs[(size_t)v];
    for (size_t k = 0; k < vd.lab.size(); ++k) {
      Slot sl;
      sl.lab = vd.lab[k];
      sl.jahr = vd.jahr[k];
      slots_.push_back(sl);
    }
    ++inst_;
  }
  void VarFertig() {
    if (++vi_ < nvar_) { z_ = kNLab; return; }
    if (++pi_ < form_[(size_t)ti_].size()) { z_ = kNVar; n_ = 0; acc_ = 0; return; }
    if (++ti_ == kNTypen) KopfFertigAlle(); else { z_ = kNPos; n_ = 0; acc_ = 0; }
  }
  // the head of the line is read: from the next byte on the records
  void KopfFertigAlle() {
    saetze_an_ = true;
    // the mirror starts with the state the decoder of the line starts with (LiesSaetze)
    anl4_tab.clear();
    KreislisteLeer();
    SatzStart();
  }
  // the line is read: decode its records once (the values per page)
  void WerteFertig() {
    saetze_an_ = false;
    if (!WerteAusZeile(saetze_, form_)) Bruch('\n');   // form_ moves out (no reader behind the line)
    { std::string w(std::move(saetze_)); }   // the buffer goes
  }
  void Schritt(unsigned char c) {
    switch (z_) {
      case kZMagic:
        if (c != (unsigned char)census_block::kMagic[n_]) Bruch(c);
        if (++n_ == 4) { z_ = kNPos; ti_ = 0; n_ = 0; acc_ = 0; }   // once: form_ is empty
        return;
      case kNPos:
        if (!Ziffer100(c, 2)) return;
        std::vector<std::vector<VarD>>((size_t)acc_).swap(form_[(size_t)ti_]);   // empty before: as resize(acc_)
        pi_ = 0;
        n_ = 0; acc_ = 0;
        if (form_[(size_t)ti_].empty()) { if (++ti_ == kNTypen) KopfFertigAlle(); return; }
        z_ = kNVar;
        return;
      case kNVar:
        if (!Ziffer100(c, 2)) return;
        nvar_ = acc_; vi_ = 0; n_ = 0; acc_ = 0;
        z_ = kNLab;
        if (nvar_ == 0) Bruch(c);
        return;
      case kNLab:     // zcb3: number of labels of the variant (its text is in the pseudo article)
        if (!Ziffer100(c, 1)) return;
        form_[(size_t)ti_][pi_].emplace_back();
        nlab_ = (int)acc_;
        n_ = 0; acc_ = 0;
        if (nlab_ == 0) { VarFertig(); return; }
        z_ = kLab;
        return;
      case kJahr:     // zcb3: the year of a row of the population table
        if (!Ziffer100(c, 2)) return;
        form_[(size_t)ti_][pi_].back().jahr.back() = (int)acc_;
        n_ = 0; acc_ = 0;
        if (--nlab_ == 0) VarFertig(); else z_ = kLab;
        return;
      case kLab:
        if (!Ziffer100(c, 1)) return;
        form_[(size_t)ti_][pi_].back().lab.push_back((unsigned char)acc_);
        {
          const int j0 = 0;   // lvalue: the push_back of the decoder (.text)
          form_[(size_t)ti_][pi_].back().jahr.push_back(j0);
        }
        if (acc_ == zahl::F_PV) { n_ = 0; acc_ = 0; z_ = kJahr; return; }
        n_ = 0; acc_ = 0;
        if (--nlab_ == 0) VarFertig();
        else z_ = kLab;
        return;
      case kStaat:
        if (!Ziffer100(c, 1)) return;
        staat_ = (int)acc_; n_ = 0; acc_ = 0;
        z_ = kTCode;
        return;
      case kTCode:   // type word code of the stripped title
        if (!Ziffer100(c, 1)) return;
        if (acc_ >= kTypworte || staat_ < 0 || staat_ >= kStaaten) Bruch(c);
        tcode_ = (int)acc_;
        KopfFertig();   // sets the next state
        return;
      case kLaufKopf:   // first page and length of the census range
        if (!Ziffer100(c, 6)) return;
        lr0_ = acc_ / 1000000;
        lrn_ = acc_ % 1000000;
        lri_ = 0;
        n_ = 0; acc_ = 0;
        lauf_.clear();
        for (auto& v : lsei_) v.clear();
        z_ = kLauf;
        if (lrn_ == 0) { lauf_da_ = true; SatzStart(); }
#if PORT_CMX_MEAS
        else if (zahl::g_var & 4) { unsigned char b[48]; Inject(b, KontextBytes((size_t)lr0_, b)); }   // research RK
#endif
        return;
      case kLauf: {   // one symbol per page of the range
        const int v = (int)c - 0x80;
        if (v < 0 || v > kNTypen) Bruch(c);
        if (v < kNTypen) lsei_[(size_t)v].push_back(lr0_ + lri_);
#if PORT_CMX_MEAS
        if (++lri_ < lrn_) {
          if (zahl::g_var & 4) { unsigned char b[48]; Inject(b, KontextBytes((size_t)(lr0_ + lri_), b)); }   // research RK
          return;
        }
#else
        if (++lri_ < lrn_) return;
#endif
        for (int ti = 0; ti < kNTypen; ++ti) for (long pg : lsei_[(size_t)ti]) lauf_.push_back(std::make_pair(ti, pg));
        for (auto& w : lsei_) { w.clear(); w.shrink_to_fit(); }
        lauf_da_ = true;
        SatzStart();
        return;
      }
      case kZeilenEnde:
        if (c != '\n') Bruch(c);
        z_ = kFertig;
        zeile_endet_ = true;
        WerteFertig();
        return;
      case kFlag:
        if (!Ziffer100(c, 1)) return;
        if (acc_ == 0) { FlagWeiter(); return; }
        if (acc_ == 99) { z_ = kAnzahl; n_ = 0; acc_ = 0; return; }
        Instanz((int)acc_ - 1);
        FlagWeiter();
        return;
      case kAnzahl:
        if (!Ziffer100(c, 2)) return;
        anzahl_ = acc_; n_ = 0; acc_ = 0;
        if (anzahl_ == 0) { FlagWeiter(); return; }
        z_ = kVarNr;
        return;
      case kVarNr:
        if (!Ziffer100(c, 1)) return;
        Instanz((int)acc_);
        n_ = 0; acc_ = 0;
        if (--anzahl_ == 0) FlagWeiter();
        return;
      case kLuecke:
        if (c < 0x80 || c > 0xE3) Bruch(c);
        if (++n_ == nluecke_) NachLuecken();
        return;
      case kWert:
        if (c != '\t') {
          tokcur_.push_back((char)c);
          return;
        }
        toks_.push_back(tokcur_);
        tokcur_.clear();
        if (++wi_ < felder_.size()) Feldkennung(wi_);
        else z_ = kSatzEnde;
        return;
      case kSatzEnde:
        if (c != 0x0B) Bruch(c);
        SatzFertig();
        SatzStart();
        return;
      default:
        Bruch(c);
    }
  }

  int z_ = kAus, n_ = 0;
  bool zeile_endet_ = false;
  std::string ring_;                 // the last stream bytes of the pages (rule anchors)
  bool saetze_an_ = false;   // the record bytes are collected (head read, line not ended)
  std::string saetze_;       // the record bytes of the census line, up to its line feed
  long acc_ = 0;
  std::vector<std::vector<VarD>> form_[kNTypen];   // type, position, variant (the types as a fixed array)
  int ti_ = 0;
  size_t pi_ = 0;
  long nvar_ = 0, vi_ = 0, anzahl_ = 0;
  int nlab_ = 0;
  int rt_ = -1, staat_ = -1;
  long seite_ = -1;
  bool lauf_da_ = false;                          // the run is read
  long lr0_ = 0, lrn_ = 0, lri_ = 0;              // first page, length, position in the run
  std::vector<long> lsei_[kNTypen];               // pages per type while the run is read
  std::vector<std::pair<int, long>> lauf_;        // (type, page) of the records in block order
  size_t lk_ = 0;                                 // next record
  long inst_ = 0;
  int nluecke_ = 0;
  size_t wi_ = 0;
  std::vector<Slot> slots_;
  std::vector<unsigned char> felder_;   // labels of the value tokens of the running record, in token order
  zahl::Typmodell trocken_;   // only for the dry run (never trained)
  zahl::XfiTafel xfi_;                 // transitions of the X-field prediction
  zahl::Typmodell spiegel_[kNTypen];   // mirror of the coder model per type
  std::vector<std::string> toks_;      // value tokens of the running record so far
  std::string tokcur_;
  unsigned char okb_[128];
  int okb_n_ = 0;
  bool tf_ = false;          // the transformer sees this injection (computed text)
  int tcode_ = 0;
  std::vector<std::array<long long, 4>> trec_;   // (page, state, type word, census type) while the line is read
  unsigned long long anker_ = 0;
  long long seite_akt_ = -1;
};
inline Aut aut;

}  // namespace census_block
#endif
