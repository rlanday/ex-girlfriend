// Comment numbers.  A recurring edit comment becomes the NUMBER of its first
// occurrence in the comment list of its author (per author the list of his
// distinct comments in the order of first appearance); a comment already on the
// list is written as its list number ("2" = the author's second distinct
// comment).  On enwik9 about 46 % of the non-empty comments repeat an earlier
// comment of the same author (count over all pages), and a short number is
// cheaper to code than the repeated text.
//
// Encoder, per page of .main_reordered (XML in stream order, behind reorder()
// and behind the section index of the comments):
//   author = the bytes between <contributor> and </contributor> (user name and
//            id, or the ip), one key per contributor record.
//   rest   = the comment body that is compared: what the section index left
//            (the section reference "/* SEC */" stands as the heading index
//            behind the text).
//   empty rest              -> untouched (the empty-comment escape of the page form)
//   rest on the author list -> its list number in decimal digits (1-based)
//   rest not on the list    -> the rest as text, appended to the list; a rest of
//                              digits only (9 on enwik9) or one starting with the
//                              escape byte gets the escape byte 0x04 in front.
// 0x04 is in the fixed byte set, passes phda9 and the WRT unchanged and occurs
// nowhere else at this place (elsewhere only the IP hint in the metadata block
// and the control bytes of the census transform inside article text).
//
// Decoder: .main_decomp_restored (behind phda9_resto() and the census block,
// before sort() and before the inverse section index), the same pages in the
// same order; the author is known there (phda9 has rebuilt <contributor> from
// the metadata block).  The lists are built from the decoded pages exactly as
// the encoder built them.  The number of lines of the stream may change (a
// comment with a line feed becomes one line); split4Decomp derives its cut
// lines from the total line count, intro and coda stay untouched.
//
// Zone: the redirect front block (the first pages, a constant known to both
// sides) keeps every comment as text; behind it the number form applies, and
// the lists start empty at the first page behind the block.  The file stage
// counts the pages of the whole file.  With the numbers the author injection
// of the comment line carries the day number of the revision in front of the
// name.
#ifndef KOMM_TRANSFORM_H_
#define KOMM_TRANSFORM_H_
#include "../meld.h"
#include "../enc_only.h"
#include "page_stream.h"
#include "cmtidx_transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../redirect_table.h"

namespace comment_form {

static const char kEsc = 0x04;

static void Oom() { msg_stop(480); exit(1); }

// Byte arena of author keys and rests.
static char* ar = 0;
static long arn = 0, arc = 0;
static long Ablegen(const char* p, long n) {
  if (arn + n > arc) {
    long c = arc ? arc : (1 << 22);
    while (arn + n > c) c *= 2;
    char* q = (char*)realloc(ar, (size_t)c);
    if (!q) Oom();
    ar = q; arc = c;
  }
  memcpy(ar + arn, p, (size_t)n);
  arn += n;
  return arn - n;
}

static unsigned long long Hash(const char* p, long n, unsigned long long h) {
  for (long i = 0; i < n; ++i) h = (h ^ (unsigned char)p[i]) * 0x100000001b3ULL;
  return h;
}

// Author: key in the arena, list of rest ids (order 1: first occurrence,
// order 2: move-to-front), rest used last (statistics).
struct Autor { long off; int len, n, cap, letzt; int* liste; };
// Rest: text in the arena, author, 1-based number in the first-occurrence list.
struct Rest { long off; int len, autor, nr; };
static Autor* au = 0;
static int aun = 0, auc = 0;
static Rest* re = 0;
static int ren = 0, rec = 0;

// Open addressing, value = index + 1 (0 = empty), capacity a power of two.
struct Tab { unsigned long long* h; int* v; long cap, n; };
static Tab ta = {0, 0, 0, 0}, tr = {0, 0, 0, 0};
static void TabNeu(Tab* t, long cap) {
  t->cap = cap; t->n = 0;
  t->h = (unsigned long long*)calloc((size_t)cap, sizeof(unsigned long long));
  t->v = (int*)calloc((size_t)cap, sizeof(int));
  if (!t->h || !t->v) Oom();
}
static void TabRein(Tab* t, unsigned long long h, int v) {
  long i = (long)(h & (unsigned long long)(t->cap - 1));
  while (t->v[i]) i = (i + 1) & (t->cap - 1);
  t->h[i] = h; t->v[i] = v; ++t->n;
}
static void TabPlatz(Tab* t) {
  if (!t->cap) { TabNeu(t, 1 << 16); return; }
  if (2 * (t->n + 1) <= t->cap) return;
  Tab u;
  TabNeu(&u, t->cap * 2);
  for (long i = 0; i < t->cap; ++i) if (t->v[i]) TabRein(&u, t->h[i], t->v[i]);
  free(t->h); free(t->v);
  *t = u;
}

static int AutorId(const char* p, long n) {
  TabPlatz(&ta);
  const unsigned long long h = Hash(p, n, 0xcbf29ce484222325ULL);
  long i = (long)(h & (unsigned long long)(ta.cap - 1));
  for (; ta.v[i]; i = (i + 1) & (ta.cap - 1)) {
    if (ta.h[i] != h) continue;
    const Autor& a = au[ta.v[i] - 1];
    if (a.len == n && memcmp(ar + a.off, p, (size_t)n) == 0) return ta.v[i] - 1;
  }
  if (aun == auc) {
    auc = auc ? 2 * auc : 1 << 16;
    au = (Autor*)realloc(au, (size_t)auc * sizeof(Autor));
    if (!au) Oom();
  }
  Autor& a = au[aun];
  a.off = Ablegen(p, n); a.len = (int)n; a.n = 0; a.cap = 0; a.letzt = -1; a.liste = 0;
  TabRein(&ta, h, aun + 1);
  return aun++;
}

static unsigned long long RestHash(int a, const char* p, long n) {
  return Hash(p, n, 0xcbf29ce484222325ULL ^ ((unsigned long long)(unsigned)a * 0x9e3779b97f4a7c15ULL));
}

// Rest id of (author, text), -1 = not on the list.
static int RestSuche(int a, const char* p, long n) {
  if (!tr.cap) return -1;
  const unsigned long long h = RestHash(a, p, n);
  for (long i = (long)(h & (unsigned long long)(tr.cap - 1)); tr.v[i]; i = (i + 1) & (tr.cap - 1)) {
    if (tr.h[i] != h) continue;
    const Rest& r = re[tr.v[i] - 1];
    if (r.autor == a && r.len == n && memcmp(ar + r.off, p, (size_t)n) == 0) return tr.v[i] - 1;
  }
  return -1;
}

// A new rest of author a; order 1 appends it, order 2 puts it in front.
static int RestNeu(int a, const char* p, long n) {
  TabPlatz(&tr);
  if (ren == rec) {
    rec = rec ? 2 * rec : 1 << 16;
    re = (Rest*)realloc(re, (size_t)rec * sizeof(Rest));
    if (!re) Oom();
  }
  Autor& A = au[a];
  if (A.n == A.cap) {
    A.cap = A.cap ? 2 * A.cap : 4;
    A.liste = (int*)realloc(A.liste, (size_t)A.cap * sizeof(int));
    if (!A.liste) Oom();
  }
  Rest& r = re[ren];
  r.off = Ablegen(p, n); r.len = (int)n; r.autor = a; r.nr = A.n + 1;
  A.liste[A.n] = ren;
  ++A.n;
  A.letzt = ren;
  TabRein(&tr, RestHash(a, p, n), ren + 1);
  return ren++;
}

// Number of rest r on the list of author a (encoder); order 2 moves it to the front.
static long Nummer(int a, int r) {
  Autor& A = au[a];
  A.letzt = r;
  return re[r].nr;
}

// Rest at number nr of author a (decoder), -1 = no such number.
static int Bei(int a, long nr) {
  Autor& A = au[a];
  if (nr < 1 || nr > A.n) return -1;
  const int r = A.liste[nr - 1];
  A.letzt = r;
  return r;
}

// Contributor key of a page: span between <contributor> and </contributor>
// (empty if the page has none).
static void Kontributor(const char* p, long n, long* ks, long* kn) {
  *ks = 0; *kn = 0;
  const long c = page_find(p, n, 0, "<contributor>", 13);
  if (c < 0) return;
  const long e = page_find(p, n, c + 13, "</contributor>", 14);
  if (e < 0) return;
  *ks = c + 13; *kn = e - c - 13;
}

static char* ob = 0;      // output page
static long obc = 0;
static long seiten = 0, kommentare = 0, leer = 0, hits = 0, neu = 0, escapes = 0,
            gleich_letzt = 0, nur_abschnitt = 0, mit_abschnitt = 0;
static long stellen[6] = {0, 0, 0, 0, 0, 0}, nr_eins = 0;
// ZONE: comments inside the redirect block (they stay text): repeats, new rests,
// and (ZONE=2) comments kept out of the lists.
static long blk_wdh = 0, blk_neu = 0, blk_aus = 0;

static bool Ziffern(const char* p, long n) {
  if (n <= 0) return false;
  for (long i = 0; i < n; ++i) if (p[i] < '0' || p[i] > '9') return false;
  return true;
}

static inline bool ImBlock(long k) { return k >= 0 && k < (long)71148; }

// Encoder, one page.  1 = changed (result in ob), 0 = unchanged, -1 = error.
static ENC_ONLY int KodiereSeite(const char* p, long n, long* outn) {
  const long k = seiten++;
  (void)k;
  int geaendert = 0;
  long ls, le, bs, be, ks, kn;
  if (!cmt_find_comment(p, n, &ls, &le, &bs, &be)) goto unveraendert;
  ++kommentare;
  if (be == bs) {
    ++leer;
    goto unveraendert;
  }
  {
    long pl = 0;
    const char* r = p + bs + pl;
    const long rn = be - bs - pl;
    Kontributor(p, n, &ks, &kn);
    const int a = AutorId(p + ks, kn);
    if (ImBlock(k)) {
      const char art = 'b';
      ++blk_aus;
      (void)art;
      goto unveraendert;
    }
    const int id = RestSuche(a, r, rn);
    char num[24];
    long nn = 0;
    const char* pay = r;
    long payn = rn;
    int esc = 0;
    if (id >= 0) {
      if (au[a].letzt == id) ++gleich_letzt;
      const long nr = Nummer(a, id);
      nn = snprintf(num, sizeof num, "%ld", nr);
      pay = num; payn = nn;
      ++hits;
      if (nr == 1) ++nr_eins;
      ++stellen[nn < 5 ? nn : 5];
    } else {
      esc = Ziffern(r, rn) || r[0] == kEsc;
      RestNeu(a, r, rn);
      ++neu;
      if (esc) ++escapes;
    }
    if (!cmt_grow(&ob, &obc, n + 32)) Oom();
    long o = 0;
    memcpy(ob, p, (size_t)(bs + pl)); o = bs + pl;
    if (esc) ob[o++] = kEsc;
    memcpy(ob + o, pay, (size_t)payn); o += payn;
    memcpy(ob + o, p + be, (size_t)(n - be)); o += n - be;
    *outn = o;
    return 1;
  }
unveraendert:
  if (!geaendert) return 0;
  if (!cmt_grow(&ob, &obc, n + 1)) Oom();
  memcpy(ob, p, (size_t)n);
  *outn = n;
  return 1;
}

// Decoder, one page: mirror image of KodiereSeite.
static int DekodiereSeite(const char* p, long n, long* outn) {
  const long k = seiten++;
  long ls, le, bs, be, ks, kn;
  if (!cmt_find_comment(p, n, &ls, &le, &bs, &be)) return 0;
  ++kommentare;
  if (be == bs) { ++leer; return 0; }
  long pl = 0;
  const char* d = p + bs + pl;
  const long dn = be - bs - pl;
  Kontributor(p, n, &ks, &kn);
  const int a = AutorId(p + ks, kn);
  if (ImBlock(k)) {
    ++blk_aus;
    return 0;
  }
  const char* r;
  long rn;
  bool text = false;
  if (Ziffern(d, dn)) {
    if (dn > 9) return -1;
    long nr = 0;
    for (long i = 0; i < dn; ++i) nr = nr * 10 + (d[i] - '0');
    const int id = Bei(a, nr);
    if (id < 0) return -1;
    r = ar + re[id].off; rn = re[id].len;
    ++hits;
  } else {
    r = d; rn = dn;
    if (d[0] == kEsc) { ++r; --rn; ++escapes; }
    // a text rest is always a new one; finding it on the list means the stream
    // was not written by this transform
    if (rn <= 0 || RestSuche(a, r, rn) >= 0) return -1;
    ++neu;
    text = true;
  }
  if (!cmt_grow(&ob, &obc, n + rn + 1)) Oom();
  long o = 0;
  memcpy(ob, p, (size_t)(bs + pl)); o = bs + pl;
  memcpy(ob + o, r, (size_t)rn); o += rn;
  memcpy(ob + o, p + be, (size_t)(n - be)); o += n - be;
  if (text) RestNeu(a, r, rn);   // after the copy: r points into the page, RestNeu may move the arena
  *outn = o;
  return 1;
}

static ENC_ONLY int KodiereCb(const char* p, long n, long* outn, const char** obp) {
  const int c = KodiereSeite(p, n, outn);
  *obp = ob;
  return c;
}
static int DekodiereCb(const char* p, long n, long* outn, const char** obp) {
  const int c = DekodiereSeite(p, n, outn);
  *obp = ob;
  return c;
}

static void Bericht(const char* page_index) {
  (void)page_index;
}

template <int enc>
static DIR_INLINE int ProcessFile(const char* path) {
  if (!page_file(path, ".komm_tmp", enc ? KodiereCb : DekodiereCb, 1)) return 0;
  Bericht(enc ? "compress" : "decompress");
  return 1;
}

static ENC_ONLY int Kodiere(const char* path) { return ProcessFile<1>(path); }
static int Dekodiere(const char* path) { return ProcessFile<0>(path); }

}  // namespace comment_form

#endif
