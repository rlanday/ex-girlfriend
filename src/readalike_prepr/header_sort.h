
#ifndef HEADER_SORT_H_
#define HEADER_SORT_H_

#include "../ts_progress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Metadata block split: each record is cut into part A (page id delta,
// revision id, optional restrictions; kept in stream order) and part B (timestamp,
// contributor; sorted by revision id).  The decoder pairs them again by the
// rank of the revision id, so the permutation costs no side information.
typedef struct {
  long long rev;        // revision id (absolute)
  int        idx;       // stream index (second key, makes the rank unique)
  size_t     a0, a1;    // part A of this record  [a0, a1)
  size_t     b0, b1;    // part B of this record  [b0, b1)
} hsort_rec;

static int hsort_cmp(const void *x, const void *y) {
  const hsort_rec *a = (const hsort_rec*)x, *b = (const hsort_rec*)y;
  if (a->rev < b->rev) return -1;
  if (a->rev > b->rev) return  1;
  return a->idx < b->idx ? -1 : (a->idx > b->idx ? 1 : 0);
}

static int hsort_is_head(const char *p, int len) {
  int i = 1;
  if (len < 3 || p[0] != '>') return 0;
  if (p[i] == '-') ++i;
  if (i >= len - 1) return 0;
  for (; i < len - 1; ++i) if (p[i] < '0' || p[i] > '9') return 0;
  return p[len - 1] == '\n';
}

static int hsort_is_contrib(const char *p, int len) {
  return (len >= 12 && memcmp(p, "contributor>", 12) == 0);
}

static const char *hsort_val(const char *p, int len, int taglen, int *vl) {
  *vl = len - taglen - 1;
  return p + taglen;
}
static void hsort_put(char *out, size_t *o, const char *p, int len) {
  memcpy(out + *o, p, (size_t)len); *o += (size_t)len;
}

// Guard: stop if an identifier lies outside the range the constants assume.
static void idnum_guard(long long v, long long lo, long long hi,
                              const char *what) {
  if (v < lo || v > hi) {
    fprintf(stderr, "guard - %s = %lld lies outside the "
                    "corpus invariant [%lld, %lld]; the stream does not match "
                    "the compiled-in constants\n", what, v, lo, hi);
    exit(1);
  }
}

static void idnum_pad(char *out, size_t *o, long long v, int dig) {
  char b[24];
  for (int i = dig - 1; i >= 0; --i) { b[i] = (char)('0' + (int)(v % 10)); v /= 10; }
  memcpy(out + *o, b, (size_t)dig); *o += (size_t)dig;
}

static void idnum_dec(char *out, size_t *o, long long v) {
  char b[24]; int n = 0;
  if (v < 0) { out[(*o)++] = '-'; v = -v; }
  if (v == 0) b[n++] = '0';
  while (v) { b[n++] = (char)('0' + (int)(v % 10)); v /= 10; }
  for (int i = n - 1; i >= 0; --i) out[(*o)++] = b[i];
}
static long long idnum_num(const char *p, int len) {
  long long v = 0; int i = 0, neg = 0;
  if (len > 0 && p[0] == '-') { neg = 1; i = 1; }
  for (; i < len; ++i) v = v * 10 + (p[i] - '0');
  return neg ? -v : v;
}

static size_t header_sort_apply(const char *in, size_t n, char *out, int enc) {
  const char *end = in + n;
  size_t cap = 4096, cnt = 0;
  hsort_rec *r = (hsort_rec*)malloc(cap * sizeof(hsort_rec));
  if (!r) { fprintf(stderr, "header sort: out of memory\n"); exit(1); }
  size_t o = 0;

  if (enc) {
    // ---------------- encoder: split the records --------------------------
    size_t i = 0;
    while (i < n) {
      int len = phda9_line_len(in + i, end);
      if (!hsort_is_head(in + i, len)) {
        fprintf(stderr, "header sort: no record start at %lu\n", (unsigned long)i);
        exit(1);
      }
      if (cnt == cap) {
        cap *= 2;
        r = (hsort_rec*)realloc(r, cap * sizeof(hsort_rec));
        if (!r) { fprintf(stderr, "header sort: out of memory\n"); exit(1); }
      }
      hsort_rec *c = &r[cnt];
      c->idx = (int)cnt; c->a0 = i; c->rev = 0;
      size_t p = i + (size_t)len;
      size_t bstart = 0;
      while (p < n) {
        int l2 = phda9_line_len(in + p, end);
        if (hsort_is_head(in + p, l2)) break;
        if (!bstart) {
          if (hsort_is_contrib(in + p, l2) ||
              (l2 > 10 && memcmp(in + p, "timestamp>", 10) == 0)) bstart = p;
          else if (l2 > 3 && memcmp(in + p, "id>", 3) == 0)
            c->rev = strtoll(in + p + 3, 0, 10);
        }
        p += (size_t)l2;
      }
      if (!bstart) { fprintf(stderr, "header sort: record without part B\n"); exit(1); }
      c->a1 = bstart; c->b0 = bstart; c->b1 = p;
      ++cnt; i = p;
    }
    // part A in stream order
    for (size_t k = 0; k < cnt; ++k) {
      size_t q = r[k].a0; const char *rest = 0; int restl = 0, dl = 0, rl = 0;
      const char *dv = 0, *rv = 0;
      while (q < r[k].a1) {
        int l = phda9_line_len(in + q, end);
        if (in[q] == '>')                                      dv = hsort_val(in + q, l, 1, &dl);
        else if (l > 13 && memcmp(in + q, "restrictions>", 13) == 0)
                                                               rest = hsort_val(in + q, l, 13, &restl);
        else if (l > 3 && memcmp(in + q, "id>", 3) == 0)        rv = hsort_val(in + q, l, 3, &rl);
        q += (size_t)l;
      }
      if (!dv || !rv) { fprintf(stderr, "header sort: incomplete record\n"); exit(1); }
      hsort_put(out, &o, dv, dl);
      out[o++] = '\t';
      hsort_put(out, &o, rv, rl);
      if (rest) { out[o++] = '\t'; hsort_put(out, &o, rest, restl); }
      out[o++] = '\n';
    }
    // part B in revision id order
    hsort_rec *s = (hsort_rec*)malloc(cnt * sizeof(hsort_rec));
    if (!s) { fprintf(stderr, "header sort: out of memory\n"); exit(1); }
    memcpy(s, r, cnt * sizeof(hsort_rec));
    qsort(s, cnt, sizeof(hsort_rec), hsort_cmp);
    for (size_t k = 1; k < cnt; ++k)
      if (s[k].rev == s[k-1].rev) {
        fprintf(stderr, "header sort: duplicate revid %lld\n",
                s[k].rev);
        exit(1);
      }
    name_index_t ut;                              // name -> uid, encoder side
    name_index_init(&ut, (int)cnt, n);
    unsigned long long ue_weg = 0, ue_marke = 0;
    for (size_t k = 0; k < cnt; ++k) {
      size_t q = s[k].b0; const char *ts = 0, *nm = 0, *ui = 0;
      int tl = 0, nl2 = 0, ul = 0;
      while (q < s[k].b1) {
        int l = phda9_line_len(in + q, end);
        if (l > 10 && memcmp(in + q, "timestamp>", 10) == 0)   ts = hsort_val(in + q, l, 10, &tl);
        else if (l > 9 && memcmp(in + q, "username>", 9) == 0)  nm = hsort_val(in + q, l, 9, &nl2);
        else if (l > 3 && memcmp(in + q, "ip>", 3) == 0)        nm = hsort_val(in + q, l, 3, &nl2);
        else if (l > 3 && memcmp(in + q, "id>", 3) == 0)        ui = hsort_val(in + q, l, 3, &ul);
        q += (size_t)l;
      }
      if (!ts || !nm) { fprintf(stderr, "header sort: incomplete B record\n"); exit(1); }
      hsort_put(out, &o, ts, tl);
      out[o++] = '\t';
      hsort_put(out, &o, nm, nl2);
      {
        if (ui && (ul <= 0 || memchr(ui, '\t', (size_t)ul))) {
          fprintf(stderr, "user ids: empty uid or uid with tab\n");
          exit(1);
        }
        int e = name_index_find(&ut, nm, nl2);
        if (ui) {
          if (e < 0) name_index_add(&ut, nm, nl2, ui, ul);
          if (e >= 0 && ut.ulen[e] == ul && memcmp(ut.arena + ut.uoff[e], ui, (size_t)ul) == 0) {
            ++ue_weg;
          } else {
            out[o++] = '\t'; hsort_put(out, &o, ui, ul);
          }
        } else if (e >= 0) {
          if (ul != 0 || nl2 <= 0 || memchr(nm, '\t', (size_t)nl2)) {
            fprintf(stderr, "user ids: collision record not of the form timestamp<tab>address\n");
            exit(1);
          }
          ++ue_marke;
          out[o++] = '\t';
        } else {
        }
      }
      out[o++] = '\n';
    }
    name_index_free(&ut);
    free(s);
  } else {
    size_t lines = 0;
    for (size_t q = 0; q < n; ) { q += (size_t)phda9_line_len(in + q, end); ++lines; }
    if (lines & 1) { fprintf(stderr, "header sort: odd line count\n"); exit(1); }
    cnt = lines / 2;
    if (cnt > cap) {
      cap = cnt;
      r = (hsort_rec*)realloc(r, cap * sizeof(hsort_rec));
      if (!r) { fprintf(stderr, "header sort: out of memory\n"); exit(1); }
    }
    size_t q = 0;
    for (size_t k = 0; k < cnt; ++k) {                       // read part A
      int l = phda9_line_len(in + q, end);
      const char *t = (const char*)memchr(in + q, '\t', (size_t)l);
      if (!t) { fprintf(stderr, "header sort: A record without separator\n"); exit(1); }
      r[k].idx = (int)k; r[k].a0 = q; r[k].a1 = q + (size_t)l;
      r[k].rev = strtoll(t + 1, 0, 10);
      q += (size_t)l;
    }
    size_t *bo = (size_t*)malloc(cnt * sizeof(size_t));
    size_t *bl = (size_t*)malloc(cnt * sizeof(size_t));
    if (!bo || !bl) { fprintf(stderr, "header sort: out of memory\n"); exit(1); }
    for (size_t k = 0; k < cnt; ++k) {                       // read part B
      int l = phda9_line_len(in + q, end);
      bo[k] = q; bl[k] = (size_t)l; q += (size_t)l;
    }
    name_index_t ut;
    name_index_init(&ut, (int)cnt, n);
    int *ue_art = (int*)malloc(cnt * sizeof(int));
    int *ue_b = (int*)malloc(cnt * sizeof(int));  // page -> part-B record
    if (!ue_art || !ue_b) { fprintf(stderr, "user ids: out of memory\n"); exit(1); }
    for (size_t k = 0; k < cnt; ++k) {
      const char *bp = in + bo[k]; const char *le = bp + bl[k] - 1;   // the '\n'
      const char *u1 = (const char*)memchr(bp, '\t', bl[k]);
      if (!u1) { fprintf(stderr, "header sort: B record without separator\n"); exit(1); }
      const char *u2 = (const char*)memchr(u1 + 1, '\t', (size_t)(le - u1 - 1));
      const char *nm = u1 + 1; int nl2 = (int)((u2 ? u2 : le) - nm);
      int e = name_index_find(&ut, nm, nl2);
      if (u2 && u2 + 1 < le) {                    // the third field carries the uid
        if (e < 0) name_index_add(&ut, nm, nl2, u2 + 1, (int)(le - u2 - 1));
        ue_art[k] = -2;
      } else if (u2) {                            // empty third field: address
        ue_art[k] = -1;
      } else {
        ue_art[k] = (e >= 0) ? e : -1;            // hit: user name, else address
      }
    }
    hsort_rec *sr = (hsort_rec*)malloc(cnt * sizeof(hsort_rec));
    if (!sr) { fprintf(stderr, "header sort: out of memory\n"); exit(1); }
    memcpy(sr, r, cnt * sizeof(hsort_rec));
    qsort(sr, cnt, sizeof(hsort_rec), hsort_cmp);
    for (size_t k = 0; k < cnt; ++k) { r[sr[k].idx].b0 = bo[k]; r[sr[k].idx].b1 = bo[k] + bl[k]; }
    for (size_t k = 0; k < cnt; ++k) ue_b[sr[k].idx] = (int)k;
    free(sr); free(bo); free(bl);
    for (size_t k = 0; k < cnt; ++k) {
      const char *ap = in + r[k].a0; int al = (int)(r[k].a1 - r[k].a0);
      const char *t1 = (const char*)memchr(ap, '\t', (size_t)al);
      const char *t2 = (const char*)memchr(t1 + 1, '\t', (size_t)(ap + al - t1 - 1));
      out[o++] = '>';
      memcpy(out + o, ap, (size_t)(t1 - ap)); o += (size_t)(t1 - ap);
      out[o++] = '\n';
      if (t2) {
        memcpy(out + o, "restrictions>", 13); o += 13;
        memcpy(out + o, t2 + 1, (size_t)(ap + al - 1 - t2 - 1)); o += (size_t)(ap + al - 1 - t2 - 1);
        out[o++] = '\n';
      }
      memcpy(out + o, "revision>\n", 10); o += 10;
      memcpy(out + o, "id>", 3); o += 3;
      { const char *e2 = t2 ? t2 : ap + al - 1;
        memcpy(out + o, t1 + 1, (size_t)(e2 - t1 - 1)); o += (size_t)(e2 - t1 - 1);
      }
      out[o++] = '\n';
      const char *bp = in + r[k].b0; int blen = (int)(r[k].b1 - r[k].b0);
      const char *u1 = (const char*)memchr(bp, '\t', (size_t)blen);
      const char *u2 = u1 ? (const char*)memchr(u1 + 1, '\t', (size_t)(bp + blen - u1 - 1)) : 0;
      if (!u1) { fprintf(stderr, "header sort: B record without separator\n"); exit(1); }
      const char *nend = u2 ? u2 : bp + blen - 1;
      char tsbuf[64]; int tsl;
      tsl = (int)(u1 - bp);
      memcpy(tsbuf, bp, (size_t)tsl);
      memcpy(out + o, "contributor>\n", 13); o += 13;
      const int art = ue_art[ue_b[k]];
      if (art != -1) { memcpy(out + o, "username>", 9); o += 9; }
      else           { memcpy(out + o, "ip>", 3); o += 3; }
      memcpy(out + o, u1 + 1, (size_t)(nend - u1 - 1)); o += (size_t)(nend - u1 - 1);
      out[o++] = '\n';
      if (art == -2) {
        memcpy(out + o, "id>", 3); o += 3;
        memcpy(out + o, u2 + 1, (size_t)(bp + blen - 1 - u2 - 1)); o += (size_t)(bp + blen - 1 - u2 - 1);
        out[o++] = '\n';
      } else if (art >= 0) {
        memcpy(out + o, "id>", 3); o += 3;
        memcpy(out + o, ut.arena + ut.uoff[art], (size_t)ut.ulen[art]); o += (size_t)ut.ulen[art];
        out[o++] = '\n';
      }
      memcpy(out + o, "/contributor>\n", 14); o += 14;
      memcpy(out + o, "timestamp>", 10); o += 10;
      memcpy(out + o, tsbuf, (size_t)tsl); o += (size_t)tsl; out[o++] = '\n';
    }
    name_index_free(&ut); free(ue_art); free(ue_b);
    free(r);
    return o;
  }
  free(r);
  return o;
}

#endif
