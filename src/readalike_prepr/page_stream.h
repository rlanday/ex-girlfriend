// The page loop shared by the census transforms and the edit-comment transforms.
// Reads a stream in 1-MiB chunks, copies everything outside <page>...</page>
// unchanged and hands every complete page to a page function.  The page
// function returns 1 (page changed, result in *ob / *outn), 0 (unchanged) or
// -1 (error: the page is copied unchanged and the loop returns 0).
#ifndef PAGE_STREAM_H_
#define PAGE_STREAM_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../meld.h"

typedef int (*page_fn)(const char* p, long n, long* outn, const char** ob);

// First occurrence of nd[0..m) in h[from..n), or -1.
static long page_find(const char* h, long n, long from, const char* nd, long m) {
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

static int page_stream_each(FILE* in, FILE* out, page_fn page) {
  const long CH = 1 << 20;
  long cap = 4 * CH, n = 0, pos;
  char* buf = (char*)malloc((size_t)cap);
  if (!buf) return 0;
  int inside = 0, eof = 0, err = 0;
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
        long s = page_find(buf, n, pos, "<page>", 6);
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
        long e = page_find(buf, n, pos, "</page>", 7);
        if (e < 0) break;
        e += 7;
        long pl = e - pos, ol = 0;
        const char* ob = 0;
        int changed = page(buf + pos, pl, &ol, &ob);
        if (changed < 0) { err = 1; changed = 0; }
        if (changed) fwrite(ob, 1, (size_t)ol, out);
        else         fwrite(buf + pos, 1, (size_t)pl, out);
        pos = e;
        inside = 0;
      }
    }
    if (pos > 0) { memmove(buf, buf + pos, (size_t)(n - pos)); n -= pos; }
  }
  if (n > 0) fwrite(buf, 1, (size_t)n, out);
  free(buf);
  return err ? 0 : 1;
}

// The page loop from file in to file out; ren: out (a temporary file) then
// replaces in.  0 on failure (the caller names its stage).
static int page_file(const char* in, const char* out, page_fn page, int ren) {
  FILE* a = fopen(in, "rb");
  FILE* b = a ? fopen(out, "wb") : 0;
  if (!b) {
    if (a) fclose(a);
    msg_note(116);
    return 0;
  }
  int ok = page_stream_each(a, b, page);
  fclose(a);
  if (fclose(b) != 0) ok = 0;
  if (ok && ren) {
    remove(in);
    ok = rename(out, in) == 0;
  }
  return ok;
}

#endif
