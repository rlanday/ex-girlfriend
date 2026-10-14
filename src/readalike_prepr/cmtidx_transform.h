
#ifndef CMTIDX_TRANSFORM_H_
#define CMTIDX_TRANSFORM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CMT_MARK 0x02
#define CMT_ESC  0x03

static const long cmt_base_lines = 13147028L;

// Comment index transform.  An edit comment that starts with "/* SEC */" loses
// that prefix; a mark line after the page text carries the index of the
// page heading that equals SEC, or, if no heading matches, an escape byte and
// SEC itself.  The decoder puts the prefix back.  The regular expressions in
// the comments below describe the rules.

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
static char*  cmt_ob = 0;      // output buffer of one page
static long   cmt_obcap = 0;
static char*  cmt_tb = 0;      // scratch buffer (page with shortened comment)
static long   cmt_tbcap = 0;
static char*  cmt_sb = 0;      // copy of the section name (decoder)
static long   cmt_sbcap = 0;
static long   cmtidx_pages = 0;   // changed pages (statistics)
static long   cmtidx_escapes = 0; // of those, with escape

static int cmt_grow(char** p, long* cap, long need) {
  if (*cap >= need) return 1;
  long nc = *cap ? *cap : 1 << 16;
  while (nc < need) nc *= 2;
  char* q = (char*)realloc(*p, (size_t)nc);
  if (!q) return 0;
  *p = q; *cap = nc;
  return 1;
}

static long cmt_find(const char* h, long n, long from, const char* pat, long pl) {
  if (from < 0) from = 0;
  for (long i = from; i + pl <= n; ++i) {
    if (h[i] == pat[0] && memcmp(h + i, pat, (size_t)pl) == 0) return i;
  }
  return -1;
}

static int cmt_isws(char c) { return c == ' ' || c == '\t'; }

// ---------------------------------------------------------------------------
// (1) comment line:  \n( *<comment>.*?</comment>\n)     [re.S]
//     returns 1 = found.  ls..le  = span of group 1 (line incl. \n),
//     bs..be = span of the comment body between the tags.
// ---------------------------------------------------------------------------
static int cmt_find_comment(const char* p, long n,
                            long* ls, long* le, long* bs, long* be) {
  long at = 0;
  for (;;) {
    long c = cmt_find(p, n, at, "<comment>", 9);
    if (c < 0) return 0;
    at = c + 1;
    // ' *' back to a '\n' - re.search requires the line start
    long q = c;
    while (q > 0 && p[q-1] == ' ') --q;
    if (q == 0 || p[q-1] != '\n') continue;
    // '.*?</comment>' lazy: first '</comment>' followed by a '\n'
    long e = c + 9;
    for (;;) {
      long t = cmt_find(p, n, e, "</comment>", 10);
      if (t < 0) break;
      if (t + 10 < n && p[t+10] == '\n') {
        *ls = q; *le = t + 11; *bs = c + 9; *be = t;
        return 1;
      }
      e = t + 1;
    }
    // no matching end -> try the next start position
  }
}

// ---------------------------------------------------------------------------
// (2) canonical form:  ^/\* ([^\n ](?:[^\n]*[^\n ])?) \*/(?: (?=.)|$)
//     b/bn = comment body.  returns 1 = match;
//     *ss/*sn = span of SEC, *mend = end of the whole match in b.
//     The group is greedy -> the LAST admissible " */" wins.
// ---------------------------------------------------------------------------
static int cmt_seccanon(const char* b, long bn, long* ss, long* sn, long* mend) {
  if (bn < 7) return 0;
  if (b[0] != '/' || b[1] != '*' || b[2] != ' ') return 0;
  long lineend = bn;
  for (long i = 0; i < bn; ++i) if (b[i] == '\n') { lineend = i; break; }
  // SEC = b[3..k), " */" at k..k+2; k >= 4 (SEC at least 1 character),
  // SEC must not contain a '\n' -> k <= lineend.
  for (long k = (lineend < bn - 2 ? lineend : bn - 3); k >= 4; --k) {
    if (k + 3 > bn) continue;
    if (b[k] != ' ' || b[k+1] != '*' || b[k+2] != '/') continue;
    char first = b[3], last = b[k-1];
    if (first == ' ' || first == '\n') continue;
    if (last == ' ' || last == '\n') continue;
    long pos = k + 3;
    // alternative 1: ' ' consumed, then a character != '\n' must follow
    if (pos < bn && b[pos] == ' ' && pos + 1 < bn && b[pos+1] != '\n') {
      *ss = 3; *sn = k - 3; *mend = pos + 1;
      return 1;
    }
    // alternative 2: '$' - end of the body or right before a trailing '\n'
    if (pos == bn || (pos == bn - 1 && b[pos] == '\n')) {
      *ss = 3; *sn = k - 3; *mend = pos;
      return 1;
    }
  }
  return 0;
}

static long cmt_line_after(const char* p, long n, long from) {
  long i = from;
  while (i < n && p[i] != '\n') ++i;
  return i < n ? i + 1 : -1;
}

static int cmt_find_text(const char* p, long n, long* tbs, long* tbe, long* tend) {
  long at = 0;
  for (;;) {
    long t = cmt_find(p, n, at, "<text", 5);
    if (t < 0) return 0;
    at = t + 1;
    long j = t + 5;
    while (j < n && p[j] != '>') ++j;
    if (j >= n) continue;
    long e = cmt_find(p, n, j + 1, "</text>\n", 8);
    if (e < 0) continue;
    *tbs = j + 1; *tbe = e; *tend = e + 8;
    return 1;
  }
}

// ---------------------------------------------------------------------------
// (4) headings:  ^(={1,6})[ \t]*(.*?)[ \t]*\1[ \t]*$   per line [re.M]
//     Rebuilt with the same backtracking: n descending 6..1, leading
//     [ \t]* greedy (with give-back), (.*?) lazy, trailing [ \t]* greedy.
//     returns 1 = the line is a heading; *hs/*hn = span of the name.
// ---------------------------------------------------------------------------
static int cmt_head_of_line(const char* L, long n, long* hs, long* hn) {
  for (int k = 6; k >= 1; --k) {
    if (n < k) continue;
    int ok = 1;
    for (int i = 0; i < k; ++i) if (L[i] != '=') { ok = 0; break; }
    if (!ok) continue;
    long w1max = k;
    while (w1max < n && cmt_isws(L[w1max])) ++w1max;
    for (long w1 = w1max; w1 >= k; --w1) {
      for (long gend = w1; gend <= n; ++gend) {
        long w2max = gend;
        while (w2max < n && cmt_isws(L[w2max])) ++w2max;
        for (long w2 = w2max; w2 >= gend; --w2) {
          if (w2 + k > n) continue;
          int hit = 1;
          for (int i = 0; i < k; ++i) if (L[w2+i] != '=') { hit = 0; break; }
          if (!hit) continue;
          long q = w2 + k;
          while (q < n && cmt_isws(L[q])) ++q;
          if (q != n) continue;
          *hs = w1; *hn = gend - w1;
          return 1;
        }
      }
    }
  }
  return 0;
}

// Index of the first heading that is verbatim SEC; -1 = none.
static long cmt_head_index(const char* body, long bn, const char* sec, long sn) {
  long idx = 0, i = 0;
  while (i <= bn) {
    long j = i;
    while (j < bn && body[j] != '\n') ++j;
    long hs, hn;
    if (cmt_head_of_line(body + i, j - i, &hs, &hn)) {
      if (hn == sn && memcmp(body + i + hs, sec, (size_t)sn) == 0) return idx;
      ++idx;
    }
    if (j >= bn) break;
    i = j + 1;
  }
  return -1;
}

// SEC of the idx-th heading; returns 1 = found.
static int cmt_head_at(const char* body, long bn, long idx,
                       long* hs_out, long* hn_out) {
  long cnt = 0, i = 0;
  while (i <= bn) {
    long j = i;
    while (j < bn && body[j] != '\n') ++j;
    long hs, hn;
    if (cmt_head_of_line(body + i, j - i, &hs, &hn)) {
      if (cnt == idx) { *hs_out = i + hs; *hn_out = hn; return 1; }
      ++cnt;
    }
    if (j >= bn) break;
    i = j + 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Page encoder.  Returns 1 = changed (result in cmt_ob/*outn).
// ---------------------------------------------------------------------------
static int cmt_encode_page(const char* p, long n, long* outn) {
  long ls, le, bs, be;
  if (!cmt_find_comment(p, n, &ls, &le, &bs, &be)) return 0;
  long ss, sn, mend;
  if (!cmt_seccanon(p + bs, be - bs, &ss, &sn, &mend)) return 0;
  long tbs, tbe, tend;
  if (!cmt_find_text(p, n, &tbs, &tbe, &tend)) return 0;

  const char* sec = p + bs + ss;
  long idx = cmt_head_index(p + tbs, tbe - tbs, sec, sn);

  // phda9 compatibility rule for the escape case.  In an escape the section
  // name travels verbatim into the mark line.  phda9 treats XML entities in
  // that line asymmetrically: the encoder leaves them alone, the restore
  // step expands '&' to '&amp;' (hent/removeamp/restoreamp in
  // phda9_preprocess.h).  If the name contains one of the four affected
  // characters, the page is left untouched (13 of 31,099 pages; the index
  // case, digits only, is never affected).  Without this rule the roundtrip
  // over enwik9 differs in 64 bytes.
  if (idx < 0) {
    for (long i = 0; i < sn; ++i) {
      char c = sec[i];
      if (c == '&' || c == '<' || c == '>' || c == '"') return 0;
    }
  }

  char ins[64];
  long inl = 0;
  ins[inl++] = CMT_MARK;
  const char* escsec = 0; long escn = 0;
  if (idx >= 0) {
    inl += snprintf(ins + inl, sizeof ins - (size_t)inl, "%ld", idx);
    ins[inl++] = '\n';
  } else {
    ins[inl++] = CMT_ESC;
    escsec = sec; escn = sn;   // SEC goes into the mark unchanged
  }

  // intermediate: page with shortened comment body
  long tn = n - mend;
  if (!cmt_grow(&cmt_tb, &cmt_tbcap, tn + 1)) return 0;
  if (escsec) {                          // SEC lies in the cut-away piece
    if (!cmt_grow(&cmt_sb, &cmt_sbcap, escn + 1)) return 0;
    memcpy(cmt_sb, escsec, (size_t)escn);
    escsec = cmt_sb;
  }
  memcpy(cmt_tb, p, (size_t)bs);                          // up to behind <comment>
  memcpy(cmt_tb + bs, p + bs + mend, (size_t)(n - bs - mend));

  // find the text end again in the shortened stream - like the reference
  long tbs2, tbe2, tend2;
  if (!cmt_find_text(cmt_tb, tn, &tbs2, &tbe2, &tend2)) return 0;
  long insat = cmt_line_after(cmt_tb, tn, tend2);
  if (insat < 0) return 0;

  long total = tn + inl + escn + (escsec ? 1 : 0);
  if (!cmt_grow(&cmt_ob, &cmt_obcap, total + 1)) return 0;
  long o = 0;
  memcpy(cmt_ob + o, cmt_tb, (size_t)insat); o += insat;
  memcpy(cmt_ob + o, ins, (size_t)inl); o += inl;
  if (escsec) {
    memcpy(cmt_ob + o, escsec, (size_t)escn); o += escn;
    cmt_ob[o++] = '\n';
    ++cmtidx_escapes;
  }
  memcpy(cmt_ob + o, cmt_tb + insat, (size_t)(tn - insat)); o += tn - insat;
  *outn = o;
  ++cmtidx_pages;
  return 1;
}

// ---------------------------------------------------------------------------
// Page decoder.  Returns 1 = changed, -1 = error.
// ---------------------------------------------------------------------------
static int cmt_decode_page(const char* p, long n, long* outn) {
  long tbs, tbe, tend;
  if (!cmt_find_text(p, n, &tbs, &tbe, &tend)) return 0;
  long insat = cmt_line_after(p, n, tend);
  if (insat < 0) return 0;
  if (insat >= n || (unsigned char)p[insat] != CMT_MARK) return 0;
  long j = insat + 1;
  while (j < n && p[j] != '\n') ++j;
  if (j >= n) return -1;
  const char* tok = p + insat + 1;
  long tokn = j - (insat + 1);

  const char* sec; long sn;
  if (tokn > 0 && (unsigned char)tok[0] == CMT_ESC) {
    sec = tok + 1; sn = tokn - 1;
    ++cmtidx_escapes;
  } else {
    if (tokn <= 0) return -1;
    long idx = 0;
    for (long i = 0; i < tokn; ++i) {
      if (tok[i] < '0' || tok[i] > '9') return -1;
      idx = idx * 10 + (tok[i] - '0');
    }
    long hs, hn;
    if (!cmt_head_at(p + tbs, tbe - tbs, idx, &hs, &hn)) return -1;
    sec = p + tbs + hs; sn = hn;
  }
  // save SEC: in the escape case it lies in the line about to be dropped.
  if (!cmt_grow(&cmt_sb, &cmt_sbcap, sn + 1)) return -1;
  memcpy(cmt_sb, sec, (size_t)sn);

  // remove the mark line
  long cut = j + 1 - insat;
  long tn = n - cut;
  if (!cmt_grow(&cmt_tb, &cmt_tbcap, tn + 1)) return -1;
  memcpy(cmt_tb, p, (size_t)insat);
  memcpy(cmt_tb + insat, p + j + 1, (size_t)(n - j - 1));

  long ls, le, bs, be;
  if (!cmt_find_comment(cmt_tb, tn, &ls, &le, &bs, &be)) return -1;
  long bodyn = be - bs;

  long total = tn + 3 + sn + 3 + (bodyn ? 1 : 0);
  if (!cmt_grow(&cmt_ob, &cmt_obcap, total + 1)) return -1;
  long o = 0;
  memcpy(cmt_ob + o, cmt_tb, (size_t)bs); o += bs;
  memcpy(cmt_ob + o, "/* ", 3); o += 3;
  memcpy(cmt_ob + o, cmt_sb, (size_t)sn); o += sn;
  memcpy(cmt_ob + o, " */", 3); o += 3;
  if (bodyn) cmt_ob[o++] = ' ';
  memcpy(cmt_ob + o, cmt_tb + bs, (size_t)(tn - bs)); o += tn - bs;
  *outn = o;
  ++cmtidx_pages;
  return 1;
}

static int cmtidx_stream(FILE* in, FILE* out, int enc) {
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
        long s = cmt_find(buf, n, pos, "<page>", 6);
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
        long e = cmt_find(buf, n, pos, "</page>", 7);
        if (e < 0) break;
        e += 7;
        long pl = e - pos, ol = 0;
        int changed = enc ? cmt_encode_page(buf + pos, pl, &ol)
                          : cmt_decode_page(buf + pos, pl, &ol);
        if (changed < 0) { err = 1; changed = 0; }
        if (changed) fwrite(cmt_ob, 1, (size_t)ol, out);
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

static int cmtidx_file(const char* inpath, const char* outpath, int enc) {
  FILE* in = fopen(inpath, "rb");
  if (!in) { fprintf(stderr, "comment index: cannot read %s\n", inpath); return 0; }
  FILE* out = fopen(outpath, "wb");
  if (!out) { fclose(in); fprintf(stderr, "comment index: cannot write %s\n", outpath); return 0; }
  cmtidx_pages = 0; cmtidx_escapes = 0;
  int ok = cmtidx_stream(in, out, enc);
  fclose(in);
  if (fclose(out) != 0) ok = 0;
  return ok;
}

// Rewrite a file in place (via a temporary file).
static int cmtidx_inplace(const char* path, const char* tmp, int enc) {
  if (!cmtidx_file(path, tmp, enc)) return 0;
  remove(path);
  if (rename(tmp, path) != 0) {
    fprintf(stderr, "comment index: rename %s -> %s failed\n", tmp, path);
    return 0;
  }
  return 1;
}

// Count '\n' in a file (for the cut lines of split4Decomp).
static long cmtidx_count_lines(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  const long CH = 1 << 20;
  char* b = (char*)malloc((size_t)CH);
  if (!b) { fclose(f); return -1; }
  long cnt = 0;
  size_t got;
  while ((got = fread(b, 1, (size_t)CH, f)) > 0) {
    for (size_t i = 0; i < got; ++i) if (b[i] == '\n') ++cnt;
  }
  free(b);
  fclose(f);
  return cnt;
}

// Shift of the split4Decomp cut lines relative to the base stream.
static long cmtidx_line_shift(const char* path) {
  long tot = cmtidx_count_lines(path);
  if (tot < 0) return 0;
  return tot - cmt_base_lines;
}

#endif
