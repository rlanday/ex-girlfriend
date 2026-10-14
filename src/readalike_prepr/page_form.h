
#ifndef PAGE_FORM_H_
#define PAGE_FORM_H_

#include "../author_block.h"
#include "../redirect_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Page form: the encoder turns each XML page into title line, body, "minor"
// as a word, edit comment on the last line and one separator; the decoder
// (g1_expand) writes the constant XML back.  Names: seiten = pages,
// leerkom = empty comments, offen = unread output left.
static char  *g1_pb = 0;         // page buffer (XML form)
static size_t g1_pn = 0, g1_pc = 0;
static char  *g1_ob = 0;         // output buffer (page form)
static size_t g1_oc = 0;
static long long g1_seiten = 0, g1_self = 0, g1_leerkom = 0, g1_nlkom = 0;

static long long g1_rc_enc[7] = {0}, g1_rc_dec[7] = {0};
static long long g1_rc_dseiten = 0;

static void g1_oom(void) { fprintf(stderr, "page form: out of memory\n"); exit(1); }

static void g1_reserve(char **b, size_t *c, size_t need) {
  if (need <= *c) return;
  *c = need * 2 + 65536;
  *b = (char*)realloc(*b, *c);
  if (!*b) g1_oom();
}

static void g1_add(const char *t) {
  size_t l = strlen(t);
  g1_reserve(&g1_pb, &g1_pc, g1_pn + l + 1);
  memcpy(g1_pb + g1_pn, t, l);
  g1_pn += l;
  g1_pb[g1_pn] = 0;
}

// Substring search without memmem (a GNU extension), so the build does not
// depend on _GNU_SOURCE.  The buffer is one page, the patterns are short.
static long g1_find(const char *hay, size_t n, size_t from,
                    const char *pat, size_t pl) {
  if (pl == 0 || from >= n || pl > n - from) return -1;
  const char c0 = pat[0];
  for (size_t i = from; i + pl <= n; ++i) {
    if (hay[i] != c0) continue;
    if (memcmp(hay + i, pat, pl) == 0) return (long)i;
  }
  return -1;
}

static long g1_rfind_str(const char *hay, size_t n, const char *pat, size_t pl) {
  if (pl == 0 || n < pl) return -1;
  for (size_t i = n - pl + 1; i > 0; --i)
    if (hay[i - 1] == pat[0] && memcmp(hay + i - 1, pat, pl) == 0) return (long)(i - 1);
  return -1;
}

static void g1_put(size_t *o, const char *p, size_t l) {
  memcpy(g1_ob + *o, p, l); *o += l;
}
static void g1_puts(size_t *o, const char *s) { g1_put(o, s, strlen(s)); }

static void g1_page_out(FILE *out) {
  const size_t n = g1_pn;
  const char *b = g1_pb;
  ++g1_seiten;

  static const char kPageHead[] = "  <page>\n    <title>";
  const size_t kPageHeadLen = sizeof(kPageHead) - 1;
  if (n < kPageHeadLen || memcmp(b, kPageHead, kPageHeadLen) != 0) {
    fprintf(stderr, "page form: page %lld does not start with `  <page>` + title\n",
            g1_seiten);
    exit(1);
  }
  const long t0 = 9;                       // behind `  <page>\n`
  const long tv = g1_find(b, n, (size_t)t0 + 11, "</title>\n", 9);
  if (tv < 0) {
    fprintf(stderr, "page form: no title end at page %lld\n", g1_seiten);
    exit(1);
  }
  const char *titel = b + t0 + 11;
  const size_t titell = (size_t)(tv - t0 - 11);
  const size_t head = (size_t)tv + 9;

  const char *body = "";
  size_t bodyl = 0;
  const long ts = g1_find(b, n, head, "      <text xml:space=\"preserve\">", strlen("      <text xml:space=\"preserve\">"));
  if (ts < 0) {
    if (g1_find(b, n, head, "      <text xml:space=\"preserve\" />\n", strlen("      <text xml:space=\"preserve\" />\n")) < 0) {
      fprintf(stderr, "page form: page %lld has no text element\n", g1_seiten);
      exit(1);
    }
    ++g1_self;
  } else {
    const size_t bs = (size_t)ts + strlen("      <text xml:space=\"preserve\">");
    const long te = g1_find(b, n, bs, "</text>\n", 8);
    if (te < 0) {
      fprintf(stderr, "page form: page %lld without </text>\n", g1_seiten);
      exit(1);
    }
    body = b + bs; bodyl = (size_t)te - bs;
    if (bodyl == 0) {
      fprintf(stderr, "page form: empty <text ...></text> at page %lld\n", g1_seiten);
      exit(1);
    }
  }

  // comment payload
  const char *kom = 0; size_t koml = 0;
  const size_t head_end = ts < 0 ? n : (size_t)ts;
  const long c0 = g1_find(b, head_end, head, "      <comment>", 15);
  long c1 = -1;
  if (c0 >= 0) {
    c1 = g1_find(b, n, (size_t)c0 + 15, "</comment>\n", 11);
    if (c1 < 0) {
      fprintf(stderr, "page form: page %lld without </comment>\n", g1_seiten);
      exit(1);
    }
    kom = b + c0 + 15; koml = (size_t)c1 - c0 - 15;
    if (koml == 0) ++g1_leerkom;
  }

  const int minor = g1_find(b, head_end, head, "      <minor />\n", 16) >= 0;

  const long rv = g1_rfind_str(b, n, "    </revision>\n", 16);
  if (rv < 0 || n < 10) {
    fprintf(stderr, "page form: page %lld without </revision>\n", g1_seiten);
    exit(1);
  }
  const char *mark = b + rv + 16;
  const size_t markl = n - 10 - (size_t)(rv + 16);

  unsigned char rc_code = 0;
  if (redirect_block(g1_seiten - 1)) {
    size_t rc_skip = 0;
    rc_code = redirect_encode(body, bodyl, &rc_skip, g1_rc_enc);
    body += rc_skip;
    bodyl -= rc_skip;
  }
  g1_reserve(&g1_ob, &g1_oc, n + 2 * koml + 4096);
  size_t o = 0;
  g1_put(&o, titel, titell); g1_ob[o++] = '\n';
  if (rc_code) g1_ob[o++] = (char)rc_code;
  g1_put(&o, body, bodyl);   g1_ob[o++] = '\n';
  const size_t komzeile = o;
  if (kom) {
    g1_ob[o++] = (char)(unsigned char)3;
    if (koml == 0) {
      g1_ob[o++] = 0x02;
    } else {
      int nl = 0;
      for (size_t i = 0; i < koml; ++i) {
        if (kom[i] == '\n') { g1_ob[o++] = 0x02; g1_ob[o++] = 0x02; nl = 1; }
        else if ((unsigned char)kom[i] == 0x02) {
          fprintf(stderr, "page form: 0x02 in the comment at page %lld\n", g1_seiten);
          exit(1);
        }
        else                  g1_ob[o++] = kom[i];
      }
      if (nl) ++g1_nlkom;
    }
  }
  g1_ob[o++] = '\n';
  g1_put(&o, mark, markl);
  if (minor) g1_puts(&o, "minor");
  g1_puts(&o, "  </page>\n");
  if (o > 10 && g1_find(g1_ob, o - 10, 0, "  </page>\n", 10) >= 0) {
    fprintf(stderr, "page form: anchor inside page %lld\n", g1_seiten);
    exit(1);
  }
  {
    const unsigned char m = (unsigned char)3;
    for (size_t z = 0; z < o; ) {
      if ((unsigned char)g1_ob[z] == m && !(kom && z == komzeile)) {
        fprintf(stderr, "page form: marker byte 0x%02x outside the comment line (offset %zu), page %lld\n",
                (unsigned)m, z, g1_seiten);
        exit(1);
      }
      const char *nl = (const char*)memchr(g1_ob + z, '\n', o - z);
      if (!nl) break;
      z = (size_t)(nl - g1_ob) + 1;
    }
  }
  if (kom && koml == 0 && markl == 0) {
    fprintf(stderr, "page form: empty comment without index line at page %lld\n", g1_seiten);
    exit(1);
  }
  blockwrite(out, (U8*)g1_ob, (U64)o);
  g1_pn = 0;
}

static char  *g1_db = 0;  static size_t g1_dn = 0, g1_dc = 0;  // page as coded
static char  *g1_eb = 0;  static size_t g1_en = 0, g1_ec = 0;  // expanded XML form
static size_t g1_epos = 0;                                     // read position

static long g1_rfind(const char *b, size_t n, char c) {
  for (size_t i = n; i > 0; --i) if (b[i - 1] == c) return (long)(i - 1);
  return -1;
}

static void g1_expand(void) {
  const char *b = g1_db;
  if (g1_dn < 10 || memcmp(b + g1_dn - 10, "  </page>\n", 10) != 0) {
    fprintf(stderr, "page form: page without anchor\n"); exit(1);
  }
  size_t e = g1_dn - 10;
  const char *titel = 0, *body = 0, *kom = 0, *flag = 0, *mark = 0;
  size_t titell = 0, bodyl = 0, koml = 0, flagl = 0, markl = 0;

  {
    long nl = -1;
    for (size_t i = 0; i < e; ++i) if (b[i] == '\n') { nl = (long)i; break; }
    if (nl < 0) { fprintf(stderr, "page form: page without title line\n"); exit(1); }
    titel = b; titell = (size_t)nl;
    const char *rest = b + nl + 1;
    size_t restl = e - (size_t)nl - 1;

    long l = g1_rfind(rest, restl, '\n');
    if (l < 0) { fprintf(stderr, "page form: page without flag line\n"); exit(1); }
    flag = rest + l + 1; flagl = restl - (size_t)l - 1;
    restl = (size_t)l;

    l = g1_rfind(rest, restl, '\n');
    if (l >= 0 && (size_t)l + 1 < restl && rest[l + 1] == 0x02 &&
        !((size_t)l + 2 < restl && rest[l + 2] == 0x02)) {
      mark = rest + l + 1; markl = restl - (size_t)l - 1;
      restl = (size_t)l;
    }
    l = g1_rfind(rest, restl, '\n');
    if (l < 0) { fprintf(stderr, "page form: page without comment line\n"); exit(1); }
    body = rest; bodyl = (size_t)l;
    kom = rest + l + 1; koml = restl - (size_t)l - 1;
    if (koml) {
      if ((unsigned char)kom[0] != (unsigned char)3) {
        fprintf(stderr, "page form: comment line without marker byte 0x%02x\n", (unsigned)(unsigned char)3);
        exit(1);
      }
      ++kom; --koml;
    }
  }

  const int textda = bodyl != 0;
  const char *rc_vor = 0;
  if (redirect_block(g1_rc_dseiten)) {
    size_t rc_drop = 0;
    rc_vor = redirect_decode(body, bodyl, &rc_drop, g1_rc_dec);
    body += rc_drop;
    bodyl -= rc_drop;
  }
  ++g1_rc_dseiten;
  g1_reserve(&g1_eb, &g1_ec, g1_dn + 4096);
  size_t o = 0;
  char *ob = g1_eb;
#define G1_PUTS(str) do { size_t _l = strlen(str); memcpy(ob + o, str, _l); o += _l; } while (0)
#define G1_PUT(p, l) do { memcpy(ob + o, (p), (l)); o += (l); } while (0)
  G1_PUTS("  <page>\n    <title>");
  G1_PUT(titel, titell);
  G1_PUTS("</title>\n");
  if (flagl) G1_PUTS("      <minor />\n");
  if (koml) {
    G1_PUTS("      <comment>");
    if (!(koml == 1 && kom[0] == 0x02)) {
      for (size_t i = 0; i < koml; ++i) {
        if (kom[i] == 0x02 && i + 1 < koml && kom[i + 1] == 0x02) { ob[o++] = '\n'; ++i; }
        else ob[o++] = kom[i];
      }
    }
    G1_PUTS("</comment>\n");
  }
  if (textda) {
    G1_PUTS("      <text xml:space=\"preserve\">");
    if (rc_vor) G1_PUTS(rc_vor);
    G1_PUT(body, bodyl);
    G1_PUTS("</text>\n");
  } else {
    G1_PUTS("      <text xml:space=\"preserve\" />\n");          // corpus invariant: an empty text part is self-closing
  }
  G1_PUTS("    </revision>\n");
  if (markl) { G1_PUT(mark, markl); ob[o++] = '\n'; }
  G1_PUTS("  </page>\n");
#undef G1_PUTS
#undef G1_PUT
  g1_en = o; g1_epos = 0;
}

static int g1_dec_getline(char *s, FILE *in, U64 tsize) {
  if (g1_epos >= g1_en) {
    if ((U64)curpos(in) >= tsize) { s[0] = 0; return 0; }
    g1_dn = 0;
    for (;;) {
      char t[8192 * 8];
      int j = wfgets(t, 8192 * 8, in);
      if ((U64)curpos(in) > tsize) { j = j - (int)(curpos(in) - tsize); t[j] = 0; }
      if (j <= 0) break;
      g1_reserve(&g1_db, &g1_dc, g1_dn + (size_t)j + 1);
      memcpy(g1_db + g1_dn, t, (size_t)j); g1_dn += (size_t)j; g1_db[g1_dn] = 0;
      if (g1_dn >= 10 && memcmp(g1_db + g1_dn - 10, "  </page>\n", 10) == 0) break;
      if ((U64)curpos(in) >= tsize) break;
    }
    if (!g1_dn) { s[0] = 0; return 0; }
    g1_expand();
  }
  size_t i = g1_epos;
  while (i < g1_en && g1_eb[i] != '\n') ++i;
  if (i < g1_en) ++i;                       // include the newline
  const size_t l = i - g1_epos;
  memcpy(s, g1_eb + g1_epos, l); s[l] = 0;
  g1_epos = i;
  return (int)l;
}

static int g1_dec_offen(void) { return g1_epos < g1_en; }

static void g1_bericht(void) {
}

#endif
