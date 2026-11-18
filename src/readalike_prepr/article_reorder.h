// Based on STARLIT (Artemiy Margaritov, github.com/amargaritov/starlit,
// GPL-3.0), src/readalike_prepr/; inherited through fx2-cmix and cmix-lex.
#ifndef ARTICLE_REORDER_H
#define ARTICLE_REORDER_H
#include "../meld.h"

#include <string.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <unordered_map>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>

#include <stdlib.h>
#include "order_codec.h"

#define NUM_OF_ARTICLES 243425

struct Accumulator {
  int id;
  int start;
  int end;
};

enum ParserState {
  expect_page = 0,
  expect_id,
  expect_pageend
};

int line_count = 0;
static char s[8192*8];

const char *patterns1[] = { "<page>", "<id>", "</page>" };
const int transitions1[3] = { expect_id, expect_pageend, expect_page };
// pages of the loaded file: id and line range
static Accumulator* vec = nullptr;
static size_t vec_n = 0, vec_cap = 0;
static char* line_buf = nullptr;      // text of all lines, back to back
static uint64_t* line_off = nullptr;  // line i = line_buf[line_off[i] .. line_off[i+1])
static size_t line_n = 0, line_cap = 0;
// in phda9
int wfgets(char *str, int count, FILE  *fp);
void wfputs(const char *str,FILE *fp);

// Write lines a..b (inclusive) of the store.
static void writeLines(int a, int b, FILE* out) {
  fwrite(line_buf + line_off[a], 1, (size_t)(line_off[b + 1] - line_off[a]), out);
}

static void freeLines() {
  free(line_buf); line_buf = nullptr;
  free(line_off); line_off = nullptr;
  line_n = line_cap = 0;
  free(vec); vec = nullptr;
  vec_n = vec_cap = 0;
}

void loadFile(const char *fname) {

    line_count = 0;
    int  state = expect_page;
    Accumulator acc;
    {
      FILE* sf = fopen(fname, "rb");   // size without stat (one import less)
      size_t n = 0; if (sf) { fseek(sf, 0, SEEK_END); n = (size_t)ftell(sf); fclose(sf); }
      freeLines();
      line_buf = (char*)malloc(n + 1);  // the text never outgrows the file
      line_cap = n / 40 + 2;            // enwik9 averages ~75 B per line
      line_off = (uint64_t*)malloc(line_cap * sizeof(uint64_t));
      if (!line_buf || !line_off) exit(1);
      line_off[0] = 0;
    }
    FILE* file = fopen(fname, "rb");
    while (wfgets(s, 8192*8, file) )  {
    char *p = strstr(s, patterns1[state]);
    if (p) {
      if (state==expect_page){
           acc.start = line_count;
      }else if(state==expect_id){
           char *p = strstr(s, patterns1[1]); //id
           if (p){
               p=p+4;
               acc.id = atoi(p);
           }
      }else if(state==expect_pageend){
           acc.end = line_count;
      }
      state = transitions1[state];
      if (state == expect_page){
        if (vec_n == vec_cap) {
          vec_cap = vec_cap * 2 + 262144;
          vec = (Accumulator*)realloc(vec, vec_cap * sizeof(Accumulator));
          if (!vec) exit(1);
        }
        vec[vec_n++] = acc;
       }
    }
    line_count++;
    {
      const size_t len = strlen(s), o = line_off[line_n];
      memcpy(line_buf + o, s, len);
      if (++line_n == line_cap) {
        line_cap *= 2;
        line_off = (uint64_t*)realloc(line_off, line_cap * sizeof(uint64_t));
        if (!line_off) exit(1);
      }
      line_off[line_n] = o + len;
    }
  }
  fclose(file);

}
// The store holds .main up to its first NUL, and .main has none: the lines
// read below are those of the file.
void reorder() {
    loadFile(".main");
    const char* const txt = line_buf;
    const char* const txt_end = line_buf + line_off[line_n];

    // Article numbers need to be remapped (see article_remap.cpp): at every
    // page line the number of non-redirect pages before the previous page is
    // mapped to the index of that page (the last write wins, an unwritten
    // number maps to 0).  remap[u + 1] is the page of unit u.
    int* remap = (int*)calloc(NUM_OF_ARTICLES + 2, sizeof(int));
    {
      int count1 = -1, count2 = -1;
      bool redirect = false;
      for (const char* p = txt; p < txt_end; ) {
        const char* e = (const char*)memchr(p, '\n', (size_t)(txt_end - p));
        const size_t n = e ? (size_t)(e - p) : (size_t)(txt_end - p);
        if (ord::IsRedirLine(p, n)) redirect = true;
        if (n == 8 && memcmp(p, "  <page>", 8) == 0) {
          if (count2 + 1 <= NUM_OF_ARTICLES + 1) remap[count2 + 1] = count1;
          if (!redirect) ++count2;
          ++count1;
          redirect = false;
        }
        p += n + 1;
      }
    }

    int* units = (int*)malloc(NUM_OF_ARTICLES * sizeof(int));
    const int nu = ord::DecodeOrderV3(".new_article_order", txt, (size_t)(txt_end - txt),
                                     units, NUM_OF_ARTICLES);
    if (nu < 0) {
      msg_stop(115);
      exit(1);
    }
    int* positions = (int*)malloc((size_t)(nu + NUM_OF_ARTICLES) * sizeof(int));
    unsigned char* used = (unsigned char*)calloc(NUM_OF_ARTICLES, 1);
    int np = 0;
    for (int i = 0; i < nu; i++) {
      int res = remap[units[i] + 1];    // Article numbers get remapped.
      positions[np++] = res;
      used[res] = 1;
    }
    free(units); free(remap);

    const int n_front = np;
    if (np < NUM_OF_ARTICLES) {
        for (int i = 0; i < NUM_OF_ARTICLES; i++) {
            if (used[i] == 0) {
                positions[np++] = i;
            }
       }
    }
    free(used);

  FILE* out = fopen(".main_reordered", "wb");
  const int n_rest = np - n_front;
  for(int i = 0; i < np; i++) {
    int pos = positions[i < n_rest ? n_front + i : i - n_rest];
    writeLines(vec[pos].start, vec[pos].end, out);
  }
  fclose(out);
  free(positions);
  freeLines();
}

// pages by id ascending (the ids of enwik9 are distinct)
static int AccLess(const void* a, const void* b) {
  const int x = ((const Accumulator*)a)->id, y = ((const Accumulator*)b)->id;
  return (x > y) - (x < y);
}

void sort() {
  loadFile(".main_decomp_restored");

  qsort(vec, vec_n, sizeof(Accumulator), AccLess);

  FILE* out = fopen(".main_decomp_restored_sorted", "wb");
  for(size_t i =0; i < vec_n; i++) {
    writeLines(vec[i].start, vec[i].end, out);
  }

  fclose(out);
  freeLines();
}
#endif // ARTICLE_REORDER_H
