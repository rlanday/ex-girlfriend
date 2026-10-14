// Based on STARLIT (Artemiy Margaritov, github.com/amargaritov/starlit,
// GPL-3.0), src/readalike_prepr/; inherited through fx2-cmix and cmix-lex.
#ifndef ARTICLE_REORDER_H
#define ARTICLE_REORDER_H

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

void bubblesort(std::vector<Accumulator>& mylist) {
    for (int i = 1; i < mylist.size(); i++)	{
        for (int j = 0; j < mylist.size() - i; j++) {
            if (mylist[j].id > mylist[j + 1].id) {
            std::swap(mylist[j], mylist[j + 1]);
            }
       }
   }
}

const char *patterns1[] = { "<page>", "<id>", "</page>" };
const int transitions1[3] = { expect_id, expect_pageend, expect_page };
std::vector<Accumulator> vec;
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
}

void loadFile(const char *fname) {

    line_count = 0;
    int  state = expect_page;
    Accumulator acc;
    {
      struct stat st;
      const size_t n = stat(fname, &st) == 0 ? (size_t)st.st_size : 0;
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
        vec.push_back(acc);
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
void reorder() {
    std::vector<int> positions;
    std::vector<int> used(NUM_OF_ARTICLES, 0);

    loadFile(".main");

    // Article numbers need to be remapped (see article_remap.cpp).
    // Here we read enwik9 in order to generate the article number mapping.
    std::unordered_map<int, int> remap;
    std::ifstream infile(".main");
    std::string line;
    int count1 = -1, count2 = -1;
    bool redirect = false;
    std::vector<std::string> prefix = {
      "      <text xml:space=\"preserve\">#REDIRECT",
      "      <text xml:space=\"preserve\">#redirect",
      "      <text xml:space=\"preserve\">#Redirect",
      "      <text xml:space=\"preserve\">#REdirect",
      "      <text xml:space=\"preserve\">{{softredirect",};
    while (getline(infile, line)) {
      for (auto pre : prefix) {
        if (line.rfind(pre, 0) == 0) {
          redirect = true;
          break;
        }
      }
      if (line == "  <page>") {
        remap[count2] = count1;
        if (!redirect) {
          ++count2;
        }
        ++count1;
        redirect = false;
      }
    }

    {
      std::vector<int> units;
      if (!ord::DecodeOrderV3(".new_article_order", ".main", &units)) {
        fprintf(stderr, "article order: reconstruction failed\n");
        exit(1);
      }
      for (size_t i = 0; i < units.size(); i++) {
        int res = remap[units[i]];    // Article numbers get remapped.
        positions.push_back(res);
        used[res] = 1;
      }
    }

    const int n_listed = (int)positions.size();
    const int n_front = n_listed;
    if (positions.size() < NUM_OF_ARTICLES) {
        for (int i = 0; i < NUM_OF_ARTICLES; i++) {
            if (used[i] == 0) {
                positions.push_back(i);
            }
       }
    }

  FILE* out = fopen(".main_reordered", "wb");
  for(int i = 0; i < positions.size(); i++) {
    const int n_rest = (int)positions.size() - n_front;
    int pos = positions[i < n_rest ? n_front + i : i - n_rest];
    writeLines(vec[pos].start, vec[pos].end, out);
  }
  fclose(out);
  vec.clear();
  std::vector<Accumulator>(vec).swap(vec);  // Why C++? why
  freeLines();
}

void sort() {
  loadFile(".main_decomp_restored");

  bubblesort(vec);

  FILE* out = fopen(".main_decomp_restored_sorted", "wb");
  for(int i =0; i < vec.size(); i++) {
    writeLines(vec[i].start, vec[i].end, out);
  }

  fclose(out);
  vec.clear();
  std::vector<Accumulator>(vec).swap(vec);
  freeLines();
}
#endif // ARTICLE_REORDER_H
