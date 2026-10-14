// Based on STARLIT (Artemiy Margaritov, github.com/amargaritov/starlit,
// GPL-3.0), src/readalike_prepr/; inherited through fx2-cmix and cmix-lex.
#ifndef SELF_EXTRACT_H
#define SELF_EXTRACT_H

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>

#include <string>

#include "../preprocess/dict_derive.h"
#include "../attributes.h"
#include "../helper_process.h"

static char fx_selfbuf[4096];
static const char* fx_self_path(const char* fallback) {
  if (fx_selfbuf[0] == 0) {
    ssize_t n = readlink("/proc/self/exe", fx_selfbuf, sizeof(fx_selfbuf) - 1);
    fx_selfbuf[n > 0 ? (size_t)n : 0] = 0;
    {
      static const char fx_mark[16] = {'B','5','C','2','S','E','L','F',
                                       'P','A','T','H','_','V','1','\0'};
      FILE* s = fopen("/proc/self/exe", "rb");
      if (s) {
        char m[16];
        unsigned int plen = 0;
        if (fseek(s, -16L, SEEK_END) == 0 && fread(m, 1, 16, s) == 16 &&
            memcmp(m, fx_mark, 16) == 0 &&
            fseek(s, -20L, SEEK_END) == 0 && fread(&plen, 1, 4, s) == 4 &&
            plen > 0 && plen < sizeof(fx_selfbuf) &&
            fseek(s, -(long)(20u + plen), SEEK_END) == 0 &&
            fread(fx_selfbuf, 1, plen, s) == plen) {
          fx_selfbuf[plen] = 0;
        }
        fclose(s);
      }
      // An unreadable /proc/self/exe is reported once; the caller's fallback
      // name is used then.
      static bool warned = false;
      if ((n <= 0 || !s) && !warned) {
        warned = true;
        fprintf(stderr, "/proc/self/exe unreadable\n");
      }
    }
  }
  return fx_selfbuf[0] ? fx_selfbuf : fallback;
}

static FILE* fx_open(const char* name, const char* mode) {
  FILE* f = fopen(name, mode);
  if (!f) {
    fprintf(stderr, "self-extraction: cannot open %s\n", name);
    exit(1);
  }
  return f;
}

#define FX_SELF_NOTF ""

#define FX_SELF_D "-D"
#define FX_SELF_C "-C"

static int fx_self_d(const char* self, const char* in, const char* out) {
  static char cmd[4400];
  snprintf(cmd, sizeof(cmd), FX_SELF_NOTF "\"%s\" " FX_SELF_D " %s %s",
      self, in, out);
  helper_process::VorStart(in);
  return system(cmd);
}

static int fx_self_c(const char* self, const char* in, const char* out) {
  static char cmd_c[4400];
  snprintf(cmd_c, sizeof(cmd_c), FX_SELF_NOTF "\"%s\" " FX_SELF_C " %s %s",
      self, in, out);
  helper_process::VorStart(in);
  return system(cmd_c);
}

struct HeaderInfo {
  int dict_size;
  int new_article_order_size;
  int decomp_input_size;
  int weights_size;
};

void write(const char* file_name, HeaderInfo& data) {
  FILE *out = fx_open(file_name , "wb" );
  fwrite(&data , 1 , sizeof(HeaderInfo) , out );
  fclose(out);
}

void read(const char* file_name, HeaderInfo& data) {
  FILE *in = fx_open(file_name , "rb" );
  fread(&data , 1 , sizeof(HeaderInfo) , in );
  fclose(in);
}

static size_t decomp_prefix_len(unsigned char* p1, size_t pkg_size) {
  static const char kMagic[15] = {'B','5','J','P','A','Y','L','O','A','D',
                                  '_','V','1','_','_'};
  size_t lim = pkg_size < (size_t)(1u << 20) ? pkg_size : (size_t)(1u << 20);
  size_t i;
  for (i = 0; i + 16 + 48 <= lim; ++i) {
    unsigned char* h;
    unsigned long long f[6];
    int k, b;
    if (memcmp(p1 + i, kMagic, 15) != 0) continue;
    h = p1 + i + 16;
    for (k = 0; k < 6; ++k) {
      unsigned long long v = 0;
      for (b = 7; b >= 0; --b) v = (v << 8) | h[k * 8 + b];
      f[k] = v;
    }
    /* 0 payload_off  1 comp_len  2 decode_len  3 image_len
       4 bcj_split (historical name: the cut point Nd)    5 dec_clen (=K) */
    if (f[0] == 0 || f[5] == 0) return 0;
    if (f[0] + f[1] != (unsigned long long)pkg_size) return 0;
    if (f[5] > f[1] || f[4] == 0 || f[4] >= f[3]) return 0;
    for (b = 0; b < 8; ++b) {
      h[1 * 8 + b] = (unsigned char)((f[5] >> (8 * b)) & 0xff);  /* comp_len   <- K  */
      h[2 * 8 + b] = (unsigned char)((f[4] >> (8 * b)) & 0xff);  /* decode_len <- Nd */
    }
    return (size_t)(f[0] + f[5]);
  }
  return 0;
}

// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it)
// Writes the transformer weight container out of the program's own image for
// the single-file modes (-c/-d/-n); selfextract_comp() runs only on the -e
// path.  Extracts only the fourth block, does nothing if the container is
// already present, and returns quietly if nothing usable is found (the engine
// then reports the missing container).
static void extract_weights_only() {
  FILE* present = fopen("weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4", "rb");
  if (present) { fclose(present); return; }      // already there

  const char* self = fx_self_path("cmix");
  FILE* f = fopen(self, "rb");
  if (!f) return;
  fseek(f, 0, SEEK_END);
  size_t fsize = (size_t)ftell(f);
  if (fsize <= sizeof(HeaderInfo)) { fclose(f); return; }

  HeaderInfo header;
  fseek(f, (long)(fsize - sizeof(HeaderInfo)), SEEK_SET);
  if (fread(&header, sizeof(HeaderInfo), 1, f) != 1) { fclose(f); return; }
  size_t w = (size_t)header.weights_size;
  if (w == 0) { fclose(f); return; }
  size_t d = (size_t)header.dict_size;
  size_t o = (size_t)header.new_article_order_size;
  if (w + d + o + sizeof(HeaderInfo) > fsize) { fclose(f); return; }

  // The container is the fourth block, directly in front of the header.
  size_t off = fsize - sizeof(HeaderInfo) - w;
  unsigned char* buf = (unsigned char*)malloc(w);
  if (!buf) { fclose(f); return; }
  fseek(f, (long)off, SEEK_SET);
  size_t got = fread(buf, 1, w, f);
  fclose(f);
  if (got != w) { free(buf); return; }

  FILE* fo = fopen("weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4", "wb");
  if (fo) { fwrite(buf, 1, w, fo); fclose(fo); }
  free(buf);
}

int selfextract_comp() {
  HeaderInfo header;

// open itslef to read auxilary data (dictionary and neworder)
  FILE *f = NULL, *fo = NULL;
  const char* self = fx_self_path("cmix");
  f = fx_open(self, "rb");

  // get the size of the whole binary
  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  // read the whole binary to memory
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = fx_open("test.dat", "wb");
  memcpy(&header, p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo));
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);

  //Remove dictionary if present
  remove(".dict");

  size_t weights_len = (size_t)header.weights_size;
  size_t decmpressor_binary_size = fsize - header.dict_size
      - header.new_article_order_size - weights_len - sizeof(HeaderInfo);

  {
    size_t dbin_size = decmpressor_binary_size;
    size_t decomp_prefix = decomp_prefix_len(p1, decmpressor_binary_size);
    if (decomp_prefix) dbin_size = decomp_prefix;
    fo = fx_open(".decomp_bin", "wb");
    fwrite(p1, dbin_size, 1, fo);
    fclose(fo);
  }

  fo = fx_open(".hints.comp", "wb");
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);

// produce article order and decompress it
  fo = fx_open(".new_article_order.comp", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size, header.new_article_order_size, 1, fo);
  fclose(fo);

  if (weights_len) {
    fo = fx_open("weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4", "wb");
    fwrite(p1 + decmpressor_binary_size + header.dict_size
              + header.new_article_order_size, weights_len, 1, fo);
    fclose(fo);
  }
  free(p1);
  p1 = NULL;
  malloc_trim(0);

  int status = fx_self_d(self, ".new_article_order.comp", ".new_article_order");
  if (status != 0) {
    fprintf(stderr, "self-extraction: decoding the article order failed (status %d)\n", status);
    free(p1);
    return 1;
  }

//  std::cout << "Decompressing dictionary..." << std::endl;
  // the hint package is half the size of the dictionary, so this is faster
  status = fx_self_d(self, ".hints.comp", ".hints");
  if (status != 0) {
    fprintf(stderr, "self-extraction: decoding the dictionary hints failed (status %d)\n", status);
    free(p1);
    return 1;
  }
  free(p1);
  malloc_trim(0);
  return 0;
}

// Same as previous function, but used in decompressor
// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it)
int selfextract_decomp() {
  HeaderInfo header;
  FILE *f = NULL, *fo = NULL;
  const char* self = fx_self_path("archive9");
  f = fx_open(self, "rb");

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = NULL;
  (void)fo;          // with the piecewise copy BlockAusAbbild writes, fo stays unused
  if (fseek(f, (long)(fsize - sizeof(HeaderInfo)), SEEK_SET) != 0 ||
      fread(&header, sizeof(HeaderInfo), 1, f) != 1) {
    fprintf(stderr, "self-extraction: cannot read the archive trailer\n");
    return 1;
  }

  //Remove dictionary if present
  remove(".dict");

  size_t weights_len = (size_t)header.weights_size;
  size_t decmpressor_binary_size = fsize - header.dict_size
      - header.decomp_input_size - weights_len - sizeof(HeaderInfo);

  // .ready4cmix_decomp is written only after the helper: without p1 waiting
  // costs nothing, and its page cache does not load the cgroup while the
  // helper runs.
  if (helper_process::BlockAusAbbild(f, (long)decmpressor_binary_size,
                                     (size_t)header.dict_size,
                                     ".dict.comp_decomp") != 0) {
    fprintf(stderr, "self-extraction: cannot extract %s\n", ".dict.comp_decomp");
    return 1;
  }
  if (weights_len &&
      helper_process::BlockAusAbbild(f,
          (long)(decmpressor_binary_size + header.dict_size),
          weights_len, "weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4") != 0) {
    fprintf(stderr, "self-extraction: cannot extract %s\n", "weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4");
    return 1;
  }

  helper_process::CacheZurueck(f);

  int status = fx_self_d(self, ".dict.comp_decomp", ".dict");
  if (status != 0) {
    fprintf(stderr, "self-extraction: decoding the dictionary failed (status %d)\n", status);
    free(p1);
    return 1;
  }

  if (helper_process::BlockAusAbbild(f,
          (long)(decmpressor_binary_size + header.dict_size
                 + weights_len
                 ), (size_t)header.decomp_input_size,
          ".ready4cmix_decomp") != 0) {
    fprintf(stderr, "self-extraction: cannot extract %s\n", ".ready4cmix_decomp");
    return 1;
  }
  helper_process::CacheZurueck(f);
  fclose(f);

  free(p1);
  malloc_trim(0);
  return 0;
}

#endif // PREPR_H
