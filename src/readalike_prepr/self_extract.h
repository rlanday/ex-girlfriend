// Based on STARLIT (Artemiy Margaritov, github.com/amargaritov/starlit,
// GPL-3.0), src/readalike_prepr/; inherited through fx2-cmix and cmix-lex.
#ifndef SELF_EXTRACT_H
#define SELF_EXTRACT_H
#include "../meld.h"

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
      if (!s && !warned) {
        warned = true;
        msg_note(163);
      }
    }
  }
  return fx_selfbuf[0] ? fx_selfbuf : fallback;
}

static FILE* fx_open(const char* name, const char* mode) {
  FILE* f = fopen(name, mode);
  if (!f) {
    msg_stop(164);
    exit(1);
  }
  return f;
}

#define FX_SELF_NOTF ""

#define FX_SELF_D "-D"
#define FX_SELF_C "-C"

#include "../enc_only.h"
ENC_ONLY
static int fx_self_d(const char* self, const char* in, const char* out) {
  static char cmd[4400];
  snprintf(cmd, sizeof(cmd), FX_SELF_NOTF "\"%s\" " FX_SELF_D " %s %s",
      self, in, out);
  helper_process::VorStart(in);
  return system(cmd);
}

struct HeaderInfo {
  int dict_size;
  int new_article_order_size;
  int weights_size;
};
struct ArchivKopf {
  int decomp_input_size;   // the payload
  int weights_size;        // the weight container
};

static size_t decomp_prefix_len(unsigned char* p1, size_t pkg_size) {
  unsigned char* h = p1 + 120;
  unsigned long long f[6];
  int k, b;
  if (pkg_size < 120 + 24 || memcmp(p1, "\177ELF", 4) != 0 || p1[56] != 1 || p1[57] != 0) return 0;
  for (k = 0; k < 6; ++k) {
    unsigned long long v = 0;
    for (b = 3; b >= 0; --b) v = (v << 8) | h[k * 4 + b];
    f[k] = v;
  }
  /* 0 payload_off  1 comp_len  2 decode_len  3 image_len
     4 bcj_split (historical name: the cut point Nd)    5 dec_clen (=K) */
  if (f[0] == 0 || f[5] == 0) return 0;
  if (f[0] + f[1] != (unsigned long long)pkg_size) return 0;
  if (f[5] > f[1] || f[4] == 0 || f[4] >= f[3]) return 0;
  for (b = 0; b < 4; ++b) {
    h[1 * 4 + b] = (unsigned char)((f[5] >> (8 * b)) & 0xff);  /* comp_len   <- K  */
    h[2 * 4 + b] = (unsigned char)((f[4] >> (8 * b)) & 0xff);  /* decode_len <- Nd */
  }
  return (size_t)(f[0] + f[5]);
}

// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it)

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
  memcpy(&header, p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo));

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
    fo = fx_open("weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5", "wb");
    fwrite(p1 + decmpressor_binary_size + header.dict_size
              + header.new_article_order_size, weights_len, 1, fo);
    fclose(fo);
  }
  free(p1);
  p1 = NULL;
  malloc_trim(0);

  int status = fx_self_d(self, ".new_article_order.comp", ".new_article_order");
  if (status != 0) {
    msg_note(165);
    free(p1);
    return 1;
  }

//  std::cout << "Decompressing dictionary..." << std::endl;
  // the hint package is half the size of the dictionary, so this is faster
  status = fx_self_d(self, ".hints.comp", ".hints");
  if (status != 0) {
    msg_note(166);
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
  ArchivKopf header;
  FILE *f = NULL, *fo = NULL;
  const char* self = fx_self_path("archive9");
  f = fx_open(self, "rb");

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = NULL;
  (void)fo;          // with the piecewise copy BlockAusAbbild writes, fo stays unused
  if (fseek(f, (long)(fsize - sizeof(header)), SEEK_SET) != 0 ||
      fread(&header, sizeof(header), 1, f) != 1) {
    msg_note(167);
    return 1;
  }

  //Remove dictionary if present
  remove(".dict");

  const size_t dict_block_size = 0;   // no dictionary block (slot 1 of the stream)
  size_t weights_len = (size_t)header.weights_size;
  size_t decmpressor_binary_size = fsize - dict_block_size
      - header.decomp_input_size - weights_len - sizeof(header);

  // .ready4cmix_decomp is written only after the helper: without p1 waiting
  // costs nothing, and its page cache does not load the cgroup while the
  // helper runs.
  if (weights_len &&
      helper_process::BlockAusAbbild(f,
          (long)(decmpressor_binary_size + dict_block_size),
          weights_len, "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5") != 0) {
    msg_note(168);
    return 1;
  }

  if (helper_process::BlockAusAbbild(f,
          (long)(decmpressor_binary_size + dict_block_size
                 + weights_len
                 ), (size_t)header.decomp_input_size,
          ".ready4cmix_decomp") != 0) {
    msg_note(168);
    return 1;
  }
  helper_process::CacheZurueck(f);
  fclose(f);

  free(p1);
  malloc_trim(0);
  return 0;
}

#endif // PREPR_H
