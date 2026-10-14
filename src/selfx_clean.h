#ifndef SELFX_CLEAN_H_
#define SELFX_CLEAN_H_

#include <stdio.h>
#include <string.h>


namespace selfx_clean {

// The names are ONE blob of NUL-separated strings, closed by an empty one.
// That is the whole data structure: no pointer table, no relocations, and the
// packer sees the names next to each other.  A name that is not there costs a
// failed `remove()` and nothing else.
inline void Sweep(const char* p) {
  for (; *p; p += strlen(p) + 1) remove(p);
}

// archive9, the `argc == 1` path.  Order as the run creates them.
inline void AfterDecompress() {
  Sweep(
      ".dict.comp_decomp\0"
      ".dict\0"
      ".ready4cmix_decomp\0"
      ".input_decomp\0"
      ".main_decomp\0"
      ".intro_decomp\0"
      ".coda_decomp\0"
      ".main_decomp_restored\0"
      ".main_decomp_restored_sorted\0"
      "un1_d\0"
      "weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4" "\0"
      "");
}

// The `-e` path of the compressor.  argv[3] is deliberately not reachable from
// here; it is the one file the caller asked for.
inline void AfterCompress() {
  Sweep(
      "test.dat\0"
      ".new_article_order.comp\0"
      ".new_article_order\0"
      ".dict\0"
      ".main\0"
      ".intro\0"
      ".coda\0"
      ".main_reordered\0"
      ".main_phda9prepr\0"
      "un1\0"
      ".ready4cmix\0"
      "dec1\0"
      "dec2\0"
      "header4archive.dat\0"
      "weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4" "\0"
      ".decomp_bin\0"
      ".dict.comp\0"
      "");
}

}

#endif
