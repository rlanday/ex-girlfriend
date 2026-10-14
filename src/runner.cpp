#include <fstream>
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <malloc.h>

#include "preprocess/preprocessor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "predictor.h"

#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/self_extract.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/misc.h"
#include "r1_reorder_transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "isa_floor.h"
#include "page_contexts.h"
#include "fast_math.h"
#include "word_recency.h"
#include "simd_scan.h"
#include "invariants.h"
#include "huge_pages.h"
#include "mixer_lr.h"
#include "lstm_fx_link.h"
#include "mixer_decay.h"
#include "fx_mixer_lr.h"
#include "mixer_bank.h"
#include "logit_table.h"
#include "mixer_weights.h"
#include "lstm_sharpen.h"
#include "vocab_frozen.h"
#include "run_mode.h"

static_assert(54857 <= 65360,
    "exceeds the capacity of the code plan "
    "(dictionary.cpp kBoundary4 or kBoundary5): the words above it "
    "would get NO code word and be skipped silently.  Use a smaller "
    "dictionary, or open the fifth band and update EVERY reader of the "
    "band plan.");

static_assert(54857 <= run_mode::kBand4,
    "exceeds the bands the grammar automaton knows (kBand4): "
    "Idx3() would return a negative index for the fifth band and "
    "BuildMasks() would reject its second byte.  The grammar automaton "
    "needs a fifth band first.");

static_assert(54857 <= (80 + 3840 + 40960 + 20480),
    "exceeds the bands the PPMd warm-up knows "
    "(Code(), kBandKap4): its last branch would write a wrong code word for "
    "the fifth band and Material() would drop the words above.  Add the "
    "fifth band there first.");

// Run-time parameters: each one returns its compiled default.  The
// manifest writer is empty in this build (see its call in main()).
namespace run_options {
static double ParamD(const char* name, double def) {
  (void)name;
  return def;
}
static long ParamI(const char* name, long def) {
  double v = ParamD(name, (double)def);
  return (long)(v < 0 ? v - 0.5 : v + 0.5);
}
static void WriteManifest(const char* path, int argc, char** argv) {
  (void)path; (void)argc; (void)argv;
}
}

#include "attributes.h"

#include "run_mode.h"

#include "transformer.h"

#include "alias_context.h"
#include "alias_title_subst.h"
#include "author_block.h"
#include "metablock.h"
#include "ip_pack.h"
#include "metablock_form.h"
#include "ts_progress.h"



#include <malloc.h>
#ifndef AUX_BUILD
#define AUX_BUILD 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

namespace {
  constexpr long io_buffer_kb = 256;
  long progress_steps = 10000;  // progress ticks per run (they also drive the huge-page collapse)
}

static bool provide_dict_file(const char* src) {
  FILE* in = fopen(src, "rb");
  if (!in) return false;
  FILE* out = fopen(".dict", "wb");
  if (!out) { fclose(in); return false; }
  char buf[1 << 16];
  size_t n;
  bool ok = true;
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
    if (fwrite(buf, 1, n, out) != n) { ok = false; break; }
  }
  fclose(in);
  if (fclose(out) != 0) ok = false;
  return ok;
}

int Help() {
  printf("cmix -e enwik9 out | -c [dict] in out | -d [dict] in out | -h a b c\n");
  return -1;
}

size_t getFileSize(const std::string& path) {
  // // get the size of the output file
  FILE *f = fopen(path.c_str(), "rb");
  if (f == NULL) {
    printf("can't open file for measuring its size\n");
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size_t output_size = ftell(f);
  fclose(f);
  return output_size;
}

void WriteStorageHeader(FILE* out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used) c = 0x80;
    putc(c, out);
  }
}

void ClearOutput() {

}

#include "preprocess/dict_derive.h"
#include "huge_pages.h"
#include "ppm_logit_prior.h"

#include "vocab_frozen.h"
#include "console_progress.h"

// Console marks of the stream sections and of the BITLSTM thaw, checked at
// every progress tick of Compress() and Decompress().  The sources see the
// same stream bytes on both sides and the tick sits at the same place of
// both loops (after the byte is complete), so both directions print the same
// lines.  Metadata and redirect block print their exact first byte (the
// metablock unit counts its lines), the main text block and the thaw the
// position of the tick that first sees them (at most stream/10000 later).
static int progress_sections = 0;       // stream sections announced so far
static bool progress_thawed = false;   // thaw announced
static void ProgressMarks(unsigned long long pos, unsigned long long total) {
  if (console_progress::Quiet()) return;
  if (progress_sections == 0 && metablock::mk.Offen()) {
    console_progress::Mark("entering metadata block", metablock::mk.Ab(), total);
    progress_sections = 1;
  }
  if (progress_sections == 1 && metablock::mk.Fertig()) {
    console_progress::Mark("entering redirect block", metablock::mk.Bis(), total);
    progress_sections = 2;
  }
  if (progress_sections == 2 && !redirect_block(author_block::aut.Seite())) {
    console_progress::Mark("entering main text block", pos, total);
    progress_sections = 3;
  }
  if (!progress_thawed && bit_head::g_unfrozen) {
    console_progress::Mark("unfreeze BITLSTM", pos, total);
    progress_thawed = true;
  }
  (void)pos; (void)progress_thawed;
}

#include "selfx_clean.h"

void WriteHeaderVocfix(unsigned long long length, bool dictionary_used,
    std::ofstream* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x7F;
      if (dictionary_used) c |= 0x80;
      if (vocab_frozen::g_aux) c |= 0x40;
    }
    os->put(c);
  }
}

void ReadHeaderVocfix(std::ifstream* is, unsigned long long* length,
    bool* dictionary_used, std::vector<bool>* vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = is->get();
    if (i == 0) {
      if (c&0x80) *dictionary_used = true;
      else *dictionary_used = false;
      c &= 0x7F;
      vocab_frozen::g_aux = (c & 0x40) != 0;
      c &= 0x3F;
    }
    *length += c;
  }
  vocab_frozen::FillVocab(vocab);
}

static inline bool AuthorBlockInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

static inline bool MetablockInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

static inline bool TimestampProgressInert() {
  if (transformer::BlobForm()) return true;
  if (vocab_frozen::g_aux) return true;
  return false;
}

void Compress(unsigned long long input_bytes, std::ifstream* is,
    std::ofstream* os, unsigned long long* output_bytes, Predictor* p) {
  Encoder e(os, p);

  unsigned long long percent =
      1 + (input_bytes / (unsigned long long)progress_steps);
  unsigned long long prog_left = 1;
  ClearOutput();
  size_t buffer_size = (size_t)io_buffer_kb * 1024;
  std::vector<char> buffer(buffer_size);
  unsigned long long pos = 0;
  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer_size, input_bytes - pos));
    is->read(buffer.data(), chunk);
    const size_t got = static_cast<size_t>(is->gcount());
    if (got == 0) break;
    for (size_t i = 0; i < got; ++i) {
      unsigned char c = static_cast<unsigned char>(buffer[i]);
      if (run_mode::gram.ByteActive(pos)) {
        unsigned int ctx = 1;
        for (int j = 7; j >= 0; --j) {
          const int bit = (c >> j) & 1;
          const int f = run_mode::gram.Forced(ctx);
          if (f < 0) {
            run_mode::gram.CountFree(ctx);
            e.Encode(bit);
          } else {
            run_mode::gram.CountKnown(ctx);
            if (f != bit) run_mode::gram.Violation(pos, ctx, f, bit);
            e.ObserveKnownBit(bit);
          }
          ctx += ctx + bit;
        }
        run_mode::gram.NextByte(c);
      } else
      {
      for (int j = 7; j >= 0; --j) {
        e.Encode((c>>j)&1);
      }
      }
      if (!MetablockInert()) {
        metablock::mk.Byte(c);
        if (metablock::mk.Endet()) {
          run_mode::gram.ZoneOeffnen(metablock::mk.Bis());
          p->WordlblBlockEnde();
          p->MetalrBlockEnde();
        }
      }
      // The byte is fully coded, so it is fixed on both sides.  If it closed a
      // title span, the alias rows of this article are pushed through the
      // transformer here, and through no other model: observed, never coded.
      // Beginn/Ende (begin/end) bracket them for it.  Mirrored in Decompress().
      alias_context::aut.Observe(c);
      {
        const unsigned char* blk = nullptr;
        const int n = alias_context::aut.TakeBlock(&blk);
        if (n) alias_prompt::Beginn(n);
        if (n) alias_title_subst::Beginn(alias_context::aut.PraefixLen());
        for (int k = 0; k < n; ++k) e.ObserveTfOnlyByte(blk[k]);
        alias_prompt::Ende();
        alias_title_subst::Ende();
      }
      if (!AuthorBlockInert()) author_block::aut.Observe(c);
      {
        const unsigned char* ab = nullptr;
        const int an = author_block::aut.TakeBlock(&ab);
        // The transformer must classify these bytes as a PROMPT block (feed 1
        // / predict 0), exactly as the alias prompt form does - mirrored, or
        // the archive is unreadable.
        if (an) tf_author_prompt::Beginn(an);
        for (int k = 0; k < an; ++k) e.ObserveKnownByte(ab[k]);
        tf_author_prompt::Ende();
      }
      if (!TimestampProgressInert()) {
        ts_progress::aut.Observe(c);
        const unsigned char* tb = nullptr;
        const int tn = ts_progress::aut.TakeBlock(&tb);
        if (tn) timestamp_skip::Beginn();
        for (int k = 0; k < tn; ++k) e.ObserveKnownByte(tb[k]);
        timestamp_skip::Ende();
      }
      if (--prog_left == 0) {
        prog_left = percent;
        double frac = 100.0 * pos / input_bytes;

        HUGEPAGE_TICK(pos);
        console_progress::Tick(pos, input_bytes);
        ProgressMarks(pos, input_bytes);
      }
      ++pos;
    }
  }
  e.Flush();
  run_mode::gram.Report("compress");
  if (!MetablockInert()) {
    metablock::mk.Report("compress");
    metablock::mk.ByteWacht("compress", pos);
  }
  redirect_table::tab.Schluss();
  alias_table::tab.Schluss();
  alias_context::aut.Finish();
  alias_context::aut.Report("compress");
  if (!AuthorBlockInert()) {
    author_block::aut.Finish();
    author_block::aut.Report("compress");
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Finish();
    ts_progress::aut.Report("compress");
  }
  *output_bytes = os->tellp();
}

void Decompress(unsigned long long output_length, std::ifstream* is,
                std::ofstream* os, Predictor* p) {
  Decoder d(is, p);
  unsigned long long percent = 1 + (output_length / 10000);
  unsigned long long prog_left = 1;
  ClearOutput();
  for(unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    if (run_mode::gram.ByteActive(pos)) {
      while (byte < 256) {
        const int f = run_mode::gram.Forced((unsigned int)byte);
        int bit;
        if (f < 0) {
          run_mode::gram.CountFree((unsigned int)byte);
          bit = d.Decode();
        } else {
          run_mode::gram.CountKnown((unsigned int)byte);
          bit = f;
          d.ObserveKnownBit(bit);
        }
        byte += byte + bit;
      }
      run_mode::gram.NextByte((unsigned int)byte & 255);
    } else
    {
    while (byte < 256) {
      byte += byte + d.Decode();
    }
    }
    os->put(byte);
    if (!MetablockInert()) {
      metablock::mk.Byte((unsigned char)(byte & 255));
      if (metablock::mk.Endet()) {
        run_mode::gram.ZoneOeffnen(metablock::mk.Bis());
        p->WordlblBlockEnde();
        p->MetalrBlockEnde();
      }
    }
    alias_context::aut.Observe((unsigned char)(byte & 255));
    {
      const unsigned char* blk = nullptr;
      const int n = alias_context::aut.TakeBlock(&blk);
      if (n) alias_prompt::Beginn(n);
      if (n) alias_title_subst::Beginn(alias_context::aut.PraefixLen());
      for (int k = 0; k < n; ++k) d.ObserveTfOnlyByte(blk[k]);
      alias_prompt::Ende();
      alias_title_subst::Ende();
    }
    if (!AuthorBlockInert())
      author_block::aut.Observe((unsigned char)(byte & 255));
    {
      const unsigned char* ab = nullptr;
      const int an = author_block::aut.TakeBlock(&ab);
      if (an) tf_author_prompt::Beginn(an);   // mirrored
      for (int k = 0; k < an; ++k) d.ObserveKnownByte(ab[k]);
      tf_author_prompt::Ende();
    }
    if (!TimestampProgressInert()) {
      ts_progress::aut.Observe((unsigned char)(byte & 255));
      const unsigned char* tb = nullptr;
      const int tn = ts_progress::aut.TakeBlock(&tb);
      if (tn) timestamp_skip::Beginn();   // mirrored
      for (int k = 0; k < tn; ++k) d.ObserveKnownByte(tb[k]);
      timestamp_skip::Ende();
    }
    if (--prog_left == 0) {
      prog_left = percent;
      double frac = 100.0 * pos / output_length;
      HUGEPAGE_TICK(pos);
      console_progress::Tick(pos, output_length);
      ProgressMarks(pos, output_length);
    }
  }
  run_mode::gram.Report("decompress");
  if (!MetablockInert()) {
    metablock::mk.Report("decompress");
    metablock::mk.ByteWacht("decompress", output_length);
  }
  redirect_table::tab.Schluss();
  alias_table::tab.Schluss();
  alias_context::aut.Finish();
  alias_context::aut.Report("decompress");
  if (!AuthorBlockInert()) {
    author_block::aut.Finish();
    author_block::aut.Report("decompress");
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Finish();
    ts_progress::aut.Report("decompress");
  }
}

bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
  preprocessor::Encode(data_in, data_out, *input_bytes, temp_path, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

bool RunCompression(bool enable_preprocess, const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* temp_out = fopen(temp_path.c_str(), "wb");
  if (!temp_out) return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);

  if (enable_preprocess) {
    preprocessor::Encode(data_in, temp_out, *input_bytes, temp_path,
        dictionary);
  } else {
    preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
  }
  fclose(data_in);
  fclose(temp_out);

  if (!metablock_form::Pack(temp_path)) {
    fprintf(stderr, "metablock codec: pack failed\n");
    return false;
  }

  if (!ip_pack::Pack(temp_path)) {
    fprintf(stderr, "ip pack: pack failed\n");
    return false;
  }

  console_progress::PhaseEnd();   // "preprocessing enwik9" ends here

  std::ifstream temp_in(temp_path, std::ios::in | std::ios::binary);
  if (!temp_in.is_open()) return false;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  temp_in.seekg(0, std::ios::end);
  unsigned long long temp_bytes = temp_in.tellg();
  temp_in.seekg(0, std::ios::beg);

  std::vector<bool> vocab(256, false);
#if AUX_BUILD
  vocab_frozen::g_aux = true;
#endif
  vocab_frozen::FillVocab(&vocab);
  WriteHeaderVocfix(temp_bytes, dictionary != NULL, &data_out);

  Predictor p(vocab);
  if (dictionary) console_progress::Phase("pretraining the model on the dictionary");
  if (enable_preprocess) preprocessor::Pretrain(&p, dictionary);
  HUGEPAGE_COLLAPSE("c-init");
  if (dictionary != NULL) {
    run_mode::gram.Init(dictionary, temp_bytes,
        post_wrt_side_path ? (temp_bytes - getFileSize(post_wrt_side_path)) : 0);
    run_mode::gram.ZoneSperren();
  }
  malloc_trim(0);
  console_progress::PhaseEnd();   // "pretraining ..." ends here
  Compress(temp_bytes, &temp_in, &data_out, output_bytes, &p);
  temp_in.close();
  data_out.close();
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  std::vector<bool> vocab(256, false);
  bool dictionary_used;
  ReadHeaderVocfix(&data_in, output_bytes, &dictionary_used, &vocab);
  if (!dictionary_used && dictionary != NULL) return false;
  if (dictionary_used && dictionary == NULL) return false;

  {
    Predictor p(vocab);
    if (dictionary) console_progress::Phase("pretraining the model on the dictionary");
    if (dictionary_used) preprocessor::Pretrain(&p, dictionary);
    HUGEPAGE_COLLAPSE("d-init");
    if (dictionary_used && dictionary != NULL) {
      run_mode::gram.Init(dictionary, *output_bytes, 0);
      run_mode::gram.ZoneSperren();
    }

    std::ofstream temp_out(temp_path, std::ios::out | std::ios::binary);
    if (!temp_out.is_open()) return false;

    console_progress::PhaseEnd();   // "pretraining ..." ends here
    Decompress(*output_bytes, &data_in, &temp_out, &p);
    // the inverse transforms of the main run take some minutes
    if (dictionary) console_progress::Phase("undoing the preprocessing");
    p.FreeFxcmMemory();
    p.FreeFxcmRest();
    data_in.close();
    temp_out.close();
  }
  malloc_trim(0);

  if (!ip_pack::Unpack(temp_path)) {
    fprintf(stderr, "ip pack: unpack failed\n");
    return false;
  }

  if (!metablock_form::Unpack(temp_path)) {
    fprintf(stderr, "metablock codec: unpack failed\n");
    return false;
  }

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

namespace ppmd_page { extern int hilfs; }
int main(int argc, char** argv) {
  console_progress::Begin();
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'c' || argv[1][1] == 'd'
      || argv[1][1] == 'C' || argv[1][1] == 'D'))
    ppmd_page::hilfs = 1;
  mallopt(M_MMAP_THRESHOLD, 1 << 20);
  if (PredictorLayoutTagFast() != ((unsigned long long)sizeof(Predictor) * 4ULL + (unsigned long long)(1) * 2ULL + (unsigned long long)(0))) {
    fprintf(stderr,
        "\nbuild check: translation units disagree on sizeof(Predictor) (%llu, %llu)\n",
        (unsigned long long)((unsigned long long)sizeof(Predictor) * 4ULL + (unsigned long long)(1) * 2ULL + (unsigned long long)(0)),
        PredictorLayoutTagFast());
    fflush(stderr);
    return -1;
  }
  if ((argc != 1) && (argv[1][1] != 'h') && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'x' && argv[1][1] != 's' &&
      argv[1][1] != 'n' && argv[1][1] != 'e'
      && argv[1][1] != 'C' && argv[1][1] != 'D'
      ))) {
    return Help();
  }
  if (argc > 1 && (argv[1][1] == 'C' || argv[1][1] == 'D')) {
    transformer::SetBlobForm();
    argv[1][1] = (argv[1][1] == 'C') ? 'c' : 'd';
  }
  if (argc > 1
      && !transformer::BlobForm()
     ) {
    const char run_mode = argv[1][1];
    if (run_mode == 'c' || run_mode == 'd' || run_mode == 'n' ||
        run_mode == 's' || run_mode == 'x' || run_mode == 'e') {
      extract_weights_only();
    }
  }
  srand((unsigned int)run_options::ParamI("seed", SEED));
  progress_steps = run_options::ParamI("progress_steps", 10000);
  if (progress_steps < 1) progress_steps = 10000;

  clock_t start = clock();

  bool enable_preprocess = true;
  std::string input_path ;
  std::string output_path;
  FILE* dictionary = NULL;

  if ((argc > 1) && (argv[1][1] != 'h'))  {
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n') return Help();
      dictionary = fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
  }

  std::string temp_path = output_path + ".cmix.temp";
  const bool progress_encoding = argc > 1 && (argv[1][1] == 'c' || argv[1][1] == 'e');
  console_progress::Start(progress_encoding, output_path);

  // Manifest hook.  WriteManifest() writes nothing in this build; the call
  // is kept because it is part of the binary that was run.
  if (argc > 1 && !output_path.empty()) {
    const char run_mode = argv[1][1];
    if (run_mode == 'c' || run_mode == 'n' || run_mode == 'e' ||
        run_mode == 's') {
      run_options::WriteManifest((output_path + ".features.txt").c_str(), argc, argv);
    }
  }

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argc == 1) {
    //Decompress enwik9
    // unpack a) header b) cmix dictionary, c) new order of articles, d) actual cmix binary
    if (selfextract_decomp() != 0) {
      return Help();
    }

    // run compression
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp" ;
    dictionary = fopen(".dict", "rb");//_decomp

    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes,
        nullptr
        )) {
      return Help();
    }

    split4Decomp();

    // apply phda9 preprocessor
    phda9_resto();

    // change the order of articles in the input
    sort();

    if (!cmtidx_inplace(".main_decomp_restored_sorted", ".cmtidx_resto", 0)) {
      fprintf(stderr, "comment index: inverse transform failed\n");
      return Help();
    }

    // merge all input parts after preprocessing
    cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d");
    cat("un1_d", ".coda_decomp", ".census_resto");
    if (!census_file(".census_resto", "enwik9_uncompressed", 0)) {
      fprintf(stderr, "census: inverse transform failed\n");
      return Help();
    }
    remove(".census_resto");

    selfx_clean::AfterDecompress();

    goto print_end_message;
  }

  if (argv[1][1] == 'c') {
      if (argv[1][1] == 'c' && argc == 5) {
        if (!provide_dict_file(argv[2])) return Help();
      }
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'e') {
    // Compress enwik9
    input_path = argv[2];
    output_path = argv[3]; //name of a compressor output

    if (selfextract_comp() != 0) {
      return Help();
    }

    {
      console_progress::Phase("deriving the dictionary from enwik9");
      const bool derived = dict_derive::BuildDictFromInput(
          input_path.c_str(), ".hints", ".dict_derived");
      console_progress::PhaseEnd();
      if (!derived) {
        fprintf(stderr, "dictionary hints: derivation failed\n");
        return Help();
      }
      int b6_status = fx_self_c(fx_self_path("cmix"), ".dict_derived",
                                ".dict.comp");
      if (b6_status != 0) {
        fprintf(stderr, "dictionary hints: .dict.comp status=%d\n",
                b6_status);
        return Help();
      }
      remove(".dict");
      if (rename(".dict_derived", ".dict") != 0) {
        fprintf(stderr, "dictionary hints: cannot create .dict\n");
        return Help();
      }
      remove(".hints");
      remove(".hints.comp");
    }

    console_progress::Phase("preprocessing enwik9");
    if (!census_file(input_path.c_str(), ".census_prepr", 1)) {
      fprintf(stderr, "census: preprocessing failed\n");
      return Help();
    }
    split4Comp(".census_prepr");
    remove(".census_prepr");
    if (!cmtidx_inplace(".main", ".cmtidx_main", 1)) {
      fprintf(stderr, "comment index: preprocessing failed\n");
      return Help();
    }

    // change the order of articles in the input
    reorder();

    // apply phda9 preprocessor
    phda9_prepr();

    // merge all input parts after preprocessing
    cat(".main_phda9prepr", ".intro", "un1");
    cat("un1", ".coda", ".ready4cmix");

    // run compression
    input_path = ".ready4cmix";
    dictionary = fopen(".dict", "rb");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes,
        nullptr
        )) {
      return Help();
    }

    cat(".decomp_bin", ".dict.comp", "dec2");
    cat("dec2", "weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4", "dec1");

    // get the size of the output file
    size_t output_size = getFileSize(output_path);

    HeaderInfo header;
    read("test.dat", header);
    header.decomp_input_size = output_size;
    header.dict_size = (int)getFileSize(".dict.comp");
    // All four fields explicitly, none inherited from comp9's own header.
    header.new_article_order_size = 0;
    header.weights_size = (int)getFileSize("weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4");
    write("header4archive.dat", header);

    cat("dec1", output_path.c_str(), "dec2");
    cat("dec2", "header4archive.dat", "archive9");

    // make the decompressor binary executable
    char mode[] = "0777";
    char buf[100] = "archive9";
    int i = strtol(mode, 0, 8);
    chmod(buf, i);

    selfx_clean::AfterCompress();

  } else if (argv[1][1] == 'h') {
    if (argc < 6) return Help();
    HeaderInfo header;
    header.dict_size = atoi(argv[2]);
    header.new_article_order_size = atoi(argv[3]);
    header.decomp_input_size = atoi(argv[4]);
    header.weights_size = atoi(argv[5]);
    write("header.dat", header);
    goto exit;
  }
  else
  {
    if (argc == 5) {
      if (!provide_dict_file(argv[2])) return Help();
    }
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  }

print_end_message:
#if !AUX_BUILD
  {
    const double progress_s = console_progress::End(progress_encoding);
    printf("%lld bytes -> %lld bytes in %1.2f s.\n", input_bytes, output_bytes,
           progress_s);
    (void)start;
  }
#else
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      input_bytes, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);
#endif

exit:
  return 0;
}
