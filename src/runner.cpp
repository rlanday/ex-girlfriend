#include "meld.h"
#include <string>
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
#include "readalike_prepr/komm_transform.h"
#include "r1_reorder_transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "isa_floor.h"
#include "page_contexts.h"
#include "word_recency.h"
#include "simd_scan.h"
#include "invariants.h"
#include "huge_pages.h"
#include "mixer_lr.h"
#include "mixer_decay.h"
#include "fx_mixer_lr.h"
#include "mixer_bank.h"
#include "logit_table.h"
#include "mixer_weights.h"
#include "vocab_frozen.h"
#include "run_mode.h"
#include "port_cfg.h"
#if PORT_CMX_MEAS
#include <math.h>
#include <unistd.h>
#include "cmx_meas.h"   // research measurement knobs (src/port_cfg.h)
#endif

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
#include "census_block.h"
#include "computed_values.h"
static bool census_block_active = false;

#include "ip_pack.h"
#include "isbn_check.h"
#include "metablock_form.h"
#include "text_start_pack.h"
#include "rule_stages_file.h"
#include "ts_progress.h"


#include "no_args.h"


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

#if !(1 && !AUX_BUILD)
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
#endif

int Help() {
  msg_note(256);
  return -1;
}

size_t getFileSize(const char* path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    msg_note(170);
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
#include "vocab_frozen.h"
#include "console_progress.h"

#include "selfx_clean.h"

void WriteHeaderVocfix(unsigned long long length, bool dictionary_used,
    FILE* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x7F;
      if (dictionary_used) c |= 0x80;
      if (vocab_frozen::g_aux) c |= 0x40;
    }
    putc(c, os);
  }
}

void ReadHeaderVocfix(FILE* is, unsigned long long* length,
    bool* dictionary_used, std::vector<bool>* vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = getc(is);
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

// The known bytes b[0..n) of the model side, coder and decoder alike: every
// model observes their bits, none is coded.  One function for both
// directions, so the hooks below need no coder object.
__attribute__((noinline)) static void ObserveKnownBytes(Predictor* p, const unsigned char* b, int n) {
  invariants::g_injected_byte = 1;
  for (int k = 0; k < n; ++k)
    for (int bit = 7; bit >= 0; --bit) p->ObserveKnownBit((b[k] >> bit) & 1);
  invariants::g_injected_byte = 0;
}

__attribute__((noinline)) static void CensusBlockByte(Predictor* p, unsigned char c) {
  if (MetablockInert()) return;
  if (metablock::mk.TaByte()) {
    // slot 3 is the pseudo article: at its end mark the automaton takes it
    if (metablock::mk.TaSlot() == 4 && metablock::mk.TaSlotEndet()) {
      size_t n = 0;
      const unsigned char* q = metablock::mk.TaPuffer(4, &n);
      census_block::aut.Slot3(q, n);
    }
    return;
  }
  census_block::aut.Observe(c);
  const unsigned char* zb = nullptr;
  const int zn = census_block::aut.TakeBlock(&zb);
  if (!zn) return;
  // of the census injections the transformer sees only the computed text (TfSieht)
  const bool tf = census_block::aut.TfSieht();
  if (!tf) timestamp_skip::Beginn();
  if (tf) tf_author_prompt::Beginn(zn);
  {
    const int kn = census_block::aut.EndeNach(zn);
    tf_copy_end::g_nach = tf ? kn : 0;
  }
  ObserveKnownBytes(p, zb, zn);
  tf_copy_end::g_nach = 0;
  if (tf) tf_author_prompt::Ende();
  if (!tf) timestamp_skip::Ende();
  (void)tf;
}

__attribute__((noinline)) static void TextStartHead(Predictor* p, int slot) {
  static const char kKopf2[] = "neif\tpre authors\n";
  static const char kKopf3[] = "neif\tpre census\n";
  static const char kKopf1[] = "neif\tpre dictionary\n";
  const char* k = slot == 2 ? kKopf2 : (slot == 4 ? kKopf3 : (slot == 1 ? kKopf1 : ""));
  const int n = (int)strlen(k);
  if (!n) return;
  tf_author_prompt::Beginn(n);
  ObserveKnownBytes(p, (const unsigned char*)k, n);
  tf_author_prompt::Ende();
}

static_assert(54857 == 54857, "must equal DICT_WORDS");
namespace fxcmv1 { int DictNachladen(); }
__attribute__((noinline)) static void DictArticleSlotEnd() {
  size_t n = 0;
  const unsigned char* b = metablock::mk.TaPuffer(1, &n);
  dict_article::Schreiben(b, n);
  const int w = fxcmv1::DictNachladen();
  if (dict_article::wrt) {
    run_mode::gram.Init(NULL, dict_article::strom, 0);
    run_mode::gram.ZoneSperren();
  }
  computed_values::g_dict_da = true;
  (void)w;
}

namespace dict_runs {
constexpr bool kRegel = 1 == 1;
// fixed bounds, checked against ./.dict by the coder (E551)
constexpr unsigned kTxt = dict_codes::kLaufTxt, kWort = dict_codes::kLaufWort;
struct RunState {                      // static (.bss), lives for the run
  size_t gez;                         // bytes of b followed
  unsigned nt, nw, nc;                // letters in txt, words finished, letters of the current word
  int np;                             // bytes of an open code word
  bool hinter;                        // a run mark in front of the current word
  unsigned char pend[3];              // the open code word
  unsigned char cur[kWort];           // the current word decoded so far
  unsigned ab[54857 + 1];       // start of word r in txt
  unsigned char txt[kTxt];            // the decoded words back to back
};
static RunState z;
static const unsigned char* Text(unsigned r, size_t* l) {
  *l = (r + 1 < z.nw ? z.ab[r + 1] : z.nt) - z.ab[r];
  return z.txt + z.ab[r];
}
// the rank intervals [lo[q], hi[q]] of the code words that begin with pre[0..k)
static int Intervalle(const unsigned char* pre, int k, unsigned* lo, unsigned* hi) {
  const unsigned b0 = pre[0];
  if (k == 1) {
    if (b0 <= 0xCF) { lo[0] = hi[0] = b0 - 0x80; return 1; }
    lo[0] = 80 + (b0 - 0xD0) * 80; hi[0] = lo[0] + 79;
    if (b0 < 0xF0) return 1;
    lo[1] = 3920 + (b0 - 0xF0) * 2560; hi[1] = lo[1] + 2559;
    lo[2] = 44880 + (b0 - 0xF0) * 1280; hi[2] = lo[2] + 1279;
    return 3;
  }
  const unsigned b1 = pre[1];
  if (b1 <= 0xCF) { lo[0] = hi[0] = 80 + (b0 - 0xD0) * 80 + (b1 - 0x80); return 1; }
  lo[0] = b1 <= 0xEF ? 3920 + ((b0 - 0xF0) * 32 + (b1 - 0xD0)) * 80
                     : 44880 + ((b0 - 0xF0) * 16 + (b1 - 0xF0)) * 80;
  if (k == 2) { hi[0] = lo[0] + 79; return 1; }
  lo[0] += pre[2] - 0x80; hi[0] = lo[0];
  return 1;
}
static void Add(const unsigned char* t, size_t l) {   // to the current word
  if (l > kWort - z.nc) l = kWort - z.nc;
  memcpy(z.cur + z.nc, t, l);
  z.nc += (unsigned)l;
}
static void Folge(unsigned char c) {  // the state behind byte c
  if (c == 0x0B) return;
  if (z.np || c >= 0x80) {
    z.pend[z.np++] = c;
    const int L = z.pend[0] <= 0xCF ? 1 : z.np < 2 ? 0 : z.pend[1] <= 0xCF ? 2 : 3;
    if (L && z.np == L) {
      size_t k = 0;
      const long r = dict_codes::Rang(z.pend, (size_t)L, &k);
      if (r >= 0 && (size_t)r < z.nw) {
        size_t l = 0;
        const unsigned char* t = Text((unsigned)r, &l);
        Add(t, l);
      }
      z.np = 0;
    }
    return;
  }
  if (c == 0x0A) {
    if (!z.nc) {
      z.hinter = true;
    } else if (z.nw <= 54857) {   // the current word is finished
      z.ab[z.nw++] = z.nt;
      const unsigned lc = z.nc < kTxt - z.nt ? z.nc : kTxt - z.nt;
      memcpy(z.txt + z.nt, z.cur, lc);
      z.nt += lc;
      z.nc = 0;
      z.hinter = false;
    }
    return;
  }
  Add(&c, 1);
}
// some code word beginning with pre[0..k) names a rank < i; with genau (exact)
// (inside an open code word, k >= 2: at most 80 ranks) one of them names a
// word T that keeps the order, S[0..ls) the rest of the previous word (equal
// so far): behind a mark T < S, in a run T >= S or T a proper prefix of S
static bool CodeOk(const unsigned char* pre, int k, unsigned long long i, bool genau,
                   const unsigned char* S, size_t ls) {
  unsigned lo[3], hi[3];
  const int n = Intervalle(pre, k, lo, hi);
  for (int q = 0; q < n; ++q) {
    if (lo[q] >= i) continue;
    if (!genau) return true;
    const unsigned e = hi[q] < i - 1 ? hi[q] : (unsigned)(i - 1);
    for (unsigned r = lo[q]; r <= e; ++r) {
      size_t lt = 0;
      const unsigned char* t = Text(r, &lt);
      const int c = memcmp(t, S, lt < ls ? lt : ls);
      if (z.hinter ? (c < 0 || (!c && lt < ls)) : c >= 0) return true;
    }
  }
  return false;
}
__attribute__((noinline)) static int Menge(const unsigned char* b, size_t m, unsigned char* v, int* art) {
  if (m < z.gez) z.gez = z.nt = z.nw = z.nc = 0, z.np = 0, z.hinter = false;   // a new slot 1
  for (; z.gez < m; ++z.gez) Folge(b[z.gez]);
  int n = 0;
  *art = 3;
  if (m && b[m - 1] == 0x0B) {
    v[n++] = 0x0B;
    return n;
  }
  const unsigned long long i = z.nw;             // index of the current word
  const bool regel = kRegel && i > dict_codes::kLaufAb;
  size_t lp = 0;
  const unsigned char* p = i ? Text((unsigned)(i - 1), &lp) : nullptr;
  const size_t k = z.nc;
  const bool eq = regel && k <= lp && (!k || !memcmp(z.cur, p, k));
  int lo = 0x61, hi = 0x7A, clo = 0x80, chi = 0xFF;
  bool lf = false, ende = false, genau = false;
  if (z.np) {
    lo = 1, hi = 0;                              // inside a code word: no letter
    chi = (z.np == 1 && z.pend[0] >= 0xF0) ? 0xFF : 0xCF;
    if (eq) genau = true, *art = 4;
  } else if (!m) {
  } else if (!k && b[m - 1] == 0x0A) {           // the start of a word
    if (z.hinter) {
      if (regel) hi = lp > 1 ? p[0] : p[0] - 1, *art = 4;
    } else {
      ende = true;
      lf = i > dict_codes::kLaufAb;           // a run mark
      if (regel) lo = p[0], *art = 4;
    }
  } else {
    // inside a word: no code word (the prefix rule); the line feed only
    // behind a letter
    clo = 1, chi = 0;
    lf = b[m - 1] < 0x80;
    if (eq && !z.hinter) {
      lf = false, *art = 4;
      if (k < lp) lo = p[k];
    } else if (eq && k < lp) {
      hi = k + 1 < lp ? p[k] : p[k] - 1, *art = 4;
    }
  }
  if (lf) v[n++] = 0x0A;
  if (ende) v[n++] = 0x0B;
  for (int c = lo; c <= hi; ++c) v[n++] = (unsigned char)c;
  unsigned char pre[3] = {z.pend[0], z.pend[1], 0};
  for (int c = clo; c <= chi; ++c) {
    pre[z.np] = (unsigned char)c;
    if (kRegel && !CodeOk(pre, z.np + 1, i, genau, p + k, lp - k)) continue;
    v[n++] = (unsigned char)c;
  }
  return n;
}
}  // namespace dict_runs
__attribute__((noinline)) static bool DictArticleSep() {
  unsigned char v[156];
  int n = 0;
  if (metablock::mk.TaNaechsterSlot() == 1) {
    n = dict_runs::Menge(nullptr, 0, v, &metablock::zw::art);
  } else if (metablock::mk.TaByte() && metablock::mk.TaSlot() == 1 &&
             !metablock::mk.TaSlotEndet()) {
    size_t m = 0;
    const unsigned char* b = metablock::mk.TaPuffer(1, &m);
    n = dict_runs::Menge(b, m, v, &metablock::zw::art);
  } else {
    return false;
  }
  return metablock::zw::Setze(v, n);
}

__attribute__((noinline)) static void TextStartByte(Predictor* p) {
  if (MetablockInert()) return;
  metablock::zw::an = false;
  if (!metablock::mk.Endet() && !metablock::mk.TaByte()) return;
  {
    bool jetzt = metablock::mk.TaSlotEndet() && metablock::mk.TaSlot() == 1;
    if (jetzt) DictArticleSlotEnd();
    if (jetzt) p->PpmdWaermen();
  }
  if (metablock::mk.TaSlotEndet() && metablock::mk.TaSlot() == 2) {
    p->WordlblBlockEnde();
  }
  const int s = metablock::mk.TaNaechsterSlot();
  if (s) TextStartHead(p, s);
  const int j = metablock::mk.TaZeileFolgt();
  if (j >= 0) {
    const unsigned char* q = nullptr;
    const int n = metablock::mk.TaOkb(j, &q);
    // the transformer gets the context as well - an author
    // prompt block like the slot head (fed, not predicted, masked in the dump)
    if (n) tf_author_prompt::Beginn(n);
    ObserveKnownBytes(p, q, n);
    if (n) tf_author_prompt::Ende();
  }
  // stage 2: a byte of a repetition line follows - its known bits
  metablock::zw::an = metablock::mk.TaZwSetze();
  if (!metablock::zw::an) metablock::zw::an = DictArticleSep();   // slot 1
  if (metablock::mk.TaFertig()) {
    // the transform grammar holds from the first byte of page 0 on (the slots
    // carry text outside it: the dictionary in plain text, the census texts)
    run_mode::gram.ZoneOeffnen(metablock::mk.Pos());
    p->MetalrBlockEnde();
  }
  (void)p;
}

ENC_ONLY static void TextStartSepEnc(Encoder& e, unsigned int c, unsigned long long pos) {
  unsigned int ctx = 1;
  const int a = metablock::zw::art;
  for (int j = 7; j >= 0; --j) {
    const int bit = (int)((c >> j) & 1u);
    const int f = metablock::zw::line_buf[ctx & 255u];
    if (f < 0) {
      e.Encode(bit);
    } else {
      if (f != bit) {
        msg_stop(542);
        exit(3);
      }
      e.ObserveKnownBit(bit);
    }
    ctx += ctx + (unsigned int)bit;
  }
}
__attribute__((noinline)) static int TextStartSepDec(Decoder& d) {
  int byte = 1;
  while (byte < 256) {
    const int f = metablock::zw::line_buf[byte];
    int bit;
    if (f < 0) {
      bit = d.Decode();
    } else {
      bit = f;
      d.ObserveKnownBit(bit);
    }
    byte += byte + bit;
  }
  return byte;
}

// After every coded (Compress) or decoded (Decompress) stream byte c: the
// units and automata that read the stream, and the bytes they inject.  ONE
// function for both directions, so coder and decoder run the same code in
// the same order; noinline keeps it out of the two byte loops.
__attribute__((noinline)) static void AfterStreamByte(Predictor* p, unsigned char c) {
  if (!MetablockInert()) {
    metablock::mk.Byte(c);
    if (metablock::mk.MetaEndet()) {
      ppmd_meta::block_fertig = true;
    }
    if (metablock::mk.Endet()) {
      p->WordlblBlockEnde();
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
    for (int k = 0; k < n; ++k) p->ObserveTfOnlyByte(blk[k]);
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
    ObserveKnownBytes(p, ab, an);
    tf_author_prompt::Ende();
  }
  CensusBlockByte(p, c);
  if (census_block::aut.ZeileEndet() && !metablock::mk.TaByte()) {
    p->MetalrBlockEnde();
    ppmd_meta::zeile_fertig = true;
  }
  if (!TimestampProgressInert()) {
    ts_progress::aut.Observe(c);
    const unsigned char* tb = nullptr;
    const int tn = ts_progress::aut.TakeBlock(&tb);
    if (tn) timestamp_skip::Beginn();
    ObserveKnownBytes(p, tb, tn);
    timestamp_skip::Ende();
  }
  TextStartByte(p);
  (void)p; (void)c;
}

#if PORT_CMX_MEAS
// PORT_CMX_MEAS (src/port_cfg.h): the encoder-side observers of a research run (Claude Opus 5.5, 2026-10-31).
// They read the coder state after a stream byte is fully coded and all its readers (AfterStreamByte) have run;
// nothing here writes a model or the coder, except the coder flush at the stop position.
namespace cmx_meas {
// the exact code length so far in bits: 8 per byte the coder has written plus the bits the open interval
// [x1, x2] has used (the arithmetic coder narrows it by exactly the coded probability of each bit)
static double CodeBits(Encoder& e) {
  return 8.0 * (double)e.OutputSize() + 32.0 - log2((double)(e.CmxX2() - e.CmxX1()) + 1.0);
}
static std::vector<float>& ByteCost() {
  static std::vector<float> v;
  return v;
}
static double g_prev_bits = 0.0;
static bool g_text = false;          // text_start seen: the page count of the order runs
static bool g_first_article = false;
static int g_at_i = 0, g_pages_i = 0;
static void Line(const char* tag, const char* name, Encoder& e, unsigned long long n) {
  fprintf(stderr, "%s%s%s %llu %zu %.3f %.1f\n", tag, name[0] ? " " : "", name, n, e.OutputSize(), CodeBits(e),
          console_progress::Now() - console_progress::t_start);
  fflush(stderr);
}
static void WriteByteCost() {
  if (!g_bytecost || ByteCost().empty()) return;
  FILE* bf = fopen(g_bytecost_path ? g_bytecost_path : "bytecost.f32", "wb");
  if (bf) {
    fwrite(ByteCost().data(), sizeof(float), ByteCost().size(), bf);
    fclose(bf);
  }
  fprintf(stderr, "CMX bytecost.f32 %zu values\n", ByteCost().size());
}
// research CMX_CM_TICKS (cmx_meas.h): per region one line of the pipeline sections ("name=cycles/laps"; E2E without
// the CMX_VALTEL work inside it, which is VALTEL), then with level 2 one line of the fxcm sites and the mxA step count
static void CkReport(const char* tag) {
  const double hz = CkTscHz();
  for (int r = 0; r < 4; ++r) {
    if (!g_ckn[r][CK_LOOP] && !g_ckn[r][CK_FXUPD]) continue;
    fprintf(stderr, "CMXTEL cmticks %s region=%d tsc_hz=%.0f ovh=%llu", tag, r, hz, g_ck_ovh);
    for (int s = 0; s < CK_N; ++s) {
      if (!g_ckn[r][s]) continue;
      unsigned long long c = g_ck[r][s];
      if (s == CK_E2E) c = c > g_ck[r][CK_VALTEL] ? c - g_ck[r][CK_VALTEL] : 0ull;
      fprintf(stderr, " %s=%llu/%llu", kCkName[s], c, g_ckn[r][s]);
    }
    fprintf(stderr, "\n");
    if (g_cm_ticks < 2) continue;
    fprintf(stderr, "CMXTEL fxticks %s region=%d mxA_steps=%llu/%llu", tag, r, g_fx_trn[r][1], g_fx_trn[r][0]);
    for (int s = 0; s < kFxN; ++s)
      if (g_fxkn[r][s] && g_fxname[s]) fprintf(stderr, " %s=%llu/%llu", g_fxname[s], g_fxk[r][s], g_fxkn[r][s]);
    fprintf(stderr, "\n");
  }
}
// the layout of fxcm's sites as last recorded: "name=pred_lo-pred_hi:int_lo-int_hi" (layer-0 index = pred + 1)
static void FxLayoutReport(const char* tag) {
  fprintf(stderr, "CMXTEL fxlayout %s", tag);
  for (int s = 0; s < kFxN; ++s)
    if (g_fxname[s])
      fprintf(stderr, " %s=%u-%u:%u-%u", g_fxname[s], g_fxlay[s][0], g_fxlay[s][1], g_fxlay[s][2], g_fxlay[s][3]);
  fprintf(stderr, "\n");
}
// the zero-byte telemetry lines (CMX_MIXTEL, CMX_SEMTEL, CMX_MIXSKIP counters)
static void TelReport(Predictor* p, const char* tag) {
  if (g_mixtel) p->CmxMixReport(tag);
  if (g_semtel) p->CmxSemReport(tag);
  if (g_fw_ticks)
    fprintf(stderr, "CMXTEL fwticks %s first_block=%d apply_cycles=%llu head_cycles=%llu update_cycles=%llu updates=%llu\n",
            tag, g_fw_first, g_fw_cyc_apply, g_fw_cyc_head, g_fw_cyc_update, g_fw_n_update);
  if (g_fx_ticks || g_fx_lazy)
    fprintf(stderr, "CMXTEL fxreset %s lazy=%d page_resets=%llu reset_block_cycles=%llu lazy_bucket_zeroings=%llu\n", tag,
            g_fx_lazy, g_fx_resets, g_fx_reset_cyc, g_fx_lazy_zero);
  if (g_skip_on)
    for (int r = 0; r < 4; ++r)
      for (int ly = 0; ly < 2; ++ly)
        if (g_upd[r][ly])
          fprintf(stderr, "CMXTEL mixskip %s region=%d layer=%d updates=%llu skipped=%llu frac=%.4f\n", tag, r, ly,
                  g_upd[r][ly], g_skp[r][ly], (double)g_skp[r][ly] / (double)g_upd[r][ly]);
  if (g_adm)   // research CMX_ADEPTH_MODE: gate bits (|z| > tau) among the non-forced bits
    for (int r = 0; r < 4; ++r)
      if (g_adm_bits[r])
        fprintf(stderr, "CMXTEL adepth %s mode=%d region=%d bits=%llu gate=%llu frac=%.4f\n", tag, g_adm, r,
                g_adm_bits[r], g_adm_gate[r], (double)g_adm_gate[r] / (double)g_adm_bits[r]);
  if (!g_adm && g_tauc_set)   // research CMX_ADEPTH_TAU_CODE: learn-skip and code-skip bits among the non-forced bits
    for (int r = 0; r < 4; ++r)
      if (g_adm_bits[r])
        fprintf(stderr, "CMXTEL adepth_split %s tau_code=%.9g tau_learn=%.9g region=%d bits=%llu learn_skip=%llu "
                "code_skip=%llu learn_frac=%.4f code_frac=%.4f\n", tag, (double)g_tauc,
                (double)(g_tau_set ? g_tau : adaptive_depth::kTau), r, g_adm_bits[r], g_adm_gate[r], g_adm_cgate[r],
                (double)g_adm_gate[r] / (double)g_adm_bits[r], (double)g_adm_cgate[r] / (double)g_adm_bits[r]);
  if (g_semin) p->CmxSemInReport(tag);   // research CMX_SEMIN: the extra weights
  if (g_nsk) p->CmxSem2Report(tag);   // research CMX_SEMKEYS / CMX_SEMKEY2: the extra caches' votes
  if (g_semtop)   // research CMX_SEMTOP: bytes with neighbours by class 1..L
    for (int r = 0; r < 4; ++r) {
      unsigned long long n = 0;
      for (int c = 0; c < 13; ++c) n += g_st_cnt[r][c];
      if (!n) continue;
      fprintf(stderr, "CMXTEL semtop %s mode=%d region=%d bytes=%llu class_frac:", tag, g_semtop, r, n);
      for (int c = 1; c <= g_st_lv; ++c) fprintf(stderr, " %.4f", (double)g_st_cnt[r][c] / (double)n);
      fprintf(stderr, "\n");
    }
  if (g_fxls)   // research CMX_FXLS: fxcm mixer learning steps taken / skipped (gate bits)
    for (int r = 0; r < 4; ++r)
      if (g_fxls_n[r][0] + g_fxls_n[r][1])
        fprintf(stderr, "CMXTEL fxls %s region=%d steps=%llu skipped=%llu frac=%.4f\n", tag, r, g_fxls_n[r][0],
                g_fxls_n[r][1], (double)g_fxls_n[r][1] / (double)(g_fxls_n[r][0] + g_fxls_n[r][1]));
  if (g_cm_ticks) CkReport(tag);   // research CMX_CM_TICKS: cycles per section (and fxcm site)
  if (g_fxrec) FxLayoutReport(tag);   // research CMX_CM_TICKS=2 / CMX_VALTEL: fxcm's input layout
  if (g_valtel) p->CmxValReport(tag);   // research CMX_VALTEL: frozen-weight occlusion per group
  fflush(stderr);
}
static bool g_rekeyed = false;
// ---- CMX_SEMDUMP (zero-byte observer): records, page index, meta ----------------------------------------------
static FILE* g_sd_rec = nullptr;
static FILE* g_sd_pg = nullptr;
static long long g_sd_n = 0, g_sd_valid[4] = {0, 0, 0, 0}, g_sd_cnt[4] = {0, 0, 0, 0};
static long long g_sd_last_page = -2;
static long long g_mk_census_end = -1, g_mk_text_start = -1, g_mk_first_article = -1;
static int SdRegion() { return g_first_article ? 3 : (g_text ? 2 : (g_mk_census_end >= 0 ? 1 : 0)); }
// before coder byte pos is coded (SemByteStart has just filled g_semrec for it)
__attribute__((noinline)) static void SemDumpByte(unsigned long long pos) {
  if ((long long)pos < g_sd_from || (long long)pos >= g_sd_to) return;
  if (!g_sd_rec) {
    char path[4096];
    snprintf(path, sizeof path, "%s/semcache_records.bin", g_semdump);
    g_sd_rec = fopen(path, "wb");
    snprintf(path, sizeof path, "%s/pages.u32", g_semdump);
    g_sd_pg = fopen(path, "wb");
    if (!g_sd_rec || !g_sd_pg) { fprintf(stderr, "CMX semdump: cannot open %s\n", g_semdump); _exit(9); }
    static char b1[1 << 22], b2[1 << 16];
    setvbuf(g_sd_rec, b1, _IOFBF, sizeof b1);
    setvbuf(g_sd_pg, b2, _IOFBF, sizeof b2);
  }
  fwrite(&g_semrec, sizeof g_semrec, 1, g_sd_rec);
  const int rg = SdRegion();
  ++g_sd_cnt[rg];
  if (g_semrec.flags & 1u) ++g_sd_valid[rg];
  if (g_text && !AuthorBlockInert()) {
    const long long sp = author_block::aut.Seite();
    if (sp != g_sd_last_page) {
      const uint32_t w[2] = {(uint32_t)sp, (uint32_t)g_sd_n};
      fwrite(w, sizeof w, 1, g_sd_pg);
      g_sd_last_page = sp;
    }
  }
  ++g_sd_n;
}
// ---- CMX_MATCHDUMP: the record of byte pos is filled during its first bit and written before the next byte -------
static FILE* g_md_rec = nullptr;
static FILE* g_md_pg = nullptr;
static long long g_md_n = 0, g_md_pend_pos = -1, g_md_pend_page = -1, g_md_last_page = -2;
static int g_md_pend_rg = 0;
static long long g_md_cnt[4] = {0, 0, 0, 0}, g_md_has[4] = {0, 0, 0, 0};
static void MatchDumpFlush() {   // the pending record (its byte is fully coded now)
  if (g_md_pend_pos < 0) return;
  fwrite(&g_mdrec, sizeof g_mdrec, 1, g_md_rec);
  ++g_md_cnt[g_md_pend_rg];
  if (g_mdrec.flags & 3u) ++g_md_has[g_md_pend_rg];
  if (g_md_pend_page >= 0 && g_md_pend_page != g_md_last_page) {
    const uint32_t w[2] = {(uint32_t)g_md_pend_page, (uint32_t)g_md_n};
    fwrite(w, sizeof w, 1, g_md_pg);
    g_md_last_page = g_md_pend_page;
  }
  ++g_md_n;
  g_md_pend_pos = -1;
}
// before coder byte pos is coded: write the previous record, arm the capture for this byte
__attribute__((noinline)) static void MatchDumpArm(unsigned long long pos) {
  if (g_md_rec) MatchDumpFlush();
  if ((long long)pos < g_sd_from || (long long)pos >= g_sd_to) return;
  if (!g_md_rec) {
    char path[4096];
    snprintf(path, sizeof path, "%s/match_records.bin", g_matchdump);
    g_md_rec = fopen(path, "wb");
    snprintf(path, sizeof path, "%s/pages.u32", g_matchdump);
    g_md_pg = fopen(path, "wb");
    if (!g_md_rec || !g_md_pg) { fprintf(stderr, "CMX matchdump: cannot open %s\n", g_matchdump); _exit(9); }
    static char b1[1 << 22], b2[1 << 16];
    setvbuf(g_md_rec, b1, _IOFBF, sizeof b1);
    setvbuf(g_md_pg, b2, _IOFBF, sizeof b2);
  }
  memset(&g_mdrec, 0, sizeof g_mdrec);
  g_mdrec.cm_sym = g_mdrec.fx_sym = 255u;
  g_md_pend_pos = (long long)pos;
  g_md_pend_rg = SdRegion();
  g_md_pend_page = (g_text && !AuthorBlockInert()) ? author_block::aut.Seite() : -1;
  g_md_arm = true;
}
static void MatchDumpClose() {
  if (!g_md_rec) return;
  MatchDumpFlush();
  fclose(g_md_rec);
  fclose(g_md_pg);
  g_md_rec = g_md_pg = nullptr;
  char path[4096];
  snprintf(path, sizeof path, "%s/meta.json", g_matchdump);
  FILE* f = fopen(path, "w");
  if (!f) return;
  const auto rel = [](long long m) { return m >= 0 ? m - g_sd_from : -1LL; };
  fprintf(f, "{\n \"format\": \"match_records.bin: n_records x 8 B packed little-endian, record i = coder position from + i,"
             " captured at the byte's first bit before it is coded: u8 flags (bit0 cmix match length > 0, bit1 fxcm "
             "MatchModel2 candidate length > 0, bit2 both present with the same symbol, bit3 fxcm delta mode, bits 4..6 "
             "index of the longest of cmix's 5 match models: orders 0,1,7,11,13), u8 src (stronger source: 0 none, 1 "
             "cmix, 2 fxcm), u8 cm_sym (token index 0..207 = transformer::g_b2i[byte], 255 none), u8 cm_len (verified "
             "bits, 0..255), u8 fx_sym, u8 fx_len (fxcm rebased length, 1 = shortest accepted match, capped 255), u16 "
             "cm_conf (cmix P(first bit follows the prediction) x 65535); pages.u32: (u32 page, u32 first record) from "
             "the text start on\",\n");
  fprintf(f, " \"from\": %lld,\n \"to\": %lld,\n \"n_records\": %lld,\n \"record_bytes\": 8,\n", g_sd_from, g_sd_to, g_md_n);
  // boundary names (driver 11-01, Codex erratum): every boundary is the BEGIN of the region it names; the program's
  // internal "text_start" mark (g_mk_text_start) is the redirect-block begin, not the begin of the text-start region
  fprintf(f, " \"region_begin_pos\": {\"census_line_end\": %lld, \"text_start_begin\": %lld, \"redirect_begin\": %lld, "
             "\"article_begin\": %lld},\n", g_mk_census_end, g_mk_census_end, g_mk_text_start, g_mk_first_article);
  fprintf(f, " \"region_begin_rec\": {\"census_line_end\": %lld, \"text_start_begin\": %lld, \"redirect_begin\": %lld, "
             "\"article_begin\": %lld},\n", rel(g_mk_census_end), rel(g_mk_census_end), rel(g_mk_text_start),
          rel(g_mk_first_article));
  fprintf(f, " \"region_begin_note\": \"half-open regions in coder positions (_pos) and record indices (_rec): census "
             "line [.., census_line_end), text start [text_start_begin, redirect_begin), redirect block [redirect_begin, "
             "article_begin), articles [article_begin, to); census_line_end == text_start_begin\",\n");
  fprintf(f, " \"regions\": {\"0_census_line\": [%lld, %lld], \"1_text_start\": [%lld, %lld], \"2_redirect\": [%lld, %lld], "
             "\"3_article\": [%lld, %lld]},\n \"region_note\": \"[records, records with a match]\"\n}\n",
          g_md_cnt[0], g_md_has[0], g_md_cnt[1], g_md_has[1], g_md_cnt[2], g_md_has[2], g_md_cnt[3], g_md_has[3]);
  fclose(f);
  fprintf(stderr, "CMX matchdump %s: %lld records, with a match per region: %lld/%lld %lld/%lld %lld/%lld %lld/%lld\n",
          g_matchdump, g_md_n, g_md_has[0], g_md_cnt[0], g_md_has[1], g_md_cnt[1], g_md_has[2], g_md_cnt[2], g_md_has[3],
          g_md_cnt[3]);
}
static void SemDumpClose() {
  if (!g_sd_rec) return;
  fclose(g_sd_rec);
  fclose(g_sd_pg);
  g_sd_rec = g_sd_pg = nullptr;
  char path[4096];
  snprintf(path, sizeof path, "%s/meta.json", g_semdump);
  FILE* f = fopen(path, "w");
  if (!f) return;
  const auto rel = [](long long m) { return m >= 0 ? m - g_sd_from : -1LL; };
  fprintf(f, "{\n \"format\": \"semcache_records.bin: n_records x 58 B packed little-endian, record i = coder position "
             "from + i: u8 flags (bit0 valid >= 1 neighbour, bit1 lookup ran, bit2 cache top-1 == transformer top-1, bit3 "
             "stored), u8 nfresh, u8 nnb, u8 nsym, i16 sim (best cosine x 32767), u16 hrate (running top-1 hit rate x "
             "65535), u16 resid (mass outside the 16 symbols x 65535), u8 sym[16] (transformer token index 0..207 = "
             "transformer::g_b2i[byte], heaviest first), u16 prob[16] (share x 65535); pages.u32: pairs (u32 page index "
             "of the order as author_block counts it, u32 first record index) from the text start on\",\n");
  fprintf(f, " \"from\": %lld,\n \"to\": %lld,\n \"n_records\": %lld,\n \"record_bytes\": 58,\n", g_sd_from, g_sd_to,
          g_sd_n);
  // boundary names (driver 11-01, Codex erratum): every boundary is the BEGIN of the region it names; the program's
  // internal "text_start" mark (g_mk_text_start) is the redirect-block begin, not the begin of the text-start region
  fprintf(f, " \"region_begin_pos\": {\"census_line_end\": %lld, \"text_start_begin\": %lld, \"redirect_begin\": %lld, "
             "\"article_begin\": %lld},\n", g_mk_census_end, g_mk_census_end, g_mk_text_start, g_mk_first_article);
  fprintf(f, " \"region_begin_rec\": {\"census_line_end\": %lld, \"text_start_begin\": %lld, \"redirect_begin\": %lld, "
             "\"article_begin\": %lld},\n", rel(g_mk_census_end), rel(g_mk_census_end), rel(g_mk_text_start),
          rel(g_mk_first_article));
  fprintf(f, " \"region_begin_note\": \"half-open regions in coder positions (_pos) and record indices (_rec): census "
             "line [.., census_line_end), text start [text_start_begin, redirect_begin), redirect block [redirect_begin, "
             "article_begin), articles [article_begin, to); census_line_end == text_start_begin\",\n");
  fprintf(f, " \"regions\": {\"0_census_line\": [%lld, %lld], \"1_text_start\": [%lld, %lld], \"2_redirect\": [%lld, %lld], "
             "\"3_article\": [%lld, %lld]},\n \"region_note\": \"[records, valid records]\"\n}\n",
          g_sd_cnt[0], g_sd_valid[0], g_sd_cnt[1], g_sd_valid[1], g_sd_cnt[2], g_sd_valid[2], g_sd_cnt[3], g_sd_valid[3]);
  fclose(f);
  fprintf(stderr, "CMX semdump %s: %lld records, valid per region (census line, text start, redirect, article): "
          "%lld/%lld %lld/%lld %lld/%lld %lld/%lld\n", g_semdump, g_sd_n, g_sd_valid[0], g_sd_cnt[0], g_sd_valid[1],
          g_sd_cnt[1], g_sd_valid[2], g_sd_cnt[2], g_sd_valid[3], g_sd_cnt[3]);
}
// after coder byte n (n = bytes coded so far)
__attribute__((noinline)) static void AfterByte(Encoder& e, FILE* os, unsigned long long n, Predictor* p) {
  if (g_bytecost) {
    const double b = CodeBits(e);
    ByteCost().push_back((float)(b - g_prev_bits));
    g_prev_bits = b;
  }
  if (!MetablockInert()) {
    // region boundaries of the stream: CMXMARK <name> <coder bytes> <coded bytes> <code bits> <wall s>
    if (metablock::mk.MetaEndet()) {
      Line("CMXMARK", "meta_end", e, n);
      if (g_z3_mul != 1.0f) g_z3_on = true;   // CMX_Z3LR: from the first byte of the census line to its end
    }
    if (metablock::mk.Endet()) {
      Line("CMXMARK", "census_end", e, n);
      g_mk_census_end = (long long)n;
      if (g_wl_text) g_wl_live = true;   // research CMX_WL_TEXT: the kappa term from text_start_begin on
    }
    if (metablock::mk.TaSlotEndet()) {
      char nm[32];
      snprintf(nm, sizeof nm, "slot%d_end", metablock::mk.TaSlot());
      Line("CMXMARK", nm, e, n);
    }
    if (metablock::mk.TaFertig()) {   // the next coder byte is page 0
      Line("CMXMARK", "text_start", e, n);
      g_mk_text_start = (long long)n;
      g_text = true;
      g_region = 1;
    }
  }
  if (g_text && !AuthorBlockInert()) {
    // pages of the order: author_block counts the page anchors behind the metablock (Seite() = the page being read,
    // as the redirect-block test of the coder reads it); after the anchor byte of page k-1 it is k
    const long long sp = author_block::aut.Seite();
    if (!g_first_article && !redirect_block(sp)) {   // behind the front redirect block (pages 0..71147)
      Line("CMXMARK", "first_article", e, n);
      g_mk_first_article = (long long)n;
      g_first_article = true;
      if (g_sem_kcen0) p->CmxSemKeyReset();   // CMX_SEM_KCEN0: the key mean is counted from the first article
      if (g_region < 2) g_region = 2;
    }
    if ((g_mixrow == 1 || g_mixrow == 2) && !g_rekeyed && sp >= 1) {   // CMX_MIXROW: re-key at page 1
      Line("CMXMARK", "mixrow_rekey", e, n);
      p->CmxMixRekey();
      g_rekeyed = true;
    }
    while (g_pages_i < g_pages_n && sp >= g_pages[g_pages_i]) {
      char nm[32];
      snprintf(nm, sizeof nm, "page%lld", g_pages[g_pages_i]);
      Line("CMXMARK", nm, e, n);
      ++g_pages_i;
    }
  }
  if (n >= 20000000ULL && g_region < 3) g_region = 3;   // telemetry regions: the 20 M base of Delta(50M - 20M)
  bool at = false;
  while (g_at_i < g_at_n && n >= (unsigned long long)g_at[g_at_i]) { at = true; ++g_at_i; }
  const bool stop = g_stop_at && n >= (unsigned long long)g_stop_at;
  if ((g_log_every && n % (unsigned long long)g_log_every == 0) || stop || at) Line("CMX", "", e, n);
  if (stop) {
    WriteByteCost();
    TelReport(p, "stop");
    SemDumpClose();
    MatchDumpClose();
    e.Flush();   // the output file is now the archive payload of the first n coder bytes (header included)
    fflush(os);
    fprintf(stderr, "CMX stop at %llu\n", n);
    fflush(nullptr);
    _exit(0);
  }
}
}  // namespace cmx_meas
#endif

void Compress(unsigned long long input_bytes, FILE* is,
    FILE* os, unsigned long long* output_bytes, Predictor* p) {
#if PORT_CMX_MEAS
  // research CMX_TFOFF=2: BITLSTM's transformer-state input reads zeros (registered by the Encoder constructor)
  if (cmx_meas::g_tfoff >= 2 && transformer::Active()) {
    static float kCmxNoTfState[192];
    p->bitl_hid_ = kCmxNoTfState;
  }
  if (cmx_meas::g_bytecost)
    cmx_meas::ByteCost().reserve((size_t)(cmx_meas::g_stop_at && (unsigned long long)cmx_meas::g_stop_at < input_bytes
                                          ? (unsigned long long)cmx_meas::g_stop_at : input_bytes));
#endif
  Encoder e(os, p);

  unsigned long long percent =
      1 + (input_bytes / (unsigned long long)progress_steps);
  unsigned long long prog_left = 1;
  ClearOutput();
  // the read block (BSS, one coder at a time per process)
  static char buffer[(size_t)io_buffer_kb * 1024];
  const size_t buffer_size = sizeof buffer;
  unsigned long long pos = 0;
  int sem_pb = 0;   // the previous stream byte (0 before the first)
  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer_size, input_bytes - pos));
    const size_t got = fread(buffer, 1, chunk, is);
    if (got == 0) break;
    for (size_t i = 0; i < got; ++i) {
      unsigned char c = static_cast<unsigned char>(buffer[i]);
      CMXK_START(cmx_lp)
      CMXK_START(cmx_ck)
      p->SemByteStart(sem_pb);
      CMXK_LAP(CK_SEMBYTE, cmx_ck)
#if PORT_CMX_MEAS
      if (__builtin_expect(cmx_meas::g_nsk != 0, 0)) p->CmxSem2ByteStart(sem_pb);   // research CMX_SEMKEYS (as in Decompress)
      if (__builtin_expect(cmx_meas::g_semdump != nullptr, 0)) cmx_meas::SemDumpByte(pos);
      if (__builtin_expect(cmx_meas::g_matchdump != nullptr, 0)) cmx_meas::MatchDumpArm(pos);
#endif
      sem_pb = c;
      const int zw = p->Tf1ZwangByte();
      if (zw >= 0) {
        for (int j = 7; j >= 0; --j) e.ObserveKnownBit((zw >> j) & 1);
        if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
      } else
      if (metablock::zw::an) TextStartSepEnc(e, c, pos);
      else
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
      CMXK_START(cmx_ak)
      AfterStreamByte(p, c);   // the readers of the stream, mirrored in Decompress()
      CMXK_LAP(CK_AFTER, cmx_ak)
#if PORT_CMX_MEAS
      if (cmx_meas::g_hook) cmx_meas::AfterByte(e, os, pos + 1, p);
#endif
      if (--prog_left == 0) {
        prog_left = percent;
        double frac = 100.0 * pos / input_bytes;

        HUGEPAGE_TICK(pos);
        console_progress::Tick(pos, input_bytes);
      }
      CMXK_LAP(CK_LOOP, cmx_lp)
      ++pos;
    }
  }
  e.Flush();
#if PORT_CMX_MEAS
  if (cmx_meas::g_hook) {   // the end of the stream (no stop position reached)
    cmx_meas::Line("CMXEND", "", e, pos);
    cmx_meas::WriteByteCost();
    cmx_meas::TelReport(p, "end");
    cmx_meas::SemDumpClose();
    cmx_meas::MatchDumpClose();
  }
#endif
  run_mode::gram.Report("compress");
  if (!MetablockInert()) {
    metablock::mk.Report("compress");
    census_block::aut.Report("compress");
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
  *output_bytes = ftell(os);
}

void Decompress(unsigned long long output_length, FILE* is,
                FILE* os, Predictor* p) {
  Decoder d(is, p);
  unsigned long long percent = 1 + (output_length / 10000);
  unsigned long long prog_left = 1;
  ClearOutput();
  int sem_pb = 0;   // the previous stream byte (0 before the first)
  for(unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    p->SemByteStart(sem_pb);
#if PORT_CMX_MEAS
    if (__builtin_expect(cmx_meas::g_nsk != 0, 0)) p->CmxSem2ByteStart(sem_pb);   // research CMX_SEMKEYS (as in Compress)
#endif
    const int zw = p->Tf1ZwangByte();
    if (zw >= 0) {
      for (int j = 7; j >= 0; --j) d.ObserveKnownBit((zw >> j) & 1);
      byte = 256 + zw;
      if (run_mode::gram.ByteActive(pos)) run_mode::gram.NextByte((unsigned int)zw);
    } else
    if (metablock::zw::an) byte = TextStartSepDec(d);
    else
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
    putc(byte, os);
    sem_pb = byte & 255;   // the stream byte for the next SemByteStart, as in Compress()
    AfterStreamByte(p, (unsigned char)(byte & 255));   // as in Compress()
    if (--prog_left == 0) {
      prog_left = percent;
      double frac = 100.0 * pos / output_length;
      HUGEPAGE_TICK(pos);
      console_progress::Tick(pos, output_length);
    }
  }
  run_mode::gram.Report("decompress");
  if (!MetablockInert()) {
    metablock::mk.Report("decompress");
    census_block::aut.Report("decompress");
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
  preprocessor::Encode(data_in, data_out, *input_bytes, dictionary);
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
    preprocessor::Encode(data_in, temp_out, *input_bytes, dictionary);
  } else {
    preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
  }
  fclose(data_in);
  fclose(temp_out);

  if (census_block_active && !census_block::Einfuegen(temp_path, ".censblock", dictionary)) {
    msg_note(407);
    return false;
  }
  if (!metablock_form::Pack(temp_path)) {
    msg_note(171);
    return false;
  }

  if (!ip_pack::Pack(temp_path)) {
    msg_note(172);
    return false;
  }
  if (!MetablockInert() && !dict_article::SlotDatei()) {
    msg_note(547);
    return false;
  }
  if (!MetablockInert() && !rule_stages::DateiEnc(temp_path)) {
    msg_note(553);
    return false;
  }
  if (!text_start::Pack(temp_path, !MetablockInert(), !enable_preprocess)) {
    msg_note(536);
    return false;
  }

  console_progress::PhaseEnd();   // "preprocessing" ends here

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;

  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  fseek(temp_in, 0L, SEEK_END);
  unsigned long long temp_bytes = ftell(temp_in);
  fseek(temp_in, 0L, SEEK_SET);

  std::vector<bool> vocab(256, false);
#if AUX_BUILD
  vocab_frozen::g_aux = true;
#endif
  vocab_frozen::FillVocab(&vocab);
  WriteHeaderVocfix(temp_bytes, dictionary != NULL, data_out);

  dict_article::strom = temp_bytes;
  dict_article::wrt = dictionary != NULL;
  if (!MetablockInert()) dict_article::Verbergen();
  Predictor p(vocab);
  HUGEPAGE_COLLAPSE("c-init");
  malloc_trim(0);
  console_progress::PhaseEnd();   // "pretraining ..." ends here
  Compress(temp_bytes, temp_in, data_out, output_bytes, &p);
  if (!MetablockInert()) dict_article::Pruefen();
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  std::vector<bool> vocab(256, false);
  bool dictionary_used;
  ReadHeaderVocfix(data_in, output_bytes, &dictionary_used, &vocab);
  if (!dictionary_used && dictionary != NULL) return false;

  {
    dict_article::strom = *output_bytes;
    dict_article::wrt = dictionary_used;
    if (!MetablockInert()) dict_article::Verbergen();
    Predictor p(vocab);
    HUGEPAGE_COLLAPSE("d-init");

    FILE* temp_out = fopen(temp_path.c_str(), "wb");
    if (!temp_out) return false;

    console_progress::PhaseEnd();   // "pretraining ..." ends here
    Decompress(*output_bytes, data_in, temp_out, &p);
    // the inverse transforms of the main run take some minutes
    if (dictionary_used) console_progress::Phase("postprocessing");
    p.FreeFxcmMemory();
    p.FreeFxcmRest();
    fclose(data_in);
    fclose(temp_out);
  }
  malloc_trim(0);
  if (!text_start::Unpack(temp_path, !MetablockInert())) {
    msg_note(537);
    return false;
  }
  if (!MetablockInert()) {
    if (!dict_article::DateiAusSlot()) {
      msg_note(548);
      return false;
    }
    if (dict_article::wrt) {
      if (dictionary) fclose(dictionary);
      dictionary = fopen(".dict", "rb");
    }
  }
  if (!MetablockInert() && !rule_stages::DateiDec(temp_path)) {
    msg_note(554);
    return false;
  }

  if (!ip_pack::Unpack(temp_path)) {
    msg_note(173);
    return false;
  }

  if (!metablock_form::Unpack(temp_path)) {
    msg_note(174);
    return false;
  }
  if (census_block_active && !census_block::Entnehmen(temp_path, ".censblock", dictionary)) {
    msg_note(408);
    return false;
  }

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  isbn_check::after_at.Put(-1, data_out);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

#if PORT_CMX_MEAS
// ---- research bed mode (PORT_CMX_MEAS; Claude Opus 5.5, 2026-10-31) -----------------------------------------------
// comp9 -n STREAM OUT codes a FINISHED coder stream (the file RunCompression hands to Compress: archive.cmix.temp of
// an -e run, e.g. the regenerated neif2_stream_CPU_20261031/stream/out.cmix.temp, or a prefix of it) with the model
// set-up of RunCompression behind all of its packing stages: header with the dictionary flag (the -e run always has
// its dictionary), dict_article state, Verbergen (an ./.dict in the run dir waits as .dict.ta, as in -e; without one
// the run is the decoder's case, slot 1 writes ./.dict), the Predictor, Compress.  No preprocessing: the coder sees
// exactly the bytes of STREAM.  The CMX_* knobs apply as in -e.  Needs the weight container in the run dir.
static bool CmxRunBed(const char* in, const char* out, unsigned long long* ib, unsigned long long* ob) {
  FILE* temp_in = fopen(in, "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(out, "wb");
  if (!data_out) return false;
  fseek(temp_in, 0L, SEEK_END);
  const unsigned long long temp_bytes = ftell(temp_in);
  fseek(temp_in, 0L, SEEK_SET);
  *ib = temp_bytes;
  std::vector<bool> vocab(256, false);
#if AUX_BUILD
  vocab_frozen::g_aux = true;
#endif
  vocab_frozen::FillVocab(&vocab);
  WriteHeaderVocfix(temp_bytes, true, data_out);
  dict_article::strom = temp_bytes;
  dict_article::wrt = true;
  if (!MetablockInert()) dict_article::Verbergen();
  Predictor p(vocab);
  HUGEPAGE_COLLAPSE("c-init");
  malloc_trim(0);
  console_progress::PhaseEnd();
  Compress(temp_bytes, temp_in, data_out, ob, &p);
  if (!MetablockInert()) dict_article::Pruefen();
  fclose(temp_in);
  fclose(data_out);
  return true;
}
__attribute__((noinline)) static int CmxBedMain(const char* in, const char* out) {
  srand((unsigned int)run_options::ParamI("seed", SEED));
  progress_steps = run_options::ParamI("progress_steps", 10000);
  if (progress_steps < 1) progress_steps = 10000;
  census_block::zahl::g_var = cmx_meas::g_zvar;
  console_progress::Start(true, out);
  unsigned long long ib = 0, ob = 0;
  if (!CmxRunBed(in, out, &ib, &ob)) {
    fprintf(stderr, "CMX bed -n failed\n");
    return 7;
  }
  const double t = console_progress::End(true);
  printf("%lld bytes -> %lld bytes in %1.2f s.\n", ib, ob, t);
  return 0;
}
#if !PORT_ENTRY || PORT_ENTRY_TEST   // not in the shipping form of the entry (it reads CMX_RECODE_LIMIT)
// comp9 -r IN OUT V: the census line of the coder stream IN re-encoded with number-coder variant V (decoded with the
// release coder, variant 0), the rest copied; OUT gets the first CMX_RECODE_LIMIT bytes (0 = all).  Port of the side
// agent's ProbeRecode (newlev_side/probe.patch).  Prints the length change of the line.
__attribute__((noinline)) static int CmxRecodeMain(const char* in, const char* out, int V) {
  using namespace census_block;
  std::string d;
  {
    FILE* f = fopen(in, "rb");
    if (!f) return 1;
    static char b[1 << 16];
    size_t g;
    while ((g = fread(b, 1, sizeof b, f)) > 0) d.append(b, g);
    fclose(f);
  }
  const size_t i0 = d.find("zcb3");
  if (i0 == std::string::npos) return 2;
  const size_t j0 = d.find('\n', i0);
  const unsigned char* base = (const unsigned char*)d.data();
  const unsigned char* p = base + i0 + 4;
  const unsigned char* e = base + j0;
  static std::vector<std::vector<VarD>> form[kNTypen];
  for (int ti = 0; ti < kNTypen; ++ti) {
    const long npos = LiesB100(p, e, 2);
    form[ti].assign((size_t)npos, std::vector<VarD>());
    for (long q = 0; q < npos; ++q) {
      const long nvar = LiesB100(p, e, 2);
      for (long v = 0; v < nvar; ++v) {
        const long nlab = LiesB100(p, e, 1);
        VarD vd;
        for (long k = 0; k < nlab; ++k) {
          const long lab = LiesB100(p, e, 1);
          long yr = 0;
          if (lab == zahl::F_PV) yr = LiesB100(p, e, 2);
          vd.lab.push_back((unsigned char)lab);
          vd.jahr.push_back((int)yr);
        }
        form[ti][(size_t)q].push_back(vd);
      }
    }
  }
  std::string o(d.data(), (size_t)(p - base));   // everything up to the run
  const unsigned char* r0p = p;
  std::vector<std::pair<int, long>> run_list;
  {
    const long r0 = LiesB100(p, e, 3), rn = LiesB100(p, e, 3);
    std::vector<long> lsx[kNTypen];
    for (long i = 0; i < rn; ++i) { const int v = (int)*p++ - 0x80; if (v < kNTypen) lsx[v].push_back(r0 + i); }
    for (int ti = 0; ti < kNTypen; ++ti) for (long pg : lsx[ti]) run_list.push_back(std::make_pair(ti, pg));
  }
  o.append((const char*)r0p, (size_t)(p - r0p));
  anl4_tab.clear(); LueckeLeer(); KreislisteLeer(); KreisArtLeer(); nachbar_ang.clear();
  static zahl::Typmodell tm_old[kNTypen], tm_new[kNTypen];
  zahl::Run::ZweigeSetzen();
  long ntok = 0, nchg = 0, bytes_old = 0, bytes_new = 0;
  for (size_t lk = 0; lk < run_list.size(); ++lk) {
    const unsigned char* rs = p;
    Satz s;
    s.typ = run_list[lk].first;
    s.page_index = run_list[lk].second;
    s.staat = (int)LiesB100(p, e, 1);
    s.titel = (int)LiesB100(p, e, 1);
    if (kTypen[s.typ] == 'k') zahl::kreisliste[(size_t)s.staat].push_back(s.page_index);
    if (kTypen[s.typ] == 'k') KreisArtSetze(s.page_index, s.titel, s.staat);
    const auto& F = form[(size_t)s.typ];
    for (size_t j = 0; j < F.size(); ++j) {
      const long f = LiesB100(p, e, 1);
      if (f == 0) continue;
      if (f == 99) {
        const long c = LiesB100(p, e, 2);
        for (long q = 0; q < c; ++q) { const long v = LiesB100(p, e, 1); s.inst.push_back(std::make_pair((int)j, (int)v)); }
      } else s.inst.push_back(std::make_pair((int)j, (int)(f - 1)));
    }
    std::vector<Slot> slots;
    for (const auto& iv : s.inst) {
      const VarD& vd = F[(size_t)iv.first][(size_t)iv.second];
      for (size_t k = 0; k < vd.lab.size(); ++k) { Slot sl; sl.lab = vd.lab[k]; sl.jahr = vd.jahr[k]; slots.push_back(sl); }
    }
    s.kreis_zusatz = KreisZusatz(kTypen[s.typ], &slots);
    const size_t nb = s.inst.size() + 1;
    for (size_t g = 0; g < nb; g += 6) LiesB100(p, e, 1);
    o.append((const char*)rs, (size_t)(p - rs));   // record head and gap digits as they are
    std::vector<std::string> tok;
    for (size_t i = 0; i < slots.size(); ++i) {
      const unsigned char* q = p;
      while (q < e && *q != '\t') ++q;
      tok.push_back(std::string((const char*)p, (size_t)(q - p)));
      p = q + 1;
    }
    Anl4Setzen(s.typ, s.page_index, s.staat);
    zahl::g_cur_page = s.page_index;
    zahl::g_var = 0;
    if (!Zahlen(tm_old[s.typ], kTypen[s.typ] == 'k', s.staat, slots, &s.slot_values, &tok, false)) {
      fprintf(stderr, "recode: decode fail rec %zu\n", lk);
      return 3;
    }
    std::vector<std::string> sv = s.slot_values, tok2;
    zahl::g_var = V;
    if (!Zahlen(tm_new[s.typ], kTypen[s.typ] == 'k', s.staat, slots, &sv, &tok2, true)) {
      fprintf(stderr, "recode: encode fail rec %zu\n", lk);
      return 4;
    }
    zahl::g_var = 0;
    if (tok2.size() != tok.size()) {
      fprintf(stderr, "recode: token count %zu vs %zu rec %zu\n", tok2.size(), tok.size(), lk);
      return 5;
    }
    for (size_t i = 0; i < tok2.size(); ++i) {
      ++ntok; nchg += tok2[i] != tok[i]; bytes_old += (long)tok[i].size(); bytes_new += (long)tok2[i].size();
      o += tok2[i];
      o.push_back('\t');
    }
    for (size_t q = 0; q < slots.size() && q < s.slot_values.size(); ++q)
      if (slots[q].lab == zahl::F_KREIS) { s.kreis = strtol(s.slot_values[q].c_str(), nullptr, 10); break; }
    Anl4Merken(s.typ, s.page_index, s.staat, kTypen[s.typ] == 'k' ? s.page_index : s.kreis);
    if (*p != 0x0B) { fprintf(stderr, "recode: no record end %zu\n", lk); return 6; }
    ++p;
    o.push_back('\x0b');
  }
  if (p != e) { fprintf(stderr, "recode: line rest %ld\n", (long)(e - p)); return 7; }
  o.append(d, j0, std::string::npos);
  const unsigned long long lim = (unsigned long long)cmx_meas::Env("CMX_RECODE_LIMIT");
  const size_t w = lim && lim < o.size() ? (size_t)lim : o.size();
  FILE* f = fopen(out, "wb");
  if (!f) return 8;
  fwrite(o.data(), 1, w, f);
  fclose(f);
  fprintf(stderr, "CMX recode V=%d: records %zu tokens %ld changed %ld token bytes %ld -> %ld, census line %zu -> %zu "
          "(delta %ld), stream %zu -> %zu, written %zu\n", V, run_list.size(), ntok, nchg, bytes_old, bytes_new,
          j0 - i0 + 1, j0 - i0 + 1 + (size_t)(bytes_new - bytes_old), bytes_new - bytes_old, d.size(), o.size(), w);
  return 0;
}
#endif   // !PORT_ENTRY || PORT_ENTRY_TEST
#endif

namespace ppmd_page { extern int hilfs; }
int main(int argc, char** argv) {
  console_progress::Begin();
  if (argc == 1 && no_args::IsCompressor()) {
    argc = 4;
    argv = (char**)no_args::kArgVector.argv;
  }
#if PORT_ENTRY
  // PORT_ENTRY (src/port_cfg.h): the recipe in every mode that codes or decodes the main stream.  The mode is read off
  // the argument vector as given, before the -D -> -d rewrite below: a -D run is a helper block (the two -D children
  // of an -e run; their blocks were coded at build time by cmix_aux -c with the release defaults) and keeps the
  // release defaults.  Everything else gets the recipe: -e (also the no-argument compressor, rewritten above), -n, a
  // top-level -d, and the no-argument archive (the self-extracting decoder, argc == 1).  The build helper
  // (AUX_BUILD) never forces and never reads the environment.
#if !AUX_BUILD
  {
    const bool helper_block = argc > 1 && argv[1][0] == '-' && argv[1][1] == 'D';
#if PORT_ENTRY_TEST
    // test form: the observers of a top-level -e / -n run from the environment; EntryApply() resets every coding knob
    if (!helper_block && argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'e' || argv[1][1] == 'n') && argv[1][2] == '\0')
      cmx_meas::Read();
#endif
    if (!helper_block) cmx_meas::EntryApply();
  }
#endif
#elif PORT_CMX_MEAS
  // research knobs: a top-level -e run only (its -D children inherit the environment and keep the defaults)
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'e' || argv[1][1] == 'n') && argv[1][2] == '\0') cmx_meas::Read();
#endif
#if !AUX_BUILD
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] | 0x20) == 'd')   // -d or -D
#else
  if (argc > 1 && argv[1][0] == '-' && (argv[1][1] == 'c' || argv[1][1] == 'd'
      || argv[1][1] == 'C' || argv[1][1] == 'D'))
#endif
    ppmd_page::hilfs = 1;
  mallopt(M_MMAP_THRESHOLD, 1 << 20);
#if PORT_CMX_MEAS
  // research modes (not in the release CLI): -n STREAM OUT (bed), -r IN OUT V (census line recode)
  if (argc == 4 && strcmp(argv[1], "-n") == 0) return CmxBedMain(argv[2], argv[3]);
#if !PORT_ENTRY || PORT_ENTRY_TEST
  if (argc == 5 && strcmp(argv[1], "-r") == 0) return CmxRecodeMain(argv[2], argv[3], atoi(argv[4]));
#endif
#endif
  if (PredictorLayoutTagFast() != ((unsigned long long)sizeof(Predictor) * 4ULL + (unsigned long long)(1) * 2ULL + (unsigned long long)(0))) {
    msg_note(175)

;
    fflush(stderr);
    return -1;
  }
#if !AUX_BUILD
  // The release form takes -e (compress), -d (decompress; a fifth argument, a
  // dictionary path, is accepted and not read) and -D (an embedded block of
  // the self-extraction).  Builds without this restriction, the build helper
  // among them, also take -c/-n/-s/-x and -C.
  if ((argc != 1) && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'd' && argv[1][1] != 'e'
      && argv[1][1] != 'D'
      ))) {
    return Help();
  }
#else
  if ((argc != 1) && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'x' && argv[1][1] != 's' &&
      argv[1][1] != 'n' && argv[1][1] != 'e'
      && argv[1][1] != 'C' && argv[1][1] != 'D'
      ))) {
    return Help();
  }
#endif
#if !AUX_BUILD
  if (argc > 1 && argv[1][1] == 'D') {
#else
  if (argc > 1 && (argv[1][1] == 'C' || argv[1][1] == 'D')) {
#endif
    transformer::SetBlobForm();
#if !AUX_BUILD
    argv[1][1] = 'd';
#else
    argv[1][1] = (argv[1][1] == 'C') ? 'c' : 'd';
#endif
  }
  srand((unsigned int)run_options::ParamI("seed", SEED));
  progress_steps = run_options::ParamI("progress_steps", 10000);
  if (progress_steps < 1) progress_steps = 10000;

#if !1 || AUX_BUILD
  clock_t start = clock();   // only for the end line of the build helper
#endif

  bool enable_preprocess = true;
  std::string input_path ;
  std::string output_path;
  FILE* dictionary = NULL;

  if (argc > 1)  {
#if !AUX_BUILD
    // the last two arguments; a dictionary argument of -d is not read, the
    // dictionary is slot 1 of the stream
    input_path = argv[argc - 2];
    output_path = argv[argc - 1];
#else
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n') return Help();
      dictionary = fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
#endif
  }

  std::string temp_path = output_path + ".cmix.temp";
  const bool progress_encoding = argc > 1 && (argv[1][1] == 'c' || argv[1][1] == 'e');
  console_progress::Start(progress_encoding, output_path.c_str());

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argc == 1) {
    //Decompress enwik9
    // unpack a) header b) cmix dictionary, c) new order of articles, d) actual cmix binary
    census_block_active = true;
    if (selfextract_decomp() != 0) {
      return Help();
    }

    // run compression
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp" ;
    dictionary = fopen(".dict", "rb");
    isbn_check::after_at.an = true;

    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes,
        nullptr
        )) {
      return Help();
    }

    split4Decomp();

    // apply phda9 preprocessor
    phda9_resto();
    if (!census_block::Setze(".main_decomp_restored", ".censblock")) {
      msg_note(409);
      return Help();
    }
    if (!comment_form::Dekodiere(".main_decomp_restored")) {
      msg_note(484);
      return Help();
    }

    // change the order of articles in the input
    sort();

    if (!cmtidx_inplace<0>(".main_decomp_restored_sorted", ".cmtidx_resto")) {
      msg_note(176);
      return Help();
    }

    // merge all input parts after preprocessing; the census inverse transform
    // follows as the last step - mirror image of compression, where it ran first.
    {
      static const char* const teile[] = {".intro_decomp", ".main_decomp_restored_sorted",
                                          ".coda_decomp", nullptr};
      if (!cat(".census_resto", teile)) return Help();
    }
    if (!census_file<0>(".census_resto", "enwik9_uncompressed")) {
      msg_note(177);
      return Help();
    }
    remove(".census_resto");

    selfx_clean::AfterDecompress();

    goto print_end_message;
  }

#if !(1 && !AUX_BUILD)   // the package form has no -c
  if (argv[1][1] == 'c') {
      if (argv[1][1] == 'c' && argc == 5) {
        if (!provide_dict_file(argv[2])) return Help();
      }
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else
#endif
  if (argv[1][1] == 'e') {
#define ENC_FAIL return Help()

    if (selfextract_comp() != 0) {
      ENC_FAIL;
    }

    {
      console_progress::Phase("dictionary");
      const bool derived = dict_derive::BuildDictFromInput(
          input_path.c_str(), ".hints", ".dict");
      console_progress::PhaseEnd();
      if (!derived) {
        msg_note(178);
        ENC_FAIL;
      }
      remove(".hints");
      remove(".hints.comp");
    }

    console_progress::Phase("preprocessing");
    if (!census_file_enc(input_path.c_str(), ".census_prepr")) {
      msg_note(181);
      ENC_FAIL;
    }
    split4Comp(".census_prepr");
    remove(".census_prepr");
    if (!cmtidx_inplace_enc(".main", ".cmtidx_main")) {
      msg_note(182);
      ENC_FAIL;
    }

    // change the order of articles in the input
    reorder();
    if (!comment_form::Kodiere(".main_reordered")) {
      msg_note(483);
      ENC_FAIL;
    }
    if (!census_block::Zerlege(".main_reordered", ".censblock")) {
      msg_note(406);
      ENC_FAIL;
    }
    census_block_active = true;

    // apply phda9 preprocessor
    phda9_prepr();

    // merge all input parts after preprocessing
    {
      static const char* const teile[] = {".main_phda9prepr", ".intro", ".coda", nullptr};
      if (!cat(".ready4cmix", teile)) ENC_FAIL;
    }
    isbn_check::Pack();

    // run compression
    input_path = ".ready4cmix";
    dictionary = fopen(".dict", "rb");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes,
        nullptr
        )) {
      ENC_FAIL;
    }

    // construct a selfextracting decompressor binary; in the release
    // configuration
    //   archive9 = [decoder][container][payload][trailer]
    // (the dictionary travels as slot 1 of the stream; a build without the
    // dictionary article puts .dict.comp behind the decoder, one without the
    // embedded container leaves the container out).
    // It is written under the temporary name first: the payload may be the
    // file archive9 itself (the jury form writes its output there).
    {
      ArchivKopf header;
      header.weights_size = (int)getFileSize("weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5");
      header.decomp_input_size = (int)getFileSize(output_path.c_str());
      const char* const teile[] = {".decomp_bin",
                                   "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5",
                                   output_path.c_str(), nullptr};
      if (!cat(temp_path.c_str(), teile, &header, sizeof(header)) ||
          rename(temp_path.c_str(), "archive9") != 0)
        ENC_FAIL;
    }
    // make the decompressor binary executable
    chmod("archive9", 0777);

    selfx_clean::AfterCompress();

#undef ENC_FAIL
  }
  else
  {
#if !(1 && !AUX_BUILD)
    if (argc == 5) {
      if (!provide_dict_file(argv[2])) return Help();
    }
#endif
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
  }
#else
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      input_bytes, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);
#endif

  return 0;
}
