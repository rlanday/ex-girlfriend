#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_


#include "author_block.h"

// The pre-trained transformer in the byte mixer.
//
// The inference engine comes from fx2-cmix-transformer (Vladimir Ivanov,
// GPL-3).  After every byte the engine takes the PPMd byte distribution as its
// prior input and produces the distribution over the next byte, which it hands
// to a byte-model slot of its own (Tf1Slot::SetProbs, mixer/byte-mixer.h).
//
// The engine works on the frozen 208-value byte set below.  At construction it
// compares the runtime vocabulary element by element against that set and
// fails on any byte outside it; checking only the size (205 in the original)
// would not catch a different set of the same size.
//
// The network of the release configuration: 16 layers of width 192 in the
// layer pattern of the original (three linear-attention layers of the Kimi
// type, then one sliding-window attention layer, here over 2048 tokens), MLP
// width up to 1024, and the form of cmix-neif-beehive (Vladimir Malinovskii,
// 2026): an AttnRes depth mix in place of the residual sums
// (Kimi Team, "Attention Residuals", arXiv 2603.15031), one K/V head shared
// by the three query heads of a window layer, value embeddings in the window
// layers and the per-token prior logit head of the original training code.

// ---- the transformer side of the text form ---------------------------------
// In the release configuration the prompt of the transformer carries:
//   * the block mark "#<word>" as the first prompt row of every page of a
//     block of the article order, in the place where the pages of the front
//     redirect block get `#rd`.  Membership by page index: the order puts
//     these pages into contiguous ranges (stamped table, built into tf1/tf1_prompt.cpp):
//     class marks in stream form (calendar, census with the US-place runs in
//     front of it, disambiguation pages of three kinds), the ship block and
//     the topic regions; no mark in the image and project blocks.  Causal:
//     the page index is known at the separator.
//   * the forced title prefixes: `Image:`, `Help:`, `MediaWiki:` and
//     `Wikipedia:` stand without exception in contiguous page ranges of the
//     order (the same table).  Coder and decoder take these bytes from
//     the table and observe them without coding (Predictor::ObserveKnownBit);
//     page index and title position come from the prompt automaton.  The
//     training dump keeps the rows that predict them masked (flags bit 1).
//   * the link context (below).
// The ranges of that table are page indices of the order with the
// calendar block at the calendar position; with another order they would
// mark and force the wrong pages.

#include "alias_title_subst.h"

namespace alias_prompt {
// Reinjection counter for alias rows.  Beginn/Ende/Laeuft/Verbrauche mean
// begin, end, running and consume.
// How many reinjected bytes of the current block are still to come, counting
// the one being handled.  `Beginn(n)` is called by the coder AND by the
// decoder immediately before the block is pushed through the models, so both
// sides see the identical sequence (src/runner.cpp, Compress/Decompress).
// 0 = no reinjection running.
inline int g_rest = 0;
inline void Beginn(int n) { g_rest = n; }
inline void Ende() { g_rest = 0; }
// True while a reinjected byte is being handled.  Read by the predictor (the
// transformer slot abstains) and by Tf1Engine::Step (prompt-class row).
inline bool Laeuft() { return g_rest > 0; }
// The engine consumes one byte of the block and says whether it was the last
// one.  Only that row predicts a real stream token and therefore gets its
// PPMd prior instead of prior0.
inline bool Verbrauche() { return --g_rest == 0; }
}
namespace tf_author_prompt {
inline bool g_ist_autor = false;

inline void Beginn(int n) {
  g_ist_autor = true;
  comment_tag::g_j = comment_tag::g_offen ? 0 : 2;
  comment_tag::g_offen = false;
  alias_prompt::Beginn(n);
}
inline void Ende() { alias_prompt::Ende(); g_ist_autor = false; }
inline bool Laeuft() { return g_ist_autor; }
}  // namespace tf_author_prompt
namespace timestamp_skip {
inline bool g_an = false;
inline void Beginn() { g_an = true; }
inline void Ende() { g_an = false; }
inline bool Laeuft() { return g_an; }
}  // namespace timestamp_skip
#include "tf1/tf1_config.h"

namespace tf_copy_end {
// the end mark follows the row at which the rest counter of the prompt class
// (alias_prompt::g_rest, that byte counted) equals g_nach; 0 = none.  Set
// around the block by the caller, on both sides.
inline int g_nach = 0;
}

#include "alias_substring_dedup.h"
#include "alias_title_subst.h"

#include "tf1/tf1_prior0_set_v14.h"


namespace tf_z3 {

constexpr int kEndeKlasse = 5;
}  // namespace tf_z3

#include <stdlib.h>
#include <string.h>


#include "vocab_frozen.h"

namespace transformer {

constexpr unsigned char kSetBitmap[32] = { 0xfe, 0x1e, 0x00, 0x00, 0xff, 0xff, 0xff, 0x03, 0x01, 0xfc, 0x0f, 0xf9, 0xfe, 0xff, 0xff, 0x07, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
constexpr int kByteToIndex[256] = { 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 48, 48, 48, 48, 49, 49, 49, 50, 51, 52, 53, 54, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 80, 80, 80, 80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207 };
constexpr int kSetSize = 208;

static_assert(kSetSize == 208,
    "the transformer width must equal the frozen byte "
    "set size.  Re-stamp the set or rebuild the model.");
static_assert(208 % 16 == 0,
    "V must be a multiple of 16 (the ported kernels rely on it).");

constexpr bool InSet(int v) {
  return ((kSetBitmap[v >> 3] >> (v & 7)) & 1) != 0;
}

// The run-time readers index g_b2i, a copy
// of kByteToIndex in .bss filled once before main (tf1_engine.cpp); the
// constant table stays for the compile-time uses only.  Hidden, so a reader
// keeps its absolute addressing (no GOT load).
extern int g_b2i[256] __attribute__((visibility("hidden")));
constexpr bool B2iFormel() {
  int k = 0;
  for (int i = 0; i < 256; ++i)
    if (kByteToIndex[i] != (InSet(i) ? k++ : k)) return false;
  return true;
}
static_assert(B2iFormel(), "g_b2i is filled by the rank rule (set bytes below i), which must give kByteToIndex");

inline bool& BlobForm() {
  static bool blob = false;
  return blob;
}
inline void SetBlobForm() { BlobForm() = true; }

// Weights file, fingerprint and mode are compile-time constants, so
// predictor.cpp folds its `Active()` branches.  The only run-time states are
// the blob form (the self-invocations with -C/-D, which code the appended data
// blocks without the transformer) and the auxiliary-stream flag below; nothing
// is read from the environment.
inline const char* WeightsPath() { return "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5"; }
inline unsigned long long WeightsFingerprint() {
  return 0xaec368ac6a568632ULL;
}
inline bool Mute() { return false; }

inline bool AuxStream() {
  return vocab_frozen::g_aux;
}

inline bool Active() { return !BlobForm() && !AuxStream(); }
inline bool Running() { return !BlobForm() && !AuxStream(); }

// Output paths of the training-data dumps (link-context stream, prompt rows,
// match channel).  The release configuration writes no dumps.
inline const char* Z3DumpPath() {
  return nullptr;
}

inline const char* PromptDumpPath() {
  return nullptr;
}
inline const char* MatchDumpPath() {
  return nullptr;
}
// Match length class as in the training package: min(raw_bits / 8, 15).
inline int MatchLengthClass(int raw_bits) {
  int c = raw_bits / 8;
  return c > 15 ? 15 : c;
}

}  // namespace transformer

#endif
