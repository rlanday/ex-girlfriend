//
// ---------------------------------------------------------------------------
// ATTRIBUTION
//   Origin: zmix 1.0 (James Byrne), GPL-3.0-or-later, build option -Dwordlbl
//   (src/models/wordlbl.zig, its v1 / -Dwordlbl-ctx=0 configuration and the
//   -Dwordlbl-ctx>0 tied-decay variant).  The MECHANISM and the swept
//   hyper-parameters (D=96, eta_N=0.060, eta_E=0.030, lambda=0.85) are taken
//   from there; the C++ below is a port of that Zig source and is stamped onto
//   OUR dictionary geometry, not theirs.  zmix in turn cites
//     Mnih and Hinton 2007, "Three New Graphical Models for Statistical
//       Language Modelling" (the log-bilinear context model), and
//     Morin and Bengio 2005, "Hierarchical Probabilistic Neural Network
//       Language Model" (the hierarchical softmax over a word tree).
//   The -Dwordlbl-attn* family (content-addressed induction memory) is
//   deliberately NOT ported.
//
// ---------------------------------------------------------------------------
// WHAT IT DOES, IN ONE PARAGRAPH
//   Every word model in the PAQ -> lpaq -> cmix -> fxcm lineage estimates word
//   identity through HASHED LOOKUP TABLES: each of our 54,857 dictionary words
//   is learned on its own, and nothing learned about one word transfers to a
//   word that behaves alike.  This module adds the one parameterisation that
//   DOES share: an online LOG-BILINEAR model.  Every word rank r owns a dense
//   embedding row E[r] of D floats; the context vector is the (weighted) sum of
//   the embeddings of the previous words; a trie node n owns a weight row N[n]
//   plus a bias, and the probability of the next coded bit of the current WRT
//   codeword is squash(bias[n] + <cvec, N[n]>).  The trie is the codeword's own
//   mixed-radix digit tree, so the cost is O(D) per coded bit and never O(V).
//   Learning is plain SGD (NO AdaGrad, NO momentum): the node row descends
//   eta_N * g * cvec, the bias descends eta_N * g, and the per-token gradient
//   accumulated over the codeword's bits descends into the embedding rows with
//   eta_E, clamped elementwise at +-0.5.  The single output is ONE layer-0
//   mixer input carrying the raw logit (0.0 on every bit the expert cannot
//   speak about, so a quiescent bit is exactly the neutral input).
//
// ---------------------------------------------------------------------------
// WHERE OUR PORT DIFFERS FROM the zmix instrument (declared, as zmix declares
// its own divergences from wordlbl.c):
//   A. CODEWORD GEOMETRY.  zmix parses three bands (1 byte 0x80..0xCF,
//      2 bytes, 3 bytes with b1 in 0xD0..0xEF), max rank 44,879.  OUR
//      dictionary has 54,857 entries and our encoder assigns a FOURTH band;
//      the parser here follows decodeCodeWord() in src/models/fxcmv1.cpp
//      band for band.
//   B. exp.  zmix calls its vendored f64 exp and casts to f32.  We call
//      std::exp(double) from a translation unit with strict FP semantics;
//      src/detmath.cpp supplies the symbol.
//   C. calloc instead of alloc+memset for the node table.  zmix allocates and
//      then memsets to zero, which makes all 48 MiB resident before the first
//      byte; calloc'd pages are already zero and stay lazy until touched.
//      Identical values, smaller working set.
//   D. The embedding tables are seeded from the SAME fixed-seed xorshift in
//      the SAME draw order as the instrument, so the initial state is a pure
//      function of D and VMAX -- no entropy, no clock, no file.  It is one
//      more consumer of the one fixed pseudo-random stream of this program,
//      and its draw count moves with D and VMAX, which is worth knowing when
//      either is changed.
//
// ---------------------------------------------------------------------------
// MEMORY: with D = 96 the node table is 2^NODEBITS * D * 4 B (48 MiB at 17
// bits), the bias table 2^NODEBITS * 4 B, and the embeddings VMAX * D * 4 B
// per table; about 109 MiB in the v1 form (three tables) or about 69 MiB
// with the tied-decay context (CTX > 0).
// ---------------------------------------------------------------------------
#ifndef WORD_LABEL_H_
#define WORD_LABEL_H_

static_assert(96 > 0 && (96 % 8) == 0,
    "must be a positive multiple of 8 (the elementwise loops "
    "are written so the compiler can vectorise them without a scalar tail)");
static_assert(17 >= 15 && 17 <= 24,
    "must be in 15..24: below 15 the byte-2 hash range "
    "(2^bits - 1 - 16640) would vanish, above 24 the node table exceeds 6 GiB");
static_assert(54857 > 0, "must be positive");
static_assert(60 > 0 && 30 > 0,
    "learning rates must be positive");
static_assert(16 >= 0 && 16 <= 64,
    "must be in 0..64 (0 = the untied three-table form)");
static_assert(16 == 0 ||
    (850 > 0 && 850 < 1000),
    "a context depth > 0 needs a decay in 1..999 per mille");

namespace word_label {

inline constexpr bool Enabled() { return true; }

struct State;

// Cold: allocate and seed.  Aborts loudly on an allocation failure rather than
// running with a null table.
State* Create();
void Destroy(State* s);

// Hot: the layer-0 input for THIS coded bit.  Returns the raw logit on the
// payload bits (bits 6..0) of a WRT codeword byte and EXACTLY 0.0f on every
// other bit -- so on a quiescent bit the slot carries the same neutral 0.0f a
// build without the model would leave there.  Call once per bit, before Mix().
float Forward(State* s);

// Hot: train on the observed bit and advance the causal codeword parse.  Call
// once per bit, after the bit is known -- in Predictor::Perceive(), which is
// the one place both the encoder and the decoder pass through for every coded
// bit (and, via ObserveKnownBit, for every structurally determined one).
void Perceive(State* s, int bit);

}

#endif
