// The word-identity distributed-representation expert.  Implementation in
// src/word_label.cpp, hook in src/predictor.cpp.
//
// ---------------------------------------------------------------------------
// ATTRIBUTION
//   Origin: zmix 1.0 (James Byrne), GPL-3.0-or-later, build option -Dwordlbl
//   with -Dwordlbl-ctx>0, the tied-decay variant (src/models/wordlbl.zig).
//   The MECHANISM and the swept hyper-parameters (D=96, eta_N=0.060,
//   eta_E=0.030, lambda=0.85) are taken from there; the C++ below is a port
//   of that Zig source and is stamped onto OUR dictionary geometry, not
//   theirs.  zmix in turn cites
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
//   embedding row E[r] of D floats; the context vector is the decayed sum of
//   the embeddings of the previous 16 words plus a topic average; a trie node
//   n owns a weight row N[n] plus a bias, and the probability of the next
//   coded bit of the current WRT codeword is squash(bias[n] + <cvec, N[n]>).
//   The trie is the codeword's own mixed-radix digit tree, so the cost is
//   O(D) per coded bit and never O(V).  Learning is plain SGD (NO AdaGrad, NO
//   momentum): the node row descends eta_N * g * cvec, the bias descends
//   eta_N * g, and the per-token gradient accumulated over the codeword's bits
//   descends into the embedding rows with eta_E, clamped elementwise at
//   +-0.5.  The single output is ONE layer-0 mixer input carrying the raw
//   logit (0.0 on every bit the expert cannot speak about, so a quiescent bit
//   is exactly the neutral input).
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
//   C. ONE calloc for the state and all tables instead of alloc+memset per
//      table.  zmix allocates and then memsets to zero, which makes the
//      48 MiB node table resident before the first byte; calloc'd pages are
//      already zero and stay lazy until touched.  Identical values, smaller
//      working set.
//   D. The embedding table is seeded from the SAME fixed-seed xorshift in
//      the SAME draw order as the instrument, so the initial state is a pure
//      function of D and VMAX -- no entropy, no clock, no file.
//
// ---------------------------------------------------------------------------
// FIXED FORM: the tied-decay context over 16 words, D = 96, 2^17 trie nodes,
// 54,857 embedding rows, eta_N 0.060 (also for the byte-0 nodes), eta_E
// 0.030, lambda 0.85; constants of this unit, not build options.
// MEMORY: one block of about 69 MiB (node rows 48 MiB, embeddings 20.1 MiB,
// biases 0.5 MiB, the state).
#ifndef WORD_LABEL_H_
#define WORD_LABEL_H_
#include "port_cfg.h"

namespace word_label {

// Number of embedding rows.  Must cover every rank our encoder can assign,
// i.e. the dictionary size; the tie (and the band plan) is checked in
// src/models/fxcmv1.cpp next to decodeCodeWord(), the file that sees both.
constexpr int kVMax = 54857;

// Opaque; the state lives entirely inside src/word_label.cpp, so
// predictor.h only ever holds a pointer and its layout does not depend on the
// model geometry.
struct State;

// Cold: allocate and seed.  Aborts loudly on an allocation failure rather than
// running with a null table.  Neu() frees the old state first (the resets at
// the metablock end and at the end of slot 2).
State* Create();
State* Neu(State* alt);

// Hot: the layer-0 input for THIS coded bit.  Returns the raw logit on the
// payload bits (bits 6..0) of a WRT codeword byte and EXACTLY 0.0f on every
// other bit -- so on a quiescent bit the slot carries a neutral 0.0f.  Call
// once per bit, before Mix().
float Forward(State* s);

// Hot: train on the observed bit and advance the causal codeword parse.  Call
// once per bit, after the bit is known -- in Predictor::Perceive(), which is
// the one place both the encoder and the decoder pass through for every coded
// bit (and, via ObserveKnownBit, for every structurally determined one).
void Perceive(State* s, int bit);
#if PORT_CMX_MEAS
// research WL-MIXGRAD (src/cmx_meas.h): Perceive with the node gradient g + extra (extra = 0: Perceive exactly)
void PerceiveG(State* s, int bit, float extra);
#endif

}  // namespace word_label

#endif
