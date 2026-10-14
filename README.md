# Ex-Girlfriend 1.0

The project is named after **“Ex-Girlfriend,” the No Doubt song**.

Ex-Girlfriend is Ryan Landay's modification of **cmix-neif 1.0**, by Marcus Wölk,
for lossless compression of the 1,000,000,000-byte **enwik9** benchmark.
The verified self-extracting archive is **91,778,237 bytes**, including the
decoder, model weights and required assets. It is **135,106 bytes smaller
(0.146993%)** than cmix-neif 1.0's 91,913,343-byte archive.

**Private review candidate, October 14, 2026.** This repository and its draft
release have not been published or submitted. The verified release artifact
is identified as **FINAL** in the experiment evidence. The [LTCB table](https://mattmahoney.net/dc/text.html)
checked on October 14 still lists cmix-neif first; our result would improve
that published size, subject to the benchmark maintainer's verification.
This is an LTCB submission candidate, not a claim to a Hutter Prize award.

## Verified result and computational cost

| Measurement | Compression | Fresh decompression |
|---|---:|---:|
| Wall time | 220,071.89 s / 61.13 h | 241,636.66 s / 67.12 h |
| User CPU time | 206,994.02 s | 227,764.75 s |
| System CPU time | 12,422.59 s | 13,807.27 s |
| Peak RSS | 10,204.5 MiB / 9.965 GiB | 10,194.0 MiB / 9.955 GiB |
| Exit status | 0 | 0 |

Measured on an **AMD Ryzen Threadripper 3970X (Zen 2), Linux x86-64**, with one
CPU core assigned to each codec run, while the server also ran other workloads.
Competition for memory bandwidth, disk access and other shared resources can
affect wall time. These observations are not controlled estimates of the overhead of each
individual feature, nor a prediction of another machine's runtime. No GPU is
used for compression or decompression.

The full fresh decode completed on October 14 around **07:01 UTC**. Its output
was exactly 1,000,000,000 bytes and passed `cmp` against the original enwik9.
A separate SHA-256 check of both complete files also agreed. The previous
interrupted decode is not the verification used here. See the retained
[encode log](verification/encode.log), [decode log](verification/decode.log)
and [machine-readable receipt](verification/result.json).

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `archive9` | 91,778,237 | `a85629443ad1afabf36c88fb1f2dc1f3bcdaf8f19ce3f31d0948e93bb0128a6c` |
| `comp9` | 3,995,122 | `952ce43a221b19ea204e0f2dcf8afd86dbf8e85f823eb5f0949dd61ad1b87e67` |
| Original/restored `enwik9` | 1,000,000,000 | `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc` |

The release-tree rebuild produced the exact measured `comp9`. Runtime and
constructor inputs in this repository are preserved from that checked tree;
new release documentation does not change them. [Source and build evidence](verification/source_identity.json)
records this distinction. A static x86-64-v3 portability review passed; no
separate full Intel decode is claimed.

## Provenance: what we inherited

The immediate base is [Marcus Wölk's cmix-neif 1.0](https://github.com/marwoelk/cmix-neif-1.0),
pinned at commit [`1c9ea5ed57ec9718bcb681d20b79e81a60b7a728`](https://github.com/marwoelk/cmix-neif-1.0/tree/1c9ea5ed57ec9718bcb681d20b79e81a60b7a728).
This is a derivative compressor, not a new implementation of the whole system.
We retain neif's transformer architecture, its trained transformer and BITLSTM
weights, dictionary, article order, prompt machinery, PPMd, classical predictors,
arithmetic coding and custom executable packer. The metadata changes below alter
two reversible representations/forecasts; most preprocessing is inherited.

Important parts of that ancestry are:

| Component | Attribution and relationship |
|---|---|
| cmix, cmix-hp, fast-cmix, fx-cmix/fx2-cmix, cmix-lex | Byron Knoll, Saurabh Kumar, Kaido Orav and Ibrahim Marcouch; inherited context mixing, models, data structures and compressor base |
| fxcm text model and PAQ/PPMd lineage | Kaido Orav and the contributors named in `NOTICE.md`, including Matt Mahoney, Alexander Rhatushnyak, Przemysław Skibiński, Dmitry Shkarin and Eugene Shelwien |
| Transformer inference engine, layer pattern and training ancestry | Vladimir Ivanov's fx2-cmix-transformer, adapted by Marcus Wölk; the KDA design also builds on Kimi Linear and flash-linear-attention |
| Shipped transformer and BITLSTM weights | Trained by Marcus Wölk for cmix-neif on enwik9; reused unchanged here, not trained by this release's authors |
| BITLSTM correction model and byte-LSTM PPMd-prior mechanism | David Freelan's cmix-obias, as integrated and trained by neif |
| Adaptive mixer depth, end-to-end mixer gradient, word-label model and indel recovery | James Byrne's zmix, already ported into neif; not new contributions of Ex-Girlfriend |
| Article-order/preprocessing ancestry | STARLIT by Artemiy Margaritov, phda9 by Alexander Rhatushnyak, and subsequent fx2-cmix/cmix-lex/neif work |
| Executable packer | Neif's custom context-mixing packer, drawing on PAQ and BCJ/Crinkler ideas; this release does not replace it with UPX |

This abbreviated table does not replace the full [inherited notices](NOTICE.md),
[training attribution](training/ATTRIBUTION.md), [third-party licenses](third_party/README.md)
or [upstream README](README.cmix-neif.md). Those documents distinguish copied
code, rewritten implementations and ideas. The dictionary-ordering pipeline
used external embedding models offline; their model weights do not ship.
Fixed upstream assets and their hashes are included for reproducibility.

The project changes were developed by **Ryan Landay with Claude (Anthropic)
and Codex (OpenAI)**. The credits below identify conceptual sources separately
from our implementation. We do not claim to have invented fast weights,
online adaptation, RMS preconditioning, larger context tables or interpolation.

## Changes, measured savings and costs

Savings below are positive numbers of bytes avoided. Prefix experiments are
separate from the full release result. In particular, the historical “160 MB”
input is a prepared-input prefix corresponding to about 100 MB of coder
stream; it is not a full enwik9 test. Payload savings exclude decoder fees.
The variants interact, and there is no full-corpus factorial that separates
every contribution. **Do not add the rows to predict full-corpus savings.**

| Change | Best retained measurement | Computational cost and limit |
|---|---|---|
| One-block online fast-weight auxiliary head plus direct final-mixer input (B: FW-DUAL/FWRH/FW1) | 9,047 payload bytes on the 160 MB screen, fresh decode passed. Full B encode: 46,079 complete-archive bytes below stock; B's full decode was interrupted. | 160 MB A1/B encode: 33,524.13 → 37,838.54 s wall (+12.87%); 9,864.5 → 9,865.6 MiB RSS. Shared-host measurement. No isolated neif FW1-only result. |
| Four feature blocks rather than one (F4) | Another 5,434 payload bytes versus B on the 160 MB encode. A 12 MB fresh decode passed; no separate F4-only 160 MB decode is claimed. | Four matrices occupy 2.25 MiB versus 0.5625 MiB for one. No controlled isolated incremental F4 runtime is available. |
| Larger classical text tables (W3) | 5,496 payload bytes versus B on the 160 MB encode; separate 12 MB round trip passed. | At 12 MB, peak RSS rises 6,026.1 → 9,735.1 MiB. At larger sizes, PPM paging masks much of the added allocation. No controlled W3 speedup is claimed. |
| Page-id forecast for old revision ids (E1) | 20,782 payload bytes on a matched 12 MB screen. | In that screen, wall time fell by 88.66 s (4.62%); integer interpolation can simplify the coded residual stream. |
| Post-migration timestamp slope (E3b), on top of E1 | Another 2,040 payload bytes on the same 12 MB screen. | +0.24 s wall and +0.5 MiB RSS in that pair; this is too small a timing difference to establish a meaningful slowdown. |
| Complete FINAL bundle | **135,106 complete-archive bytes** on full enwik9, fresh decode and exact comparison passed. | Measured release costs are the encode/decode table above. |

The full release saves **132,106 payload bytes** (87,896,203 → 87,764,097) and
**3,000 bytes of archive overhead**, for the 135,106-byte total. Relative to
the earlier B archive, FINAL saves 89,027 bytes, but that comparison combines
F4, W3, E1/E3b and package interactions. The overhead difference is not an
isolated “smaller code” experiment. The compressor is 5,567 bytes smaller
than stock; compressor size does not enter the LTCB self-extracting-archive
ranking. Detailed controls and their qualification are in [ABLATIONS.md](docs/ABLATIONS.md).

## Online fast-weight prediction

Ex-Girlfriend adds an online-learned predictor beside cmix-neif's frozen transformer. Both the compressor and decompressor reconstruct its state from the symbols already coded, so no additional trained weights are stored in the archive. The pretrained transformer and its original probability output remain intact; the mixers receive an additional prediction that can adapt to the file as coding proceeds.

The following excerpts are copied exactly from the release source. Source links and line numbers refer to this release.

### Sparse features from the last four MLP blocks

The build sets `PORT_FW_LAYERS` to four in [src/port_cfg.h:50](src/port_cfg.h#L50). The transformer has 16 blocks, so the added predictor reads the final four blocks' quantized ReLU² activations. It captures features only on rows whose output is used. Intermediate prompt rows that do not produce a read output must not overwrite the feature state needed for the next update.

[src/tf1/opt/model_opt.cpp:594–600](src/tf1/opt/model_opt.cpp#L594):

```cpp
#if PORT_FW
    // FW: the relu^2 code (raw u8 x static scale) of the last FWK MLPs, on
    // read rows only.  mlp_block(NL-1) runs on read rows only (an unread row
    // stops before the last layer); the earlier blocks also run on unread
    // rows, whose code must not replace the read row's (fw_gel).
    if (fw_gel && l >= NL - FWK)
      fw_apply(l - (NL - FWK), mm.up.m.d_out, mm.down_s_act);
```

Each block has a zero-initialized correction matrix with 192 output coordinates and space for 768 features. Only nonzero features are visited; each quantized activation is multiplied by its fixed activation scale. The first block clears the accumulated correction, and all four blocks add into it:

[src/tf1/opt/model_opt.cpp:615–630](src/tf1/opt/model_opt.cpp#L615):

```cpp
    for (int i = 0; i < w; i++) {
      if (q768[i] == 0) continue;
      const float hv = float(q768[i]) * s;
      idx[nnz] = uint16_t(i);
      h[nnz] = hv;
      nnz++;
    }
    fw_nnz[b] = nnz;
    if (b == 0)
      for (int j = 0; j < D; j++) fw_dx[j] = 0.0f;
    const float* const dWb = fw_dW.data() + size_t(b) * DMLP * D;
    for (int k = 0; k < nnz; k++) {
      const float* col = dWb + size_t(idx[k]) * D;
      const float hv = h[k];
      for (int j = 0; j < D; j++) fw_dx[j] += col[j] * hv;
    }
```

In mathematical terms, the accumulated correction is `δx = Σ_b ΔW_b h_b`. The four matrices persist across article boundaries. They consume 2,359,296 bytes (2.25 MiB) in total, plus statistics and other scratch storage. The correction is applied at the final readout; it is not inserted into the intermediate transformer blocks.

### A separate output head preserves the pretrained prediction

The additional head reads `x + δx` through the existing RMS normalization, activation quantization, frozen unembedding, logit soft cap and softmax. It uses separate buffers for its normalized state, logits and probabilities:

[src/tf1/opt/model_opt.cpp:635–648](src/tf1/opt/model_opt.cpp#L635):

```cpp
  void fw_head() {
    for (int i = 0; i < D; i++) fw_x2[i] = x[i] + fw_dx[i];
    rms_norm_quant192(fw_x2, fw_xn2, M.unembed.s_act, fw_q);
    float sx = 0.0f, sn = 0.0f;
    for (int i = 0; i < D; i++) {
      sx += fw_x2[i] * fw_x2[i];
      sn += fw_xn2[i] * fw_xn2[i];
      fw_xn[i] = fw_xn2[i];
    }
    fw_denom = sx > 0.0f ? std::sqrt(sn / sx) : 0.0f;
    qgemv_f32(M.unembed.m, fw_q, fw_logits);  // rows_padded 208
    head_softcap_softmax(fw_logits, fw_probs208);  // fw_logits -> 15 tanh(l/15)
    std::memcpy(fw_lc, fw_logits, sizeof(float) * V);
    std::memcpy(fw_p, fw_probs208, sizeof(float) * V);
```

The original head's probabilities are copied to its output before the additional head runs. Its original residual state `x` is only read by the additional head:

[src/tf1/opt/model_opt.cpp:865–869](src/tf1/opt/model_opt.cpp#L865):

```cpp
      head_softcap_softmax(logits, probs208);
      std::memcpy(probs_out, probs208, sizeof(float) * V);
#if PORT_FW
      fw_head();  // DUAL: the fast-weight distribution; probs_out is untouched
#endif
```

Keeping both predictions lets the context mixers learn how to use the adapted signal alongside the pretrained one. The shipped transformer parameters are not changed by this process.

### A normalized online update

After the corresponding stream target is known, the auxiliary predictor forms its own cross-entropy error. It differentiates the soft cap, applies a float reconstruction of the frozen unembedding, and uses the RMS-normalization backward rule:

[src/tf1/opt/model_opt.cpp:658–675](src/tf1/opt/model_opt.cpp#L658):

```cpp
    for (int o = 0; o < V; o++) {
      const float d = fw_p[o] - (o == int(y) ? 1.0f : 0.0f);
      const float c = fw_lc[o] * (1.0f / 15.0f);
      dl[o] = d * (1.0f - c * c);
    }
    float gxn[D];
    for (int i = 0; i < D; i++) gxn[i] = 0.0f;
    const float* U = M.unembed_f32.data();
    for (int o = 0; o < V; o++) {
      const float g = dl[o];
      const float* row = U + size_t(o) * D;
      for (int i = 0; i < D; i++) gxn[i] += row[i] * g;
    }
    float m = 0.0f;
    for (int i = 0; i < D; i++) m += gxn[i] * fw_xn[i];
    m *= (1.0f / float(D));
    float gx[D];
    for (int i = 0; i < D; i++) gx[i] = fw_denom * (gxn[i] - fw_xn[i] * m);
```

The forward head contains activation quantization. This backward calculation therefore supplies a straight-through-style **surrogate gradient**, not an exact derivative through that quantization. It updates only the correction matrices; it does not backpropagate into the pretrained transformer or through the final context-mixer/SSE/BITLSTM probability.

The step size begins at 0.006 and approaches 0.0015. Its clock counts eligible auxiliary-predictor updates:

[src/tf1/opt/model_opt.cpp:677–678](src/tf1/opt/model_opt.cpp#L677):

```cpp
    const double kap = 1500.0e-6 + (6000.0e-6 - 1500.0e-6) / (1.0 + double(fw_t) / 8000.0);
    fw_t++;
```

Each block tracks an exponential moving average of each feature's squared activation, with decay 0.999. Its mean feature RMS normalizes the per-feature scaling. The small positive denominator terms keep zero or very small features well behaved:

[src/tf1/opt/model_opt.cpp:683–690](src/tf1/opt/model_opt.cpp#L683):

```cpp
    const float fb = 999000.0e-6f;
    float hsh = 0.0f;
    for (int b = 0; b < FWK; b++) {
      float* const v = fw_v[b];
      const uint16_t* const idx = fw_idx[b];
      const float* const h = fw_h[b];
      float* const sh = fw_sh[b];
      const int nnz = fw_nnz[b];
```

[src/tf1/opt/model_opt.cpp:691–700](src/tf1/opt/model_opt.cpp#L691):

```cpp
      for (int i = 0; i < DMLP; i++) v[i] *= fb;
      for (int k = 0; k < nnz; k++) v[idx[k]] += (1.0f - fb) * h[k] * h[k];
      float msd = 0.0f;
      for (int i = 0; i < DMLP; i++) msd += std::sqrt(v[i]);
      msd *= 1.0f / float(DMLP);
      for (int k = 0; k < nnz; k++) {
        const float sk = msd / (std::sqrt(v[idx[k]]) + 1e-3f * msd + 1e-12f);
        sh[k] = sk * h[k];
        hsh += h[k] * sh[k];
      }
```

Here `sh` is the scaled feature vector. One normalized-LMS-style step is shared by all four blocks: its denominator is `1 + Σ_b h_b·sh_b`. Each eligible update writes the outer product of the output error direction and the scaled sparse features:

[src/tf1/opt/model_opt.cpp:702–713](src/tf1/opt/model_opt.cpp#L702):

```cpp
    const float step = float(kap) / (1.0f + hsh);
    for (int b = 0; b < FWK; b++) {
      const uint16_t* const idx = fw_idx[b];
      const float* const sh = fw_sh[b];
      const int nnz = fw_nnz[b];
      float* const dWb = fw_dW.data() + size_t(b) * DMLP * D;
      for (int k = 0; k < nnz; k++) {
        float* col = dWb + size_t(idx[k]) * D;
        const float c = step * sh[k];
        for (int j = 0; j < D; j++) col[j] -= c * gx[j];
      }
    }
```

This combines a persistent learned correction, per-feature RMS preconditioning and a decreasing step-size schedule. The matrices are initialized to zero, so they add no correction initially.

### Updates use only targets already coded

A read row records that its auxiliary prediction is waiting for a target:

[src/tf1/tf1_engine.cpp:277–280](src/tf1/tf1_engine.cpp#L277):

```cpp
    opt->step((uint8_t)tok, prior, out, ch);
#if PORT_FW
    if (gelesen) fw_have = true;
#endif
```

The target is the next stream symbol after any inserted prompt rows. A document's final read row can predict its closing symbol, even though that closing symbol is not itself fed into the transformer. At the beginning of the next stream step, the engine updates only when a pending read prediction exists **and** the slot actually used an auxiliary-head row:

[src/tf1/tf1_engine.cpp:469–478](src/tf1/tf1_engine.cpp#L469):

```cpp
  int token = s.byte_to_index[byte];
  if (token < 0) {
    Fail("byte 0x%02x is not in the frozen byte set", byte);
  }
#if PORT_FW
  // FW target: this stream token, if it was coded with an FW-head row.
  if (s.fw_have && s.fw_live && s.opt) s.opt->fw_update((uint8_t)token);
  s.fw_have = false;
  s.fw_live = false;
#endif
```

`fw_have` tracks the pending read prediction; `fw_live` distinguishes the auxiliary-head row from a PPMd fallback or a neutral/off slot. The two flags are cleared after this decision. Thus the algorithm updates on eligible predictions, not indiscriminately after every byte. Encoder and decoder know the same target when the update occurs.

### The same probability representation as the static slot

The auxiliary row passes through the same float16 round trip and `1e−6` probability floor as the static transformer row. If the engine is using a fallback row, that row is copied and auxiliary learning is disabled for the corresponding prediction:

[src/tf1/tf1_engine.cpp:731–742](src/tf1/tf1_engine.cpp#L731):

```cpp
  if (s.probs_from_tf && s.opt) {
    // the same float16 rounding and 1e-6 floor as the transformer row
    FloatsToHalves(s.opt->fw_probs(), s.fw_half.data(), 208);
    HalvesToFloats(s.fw_half.data(), s.fw_row.data(), 208);
    for (int i = 0; i < 208; ++i) {
      if (!(s.fw_row[i] >= 1e-6f)) s.fw_row[i] = 1e-6f;
    }
    s.fw_live = true;
  } else {
    memcpy(s.fw_row.data(), s.probs.data(), sizeof(float) * 208);
    s.fw_live = false;
  }
```

The predictor also neutralizes the auxiliary slot when the transformer is gated for metadata, alias feeding or timestamp skip paths; see [src/predictor.cpp:660–685](src/predictor.cpp#L660).

### The adapted signal reaches both context-mixer stages

A separate byte-slot object conditions the auxiliary distribution on the bits already coded in the current byte. Its prediction is appended as input 588 of the first mixer stage, increasing that stage from 588 to 589 inputs:

[src/predictor.cpp:427–433](src/predictor.cpp#L427):

```cpp
  const unsigned int fw_index = input_index++;
#if PORT_FW
  layers_[0].SetInput(fw_index, fw_slot_output);
#else
  layers_[0].SetZero(fw_index);
#endif
#endif
```

The final mixer receives both the static transformer input and the auxiliary input directly. This is the additional final-stage connection called FW1 in the source; the final stage grows from 27 to 28 inputs:

[src/predictor.cpp:466–476](src/predictor.cpp#L466):

```cpp
  layers_[1].SetStretchedInput(mixer_0_.size(), aux_aux_in);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1, layers_[0].Inputs()[byte_mixer_index]);
  layers_[1].SetStretchedInput(mixer_0_.size() + 2, layers_[0].Inputs()[tf1_slot_index]);
#if PORT_GEOM
#if PORT_FW
  // FW1: the FW row's stretched input also feeds the final mixer.
  layers_[1].SetStretchedInput(mixer_0_.size() + 3, layers_[0].Inputs()[fw_index]);
#else
  layers_[1].SetZero(mixer_0_.size() + 3);
#endif
#endif
```

The inherited adaptive-depth feature gather does not read the appended input, and BITLSTM keeps its existing input shape. These connections provide the additional expert to the mixers without replacing the original transformer prediction.

The general ideas have substantial prior art: [Schmidhuber's fast weights (1992)](https://people.idsia.ch/~juergen/fastweights/),
[Ba et al. (2016)](https://arxiv.org/abs/1610.06258), and
[Krause et al.'s dynamic evaluation (2018)](https://proceedings.mlr.press/v80/krause18a.html).
Our initial experiment explicitly cited [TTT-NTP (Ouyang, Cai and Hu, 2026)](https://arxiv.org/abs/2606.21803)
as motivation; this release uses a different auxiliary-head objective and is
not an implementation of that paper. The dual-head preservation, particular
preconditioner/schedule, causal wiring and four-block combination are project
adaptations, first explored in our zmix branch and then ported to neif.
No learned optimizer weights or code from those papers are shipped.

## Larger text-model tables (W3)

Context models associate past patterns with probabilities for the next bit. In a fixed-size hashed table, different patterns compete for space. Larger tables can retain more distinct state and reduce replacement or collision pressure, at the cost of memory. W3 changes capacities within the inherited text model; it does not introduce a new predictor family.

`ContextMap3::Init` receives the same requested `m1`, but FINAL multiplies it by four instead of two. Thus each affected ContextMap3 table is twice its upstream size. The bucket format and surrounding prediction/update code remain inherited.

Source: [`src/models/fxcmv1.cpp:1439–1447`](src/models/fxcmv1.cpp#L1439-L1447).

```cpp
    void __attribute__ ((noinline)) Init(U32 m1, int c1, int s2, int s3,const U8 *nn1,int cs4,int k, const int u,short *st
    ) {
        C=min(c1, MAXCXT);
#if PORT_BIGTAB_CM3
        int m=m1*4;   // W3: every ContextMap3 table x2 (neif m1*2)
#else
        int m=m1*2;
#endif
        tmask=((m>>7)-1);
```

The five direct-state-map capacities also each gain one address bit, doubling their number of entries. The inactive `#else` preserves the upstream widths for comparison; FINAL enables the first branch through `PORT_BIGTAB_DSM`.

Source: [`src/models/fxcmv1.cpp:4066–4079`](src/models/fxcmv1.cpp#L4066-L4079).

```cpp
#if PORT_BIGTAB_DSM
    // W3: every direct state map +1 bit (neif 28/29/24/27/26).
    dcsm.Init(29,dccount-1, &STA7[0][0] );
    dcsm0.Init(30,dccount, &STA7[0][0] );
    dcsm1.Init(25,2, &STA7[0][0] );
    dcsm2.Init(28,3, &STA7[0][0] );
    dcsmN.Init(27,3, &STA7[0][0] );
#else
    dcsm.Init(28,dccount-1, &STA7[0][0] );
    dcsm0.Init(29,dccount, &STA7[0][0] );
    dcsm1.Init(24,2, &STA7[0][0] );
    dcsm2.Init(27,3, &STA7[0][0] );
    dcsmN.Init(26,3, &STA7[0][0] );
#endif
```

Both switches inherit `PORT_BIGTAB=1` in `src/port_cfg.h:53–68`.

The matched 160 MB payload comparison is 18,295,321 → 18,289,825 bytes, a **5,496-byte saving**. That screen was encode-only; the W3 12 MB round trip passed, and the combined FINAL release subsequently passed a full-corpus round trip. A standalone full-corpus W3 gain has not been measured. Also, total RSS does not simply double: neif's PPM page pool adjusts to process memory, so larger text tables can displace PPM frames and increase paging. These capacity tradeoffs build on the inherited models and the measurements described in neif's README §3.3 and §4.8.

## Revision IDs as interpolation residuals (E1)

In the early revision-ID group of this corpus, page IDs and revision IDs are strongly ordered together. Storing a small prediction error is cheaper than storing the whole revision ID. E1 applies this only below the fixed threshold; later IDs keep the original four-byte base-100 form. The threshold, initial forecast, and marker are explicit source constants and count as part of the implementation.

Source: [`src/b1_interp.h:25–27`](src/b1_interp.h#L25-L27).

```cpp
static const long long kB1Max = 17000000LL;      // B1 = revid below this
static const long long kLeer = 15898943LL;       // forecast of an empty set
static const unsigned char kMark = 0x2b;         // '+': ends the page id delta of a B1 line
```

The map contains only records already processed in stream order. `lower_bound(p)` finds the first known page ID at or above the current page ID. If known neighbors exist on both sides, the code interpolates their revision IDs. With only one neighbor it extrapolates at unit slope; with no history it uses the fixed initial value. A record later in the stream is never consulted just because its page ID would be a closer neighbor. Integer floor division makes the forecast identical in the transform, inverse, and runtime metadata reader.

Source: [`src/b1_interp.h:35–51`](src/b1_interp.h#L35-L51).

```cpp
  long long Prognose(long long p) const {
    std::map<long long, long long>::const_iterator hi = m_.lower_bound(p);
    const bool h2 = hi != m_.end();
    const bool h1 = hi != m_.begin();
    if (h1 && h2) {
      std::map<long long, long long>::const_iterator lo = std::prev(hi);
      const long long q1 = lo->first, r1 = lo->second;
      const long long q2 = hi->first, r2 = hi->second;
      return r1 + ts_algebra::Fdiv((p - q1) * (r2 - r1), q2 - q1);
    }
    if (h1) {
      std::map<long long, long long>::const_iterator lo = std::prev(hi);
      return lo->second + (p - lo->first);
    }
    if (h2) return hi->second - (hi->first - p);
    return kLeer;
  }
```

The forward transform writes the existing page-ID delta, then `+` and the signed decimal revision-ID residual. The marker is distinct from the decimal digits, negative sign, separators and base-100 bytes used by this grammar. Only after computing the residual does it add the current pair to history. The alternative branch preserves the original representation for later revision IDs.

Source: [`src/metablock_form.h:236–248`](src/metablock_form.h#L236-L248).

```cpp
      out.insert(out.end(), p, t);                 // page id delta, decimal
#if NEIF_E1
      pid += LiesDelta(p, t);
      if (rv < b1_interp::kB1Max) {                // B1: '+' residual vs interpolation
        out.push_back(b1_interp::kMark);
        Dezimal(&out, rv - b1.Prognose(pid));
        b1.Add(pid, rv);
      } else {
        Fest(&out, rv, 4);
      }
#else
      Fest(&out, rv, 4);
#endif
```

The inverse parses the same marker and signed decimal residual, recreates the identical forecast from earlier records, adds the residual, and then updates history. The core reconstruction is below; the preceding parser at `src/metablock_form.h:326–337` produces `pid`, residual magnitude `res`, and sign `vz`. The complete inverse also restores the original decimal field and verifies the revision-ID ordering. This is a reversible representation change; no original ID is discarded.

Source: [`src/metablock_form.h:338–340`](src/metablock_form.h#L338-L340).

```cpp
        v = b1.Prognose(pid) + vz * res;
        if (v < 0) Fail("B1 revid negative");
        b1.Add(pid, v);
```

E1 also updates the identifier-exclusion model. That model can use the fact that absolute page IDs and revision IDs do not repeat. Residuals do repeat—zero is common—so treating a residual as another unused absolute revision ID would apply the wrong constraints. A valid early-group row still marks its page ID, but only later-group rows mark the original revision-ID exclusion table:

Source: [`src/id_exclude.h:278–288`](src/id_exclude.h#L278-L288).

```cpp
    if (good) {
      if (armed >= 64) {
        Revive();
        ++stat_mark;
        MarkId(id);
#if NEIF_E1
        if (!b1row) revt.Mark(revpfx);
#else
        revt.Mark(revpfx);
#endif
      }
```

The residual field has its own parser state, field 4, and receives no absolute-ID candidate exclusion:

Source: [`src/id_exclude.h:359–359`](src/id_exclude.h#L359-L359).

```cpp
  if (fld != 1) return;                    // (E1: no candidates in field 4)
```

At the end of the page-ID delta, the candidate alphabet now includes `+`. The existing free/total counts are scaled by the same marker share, preserving their ratio. Later revision IDs must be at least 17,000,000, so the first base-100 digit cannot be below 17; those impossible values receive zero free weight. These changes are part of the E1 arm, not separate measured improvements.

Source: [`src/id_exclude.h:378–385`](src/id_exclude.h#L378-L385).

```cpp
      AddCand((unsigned char)kB1Mark, w * kB1Share, u * kB1Share);
      {
        const double wa = w * (1.0 - kB1Share) / 100.0;
        const double ua = u * (1.0 - kB1Share) / 100.0;
        for (int d = 0; d < 100; ++d)
          AddCand((unsigned char)((unsigned)0x80 + (unsigned)d),
                  d < kB2FirstMin ? 0.0 : wa, ua);
      }
```

E1 saves **20,782 bytes** on the matched 12 MB fixture (1,653,282 → 1,632,500). With the same input, CPU 4 and trace instrumentation, wall time falls from 1,921.07 to 1,832.41 seconds in the measured pair; the transformed coder stream is 159,000 bytes shorter before arithmetic coding. The net speed result is not a claim that the map operations are free.

The local idea and reference inverse are documented in `e1_spec.py`, based on `explore2.py`/`explore6.py` in the Oct7 metadata analysis. They report about 78% exact forecasts and 91.9% within ±1 for the early group. The inherited neif preprocessing and exclusion mechanisms remain the foundation. Credit the new residual/interpolation treatment to Ryan Landay, developed with Claude and Codex, without presenting it as a new general interpolation algorithm.

## A revision-ID slope for timestamp forecasts (E3b)

Neif already supplies the context models with a timestamp forecast. E3b improves that forecast for the later revision-ID group. The next record's revision ID is already known from the decoded metadata, while its timestamp is not. E3b estimates how much timestamp changes per revision-ID increment using earlier records, then projects from the most recent known timestamp to the current revision ID.

It stores the last nine decoded later-group pairs, covering up to eight intervals. The arithmetic uses the oldest and newest available pair rather than an immediately adjacent pair, which can reduce the effect of short local irregularities. This is a causal forecast: only timestamps already decoded enter the ring. The exact computation and guards are:

Source: [`src/ts_progress.h:60–78`](src/ts_progress.h#L60-L78).

```cpp
  enum { kN = 9 };
  void Beobachte(long long rev, long long t) {
    rev_[pos_] = rev; t_[pos_] = t;
    pos_ = (pos_ + 1) % kN;
    if (n_ < kN) ++n_;
  }
  bool Prognose(long long rev, long long* f) const {
    if (n_ < 2) return false;
    const int neu = (pos_ + kN - 1) % kN;              // record k-1
    const int alt = (pos_ + kN - n_) % kN;             // record j
    const long long dr = rev_[neu] - rev_[alt];
    if (dr <= 0) return false;
    const long long drev = rev - rev_[neu];
    const long long dt = t_[neu] - t_[alt];
    const long long kLim = 1LL << 31;                  // product < 2^62
    if (drev >= kLim || drev <= -kLim || dt >= kLim || dt <= -kLim) return false;
    *f = t_[neu] + Fdiv(drev * dt, dr);
    return true;
  }
```

If fewer than two records exist, revision IDs do not advance over the baseline, or the checked product could exceed its conservative bound, the function returns false. The caller first obtains neif's original median-of-32-slopes forecast and replaces it only when the new forecast is available. The earlier revision-ID group therefore keeps the original forecast, and the later group retains a defined fallback:

Source: [`src/ts_progress.h:158–166`](src/ts_progress.h#L158-L166).

```cpp
  void Einspeisen(size_t kn) {
    block_.clear();
    letzte_prognose_ = sch_.Prognose(Rev(kn));
#if NEIF_E3B
    if (Oben(kn)) {
      long long f = 0;
      if (b2_.Prognose(Rev(kn), &f)) { letzte_prognose_ = f; ++st_.b2_basis; }
    }
#endif
```

The ring is updated only after a metadata record's timestamp has been decoded; the next forecast is prepared afterwards:

Source: [`src/ts_progress.h:106–112`](src/ts_progress.h#L106-L112).

```cpp
      sch_.Beobachte(Rev(k), t_k);
#if NEIF_E3B
      if (Oben(k)) b2_.Beobachte(Rev(k), t_k);
#endif
      ++st_.b_zeilen;
      const size_t kn = k + 1;
      if (kn < metablock::mk.RangZahl()) Einspeisen(kn);
```

This changes the predicted timestamp bytes supplied to the existing models, not the stored timestamp grammar. The ring holds nine pairs of integers; it needs no external model or table asset. Integer floor division and the same update order keep the encoder and decoder synchronized.

Adding E3b to E1 saves **another 2,040 bytes** on the 12 MB fixture, for a combined **22,822-byte saving** versus the original control (1,653,282 → 1,630,460). E13's decode and comparison passed. In that matched E1→E13 pair, wall time changes by +0.24 seconds and peak RSS by +0.5 MiB; this is effectively unresolved small overhead in a single pair, not a universal zero-cost claim. The 2,040 bytes are incremental on E1, not an E3b-alone result.

The local design study `e3b_curve.py` compared baseline spans of 4, 8 and 16 using neif's observed coding cost versus forecast error. Those proxy estimates guided the choice of span; the numbers above are actual coded-payload measurements. Credit the new slope choice to Ryan Landay with Claude and Codex while retaining attribution for neif's timestamp predictor and integer algebra.

## Build recipe used for the measured release

The shipped recipe retains link-time optimization but does not consume the original profile-guided optimization data in its `prof_use` or `cmix_aux` targets. The target name `prof_use` is historical: its current flags are what determine the build. This distinction matters when explaining reproduction and binary-size differences.

Source: [`makefile:78–82`](makefile#L78-L82).

```makefile
prof_use: CPPFLAGS_PART-THAT-CAN-BE-SLOW    +=
prof_use: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -flto
prof_use: CPPFLAGS_PART-THAT-IS-COLD        +=
prof_use: LFLAGS                            += -flto
prof_use: clean cmix
```

Source: [`makefile:172–177`](makefile#L172-L177).

```makefile
cmix_aux: CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -DAUX_BUILD=1
cmix_aux: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -flto -DAUX_BUILD=1
cmix_aux: CPPFLAGS_PART-THAT-IS-COLD        += -DAUX_BUILD=1
cmix_aux: LFLAGS                            += -flto
cmix_aux: clean cmix
	mv cmix cmix_aux
```

This is the recipe for the verified original FINAL binary, not evidence that disabling PGO alone saves a particular number of bytes or improves speed. No isolated PGO-only comparison supports such a claim here. The full measured release changes modeling and compilation together.

## Headroom and scope

The **banked margin is 135,106 bytes** over the pinned neif artifact. The
component measurements above explain why these changes were selected; they
do not prove how much further improvement each mechanism has left.
This release introduces **no offline GPU training** and reuses neif's shipped
weights. The upstream training effort belongs to neif; it is not our own
compute expenditure. We have no audited all-experiments CPU/GPU total for
developing this release, so none is claimed.

This sub-0.15% LTCB margin does not establish the Hutter Prize's required improvement
or hardware eligibility.

## Build and run

Use x86-64 Linux with AVX2/FMA (x86-64-v3). Exact reconstruction was checked
with Ubuntu clang/LLVM/lld **18.1.3-1ubuntu1** and GNU binutils **2.42**.
The inherited [install script](install.sh) specifies the tool packages and
Ubuntu snapshot; inspect it before using it, since it installs system packages.
After those dependencies are present, the build itself is offline and
unprivileged. See [BUILD.md](docs/BUILD.md) for the complete procedure and
verification commands. In brief, from an empty output directory:

```sh
bash /path/to/ex-girlfriend-source/build.sh
sha256sum comp9
# Must be 952ce43a221b19ea204e0f2dcf8afd86dbf8e85f823eb5f0949dd61ad1b87e67
./comp9 -e /path/to/enwik9 archive9
```

For the published-format self-extracting archive, copy only `archive9` to a
fresh writable directory, mark it executable, and run `./archive9` with no
arguments. It writes `enwik9_uncompressed`; compare that complete file with
the original enwik9. No external weights, dictionary, GPU, Python environment
or network access is needed to run the artifact. Allow at least **25 GB of
free working space** and room above the measured ~10 GiB process RSS for the
operating system; the inherited PPMd also uses a sparse paging file. Filesystem
cache and other applications are not included in the reported RSS.

The binary assets are attached to the **private draft v1.0 release**, rather
than stored in Git history. Until review and publication are complete, their
eventual public download links are not available to others.

## License and modification notice

The combined program and our modifications are distributed under
**GNU GPL version 3 only (`GPL-3.0-only`)**, retaining neif's license choice.
See [LICENSE](LICENSE). Copyright and license notices of inherited components
remain in force; MIT, BSD, LLVM-exception, zlib and other notices are preserved
in `third_party/` and the source. Wikipedia profile text retains its separate
documented data license; it is not relicensed as our software.

Modification dates: **September 24–October 10, 2026** for the project techniques
and neif port; **October 14, 2026** for this release documentation and evidence.
See [EX-GIRLFRIEND.md](EX-GIRLFRIEND.md) for the changed-file inventory.

**Inherited open provenance point for review:** upstream's
[ATTRIBUTION.md, section 7](training/ATTRIBUTION.md#7-open-points) expressly
leaves unresolved the licensing interpretation of Ivanov's CUDA-libdevice
sine/cosine transcription. It is present in the native compressed-weight
loader, not only optional training tools. We retain it unchanged and preserve
the disclosure; selecting GPL-3.0-only does not independently resolve that
question. Other inherited provenance gaps are also recorded there. This
private preparation does not assert that those open points have been cleared.

The [submission email](docs/SUBMISSION_EMAIL.md) is a draft for Ryan's review.
It has not been sent.
