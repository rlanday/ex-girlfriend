// replay_trainer.cpp: replays a trace of the BITLSTM's inputs through
// the model (bit_head.cpp) and trains it.  Modified for this entry in 2026.
//
// WHY THIS IS EXACT, not an approximation.  The model reads the predictor
// state plus the coded bit and returns a probability, and the predictor never
// sees that probability.  base_p, the stage-1 row, m1raw, the override flag
// and the coded bit are therefore IDENTICAL for every weight setting: one
// compression run records them once, and the model can be run and trained on
// that recording without running the compressor again.  The replayed model is
// the same source with the same floating-point flags as the compressor's.
//
// One run of this program is one pass over the trace in stream order.  The
// model learns while it reads, exactly as the coder would (truncated BPTT
// over windows of 1024 bits, Adam); REPLAY_LOAD names the weights to start
// from, REPLAY_DUMP where the weights are written at the end.  Inputs:
//   argv[1]               the trace (format in ../FORMAT.md)
//   argv[2]               number of bits to replay (0 = all)
//   argv[3]               bits from which the "tail" sums start (0 = all)
//   REPLAY_CHANNEL        the per-byte channel file (needed)
//   REPLAY_SEED           seed of the initial draw
//   REPLAY_LR1E6          learning rate x 1e6
//   REPLAY_LOAD           weight file to start from (warm start)
//   REPLAY_DUMP           weight file written at the end
// Diagnostics that were not used for the shipped weights: REPLAY_TRUST,
// REPLAY_TRUST_COL, REPLAY_WINDOW_PROFILE, REPLAY_FREEZE_AT, REPLAY_CTRL_*.
//
// Output: the model's net effect in BYTES on the replayed bits, negative =
// smaller, computed from the discretized probabilities the coder would use.
// It is an in-sample value of a model that learns while it reads, not an
// archive size.
//
// TRUST (REPLAY_TRUST=1): does the correction dampen or amplify the
// transformer?  With
//   bl = logit(base_p)      the mixer's final answer
//   tl = stage-1 column     the transformer's own answer (REPLAY_TRUST_COL,
//                           default the last column)
//   dh = logit(head_p) - bl the model's correction
// the model DAMPENS the transformer when dh points away from tl and
// AMPLIFIES it when it points towards it.  Split by gate zone, by the gate
// bit and by decile of the replayed bits.
//
// Build: build.sh.  Run: see ../tools/train_blob.sh.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bit_head.h"

namespace {

// One accounting cell of the trust evaluation.
struct Trust {
  unsigned long long n_damp = 0, n_amp = 0, n_flat = 0;
  double mass_damp = 0.0, mass_amp = 0.0;   // sum |dh|
  double gain_damp = 0.0, gain_amp = 0.0;   // rate effect in bytes
};

void TrustAdd(Trust& t, int damp, double adh, double gain) {
  if (damp < 0) { ++t.n_flat; return; }
  if (damp) { ++t.n_damp; t.mass_damp += adh; t.gain_damp += gain; }
  else      { ++t.n_amp;  t.mass_amp  += adh; t.gain_amp  += gain; }
}

void TrustPrint(const char* name, const Trust& t) {
  const double n = (double)(t.n_damp + t.n_amp);
  if (n <= 0.0) { std::printf("  trust %-16s bits=0\n", name); return; }
  std::printf("  trust %-16s bits=%llu damped=%.4f mass_d=%.4f "
              "gain_d_B=%+.1f gain_a_B=%+.1f\n",
              name, (unsigned long long)n, (double)t.n_damp / n,
              t.mass_damp / (t.mass_damp + t.mass_amp + 1e-30),
              t.gain_damp, t.gain_amp);
}

constexpr int kMaxZone = 8;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: replay_trainer <trace> [maxbits] [frombits]\n");
    return 2;
  }
  const unsigned long long maxbits =
      (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 0ULL;
  const unsigned long long frombits =
      (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 0ULL;
  FILE* f = std::fopen(argv[1], "rb");
  if (f == nullptr) { std::perror("open trace"); return 1; }
  unsigned int hdr[4];
  if (std::fread(hdr, sizeof(hdr), 1, f) != 1) {
    std::fprintf(stderr, "replay: short header\n"); return 1;
  }
  if (hdr[0] != 0x4c544942u || (hdr[1] != 2u && hdr[1] != 3u)) {
    std::fprintf(stderr, "replay: bad magic/version\n"); return 1;
  }
  const int n1 = (int)hdr[2];
  const int rec = (int)hdr[3];
  // A window list follows the header (empty in the trace of the shipped
  // weights).  The records of all windows are stored back to back; the model
  // is driven through them in one pass, so its weights carry across the gaps
  // exactly as they do in the run that recorded them.
  int nwin = 0;
  unsigned long long* wstart = nullptr;
  unsigned long long* wend = nullptr;
  {
    unsigned int nw;
    if (std::fread(&nw, 4, 1, f) != 1) {
      std::fprintf(stderr, "replay: short header\n"); return 1;
    }
    nwin = (int)nw;
    if (nwin > 0) {
      wstart = (unsigned long long*)std::malloc(sizeof(unsigned long long) * nwin);
      wend = (unsigned long long*)std::malloc(sizeof(unsigned long long) * nwin);
      for (int k = 0; k < nwin; ++k) {
        unsigned long long se[2];
        if (std::fread(se, sizeof(se), 1, f) != 1) {
          std::fprintf(stderr, "replay: short window list\n"); return 1;
        }
        wstart[k] = se[0]; wend[k] = se[1];
      }
    }
  }
  // Version 3: the trace carries only a SELECTION of the stage-1 columns; its
  // mask follows the window list.  The row is rebuilt here at FULL width n1
  // (the rest exactly 0), so every column number still means what it means in
  // the coder.  That is exact as long as the model drops the unrecorded
  // columns anyway, which is why the mask is printed first.
  unsigned long long tcols = 0ULL;
  int ncol = n1;
  int* colidx = nullptr;
  if (hdr[1] == 3u) {
    if (std::fread(&tcols, sizeof(tcols), 1, f) != 1) {
      std::fprintf(stderr, "replay: short v3 column mask\n"); return 1;
    }
    colidx = (int*)std::malloc(sizeof(int) * (size_t)(n1 > 0 ? n1 : 1));
    ncol = 0;
    for (int j = 0; j < n1; ++j) {
      if (j >= 64 || ((tcols >> j) & 1ULL)) colidx[ncol++] = j;
    }
    std::printf("trace: v3, recorded columns %d of %d, mask=%llu\n",
                ncol, n1, (unsigned long long)tcols);
  }
  if (rec != 8 + 4 * ncol || n1 <= 0 || n1 > 4096) {
    std::fprintf(stderr, "replay: bad geometry n1=%d ncol=%d rec=%d\n",
                 n1, ncol, rec);
    return 1;
  }

  // The per-byte channel.  ONE record per byte: n_tf transformer scalars, then
  // n_hid state values, then n_lhid state values, then the gate zone.  The
  // model gets POINTERS into this buffer once, before Init(); the file is
  // advanced every eight bits, the rhythm in which the compressor rewrites
  // them.
  const char* auxpath = std::getenv("REPLAY_CHANNEL");
  FILE* fa = nullptr;
  int n_tf = 0, n_hid = 0, hid_pre = 0, n_lhid = 0, lhid_pre = 0, n_zone = 0;
  float* auxbuf = nullptr;
  size_t auxw = 0;
  static unsigned int zone_val = 0;
  if (auxpath != nullptr && *auxpath != '\0') {
    fa = std::fopen(auxpath, "rb");
    if (fa == nullptr) { std::perror("open channel"); return 1; }
    unsigned int hh[7];
    if (std::fread(hh, sizeof(unsigned int), 7, fa) != 7 ||
        hh[0] != 0x315a4142u) {           // "BAZ1"
      std::fprintf(stderr, "replay: bad channel file\n"); return 1;
    }
    n_tf = (int)hh[1]; n_hid = (int)hh[2]; hid_pre = (int)hh[3];
    n_lhid = (int)hh[4]; lhid_pre = (int)hh[5]; n_zone = (int)hh[6];
    auxw = (size_t)(n_tf + n_hid + n_lhid + (n_zone > 0 ? 1 : 0));
    auxbuf = (float*)std::calloc(auxw + 1, sizeof(float));
    std::printf("channel: n_tf=%d n_hid=%d (projected=%d) n_lhid=%d "
                "(projected=%d) zone=%d\n",
                n_tf, n_hid, hid_pre, n_lhid, lhid_pre, n_zone);
  }
  // A PRE-PROJECTED transformer-state block may only be replayed at ITS OWN
  // width.  The model sketches whenever its width is below the registered
  // one, and on an already sketched vector that would be a sketch OF A SKETCH:
  // a different feature set, not the k-prefix.  (The byte-LSTM block carries
  // the flag through SetLhid and may take its prefix.)
  if (hid_pre && n_hid != bit_head::kHidSketch && bit_head::kHidSketch != 0) {
    std::fprintf(stderr,
                 "replay: the channel carries a %d-wide sketch of the "
                 "transformer state, the model reads %d columns.  A sketch "
                 "of the sketch is not the %d-sketch; build the model with "
                 "%d columns or record the raw state.\n",
                 n_hid, (int)bit_head::kHidSketch, (int)bit_head::kHidSketch, n_hid);
    return 1;
  }
  if (n_tf > 0) bit_head::SetTf(auxbuf, n_tf);
  if (n_hid > 0) bit_head::SetAux(auxbuf + n_tf, n_hid);
  if (n_lhid > 0) bit_head::SetLhid(auxbuf + n_tf + n_hid, n_lhid, lhid_pre);
  if (n_zone > 0) bit_head::SetZone(&zone_val);
  if (n_zone <= 0) {
    std::fprintf(stderr, "replay: the channel has no gate zone; without it "
                 "the metadata block cannot be skipped.  Stop.\n");
    return 2;
  }

  const char* profpath = std::getenv("REPLAY_WINDOW_PROFILE");
  double* prof_b = nullptr; double* prof_h = nullptr;
  unsigned long long* prof_n = nullptr;
  int profw = 0;

  const bool trust = std::getenv("REPLAY_TRUST") != nullptr &&
                     *std::getenv("REPLAY_TRUST") != '\0';
  // WHICH column is the transformer's own answer: by default the last one,
  // REPLAY_TRUST_COL names another.
  int tcol = n1 - 1;
  {
    const char* tc = std::getenv("REPLAY_TRUST_COL");
    if (tc != nullptr && *tc != '\0') tcol = (int)std::strtol(tc, nullptr, 10);
    if (tcol < 0 || tcol >= n1) tcol = n1 - 1;
  }
  if (trust) std::printf("trust: transformer column %d of %d\n", tcol, n1);

  if (!bit_head::Init()) {
    std::fprintf(stderr, "replay: model disabled\n");
    return 1;
  }

  if (profpath != nullptr && *profpath != '\0') {
    profw = bit_head::Wlen();
    if (profw > 0 && profw <= (1 << 20)) {
      prof_b = (double*)std::calloc((size_t)profw, sizeof(double));
      prof_h = (double*)std::calloc((size_t)profw, sizeof(double));
      prof_n = (unsigned long long*)std::calloc((size_t)profw,
                                                sizeof(unsigned long long));
    } else {
      profw = 0;
    }
  }

  const int kBatch = 65536;
  unsigned char* buf = (unsigned char*)std::malloc((size_t)kBatch * rec);
  float* stage1 = (float*)std::calloc((size_t)n1, sizeof(float));
  // Costs in bits, from the discretized probabilities.
  double cost_base = 0.0, cost_head = 0.0;
  double tail_base = 0.0, tail_head = 0.0;
  unsigned long long nbits = 0, ntail = 0;
  if (fa == nullptr) {
    std::fprintf(stderr, "replay: no channel file (REPLAY_CHANNEL); the "
                 "metadata block cannot be recognised.  Stop.\n");
    return 2;
  }
  // per-zone accounting of the tail
  double zb[kMaxZone] = {0.0}, zh[kMaxZone] = {0.0};
  unsigned long long zn[kMaxZone] = {0};
  double* wb = nullptr; double* wh = nullptr; unsigned long long* wn = nullptr;
  if (nwin > 0) {
    wb = (double*)std::calloc((size_t)nwin, sizeof(double));
    wh = (double*)std::calloc((size_t)nwin, sizeof(double));
    wn = (unsigned long long*)std::calloc((size_t)nwin, sizeof(unsigned long long));
  }
  int wcur = 0;                 // records arrive window by window, in order
  unsigned long long wleft = (nwin > 0) ? (wend[0] - wstart[0]) * 8ULL : 0ULL;
  Trust tr_all, tr_gate[2], tr_zone[kMaxZone], tr_dec[10];
  for (;;) {
    const size_t got = std::fread(buf, (size_t)rec, (size_t)kBatch, f);
    if (got == 0) break;
    for (size_t k = 0; k < got; ++k) {
      const unsigned char* r = buf + k * (size_t)rec;
      unsigned short bp16;
      std::memcpy(&bp16, r, 2);
      const unsigned int base_p = bp16;
      const int ovr = (r[2] & 1);
      const int bit = (r[2] >> 1) & 1;
      float m1raw;
      std::memcpy(&m1raw, r + 4, 4);
      if (colidx != nullptr) {
        // stage1 was zeroed once; only the recorded columns are overwritten
        // per record, the rest stays 0.
        for (int q = 0; q < ncol; ++q) {
          std::memcpy(stage1 + colidx[q], r + 8 + 4 * q, 4);
        }
      } else {
        std::memcpy(stage1, r + 8, sizeof(float) * (size_t)n1);
      }
      if (fa != nullptr && (nbits & 7ULL) == 0ULL) {
        if (std::fread(auxbuf, sizeof(float), auxw, fa) != auxw) {
          std::fprintf(stderr, "replay: channel ran out at bit %llu\n",
                       nbits);
          return 1;
        }
        if (n_zone > 0) {
          const float zv = auxbuf[auxw - 1];
          zone_val = (zv >= 0.0f) ? (unsigned int)(zv + 0.5f) : 0u;
        }
      }
      // Inside the metadata block (gate zone 3) the coder runs the model
      // neither forward nor backward; the bit costs base_p.  The replay does
      // the same, so it trains the model in the form in which it runs.
      const bool meta_skip = (zone_val == 3u);
      const unsigned int q = meta_skip
          ? base_p : bit_head::Adjust(base_p, stage1, n1, m1raw, ovr);
      if (!meta_skip) bit_head::Observe(bit);
      const double pb = (double)base_p / 65536.0;
      const double ph = (double)q / 65536.0;
      const double cb = -std::log2(bit ? pb : 1.0 - pb);
      const double ch = -std::log2(bit ? ph : 1.0 - ph);
      cost_base += cb;
      cost_head += ch;
      const int zi = (n_zone > 0 && zone_val < (unsigned int)kMaxZone)
                         ? (int)zone_val : 0;
      if (trust && !ovr) {
        // tl = the stretched probability of the transformer's mixer slot.
        const double tl = stage1[tcol];
        const double bl = std::log(pb / (1.0 - pb));
        const double hl = std::log(ph / (1.0 - ph));
        const double dh = hl - bl;
        const double dt = tl - bl;
        int damp = -1;
        if (std::fabs(dh) > 1e-9 && std::fabs(dt) > 1e-9) {
          damp = ((dh > 0.0) == (dt > 0.0)) ? 0 : 1;
        }
        const double gain = (ch - cb) / 8.0;   // bytes, negative = better
        const double adh = std::fabs(dh);
        TrustAdd(tr_all, damp, adh, gain);
        if (fa != nullptr && n_tf >= 1) {
          TrustAdd(tr_gate[auxbuf[0] >= 0.5f ? 1 : 0], damp, adh, gain);
        }
        if (n_zone > 0) TrustAdd(tr_zone[zi], damp, adh, gain);
        if (maxbits != 0ULL) {
          TrustAdd(tr_dec[(int)((nbits * 10ULL) / maxbits) % 10], damp, adh,
                   gain);
        }
      }
      if (profw > 0) {
        const int pos = (int)(nbits % (unsigned long long)profw);
        prof_b[pos] += cb; prof_h[pos] += ch; ++prof_n[pos];
      }
      if (nbits >= frombits) {
        tail_base += cb; tail_head += ch; ++ntail;
        if (n_zone > 0) { zb[zi] += cb; zh[zi] += ch; ++zn[zi]; }
      }
      if (nwin > 0 && wcur < nwin) {
        wb[wcur] += cb; wh[wcur] += ch; ++wn[wcur];
        if (--wleft == 0 && wcur + 1 < nwin) {
          ++wcur;
          wleft = (wend[wcur] - wstart[wcur]) * 8ULL;
        }
      }
      ++nbits;
      if (maxbits != 0ULL && nbits >= maxbits) { k = got; break; }
    }
    if (maxbits != 0ULL && nbits >= maxbits) break;
  }
  std::fclose(f);
  std::printf("bits=%llu cells=%d inputs=%d params=%d "
              "base_B=%.1f head_B=%.1f delta_B=%+.1f rel=%+.6f%% "
              "tailbits=%llu tail_delta_B=%+.1f\n",
              nbits, bit_head::Cells(), bit_head::NumInputs(),
              bit_head::NumParams(),
              cost_base / 8.0, cost_head / 8.0,
              (cost_head - cost_base) / 8.0,
              100.0 * (cost_head - cost_base) / cost_base,
              ntail, (tail_head - tail_base) / 8.0);
  for (int k = 0; k < nwin; ++k) {
    std::printf("  window %d start=%llu len=%llu bits=%llu delta_B=%+.1f\n",
                k, wstart[k], wend[k] - wstart[k], wn[k], (wh[k] - wb[k]) / 8.0);
  }
  if (n_zone > 0) {
    for (int k = 0; k < kMaxZone; ++k) {
      if (zn[k] == 0) continue;
      std::printf("  zone %d tailbits=%llu tail_delta_B=%+.1f\n", k, zn[k],
                  (zh[k] - zb[k]) / 8.0);
    }
  }
  if (trust) {
    TrustPrint("all", tr_all);
    if (fa != nullptr && n_tf >= 1) {
      TrustPrint("gate=0(prior)", tr_gate[0]);
      TrustPrint("gate=1(tf)", tr_gate[1]);
    }
    if (n_zone > 0) {
      for (int k = 0; k < kMaxZone; ++k) {
        char nm[24];
        std::snprintf(nm, sizeof(nm), "zone%d", k);
        if (tr_zone[k].n_damp + tr_zone[k].n_amp + tr_zone[k].n_flat > 0) {
          TrustPrint(nm, tr_zone[k]);
        }
      }
    }
    if (maxbits != 0ULL) {
      for (int k = 0; k < 10; ++k) {
        char nm[24];
        std::snprintf(nm, sizeof(nm), "decile%d", k);
        TrustPrint(nm, tr_dec[k]);
      }
    }
  }
  if (profw > 0) {
    FILE* pf = std::fopen(profpath, "w");
    if (pf != nullptr) {
      std::fprintf(pf, "# pos base_bits head_bits n delta_B\n");
      for (int k = 0; k < profw; ++k) {
        std::fprintf(pf, "%d %.6f %.6f %llu %+.6f\n", k, prof_b[k], prof_h[k],
                     prof_n[k], (prof_h[k] - prof_b[k]) / 8.0);
      }
      std::fclose(pf);
    }
  }
  bit_head::Shutdown();
  return 0;
}
