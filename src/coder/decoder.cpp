#include "../invariants.h"
#include "decoder.h"
__attribute__((cold)) Decoder::Decoder(FILE* is, Predictor* p) : is_(is), x1_(0),
    x2_(0xffffffff), x_(0), p_(p) {
  // Identical registration to Encoder::Encoder: same pointers, same
  // widths, same order, and before Init().  See there.
  bit_head::SetAux(p_->bitl_hid_, p_->bitl_n_hid_);
  bit_head::SetTf(p_->bitl_tf_, p_->bitl_n_tf_);
  bit_head::SetZone(p_->bitl_zone_);
  bit_head::Init();
  for (int i = 0; i < 4; ++i) {
    x_ = (x_ << 8) + (ReadByte() & 0xff);
  }
}

// Past the end of the archive the coder reads 0.  The form `c == EOF ? 0 : c`
// keeps the control flow the PGO profile of Decode was recorded with.
int Decoder::ReadByte() {
  const int c = getc(is_);
  return c == EOF ? 0 : c;
}

unsigned int Decoder::Discretize(float p) {
  return 1 + 65534 * p;
}

int Decoder::Decode() {
  {
    const int forced = p_->ChrForced();
    if (forced) {
      p_->Predict();
      const int bit = (forced > 0) ? 1 : 0;
      p_->Perceive(bit);
      return bit;
    }
  }
  unsigned int p = Discretize(p_->Predict());
  int bb_bitl = 0;
  if (bit_head::Enabled()
      && !bit_head::AuxOff()
      && !bit_head::MetaOff()
#if PORT_CMX_MEAS
      && !p_->AdepthLearnOff()   // research CMX_ADEPTH_MODE: = AdepthSkipped() with the knob unset (Encoder::Encode)
#else
      && !p_->AdepthSkipped()
#endif
      ) {
#if PORT_CMX_MEAS
    if (__builtin_expect(p_->AdmRun(), 0)) {
      (void)bit_head::Adjust(Discretize(p_->AdmCascadeP()), p_->bitl_stage1_, p_->bitl_n_stage1_,
                             p_->bitl_m1raw_, p_->bitl_override_);
    } else {
#endif
    p = bit_head::Adjust(p, p_->bitl_stage1_, p_->bitl_n_stage1_,
                          p_->bitl_m1raw_, p_->bitl_override_);
    bb_bitl = p_->bitl_override_ ? 0 : 1;
#if PORT_CMX_MEAS
    }
#endif
  }
  const unsigned int xmid = floor_stage::Mid(x1_, x2_, p_->SemCorrect(floor_.P32(p,
      p_->bitl_override_,
      p_->AdepthSkipped() ? 1 : 0,
      bb_bitl, p_->bb_z_, p_->bitl_m1raw_, bit_head::MetaOff() ? 1 : 0,
      (31 - __builtin_clz(p_->BbBitCtx() | 1u)) & 7)));
  int bit = 0;
  if (x_ <= xmid) {
    bit = 1;
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }
  floor_.Update(bit);
  p_->SemLearn(bit);
  if (bit_head::Enabled()
      && !bit_head::AuxOff()
      && !bit_head::MetaOff()
#if PORT_CMX_MEAS
      && !p_->AdepthLearnOff()
#else
      && !p_->AdepthSkipped()
#endif
      ) bit_head::Observe(bit);
  p_->Perceive(bit);

  while (((x1_^x2_) & 0xff000000) == 0) {
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
    x_ = (x_ << 8) + ReadByte();
  }
  return bit;
}

int Decoder::DecodeRawBit(unsigned int p) {
  if (p < 1) p = 1;
  if (p > 65534) p = 65534;
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  int bit = 0;
  if (x_ <= xmid) {
    bit = 1;
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }

  while (((x1_^x2_) & 0xff000000) == 0) {
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
    x_ = (x_ << 8) + ReadByte();
  }
  return bit;
}

void Decoder::ObserveKnownByte(unsigned int byte) {
  invariants::g_injected_byte = 1;
  for (int bit = 7; bit >= 0; --bit) {
    ObserveKnownBit((byte >> bit) & 1);
  }
  invariants::g_injected_byte = 0;
}
