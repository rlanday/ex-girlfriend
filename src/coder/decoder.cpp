#include "decoder.h"

Decoder::Decoder(std::ifstream* is, Predictor* p) : is_(is), x1_(0),
    x2_(0xffffffff), x_(0), p_(p) {
  // Identical registration to Encoder::Encoder: same pointers, same
  // widths, same order, and before Init().  See there.
  bit_head::SetAux(p_->bitl_hid_, p_->bitl_n_hid_);
  bit_head::SetTf(p_->bitl_tf_, p_->bitl_n_tf_);
  bit_head::SetLhid(p_->bitl_lhid_, p_->bitl_n_lhid_, 0);
  bit_head::SetZone(p_->bitl_zone_);
  bit_head::Init();
  for (int i = 0; i < 4; ++i) {
    x_ = (x_ << 8) + (ReadByte() & 0xff);
  }
}

int Decoder::ReadByte() {
  int byte = (unsigned char)(is_->get());
  if (!is_->good()) return 0;
  return byte;
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
  if (bit_head::Enabled()
      && !bit_head::AuxOff()
      && !bit_head::MetaOff()
      && !p_->AdepthSkipped()
      ) {
    p = bit_head::Adjust(p, p_->bitl_stage1_, p_->bitl_n_stage1_,
                          p_->bitl_m1raw_, p_->bitl_override_);
  }
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  int bit = 0;
  if (x_ <= xmid) {
    bit = 1;
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }
  if (bit_head::Enabled()
      && !bit_head::AuxOff()
      && !bit_head::MetaOff()
      && !p_->AdepthSkipped()
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
  for (int bit = 7; bit >= 0; --bit) {
    ObserveKnownBit((byte >> bit) & 1);
  }
}
