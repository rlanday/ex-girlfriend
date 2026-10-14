#include "encoder.h"

Encoder::Encoder(std::ofstream* os, Predictor* p) : os_(os), x1_(0),
    x2_(0xffffffff), p_(p) {
  // Register the per-byte channel BEFORE Init(): Init() builds the
  // head and sizes its projections from these widths.  Every pointer is stable
  // for the whole run, so this is the only registration there is: no setter
  // call per coded bit, across a translation-unit boundary, into a unit built
  // without -flto.  Encoder and Decoder register the identical things, which
  // is what keeps the head in lockstep.
  bit_head::SetAux(p_->bitl_hid_, p_->bitl_n_hid_);
  bit_head::SetTf(p_->bitl_tf_, p_->bitl_n_tf_);
  // The compressor registers the RAW state vector (pre_projected = 0); only a
  // replay on a pre-projected trace registers 1.
  bit_head::SetLhid(p_->bitl_lhid_, p_->bitl_n_lhid_, 0);
  bit_head::SetZone(p_->bitl_zone_);
  bit_head::Init();
}

void Encoder::WriteByte(unsigned int byte) {
  out_.push_back(byte);
  if (out_.size() >= (size_t)1024 * 1024) {
    os_->write(out_.data(), out_.size());
    total_out_ += out_.size();
    out_.clear();
  }
}

unsigned int Encoder::Discretize(float p) {
  return 1 + 65534 * p;
}

void Encoder::Encode(int bit) {
  if (p_->ChrForced()) {
    p_->Predict();
    p_->Perceive(bit);
    return;
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
  if (bit) {
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
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::EncodeRawBit(int bit, unsigned int p) {
  if (p < 1) p = 1;
  if (p > 65534) p = 65534;
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  if (bit) {
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }

  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::ObserveKnownByte(unsigned int byte) {
  for (int bit = 7; bit >= 0; --bit) {
    ObserveKnownBit((byte >> bit) & 1);
  }
}

void Encoder::BeginTraceByte(unsigned long long offset, unsigned int actual_byte,
    unsigned int prev4) {
  (void)offset;
  (void)actual_byte;
  (void)prev4;
}

void Encoder::EndTraceByte() {}

void Encoder::Flush() {
  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
  WriteByte(x2_ >> 24);

  auto* data = reinterpret_cast<const char*>(out_.data());
  os_->write(data, out_.size());
  total_out_ += out_.size();
  out_.clear();
  out_.shrink_to_fit();
}
