#ifndef ENC_ONLY_H
#define ENC_ONLY_H
#define ENC_ONLY __attribute__((noinline, section(".text.enc_only")))
// The two directions of a transform as one template on the direction: each
// instance keeps only its own branch, and the encoder instance is called
// from an ENC_ONLY wrapper only.
#define DIR_INLINE inline __attribute__((always_inline))
#endif  // ENC_ONLY_H
