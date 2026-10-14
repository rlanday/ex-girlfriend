#ifndef PPMD_META_H_
#define PPMD_META_H_

namespace ppmd_meta {
// Latch: once set, it stays set.  Set by the metadata block reader at the
// last byte of the block; models/ppmd.cpp::ByteUpdate() reads it and
// restarts the model at the next model byte.
inline bool block_fertig = false;
}

#endif
