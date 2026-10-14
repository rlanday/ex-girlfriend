// isa_floor.h: the instruction-set floor of the BITLSTM code.
//
// The kernels in bit_head.cpp are written for AVX2 and FMA (-march=x86-64-v3)
// and have no scalar fallback.  A build below that floor stops here with one
// clear error instead of one error per intrinsic.

#ifndef ISA_FLOOR_H_
#define ISA_FLOOR_H_

#if !defined(__AVX2__) || !defined(__FMA__)
#error "the BITLSTM requires AVX2+FMA (build with -march=x86-64-v3 or higher)"
#endif

#endif  // ISA_FLOOR_H_
