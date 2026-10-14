# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  This file is a fork of `pysrc/weights_compress.py` from
# fx2-cmix-transformer by Vladimir Ivanov (GPL-3.0); the container formats v1
# and v2, the range coder, the bit trees and the RoPE recomputation are his.
# Our addition is container format v3 (magic FX2TFWC3): the three stationary
# bit trees count instead of decaying; format v4 (FX2TFWC4) also counts on the
# two low byte planes of the f32 tensors, skips all-zero int4 rows behind a
# flag bit and predicts names and metadata from a template.  See
# ../../ATTRIBUTION.md.
#
# Copyright (C) 2026 Vladimir Ivanov (original)
# Copyright (C) 2026 Marcus Woelk (formats v3 and v4 and this modification)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""losslessly compressed variants of cpp_infer/data/weights.bin

the uncompressed format (pysrc/export_weights.py, cpp_infer/SPEC.md section 4)
stays the reference; this module writes/reads compressed containers that decode
to bit-for-bit identical tensors.  the C++ decoder lives in
cpp_infer/src/weights_io_compressed.cpp and must stay in exact sync.

format v1, magic FX2TFWC1: every DT_I8 tensor (the 15-value quantized weights,
ints in [-7, 7]) is range-coded with a uniform 1/15 model, i.e. log2(15) =
3.907 bits per weight; bf16/f32/i32 tensors and all metadata are stored raw.

format v2, magic FX2TFWC2: one range-coded stream with adaptive binary models
(LZMA-style 11-bit probabilities, shift-5 update):
  - rope.sin / rope.cos are not stored at all: the decoder recomputes them
    from rope.inv_freq with a bit-exact host port of CUDA libdevice
    __nv_sinf/__nv_cosf (the encoder verifies equality before dropping them,
    and falls back to storing the tables if they do not match)
  - DT_I8 quantized weights: adaptive 15-symbol bit tree (~order-0 entropy,
    3.79 bits/weight)
  - DT_BF16: adaptive byte models, low byte contexted on the high byte
  - DT_F32 / DT_I32: adaptive per-byte-plane models
  - names: order-2 adaptive character model; metadata: order-1 byte model

format v3, magic FX2TFWC3: identical container, identical range coder and
identical bit trees as v2; the only difference is the update rule
of the node probabilities on the three STATIONARY trees.  v2's shift-5 step is
an exponential decay with time constant 32, far too fast for a weight
distribution that does not move; v3 lets those nodes COUNT instead:

    p = 2048 * n0 // (n0 + n1), clamped to [1, 2047];  n[bit] += 1
    both counts halved (floor, min 1) once n0 + n1 exceeds a per-tree limit

  - DT_I8 quantized weights: counting 4-bit tree, limit 1023
  - names: counting order-2 character trees, limit 255
  - metadata: counting order-1 byte trees, limit 255
  - DT_BF16, DT_F32/DT_I32, ENC_RAW keep the v2 shift-5 rule; those byte
    planes are not stationary, and counting costs 516 B there
  - rope.sin/rope.cos handling is unchanged

format v4, magic FX2TFWC4: v3 plus three changes, everything else as in v3.
The compressor of this entry reads v4 (and v2 and v3); a loader without v4
support refuses the magic.
  (a) ENC_PLANE4: counting trees (limit 1023) on byte planes 0 and 1.  The f32
      tensors of a bf16 export are rounded to bf16, so those two planes are
      zero almost everywhere and the shift-5 rule pays its adaptation floor
      (about 0.18 bit per zero byte) on every one of them; planes 2 and 3 stay
      shift-5.  About 1.5 KB on the 8M container.
  (b) ENC_INT4: one counting flag bit (limit 1023) in front of every row (row
      = index of dimension 0, a tensor with ndim < 2 is one row).  Flag 1 means
      the whole row is zero and is not coded; the writer sets it for every
      all-zero row, there is no other rule in the format.
  (c) names and metadata from a template.  The canonical name replaces the
      layer number of 'blocks.<n>.' by '#'.  A counting flag (limit 255) says
      "the canonical name is the one that followed the previous tensor's
      canonical name last time"; for a block name a second flag says "same
      layer as the previous block tensor" (else the next one).  Flag 0 codes
      the name as in v3.  A third flag says "dtype, shape and encoding as at
      the last occurrence of this canonical name"; flag 0 codes them as in
      v3.  About 3.3 KB on the 8M container (names and metadata cost 4.8 KB
      in v3).
The rows that become zero are chosen BEFORE the writer, by
zero_collapsed_rows() below (an explicit parameter, default 2**-20): attention
channels that collapsed in training keep a row scale of 2**-20 ... 2**-102 and
an int4 row of noise behind it.  On the 8M container these are 86 rows of 29
channels (query, key and forget gate of the same channel); set to zero they
leave the compressed archive byte for byte unchanged (measured on two test
beds), and not coding them saves about 5.5 KB.

On a 4 M model format v3 is 1,862,480 B instead of 1,883,433 B (-20,953 B per
container; the submission ships the container twice) at unchanged decode
time.

convert / verify from the command line (run from the repository root):
    python -m pysrc.weights_compress compress   cpp_infer/data/weights.bin cpp_infer/data/weights.c2
    python -m pysrc.weights_compress compress3  cpp_infer/data/weights.bin cpp_infer/data/weights.c3
    python -m pysrc.weights_compress compress4  cpp_infer/data/weights.bin cpp_infer/data/weights.c4 [LOG2|off]
    python -m pysrc.weights_compress compress1  cpp_infer/data/weights.bin cpp_infer/data/weights.c1
    python -m pysrc.weights_compress decompress cpp_infer/data/weights.c3  /tmp/roundtrip.bin
    python -m pysrc.weights_compress verify     cpp_infer/data/weights.bin cpp_infer/data/weights.c3
(decompress/verify detect the format from the magic; compress4 first zeroes
the collapsed rows with log2|scale| < LOG2, default -20, "off" keeps them, and
verifies against the zeroed tensors.  An existing v3 container is converted
with convert_container_v4.py next to this file.)
This fork is run as `python3 tf/4_export/weights_compress_v3.py <command> <in>
<out>` with the upstream tree on PYTHONPATH (it imports pysrc.export_weights).
"""

import struct
import sys
from fractions import Fraction

import numpy as np

from pysrc.export_weights import (
    DTYPE_BF16_BITS,
    DTYPE_FLOAT32,
    DTYPE_INT8,
    DTYPE_INT32,
    _DTYPE_TO_NUMPY,
    read_tensor_file,
)

MAGIC_V1 = b"FX2TFWC1"

_TOP = 1 << 24
_MASK32 = 0xFFFFFFFF


class RangeEncoder:
    """LZMA-style byte-wise range coder (identical to the C++ decoder's model
    of the stream: 32-bit range, 64-bit low with carry propagation through a
    cache byte, renormalization at 2**24)"""

    def __init__(self) -> None:
        self.low = 0
        self.range = _MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > _MASK32:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & _MASK32

    def encode(self, cum: int, freq: int, tot: int) -> None:
        r = self.range // tot
        self.low += r * cum
        self.range = r * freq
        while self.range < _TOP:
            self.range = self.range << 8
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 1  # the first byte is the encoder's initial zero cache
        self.range = _MASK32
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self) -> int:
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode_uniform(self, tot: int) -> int:
        r = self.range // tot
        s = min(self.code // r, tot - 1)
        self.code -= r * s
        self.range = r
        while self.range < _TOP:
            self.code = ((self.code << 8) | self._byte()) & _MASK32
            self.range = self.range << 8
        return s


def _encode_i8_uniform15(values: np.ndarray) -> bytes:
    """range-code int8 values in [-7, 7] at log2(15) bits per value"""
    assert values.dtype == np.int8
    symbols = values.astype(np.int32).ravel() + 7
    assert symbols.min() >= 0 and symbols.max() <= 14
    enc = RangeEncoder()
    encode = enc.encode
    for s in symbols.tolist():
        encode(s, 1, 15)
    return enc.finish()


def _decode_i8_uniform15(stream: bytes, count: int) -> np.ndarray:
    dec = RangeDecoder(stream)
    decode = dec.decode_uniform
    out = np.empty(count, dtype=np.int8)
    out_list = [decode(15) - 7 for _ in range(count)]
    out[:] = out_list
    return out


def write_tensor_file_v1(
    filename: str, entries: list[tuple[str, int, np.ndarray]]
) -> None:
    """compressed alternative to export_weights.write_tensor_file: same
    entries, DT_I8 payloads range-coded at log2(15) bits/weight, rest raw"""
    with open(filename, "wb") as f:
        f.write(MAGIC_V1)
        f.write(struct.pack("<I", len(entries)))
        for name, dtype_code, array in entries:
            assert array.dtype == _DTYPE_TO_NUMPY[dtype_code], name
            data = np.ascontiguousarray(array)
            encoded = name.encode()
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<I", data.ndim))
            f.write(struct.pack(f"<{data.ndim}I", *data.shape))
            if dtype_code == DTYPE_INT8:
                stream = _encode_i8_uniform15(data.reshape(-1))
                f.write(struct.pack("<I", len(stream)))
                f.write(stream)
            else:
                f.write(data.tobytes())


def read_tensor_file_v1(filename: str) -> dict[str, tuple[int, np.ndarray]]:
    with open(filename, "rb") as f:
        assert f.read(8) == MAGIC_V1
        (n_tensors,) = struct.unpack("<I", f.read(4))
        tensors: dict[str, tuple[int, np.ndarray]] = {}
        for _ in range(n_tensors):
            (name_length,) = struct.unpack("<I", f.read(4))
            name = f.read(name_length).decode()
            (dtype_code,) = struct.unpack("<B", f.read(1))
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            dtype = np.dtype(_DTYPE_TO_NUMPY[dtype_code])
            count = int(np.prod(shape)) if ndim else 1
            if dtype_code == DTYPE_INT8:
                (stream_len,) = struct.unpack("<I", f.read(4))
                array = _decode_i8_uniform15(f.read(stream_len), count)
                array = array.reshape(shape)
            else:
                array = np.frombuffer(
                    f.read(count * dtype.itemsize), dtype=dtype
                ).reshape(shape)
            assert name not in tensors
            tensors[name] = (dtype_code, array)
        assert f.read(1) == b""
        return tensors


# ---------------------------------------------------------------------------
# format v2
# ---------------------------------------------------------------------------

MAGIC_V2 = b"FX2TFWC2"

# payload encodings (one byte per tensor in the v2 stream)
ENC_RAW = 0  # bytes through the order-0 raw model (fallback, any dtype)
ENC_INT4 = 1  # DT_I8 values in [-7, 7], adaptive 15-symbol tree
ENC_BF16 = 2  # DT_BF16, hi/lo byte models
ENC_PLANE4 = 3  # DT_F32/DT_I32, per-byte-plane models
ENC_ROPE_SIN = 4  # empty payload: recompute from rope.inv_freq
ENC_ROPE_COS = 5  # empty payload: recompute from rope.inv_freq


# --- bit-exact host emulation of CUDA libdevice __nv_sinf/__nv_cosf ---------
#
# algorithm transcribed from the __nv_sinf/__nv_cosf LLVM IR of
# /usr/local/cuda/nvvm/libdevice/libdevice.10.bc (CUDA 13.0): Cody-Waite
# 3-step FMA reduction for |a| < 105615, Payne-Hanek integer reduction above,
# shared polynomial kernel; cos is sin with quadrant + 1.
#
# float32 fmaf(x, y, z) is emulated as double(x)*double(y) + double(z) (the
# product is exact in double, so the double result is correctly rounded) then
# rounded to float32; that double rounding differs from a genuine single
# rounding only when the double lands within one double-ulp of a float32
# rounding boundary, which is detected and redone exactly with Fractions.

_F32_2_OVER_PI = np.uint32(0x3F22F983).view(np.float32)
_F32_RED = [np.uint32(u).view(np.float32) for u in (0xBFC90FDA, 0xB3A22168, 0xA7C234C5)]
_F32_KCOS = [np.uint32(u).view(np.float32) for u in (0x37CBAC00, 0xBAB607ED, 0x3D2AAABB, 0xBEFFFFFF)]
_F32_KSIN = [np.uint32(u).view(np.float32) for u in (0xB94D4153, 0x3C0885E4, 0xBE2AAAA8)]
_PH_SCALE = np.uint64(0x3BF921FB54442D19).view(np.float64)  # pi/2 * 2^-64
_I2OPI = (0x3C439041, 0xDB629599, 0xF534DDC0, 0xFC2757D1, 0x4E441529, 0xA2F9836E)
_TRIG_PLOSS = np.float32(105615.0)


def _round_fraction_to_f32(x: Fraction) -> np.float32:
    """round-to-nearest-even of an exact rational to float32 (finite range)"""
    if x == 0:
        return np.float32(0.0)
    f = np.float32(float(x))  # candidate; may be off by an ulp
    while True:
        below = np.nextafter(f, np.float32(-np.inf))
        above = np.nextafter(f, np.float32(np.inf))
        mid_below = (Fraction(float(f)) + Fraction(float(below))) / 2
        mid_above = (Fraction(float(f)) + Fraction(float(above))) / 2
        if x < mid_below:
            f = below
        elif x > mid_above:
            f = above
        elif x == mid_below:
            return below if int(below.view(np.uint32)) & 1 == 0 else f
        elif x == mid_above:
            return f if int(f.view(np.uint32)) & 1 == 0 else above
        else:
            return f


def _fmaf_vec(x: np.ndarray, y, z) -> np.ndarray:
    """float32 fused multiply-add, correctly rounded, on float32 arrays"""
    x64 = x.astype(np.float64)
    y64 = np.asarray(y, dtype=np.float32).astype(np.float64)
    z64 = np.asarray(z, dtype=np.float32).astype(np.float64)
    d = x64 * y64 + z64  # product exact in double -> one rounding, to double
    f = d.astype(np.float32)
    # double-rounding hazard: d within one double-ulp of a float32 boundary
    f64 = f.astype(np.float64)
    below = np.nextafter(f, np.float32(-np.inf)).astype(np.float64)
    above = np.nextafter(f, np.float32(np.inf)).astype(np.float64)
    ulp = np.spacing(np.abs(d))
    hazard = (np.abs(d - (f64 + below) * 0.5) <= ulp) | (
        np.abs(d - (f64 + above) * 0.5) <= ulp
    )
    for i in np.nonzero(hazard.ravel())[0]:
        xi = np.broadcast_to(x, f.shape).ravel()[i]
        yi = np.broadcast_to(np.asarray(y, dtype=np.float32), f.shape).ravel()[i]
        zi = np.broadcast_to(np.asarray(z, dtype=np.float32), f.shape).ravel()[i]
        exact = Fraction(float(xi)) * Fraction(float(yi)) + Fraction(float(zi))
        f.ravel()[i] = _round_fraction_to_f32(exact)
    return f


def _trig_slowpath(a: np.float32) -> tuple[np.float32, int]:
    """__internal_trig_reduction_slowpath: Payne-Hanek for |a| >= 105615"""
    ia = int(np.float32(a).view(np.uint32))
    sign = ia & 0x80000000
    e = ((ia >> 23) & 0xFF) - 128
    ia = ((ia << 8) | 0x80000000) & 0xFFFFFFFF
    result = []
    hi = 0
    for k in range(6):
        p = _I2OPI[k] * ia + hi
        result.append(p & 0xFFFFFFFF)
        hi = p >> 32
    result.append(hi)
    idx = 4 - (e >> 5)
    sh = e & 31
    rhi, rlo = result[idx + 2], result[idx + 1]
    if sh:
        rhi = ((result[idx + 2] << sh) | (result[idx + 1] >> (32 - sh))) & 0xFFFFFFFF
        rlo = ((result[idx + 1] << sh) | (result[idx] >> (32 - sh))) & 0xFFFFFFFF
    q = rhi >> 30
    nhi = ((rhi << 2) | (rlo >> 30)) & 0xFFFFFFFF
    nlo = (rlo << 2) & 0xFFFFFFFF
    top = nhi >> 31
    q += top
    if sign:
        q = -q
    s2 = sign
    if top:
        nhi ^= 0xFFFFFFFF
        nlo ^= 0xFFFFFFFF
        s2 = sign ^ 0x80000000
    prod = (nhi << 32) | nlo
    d = float(prod) * float(_PH_SCALE)  # sitofp to double, double multiply
    r = np.float32(d)  # fptrunc: rounds the genuine double, no hazard
    if s2:
        r = -r
    return r, q


def cuda_rope_table(inv_freq: np.ndarray, n_pos: int, cos: bool) -> np.ndarray:
    """(n_pos, len(inv_freq)) float32 table equal bit-for-bit to
    torch.outer(arange(n_pos), inv_freq).sin()/.cos() on a CUDA device"""
    assert inv_freq.dtype == np.float32 and inv_freq.ndim == 1
    t = np.arange(n_pos, dtype=np.float32)
    a = (t[:, None] * inv_freq[None, :]).astype(np.float32).ravel()

    # q = __float2int_rn(a * 2/pi); the product is exact in double, so one
    # float32 rounding; np.rint rounds half to even like cvt.rni
    m = (a.astype(np.float64) * float(_F32_2_OVER_PI)).astype(np.float32)
    q = np.rint(m).astype(np.int64)
    j = q.astype(np.float32)
    t_red = _fmaf_vec(j, _F32_RED[0], a)
    t_red = _fmaf_vec(j, _F32_RED[1], t_red)
    t_red = _fmaf_vec(j, _F32_RED[2], t_red)
    quadrant = q.copy()
    for i in np.nonzero(np.abs(a) >= _TRIG_PLOSS)[0]:
        t_red[i], quadrant[i] = _trig_slowpath(a[i])

    if cos:
        quadrant = quadrant + 1
    odd = (quadrant & 1) != 0
    x2 = (t_red.astype(np.float64) * t_red.astype(np.float64)).astype(np.float32)
    base = np.where(odd, np.float32(1.0), t_red)
    p = _fmaf_vec(x2, base, np.zeros_like(x2))
    c = np.where(odd, _fmaf_vec(np.full_like(x2, _F32_KCOS[0]), x2, np.full_like(x2, _F32_KCOS[1])), _F32_KSIN[0].astype(np.float32))
    c = _fmaf_vec(c, x2, np.where(odd, _F32_KCOS[2], _F32_KSIN[1]))
    c = _fmaf_vec(c, x2, np.where(odd, _F32_KCOS[3], _F32_KSIN[2]))
    z = _fmaf_vec(c, p, base)
    negate = (quadrant & 2) != 0
    z = np.where(negate, (z * np.float32(-1.0)) + np.float32(0.0), z)
    return z.reshape(n_pos, len(inv_freq))


# --- adaptive binary range coder (LZMA-style, 11-bit probs, shift-5) ---------


class BinEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.range = _MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > _MASK32:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & _MASK32

    def encode_bit(self, probs: list, idx: int, bit: int) -> None:
        p = probs[idx]
        bound = (self.range >> 11) * p
        if bit == 0:
            self.range = bound
            probs[idx] = p + ((2048 - p) >> 5)
        else:
            self.low += bound
            self.range -= bound
            probs[idx] = p - (p >> 5)
        while self.range < _TOP:
            self.range = self.range << 8
            self._shift_low()

    def encode_tree(self, probs: list, nbits: int, symbol: int) -> None:
        node = 1
        for k in range(nbits - 1, -1, -1):
            bit = (symbol >> k) & 1
            self.encode_bit(probs, node, bit)
            node = (node << 1) | bit

    # Format v3.  Same tree walk, same 11-bit probability and the same coder
    # arithmetic as encode_tree; only the node state and its update rule
    # differ (_count_prob / _count_bump, v3 section below).
    def encode_ctree(self, tree: list, nbits: int, symbol: int, limit: int) -> None:
        node = 1
        for k in range(nbits - 1, -1, -1):
            bit = (symbol >> k) & 1
            bound = (self.range >> 11) * _count_prob(tree, node)
            if bit == 0:
                self.range = bound
            else:
                self.low += bound
                self.range -= bound
            while self.range < _TOP:
                self.range = self.range << 8
                self._shift_low()
            _count_bump(tree, node, bit, limit)
            node = (node << 1) | bit

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class BinDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 1  # the first byte is the encoder's initial zero cache
        self.range = _MASK32
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self) -> int:
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode_bit(self, probs: list, idx: int) -> int:
        p = probs[idx]
        bound = (self.range >> 11) * p
        if self.code < bound:
            self.range = bound
            probs[idx] = p + ((2048 - p) >> 5)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[idx] = p - (p >> 5)
            bit = 1
        while self.range < _TOP:
            self.code = ((self.code << 8) | self._byte()) & _MASK32
            self.range = self.range << 8
        return bit

    def decode_tree(self, probs: list, nbits: int) -> int:
        node = 1
        for _ in range(nbits):
            node = (node << 1) | self.decode_bit(probs, node)
        return node - (1 << nbits)

    # format v3, the mirror image of BinEncoder.encode_ctree
    def decode_ctree(self, tree: list, nbits: int, limit: int) -> int:
        node = 1
        for _ in range(nbits):
            bound = (self.range >> 11) * _count_prob(tree, node)
            if self.code < bound:
                self.range = bound
                bit = 0
            else:
                self.code -= bound
                self.range -= bound
                bit = 1
            while self.range < _TOP:
                self.code = ((self.code << 8) | self._byte()) & _MASK32
                self.range = self.range << 8
            _count_bump(tree, node, bit, limit)
            node = (node << 1) | bit
        return node - (1 << nbits)


class _Models:
    """the adaptive model set of the v2 stream (same on both sides)"""

    def __init__(self) -> None:
        make = lambda: [1024] * 256  # bit-tree probs, indices 1..255 used
        self.meta = {}  # order-1: prev meta byte -> byte tree
        self.meta_prev = 0
        self.name = {}  # order-2: (prev2, prev1) -> byte tree
        self.raw = make()
        self.int4 = [1024] * 16  # 4-bit tree, symbols 0..14
        self.bf16_hi = make()
        self.bf16_lo = {}  # hi byte -> byte tree
        self.plane = [make() for _ in range(4)]
        self._make = make

    def meta_tree(self) -> list:
        t = self.meta.get(self.meta_prev)
        if t is None:
            t = self.meta[self.meta_prev] = self._make()
        return t

    def name_tree(self, c2: int, c1: int) -> list:
        t = self.name.get((c2, c1))
        if t is None:
            t = self.name[(c2, c1)] = self._make()
        return t

    def bf16_lo_tree(self, hi: int) -> list:
        t = self.bf16_lo.get(hi)
        if t is None:
            t = self.bf16_lo[hi] = self._make()
        return t


def _pick_encoding(
    name: str,
    dtype_code: int,
    array: np.ndarray,
    seen: dict[str, np.ndarray],
) -> int:
    if dtype_code == DTYPE_INT8:
        flat = array.astype(np.int32).ravel()
        if len(flat) == 0 or (flat.min() >= -7 and flat.max() <= 7):
            return ENC_INT4
        return ENC_RAW
    if dtype_code == DTYPE_BF16_BITS:
        return ENC_BF16
    if dtype_code == DTYPE_FLOAT32:
        if (
            name in ("rope.sin", "rope.cos")
            and array.ndim == 2
            and "rope.inv_freq" in seen
            and seen["rope.inv_freq"].shape == (array.shape[1],)
        ):
            table = cuda_rope_table(
                seen["rope.inv_freq"], array.shape[0], cos=name == "rope.cos"
            )
            if np.array_equal(
                table.view(np.uint32), np.ascontiguousarray(array).view(np.uint32)
            ):
                return ENC_ROPE_SIN if name == "rope.sin" else ENC_ROPE_COS
        return ENC_PLANE4
    if dtype_code == DTYPE_INT32:
        return ENC_PLANE4
    return ENC_RAW


def write_tensor_file_v2(
    filename: str, entries: list[tuple[str, int, np.ndarray]]
) -> None:
    """compressed alternative to export_weights.write_tensor_file (format v2)"""
    enc = BinEncoder()
    models = _Models()

    def put_meta(b: int) -> None:
        enc.encode_tree(models.meta_tree(), 8, b)
        models.meta_prev = b

    seen: dict[str, np.ndarray] = {}
    for name, dtype_code, array in entries:
        assert array.dtype == _DTYPE_TO_NUMPY[dtype_code], name
        data = np.ascontiguousarray(array)
        encoded = name.encode()
        assert len(encoded) <= 255, name

        put_meta(len(encoded))
        c2 = c1 = 0
        for ch in encoded:
            enc.encode_tree(models.name_tree(c2, c1), 8, ch)
            c2, c1 = c1, ch
        put_meta(dtype_code)
        assert data.ndim <= 255
        put_meta(data.ndim)
        for d in data.shape:
            for k in range(4):
                put_meta((d >> (8 * k)) & 0xFF)

        encoding = _pick_encoding(name, dtype_code, data, seen)
        put_meta(encoding)

        if encoding == ENC_INT4:
            probs = models.int4
            for v in (data.astype(np.int32).ravel() + 7).tolist():
                enc.encode_tree(probs, 4, v)
        elif encoding == ENC_BF16:
            for v in data.reshape(-1).tolist():
                hi = v >> 8
                enc.encode_tree(models.bf16_hi, 8, hi)
                enc.encode_tree(models.bf16_lo_tree(hi), 8, v & 0xFF)
        elif encoding == ENC_PLANE4:
            raw = data.tobytes()
            planes = models.plane
            for i, b in enumerate(raw):
                enc.encode_tree(planes[i & 3], 8, b)
        elif encoding in (ENC_ROPE_SIN, ENC_ROPE_COS):
            pass  # recomputed by the decoder
        else:
            assert encoding == ENC_RAW
            probs = models.raw
            for b in data.tobytes():
                enc.encode_tree(probs, 8, b)

        if dtype_code == DTYPE_FLOAT32:
            seen[name] = data

    with open(filename, "wb") as f:
        f.write(MAGIC_V2)
        f.write(struct.pack("<I", len(entries)))
        f.write(enc.finish())


def read_tensor_file_v2(filename: str) -> dict[str, tuple[int, np.ndarray]]:
    with open(filename, "rb") as f:
        assert f.read(8) == MAGIC_V2
        (n_tensors,) = struct.unpack("<I", f.read(4))
        dec = BinDecoder(f.read())
    models = _Models()

    def get_meta() -> int:
        b = dec.decode_tree(models.meta_tree(), 8)
        models.meta_prev = b
        return b

    tensors: dict[str, tuple[int, np.ndarray]] = {}
    for _ in range(n_tensors):
        name_len = get_meta()
        chars = bytearray()
        c2 = c1 = 0
        for _ in range(name_len):
            ch = dec.decode_tree(models.name_tree(c2, c1), 8)
            chars.append(ch)
            c2, c1 = c1, ch
        name = chars.decode()
        dtype_code = get_meta()
        ndim = get_meta()
        shape = []
        for _ in range(ndim):
            d = 0
            for k in range(4):
                d |= get_meta() << (8 * k)
            shape.append(d)
        shape = tuple(shape)
        count = int(np.prod(shape)) if ndim else 1
        encoding = get_meta()

        dtype = np.dtype(_DTYPE_TO_NUMPY[dtype_code])
        if encoding == ENC_INT4:
            assert dtype_code == DTYPE_INT8
            probs = models.int4
            array = np.array(
                [dec.decode_tree(probs, 4) - 7 for _ in range(count)],
                dtype=np.int8,
            ).reshape(shape)
        elif encoding == ENC_BF16:
            assert dtype_code == DTYPE_BF16_BITS
            vals = np.empty(count, dtype=np.uint16)
            for i in range(count):
                hi = dec.decode_tree(models.bf16_hi, 8)
                lo = dec.decode_tree(models.bf16_lo_tree(hi), 8)
                vals[i] = (hi << 8) | lo
            array = vals.reshape(shape)
        elif encoding == ENC_PLANE4:
            planes = models.plane
            raw = bytes(
                dec.decode_tree(planes[i & 3], 8) for i in range(count * 4)
            )
            array = np.frombuffer(raw, dtype=dtype).reshape(shape)
        elif encoding in (ENC_ROPE_SIN, ENC_ROPE_COS):
            assert dtype_code == DTYPE_FLOAT32 and len(shape) == 2
            inv_code, inv = tensors["rope.inv_freq"]
            assert inv_code == DTYPE_FLOAT32 and inv.shape == (shape[1],)
            array = cuda_rope_table(
                inv, shape[0], cos=encoding == ENC_ROPE_COS
            )
        else:
            assert encoding == ENC_RAW
            probs = models.raw
            raw = bytes(
                dec.decode_tree(probs, 8) for i in range(count * dtype.itemsize)
            )
            array = np.frombuffer(raw, dtype=dtype).reshape(shape)

        assert name not in tensors
        tensors[name] = (dtype_code, array)
    return tensors


# ---------------------------------------------------------------------------
# format v3
# ---------------------------------------------------------------------------

MAGIC_V3 = b"FX2TFWC3"
MAGIC_V4 = b"FX2TFWC4"  # v3 + planes 0/1, zero-row flag, template

# count limits per tree family; the two counters of a node are halved as soon
# as their sum passes the limit.  1023 on the int4 tree and 255 on the text-like
# trees are the measured optima.
CNT_LIMIT_INT4 = 1023
CNT_LIMIT_NAME = 255
CNT_LIMIT_META = 255
CNT_LIMIT_PLANE01 = 1023  # v4: byte planes 0 and 1 of ENC_PLANE4
CNT_LIMIT_ZROW = 1023  # v4: the zero-row flag of ENC_INT4
CNT_LIMIT_TMPL = 255  # v4: the three template flags of name and metadata

# v4: default limit of zero_collapsed_rows, log2 of the row scale
ZERO_ROWS_LOG2 = -20.0


def _count_tree(nbits: int) -> list:
    """bit tree with COUNTING nodes: two counters per node instead of one
    11-bit probability, both starting at 1 (a uniform Laplace prior).  Node
    indices 1 .. 2**nbits - 1 are used, exactly as in the v2 trees."""
    size = 1 << nbits
    return [[1] * size, [1] * size]


def _count_prob(tree: list, node: int) -> int:
    """the v3 node probability, 11 bit like v2's, clamped away from 0 and 2048
    (the clamp cannot fire at the limits used here; it guards the format)"""
    n0 = tree[0][node]
    n1 = tree[1][node]
    p = (2048 * n0) // (n0 + n1)
    return 1 if p < 1 else (2047 if p > 2047 else p)


def _count_bump(tree: list, node: int, bit: int, limit: int) -> None:
    tree[bit][node] += 1
    if tree[0][node] + tree[1][node] > limit:
        n0 = tree[0][node] >> 1
        n1 = tree[1][node] >> 1
        tree[0][node] = n0 if n0 else 1
        tree[1][node] = n1 if n1 else 1


class _ModelsV3:
    """the model set of the v3 stream.  meta, name and int4 count; bf16, plane
    and raw keep the v2 shift-5 trees, where counting would COST 516 B."""

    def __init__(self, v4: bool = False) -> None:
        make = lambda: [1024] * 256
        self.meta = {}  # order-1: prev meta byte -> counting byte tree
        self.meta_prev = 0
        self.name = {}  # order-2: (prev2, prev1) -> counting byte tree
        self.int4 = _count_tree(4)  # 4-bit counting tree, symbols 0..14
        self.raw = make()
        self.bf16_hi = make()
        self.bf16_lo = {}
        self.plane = [make() for _ in range(4)]
        # v4: counting trees for byte planes 0 and 1 of ENC_PLANE4
        self.cplane = [_count_tree(8), _count_tree(8)] if v4 else None
        # v4: one counting node (a 1-bit tree) for the zero-row flag of ENC_INT4
        self.zrow = _count_tree(1) if v4 else None
        # v4: template flags: name predicted, same layer, metadata predicted
        self.tname = _count_tree(1) if v4 else None
        self.tblock = _count_tree(1) if v4 else None
        self.tmeta = _count_tree(1) if v4 else None
        self._make = make

    def meta_tree(self) -> list:
        t = self.meta.get(self.meta_prev)
        if t is None:
            t = self.meta[self.meta_prev] = _count_tree(8)
        return t

    def name_tree(self, c2: int, c1: int) -> list:
        t = self.name.get((c2, c1))
        if t is None:
            t = self.name[(c2, c1)] = _count_tree(8)
        return t

    def bf16_lo_tree(self, hi: int) -> list:
        t = self.bf16_lo.get(hi)
        if t is None:
            t = self.bf16_lo[hi] = self._make()
        return t


def _canon_name(name: str) -> tuple[str, int]:
    """v4 template: 'blocks.<n>.<rest>' -> ('blocks.#.<rest>', n) when n is a
    canonical decimal of 1 to 9 digits; any other name -> (name, -1).  Same
    rule as canon_name in the C++ loader."""
    if name.startswith("blocks."):
        j = 7
        while j < len(name) and j < 16 and name[j] in "0123456789":
            j += 1
        k = j - 7
        if k >= 1 and j < len(name) and name[j] == "." and (k == 1 or name[7] != "0"):
            return "blocks.#" + name[j:], int(name[7:j])
    return name, -1


class _Template:
    """v4 template state: the successor each canonical name had last time and
    its metadata bytes (dtype, ndim, shape, encoding) at the last occurrence"""

    def __init__(self) -> None:
        self.succ: dict[str, str] = {}
        self.meta: dict[str, list[int]] = {}
        self.prev = None  # canonical name of the previous tensor
        self.prev_blk = -1

    def predicted(self):
        return self.succ.get(self.prev) if self.prev is not None else None

    def update(self, cn: str, blk: int, meta: list[int]) -> None:
        if self.prev is not None:
            self.succ[self.prev] = cn
        self.meta[cn] = meta
        self.prev = cn
        if blk >= 0:
            self.prev_blk = blk


def write_tensor_file_v3(
    filename: str, entries: list[tuple[str, int, np.ndarray]], v4: bool = False
) -> None:
    """compressed alternative to export_weights.write_tensor_file (format v3,
    or v4 with v4=True)"""
    enc = BinEncoder()
    models = _ModelsV3(v4)

    def put_meta(b: int) -> None:
        enc.encode_ctree(models.meta_tree(), 8, b, CNT_LIMIT_META)
        models.meta_prev = b

    def put_name(encoded: bytes) -> None:
        put_meta(len(encoded))
        c2 = c1 = 0
        for ch in encoded:
            enc.encode_ctree(models.name_tree(c2, c1), 8, ch, CNT_LIMIT_NAME)
            c2, c1 = c1, ch

    tmpl = _Template() if v4 else None
    seen: dict[str, np.ndarray] = {}
    for name, dtype_code, array in entries:
        assert array.dtype == _DTYPE_TO_NUMPY[dtype_code], name
        data = np.ascontiguousarray(array)
        encoded = name.encode()
        assert len(encoded) <= 255, name
        assert data.ndim <= 8, name  # the C++ loader refuses more

        encoding = _pick_encoding(name, dtype_code, data, seen)
        meta = [dtype_code, data.ndim]
        for d in data.shape:
            meta += [(d >> (8 * k)) & 0xFF for k in range(4)]
        meta.append(encoding)

        if tmpl is None:
            put_name(encoded)
            for b in meta:
                put_meta(b)
        else:
            # v4 (c): name and metadata from the template
            cn, blk = _canon_name(name)
            if blk >= 0:
                hit = tmpl.predicted() == cn and blk in (tmpl.prev_blk, tmpl.prev_blk + 1)
            else:
                # a literal 'blocks.#.' name would read back as a block name
                hit = tmpl.predicted() == cn and not cn.startswith("blocks.#.")
            enc.encode_ctree(models.tname, 1, int(hit), CNT_LIMIT_TMPL)
            if hit:
                if blk >= 0:
                    enc.encode_ctree(models.tblock, 1, int(blk == tmpl.prev_blk),
                                     CNT_LIMIT_TMPL)
            else:
                put_name(encoded)
            same = tmpl.meta.get(cn) == meta
            enc.encode_ctree(models.tmeta, 1, int(same), CNT_LIMIT_TMPL)
            if not same:
                for b in meta:
                    put_meta(b)
            tmpl.update(cn, blk, meta)

        if encoding == ENC_INT4:
            tree = models.int4
            encode_ctree = enc.encode_ctree
            if models.zrow is None:
                for v in (data.astype(np.int32).ravel() + 7).tolist():
                    encode_ctree(tree, 4, v, CNT_LIMIT_INT4)
            else:
                # v4: flag and row interleaved, in the order the reader walks
                rows = data.shape[0] if data.ndim > 1 else 1
                q2 = data.astype(np.int32).reshape(rows, -1)
                for row in q2:
                    zero = 0 if row.any() else 1
                    encode_ctree(models.zrow, 1, zero, CNT_LIMIT_ZROW)
                    if zero:
                        continue
                    for v in (row + 7).tolist():
                        encode_ctree(tree, 4, v, CNT_LIMIT_INT4)
        elif encoding == ENC_BF16:
            for v in data.reshape(-1).tolist():
                hi = v >> 8
                enc.encode_tree(models.bf16_hi, 8, hi)
                enc.encode_tree(models.bf16_lo_tree(hi), 8, v & 0xFF)
        elif encoding == ENC_PLANE4:
            raw = data.tobytes()
            planes = models.plane
            cplane = models.cplane
            for i, b in enumerate(raw):
                if cplane is not None and (i & 3) < 2:
                    enc.encode_ctree(cplane[i & 3], 8, b, CNT_LIMIT_PLANE01)
                else:
                    enc.encode_tree(planes[i & 3], 8, b)
        elif encoding in (ENC_ROPE_SIN, ENC_ROPE_COS):
            pass  # recomputed by the decoder
        else:
            assert encoding == ENC_RAW
            probs = models.raw
            for b in data.tobytes():
                enc.encode_tree(probs, 8, b)

        if dtype_code == DTYPE_FLOAT32:
            seen[name] = data

    with open(filename, "wb") as f:
        f.write(MAGIC_V4 if v4 else MAGIC_V3)
        f.write(struct.pack("<I", len(entries)))
        f.write(enc.finish())


def write_tensor_file_v4(
    filename: str, entries: list[tuple[str, int, np.ndarray]]
) -> None:
    """format v4: v3 with counting trees on planes 0/1, the
    zero-row flag on int4 and the name/metadata template.  Lossless: it codes
    the tensors it is given; rows are zeroed beforehand by
    zero_collapsed_rows(), never here."""
    write_tensor_file_v3(filename, entries, v4=True)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """fp32 values of raw bfloat16 bits (bits << 16 reinterpreted as float)"""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def zero_collapsed_rows(
    entries: list[tuple[str, int, np.ndarray]], log2_limit: float = ZERO_ROWS_LOG2
) -> tuple[list[tuple[str, int, np.ndarray]], list[tuple[str, int, float, int, int]]]:
    """set the int4 rows of collapsed channels to zero (row AND row scale)

    A row is selected when its tensor is an int4 `<x>.q` outside the MLP
    blocks, `<x>.scale` holds one bf16 scale per row, and log2|scale| <
    log2_limit.  Both the int4 row and its scale become 0, so the row is
    exactly zero whatever the engine does with the scale.  The MLP is left to
    prune_dead_neurons.py, which removes dead neurons structurally.

    Returns (new entries in the same order, report), report rows are
    (tensor, row, log2|scale|, values per row, nonzero values before).
    """
    by_name = {name: (code, array) for name, code, array in entries}
    selected: dict[str, list[int]] = {}
    for name, code, array in entries:
        if code != DTYPE_INT8 or not name.endswith(".q") or ".mlp." in name:
            continue
        sname = name[:-2] + ".scale"
        if sname not in by_name or by_name[sname][0] != DTYPE_BF16_BITS:
            continue
        rows = array.shape[0] if array.ndim > 1 else 1
        scale = _bf16_to_f32(by_name[sname][1]).astype(np.float64).ravel()
        if len(scale) != rows:
            continue
        lg = np.log2(np.maximum(np.abs(scale), 1e-300))
        hit = np.flatnonzero(lg < log2_limit)
        if len(hit):
            selected[name] = [int(j) for j in hit]

    report = []
    changed: dict[str, np.ndarray] = {}
    for qname, rows_sel in selected.items():
        sname = qname[:-2] + ".scale"
        q = np.array(by_name[qname][1], copy=True)
        s = np.array(by_name[sname][1], copy=True)
        q2 = q.reshape(q.shape[0] if q.ndim > 1 else 1, -1)
        sflat = s.reshape(-1)
        sf = _bf16_to_f32(sflat).astype(np.float64)
        for j in rows_sel:
            report.append((qname, j, float(np.log2(abs(sf[j]))), int(q2.shape[1]),
                           int((q2[j] != 0).sum())))
            q2[j, :] = 0
            sflat[j] = 0
        changed[qname] = q
        changed[sname] = s
    out = [(name, code, changed.get(name, array)) for name, code, array in entries]
    return out, report


def read_tensor_file_v3(filename: str) -> dict[str, tuple[int, np.ndarray]]:
    """reads format v3 and v4 (the magic decides)"""
    with open(filename, "rb") as f:
        magic = f.read(8)
        assert magic in (MAGIC_V3, MAGIC_V4)
        (n_tensors,) = struct.unpack("<I", f.read(4))
        dec = BinDecoder(f.read())
    models = _ModelsV3(magic == MAGIC_V4)

    def get_meta() -> int:
        b = dec.decode_ctree(models.meta_tree(), 8, CNT_LIMIT_META)
        models.meta_prev = b
        return b

    def get_name() -> str:
        name_len = get_meta()
        chars = bytearray()
        c2 = c1 = 0
        for _ in range(name_len):
            ch = dec.decode_ctree(models.name_tree(c2, c1), 8, CNT_LIMIT_NAME)
            chars.append(ch)
            c2, c1 = c1, ch
        return chars.decode()

    def get_hdr() -> list[int]:
        hdr = [get_meta(), get_meta()]
        hdr += [get_meta() for _ in range(4 * hdr[1] + 1)]
        return hdr

    tmpl = _Template() if models.tname is not None else None
    tensors: dict[str, tuple[int, np.ndarray]] = {}
    for _ in range(n_tensors):
        if tmpl is None:
            name = get_name()
            hdr = get_hdr()
        else:
            # v4 (c): name and metadata from the template
            if dec.decode_ctree(models.tname, 1, CNT_LIMIT_TMPL):
                cn = tmpl.predicted()
                assert cn is not None, "template name without a predecessor"
                if cn.startswith("blocks.#."):
                    blk = (tmpl.prev_blk
                           if dec.decode_ctree(models.tblock, 1, CNT_LIMIT_TMPL)
                           else tmpl.prev_blk + 1)
                    name = "blocks.%d%s" % (blk, cn[8:])
                else:
                    blk = -1
                    name = cn
            else:
                name = get_name()
                cn, blk = _canon_name(name)
            if dec.decode_ctree(models.tmeta, 1, CNT_LIMIT_TMPL):
                hdr = tmpl.meta[cn]
            else:
                hdr = get_hdr()
            tmpl.update(cn, blk, hdr)
        dtype_code = hdr[0]
        ndim = hdr[1]
        shape = tuple(
            sum(hdr[2 + 4 * d + k] << (8 * k) for k in range(4)) for d in range(ndim)
        )
        count = int(np.prod(shape)) if ndim else 1
        encoding = hdr[-1]

        dtype = np.dtype(_DTYPE_TO_NUMPY[dtype_code])
        if encoding == ENC_INT4:
            assert dtype_code == DTYPE_INT8
            tree = models.int4
            decode_ctree = dec.decode_ctree
            if models.zrow is None:
                array = np.array(
                    [decode_ctree(tree, 4, CNT_LIMIT_INT4) - 7 for _ in range(count)],
                    dtype=np.int8,
                ).reshape(shape)
            else:
                rows = shape[0] if ndim > 1 else 1
                cols = count // rows if rows else 0
                q2 = np.zeros((rows, cols), dtype=np.int8)
                for r in range(rows):
                    if decode_ctree(models.zrow, 1, CNT_LIMIT_ZROW):
                        continue  # v4: all-zero row, not coded
                    q2[r] = [decode_ctree(tree, 4, CNT_LIMIT_INT4) - 7
                             for _ in range(cols)]
                array = q2.reshape(shape)
        elif encoding == ENC_BF16:
            assert dtype_code == DTYPE_BF16_BITS
            vals = np.empty(count, dtype=np.uint16)
            for i in range(count):
                hi = dec.decode_tree(models.bf16_hi, 8)
                lo = dec.decode_tree(models.bf16_lo_tree(hi), 8)
                vals[i] = (hi << 8) | lo
            array = vals.reshape(shape)
        elif encoding == ENC_PLANE4:
            planes = models.plane
            cplane = models.cplane
            if cplane is None:
                raw = bytes(
                    dec.decode_tree(planes[i & 3], 8) for i in range(count * 4)
                )
            else:
                raw = bytes(
                    dec.decode_ctree(cplane[i & 3], 8, CNT_LIMIT_PLANE01)
                    if (i & 3) < 2
                    else dec.decode_tree(planes[i & 3], 8)
                    for i in range(count * 4)
                )
            array = np.frombuffer(raw, dtype=dtype).reshape(shape)
        elif encoding in (ENC_ROPE_SIN, ENC_ROPE_COS):
            assert dtype_code == DTYPE_FLOAT32 and len(shape) == 2
            inv_code, inv = tensors["rope.inv_freq"]
            assert inv_code == DTYPE_FLOAT32 and inv.shape == (shape[1],)
            array = cuda_rope_table(
                inv, shape[0], cos=encoding == ENC_ROPE_COS
            )
        else:
            assert encoding == ENC_RAW
            probs = models.raw
            raw = bytes(
                dec.decode_tree(probs, 8) for i in range(count * dtype.itemsize)
            )
            array = np.frombuffer(raw, dtype=dtype).reshape(shape)

        assert name not in tensors
        tensors[name] = (dtype_code, array)
    return tensors


def _entries_from_reference(filename: str) -> list[tuple[str, int, np.ndarray]]:
    """reference-file tensors in file order (read_tensor_file keeps order:
    python dicts preserve insertion order)"""
    return [
        (name, dtype_code, array)
        for name, (dtype_code, array) in read_tensor_file(filename).items()
    ]


def _verify_equal(
    reference: dict[str, tuple[int, np.ndarray]],
    decoded: dict[str, tuple[int, np.ndarray]],
) -> None:
    assert set(reference) == set(decoded)
    for name, (dtype_code, array) in reference.items():
        decoded_code, decoded_array = decoded[name]
        assert decoded_code == dtype_code, name
        assert decoded_array.shape == array.shape, name
        assert decoded_array.dtype == array.dtype, name
        assert np.array_equal(decoded_array, array), name


def _read_compressed(filename: str) -> dict[str, tuple[int, np.ndarray]]:
    with open(filename, "rb") as f:
        magic = f.read(8)
    if magic == MAGIC_V1:
        return read_tensor_file_v1(filename)
    if magic == MAGIC_V2:
        return read_tensor_file_v2(filename)
    if magic in (MAGIC_V3, MAGIC_V4):
        return read_tensor_file_v3(filename)
    raise ValueError(f"{filename}: unknown magic {magic!r}")


def main() -> None:
    import os

    usage = (
        "usage: python -m pysrc.weights_compress"
        " {compress|compress3|compress4|compress1|decompress|verify}"
        " <input> <output|compressed> [compress4 only: LOG2|off]"
    )
    if len(sys.argv) == 5 and sys.argv[1] == "compress4":
        zero_arg = sys.argv[4]
    elif len(sys.argv) == 4:
        zero_arg = None
    else:
        sys.exit(usage)
    command, a, b = sys.argv[1:4]
    if command not in ("compress", "compress3", "compress4", "compress1", "decompress",
                       "verify"):
        sys.exit(usage)

    # Check the input files up front so a typo is reported as such instead of
    # as a traceback from somewhere deep in the reader ("b" is an output file
    # for every command except "verify").
    inputs = [a, b] if command == "verify" else [a]
    for path in inputs:
        if not os.path.isfile(path):
            sys.exit(
                f"error: input file {path!r} does not exist"
                if not os.path.exists(path)
                else f"error: input file {path!r} is not a regular file"
            )

    if command == "compress4":
        entries = _entries_from_reference(a)
        limit = ZERO_ROWS_LOG2 if zero_arg is None else zero_arg
        if limit != "off":
            entries, report = zero_collapsed_rows(entries, float(limit))
            print(f"zero_collapsed_rows: log2|scale| < {float(limit):g}, "
                  f"{len(report)} rows set to zero")
            for qname, row, lg, cols, nonzero in report:
                print(f"  {qname}\trow {row}\tlog2|s| {lg:.2f}\t"
                      f"{nonzero} of {cols} values were nonzero")
        write_tensor_file_v4(b, entries)
        _verify_equal({n: (c, arr) for n, c, arr in entries}, _read_compressed(b))
        print(
            f"{a} ({os.path.getsize(a)} bytes) -> {b} "
            f"({os.path.getsize(b)} bytes), round-trip verified bit-exact "
            f"against the tensors written"
        )
    elif command in ("compress", "compress3", "compress1"):
        entries = _entries_from_reference(a)
        writer = {
            "compress": write_tensor_file_v2,
            "compress3": write_tensor_file_v3,
            "compress1": write_tensor_file_v1,
        }[command]
        writer(b, entries)
        _verify_equal(dict(read_tensor_file(a)), _read_compressed(b))
        print(
            f"{a} ({os.path.getsize(a)} bytes) -> {b} "
            f"({os.path.getsize(b)} bytes), round-trip verified bit-exact"
        )
    elif command == "decompress":
        from pysrc.export_weights import write_tensor_file

        tensors = _read_compressed(a)
        write_tensor_file(
            b, [(name, code, arr) for name, (code, arr) in tensors.items()]
        )
        print(f"{a} -> {b} ({os.path.getsize(b)} bytes)")
    elif command == "verify":
        _verify_equal(read_tensor_file(a), _read_compressed(b))
        print(f"{b} decodes bit-exactly to {a}")
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main()
