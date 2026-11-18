#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  The container formats v1 to v5, the range coder, the bit trees
# and the RoPE recomputation come from `../export4/weights_compress_v3.py`
# (a fork of `pysrc/weights_compress.py` of fx2-cmix-transformer by Vladimir
# Ivanov, GPL-3.0), which this script imports.  Our addition is this script:
# a Python transcription of the writer of formats NP and NP2 in
# `tf1_np_codec.cpp` next to this file (mode "np") and a reader of both formats.
# See ../../../../ATTRIBUTION.md.
#
# Copyright (C) 2026 Marcus Woelk
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""write and read the transformer weight container in format NP / NP2

Format NP (magic FX2TFWN5) and NP2 (FX2TFWP5) code the tensors of a v5
container (FX2TFWC5, weights_compress_v3.py) in two streams.  The order of
the hidden units of some groups is free for the engine; the writer chooses it
so that the digits of the order in the factorial number system (C.-A.
Laisant 1888, the Lehmer code) spell the first K bytes of the second stream,
which are then not stored (the bits-back argument of G. E. Hinton and D. van
Camp 1993).  Layout and groups: the head comment of tf1_np_codec.cpp.

  np2_container.py np <in.c5> <out> [k|nok] [fw] [np2] [dead]
      in: a v5 container.  Steps, each one checked:
        1. read in; identity probe: the v5 writer of weights_compress_v3.py
           reproduces in bit for bit (with or without ENC_F32HC);
        2. dead: zero the parameters of the dead KDA key dims;
        3. write the NP container (k: the bf16-rounded f32 tensors as
           ENC_F32HC; nok: as ENC_F32HI; default: as in the input);
        4. read it back with the reader of this script: every tensor equals
           the permuted one, the permutation is not trivial.
      The options are those of tf1_np_codec.cpp; the container of this entry
      is written with "k fw np2 dead".
  np2_container.py read <container> [<out.raw>]
      read an NP, NP2 or v5 container; prints the tensor count and the
      sha256 of the tensors in file order; with <out.raw> the decoded tensors
      raw in file order (FX2TFW01 layout, as "tf1_np_codec dump").

Needs numpy only (weights_compress_v3.py takes the dtype codes from
pysrc.export_weights of the upstream tree when it is on PYTHONPATH, else from a
stand-in with the same values).  Deterministic: same input, same output, byte
for byte.  Pure Python, one core.
"""

import hashlib
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export4"))
try:
    import pysrc.export_weights  # noqa: F401  (the upstream tree, needs torch and fla)
except ImportError:
    # weights_compress_v3.py takes only the dtype codes of the tensor file from
    # pysrc.export_weights (values of the upstream file); without the upstream
    # tree this stand-in lets the container tools run on numpy alone
    import types
    _pk = types.ModuleType("pysrc")
    _ew = types.ModuleType("pysrc.export_weights")
    _ew.DTYPE_INT8, _ew.DTYPE_BF16_BITS, _ew.DTYPE_FLOAT32, _ew.DTYPE_INT32 = 0, 1, 2, 3
    _ew._DTYPE_TO_NUMPY = {0: np.int8, 1: np.uint16, 2: np.float32, 3: np.int32}

    def _no_reader(filename):
        raise SystemExit("np2_container: reading an uncompressed tensor file needs the upstream tree")

    _ew.read_tensor_file = _no_reader
    _pk.export_weights = _ew
    sys.modules["pysrc"] = _pk
    sys.modules["pysrc.export_weights"] = _ew
import weights_compress_v3 as wc  # noqa: E402

MAGIC_NP = b"FX2TFWN5"
MAGIC_NP2 = b"FX2TFWP5"
FLAG_ALL, FLAG_BLOCKS, FLAG_FIRST = 0x40, 0x20, 0x60   # key flags in the encoding byte
FLAG_MASK = 0x60


def die(msg: str) -> None:
    sys.exit("np2_container: " + msg)


# ------------------------------------------------------------------ tensors
class Ent:
    """one tensor: name, dtype code, array (C order, as decoded)"""

    def __init__(self, name: str, code: int, array: np.ndarray) -> None:
        self.name = name
        self.code = code
        self.a = np.ascontiguousarray(array).copy()

    def rows(self) -> int:
        return self.a.shape[0] if self.a.ndim > 1 else 1

    def cols(self) -> int:
        r = self.rows()
        return self.a.size // r if r else 0

    def m2(self) -> np.ndarray:
        """2-d view (rows, cols)"""
        return self.a.reshape(self.rows(), self.cols())

    def same(self, o: "Ent") -> bool:
        return (self.name == o.name and self.code == o.code and self.a.shape == o.a.shape
                and self.a.tobytes() == o.a.tobytes())


def ents_of(tensors: dict) -> list:
    return [Ent(n, c, a) for n, (c, a) in tensors.items()]


def entries_of(es: list) -> list:
    return [(e.name, e.code, e.a) for e in es]


def tensors_sha(es: list) -> str:
    h = hashlib.sha256()
    for e in es:
        h.update(e.name.encode() + bytes([e.code]) + struct.pack("<%dI" % e.a.ndim, *e.a.shape))
        h.update(e.a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ encodings
def encodings(es: list, k: bool) -> list:
    """the encoding of each tensor, as the v5 writer picks it"""
    seen = {}
    out = []
    for e in es:
        enc = wc._pick_encoding(e.name, e.code, e.a, seen)
        out.append(wc._pick_encoding_v5(e.name, e.code, e.a, enc, k))
        if e.code == wc.DTYPE_FLOAT32:
            seen[e.name] = e.a
    return out


# ------------------------------------------------------------------ one tensor
def put_meta(enc, models, b: int) -> None:
    enc.encode_ctree(models.meta_tree(), 8, b, wc.CNT_LIMIT_META)
    models.meta_prev = b


def put_name(enc, models, encoded: bytes) -> None:
    put_meta(enc, models, len(encoded))
    c2 = c1 = 0
    for ch in encoded:
        enc.encode_ctree(models.name_tree(c2, c1), 8, ch, wc.CNT_LIMIT_NAME)
        c2, c1 = c1, ch


def code_head(enc, models, tmpl, e: Ent, enc_byte: int) -> str:
    """name and metadata through the v4 template; returns the canonical name"""
    meta = [e.code, e.a.ndim]
    for d in e.a.shape:
        meta += [(d >> (8 * j)) & 0xFF for j in range(4)]
    meta.append(enc_byte)
    cn, blk = wc._canon_name(e.name)
    if blk >= 0:
        hit = tmpl.predicted() == cn and blk in (tmpl.prev_blk, tmpl.prev_blk + 1)
    else:
        hit = tmpl.predicted() == cn and not cn.startswith("blocks.#.")
    enc.encode_ctree(models.tname, 1, int(hit), wc.CNT_LIMIT_TMPL)
    if hit:
        if blk >= 0:
            enc.encode_ctree(models.tblock, 1, int(blk == tmpl.prev_blk), wc.CNT_LIMIT_TMPL)
    else:
        put_name(enc, models, e.name.encode())
    same = tmpl.meta.get(cn) == meta
    enc.encode_ctree(models.tmeta, 1, int(same), wc.CNT_LIMIT_TMPL)
    if not same:
        for b in meta:
            put_meta(enc, models, b)
    tmpl.update(cn, blk, meta)
    return cn


def code_values(enc, models, e: Ent, encoding: int, cn: str) -> None:
    """the values of one tensor (the v5 branches of write_tensor_file_v3)"""
    data = e.a
    if encoding == wc.ENC_INT4W:
        encode_ctree = enc.encode_ctree
        trees = models.int4w
        q2 = data.astype(np.int32).reshape(e.rows(), -1)
        cs = [0] * q2.shape[1]
        n = T = N = 0
        for row in q2:
            zero = 0 if row.any() else 1
            encode_ctree(models.zrow, 1, zero, wc.CNT_LIMIT_ZROW)
            if zero:
                continue
            rs = 0
            for j, v in enumerate(row.tolist()):
                b = wc._width_bucket(T, N, rs, j, cs[j], n)
                encode_ctree(trees[b], 4, v + 7, wc.CNT_LIMIT_INT4W)
                a = v if v >= 0 else -v
                rs += a
                cs[j] += a
                T += a
                N += 1
            n += 1
    elif encoding == wc.ENC_F32HI:
        for v in (data.view(np.uint32).reshape(-1) >> 16).tolist():
            hi = v >> 8
            enc.encode_tree(models.f32_hi, 8, hi)
            enc.encode_tree(models.f32_lo_tree(hi), 8, v & 0xFF)
    elif encoding == wc.ENC_F32HC:
        s = models.f32hc_set(cn)
        cols = e.cols()
        last = wc.F32HC_COLS - 1
        for i, v in enumerate((data.view(np.uint32).reshape(-1) >> 16).tolist()):
            hi = v >> 8
            j = i % cols
            enc.encode_ctree(s[j if j < last else last], 8, hi, wc.CNT_LIMIT_F32HC)
            enc.encode_tree(models.f32_lo_tree(hi), 8, v & 0xFF)
    elif encoding == wc.ENC_BF16C:
        s = models.bf16c_set(cn)
        for v in data.reshape(-1).tolist():
            hi = v >> 8
            enc.encode_tree(s[0], 8, hi)
            enc.encode_tree(models.bf16c_lo_tree(s, hi), 8, v & 0xFF)
    elif encoding == wc.ENC_BF16:
        for v in data.reshape(-1).tolist():
            hi = v >> 8
            enc.encode_tree(models.bf16_hi, 8, hi)
            enc.encode_tree(models.bf16_lo_tree(hi), 8, v & 0xFF)
    elif encoding == wc.ENC_PLANE4:
        planes = models.plane
        cplane = models.cplane
        for i, b in enumerate(data.tobytes()):
            if (i & 3) < 2:
                enc.encode_ctree(cplane[i & 3], 8, b, wc.CNT_LIMIT_PLANE01)
            else:
                enc.encode_tree(planes[i & 3], 8, b)
    elif encoding in (wc.ENC_ROPE_SIN, wc.ENC_ROPE_COS):
        pass  # recomputed by the reader
    elif encoding == wc.ENC_RAW:
        for b in data.tobytes():
            enc.encode_tree(models.raw, 8, b)
    else:
        die("encoding %d of %s is not a v5 encoding" % (encoding, e.name))


def decode_tensor(dec, models, tmpl, known: dict) -> tuple:
    """one tensor of a v5 stream; returns (Ent, encoding byte with key flag)"""
    def get_meta() -> int:
        b = dec.decode_ctree(models.meta_tree(), 8, wc.CNT_LIMIT_META)
        models.meta_prev = b
        return b

    def get_name() -> str:
        n = get_meta()
        chars = bytearray()
        c2 = c1 = 0
        for _ in range(n):
            ch = dec.decode_ctree(models.name_tree(c2, c1), 8, wc.CNT_LIMIT_NAME)
            chars.append(ch)
            c2, c1 = c1, ch
        return chars.decode()

    def get_hdr() -> list:
        hdr = [get_meta(), get_meta()]
        hdr += [get_meta() for _ in range(4 * hdr[1] + 1)]
        return hdr

    if dec.decode_ctree(models.tname, 1, wc.CNT_LIMIT_TMPL):
        cn = tmpl.predicted()
        if cn is None:
            die("template name without a predecessor")
        if cn.startswith("blocks.#."):
            blk = (tmpl.prev_blk if dec.decode_ctree(models.tblock, 1, wc.CNT_LIMIT_TMPL)
                   else tmpl.prev_blk + 1)
            name = "blocks.%d%s" % (blk, cn[8:])
        else:
            blk = -1
            name = cn
    else:
        name = get_name()
        cn, blk = wc._canon_name(name)
    hdr = tmpl.meta[cn] if dec.decode_ctree(models.tmeta, 1, wc.CNT_LIMIT_TMPL) else get_hdr()
    tmpl.update(cn, blk, hdr)
    code, ndim = hdr[0], hdr[1]
    shape = tuple(sum(hdr[2 + 4 * d + j] << (8 * j) for j in range(4)) for d in range(ndim))
    count = int(np.prod(shape)) if ndim else 1
    enc_byte = hdr[-1]
    encoding = enc_byte & ~FLAG_MASK
    dtype = np.dtype(wc._DTYPE_TO_NUMPY[code])
    if encoding == wc.ENC_INT4W:
        rows = shape[0] if ndim > 1 else 1
        cols = count // rows if rows else 0
        q2 = np.zeros((rows, cols), dtype=np.int8)
        cs = [0] * cols
        n = T = N = 0
        for r in range(rows):
            if dec.decode_ctree(models.zrow, 1, wc.CNT_LIMIT_ZROW):
                continue
            rs = 0
            rowv = [0] * cols
            for j in range(cols):
                b = wc._width_bucket(T, N, rs, j, cs[j], n)
                v = dec.decode_ctree(models.int4w[b], 4, wc.CNT_LIMIT_INT4W) - 7
                rowv[j] = v
                a = v if v >= 0 else -v
                rs += a
                cs[j] += a
                T += a
                N += 1
            q2[r] = rowv
            n += 1
        array = q2.reshape(shape)
    elif encoding in (wc.ENC_F32HI, wc.ENC_F32HC):
        vals = np.empty(count, dtype=np.uint32)
        if encoding == wc.ENC_F32HC:
            s = models.f32hc_set(cn)
            cols = count // (shape[0] if ndim > 1 else 1)
            last = wc.F32HC_COLS - 1
        for i in range(count):
            if encoding == wc.ENC_F32HC:
                j = i % cols
                hi = dec.decode_ctree(s[j if j < last else last], 8, wc.CNT_LIMIT_F32HC)
            else:
                hi = dec.decode_tree(models.f32_hi, 8)
            lo = dec.decode_tree(models.f32_lo_tree(hi), 8)
            vals[i] = ((hi << 8) | lo) << 16
        array = vals.view(np.float32).reshape(shape)
    elif encoding == wc.ENC_BF16C:
        s = models.bf16c_set(cn)
        vals = np.empty(count, dtype=np.uint16)
        for i in range(count):
            hi = dec.decode_tree(s[0], 8)
            vals[i] = (hi << 8) | dec.decode_tree(models.bf16c_lo_tree(s, hi), 8)
        array = vals.reshape(shape)
    elif encoding == wc.ENC_BF16:
        vals = np.empty(count, dtype=np.uint16)
        for i in range(count):
            hi = dec.decode_tree(models.bf16_hi, 8)
            vals[i] = (hi << 8) | dec.decode_tree(models.bf16_lo_tree(hi), 8)
        array = vals.reshape(shape)
    elif encoding == wc.ENC_PLANE4:
        raw = bytes(dec.decode_ctree(models.cplane[i & 3], 8, wc.CNT_LIMIT_PLANE01) if (i & 3) < 2
                    else dec.decode_tree(models.plane[i & 3], 8) for i in range(count * 4))
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    elif encoding in (wc.ENC_ROPE_SIN, wc.ENC_ROPE_COS):
        inv_code, inv = known["rope.inv_freq"]
        array = wc.cuda_rope_table(inv, shape[0], cos=encoding == wc.ENC_ROPE_COS)
    elif encoding == wc.ENC_RAW:
        raw = bytes(dec.decode_tree(models.raw, 8) for _ in range(count * dtype.itemsize))
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    else:
        die("encoding %d of %s is not a v5 encoding" % (encoding, name))
    known[name] = (code, array)
    return Ent(name, code, array), enc_byte


def reset_for_b(models, np2: bool) -> None:
    """stream B continues the models of stream A except the int4 trees and the
    per-name trees of the row scales; NP2 also starts the f32 models fresh"""
    models.int4 = wc._count_tree(4)
    models.int4w = [wc._count_tree(4) for _ in range(wc.W_NB)]
    models.zrow = wc._count_tree(1)
    models.bf16c = {}
    if np2:
        models.f32_hi = [1024] * 256
        models.f32_lo = {}
        models.f32hc = {}


def new_state() -> tuple:
    return wc.BinEncoder(), wc._ModelsV3(v4=True, v5=True), wc._Template()


# ------------------------------------------------------------------ groups
class Group:
    """the elements are the rows [r0, r0+n) of the key tensor; moving element
    j moves row o+j of each (tensor, o) in rows_at, element o+j of elem_at and
    column o+j of cols_at"""

    def __init__(self, name, key, flag, r0, n) -> None:
        self.name, self.key, self.flag, self.r0, self.n = name, key, flag, r0, n
        self.rows_at, self.elem_at, self.cols_at = [], [], []


def find_t(es, name, need=True):
    for i, e in enumerate(es):
        if e.name == name:
            return i
    if need:
        die("tensor %s missing" % name)
    return -1


def group_at(name, key, r0, n, rows, elem, cols, key_rows) -> Group:
    g = Group(name, key, FLAG_ALL if n == key_rows else FLAG_BLOCKS, r0, n)
    g.rows_at = [(t, r0) for t in rows]
    g.elem_at = [(t, r0) for t in elem]
    g.cols_at = [(t, r0) for t in cols]
    return g


def np_groups(es, with_fw: bool, kda: bool) -> list:
    kimi = es[find_t(es, "config.kimi")].a.reshape(-1).astype(np.int64).tolist()
    NL = len(kimi)
    gs = []
    vi = 0  # index of the plain attention layer (value_embeddings.<i>)
    for l in range(NL):
        b = "blocks.%d." % l
        a = b + "attention."
        if kimi[l]:
            for g in ("forget_gate_projection", "output_gate_projection"):
                p = a + g
                key = find_t(es, p + ".up.weight.q")
                n = es[key].rows()
                gs.append(group_at(p, key, 0, n, [key], [find_t(es, p + ".up.weight.scale")],
                                   [find_t(es, p + ".down.weight.q")], n))
            if kda:
                for h in range(3):
                    rows = [find_t(es, a + t) for t in (
                        "query_convolution.weight", "key_convolution.weight",
                        "query_projection.weight.q", "key_projection.weight.q",
                        "forget_gate_projection.down.weight.q")]
                    elem = [find_t(es, a + t) for t in (
                        "query_projection.weight.scale", "key_projection.weight.scale",
                        "forget_gate_projection.down.weight.scale", "dt_bias")]
                    key = find_t(es, a + "query_convolution.weight")
                    gs.append(group_at(a + "key_dims.h%d" % h, key, 64 * h, 64, rows, elem, [],
                                       es[key].rows()))
                g = Group(a + "value_dims", find_t(es, a + "value_projection.weight.q"),
                          FLAG_FIRST, 0, 64)
                for h in range(3):
                    for t in ("value_convolution.weight", "value_projection.weight.q",
                              "output_gate_projection.down.weight.q"):
                        g.rows_at.append((find_t(es, a + t), 64 * h))
                    for t in ("value_projection.weight.scale",
                              "output_gate_projection.down.weight.scale"):
                        g.elem_at.append((find_t(es, a + t), 64 * h))
                    g.cols_at.append((find_t(es, a + "output_projection.weight.q"), 64 * h))
                g.elem_at.append((find_t(es, a + "output_fused_norm_gate.weight"), 0))
                gs.append(g)
        elif es[find_t(es, a + "value_projection.weight.q")].rows() == 64:
            # one K/V head (MQA): its 64 value dims, also in the three head
            # blocks of the output projection and the value-embedding table
            key = find_t(es, a + "value_projection.weight.q")
            g = group_at(a + "v_dims.mqa", key, 0, 64, [key],
                         [find_t(es, a + "value_projection.weight.scale")], [], 64)
            op = find_t(es, a + "output_projection.weight.q")
            g.cols_at = [(op, 0), (op, 64), (op, 128)]
            ve = find_t(es, "value_embeddings.%d.inner.weight.q" % vi, need=False)
            if ve >= 0:
                g.cols_at.append((ve, 0))
            gs.append(g)
            vi += 1
        else:
            for h in range(3):
                key = find_t(es, a + "value_projection.weight.q")
                gs.append(group_at(a + "v_dims.h%d" % h, key, 64 * h, 64, [key],
                                   [find_t(es, a + "value_projection.weight.scale")],
                                   [find_t(es, a + "output_projection.weight.q")], es[key].rows()))
        if l < NL - 4 or with_fw:
            key = find_t(es, b + "mlp.up.weight.q")
            n = es[key].rows()
            gs.append(group_at(b + "mlp", key, 0, n, [key], [find_t(es, b + "mlp.up.weight.scale")],
                               [find_t(es, b + "mlp.down.weight.q")], n))
    # the order in which the reader meets the key tensors (a stable sort)
    gs.sort(key=lambda g: (g.key, g.r0))
    return gs


def apply_perm(es, g: Group, perm: list) -> None:
    """new element j = old element perm[j]"""
    p = np.asarray(perm, dtype=np.int64)
    for t, o in g.rows_at:
        e = es[t]
        if e.a.ndim < 2:
            die("group %s: rows of the 1-d tensor %s" % (g.name, e.name))
        m = e.m2()
        m[o:o + g.n] = m[o:o + g.n][p].copy()
    for t, o in g.elem_at:
        f = es[t].a.reshape(-1)
        f[o:o + g.n] = f[o:o + g.n][p].copy()
    for t, o in g.cols_at:
        e = es[t]
        m = e.m2()
        if o + g.n > m.shape[1]:
            die("group %s: column block outside %s" % (g.name, e.name))
        m[:, o:o + g.n] = m[:, o:o + g.n][:, p].copy()


def key_rows(es, key: int, r0: int, n: int) -> list:
    m = es[key].m2()
    return [m[r0 + j].tobytes() for j in range(n)]


def zero_dead_key_dims(es, distinct: bool) -> tuple:
    """option dead (tf1_np_codec.cpp zero_dead_key_dims): a KDA key dim whose
    q, k and forget-gate down rows are all zero gets zero k and q convolution
    rows and dt_bias; with distinct the j-th further dead dim of a head gets
    the marker row with element j mod 4 = the f32 of high half 1 + j / 4"""
    kimi = es[find_t(es, "config.kimi")].a.reshape(-1).astype(np.int64).tolist()
    dead = qmark = 0
    for l, kd in enumerate(kimi):
        if not kd:
            continue
        a = "blocks.%d.attention." % l
        q = es[find_t(es, a + "query_projection.weight.q")].m2()
        k = es[find_t(es, a + "key_projection.weight.q")].m2()
        fd = es[find_t(es, a + "forget_gate_projection.down.weight.q")].m2()
        qc = es[find_t(es, a + "query_convolution.weight")]
        kc = es[find_t(es, a + "key_convolution.weight")].m2()
        dt = es[find_t(es, a + "dt_bias")].a.reshape(-1)
        if qc.code != wc.DTYPE_FLOAT32 or qc.cols() != 4:
            die("%squery_convolution.weight: f32 [*, 4] expected" % a)
        qcu = qc.m2().view(np.uint32)
        extra = [-1, -1, -1]
        for r in range(q.shape[0]):
            if q[r].any() or k[r].any() or fd[r].any():
                continue
            dead += 1
            kc[r] = 0
            dt[r] = 0
            qcu[r] = 0
            j = extra[r // 64]
            if distinct and j >= 0:
                qcu[r, j % 4] = (1 + j // 4) << 16
                qmark += 1
            extra[r // 64] = j + 1
    return dead, qmark


# ------------------------------------------------------------------ containers
def encode_plain(es, k: bool) -> bytes:
    """the plain v5 container (one stream)"""
    enc, models, tmpl = new_state()
    for e, en in zip(es, encodings(es, k)):
        cn = code_head(enc, models, tmpl, e, en)
        code_values(enc, models, e, en, cn)
    return wc.MAGIC_V5 + struct.pack("<I", len(es)) + enc.finish()


def capacity_bytes(gs) -> int:
    """whole bytes the groups can carry: floor((bitlen(prod n!) - 1) / 8)"""
    P = 1
    for g in gs:
        for i in range(2, g.n + 1):
            P *= i
    return (P.bit_length() - 1) // 8


def encode_np(es, k: bool, gs, np2: bool) -> tuple:
    """the NP / NP2 container:
         magic | u32 n | u32 nA | u32 lenA | u32 K | A[1..] | B[1+K..]
    stream A = the tensors up to the last one a group touches, stream B = the
    rest; B[1..K] are carried by the orders.  Returns (bytes, K, permuted)."""
    es = [Ent(e.name, e.code, e.a) for e in es]
    inA = [False] * len(es)
    keyflag = [0] * len(es)
    for g in gs:
        for v in (g.rows_at, g.elem_at, g.cols_at):
            for t, _ in v:
                inA[t] = True
                if not np2 and es[t].code == wc.DTYPE_FLOAT32:
                    die("group %s moves the f32 tensor %s: format NP2 only" % (g.name, es[t].name))
        inA[g.key] = True
        if g.flag not in (FLAG_ALL, FLAG_BLOCKS) and not (np2 and g.flag == FLAG_FIRST):
            die("group %s: key flag 0x%02x" % (g.name, g.flag))
        if keyflag[g.key] and keyflag[g.key] != g.flag:
            die("tensor %s: two key flags" % es[g.key].name)
        keyflag[g.key] = g.flag
    enc = encodings(es, k)   # an order does not change them
    last = max(i for i in range(len(es)) if inA[i])
    ia = list(range(last + 1))
    ib = list(range(last + 1, len(es)))
    # stream B: the models after the unpermuted A, minus the resets
    scratch, models, tmpl = new_state()
    for i in ia:
        cn = code_head(scratch, models, tmpl, es[i], enc[i] | keyflag[i])
        code_values(scratch, models, es[i], enc[i], cn)
    reset_for_b(models, np2)
    eb = wc.BinEncoder()
    for i in ib:
        cn = code_head(eb, models, tmpl, es[i], enc[i])
        code_values(eb, models, es[i], enc[i], cn)
    B = eb.finish()
    K = capacity_bytes(gs)
    if len(B) < K + 2:
        die("stream B too short for K")
    if B[0] != 0:
        die("stream B does not start with the zero cache byte")
    # the carried number; the reader folds the digits in file order (Horner
    # x = x * radix + digit), so they come out of X from the last one
    X = int.from_bytes(B[1:1 + K], "little")
    digs = [[0] * g.n for g in gs]
    for gi in range(len(gs) - 1, -1, -1):
        n = gs[gi].n
        for j in range(n - 1, -1, -1):
            X, digs[gi][j] = divmod(X, n - j)
    if X:
        die("carried number does not fit")
    for gi, g in enumerate(gs):
        kr = key_rows(es, g.key, g.r0, g.n)
        idx = sorted(range(g.n), key=lambda x: kr[x])
        for j in range(1, g.n):
            if kr[idx[j - 1]] == kr[idx[j]]:
                die("group %s: two equal key rows, the order is not readable" % g.name)
        perm = [idx.pop(d) for d in digs[gi]]
        apply_perm(es, g, perm)
    ea, models, tmpl = new_state()
    for i in ia:
        cn = code_head(ea, models, tmpl, es[i], enc[i] | keyflag[i])
        code_values(ea, models, es[i], enc[i], cn)
    A = ea.finish()
    if A[0] != 0:
        die("stream A does not start with the zero cache byte")
    out = (MAGIC_NP2 if np2 else MAGIC_NP) + struct.pack("<4I", len(es), len(ia), len(A) - 1, K)
    return out + A[1:] + B[1 + K:], K, es


def read_container(path: str) -> list:
    """the tensors of a v5, NP or NP2 container, in file order"""
    data = open(path, "rb").read()
    magic = data[:8]
    if magic == wc.MAGIC_V5:
        return ents_of(wc.read_tensor_file_v3(path))
    if magic not in (MAGIC_NP, MAGIC_NP2):
        die("%s: magic %r is neither v5 nor NP/NP2" % (path, magic))
    np2 = magic == MAGIC_NP2
    n, nA, lenA, K = struct.unpack("<4I", data[8:24])
    A = b"\0" + data[24:24 + lenA]
    tail = data[24 + lenA:]
    dec = wc.BinDecoder(A)
    models = wc._ModelsV3(v4=True, v5=True)
    tmpl = wc._Template()
    known = {}
    es = []
    x = 0
    for _ in range(nA):
        e, enc_byte = decode_tensor(dec, models, tmpl, known)
        es.append(e)
        flag = enc_byte & FLAG_MASK
        if not flag:
            continue
        # the digits of the stored order: per group, digit j is the rank of the
        # stored element among the remaining ones in bytewise key-row order
        if flag == FLAG_ALL:
            blocks = [(0, e.rows())]
        elif flag == FLAG_BLOCKS:
            blocks = [(r0, 64) for r0 in range(0, e.rows(), 64)]
        else:
            blocks = [(0, 64)]
        for r0, cnt in blocks:
            kr = key_rows(es, len(es) - 1, r0, cnt)
            rest = sorted(kr)
            for j in range(cnt):
                d = rest.index(kr[j])
                rest.pop(d)
                x = x * (cnt - j) + d
    if x.bit_length() > 8 * K:
        die("carried number longer than K = %d B" % K)
    reset_for_b(models, np2)
    dec = wc.BinDecoder(b"\0" + x.to_bytes(K, "little") + tail)
    for _ in range(n - nA):
        es.append(decode_tensor(dec, models, tmpl, known)[0])
    return es


def write_raw(es, path: str) -> None:
    """the decoded tensors raw in file order (FX2TFW01 layout)"""
    with open(path, "wb") as f:
        f.write(b"FX2TFW01" + struct.pack("<I", len(es)))
        for e in es:
            nm = e.name.encode()
            f.write(struct.pack("<I", len(nm)) + nm + bytes([e.code]))
            f.write(struct.pack("<I", e.a.ndim) + struct.pack("<%dI" % e.a.ndim, *e.a.shape))
            f.write(e.a.tobytes())


# ------------------------------------------------------------------ main
def main() -> None:
    av = sys.argv[1:]
    if len(av) >= 3 and av[0] == "np":
        src, out = av[1], av[2]
        kopt, fw, np2, dead = None, False, False, False
        for o in av[3:]:
            if o == "k":
                kopt = True
            elif o == "nok":
                kopt = False
            elif o == "fw":
                fw = True
            elif o == "np2":
                np2 = True
            elif o == "dead":
                dead = True
            else:
                die("unknown option %s" % o)
        if os.path.exists(out):
            die("refusing to overwrite %s" % out)
        orig = open(src, "rb").read()
        if orig[:8] != wc.MAGIC_V5:
            die("%s: not a v5 container (magic %r)" % (src, orig[:8]))
        es = ents_of(wc.read_tensor_file_v3(src))
        kin = None
        for kk in (False, True):
            if encode_plain(es, kk) == orig:
                kin = kk
                break
        if kin is None:
            die("STOP: the v5 writer does NOT reproduce %s bit for bit (with and without "
                "ENC_F32HC); nothing was written" % src)
        print("identity probe (v5 writer%s): BIT IDENTICAL, %d B" % (", ENC_F32HC" if kin else "",
                                                                      len(orig)))
        k = kin if kopt is None else kopt
        if dead:
            nd, qm = zero_dead_key_dims(es, np2)
            print("dead KDA key dims: %d zeroed (k conv, dt_bias, q conv; %d q conv marker rows)"
                  % (nd, qm))
        gs = np_groups(es, fw, np2)
        blob, K, perm = encode_np(es, k, gs, np2)
        with open(out, "wb") as f:
            f.write(blob)
        back = read_container(out)
        same = len(back) == len(perm) and all(b.same(p) for b, p in zip(back, perm))
        moved = any(e.a.tobytes() != p.a.tobytes() for e, p in zip(es, perm))
        print("read back %s against the permuted tensors: %s (%d tensors), permutation %s"
              % ("NP2" if np2 else "NP", "BIT IDENTICAL" if same else "DIFFERENT", len(back),
                 "non-trivial" if moved else "TRIVIAL"))
        if not same or not moved:
            os.remove(out)
            die("STOP: output removed")
        print("groups %d%s%s, carried K = %d B, output %s: %d B, sha256 %s"
              % (len(gs), " (with the fast-weight layers)" if fw else "",
                 " (with the KDA dims)" if np2 else "", K, out, len(blob),
                 hashlib.sha256(blob).hexdigest()))
        print("tensors (permuted) sha256 %s" % tensors_sha(perm))
        return
    if len(av) in (2, 3) and av[0] == "read":
        es = read_container(av[1])
        print("%d tensors, tensors sha256 %s" % (len(es), tensors_sha(es)))
        if len(av) == 3:
            write_raw(es, av[2])
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
