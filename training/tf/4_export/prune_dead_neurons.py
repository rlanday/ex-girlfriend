#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Derived work.  The container format (magic FX2TFWC3), the range coder and the
# bit trees this script reads and writes are `weights_compress_v3.py` next to
# this file, which is itself a fork of `pysrc/weights_compress.py` from
# fx2-cmix-transformer by Vladimir Ivanov (GPL-3.0); formats v1 and v2, the
# range coder, the bit trees and the RoPE recomputation are his.  The
# quantization arithmetic whose closed form is evaluated below (biased u8
# activation, folded row scale, relu squared epilogue) is likewise his, in
# `cpp_infer/src/opt/qmat.h` and `cpp_infer/src/opt/qmat_dense.cpp`.  Our
# addition is this script: the structural deadness test and the pruning.
# See ../../ATTRIBUTION.md.
#
# Copyright (C) 2026 Vladimir Ivanov (original)
# Copyright (C) 2026 Marcus Woelk (this script)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.  See the LICENSE file of this entry for
# the full text.

"""remove the structurally dead MLP neurons from a finished v3 container

A neuron of an MLP block is STRUCTURALLY DEAD when its quantized output is 0
for EVERY possible input, not merely for the corpus we happened to measure.
On the 8M transformer of this entry 70 of 12,288 neurons are dead that way:
training collapsed their row scale `w_scale`, and the int4 row behind it is
noise.  They cost weight bytes and contribute nothing.

The test is a closed form on the container alone, taken from the inference
kernels (qmat.h section 1, qgemv_relu2q in qmat_dense.cpp, mlp_block in
model_opt.cpp, rms_norm_quant192 in glue.cpp):

  * the activation entering `mlp.up` is BIASED u8: act_u8 = qa + 128 with qa in
    [-128, 127] (rms_norm_quant192 rounds and clamps),
  * the kernel computes  dot[o] = SUM_i act_u8[i]*qw[o][i] - corr[o]  with
    corr[o] = 128*SUM_i qw[o][i], hence exactly dot[o] = SUM_i qa[i]*qw[o][i],
  * y[o] = fold[o] * float(dot[o])  with
    fold[o] = s_act * bf16_to_f32(w_scale[o]),
  * the epilogue is q[o] = cvtps_epi32(min(relu(y[o])^2 / s_next, 127)) with
    round to nearest even, s_next = mlp.down.quantize_activation.scale.

  => q[o] == 0 for EVERY input as soon as

        max_qa relu(y[o])^2 / s_next < 0.5,  i.e.  max_y[o] < sqrt(0.5*s_next)

  and max_y has a closed form, because each qa is free in [-128, 127]:

        fold > 0:  max_dot = SUM_i ( qw_i > 0 ? 127*qw_i : 128*(-qw_i) )
        fold < 0:  min_dot = -SUM_i ( qw_i > 0 ? 128*qw_i : 127*(-qw_i) ),
                   max_y   = fold * min_dot

This is an UPPER bound: the 192 positions are not independently free, qa comes
out of an rms norm.  So the test is conservative in the safe direction.  When
it says "dead", the neuron is provably dead; when it says "could live", the
neuron is kept.  No sampling, no corpus, no threshold to tune.

Pruning a dead neuron j of layer l means:

  * drop row j of `blocks.<l>.mlp.up.weight.q`      (and of `.weight.scale`,
    and of `.bias` where a geometry carries one),
  * drop column j of `blocks.<l>.mlp.down.weight.q`.

`config.ints` is NOT touched: position 5 stays the MLP width the geometry was
trained with and remains the upper bound the loader checks against.  The
compressor of this entry takes each layer's width from the tensor form
instead of that constant.  A loader that expects the full width refuses a
pruned container loudly (shape mismatch); it never reads one wrongly.

Before anything is changed the script proves that this writer reproduces the
input container BIT FOR BIT.  If it does not, the container in hand was written
by a different coder and every byte count below would be meaningless, so the
script stops.

Deterministic: no randomness, no network, no environment.  Same input, same
output, byte for byte.

Usage, from the tree root (so that `pysrc` is importable; a torch free stand in
is used automatically when it is not):

  python3 tf/4_export/prune_dead_neurons.py <in.c3> <out.c3> [--report <tsv>]
  python3 tf/4_export/prune_dead_neurons.py --list <in.c3>
"""

import os
import struct
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _install_export_weights_stub():
    """put a torch free stand in for `pysrc.export_weights` on the module path

    `weights_compress_v3` imports four dtype constants, the dtype table and
    `read_tensor_file` from there.  All five are plain struct/numpy; the rest of
    that module is the checkpoint exporter and pulls in torch and the whole
    model.  This script never touches a checkpoint, so when the real module is
    not importable we supply exactly those five names, with the same values and
    the same file format (SPEC.md section 4).
    """
    import types

    magic = b"FX2TFW01"
    m = types.ModuleType("pysrc.export_weights")
    m.MAGIC = magic
    m.DTYPE_INT8 = 0
    m.DTYPE_BF16_BITS = 1
    m.DTYPE_FLOAT32 = 2
    m.DTYPE_INT32 = 3
    m._DTYPE_TO_NUMPY = {0: np.int8, 1: np.uint16, 2: np.float32, 3: np.int32}

    def read_tensor_file(filename):
        with open(filename, "rb") as f:
            assert f.read(8) == magic
            (n_tensors,) = struct.unpack("<I", f.read(4))
            tensors = {}
            for _ in range(n_tensors):
                (name_length,) = struct.unpack("<I", f.read(4))
                name = f.read(name_length).decode()
                (dtype_code,) = struct.unpack("<B", f.read(1))
                (ndim,) = struct.unpack("<I", f.read(4))
                shape = struct.unpack("<%dI" % ndim, f.read(4 * ndim))
                dtype = np.dtype(m._DTYPE_TO_NUMPY[dtype_code])
                count = int(np.prod(shape)) if shape else 1
                array = np.frombuffer(
                    f.read(count * dtype.itemsize), dtype=dtype
                ).reshape(shape)
                assert name not in tensors
                tensors[name] = (dtype_code, array)
            assert f.read(1) == b""
            return tensors

    m.read_tensor_file = read_tensor_file
    pkg = sys.modules.get("pysrc")
    if pkg is None:
        pkg = types.ModuleType("pysrc")
        pkg.__path__ = []
        sys.modules["pysrc"] = pkg
    pkg.export_weights = m
    sys.modules["pysrc.export_weights"] = m


try:
    import pysrc.export_weights  # noqa: F401
except Exception:
    _install_export_weights_stub()

import weights_compress_v3 as wc3  # noqa: E402


# ---------------------------------------------------------------------------
# container in, container out
# ---------------------------------------------------------------------------
def read_container(path):
    """the tensor set of a v3 container, in FILE ORDER

    `read_tensor_file_v3` returns a dict; dicts keep insertion order, and the
    decoder inserts in file order, so the order the writer needs is preserved.
    """
    tensors = wc3.read_tensor_file_v3(path)
    entries = [(name, code, array) for name, (code, array) in tensors.items()]
    return tensors, entries


def write_container(path, entries):
    wc3.write_tensor_file_v3(path, entries)
    return os.path.getsize(path)


def identity_probe(path, entries):
    """does THIS writer reproduce THIS container bit for bit?

    Everything below is a byte count against the input file, so a writer that
    is not the one that produced it would turn the whole report into fiction.
    """
    fd, tmp = tempfile.mkstemp(prefix="prune_identity_", suffix=".c3",
                               dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        write_container(tmp, entries)
        with open(path, "rb") as a, open(tmp, "rb") as b:
            return a.read() == b.read()
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# the structural deadness test
# ---------------------------------------------------------------------------
def bf16_to_f32(bits):
    """fp32 values of raw bfloat16 bits (bits << 16 reinterpreted as float)"""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def mlp_layers(tensors):
    return sorted(
        int(name.split(".")[1])
        for name in tensors
        if name.startswith("blocks.") and name.endswith(".mlp.up.weight.q")
    )


def dead_neurons(tensors):
    """per layer: the dead neuron indices, and the numbers behind the verdict

    Returns (dead, stats) with dead[l] a sorted list of neuron indices and
    stats[l] = (width, s_act, s_next, threshold, min_max_y, median_max_y).
    """
    dead = {}
    stats = {}
    for layer in mlp_layers(tensors):
        b = "blocks.%d." % layer
        q = np.asarray(tensors[b + "mlp.up.weight.q"][1], dtype=np.int32)
        w_scale = bf16_to_f32(
            np.asarray(tensors[b + "mlp.up.weight.scale"][1]).ravel()
        )
        s_act = float(
            bf16_to_f32(
                np.asarray(
                    tensors[b + "mlp.up.quantize_activation.scale"][1]
                ).ravel()
            )[0]
        )
        s_next = float(
            bf16_to_f32(
                np.asarray(
                    tensors[b + "mlp.down.quantize_activation.scale"][1]
                ).ravel()
            )[0]
        )
        fold = (np.float32(s_act) * w_scale).astype(np.float32)
        pos = np.where(q > 0, q, 0).sum(axis=1)   # sum of the positive weights
        neg = np.where(q < 0, -q, 0).sum(axis=1)  # sum of |negative weights|
        max_dot = 127 * pos + 128 * neg     # qa = +127 where qw>0, -128 where <0
        min_dot = -(128 * pos + 127 * neg)  # the other extreme
        max_y = np.where(fold >= 0, fold * max_dot, fold * min_dot).astype(
            np.float64
        )
        threshold = float(np.sqrt(0.5 * s_next))
        dead[layer] = [int(j) for j in np.flatnonzero(max_y < threshold)]
        stats[layer] = (
            int(q.shape[0]),
            s_act,
            s_next,
            threshold,
            float(np.min(max_y)),
            float(np.median(max_y)),
        )
    return dead, stats


# ---------------------------------------------------------------------------
# the pruning
# ---------------------------------------------------------------------------
_UP_ROW_SUFFIXES = (
    ".mlp.up.weight.q",
    ".mlp.up.weight.scale",
    ".mlp.up.bias",
)


def prune(entries, dead):
    """drop the dead rows of mlp.up and the dead columns of mlp.down

    Nothing else changes, in particular not `config.ints`: position 5 keeps the
    width the geometry was trained with, and stays the upper bound the loader
    checks each layer against.
    """
    out = []
    dropped_rows = dropped_cols = 0
    for name, code, array in entries:
        if array is not None and name.startswith("blocks."):
            layer = int(name.split(".")[1])
            gone = dead.get(layer)
            if gone:
                if name.endswith(_UP_ROW_SUFFIXES):
                    keep = np.setdiff1d(np.arange(array.shape[0]), gone)
                    array = np.ascontiguousarray(array[keep])
                    if name.endswith(".mlp.up.weight.q"):
                        dropped_rows += len(gone)
                elif name.endswith(".mlp.down.weight.q"):
                    keep = np.setdiff1d(np.arange(array.shape[1]), gone)
                    array = np.ascontiguousarray(array[:, keep])
                    dropped_cols += len(gone)
        out.append((name, code, array))
    return out, dropped_rows, dropped_cols


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(dead, stats, out=sys.stdout):
    total = sum(len(v) for v in dead.values())
    neurons = sum(s[0] for s in stats.values())
    print("layers with an MLP block: %d" % len(stats), file=out)
    print(
        "layer\twidth\ts_act\t\ts_next\t\tthreshold\tdead\tmin(max_y)\t"
        "median(max_y)",
        file=out,
    )
    for layer in sorted(stats):
        width, s_act, s_next, threshold, lo, mid = stats[layer]
        print(
            "%d\t%d\t%.6g\t%.6g\t%.6g\t%d\t%.6g\t%.6g"
            % (layer, width, s_act, s_next, threshold, len(dead[layer]), lo, mid),
            file=out,
        )
    print(
        "STRUCTURALLY DEAD: %d of %d (%.4f %%)"
        % (total, neurons, 100.0 * total / neurons if neurons else 0.0),
        file=out,
    )
    for layer in sorted(dead):
        for j in dead[layer]:
            print("  layer %d neuron %d" % (layer, j), file=out)
    return total


def write_report_tsv(path, dead, stats):
    with open(path, "w") as f:
        f.write("# structurally dead MLP neurons, threshold sqrt(0.5*s_next)\n")
        f.write("layer\tneuron\twidth_before\twidth_after\tthreshold\n")
        for layer in sorted(dead):
            width, _sa, _sn, threshold, _lo, _mid = stats[layer]
            after = width - len(dead[layer])
            for j in dead[layer]:
                f.write("%d\t%d\t%d\t%d\t%.9g\n" % (layer, j, width, after,
                                                    threshold))


# ---------------------------------------------------------------------------
def main(argv):
    args = list(argv[1:])
    report_path = None
    if "--report" in args:
        i = args.index("--report")
        report_path = args[i + 1]
        del args[i:i + 2]
    list_only = False
    if "--list" in args:
        args.remove("--list")
        list_only = True
    if len(args) != (1 if list_only else 2):
        print(__doc__, file=sys.stderr)
        return 2
    src = args[0]

    tensors, entries = read_container(src)
    dead, stats = dead_neurons(tensors)
    total = report(dead, stats)
    if report_path:
        write_report_tsv(report_path, dead, stats)
        print("report written: %s" % report_path)
    if list_only:
        return 0

    dst = args[1]
    if os.path.exists(dst):
        print("refusing to overwrite %s" % dst, file=sys.stderr)
        return 3

    print("identity probe: writing %s back through this coder ..." % src)
    if not identity_probe(src, entries):
        print(
            "STOP: this writer does NOT reproduce %s bit for bit.  The "
            "container in hand was written by a different coder, so every byte "
            "count here would be meaningless.  Nothing was changed." % src,
            file=sys.stderr,
        )
        return 4
    print("identity probe: BIT IDENTICAL")

    if total == 0:
        print("no structurally dead neuron; keep the input container.")
        return 0

    pruned, rows, cols = prune(entries, dead)
    size_out = write_container(dst, pruned)
    size_in = os.path.getsize(src)
    print("pruned: %d rows of mlp.up, %d columns of mlp.down" % (rows, cols))
    print("widths after pruning: %s"
          % " ".join("%d:%d" % (l, stats[l][0] - len(dead[l]))
                     for l in sorted(stats)))
    print(
        "container %d B (input %d B, %+d B; %+d B in S, the container ships "
        "twice)" % (size_out, size_in, size_out - size_in,
                    2 * (size_out - size_in))
    )
    print("the compressor must take the MLP width per layer from the tensor form "
          "to read this container.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
