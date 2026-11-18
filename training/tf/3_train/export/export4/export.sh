#!/usr/bin/env bash
# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/export/export4/export.sh, sha256 5a16a1e9a030f67b), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
# export.sh: from the reference weight file of the trained transformer to the
# container in format v4 (format v5 and NP2 follow, README section 6).  CPU
# only, deterministic, a few minutes.
#
#   TREE=<training tree> bash tf/3_train/export/export4/export.sh <in.bin> <out-dir>
#
# <in.bin> is the reference weight file (magic FX2TFW01) that export_v3_ar.py
# writes from the last checkpoint, next to its manifest <in.bin>.json.  TREE
# is the assembled training tree (README section 1); the
# tools import its pysrc package.  PYTHON defaults to python3, which needs numpy
# and torch (the CPU build is enough).  On a host without the GPU libraries of
# the training tree the script provides stand-ins for the three names
# pysrc/model.py imports from them (never called here) and for the annotation
# package jaxtyping.
#
# Steps, each one checked by the tool that performs it:
#   1. round the raw fp32 tensors to bfloat16            wpack2_bf16.py
#   2. write container format v3 (round trip verified)   weights_compress_v3.py compress3
#   3. drop the structurally dead MLP neurons             prune_dead_neurons.py
#   4. zero the collapsed attention rows, write format v4 convert_container_v4.py
#
# For the transformer of cmix-neif-2.0 the expected sha256 values are (README section 6):
#   in.bin    fa0ea606ff8fda91784b6b3ae1c875ed5e6ed5f4a39e87cbf91d45403361392c
#   step 2    2bdc20aa118eb1ba141dc89f024afff7e2795fd74ba4d5c310aa7da60b529008
#   step 3    the same (no MLP neuron is structurally dead)
#   step 4    38b7c162f1882ca6aedfa8f7831bb55b52d37582a3892a340605cf0c979b54e8
set -euo pipefail
IN="${1:?usage: TREE=<training tree> export.sh <in.bin> <out-dir>}"
OUT="${2:?usage: TREE=<training tree> export.sh <in.bin> <out-dir>}"
TREE="${TREE:?set TREE to the assembled training tree}"
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -s "$IN" ] || { echo "missing $IN" >&2; exit 2; }
[ -s "$IN.json" ] || { echo "missing manifest $IN.json (container and manifest are a pair)" >&2; exit 2; }
[ -f "$TREE/pysrc/export_weights.py" ] || { echo "no pysrc under $TREE" >&2; exit 2; }
mkdir -p "$OUT"
B="$(basename "$IN" .bin)"

STUB=""
if ! (cd "$TREE" && "$PYTHON" -c "import fla" 2>/dev/null); then
  STUB="$(mktemp -d)"
  mkdir -p "$STUB/fla/ops/kda" "$STUB/fla/modules/conv"
  : > "$STUB/fla/__init__.py"
  : > "$STUB/fla/ops/__init__.py"
  printf 'def chunk_kda(*a, **k):\n    raise RuntimeError("CPU stand-in: chunk_kda is not available")\n' \
    > "$STUB/fla/ops/kda/__init__.py"
  printf 'from torch.nn import Module\n\n\nclass FusedRMSNormGated(Module):\n    def __init__(self, *a, **k):\n        super().__init__()\n\n    def forward(self, *a, **k):\n        raise RuntimeError("CPU stand-in: FusedRMSNormGated is not available")\n' \
    > "$STUB/fla/modules/__init__.py"
  : > "$STUB/fla/modules/conv/__init__.py"
  printf 'def causal_conv1d(*a, **k):\n    raise RuntimeError("CPU stand-in: causal_conv1d is not available")\n' \
    > "$STUB/fla/modules/conv/causal_conv1d.py"
  trap 'rm -rf "$STUB"' EXIT
fi
if ! (cd "$TREE" && "$PYTHON" -c "import jaxtyping" 2>/dev/null); then
  # pysrc/model.py uses jaxtyping for type annotations only
  [ -n "$STUB" ] || { STUB="$(mktemp -d)"; trap 'rm -rf "$STUB"' EXIT; }
  printf 'class _T:\n    def __class_getitem__(cls, item):\n        return object\n\n\nFloat = Int = Bool = _T\n' \
    > "$STUB/jaxtyping.py"
fi
export PYTHONPATH="$TREE${STUB:+:$STUB}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== 1/4 bf16 rounding of the raw fp32 tensors"
"$PYTHON" "$HERE/wpack2_bf16.py" "$IN" "$OUT/$B.bin.bf16"

echo "=== 2/4 container format v3"
"$PYTHON" "$HERE/weights_compress_v3.py" compress3 "$OUT/$B.bin.bf16" "$OUT/${B}_v3bf16.c3"
cp -f "$IN.json" "$OUT/${B}_v3bf16.bin.json"

echo "=== 3/4 drop the structurally dead MLP neurons"
"$PYTHON" "$HERE/prune_dead_neurons.py" "$OUT/${B}_v3bf16.c3" "$OUT/${B}_pruned.c3" \
  --report "$OUT/${B}_pruned.tsv"
# 2.0: no dead neuron, the tool keeps its input and writes no output file
[ -f "$OUT/${B}_pruned.c3" ] || cp -f "$OUT/${B}_v3bf16.c3" "$OUT/${B}_pruned.c3"

echo "=== 4/4 zero the collapsed attention rows, container format v4"
"$PYTHON" "$HERE/convert_container_v4.py" "$OUT/${B}_pruned.c3" "$OUT/${B}_pruned_v4.c4" \
  --report "$OUT/${B}_zeroed_rows.tsv"
cp -f "$IN.json" "$OUT/${B}_pruned_v4.bin.json"

echo
( cd "$OUT" && sha256sum "${B}_v3bf16.c3" "${B}_pruned.c3" "${B}_pruned_v4.c4" "${B}_pruned_v4.bin.json" )
