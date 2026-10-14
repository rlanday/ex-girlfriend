#!/usr/bin/env bash
# export.sh: from the reference weight file of the trained transformer to the
# container this entry ships.  CPU only, deterministic, a few minutes.
#
#   TREE=<training tree> bash tf/4_export/export.sh <in.bin> <out-dir>
#
# <in.bin> is the reference weight file (magic FX2TFW01) that run/export_chain.sh
# of the training tree writes from the last checkpoint, next to its manifest
# <in.bin>.json.  TREE is the assembled training tree (README section 1); the
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
# For the shipped transformer the expected sha256 values are (README section 6):
#   in.bin    614e2ea500fd5bbb006e3759d019d5d9d2f4a9677d47a7808d575fad7473a3ac
#   step 2    8cff9bb1c695fafd6e4fb390e18108ed4c418f03e04761e527a755936c34653b
#   step 3    aa1b266f1aa2d54dd86a4d013405639980ca67631e77123e351ed0380ee3130d
#   step 4    ba982cd5da95c88c076d63b7767dad12e0c510ea3dd99adc0e213e707b45e719
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

echo "=== 4/4 zero the collapsed attention rows, container format v4"
"$PYTHON" "$HERE/convert_container_v4.py" "$OUT/${B}_pruned.c3" "$OUT/${B}_pruned_v4.c4" \
  --report "$OUT/${B}_zeroed_rows.tsv"
cp -f "$IN.json" "$OUT/${B}_pruned_v4.bin.json"

echo
( cd "$OUT" && sha256sum "${B}_v3bf16.c3" "${B}_pruned.c3" "${B}_pruned_v4.c4" "${B}_pruned_v4.bin.json" )
