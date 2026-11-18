#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Marcus Woelk
#
# reproduce.sh: the training chain of the transformer of cmix-neif-2.0, stage
# by stage, with the configurations that were executed (chain/*.json, host
# paths replaced by <W>).
#
#   TFT_TREE=<training tree> PKG=<training package> W=<work directory> \
#     NGPU=8 bash reproduce.sh [stage ...]
#
# TFT_TREE  the assembled training tree (README section 1) with its Python
#           environment ($TFT_TREE/.venv/bin/python, or PY)
# PKG       the training package of stage (c), including ppmd-probs.float16
# W         work directory; it must hold start/beehive.c4, the weight
#           container of cmix-neif-beehive (3,649,198 B, sha256
#           141ee13c316811157bea167846b3f45472affff7adaa1b8b263e30621e490885)
# NGPU      cards per run (default 8); the shipped chain ran b1 on 4 cards and
#           every later stage on 8.  The learning-rate plans do not depend on the
#           card count; the order of the floating-point reductions does.
#
# Stages, in this order (default: all; a stage whose result exists is
# skipped):
#   start   container of cmix-neif-beehive -> bf16 start state (CPU)
#   b1 b2   100 bfloat16 epochs each (rx7q.py)
#   b3      400 bfloat16 epochs (rx7q.py): epochs 1 to 29 on a cosine plan,
#           then 30 to 400 at a constant rate (the run was stopped at the end
#           of epoch 29 and resumed with the second configuration)
#   fisher  allocation of 2, 3 or 4 bits per weight row (fisher_alloc_a.py, 1 card)
#   init    lifting of all-zero rows (mlp_lift.py), quantization-aware start
#           state with the moments of b3 (rx7r.py init_only)
#   qat     96 quantization-aware epochs with the size term (rx7r.py)
#   fp32    16 epochs at full precision and the strict deployed evaluation (rx3a.py)
#   export  reference weight file, bf16 rounding, v3, dead neurons, v4, v5
#           (the last step to format NP2 needs the writer of export/np2/,
#           README section 6)
set -euo pipefail
: "${TFT_TREE:?TFT_TREE missing (assembled training tree)}"
: "${PKG:?PKG missing (training package)}"
: "${W:?W missing (work directory)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$TFT_TREE/.venv/bin/python}"
NGPU="${NGPU:-8}"
TRUN="$PY -m torch.distributed.run"
PORT="${PORT:-30731}"
export TFT_TREE PKG
# environment of the training hosts (the training form of the package)
export TFT_GEOMETRY=poc8m_w2048 PK_DUMP_DIR="$PKG" TFT_DATA_DIR="$PKG" TFT_MMAP=1 ZCTX_ARM=pkd ZCTX_SEED=20260901 \
  PK_LAST_X0ROW=1 PK_REDIRMERGE=1 PK_SLOT_DIM=16 TFT_OPTIMIZER_BATCH_TOKENS=1048576 \
  TFT_MICRO_BATCH_TOKENS=131072 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
  RANK_CHECK_STEPS=20,1000

say() { echo "$(date -u '+%F %T') $*"; }
# cfg <chain file> <out file> [python expression applied to the dict c]
cfg() {
  "$PY" - "$HERE/chain/$1" "$2" "$W" "${3:-}" <<'PY'
import json, sys
src, dst, w, expr = sys.argv[1:5]
def sub(o):
    if isinstance(o, dict):
        return {k: sub(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sub(v) for v in o]
    return o.replace("<W>", w) if isinstance(o, str) else o
c = sub(json.load(open(src)))
if expr:
    exec(expr)
json.dump(c, open(dst, "w"), indent=1)
PY
}
run() {   # run <cards> <script> <cfg>: one distributed run in the directory of the training scripts
  # (the rank check of rx7q.py/rx7r.py ends the run with an error if the replicas differ)
  local n=$1 sc=$2 c=$3
  ( cd "$HERE/tree" && $TRUN --nproc_per_node "$n" --master_port "$PORT" "$sc" "$c" ) 2>&1 | tee -a "$(dirname "$c")/stdout.log"
}
finish_block() {   # the bf16 end of a block: model, optimizer state and step for the next block
  local o=$1
  mkdir -p "$o/ende"
  cp -f "$o/model.tch" "$o/ende/model.tch"
  cp -f "$o/state/opt.pt" "$o/ende/opt.pt"
  cp -f "$o/state/state.json" "$o/ende/state.json"
}

st_start() {
  [ -s "$W/start/start_bf16.tch" ] && return 0
  [ "$(sha256sum < "$W/start/beehive.c4" | cut -c1-64)" = \
    141ee13c316811157bea167846b3f45472affff7adaa1b8b263e30621e490885 ] \
    || { echo "start/beehive.c4 is not the container of cmix-neif-beehive" >&2; exit 2; }
  ( cd "$HERE" && CUDA_VISIBLE_DEVICES= "$PY" container_to_tch.py --container "$W/start/beehive.c4" \
      --cfg "$HERE/chain/00_start_arch.json" --strict "$W/start/start_strict.tch" --warm "$W/start/start_bf16.tch" \
      --wiederbelebe --sentinel-null --report "$W/start/start_conv.json" )
  grep -q '"mismatch": 0' "$W/start/start_conv.json" || { echo "integer check of the conversion failed" >&2; exit 1; }
}
st_block() {   # st_block <name> <chain file>
  local b=$1 o="$W/$1"
  [ -s "$o/ende/model.tch" ] && return 0
  mkdir -p "$o"
  cfg "$2" "$o/cfg.json"
  run "$NGPU" rx7q.py "$o/cfg.json"
  finish_block "$o"
}
st_b3() {
  local o="$W/b3"
  [ -s "$o/ende/model.tch" ] && return 0
  mkdir -p "$o"
  if [ ! -s "$o/state/state.json" ]; then
    cfg 03_b3_bf16_epochs_1_to_29.json "$o/cfg_1_to_29.json" 'c["stop_at"] = 29'
    run "$NGPU" rx7q.py "$o/cfg_1_to_29.json"
  fi
  cfg 03_b3_bf16_epochs_30_to_400.json "$o/cfg.json"
  run "$NGPU" rx7q.py "$o/cfg.json"
  finish_block "$o"
}
st_fisher() {
  local o="$W/t3/fisher"
  [ -s "$o/alloc_1.000.npz" ] && return 0
  mkdir -p "$o"
  cfg 04_fisher.json "$o/cfg.json"
  run 1 fisher_alloc_a.py "$o/cfg.json"
}
st_init() {
  local o="$W/t3/init"
  [ -s "$W/t3/qat/state/state.json" ] && return 0
  mkdir -p "$o/qat" "$W/t3/qat"
  "$PY" "$HERE/mlp_lift.py" "$W/b3/ende/model.tch" "$o/start.tch"
  cfg 05_qat_init.json "$o/cfg.json"
  run "$NGPU" rx7r.py "$o/cfg.json"
  cp -a "$o/qat/state" "$W/t3/qat/state"
}
st_qat() {
  local o="$W/t3/qat"
  [ -s "$o/result.json" ] && return 0
  cfg 06_qat.json "$o/cfg.json"
  run "$NGPU" rx7r.py "$o/cfg.json"
}
st_fp32() {
  local o="$W/t3/fp32"
  [ -s "$o/result.json" ] && return 0
  mkdir -p "$o"
  cfg 07_fp32.json "$o/cfg.json"
  run "$NGPU" rx3a.py "$o/cfg.json"
}
st_export() {
  local o="$W/t3/export"
  mkdir -p "$o/raw" "$o/c4"
  cfg 08_export.json "$o/cfg.json"
  ( cd "$HERE/export" && EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}" "$PY" export_v3_ar.py "$o/cfg.json" \
      "$W/t3/fp32/model.tch" "$o/raw" )
  TREE="$TFT_TREE" PYTHON="$PY" bash "$HERE/export/export4/export.sh" "$o/raw/weights-poc8m_w2048.bin" "$o/c4"
  "$PY" "$HERE/export/export4/convert_container_v5.py" "$o/c4/weights-poc8m_w2048_pruned_v4.c4" "$o/v5.c5"
  ( cd "$o" && sha256sum raw/weights-poc8m_w2048.bin c4/*.c3 c4/*.c4 v5.c5 )
}

STAGES="${*:-start b1 b2 b3 fisher init qat fp32 export}"
for s in $STAGES; do
  say "stage $s"
  case $s in
    start) st_start ;;
    b1) st_block b1 01_b1_bf16.json ;;
    b2) st_block b2 02_b2_bf16.json ;;
    b3) st_b3 ;;
    fisher) st_fisher ;;
    init) st_init ;;
    qat) st_qat ;;
    fp32) st_fp32 ;;
    export) st_export ;;
    *) echo "unknown stage $s" >&2; exit 2 ;;
  esac
done
say "done: $STAGES"
