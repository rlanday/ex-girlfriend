#!/usr/bin/env bash
# zctx: the two ablation twins, screening form (bf16 phase only)
#
#   bash zctx/run_zctx.sh ref     # the reference twin: poc4m, same seed
#   bash zctx/run_zctx.sh z1      # arm z1: article ordinal + stream position
#   bash zctx/run_zctx.sh z3      # arm z3: incoming link context prefix
#
# Run all three with the SAME ZCTX_SEED and the SAME TFT_EPOCHS_BF16.  Only the
# bf16 phase is run: the question is "does the feature lower the loss", and QAT
# plus cooldown answer a different one (they cost 7 of the 12 hours and change
# every arm by the same quantisation malus).
#
# If an earlier poc4m night run already exists AND was run with a ZCTX_SEED, its
# bf16 log is the reference and `ref` can be skipped.  Without a seed it cannot:
# his recipe does not seed, so an unseeded reference differs from the arm in its
# initialisation as well as in the feature.
#
# Cost per arm on 1x RTX 5090 at the planned 55 bf16 epochs: ~5.1 h ~ 5.10 $
# (arm z3 ~3.1 % more, it runs over 3.1 % more positions).
set -euo pipefail

ARM="${1:-}"
case "$ARM" in
  ref) export ZCTX_ARM=none ;;
  z1)  export ZCTX_ARM=z1 ;;
  z3)  export ZCTX_ARM=z3 ;;
  z1z3) export ZCTX_ARM=z1z3 ;;
  z4)  export ZCTX_ARM=z4 ;;
  z4c) export ZCTX_ARM=z4c ;;
  g0)  export ZCTX_ARM=g0 ;;
  g1)  export ZCTX_ARM=g1 ;;
  m1)  export ZCTX_ARM=m1 ;;
  pk)  export ZCTX_ARM=pk ;;   # the complete context package
  *) echo "usage: bash zctx/run_zctx.sh ref|z1|z3|z1z3|z4|z4c|g0|g1|m1|pk" >&2; exit 2 ;;
esac

export TFT_GEOMETRY="${TFT_GEOMETRY:-poc4m}"
export TFT_DATA_DIR="${TFT_DATA_DIR:-data}"
export TFT_LOG_DIR="${TFT_LOG_DIR:-logs}"
export TFT_EPOCHS_BF16="${TFT_EPOCHS_BF16:-55}"
# Read the priors into main memory instead of random reads over the network
# file system (see runner.py).  TFT_MMAP=1 overrides it.
export TFT_MMAP="${TFT_MMAP:-0}"
export ZCTX_SEED="${ZCTX_SEED:-20260901}"
export ZCTX_Z3_DIR="${ZCTX_Z3_DIR:-z3}"
export TFT_RUN_TAG="${TFT_RUN_TAG:-zctx-$ARM-$(date +%Y%m%d-%H%M)}"
GPUS="${TFT_GPUS:-1}"

mkdir -p "$TFT_LOG_DIR" models
exec > >(tee -a "$TFT_LOG_DIR/zctx-$ARM-$TFT_RUN_TAG.log") 2>&1

echo "### twin  arm=$ZCTX_ARM geometry=$TFT_GEOMETRY seed=$ZCTX_SEED"
echo "### epochs(bf16)=$TFT_EPOCHS_BF16 gpus=$GPUS data=$TFT_DATA_DIR"
date -u

echo "=== 0/2  interface checks (never start a twin on a package that moved)"
uv run python -m tft.dataset check "$TFT_DATA_DIR" 208
uv run python -m zctx.zgrammar verify "$TFT_DATA_DIR"
case "$ZCTX_ARM" in
  none|z1|z4|z4c|g0|g1|m1|pk) ;;   # no z3 stream needed (z4/z4c only shift the stream, see ZDataLoader in zloader.py)
  *) uv run python -m zctx.z3_extract check "$TFT_DATA_DIR" "$ZCTX_Z3_DIR" ;;
esac
if [ "$ZCTX_ARM" = "m1" ]; then
  uv run python -m zctx.zmatch verify "$TFT_DATA_DIR"
fi
if [ "$ZCTX_ARM" = "pk" ]; then
  # Check packing, match channel and the table directory BEFORE the GPU
  # runs; if one is missing, the arm silently measures the reference.
  uv run python -m zctx.zpaket
  uv run python -m zctx.zmatch verify "$TFT_DATA_DIR"
  : "${PK_DIR:?PK_DIR missing; without the table directory there is no package arm}"
  for d in z1t syn z3; do
    test -s "$PK_DIR/$d/z3-prefix-offsets.int32" || { echo "PK_DIR/$d missing" >&2; exit 3; }
  done
  for f in title-lengths.int32 reinject-index.int32 reinject-offsets.int32 z3-prior0.float16; do
    test -s "$PK_DIR/$f" || { echo "PK_DIR/$f missing" >&2; exit 3; }
  done
fi
uv run python -m zctx.zfeat selftest
if [ "$ZCTX_ARM" = "z4" ] || [ "$ZCTX_ARM" = "z4c" ]; then
  uv run python -m zctx.zlast "${ZCTX_LAST_X:-4}"
  uv run python -m zctx.zlast verify "$TFT_DATA_DIR"
fi
if [ "$ZCTX_ARM" = "g0" ] || [ "$ZCTX_ARM" = "g1" ]; then
  uv run python -m zctx.zgate selftest
fi

echo "=== 1/2  bf16 phase"
uv run torchrun --nproc_per_node "$GPUS" -m tft.train_poc

echo "=== 2/2  done; evaluate against the reference log with"
echo "    python -m zctx.eval_zctx compare <ref.jsonl> <arm.jsonl> --params <n> --sigma <s>"
date -u
