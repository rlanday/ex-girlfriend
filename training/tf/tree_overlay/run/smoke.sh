#!/usr/bin/env bash
# end to end on a tiny window: three phases plus the whole export chain
#
# Purpose: show that the pipeline runs before spending a night on it.  Data,
# quantization-aware training and the tfwc2 container are the real ones; only
# the window and the epoch counts are small.
#
#   bash run/smoke.sh                 # cuts a 4M token window out of the package
#   TFT_SMOKE_TOKENS=1000000 bash run/smoke.sh
#
# Expect ~20-40 min on 1x RTX 5090.
set -euo pipefail

export TFT_GEOMETRY="${TFT_GEOMETRY:-poc4m}"
export TFT_RUN_TAG="${TFT_RUN_TAG:-smoke}"
SMOKE_TOKENS="${TFT_SMOKE_TOKENS:-4000000}"
SRC_DIR="${TFT_SRC_DATA_DIR:-data}"
export TFT_DATA_DIR="${TFT_DATA_DIR:-data-smoke}"
export TFT_LOG_DIR="${TFT_LOG_DIR:-logs}"

# his Logger refuses to append to an existing log file, so a repeated smoke
# run must clear its own logs first
rm -f "$TFT_LOG_DIR"/*"-$TFT_RUN_TAG.jsonl"

echo "=== 0/6  cut a $SMOKE_TOKENS token window out of $SRC_DIR"
if [ ! -f "$TFT_DATA_DIR/ppmd-bytes.uint8" ]; then
  uv run python -m tft.dataset cut "$SRC_DIR" "$TFT_DATA_DIR" "$SMOKE_TOKENS"
fi
uv run python -m tft.dataset check "$TFT_DATA_DIR" 208

echo "=== 1/6  phase bf16"
TFT_EPOCHS_BF16=2 uv run torchrun --nproc_per_node "${TFT_GPUS:-1}" -m tft.train_poc

echo "=== 2/6  phase qat"
TFT_EPOCHS_QAT=2 TFT_SCALE_STEPS=8 uv run torchrun --nproc_per_node "${TFT_GPUS:-1}" -m tft.quantize_poc

echo "=== 3/6  phase fp32"
TFT_EPOCHS_FP32=1 uv run torchrun --nproc_per_node "${TFT_GPUS:-1}" -m tft.cooldown_poc

echo "=== 4/6  export chain (weights.bin -> tfwc2 -> roundtrip -> compare)"
bash run/export_chain.sh

echo "=== 5/6  checkpoints and logs"
ls -l models/ | sed -n '1,20p'
ls -l "$TFT_LOG_DIR"/ | sed -n '1,20p'

echo "=== 6/6  loss trajectory (last line of each phase log)"
for f in "$TFT_LOG_DIR"/*"$TFT_RUN_TAG"*.jsonl; do
  echo "--- $f"
  tail -n 1 "$f"
done

echo
echo "SMOKE OK.  The loss from a 2 epoch run on a 4M token window means nothing;"
echo "what this proved is that the chain runs and produces a verified container."
