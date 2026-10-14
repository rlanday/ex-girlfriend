#!/usr/bin/env bash
# the actual PoC run: calibrate, then the three phases, then the export chain
#
#   TFT_BUDGET_HOURS=12 bash run/night.sh
#
# Start it inside tmux.  It writes everything to logs/ and models/ on the
# network volume, so a dropped ssh session costs nothing and a dead pod costs
# at most the epoch in flight (checkpoints are written every epoch).
set -euo pipefail

export TFT_GEOMETRY="${TFT_GEOMETRY:-poc4m}"
export TFT_DATA_DIR="${TFT_DATA_DIR:-data}"
export TFT_LOG_DIR="${TFT_LOG_DIR:-logs}"
export TFT_RUN_TAG="${TFT_RUN_TAG:-$(date +%Y%m%d-%H%M)}"
GPUS="${TFT_GPUS:-1}"
BUDGET="${TFT_BUDGET_HOURS:-12}"

mkdir -p "$TFT_LOG_DIR" models
exec > >(tee -a "$TFT_LOG_DIR/night-$TFT_RUN_TAG.log") 2>&1

echo "### night run  geometry=$TFT_GEOMETRY gpus=$GPUS budget=${BUDGET}h tag=$TFT_RUN_TAG"
date -u

echo "=== 0/5  data check (the interface, not a formality: a wrong prior width"
echo "         does not fail, it silently trains a differently shaped model)"
uv run python -m tft.dataset check "$TFT_DATA_DIR" 208

echo "=== 1/5  calibrate"
if [ -z "${TFT_EPOCHS_BF16:-}" ]; then
  TFT_BUDGET_HOURS="$BUDGET" uv run torchrun --nproc_per_node "$GPUS" -m tft.calibrate \
    | tee "$TFT_LOG_DIR/calibrate-$TFT_RUN_TAG.txt"
  # the last line of calibrate is an `export TFT_EPOCHS_...` statement
  eval "$(grep '^export TFT_EPOCHS_' "$TFT_LOG_DIR/calibrate-$TFT_RUN_TAG.txt" | tail -n 1)"
fi
# fall back to the planned defaults of tft/config.py if calibration was skipped
export TFT_EPOCHS_BF16="${TFT_EPOCHS_BF16:-55}"
export TFT_EPOCHS_QAT="${TFT_EPOCHS_QAT:-22}"
export TFT_EPOCHS_FP32="${TFT_EPOCHS_FP32:-3}"
echo "epochs: bf16=$TFT_EPOCHS_BF16 qat=$TFT_EPOCHS_QAT fp32=$TFT_EPOCHS_FP32"

echo "=== 2/5  phase bf16 (no quantization, the bulk of the budget)"
uv run torchrun --nproc_per_node "$GPUS" -m tft.train_poc

echo "=== 3/5  phase qat (int4 weights / int8 activations)"
uv run torchrun --nproc_per_node "$GPUS" -m tft.quantize_poc

echo "=== 4/5  phase fp32 (cooldown; this checkpoint is what gets exported)"
uv run torchrun --nproc_per_node "$GPUS" -m tft.cooldown_poc

echo "=== 5/5  export chain"
bash run/export_chain.sh

date -u
echo "### done.  Deliver the .c2, its .json and the .vocab-bitmap.u8 to the compressor build."
echo "### The loss to quote is the LAST train/epoch_loss of the fp32 phase log;"
echo "### it is a TRAINING loss on the very data that will be compressed, which"
echo "### is the setting by design, but it is not comparable to the LSTM's"
echo "### 1.229 until the C++ side measures on the same window (see README)."
