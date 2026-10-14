#!/usr/bin/env bash
# Only the remaining phases on an existing bf16 state: qat, fp32 and the export.
#   BAUM=/root/fx6m_8m ENVD=/root/chain/09_qat.env KARTEN=0,1,2,3 PORT=29931 \
#     bash /root/qat_arm.sh
# The bf16 state must be in place as models/<geometry>-pkd.tch. At the end the
# marker /root/ARM_<tag>_FERTIG ("finished") is written.
set -u
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin"
cd "$BAUM" || exit 2
set -a; . "$ENVD"; set +a
# qat always starts from models/<geometry>-pkd.tch (the bf16 end), never from
# a warm start named in the env file; the continued-schedule variables belong
# to bf16 (same rule as in kette_arm.sh before qat).
unset TFT_WARMSTART TFT_WARMSTART_OPT TFT_SCHED_TOTAL_EPOCHS TFT_SCHED_DONE_EPOCHS
export CUDA_VISIBLE_DEVICES="$KARTEN"
export TFT_GPUS=$(printf '%s' "$KARTEN" | tr ',' '\n' | grep -c .)
echo "### QAT-ARM $TFT_RUN_TAG  cards $KARTEN ($TFT_GPUS)  $(date -u +%F' '%T) UTC"
echo "### geometry $TFT_GEOMETRY, package $PK_DUMP_DIR, qat $TFT_EPOCHS_QAT, fp32 $TFT_EPOCHS_FP32"
env | grep -E '^(PK_|ZCTX_|TFT_)' | sort
ls -l "models/${TFT_GEOMETRY}-pkd.tch" || { echo "### bf16 state missing"; exit 2; }
echo "### qat starts from: $(sha256sum models/${TFT_GEOMETRY}-pkd.tch | cut -c1-16)"
uv run python -m tft.dataset check "$TFT_DATA_DIR" 208 | tail -3
echo "### qat ($TFT_EPOCHS_QAT epochs)  $(date -u +%T) UTC"
uv run torchrun --nproc_per_node "$TFT_GPUS" --master_port "$PORT" -m tft.quantize_poc || { echo "### QAT FAILED rc=$?"; exit 3; }
echo "### fp32 ($TFT_EPOCHS_FP32 epochs)  $(date -u +%T) UTC"
uv run torchrun --nproc_per_node "$TFT_GPUS" --master_port "$((PORT+1))" -m tft.cooldown_poc || { echo "### FP32 FAILED rc=$?"; exit 4; }
bash run/export_chain.sh || { echo "### EXPORT FAILED rc=$?"; exit 5; }
echo "### QAT-ARM $TFT_RUN_TAG DONE  $(date -u +%F' '%T) UTC"
touch "/root/ARM_${TFT_RUN_TAG}_FERTIG"
