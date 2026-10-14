#!/usr/bin/env bash
# fp32 with TFT_EPOCHS_FP32 epochs on the qat end, then the export.
#   BAUM=/root/fx6m_8m ENVD=/root/chain/10_fp32.env bash /root/fp32_arm.sh
# Expects models/<geometry>-pkd-q4.tch (the qat end); runs on cards 0-3.
# A checkpoint or log of an earlier fp32 attempt (models/*-q4-fp32*,
# <TFT_LOG_DIR>/*fp32*.jsonl) must be moved aside first: the logger refuses to
# write into an existing jsonl file. At the end the marker
# /root/ARM_<tag>_FERTIG ("finished") is written.
set -u
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin"
cd "${BAUM:?trainer tree, e.g. /root/fx6m_8m}" || exit 2
set -a; . "${ENVD:?env file, e.g. /root/chain/10_fp32.env}"; set +a
unset TFT_WARMSTART TFT_WARMSTART_OPT TFT_SCHED_TOTAL_EPOCHS TFT_SCHED_DONE_EPOCHS
export CUDA_VISIBLE_DEVICES=0,1,2,3; export TFT_GPUS=4
ls -l models/${TFT_GEOMETRY}-pkd-q4.tch || { echo "### qat end missing"; exit 2; }
echo "### fp32 ($TFT_EPOCHS_FP32 epochs)  $(date -u +%T) UTC"
uv run torchrun --nproc_per_node 4 --master_port 29951 -m tft.cooldown_poc || { echo "### FP32 FAILED rc=$?"; exit 4; }
bash run/export_chain.sh || { echo "### EXPORT FAILED rc=$?"; exit 5; }
echo "### FP32-ARM DONE  $(date -u +%F' '%T) UTC"
touch "/root/ARM_${TFT_RUN_TAG}_FERTIG"
