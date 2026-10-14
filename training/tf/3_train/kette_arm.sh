#!/usr/bin/env bash
# One chain arm on the GPU host: bf16, then qat, fp32 and the export.
#
#   BAUM=/root/fx6m_8m ENVD=/root/chain/05_w2048.env KARTEN=0,1,2,3 PORT=29751 \
#     bash /root/kette_arm.sh
#
# BAUM = trainer tree, ENVD = env file of the section, KARTEN = cards
# (CUDA_VISIBLE_DEVICES). At the end the marker /root/ARM_<tag>_FERTIG
# ("finished") is written.
#
# Two arms that run at the same time need two trees: tftcfg.model_path writes
# models/<geometry>-<arm>.tch relative to the working directory, and two arms
# with the same geometry and ZCTX_ARM=pkd overwrite each other in one tree
# (the fp32 phase then loads the qat state of the other arm).
set -u
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin"
: "${BAUM:?}"; : "${ENVD:?}"; : "${KARTEN:?}"; : "${PORT:=29751}"
cd "$BAUM"
set -a; . "$ENVD"; set +a
export CUDA_VISIBLE_DEVICES="$KARTEN"
export TFT_GPUS=$(printf '%s' "$KARTEN" | tr ',' '\n' | grep -c .)
echo "### ARM $TFT_RUN_TAG  tree $BAUM  cards $KARTEN ($TFT_GPUS)  $(date -u +%F' '%T) UTC"
echo "### geometry $TFT_GEOMETRY, package $PK_DUMP_DIR"
echo "### micro-batch per rank $TFT_MICRO_BATCH_TOKENS, optimizer batch $TFT_OPTIMIZER_BATCH_TOKENS => accumulation $(( TFT_OPTIMIZER_BATCH_TOKENS / TFT_MICRO_BATCH_TOKENS / TFT_GPUS ))"
env | grep -E '^(PK_|ZCTX_|TFT_)' | sort
uv run python -m tft.dataset check "$TFT_DATA_DIR" 208 | tail -3
uv run torchrun --nproc_per_node "$TFT_GPUS" --master_port "$PORT" -m tft.train_poc \
  || { echo "### BF16 FAILED rc=$?"; exit 2; }
# These variables belong to the bf16 phase. Left set, the continued schedule
# would also bend the qat and fp32 schedules, and with KINJ_OPTSTATE=1 the
# optimizer state of the warm start would be loaded into qat.
unset TFT_SCHED_TOTAL_EPOCHS TFT_SCHED_DONE_EPOCHS TFT_WARMSTART TFT_WARMSTART_OPT
echo "### qat ($TFT_EPOCHS_QAT epochs)  $(date -u +%T) UTC"
uv run torchrun --nproc_per_node "$TFT_GPUS" --master_port "$((PORT+1))" -m tft.quantize_poc \
  || { echo "### QAT FAILED rc=$?"; exit 3; }
echo "### fp32 ($TFT_EPOCHS_FP32 epochs)  $(date -u +%T) UTC"
uv run torchrun --nproc_per_node "$TFT_GPUS" --master_port "$((PORT+2))" -m tft.cooldown_poc \
  || { echo "### FP32 FAILED rc=$?"; exit 4; }
echo "### export  $(date -u +%T) UTC"
bash run/export_chain.sh || { echo "### EXPORT FAILED rc=$?"; exit 5; }
echo "### ARM $TFT_RUN_TAG DONE  $(date -u +%F' '%T) UTC"
touch "/root/ARM_${TFT_RUN_TAG}_FERTIG"
