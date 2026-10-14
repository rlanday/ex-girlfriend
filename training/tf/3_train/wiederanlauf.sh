#!/usr/bin/env bash
# Resume ("Wiederanlauf") one phase of a chain arm from a saved checkpoint,
# for instance after the GPU host was lost. Runs on the GPU host.
#
#   BAUM=/root/fx6m_8m ENVD=/root/chain/01_anchor_w2048.env KARTEN=0,1 \
#     PORT=29851 bash /root/wiederanlauf.sh <ckpt> <phase> <done_epochs>
#     <ckpt>         the saved state, e.g. /root/ckpt/a8m_2048-ep13.tch;
#                    <ckpt>.opt next to it is used when present
#     <phase>        bf16 | qat | fp32
#     <done_epochs>  epochs of this phase already done (read them from the
#                    .opt with epoche_opt.sh, do not estimate)
#
# What differs from a fresh start:
#   * it starts from the weights of the checkpoint;
#   * it carries the Adam/Muon moments (TFT_WARMSTART_OPT, needs
#     KINJ_OPTSTATE=1);
#   * it continues the cosine schedule of the whole block
#     (TFT_SCHED_TOTAL_EPOCHS / _DONE_EPOCHS, _plan_schedule in tft/runner.py)
#     instead of restarting at the warmup multiplier 8; a resume at epoch 37
#     of a 120-epoch block would otherwise run at 8.00 instead of 4.91;
#   * it runs only the remaining epochs of the phase.
# It runs this one phase only and writes no marker; after bf16 the following
# phases run with qat_arm.sh. In this chain it was used for bf16 only:
# tft/runner.py applies TFT_WARMSTART to bf16 alone (qat and fp32 load
# models/<geometry>-pkd.tch and -pkd-q4.tch).
# GESAMT = total epochs of the phase, REST = epochs left.
set -eu
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin"
CKPT="${1:?ckpt}"; PHASE="${2:?phase: bf16|qat|fp32}"; DONE="${3:?done_epochs}"
BAUM="${BAUM:?trainer tree, e.g. /root/fx6m_8m}"
ENVD="${ENVD:?env file of the section}"
KARTEN="${KARTEN:-0,1,2,3}"
PORT="${PORT:-29851}"
[ -s "$CKPT" ] || { echo "checkpoint $CKPT missing"; exit 2; }

set -a; . "$ENVD"; set +a
case "$PHASE" in
  bf16) GESAMT=$TFT_EPOCHS_BF16;;
  qat)  GESAMT=$TFT_EPOCHS_QAT;;
  fp32) GESAMT=$TFT_EPOCHS_FP32;;
  *) echo "phase must be bf16|qat|fp32"; exit 2;;
esac
REST=$(( GESAMT - DONE ))
[ "$REST" -ge 1 ] || { echo "nothing left ($DONE of $GESAMT)"; exit 3; }

export TFT_WARMSTART="$CKPT"
[ -s "$CKPT.opt" ] && export TFT_WARMSTART_OPT="$CKPT.opt"
export KINJ_OPTSTATE=1
export TFT_SCHED_TOTAL_EPOCHS="$GESAMT"
export TFT_SCHED_DONE_EPOCHS="$DONE"
case "$PHASE" in
  bf16) export TFT_EPOCHS_BF16="$REST";;
  qat)  export TFT_EPOCHS_QAT="$REST";;
  fp32) export TFT_EPOCHS_FP32="$REST";;
esac
export CUDA_VISIBLE_DEVICES="$KARTEN"
export TFT_GPUS=$(printf '%s' "$KARTEN" | tr ',' '\n' | grep -c .)

echo "### RESUME $TFT_RUN_TAG phase $PHASE: $DONE of $GESAMT done, $REST left"
echo "### warm start $TFT_WARMSTART${TFT_WARMSTART_OPT:+ (+ optimizer state)}"
echo "### continued schedule: TOTAL=$TFT_SCHED_TOTAL_EPOCHS DONE=$TFT_SCHED_DONE_EPOCHS"
cd "$BAUM"
env | grep -E '^(PK_|ZCTX_|TFT_|KINJ_)' | sort
MOD=train_poc; [ "$PHASE" = qat ] && MOD=quantize_poc; [ "$PHASE" = fp32 ] && MOD=cooldown_poc
uv run torchrun --nproc_per_node "$TFT_GPUS" --master_port "$PORT" -m "tft.$MOD"
echo "### phase $PHASE done after the resume; the following phases"
echo "### run with qat_arm.sh (qat, fp32, export)."
