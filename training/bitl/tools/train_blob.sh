#!/bin/bash
# Trains the BITLSTM weights on the first <bits> model bits of a
# trace: <passes> passes in stream order, the learning rate falling along a
# cosine from lr0 to lr1 (x 1e-6), a fixed seed for the initial draw.  Each
# pass after the first starts from the weights of the previous one (only the
# weights carry over, the optimiser moments start at zero).  The weights of
# the last pass are rounded to bf16 (nblb.py v2).
#
#   TRACE=<trace> CHANNEL=<channel> OUT=<dir> \
#     bash train_blob.sh <name> <bits> [passes] [lr0_1e6] [lr1_1e6] [seed]
#
# Defaults are the recipe of the shipped weights: 32 passes, 3000 -> 30,
# seed 1112363057, bits 418837176 (../record/).  The trace must hold
# MARGIN bits more than <bits> (default 100000, for a trace that is still
# being written; 0 for a finished prefix).  TRAINER defaults to
# ../trainer/replay_trainer (build it with ../trainer/build.sh).
# Writes $OUT/<name>/e<pass>.nblb (passes 1 and the last are kept),
# $OUT/<name>_fp32.nblb, $OUT/<name>.nblb and the log $OUT/train_<name>.log.
set -uo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
TRACE=${TRACE:?path of the trace file}
CHANNEL=${CHANNEL:?path of the per-byte channel file}
OUT=${OUT:?output directory}
MARGIN=${MARGIN:-100000}
N="${1:?name}"; B="${2:?bits}"; EP="${3:-32}"; LR0="${4:-3000}"; LR1="${5:-30}"; SEED="${6:-1112363057}"
mkdir -p "$OUT"
R=$OUT/train_$N.log
say(){ echo "$(date "+%F %T") $*" | tee -a "$R"; }
TR=${TRAINER:-$HERE/../trainer/replay_trainer}
[ -x "$TR" ] || { say "STOP: $TR missing"; exit 2; }
read REC HEAD NCOL N1 < <(python3 "$HERE/trace_geom.py" "$TRACE")
# Column mask check: n1 alone does not identify the recorded columns.
WANT_COLS=132120703; WANT_N1=28; WANT_NCOL=13
TCOLS=$(python3 -c "import struct,sys
f=open(sys.argv[1],'rb')
h=struct.unpack('<5I',f.read(20))
f.read(16*h[4])
print(struct.unpack('<Q',f.read(8))[0] if h[1]>=3 else 0)" "$TRACE")
[ "$TCOLS" = "$WANT_COLS" ] || { echo "STOP: column mask $TCOLS, expected $WANT_COLS"; exit 11; }
[ "$N1" = "$WANT_N1" ] || { echo "STOP: n1 $N1, expected $WANT_N1"; exit 11; }
[ "$NCOL" = "$WANT_NCOL" ] || { echo "STOP: $NCOL columns, expected $WANT_NCOL"; exit 11; }
echo "column mask ok: tcols $TCOLS, n1 $N1, $NCOL columns"
SR=$(( ($(stat -c %s "$TRACE")-HEAD)/REC )); CR=$(( ($(stat -c %s "$CHANNEL")-28)/276 ))
MIN=$SR; [ $((CR*8)) -lt $MIN ] && MIN=$((CR*8))
[ "$MIN" -ge $((B+MARGIN)) ] || { say "STOP: trace holds $MIN bits, needed $((B+MARGIN))"; exit 3; }
say "=== weights $N: $B bits, $EP passes, cosine $LR0 -> $LR1 /1e6, seed $SEED"
say "    trace record $REC B, header $HEAD B, $NCOL of $N1 columns; present $MIN bits (margin $((MIN-B)))"
say "    trainer $(stat -c %s "$TR") B sha $(sha256sum < "$TR" | cut -c1-16)"
W=$OUT/$N; mkdir -p "$W"
PREV=""
for e in $(seq 1 "$EP"); do
  LR=$(python3 -c "import math;e=$e;E=$EP;a=$LR0;b=$LR1;print(int(round(b+(a-b)*0.5*(1+math.cos(math.pi*(e-1)/max(E-1,1))))))")
  CUR="$W/e$e.nblb"
  ENVS=(REPLAY_CHANNEL="$CHANNEL" REPLAY_SEED="$SEED" REPLAY_LR1E6="$LR" REPLAY_DUMP="$CUR")
  [ -n "$PREV" ] && ENVS+=(REPLAY_LOAD="$PREV")
  T0=$(date +%s)
  LINE=$(env "${ENVS[@]}" nice -n 15 "$TR" "$TRACE" "$B" 0 2>&1 | grep -aE "^(bits=|trace:|channel:)" | tr "\n" " ")
  T1=$(date +%s)
  say "pass $e lr=$LR/1e6 $((T1-T0))s $LINE"
  [ -s "$CUR" ] || { say "STOP: pass $e wrote no weights"; exit 4; }
  [ -n "$PREV" ] && [ "$e" -gt 2 ] && rm -f "$PREV"
  PREV="$CUR"
done
cp "$PREV" "$OUT/${N}_fp32.nblb"
python3 "$HERE/nblb.py" v2 "$PREV" "$OUT/${N}.nblb" 2>&1 | tee -a "$R"
BL=$OUT/$N.nblb
say "weights $N: $(stat -c %s "$BL") B sha $(sha256sum < "$BL" | cut -c1-32)"
python3 "$HERE/nblb.py" info "$BL" 2>&1 | tee -a "$R"
G=$(python3 -c "import struct,sys;print('%d/%d/%d/%d'%struct.unpack('<4i',open(sys.argv[1],'rb').read(24)[8:24]))" "$BL")
[ "$G" = "16/89/96/28" ] || { say "STOP: geometry $G, expected 16/89/96/28"; exit 5; }
python3 "$HERE/blobstat.py" "$BL" 2>&1 | tee -a "$R"
say "done: $BL bits=$B passes=$EP sha=$(sha256sum < "$BL" | cut -d' ' -f1) geom=$G size=$(stat -c %s "$BL")"
