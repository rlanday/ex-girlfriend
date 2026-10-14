#!/bin/bash
# Records the trace of the BITLSTM's inputs: one compression run
# ("-e") of enwik9 with a measurement build of the development tree, which is
# not part of this package.  The build line is FEATURES_trace.txt.  The build
# options and the CMIX_BITL_TRACE* environment variables below exist only in
# that tree (../README.md, section "Trace run").
#
#   TREE=<development tree> CONTAINER=<weights .c3> ENWIK9=<enwik9> \
#   DICT=<.dict> DICT_COMP=<.dict.comp> ORDER=<.new_article_order> \
#   RUN=<run directory> bash run_trace.sh
#
# The run writes $RUN/trace.bin (60 B per model bit) and $RUN/channel.bin
# (276 B per eight model bits); they grow by about 1.3 GB and 0.7 GB per
# percent of the coder stream.  The shipped weights use the prefix up to 20 %
# of the stream, 25.1 GB and 14.4 GB (../record/mark.txt); the run can be
# stopped once ../tools/cut_trace.sh has written its record.
set -u
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=${TREE:?development tree at the state of the trace}
C=${CONTAINER:?transformer weight container (.c3)}
ENWIK9=${ENWIK9:?path of enwik9}
DICT=${DICT:?.dict of a compression run of this configuration}
DICT_COMP=${DICT_COMP:?.dict.comp of the same run}
ORDER=${ORDER:?.new_article_order of the same run}
RUN=${RUN:?run directory}
mkdir -p "$RUN"
LOG=$RUN/start.log
say(){ echo "$(date +%FT%T) $*" | tee -a "$LOG"; }
die(){ say "STOP: $*"; exit 1; }

# --- 1. stamp the weight container into the tree ------------------------------
python3 "$TREE/tools/tf1_weights_stamp.py" "$C" "$(basename "$C")" > "$TREE/src/tf1/tf1_weights_set.h.new" 2> "$RUN/stamp.log" || { cat "$RUN/stamp.log"; die "stamping failed"; }
mv "$TREE/src/tf1/tf1_weights_set.h.new" "$TREE/src/tf1/tf1_weights_set.h"
grep -q "NEIF_TF1_WEIGHTS_STAMPED 1" "$TREE/src/tf1/tf1_weights_set.h" || die "no STAMPED 1"
grep -q "$(sha256sum < "$C" | cut -d" " -f1)" "$TREE/src/tf1/tf1_weights_set.h" || die "stamped sha256 is not the container's"
say "container stamped ($(sha256sum < "$C" | cut -c1-16))"

# --- 2. build -------------------------------------------------------------------
FEATURES=$(grep -v '^#' "$HERE/FEATURES_trace.txt" | tr '\n' ' ')
( cd "$TREE" && make -s clean >/dev/null 2>&1; make cmix FEATURES="$FEATURES" -j8 ) > "$RUN/build.log" 2>&1 || { grep -aE "error|#error" "$RUN/build.log" | head -10; die "build failed"; }
cp "$TREE/cmix" "$RUN/cmix"
say "build: $(stat -c %s "$RUN/cmix") B sha $(sha256sum < "$RUN/cmix" | cut -c1-16)"
# The four options of the final configuration must be in the -D set.
for s in IND23_WEG=1 LAK_SHARED_MAP=500000 SSE_S7_HALB=1 FXSTAT_HALB=1; do
  grep -aq -- "-DNEIF_$s " "$RUN/build.log" || grep -aq -- "-DNEIF_$s$" "$RUN/build.log" || die "-DNEIF_$s not in the build log"
done
strings "$RUN/cmix" | grep -q "$(basename "$C")" || die "container stamp not in the binary"

# --- 3. inputs and start ----------------------------------------------------------
# REUSE_EXTRACTED in the build line skips the derivation of the dictionary and
# of the article order (and the child process that compresses the dictionary,
# which would otherwise inherit the trace variables): both files must be in
# the run directory.
cd "$RUN" || die "no run directory"
cp "$DICT" .dict; cp "$DICT_COMP" .dict.comp; cp "$ORDER" .new_article_order; cp "$C" .
{ [ -e trace.bin ] || [ -e channel.bin ]; } && die "trace files exist already"
export CMIX_BITL_TRACE=$RUN/trace.bin CMIX_BITL_TRACE_AUX=$RUN/channel.bin
export CMIX_BITL_TRACE_AUX_PROJ=32 CMIX_BITL_TRACE_AUX_LPROJ=32 CMIX_BITL_TRACE_COLS=132120703
{ echo "=== trace run start $(date +%FT%T) ==="
  echo "binary $(sha256sum cmix | cut -d" " -f1)"; echo "container $(sha256sum "$(basename "$C")" | cut -d" " -f1)"
  echo "FEATURES $FEATURES"; env | grep -E "^CMIX_BITL" | sort; } > compress.log
nice -n 8 ./cmix -e "$ENWIK9" out >> compress.log 2>&1 &
P=$!
say "PID $P"

# --- 4. probes ----------------------------------------------------------------------
ok=0
for t in $(seq 1 150); do
  sleep 10
  kill -0 $P 2>/dev/null || { say "STOP: process gone after ${t}0 s"; tail -10 compress.log; exit 6; }
  grep -aq "is not readable" compress.log && { say "STOP: container not readable"; kill -TERM $P; exit 7; }
  grep -aq "NEIF-TF1: opt (int8) engine" compress.log && { ok=1; say "transformer engine running"; break; }
done
[ $ok -eq 1 ] || { say "STOP: no engine line"; kill -TERM $P; exit 8; }
# The coder stream of the recorded run started with sha256 8e0b264a777cb56a6be48c58...
if [ -f out.cmix.temp ]; then
  say "coder stream sha256 $(sha256sum < out.cmix.temp | cut -c1-24) (recorded run: 8e0b264a777cb56a6be48c58)"
fi
for t in $(seq 1 720); do
  sleep 10
  kill -0 $P 2>/dev/null || { say "STOP: process gone before the first trace record"; exit 6; }
  [ "$(stat -c %s trace.bin 2>/dev/null || echo 0)" -gt 28 ] && { say "first trace record at progress $(tail -1 progress.log)"; break; }
done
sleep 180
say "trace form (record header columns n1): $(python3 "$HERE/../tools/trace_geom.py" trace.bin)  (expected: 60 28 13 28)"
wait $P; say "=== trace run ended rc=$?"
