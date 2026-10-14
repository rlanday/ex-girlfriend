#!/bin/bash
# Fixes the training range of the BITLSTM on a trace that is still
# being written: waits until the compression run has passed TARGET percent of
# the coder stream, takes the number of model bits B the trace and the channel
# both hold, keeps a margin of 200,000 bits to the growing end, rounds down to
# a multiple of 8, and records B with the size and sha256 of both prefixes.  No
# copy is made; the record identifies the prefix.
#
#   TRACE=<trace> CHANNEL=<channel> PROGRESS=<progress.log> OUT=<dir> \
#     bash cut_trace.sh <name>
#
# Writes $OUT/<name>_BITS and $OUT/mark.txt.  For the shipped weights:
# TARGET 20.00, reached at 20.01 %, B = 418,837,176 (../record/).
set -u
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
TRACE=${TRACE:?path of the trace file}
CHANNEL=${CHANNEL:?path of the per-byte channel file}
PROGRESS=${PROGRESS:?progress.log of the compression run that writes the trace}
OUT=${OUT:?output directory}
TARGET=${TARGET:-20.00}
N=${1:?name of the weight set}
mkdir -p "$OUT"
LOG=$OUT/cut_$N.log
say(){ echo "$(date +%FT%T) $*" | tee -a "$LOG"; }
die(){ say "STOP: $*"; exit 1; }

# --- 1. wait for the share of the stream --------------------------------------
p=$(tail -1 "$PROGRESS" 2>/dev/null | awk '{print $1}')
say "=== $N: target $TARGET %, trace now at ${p:-?} %"
for t in $(seq 1 2880); do
  p=$(tail -1 "$PROGRESS" 2>/dev/null | awk '{print $1}')
  if [ -n "$p" ] && python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>=float(sys.argv[2]) else 1)" "$p" "$TARGET"; then
    say "target reached at $p %"; break; fi
  [ $((t % 15)) -eq 0 ] && say "waiting, trace at ${p:-?} %"
  sleep 60
done
p=$(tail -1 "$PROGRESS" | awk '{print $1}')
python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>=float(sys.argv[2]) else 1)" "$p" "$TARGET" || die "time limit, trace at $p %"

# --- 2. form of the trace -------------------------------------------------------
read REC HEAD NCOL N1 < <(python3 "$HERE/trace_geom.py" "$TRACE") || die "trace header"
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
say "trace form: record $REC B, header $HEAD B, $NCOL of $N1 columns, mask $TCOLS"

# --- 3. the cut -----------------------------------------------------------------
CREC=276; CHEAD=28
SS=$(stat -c %s "$TRACE"); CS=$(stat -c %s "$CHANNEL")
SR=$(( (SS-HEAD)/REC )); CR=$(( (CS-CHEAD)/CREC ))
MIN=$SR; [ $((CR*8)) -lt $MIN ] && MIN=$((CR*8))
B=$(( (MIN - 200000) / 8 * 8 ))
[ "$B" -gt 1000000 ] || die "too little trace: $MIN bits"
[ "$B" -le $((MIN-100000)) ] || { say "STOP: $B bits asked, trace and channel hold $MIN"; exit 3; }
echo "$B" > "$OUT/${N}_BITS"
say "trace holds $MIN bits; cut at B=$B bits (in ${N}_BITS)"

# --- 4. the record ----------------------------------------------------------------
SP=$((HEAD + REC*B)); CP=$((CHEAD + CREC*(B/8)))
{ echo "name=$N"; echo "date=$(date "+%F %T %Z")"; echo "bits=$B"; echo "channel_records=$((B/8))";
  echo "trace_record_B=$REC"; echo "trace_header_B=$HEAD"; echo "columns=$NCOL"; echo "n1=$N1";
  echo "trace_prefix_B=$SP"; echo "channel_prefix_B=$CP";
  echo "trace_file_B=$SS"; echo "channel_file_B=$CS";
  echo "trace_records_total=$SR"; echo "channel_records_total=$CR"; echo "min_bits=$MIN";
  echo "margin_bits=$((MIN-B))"; } > "$OUT/mark.txt"
say "sha256 of the prefixes (trace $SP B, channel $CP B) ..."
SSHA=$(nice -n 19 ionice -c3 head -c $SP "$TRACE" | sha256sum | cut -d" " -f1)
CSHA=$(nice -n 19 ionice -c3 head -c $CP "$CHANNEL" | sha256sum | cut -d" " -f1)
{ echo "trace_prefix_sha256=$SSHA"; echo "channel_prefix_sha256=$CSHA"; } >> "$OUT/mark.txt"
cat "$OUT/mark.txt" >> "$LOG"
say "mark $N written: trace_prefix_sha256=$SSHA"
