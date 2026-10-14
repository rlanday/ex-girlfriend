#!/usr/bin/env bash
# Set up a GPU host for one section of the chain: python environment, window
# geometries, warm-start check, training package and priors, each checked by
# content, then the loader check.  Runs on the GPU host (we ran it under
# nohup); each step leaves a marker file, so a second run skips what is done.
# Log: /root/boot_v7.log.  Marker of success: /root/BOOT_V7_OK.
#
# Expected in place before the start (the GPU host layout of our runs):
#   /root/fx6m_8m/          trainer tree (tree_upstream + tree_overlay)
#   /root/geo_zusatz.py     window geometry helper
#   /root/kette_arm.sh      (mode acht starts it)
#   /root/.modus_v7         mode, one word (see below)
#   /root/.u_v7             4 lines: package URL, priors URL, -, -
#                           ("-" = not yet known; then the files below)
#   /root/.u_pkg            package URL (tar of the training package)
#   /root/.u_b25            URL of ppmd-bytes.uint8
#   /root/.u_p25            URL of the priors (ppmd-probs.float16)
#   /root/.u_p25_sha        sha256 of the first MiB of the priors
#   /root/.u_p25_vollsha    sha256 of the whole priors file
#   /root/.anker_pfad       mode zweig: path of the warm-start checkpoint
#   /root/.anker_sha16      mode zweig: its sha256, first 16 hex digits
#   /root/.prio_stroeme     optional: number of parallel download streams
# URLs and tokens live only in these files, never in this script.
#
# Modes: zweig ("branch") = warm start, checked by sha, no training start;
#        acht ("eight") = cold 8M anchor, starts section 1 itself on cards 0,1
#        with ANCHOR_ENV as env file;
#        pilot / kette ("chain") = earlier 6M runs, not used in this chain.
#
# German names: sag = log line, hole = fetch, Paket = package,
# Prioren = priors, Stroeme = streams, Stuecke = pieces, Anker = warm-start
# checkpoint, Baeume = trees, Modus = mode, Marke = tag, Ziel = target,
# Soll/Ist = expected/actual, Voll = whole file, GETEILT = files shared by all
# arms, FEHLER = errors, schlecht = bad. "25" in .u_p25, U_P25 and PRIO_25_OK is the tag of the main
# priors file (named after PPMd order 25 of the first package; the tag stayed
# for the later packages).
set -u
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin"
L=/root/boot_v7.log
sag () { echo "$(date -u +%T) $*" | tee -a $L; }
STROEME=${PRIO_STROEME:-64}
[ -s /root/.prio_stroeme ] && STROEME=$(cat /root/.prio_stroeme)

hole_parallel () {   # $1 URL  $2 target file  $3 expected bytes (0 = from HEAD)
  U=$1; ZIEL=$2; LEN=${3:-0}
  if [ "$LEN" = 0 ]; then
    LEN=$(curl -sSI --max-time 60 "$U" | tr -d "\r" \
          | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  fi
  [ -n "$LEN" ] && [ "$LEN" -gt 0 ] || { sag "no length for $ZIEL"; return 1; }
  # Do NOT use `N` here: the caller keeps n_tokens in it.
  HP_N=$STROEME; MB=$((1024*1024))
  truncate -s "$LEN" "$ZIEL"
  BL=$((LEN/MB)); REST=$((LEN%MB)); JE=$((BL/HP_N)); T0=$(date +%s)
  sag "fetching $(basename "$ZIEL") ($((LEN/1000000)) MB) with $HP_N streams"
  for i in $(seq 0 $((HP_N-1))); do
    SK=$((i*JE)); CO=$JE; [ $i -eq $((HP_N-1)) ] && CO=$((BL-SK))
    A=$((SK*MB)); B=$((A+CO*MB-1))
    ( curl -sS --fail --retry 5 --retry-delay 5 -H "Range: bytes=$A-$B" "$U" \
      | dd of="$ZIEL" bs=$MB seek=$SK conv=notrunc status=none ) &
  done
  [ $REST -gt 0 ] && ( A=$((BL*MB)); curl -sS --fail --retry 5 -H "Range: bytes=$A-$((LEN-1))" "$U" \
      | dd of="$ZIEL" bs=1 seek=$A conv=notrunc status=none ) &
  wait
  T1=$(date +%s); D=$((T1-T0)); [ $D -lt 1 ] && D=1
  sag "$(basename "$ZIEL"): $((LEN/D/1000000)) MB/s in $D s"
  [ "$(stat -c%s "$ZIEL")" = "$LEN" ] || { sag "WRONG LENGTH of $ZIEL"; return 1; }
  return 0
}
V=208
MODUS=$(cat /root/.modus_v7 2>/dev/null || echo pilot)
U_PKG=$(sed -n 1p /root/.u_v7)
U_P25=$(sed -n 2p /root/.u_v7)
U_P20=$(sed -n 3p /root/.u_v7)
U_B20=$(sed -n 4p /root/.u_v7)
case "$MODUS" in
  pilot) BAEUME="/root/fx6m_p25 /root/fx6m_p20"; PAKETE="/root/pkg_v7 /root/pkg_v7o20";;
  kette) BAEUME="/root/fx6m_k";                  PAKETE="/root/pkg_v7";;
  acht)  BAEUME="/root/fx6m_8m";                 PAKETE="/root/pkg_v7";;
  zweig) BAEUME="/root/fx6m_8m";                 PAKETE="/root/pkg_v7";;
  *) echo "MODUS must be pilot, kette, acht or zweig"; exit 2;;
esac

sag "### SETUP ($MODUS) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv | tee -a $L
df -h / | tail -1 | tee -a $L

# 1 python environment per tree ---------------------------------------------
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$PATH:/root/.local/bin; }
for D in $BAEUME; do
  [ -d "$D/.venv" ] && { sag "venv $D present"; continue; }
  cd "$D"
  for i in 1 2 3; do uv sync >>$L 2>&1 && break; sag "  uv sync $D attempt $i"; sleep 15; done
  test -d "$D/.venv" || { sag "VENV MISSING in $D"; exit 3; }
  sag "venv $D built"
done

# 2 window geometries in every tree -----------------------------------------
N_BAEUME=0
for D in $BAEUME; do
  N_BAEUME=$((N_BAEUME+1))
  cd "$D"
  if [ "$MODUS" = acht ] || [ "$MODUS" = zweig ]; then
    for G in poc8m_w1024 poc8m_w1536 poc8m_w2048; do
      python3 /root/geo_zusatz.py $G tft/param_count.py >>$L 2>&1 || sag "geometry helper $D $G reports an error"
    done
  else
  python3 /root/geo_zusatz.py 6m208_w1664 tft/param_count.py >>$L 2>&1 || sag "geometry helper $D reports an error"
  fi
  if [ "$MODUS" = acht ] || [ "$MODUS" = zweig ]; then
    .venv/bin/python -c "
from tft.param_count import GEOMETRIES as G
for n in ('poc8m_w1024','poc8m_w1536','poc8m_w2048'):
    g=G[n]; print('GEO8 $D %s: window=%d d_model=%d layers=%d' % (n,g.window_size,g.d_model,g.n_layers))" 2>&1 | tee -a $L
  else
  .venv/bin/python -c "
from tft.param_count import GEOMETRIES as G
g=G['6m208_w1664']; print('GEO $D 6m208_w1664: window=%d d_model=%d layers=%d' % (g.window_size,g.d_model,g.n_layers))" 2>&1 | tee -a $L
  fi
done
if [ "$MODUS" = acht ] || [ "$MODUS" = zweig ]; then
  for W in 1024 1536 2048; do grep -q "GEO8 .* window=$W" $L || { sag "GEOMETRY poc8m_w$W MISSING"; exit 5; }; done
  grep -q "GEO8 .* layers=16" $L || { sag "poc8m does not have 16 layers"; exit 5; }
else
[ "$(grep -c 'GEO .* window=1664' $L)" -ge "$N_BAEUME" ] || { sag "GEOMETRY 1664 MISSING in a tree"; exit 5; }
fi

# 3 check the warm start ----------------------------------------------------
if [ "$MODUS" = zweig ]; then
  # path and expected sha256 prefix come from /root/.anker_pfad and /root/.anker_sha16
  AP=$(head -1 /root/.anker_pfad 2>/dev/null); AS=$(head -1 /root/.anker_sha16 2>/dev/null | tr -cd 0-9a-f)
  [ -n "$AP" ] && [ ${#AS} = 16 ] || { sag "WARM START: /root/.anker_pfad or /root/.anker_sha16 missing"; exit 10; }
  SHA=$(sha256sum "$AP" 2>/dev/null | cut -c1-16)
  [ "$SHA" = "$AS" ] || { sag "WARM START SHA WRONG: $SHA instead of $AS ($AP)"; exit 10; }
  sag "warm start $AP present, sha $SHA"
elif [ "$MODUS" = acht ]; then
  sag "COLD START (8M): no warm-start check"
else
SHA=$(sha256sum /root/ckpt/6m208_w1536-400-pkd.tch | cut -c1-16)
[ "$SHA" = "71fe98fe8f2efa30" ] || { sag "WARM START SHA WRONG: $SHA instead of 71fe98fe8f2efa30"; exit 10; }
sag "warm start bf16-400 present, sha $SHA"
fi

# 4 training package (without the priors) -----------------------------------
mkdir -p $PAKETE
if [ ! -f /root/PKG_OK ]; then
  # The package URL may come later (/root/.u_pkg), so that a host set up early
  # waits for a late package instead of failing.
  while [ -z "$U_PKG" ] || [ "$U_PKG" = "-" ]; do
    if [ -s /root/.u_pkg ]; then U_PKG=$(cat /root/.u_pkg); break; fi
    sag "waiting for the package URL (/root/.u_pkg)"
    sleep 120
  done
  for i in 1 2 3; do
    hole_parallel "$U_PKG" /root/pkg.tar 0 && break
    sag "  package attempt $i"; sleep 15
  done
  [ -s /root/pkg.tar ] || { sag "PACKAGE NOT FETCHED"; exit 6; }
  tar xf /root/pkg.tar -C /root/pkg_v7
  rm -f /root/pkg.tar
  touch /root/PKG_OK
fi
GETEILT="zeilen.dump doc-rows.int64 doc-last-ziel.int32 article-boundaries.int32 \
title-lengths.int32 dump-meta.json z3-prior0.float16 vocab-bitmap.u8 zonenmaske.u8"
for f in $GETEILT; do
  [ -f /root/pkg_v7/$f ] || { sag "PACKAGE CHECK FAILED: $f missing"; exit 11; }
  sag "package check: $f $(stat -c%s /root/pkg_v7/$f) B"
done
N=$(python3 -c "import json;print(json.load(open('/root/pkg_v7/dump-meta.json'))['n_tokens'])")
sag "n_tokens $N"
# ppmd-bytes.uint8 is not in the package tar (the package is built before the
# priors); its URL comes like the priors URL: /root/.u_b25.
if [ "$(stat -c%s /root/pkg_v7/ppmd-bytes.uint8 2>/dev/null || echo 0)" != "$N" ]; then
  U_B25=""
  while [ -z "$U_B25" ]; do
    if [ -s /root/.u_b25 ]; then U_B25=$(cat /root/.u_b25); break; fi
    sag "waiting for the URL of ppmd-bytes (/root/.u_b25)"
    sleep 120
  done
  hole_parallel "$U_B25" /root/pkg_v7/ppmd-bytes.uint8 "$N" \
    || { sag "ppmd-bytes NOT FETCHED"; exit 12; }
fi
[ "$(stat -c%s /root/pkg_v7/ppmd-bytes.uint8)" = "$N" ] || { sag "ppmd-bytes is $(stat -c%s /root/pkg_v7/ppmd-bytes.uint8) B, expected $N"; exit 12; }

# 5 priors (always first) ---------------------------------------------------
hole_prioren () {   # $1 target path  $2 URL  $3 tag
  Z=$1; U=$2; MK=$3
  SOLLSHA=$(head -1 /root/.u_p${MK}_sha 2>/dev/null | tr -cd '0-9a-f')
  [ ${#SOLLSHA} = 64 ] || { sag "IDENTITY: /root/.u_p${MK}_sha missing or not a sha256; abort (no unchecked priors)"; exit 9; }
  istsha(){ head -c 1048576 "$Z" 2>/dev/null | sha256sum | cut -d' ' -f1; }
  if [ -f "/root/PRIO_${MK}_OK" ]; then
    if [ "$(istsha)" = "$SOLLSHA" ]; then sag "priors $MK present, identity of the first MiB confirmed (${SOLLSHA:0:16})"; return 0; fi
    sag "priors $MK present but the WRONG table (first MiB $(istsha | cut -c1-16) != expected ${SOLLSHA:0:16}); fetching again"
    rm -f "/root/PRIO_${MK}_OK" "$Z"
  fi
  # A `curl --retry | dd seek` pipe can write a range a second time behind the
  # bytes already written after a dropped connection (a hole or shift in the
  # file).  Hence: 256-MiB pieces, each into its own file, size checked, only
  # then copied into place; up to 8 attempts per piece; then sha256 over the
  # whole file.
  LEN=$(( N * V * 2 )); ST=$STROEME; CH=$((256*1024*1024))
  truncate -s $LEN "$Z"; mkdir -p /root/.stk; rm -f /root/.stk/*
  NCH=$(( (LEN + CH - 1) / CH )); T0=$(date +%s)
  sag "fetching priors $MK ($((LEN/1000000000)) GB, $NCH pieces of 256 MiB, $ST in parallel, size check per piece)"
  export Z U LEN CH
  seq 0 $((NCH-1)) | xargs -P $ST -I{} bash -c '
    k={}; A=$((k*CH)); B=$((A+CH-1)); [ $B -ge $LEN ] && B=$((LEN-1)); S=$((B-A+1)); T=/root/.stk/$k
    for v in 1 2 3 4 5 6 7 8; do
      rm -f $T; curl -sS --fail --max-time 900 -H "Range: bytes=$A-$B" "$U" -o $T 2>/dev/null
      [ "$(stat -c%s $T 2>/dev/null)" = "$S" ] && dd if=$T of="$Z" bs=1M seek=$((A/1048576)) conv=notrunc status=none && { rm -f $T; exit 0; }
      sleep 5
    done
    echo "PIECE $k WRONG AFTER 8 ATTEMPTS" >> /root/.stk/FEHLER; exit 1'
  [ -s /root/.stk/FEHLER ] && { sag "PRIORS $MK: $(wc -l < /root/.stk/FEHLER) pieces could not be fetched"; exit 8; }
  T1=$(date +%s); DT=$((T1-T0)); [ $DT -lt 1 ] && DT=1
  sag "priors $MK: $((LEN/DT/1000000)) MB/s in $DT s"
  [ $DT -lt 60 ] && { sag "ABORT $MK: impossibly fast, nothing arrived"; exit 7; }
  [ "$(stat -c%s "$Z")" = "$LEN" ] || { sag "LENGTH $MK WRONG"; exit 8; }
  VOLLSOLL=$(head -1 /root/.u_p${MK}_vollsha 2>/dev/null | tr -cd '0-9a-f')
  [ ${#VOLLSOLL} = 64 ] || { sag "IDENTITY: /root/.u_p${MK}_vollsha missing; abort"; exit 9; }
  sag "priors $MK: sha256 of the whole file ..."
  VOLLIST=$(sha256sum "$Z" | cut -d' ' -f1)
  [ "$VOLLIST" = "$VOLLSOLL" ] || { sag "PRIORS $MK BROKEN: sha256 ${VOLLIST:0:16} != expected ${VOLLSOLL:0:16}"; exit 9; }
  sag "priors $MK: sha256 of the whole file confirmed (${VOLLSOLL:0:16})"
  python3 - "$Z" "$LEN" <<'PY' | tee -a $L
import struct, sys, random
p, LEN = sys.argv[1], int(sys.argv[2]); V = 208; ROW = V*2; n = LEN//ROW
random.seed(11); schlecht = 0
with open(p, "rb") as f:
    for _ in range(64):
        i = random.randrange(n); f.seek(i*ROW)
        s = sum(struct.unpack("<%de" % V, f.read(ROW)))
        if not (0.99 < s < 1.01):
            schlecht += 1; print("  row %d -> %.4f" % (i, s))
print("row sums: %d of 64 bad" % schlecht)
sys.exit(1 if schlecht else 0)
PY
  [ ${PIPESTATUS[0]:-0} -eq 0 ] || { sag "PRIORS $MK BROKEN"; exit 9; }
  [ "$(istsha)" = "$SOLLSHA" ] || { sag "PRIORS $MK: first MiB $(istsha | cut -c1-16) != expected ${SOLLSHA:0:16}; WRONG TABLE"; exit 9; }
  sag "priors $MK: identity of the first MiB confirmed (${SOLLSHA:0:16})"
  touch "/root/PRIO_${MK}_OK"
}
# The priors URL may come later: a host set up early overlaps image pull,
# venv and package download with the making of the priors.  To hand it in,
# write the URL to /root/.u_p25 (mode 600).
while [ -z "$U_P25" ] || [ "$U_P25" = "-" ]; do
  if [ -s /root/.u_p25 ]; then U_P25=$(cat /root/.u_p25); break; fi
  sag "waiting for the priors URL (/root/.u_p25); everything else is done"
  sleep 120
done
hole_prioren /root/pkg_v7/ppmd-probs.float16 "$U_P25" 25

# 6 second arm, pilot mode only ---------------------------------------------
if [ "$MODUS" = pilot ]; then
  for f in $GETEILT; do
    [ -f /root/pkg_v7o20/$f ] || ln /root/pkg_v7/$f /root/pkg_v7o20/$f
  done
  if [ ! -s /root/pkg_v7o20/ppmd-bytes.uint8 ]; then
    hole_parallel "$U_B20" /root/pkg_v7o20/ppmd-bytes.uint8 "$N" \
      || { sag "ppmd-bytes(20) NOT FETCHED"; exit 12; }
  fi
  [ "$(stat -c%s /root/pkg_v7o20/ppmd-bytes.uint8)" = "$N" ] || { sag "ppmd-bytes(20) has the wrong length"; exit 12; }
  # Both arms share zeilen.dump, doc-rows and doc-last-ziel, so masks and
  # targets are the same and the losses compare directly, provided ppmd-bytes
  # holds the stream token of each position.
  S25=$(sha256sum /root/pkg_v7/ppmd-bytes.uint8 | cut -c1-16)
  S20=$(sha256sum /root/pkg_v7o20/ppmd-bytes.uint8 | cut -c1-16)
  if [ "$S25" = "$S20" ]; then
    sag "ppmd-bytes of both orders EQUAL ($S25) => same targets, losses compare directly"
  else
    sag "NOTE: ppmd-bytes DIFFER (25:$S25 20:$S20) => different target tokens; compare losses only on the common set"
  fi
  while [ -z "$U_P20" ] || [ "$U_P20" = "-" ]; do
    if [ -s /root/.u_p20 ]; then U_P20=$(cat /root/.u_p20); break; fi
    sag "waiting for the priors URL of order 20 (/root/.u_p20)"
    sleep 120
  done
  hole_prioren /root/pkg_v7o20/ppmd-probs.float16 "$U_P20" 20
fi

# 7 loader check per arm ----------------------------------------------------
I=0
for D in $BAEUME; do
  I=$((I+1)); P=$(echo $PAKETE | cut -d' ' -f$I)
  cd "$D" && uv run python -m tft.dataset check "$P" 208 | tail -4 | tee -a $L
done
# the line printed by `tft.dataset check` (tree_overlay) on success
[ "$(grep -c 'package matches the training-data interface' $L)" -ge "$N_BAEUME" ] || { sag "LOADER CHECK FAILED"; exit 13; }

sag "### SETUP DONE ($MODUS)"
touch /root/BOOT_V7_OK

# 8 start -------------------------------------------------------------------
if [ "$MODUS" = pilot ]; then
  tmux new-session -d -s p25 "BAUM=/root/fx6m_p25 ENVD=/root/final6m_v7_p25.env KARTEN=0,1 PORT=29751 bash /root/kette_arm.sh > /root/lauf_p25.log 2>&1"
  tmux new-session -d -s p20 "BAUM=/root/fx6m_p20 ENVD=/root/final6m_v7_p20.env KARTEN=2,3 PORT=29781 bash /root/kette_arm.sh > /root/lauf_p20.log 2>&1"
  sag "### BOTH ARMS STARTED (tmux p25 cards 0,1 / p20 cards 2,3)"
elif [ "$MODUS" = zweig ]; then
  sag "### SETUP DONE; the arm is started separately"
elif [ "$MODUS" = acht ]; then
  tmux new-session -d -s acht "BAUM=/root/fx6m_8m ENVD=${ANCHOR_ENV:?env file of section 1, e.g. /root/chain/01_anchor_w2048.env} KARTEN=0,1 PORT=29751 bash /root/kette_arm.sh > /root/lauf_8m.log 2>&1"
  sag "### 8M ANCHOR STARTED (tmux acht, 2 cards)"
else
  tmux new-session -d -s kette "BAUM=/root/fx6m_k ENVD=/root/final6m_v7.env KARTEN=0,1,2,3 PORT=29751 bash /root/kette_arm.sh > /root/lauf_kette.log 2>&1"
  sag "### CHAIN STARTED (tmux kette, 4 cards)"
fi
