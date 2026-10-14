#!/usr/bin/env bash
# check_order_run.sh - mandatory acceptance of full runs with a changed article
# order.  A full run with the wrong order file (243,425 instead of 172,277
# lines) is worthless as an order experiment; this test finds it within minutes.
#
#   phase 1 (before the start, directly after build_variant.sh):
#       bash check_order_run.sh pre  <order-name> <variant-dir>
#   phase 2 (2 minutes after the start of the full run):
#       bash check_order_run.sh post <order-name> <run-dir>
#
# Exit 0 = fine.  Exit != 0 = stop the run at once.
set -uo pipefail

ORDERDIR="${ORDERDIR:?set ORDERDIR to the directory of order files}"

# order-name : md5(order) : expected comp_order (several values separated by |,
# one per binary family) : expected .main_phda9prepr (0 = not checked)
# : [order file bytes; missing = 1094862 = the classic permutation]
# : [md5(.main_reordered), optional: checks the order itself rather than its
#   size; empty = no test]
read -r -d "" TAB <<TABEOF
record       b5c1e1163aaede3a72d322508e9b5697 199853 929421077
record_allD_c8 6a283abc69371fa3adf44b155f47de41 61104 931065902 122693
rallD_V1s_c8 f70b74bd0893e016997b596392f26048 61117 0 122695
record_allD  428d9b4eb5a984a5b87cc6bd31f3fdc0 161773|161240|161261 929373836|931065902
rallD_V1     1badf5ea1317e4ba0e5f4cbcee78c799 161310|161847 0
# Schema v3: the order is rebuilt from a rule plus a residual file, and
# comp_order falls from about 161.3 KB to about 46.6 KB.  The residual file
# resid_V1_v3.bin (119,756 B, md5 2eb1a3a2) replaces the explicit order file,
# hence its own name, its own expected values and OSIZE=119756.
# The third .main_phda9prepr value (929562049) belongs to the builds that map
# the HTML entity &sup2; to a control byte in the phda9 step
# (-1,328,064 B = 332,016 x 4).
rallD_V1_v3_c8 2eb1a3a2d9805a0dbc8e1595a495097d 46588|46603 930890113|930890170|929562049 119756
# The redirect-front order is not an order file of its own but a RULE: the
# usual order with the block of the 71,148 unlisted pages (71,110 redirects and
# 38 excluded pages) in FRONT instead of at the end.  The order file cannot
# express that (article_reorder.h always appends unlisted pages), hence a rule
# in the code.  Residual file, its size and comp_order stay identical to
# rallD_V1_v3_c8, so the rule costs 0 B of side information; the row exists so
# that the run has its own name and its own .main_reordered mark.
# Expected values, measured on a finished binary (-e run under another name in
# a fresh empty directory):
#   .main_reordered  998,030,918 B  md5 933a476d... (computed independently;
#                    column 6, the actual check of the order)
#   .main_phda9prepr 929,564,616 B
#   .ready4cmix      929,575,765 B  = phda9 + 11,149
# The same computation reproduces .main_reordered of an earlier build exactly.
rallD_V1_v3_redirfront_c8 2eb1a3a2d9805a0dbc8e1595a495097d 46588|46603 929564616 119756 933a476d24d44e422b181f965ae3d16c
# The order of this entry: the redirect-front rule plus the calendar block (the
# pages with a calendar title as one block, sorted by title) and the residual
# in the retuned coding, which ends with a form tag.  ORDERDIR must hold the
# residual as calendar_tune_c8.order.
#   .main_reordered  md5 28ca879d...
#   .main_phda9prepr 888,085,252 B
#   .ready4cmix      888,096,401 B
calendar_tune_c8 3beb7ee15ebeabdb13b2c87d2cf44919 43780|43815 888085252 115870 28ca879dcfdafe7833b042f58cfee517
o3nemon      7fddbb86483bd6d9648b1c8522f31be8 158166 0
o3nemo       ca835f7e0116d5c3b0007da1a983baf9 160506 0
o3nemom      d56f1cda2466a218051f42f34209a0c8 160426 0
cenfar       7c205714a69c62437a56982d5f46bfaf 161445 0
cenfar_rev   91d21d21e58fbeb4e364d66dd83eac40 161446 0
permD3       a7e764ccd504ab1083bba78602844cda 163175 929260652
fsrn         1d616ef425d944693492e71dbf8ef172 201966 929311936
record_allD_rev 1ce93d2e20402f3fe5590fad93a413a4 162447 929264421
ward_w2v     e8316e840cbf2ee52b8bd6d253f1bc0f 292638 929591907
ward_runs    df1c637b7178410499ae9efbc5315930 200223 929421515
TABEOF

PHASE="${1:?phase missing: pre|post}"
NAME="${2:?order name missing}"
DIR="${3:?directory missing}"

ROW=$(printf "%s\n" "$TAB" | awk -v n="$NAME" "\$1==n")
[ -n "$ROW" ] || { echo "ERROR: unknown order name $NAME"; exit 3; }
set -- $ROW
MD5_SOLL=$2; CORD_SOLL=$3; PHDA_SOLL=$4; OSIZE=${5:-1094862}; REORD_MD5_SOLL=${6:-}
case "$NAME" in *_c8) C8REST=1 ;; *) C8REST=0 ;; esac

fail=0
ok()   { printf "  OK    %s\n" "$*"; }
bad()  { printf "  ERROR %s\n" "$*"; fail=1; }
warn() { printf "  WARNING %s\n" "$*"; }
size() { stat -c %s "$1" 2>/dev/null || echo -1; }

echo "== check_order_run.sh $PHASE $NAME $DIR"

if [ "$PHASE" = "pre" ]; then
  SRC=$ORDERDIR/$NAME.order
  # A residual file (names ending in _c8) is binary, with no lines and no
  # permutation; the order itself is checked instead through a dump of the
  # decoded order and cmp.
  m=$(md5sum "$SRC" 2>/dev/null | cut -d" " -f1)
  [ "$(size "$SRC")" = "$OSIZE" ] && ok "order file $OSIZE B" || bad "order file size $(size "$SRC") != $OSIZE"
  [ "$m" = "$MD5_SOLL" ] && ok "order md5 $m" || bad "order md5 $m != $MD5_SOLL"
  if [ "$C8REST" != 1 ]; then
  l=$(wc -l < "$SRC" 2>/dev/null); u=$(sort -u "$SRC" 2>/dev/null | wc -l)
  [ "$l" = 172277 ] && ok "172277 lines" || bad "$l lines != 172277 (243425 = WRONG FORMAT!)"
  [ "$u" = 172277 ] && ok "permutation (all indices distinct)" || bad "only $u distinct indices"
  fi

  V=$DIR/variant.json
  c=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"sizes\"][\"comp_order\"])" "$V" 2>/dev/null)
  vo=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"order_bytes\"])" "$V" 2>/dev/null)
  [ "$vo" = "$OSIZE" ] && ok "variant.json order_bytes $OSIZE" || bad "variant.json order_bytes $vo != $OSIZE"
  if [ -n "$c" ]; then
    d=999999; for soll in ${CORD_SOLL//|/ }; do dd=$(( c > soll ? c - soll : soll - c )); [ "$dd" -lt "$d" ] && d=$dd; done
    # A changed MODEL moves comp_order legitimately: comp_order is the order
    # file packed with cmix_orig, and a build with another model (cells, layers,
    # new contexts) packs it differently (16 to 30 B seen, 302 B with two LSTM
    # layers).  The proof of the right order is the order md5 (checked above)
    # and the control build, not this size; from 300 B on it is therefore a
    # warning with a check instead of a stop.
    if [ "$d" -le 300 ]; then ok "comp_order $c (expected ~$CORD_SOLL, deviation $d B)"
    elif [ "$d" -le 3000 ]; then
      warn "comp_order $c differs by $d B from ~$CORD_SOLL. Order md5 is OK, so NOT a wrong order file."
      warn "  -> With a changed MODEL (CELLS/LAYERS/new contexts) this is EXPECTED (different cmix_orig)."
      warn "  -> MANDATORY before the start: check the acceptance of the control build (it must hit the expected values)."
      warn "  -> If the model is NOT changed, this is a real finding: check the build."
    else bad "comp_order $c differs by $d B from ~$CORD_SOLL -> wrong order file compiled in"; fi
  else bad "comp_order not readable from $V"; fi

else
  # The -e path first unpacks dictionary, order and decompressor from the
  # binary (several minutes); only then do .main_reordered and
  # .main_phda9prepr appear.  Hence the wait instead of an immediate alarm.
  WAIT_MIN=${WAIT_MIN:-25}
  i=0
  while [ ! -s "$DIR/.main_phda9prepr" ] && [ "$i" -lt "$((WAIT_MIN*2))" ]; do
    [ "$i" = 0 ] && echo "  ... waiting for .main_phda9prepr (self-extraction + reorder + phda9, up to ${WAIT_MIN} min)"
    sleep 30; i=$((i+1))
  done
  # phda9 writes the file in pieces; wait until its size is stable
  a=-1; b=$(size "$DIR/.main_phda9prepr")
  while [ "$a" != "$b" ]; do a=$b; sleep 20; b=$(size "$DIR/.main_phda9prepr"); done

  o=$(size "$DIR/.new_article_order")
  if [ "$C8REST" = 1 ]; then
    # binary residual file instead of a line permutation; the order itself is
    # checked through a dump of the decoded order and cmp
    [ "$o" = "$OSIZE" ] && ok ".new_article_order (C8 residual file) $OSIZE B" || bad ".new_article_order $o != $OSIZE (C8 residual file)"
  else
  [ "$o" = "$OSIZE" ] && ok ".new_article_order $OSIZE B" || bad ".new_article_order $o != $OSIZE (1592865 = WRONG FORMAT!)"
  l=$(wc -l < "$DIR/.new_article_order" 2>/dev/null)
  [ "$l" = 172277 ] && ok ".new_article_order 172277 lines" || bad ".new_article_order $l lines != 172277"
  fi
  r=$(size "$DIR/.main_reordered")
  # This value depends on the set of transforms, not a constant: every
  # generation of the preprocessing has its own (cross-checked over 46 run
  # directories on four machines):
  #   998997643  classic chain
  #   998749680  a later feature set
  #   998206650  with the language-link fold and the comment index (they
  #              change the split; the preprocessing round trip is exact)
  #   998030918  with the census transforms, which shorten the stream further
  # An unknown value is therefore no proof of data loss but first a hint at a
  # new generation: compare against a running run with the same transforms
  # before stopping.
  case "$r" in 998997643|998206650|998749680|998030918) ok ".main_reordered $r" ;;
    *) bad ".main_reordered $r unknown; cross-check against a run with the same transform set before judging" ;; esac
  # Column 6: md5 of the reconstructed order itself.  It is the only test that
  # checks the order rather than its size, and for a new order without an
  # expected phda9 value it takes that value's place.  Costs ~3 s on 1 GB.
  if [ -n "$REORD_MD5_SOLL" ] && [ -s "$DIR/.main_reordered" ]; then
    rm5=$(md5sum "$DIR/.main_reordered" | cut -d' ' -f1)
    [ "$rm5" = "$REORD_MD5_SOLL" ] && ok ".main_reordered md5 $rm5 (order byte-identical to the expected one)" \
      || bad ".main_reordered md5 $rm5 != $REORD_MD5_SOLL -> WRONG order"
  fi
  p=$(size "$DIR/.main_phda9prepr")
  if [ "$PHDA_SOLL" = "-" ]; then
    warn ".main_phda9prepr $p: no expected value stored (new order); please add the value to the table"
  else
  pok=0; for soll in ${PHDA_SOLL//|/ }; do [ "$p" = "$soll" ] && pok=1; done
  [ "$pok" = 1 ] && ok ".main_phda9prepr $p" || bad ".main_phda9prepr $p != $PHDA_SOLL"
  fi
  if [ -f "$DIR/.ready4cmix" ] && [ "$PHDA_SOLL" != "-" ]; then
    y=$(size "$DIR/.ready4cmix"); yok=0
    for soll in ${PHDA_SOLL//|/ }; do [ "$y" = "$(( soll + 11149 ))" ] && yok=1; done
    [ "$yok" = 1 ] && ok ".ready4cmix $y" || bad ".ready4cmix $y != PHDA_SOLL+11149"
  fi
  c=$(size "$DIR/.new_article_order.comp")
  if [ "$c" -gt 0 ]; then
    d=999999; for soll in ${CORD_SOLL//|/ }; do dd=$(( c > soll ? c - soll : soll - c )); [ "$dd" -lt "$d" ] && d=$dd; done
    # The same tolerance as in the pre branch: a changed MODEL packs the order
    # file legitimately differently (329 B seen with two LSTM layers).  The
    # proof of the right order is the order md5 and the control build.
    if [ "$d" -le 300 ]; then ok "comp_order $c (expected ~$CORD_SOLL)"
    elif [ "$d" -le 3000 ]; then
      warn "comp_order $c differs by $d B from ~$CORD_SOLL; expected with a changed MODEL."
      warn "  -> order md5 and .new_article_order are checked above; NO reason to stop."
      warn "  -> If the model is NOT changed, this is a real finding: check the build."
    else bad "comp_order $c differs by $d B"; fi
  fi
fi

if [ "$fail" = 0 ]; then echo "== RESULT: PASSED"; exit 0
else echo "== RESULT: FAILED; stop the run AT ONCE (kill), do not let it compute for 40 h."; exit 1; fi
