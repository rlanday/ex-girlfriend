#!/bin/bash
# Builds the replay trainer of the BITLSTM.
#   bash build.sh [output]        (default: ./replay_trainer)
# The flags are those the shipped weights were trained with.  bit_head.cpp
# must keep -ffp-model=precise -ffp-contract=off: the model has to compute
# the same bits as the compressor's copy.  CXX defaults to clang++-18; the
# binary the weights were trained with was built with clang 18.1.3.
set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-clang++-18}
OUT=${1:-replay_trainer}
FLAGS="-O3 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie \
-ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin \
-fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables \
-fdata-sections -ffunction-sections -Wall -Wno-unused-variable"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$CXX" $FLAGS -c bit_head.cpp -o "$TMP/bit_head.o"
"$CXX" $FLAGS -no-pie replay_trainer.cpp "$TMP/bit_head.o" -o "$OUT"
echo "built $OUT"
