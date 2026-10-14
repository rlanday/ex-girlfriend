#!/bin/bash -x

set -euo pipefail

#SEED="$1"
#UPDATE_LIMIT="$2"

SEED="923"
UPDATE_LIMIT="3000"

rm -rf pgo_data 
mkdir -p pgo_data 

# building with PGO
CFLAGS_DEFINES="-DSEED=$SEED -DUPDATE_LIMIT=$UPDATE_LIMIT"
make CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_gen -j 

./cmix -c ./prof_input/input ./prof_comp > ./prof_output
rm ./prof_comp ./prof_output 
llvm-profdata-17 merge -output=default.profdata ./pgo_data/*
mv default.profdata pgo_data/

make CFLAGS_DEFINES="$CFLAGS_DEFINES" prof_use -j
 upx-ucl -9 cmix 

# this is a directory where the compressor binary will be placed 
DIR=run
mkdir -p ./$DIR
ROOT=$(pwd)
cp ./cmix $DIR/cmix_orig
# git diff > $DIR/patch
# exit
# building a selfextracting binary 
pushd $DIR
# creating a compressed version of dictionary
./cmix_orig -c $ROOT/dictionary/english.dic ./comp_dict
# creating a compressed verions of a file with new order
./cmix_orig -c $ROOT/src/readalike_prepr/data/new_article_order ./comp_order
# the transformer weights are already losslessly compressed (FX2TFWC2, made by
# pysrc/weights_compress.py from the export of models/6m-q4-fp32.tch) and are
# loaded directly in that format, so they are appended as-is
cp $ROOT/models/6m-q4-fp32.tfwc2 ./comp_tfweights
# creating a header with size of the above files
./cmix_orig -h $(wc -c ./comp_dict | awk '{print $1}') $(wc -c ./comp_order | awk '{print $1}') 0 $(wc -c ./comp_tfweights | awk '{print $1}')

# merging the above files and setting permissions for the final executable file
cat ./cmix_orig ./comp_dict ./comp_order ./comp_tfweights header.dat > ./cmix
chmod +x ./cmix
