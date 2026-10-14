#!/usr/bin/env bash
# export the trained PoC checkpoint all the way to a verified tfwc2 container
#
#   models/<geometry>-q4-fp32.tch                     (cooldown phase output)
#     -> cpp_infer/data/weights-<geometry>.bin        (his tensor container)
#     -> cpp_infer/data/weights-<geometry>.c2         (tfwc2, range coded)
#     -> /tmp/roundtrip-<geometry>.bin                (decompressed again)
#     -> byte comparison against the .bin             (the verification step)
#
# Run from the fx2-cmix-transformer tree root, on the pod (needs CUDA: his
# export replicates the quantization arithmetic on the GPU on purpose, so that
# the exported ints are bit-identical to what training used).
#
#   TFT_GEOMETRY=poc4m bash run/export_chain.sh
#
set -euo pipefail

GEOMETRY="${TFT_GEOMETRY:-poc4m}"
export TFT_GEOMETRY
EXPORT_DIR="${TFT_EXPORT_DIR:-cpp_infer/data}"
BIN="$EXPORT_DIR/weights-$GEOMETRY.bin"
C2="$EXPORT_DIR/weights-$GEOMETRY.c2"
ROUNDTRIP="${TMPDIR:-/tmp}/roundtrip-$GEOMETRY.bin"

echo "=== 1/4  export fp32 checkpoint -> $BIN"
uv run python -m tft.export_weights_poc

echo "=== 2/4  compress -> $C2 (tfwc2)"
# his compressor verifies its own round trip internally and prints both sizes
uv run python -m pysrc.weights_compress compress "$BIN" "$C2"

echo "=== 3/4  decompress -> $ROUNDTRIP"
uv run python -m pysrc.weights_compress decompress "$C2" "$ROUNDTRIP"

echo "=== 4/4  verify"
# (a) his own verifier: every tensor decodes bit-exactly
uv run python -m pysrc.weights_compress verify "$BIN" "$C2"
# (b) an independent byte comparison of the decompressed container.  The
#     tensor ORDER is preserved by write_tensor_file, so the files must be
#     byte identical, not merely equivalent.
if cmp -s "$BIN" "$ROUNDTRIP"; then
  echo "roundtrip byte identical: OK"
else
  echo "roundtrip NOT byte identical; tensor order or metadata differ." >&2
  echo "his verify() above only compares tensor CONTENT, so this is a" >&2
  echo "container-level difference; report it before shipping the weights." >&2
  ls -l "$BIN" "$ROUNDTRIP" >&2
  exit 1
fi

echo
echo "sizes:"
ls -l "$BIN" "$C2" "$BIN.json" 2>/dev/null || true
echo
echo "deliver to the compressor build:"
echo "  $C2                    (the weights)"
echo "  $BIN.json              (geometry + vocabulary bitmap + tensor counts)"
echo "  $BIN.vocab-bitmap.u8   (32 bytes, index -> byte value)"
