set -euo pipefail

make data/enwik9
./build_and_construct_comp.sh

pushd run
./cmix -e ../data/enwik9 enwik9.comp --ppmd-only --save-ppmd-bytes ../data/ppmd-bytes.uint8 --save-ppmd-probs ../data/ppmd-probs.float16 --save-article-boundaries ../data/article-boundaries.int32
popd
