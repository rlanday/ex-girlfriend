set -euo pipefail

sudo apt-get update
sudo apt-get install upx-ucl make -y

wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 17
rm llvm.sh
