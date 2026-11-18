// Change in this file: side-channel inputs (paket_token_row and the
// four-argument step) and the sizes for 16 layers and 208 values.

// Origin: fx2-cmix-transformer by Vladimir Ivanov, GPL-3.0, file cpp_infer/src/model.cpp.
// Ported and modified for this entry in 2026; the changes are listed in
// sections 3 and 4 of README.md and of the README of cmix-neif-1.0.
#include "../meld.h"
#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "kernels.h"
#include "weights_io.h"
#include "tf1_config.h"

