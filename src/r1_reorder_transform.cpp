// Payload-lex regime-1 transform for the fixed post-WRT enwik9 tail.
//
// This intentionally implements only the active submission path: reorder the
// known 45,332,670-byte encoded tail, append a compact side permutation, and
// restore it before dictionary decode.

#include "r1_reorder_transform.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "attributes.h"

#include <stdlib.h>

#include "r1_reorder_impl.inc"
