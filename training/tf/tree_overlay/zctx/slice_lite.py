#!/usr/bin/env python3
"""cut a "lite" package: tokens + boundaries + bitmap, WITHOUT the priors

The z3 extractor and every grammar check read only those three files, so a
slice of a few megabytes is enough to run the whole dry run on real data,
instead of moving 83 GB of priors around.  The cut lands on an article
boundary, so `zgrammar.verify` and `tft.dataset check` (minus the priors) still
apply.

    python -m zctx.slice_lite <src-package> <dst-dir> <max-tokens>
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np


def slice_lite(src: str, dst: str, max_tokens: int) -> dict:
    os.makedirs(dst, exist_ok=True)
    boundaries = np.fromfile(
        os.path.join(src, "article-boundaries.int32"), dtype=np.int32
    ).astype(np.int64)
    keep = boundaries[boundaries <= max_tokens]
    if len(keep) < 2:
        raise SystemExit("max_tokens is shorter than the first article")
    n = int(keep[-1])
    tokens = np.fromfile(os.path.join(src, "ppmd-bytes.uint8"), dtype=np.uint8, count=n)
    tokens.tofile(os.path.join(dst, "ppmd-bytes.uint8"))
    keep.astype(np.int32).tofile(os.path.join(dst, "article-boundaries.int32"))
    bitmap = os.path.join(src, "vocab-bitmap.u8")
    if os.path.isfile(bitmap):
        shutil.copy(bitmap, os.path.join(dst, "vocab-bitmap.u8"))
    return {"tokens": n, "articles": int(len(keep)) - 1, "dir": dst}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    for key, value in slice_lite(sys.argv[1], sys.argv[2], int(sys.argv[3])).items():
        print(str(key).ljust(12) + str(value))
