# Record of the shipped BITLSTM weights

`F2L` is the name the shipped weights carry in these records; the files below
are the outputs of `../tools/cut_trace.sh` and `../tools/train_blob.sh` for
them.

| file | content |
|---|---|
| `F2L_BITS` | B = 418,837,176, the number of model bits of the training range |
| `mark.txt` | the cut: B, the trace and channel sizes at the time of the cut, the prefix sizes 28 + 60 x B and 28 + 276 x B/8, and the sha256 of both prefixes |
| `training.txt` | one line per training pass: learning rate and the in-sample replay value |
| `SHA256SUMS` | the weights after pass 1 and pass 32 (f32, NBLB version 1) and the shipped bf16 file (NBLB version 2), in the layout `train_blob.sh` writes |
| `F2L_fp32.nblb` | the f32 weights after pass 32 (14,748 bytes, = `F2L/e32.nblb`); `python3 ../tools/nblb.py v2 F2L_fp32.nblb out.nblb` gives the shipped file |

The shipped file `F2L.nblb` (7,386 bytes) is the byte array compiled into the
compressor.  The trace itself (25.1 GB and 14.4 GB up to the cut) is not part of
this package; `mark.txt` identifies it.
