# Third-party licences

Three embedded libraries and code from two further sources are under
licences other than the GPL. Two of the libraries are PyTorch `c10` forms of
LLVM files, and the PyTorch changes are under BSD-3-Clause. The notices are
in the files themselves; the full licence texts they refer to are here.

| file in this package | component | licence | text |
|---|---|---|---|
| `src/ds/emhash_map.hpp`, `src/ds/emhash_set.hpp`, and the unchanged copies in `training/tf/tree_upstream/src/ds/` | emhash, Huang Yuanbing (`ktprime`) | MIT | `emhash-LICENSE.txt` (identical to the notice in the file header) |
| `src/ds/SmallVector.h`, `src/ds/SmallVector.hpp`, and the unchanged copies in `training/tf/tree_upstream/src/ds/` | LLVM `SmallVector`, in the form of PyTorch `c10` | Apache-2.0 WITH LLVM-exception; the PyTorch changes BSD-3-Clause | `LLVM-LICENSE.txt`, `PyTorch-LICENSE.txt` |
| `src/ds/AlignOf.h`, and the unchanged copy in `training/tf/tree_upstream/src/ds/` | LLVM `AlignOf` (legacy), in the form of PyTorch `c10` | University of Illinois/NCSA; the PyTorch changes BSD-3-Clause | `LLVM-LICENSE.txt`, section "Legacy LLVM License"; `PyTorch-LICENSE.txt` |
| `src/bit_head.cpp` and `training/bitl/trainer/bit_head.cpp` (`Exp8`), `src/tf1/opt/vec_math.h` (`exp256_ps`, `log256_ps`), `src/tf1/opt/kda_math.h` (`exp_full8`); in the unchanged upstream tree `training/tf/tree_upstream/cpp_infer/`: `src/opt/vec_math.h`, `src/opt/kda_math.h`, `bench/misc.cpp` | altered versions of `exp256_ps` and `log256_ps` of `avx_mathfun.h` (Giovanni Garberoglio), based on `sse_mathfun.h` (Julien Pommier) | zlib | `mathfun-LICENSE.txt` |
| `tools_src/packer/cm_core.h` (`cm_hashn`), `src/lstm_aux_contexts.h` (`BigCol`, `TriCol`); not called by the program: `src/invariants.h` (`Mix32`), `src/lstm_fx_link.h` (`UniByte`, `UniProb`) | the primes of XXH32 and its avalanche step, in full or its first steps, from xxHash (Yann Collet) | BSD-2-Clause | `xxHash-LICENSE.txt` (xxHash v0.8.4) |

`src/ds/SmallVector.h` was changed further for this entry; the notice at its
top lists the changes.

Everything else in this package is GPL-3, not "or later" (see the licence
section of `README.md`, `LICENSE` and `NOTICE.md`), except the Wikipedia
excerpts in `prof_input/` and `training/tf/tree_upstream/prof_input/`, which are
under the GFDL and CC BY-SA (see `NOTICE.md`).
