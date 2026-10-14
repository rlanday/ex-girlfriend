# cmix - the compressor of this entry.
#
#   make cmix        the compressor binary ./cmix
#   make prof_gen    first pass of the profile-guided build
#   make prof_use    ./cmix with profile and LTO  <- the submitted recipe
#   make cmix_aux    the build helper (writes the embedded blocks; NOT shipped)
#   make sstrip      the section-header stripper used when packing
#   make clean
#
# The complete submitted binary is produced by
#   bash build_and_construct_comp.sh
# (profile, LTO, strip, pack, embed dictionary and article order).
#
# There are no build options.  Every value below is fixed; the source contains
# no conditional compilation beyond the platform guards it inherited.  The three
# translation groups (fast / slow / cold) and the per-file flag sets are NOT
# cosmetic: several files must be compiled with -ffp-model=precise and without
# contraction, and the binary is only reproducible when they are.

ADEPTHDET_CXXFLAGS := -O2 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections -Wall -Wno-unused-variable
ADEPTHDET_DEP := adepthdet
ADEPTHDET_OBJ := adaptive_depth.o
ADEPTHDET_SRC := src/adaptive_depth.cpp
ALLOC2_DEP := 
ALLOC2_OBJ := 
ARCH_FLAGS := -march=x86-64-v3 -mtune=znver2
B6_HINTS_DEP := b6hints
B6_HINTS_OBJ := dict_derive.o
BITL_CXXFLAGS := -O3 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections -Wall -Wno-unused-variable
BITL_DEP := bitl
BITL_OBJ := bit_head.o
BITL_SRC := src/bit_head.cpp
CC ?= clang++-18
CFLAGS_DEFINES ?= -DSEED=923 -DUPDATE_LIMIT=3000
CPPFLAGS_PART-THAT-CAN-BE-SLOW := -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG -m64 -Wall -std=c++17 -ffp-model=fast -mrecip=none -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-threadsafe-statics -Wno-unknown-escape-sequence -Wno-unused-variable -Wno-unneeded-internal-declaration -Wno-unused-but-set-variable -Wno-format -march=x86-64-v3 -mtune=znver2 -Os -fdata-sections -ffunction-sections -fno-unroll-loops -fno-pie -fmerge-all-constants -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -Oz
CPPFLAGS_PART-THAT-IS-COLD := -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG -m64 -Wall -std=c++17 -ffp-model=fast -mrecip=none -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-threadsafe-statics -Wno-unknown-escape-sequence -Wno-unused-variable -Wno-unneeded-internal-declaration -Wno-unused-but-set-variable -Wno-format -Oz -fdata-sections -ffunction-sections -march=x86-64-v3 -mtune=znver2 -fno-unroll-loops -fno-pie -fmerge-all-constants -fno-direct-access-external-data -fno-rtti -fvisibility=hidden
CPPFLAGS_PART-THAT-SHOULD-BE-FAST := -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG -m64 -Wall -std=c++17 -ffp-model=fast -mrecip=none -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-threadsafe-statics -Wno-unknown-escape-sequence -Wno-unused-variable -Wno-unneeded-internal-declaration -Wno-unused-but-set-variable -Wno-format -march=x86-64-v3 -mtune=znver2 -O3 -fdata-sections -ffunction-sections -fno-unroll-loops -fno-pie -fmerge-all-constants -fno-direct-access-external-data -fno-rtti -fvisibility=hidden
DETNUM_CXXFLAGS := -O2 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections
DETNUM_DEP := detnum
DETNUM_OBJ := detmath.o
DETNUM_SRC := src/detmath.cpp
DET_ARCH := -march=x86-64-v3 -mtune=znver2
DET_FP := -ffp-model=fast -mrecip=none
FP_MODEL := -ffp-model=fast -mrecip=none
LFLAGS := -m64 -fuse-ld=lld-18 -Wl,--gc-sections -std=c++17 -no-pie -Wl,--icf=all -Wl,-z,norelro -Wl,--version-script=runtime_exports.map -Wl,--as-needed -Wl,--build-id=none
STRIP_FLAG ?= -s
OKB_EXTRA := 
OPT1VEC_CXXFLAGS := -O3 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections
OPT1VEC_DEP := opt1vec
OPT1VEC_OBJ := optimizer_vec.o
OPT1VEC_SRC := src/optimizer_vec.cpp
RT_COMPAT_OBJ := glibc_compat.o
S26VEC_CXXFLAGS := -O3 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections
S26VEC_DEP := s26vec
S26VEC_OBJ := fast_math_vec.o
S26VEC_SRC := src/fast_math_vec.cpp
TF1_CXXFLAGS := -O3 -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -mrecip=none -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections -Wno-unused-variable -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG
TF1_DEP := tf1
TF1_GATE_G1_DEFS := 
TF1_OBJ := tf1_engine.o weights_io.o weights_io_compressed.o kernels.o model.o arena_build.o attn.o glue.o kda.o qmat_dense.o qmat_sparse.o model_opt.o tf1_z3.o tf1_prompt.o tf1_gate.o
TF1_OZ_FLAGS := -Oz
TF1_SRCS := src/tf1/tf1_engine.cpp src/tf1/weights_io.cpp src/tf1/weights_io_compressed.cpp src/tf1/kernels.cpp src/tf1/model.cpp src/tf1/opt/arena_build.cpp src/tf1/opt/attn.cpp src/tf1/opt/glue.cpp src/tf1/opt/kda.cpp src/tf1/opt/qmat_dense.cpp src/tf1/opt/qmat_sparse.cpp src/tf1/opt/model_opt.cpp src/tf1/tf1_z3.cpp src/tf1/tf1_prompt.cpp src/tf1/tf1_gate.cpp
TF1_SRCS_OZ := src/tf1/weights_io.cpp src/tf1/weights_io_compressed.cpp src/tf1/opt/arena_build.cpp
WORDLBL_CXXFLAGS := -Os -m64 -std=c++17 -march=x86-64-v3 -mtune=znver2 -fno-pie -fno-direct-access-external-data -fno-rtti -fvisibility=hidden -ffp-model=precise -ffp-contract=off -fno-math-errno -fno-builtin -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections -Wall -Wno-unused-variable
WORDLBL_DEP := wordlbl
WORDLBL_OBJ := word_label.o
WORDLBL_SRC := src/word_label.cpp

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
LFLAGS += -Wl,-T,$(ROOT_DIR)/encoder_split.lds

prof_gen: CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: CPPFLAGS_PART-THAT-IS-COLD        += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: LFLAGS                            += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: clean cmix

prof_use: CPPFLAGS_PART-THAT-CAN-BE-SLOW    +=
prof_use: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -flto
prof_use: CPPFLAGS_PART-THAT-IS-COLD        +=
prof_use: LFLAGS                            += -flto
prof_use: clean cmix


slow: src/preprocess/preprocessor.cpp src/preprocess/preprocessor.h src/preprocess/dictionary.cpp src/preprocess/dictionary.h
	$(CC) $(CPPFLAGS_PART-THAT-CAN-BE-SLOW) src/preprocess/preprocessor.cpp src/preprocess/dictionary.cpp -c 

cold: src/r1_reorder_transform.cpp src/r1_reorder_transform.h src/runner.cpp src/readalike_prepr/cmtidx_transform.h src/readalike_prepr/misc.h
	$(CC) $(CPPFLAGS_PART-THAT-IS-COLD) src/r1_reorder_transform.cpp src/runner.cpp -c

fast: src/coder/decoder.cpp src/coder/decoder.h src/coder/encoder.cpp src/coder/encoder.h src/context-manager.cpp src/context-manager.h src/contexts/bit-context.cpp src/contexts/bit-context.h src/contexts/bracket-context.cpp src/contexts/bracket-context.h src/contexts/combined-context.cpp src/contexts/combined-context.h src/contexts/context-hash.cpp src/contexts/context-hash.h src/contexts/context.h src/contexts/indirect-hash.cpp src/contexts/indirect-hash.h src/contexts/interval-hash.cpp src/contexts/interval-hash.h src/contexts/interval.cpp src/contexts/interval.h src/contexts/sparse.cpp src/contexts/sparse.h src/models/bracket.cpp src/models/bracket.h src/models/byte-model.cpp src/models/byte-model.h src/models/direct-hash.cpp src/models/direct-hash.h src/models/direct.cpp src/models/direct.h src/models/indirect.cpp src/models/indirect.h src/models/match.cpp src/models/match.h src/models/grammar-match.cpp src/models/grammar-match.h src/models/model.h src/models/fxcmv1.h src/models/ppmd.cpp src/models/ppmd.h src/states/nonstationary.cpp src/states/nonstationary.h src/states/run-map.cpp src/states/run-map.h src/states/state.h src/mixer/byte-mixer.cpp src/mixer/byte-mixer.h src/mixer/lstm-layer.h src/mixer/lstm.h src/mixer/mixer-input.cpp src/mixer/mixer-input.h src/mixer/mixer.cpp src/mixer/mixer.h src/mixer/sigmoid.cpp src/mixer/sigmoid.h src/mixer/sse.cpp src/mixer/sse.h src/predictor.h src/predictor.cpp src/page_contexts.h src/fast_math.h src/fx_mixer_lr.h
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) src/coder/decoder.cpp src/coder/encoder.cpp src/context-manager.cpp src/contexts/bit-context.cpp src/contexts/bracket-context.cpp src/contexts/combined-context.cpp src/contexts/context-hash.cpp src/contexts/indirect-hash.cpp src/contexts/interval-hash.cpp src/contexts/interval.cpp src/contexts/sparse.cpp src/models/bracket.cpp src/models/byte-model.cpp src/models/direct-hash.cpp src/models/direct.cpp src/models/indirect.cpp src/models/match.cpp src/models/grammar-match.cpp src/models/fxcmv1.cpp src/models/ppmd.cpp src/states/nonstationary.cpp src/states/run-map.cpp src/mixer/byte-mixer.cpp src/mixer/mixer-input.cpp src/mixer/mixer.cpp src/mixer/sigmoid.cpp src/mixer/sse.cpp -c src/predictor.cpp

# Forwarders for the __isoc23_* entry points of newer glibc headers
# (src/glibc_compat.cpp).
glibc_compat.o: src/glibc_compat.cpp
	$(CC) $(CPPFLAGS_PART-THAT-IS-COLD) -c src/glibc_compat.cpp -o glibc_compat.o

# Own transcendental functions: own translation unit with strict FP semantics,
# no PGO/LTO.  A prerequisite of cmix only in builds that use them
# (DETNUM_DEP).
detnum: $(DETNUM_SRC)
	$(CC) $(DETNUM_CXXFLAGS) -c $(DETNUM_SRC) -o detmath.o

# Vectorised fast-math kernels: strict FP flags, see S26VEC_CXXFLAGS above.
s26vec: $(S26VEC_SRC) src/fast_math.h
	$(CC) $(S26VEC_CXXFLAGS) -c $(S26VEC_SRC) -o fast_math_vec.o

# Vectorised optimizer step: strict FP flags, see OPT1VEC_CXXFLAGS above.
opt1vec: $(OPT1VEC_SRC) src/optimizer.h
	$(CC) $(OPT1VEC_CXXFLAGS) -c $(OPT1VEC_SRC) -o optimizer_vec.o

# See WORDLBL_CXXFLAGS above.
wordlbl: $(WORDLBL_SRC) src/word_label.h
	$(CC) $(WORDLBL_CXXFLAGS) -c $(WORDLBL_SRC) -o word_label.o

# Correction model: strict FP flags, see BITL_CXXFLAGS above.
bitl: $(BITL_SRC) src/bit_head.h
	$(CC) $(BITL_CXXFLAGS) -c $(BITL_SRC) -o bit_head.o

# see ADEPTHDET_CXXFLAGS above.
adepthdet: $(ADEPTHDET_SRC) src/adaptive_depth.h
	$(CC) $(ADEPTHDET_CXXFLAGS) -c $(ADEPTHDET_SRC) -o adaptive_depth.o


# Dictionary derivation from the hint package, cold flag set (-Oz).
b6hints: src/preprocess/dict_derive.cpp src/preprocess/dict_derive.h
	$(CC) $(CPPFLAGS_PART-THAT-IS-COLD) src/preprocess/dict_derive.cpp -c

# -Wl,-Map=cmix.map writes the linker map for the section guard
# (tools_src/packer/cut_dpath.py, run from tools_src/packer/build_packer.sh
# on the image before sstrip).  The guard stops the build if a function that
# the linker script moved to the encoder-only section is reachable from the
# decompression path; the decompressor would otherwise crash late in a run.
# The map is a side file; the binary is identical with and without it.
comma := ,
# Transformer inference engine (src/tf1/); the load path is compiled for
# size.
tf1: $(TF1_SRCS)
	$(CC) $(TF1_CXXFLAGS) -c $(filter-out $(TF1_SRCS_OZ),$(TF1_SRCS))
	$(CC) $(TF1_CXXFLAGS) $(TF1_OZ_FLAGS) -c $(filter $(TF1_SRCS_OZ),$(TF1_SRCS))

cmix: fast slow cold $(RT_COMPAT_OBJ) $(DETNUM_DEP) $(S26VEC_DEP) $(OPT1VEC_DEP) $(B6_HINTS_DEP) $(BITL_DEP) $(TF1_DEP) $(WORDLBL_DEP) $(ADEPTHDET_DEP) $(ALLOC2_DEP)
	$(CC) $(LFLAGS) -Wl,-Map=cmix.map bit-context.o bracket-context.o bracket.o byte-mixer.o byte-model.o combined-context.o context-hash.o context-manager.o decoder.o dictionary.o direct-hash.o direct.o encoder.o indirect-hash.o indirect.o interval-hash.o interval.o match.o grammar-match.o mixer-input.o mixer.o nonstationary.o fxcmv1.o ppmd.o predictor.o preprocessor.o r1_reorder_transform.o run-map.o runner.o sigmoid.o sparse.o sse.o $(RT_COMPAT_OBJ) $(DETNUM_OBJ) $(S26VEC_OBJ) $(OPT1VEC_OBJ) $(B6_HINTS_OBJ) $(BITL_OBJ) $(TF1_OBJ) $(WORDLBL_OBJ) $(ADEPTHDET_OBJ) $(ALLOC2_OBJ) $(STRIP_FLAG) -o cmix
	rm -f *.o

# sstrip drops the section header table and .shstrtab from the linked ELF
# before packing (-484 B packed, -968 B in S).  The build uses
# `llvm-objcopy --strip-sections` (same result) and falls back to this program
# only where that option is missing.  Source: tools_src/sstrip.c
sstrip: tools_src/sstrip.c
	$(CC) -O2 -x c++ tools_src/sstrip.c -o sstrip

clean:
	rm -f *.o
	rm -f cmix
	rm -f tf1_dump
	rm -f tf1_z3
	rm -f remap
	rm -f sstrip

all: cmix

.PHONY: clean all prof_gen prof_use detnum s26vec opt1vec b6hints bitl tf1 cmix_aux fast slow cold


# The BUILD HELPER.  It is compiled from the same sources with AUX_BUILD=1 and
# writes the two blocks that get embedded into the compressor (the hint package
# and the article order).  It is never shipped and never packed; with
# AUX_BUILD=1 the transformer head does not run, so the blocks are written the
# same way they are later read.
cmix_aux: CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -DAUX_BUILD=1
cmix_aux: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -flto -DAUX_BUILD=1
cmix_aux: CPPFLAGS_PART-THAT-IS-COLD        += -DAUX_BUILD=1
cmix_aux: LFLAGS                            += -flto
cmix_aux: clean cmix
	mv cmix cmix_aux
