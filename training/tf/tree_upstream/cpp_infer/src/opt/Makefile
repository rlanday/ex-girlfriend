# Optimized quantized-matmul kernel suite (qmat). clang++-17 is the
# production compiler (MACHINE.md: gcc spills in the inner loops).
CXX      ?= clang++-17
CXXFLAGS ?= -O3 -march=znver2 -std=c++17 -Wall -Wextra -fno-math-errno
# NO -ffast-math (SPEC forbids it)

BIN := ../../bin/opt
SRCS := qmat_dense.cpp qmat_sparse.cpp
HDRS := qmat.h qmat_dense.h qmat_sparse.h
OBJS := $(BIN)/qmat_dense.o $(BIN)/qmat_sparse.o

all: $(BIN)/test_qmat $(BIN)/bench_qmat

$(BIN):
	mkdir -p $(BIN)

$(BIN)/%.o: %.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/kernels.o: ../kernels.cpp ../kernels.h | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/test_qmat: test_qmat.cpp $(OBJS) $(BIN)/kernels.o $(HDRS)
	$(CXX) $(CXXFLAGS) test_qmat.cpp $(OBJS) $(BIN)/kernels.o -o $@ -lm

$(BIN)/bench_qmat: bench_qmat.cpp $(OBJS) $(BIN)/kernels.o $(HDRS) ../../bench/bench_common.h
	$(CXX) $(CXXFLAGS) bench_qmat.cpp $(OBJS) $(BIN)/kernels.o -o $@ -lm

clean:
	rm -rf $(BIN)

.PHONY: all clean
