CXX      := g++-13
CLANGXX  ?= clang++-17
CXXFLAGS := -O3 -march=znver2 -std=c++17 -Wall -Wextra

BINS := alu_peaks membw gemv misc gemv_clang

all: $(BINS)

alu_peaks: alu_peaks.cpp bench_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<
membw: membw.cpp bench_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<
gemv: gemv.cpp bench_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<
misc: misc.cpp bench_common.h
	$(CXX) $(CXXFLAGS) -o $@ $<
gemv_clang: gemv.cpp bench_common.h
	$(CLANGXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
