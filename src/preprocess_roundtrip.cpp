#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "preprocess/preprocessor.h"
#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/misc.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "r1_reorder_transform.h"

namespace {

bool CopyFile(const std::string& from, const std::string& to) {
  std::ifstream in(from, std::ios::binary);
  std::ofstream out(to, std::ios::binary | std::ios::trunc);
  if (!in.is_open() || !out.is_open()) return false;
  std::vector<char> buffer(1 << 20);
  while (in) {
    in.read(buffer.data(), buffer.size());
    const std::streamsize got = in.gcount();
    if (got > 0) out.write(buffer.data(), got);
  }
  return out.good();
}

size_t FileSize(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) return 0;
  return static_cast<size_t>(in.tellg());
}

bool FilesEqual(const std::string& a, const std::string& b) {
  std::ifstream left(a, std::ios::binary);
  std::ifstream right(b, std::ios::binary);
  if (!left.is_open() || !right.is_open()) return false;
  std::vector<char> lb(1 << 20);
  std::vector<char> rb(1 << 20);
  while (left && right) {
    left.read(lb.data(), lb.size());
    right.read(rb.data(), rb.size());
    const std::streamsize lg = left.gcount();
    const std::streamsize rg = right.gcount();
    if (lg != rg) return false;
    if (lg > 0 && std::memcmp(lb.data(), rb.data(), static_cast<size_t>(lg)) != 0) {
      return false;
    }
  }
  return left.eof() && right.eof();
}

void RemoveRoundtripScratch(const std::string& output_path) {
  const char* files[] = {
      ".roundtrip_ready_work",
      ".r1_payload_lex_side_roundtrip",
      ".input_decomp",
      ".intro_decomp",
      ".main_decomp",
      ".coda_decomp",
      ".main_decomp_restored",
      ".main_decomp_restored_sorted",
      "un1_d",
      ".census_resto",
      ".cmtidx_resto",
  };
  for (const char* file : files) std::remove(file);
  std::remove(output_path.c_str());
}

bool DecodeReadyStream(const std::string& ready_stream,
    const std::string& output_path) {
  FILE* in = std::fopen(ready_stream.c_str(), "rb");
  if (!in) {
    std::fprintf(stderr, "failed to open restored ready stream: %s\n",
        ready_stream.c_str());
    return false;
  }
  FILE* dict = std::fopen(".dict", "rb");
  if (!dict) {
    std::fprintf(stderr, "failed to open .dict; run packaged cmix -e prepare first\n");
    std::fclose(in);
    return false;
  }
  FILE* out = std::fopen(output_path.c_str(), "wb");
  if (!out) {
    std::fprintf(stderr, "failed to create dictionary-decoded stream: %s\n",
        output_path.c_str());
    std::fclose(in);
    std::fclose(dict);
    return false;
  }
  preprocessor::Decode(in, out, dict);
  std::fclose(in);
  std::fclose(dict);
  std::fclose(out);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr,
        "usage: %s transformed_ready_stream output_enwik9 [original_enwik9]\n",
        argv[0]);
    return 2;
  }

  const std::string transformed_ready = argv[1];
  const std::string output_enwik9 = argv[2];
  const std::string original_enwik9 = argc == 4 ? argv[3] : "";
  const std::string work_ready = ".roundtrip_ready_work";
  const std::string side = ".r1_payload_lex_side_roundtrip";

  RemoveRoundtripScratch(output_enwik9);

  std::cout << "roundtrip input ready stream: " << transformed_ready
            << " (" << FileSize(transformed_ready) << " bytes)\n";
  if (!CopyFile(transformed_ready, work_ready)) {
    std::fprintf(stderr, "failed to copy ready stream to scratch file\n");
    return 1;
  }

  if (!r1_reorder::ExtractSideFromFile(work_ready, side)) {
    std::fprintf(stderr, "payload_lex side extraction failed\n");
    return 1;
  }
  std::cout << "extracted side: " << FileSize(side) << " bytes\n";

  if (!r1_reorder::RestoreEncodedTailFile(work_ready, side)) {
    std::fprintf(stderr, "payload_lex tail restore failed\n");
    return 1;
  }
  std::cout << "restored ready stream: " << FileSize(work_ready) << " bytes\n";

  if (!DecodeReadyStream(work_ready, ".input_decomp")) {
    return 1;
  }
  std::cout << "dictionary-decoded stream: " << FileSize(".input_decomp")
            << " bytes\n";

  split4Decomp();
  phda9_resto();
  ::sort();
  if (!cmtidx_inplace(".main_decomp_restored_sorted", ".cmtidx_resto", 0)) {
    std::fprintf(stderr, "comment index: inverse transform failed\n");
    return 1;
  }
  if (!cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d") ||
      !cat("un1_d", ".coda_decomp", ".census_resto")) {
    std::fprintf(stderr, "final enwik9 merge failed\n");
    return 1;
  }
  if (!census_file(".census_resto", output_enwik9.c_str(), 0)) {
    std::fprintf(stderr, "census: inverse transform failed\n");
    return 1;
  }
  std::remove(".census_resto");

  std::cout << "reconstructed enwik9: " << FileSize(output_enwik9)
            << " bytes\n";
  if (!original_enwik9.empty()) {
    if (!FilesEqual(original_enwik9, output_enwik9)) {
      std::fprintf(stderr, "roundtrip mismatch: %s != %s\n",
          original_enwik9.c_str(), output_enwik9.c_str());
      return 1;
    }
    std::cout << "roundtrip byte-compare: OK\n";
  }
  return 0;
}
