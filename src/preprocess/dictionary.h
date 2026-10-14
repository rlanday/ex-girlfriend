#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <stdio.h>
#include "../attributes.h"
#include <string>
#include <array>
#include <list>

#include <stdlib.h>
#include <string.h>

namespace preprocessor {

class FastWordMap {
 public:
  ~FastWordMap() { free(blob_); free(off_); free(len_); free(code_);
                  free(tab_); free(rtab_); }

  void Reserve(unsigned int words, unsigned int blob_bytes) {
    cap_ = words ? words : 1;
    off_  = (unsigned int*)malloc(sizeof(unsigned int) * cap_);
    len_  = (unsigned int*)malloc(sizeof(unsigned int) * cap_);
    code_ = (unsigned int*)malloc(sizeof(unsigned int) * cap_);
    blob_cap_ = blob_bytes ? blob_bytes : 1;
    blob_ = (char*)malloc(blob_cap_);
    mask_ = 1; while (mask_ < 2 * cap_) mask_ <<= 1; mask_ -= 1;
    tab_  = (unsigned int*)calloc(mask_ + 1, sizeof(unsigned int));
    rtab_ = (unsigned int*)calloc(mask_ + 1, sizeof(unsigned int));
  }

  void Add(const char* w, unsigned int len, unsigned int code,
      bool encode, bool decode) {
    memcpy(blob_ + blob_len_, w, len);
    off_[n_] = blob_len_; len_[n_] = len; code_[n_] = code;
    blob_len_ += len;
    ++n_;
    if (encode) {
      unsigned int h = HashWord(w, len) & mask_;
      while (tab_[h]) h = (h + 1) & mask_;
      tab_[h] = n_;                      // entry index + 1
    }
    if (decode) {
      unsigned int h = HashCode(code) & mask_;
      while (rtab_[h]) h = (h + 1) & mask_;
      rtab_[h] = n_;
    }
  }

  bool Find(const char* w, unsigned int len, unsigned int* code) const {
    if (!tab_) return false;
    unsigned int h = HashWord(w, len) & mask_;
    while (tab_[h]) {
      const unsigned int i = tab_[h] - 1;
      if (len_[i] == len && memcmp(blob_ + off_[i], w, len) == 0) {
        *code = code_[i]; return true;
      }
      h = (h + 1) & mask_;
    }
    return false;
  }

  bool FindCode(unsigned int code, const char** w, unsigned int* len) const {
    if (!rtab_) return false;
    unsigned int h = HashCode(code) & mask_;
    while (rtab_[h]) {
      const unsigned int i = rtab_[h] - 1;
      if (code_[i] == code) { *w = blob_ + off_[i]; *len = len_[i]; return true; }
      h = (h + 1) & mask_;
    }
    return false;
  }

 private:
  static unsigned int HashWord(const char* w, unsigned int len) {
    unsigned int h = 2166136261u;
    for (unsigned int i = 0; i < len; ++i) {
      h ^= (unsigned char)w[i]; h *= 16777619u;
    }
    return h;
  }
  static unsigned int HashCode(unsigned int c) {
    unsigned int h = c * 2654435761u;
    return h ^ (h >> 15);
  }

  char* blob_ = nullptr; unsigned int blob_len_ = 0, blob_cap_ = 0;
  unsigned int* off_ = nullptr; unsigned int* len_ = nullptr;
  unsigned int* code_ = nullptr; unsigned int n_ = 0, cap_ = 0;
  unsigned int* tab_ = nullptr; unsigned int* rtab_ = nullptr;
  unsigned int mask_ = 0;
};

class Dictionary {
 public:
  Dictionary(FILE* dictionary, bool encode, bool decode);
  void Encode(FILE* input, int len, FILE* output);
  void EncodeString(const std::string& input, std::string* output);
  unsigned char Decode(FILE* input);

 private:
  void EncodeWord(const std::string& word, int num_upper, bool next_lower,
      FILE* output);
  void EncodeWordString(const std::string& word, int num_upper,
      bool next_lower, std::string* output);
  bool EncodeSubstring(const std::string& word, FILE* output);
  bool EncodeSubstringString(const std::string& word, std::string* output);
  void AddToBuffer(FILE* input);

  FastWordMap word_map_;
  std::list<unsigned char> output_buffer_;
  bool decode_upper_ = false, decode_capital_ = false;
  unsigned int longest_word_ = 0;
};

}

#endif
