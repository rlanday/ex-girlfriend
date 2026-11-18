#include "dictionary.h"

#include <stdio.h>
#include <string>

namespace preprocessor {

namespace {

const unsigned char kCapitalized = 0x40;
const unsigned char kUppercase = 0x07;
const unsigned char kEndUpper = 0x06;
const unsigned char kEscape = 0x0C;

void EncodeByte(unsigned char c, FILE* output) {
  if (c == kEndUpper || c == kEscape || c == kUppercase ||
      c == kCapitalized || c >= 0x80) {
    putc(kEscape, output);
  }
  putc(c, output);
}

void EncodeBytes(unsigned int bytes, FILE* output) {
  putc((unsigned char)bytes&0xFF, output);
  if (bytes & 0xFF00) {
    putc((unsigned char)((bytes&0xFF00)>>8), output);
  } else {
    return;
  }
  if (bytes & 0xFF0000) {
    putc((unsigned char)((bytes&0xFF0000)>>16), output);
  }
}

}

// See dictionary.h.
__attribute__((noinline)) unsigned char* CodeWord(unsigned r, unsigned char* o) {
  if (r < 80) { *o++ = (unsigned char)(0x80 + r); return o; }
  if (r < 3920) {
    r -= 80; *o++ = (unsigned char)(0xD0 + r / 80);
  } else if (r < 44880) {
    r -= 3920; *o++ = (unsigned char)(0xF0 + r / 2560);
    *o++ = (unsigned char)(0xD0 + (r / 80) % 32);
  } else {
    r -= 44880; *o++ = (unsigned char)(0xF0 + r / 1280);
    *o++ = (unsigned char)(0xF0 + (r / 80) % 16);
  }
  *o++ = (unsigned char)(0x80 + r % 80);
  return o;
}

Dictionary::Dictionary(FILE* dictionary, bool encode, bool decode) {
  fseek(dictionary, 0L, SEEK_END);
  unsigned long long len = ftell(dictionary);
  fseek(dictionary, 0L, SEEK_SET);
  word_map_.Reserve((unsigned int)(len / 2 + 64), (unsigned int)(len + 64));
  char line[256];
  char* p = line;
  int line_count = 0;
  const int kBoundary1 = 80, kBoundary2 = kBoundary1 + 3840,
      kBoundary3 = kBoundary2 + 40960;
  const int kBoundary4 = kBoundary3 + 20480;
  for (unsigned pos = 0; pos < len; ++pos) {
    unsigned char c = getc(dictionary);
    if (c >= 'a' && c <= 'z') {
      if (p == line + sizeof(line)) abort();
      *p++ = (char)c;
    } else if (p != line) {
      const unsigned int n = (unsigned int)(p - line);
      if (n > longest_word_) longest_word_ = n;
      unsigned int bytes;
      bytes = 0;
      if (line_count < kBoundary4) {   // bands 1-4: byte i of the code word in bits 8i
        unsigned char cw[4] = {0, 0, 0, 0};
        CodeWord((unsigned)line_count, cw);
        bytes = cw[0] | (cw[1] << 8) | (cw[2] << 16);
      }
      if (bytes == 0) { ++line_count; p = line; continue; }
      if (encode || decode) {
        word_map_.Add(line, n, bytes, encode, decode);
      }
      ++line_count;
      p = line;
    }
  }
}

void Dictionary::Encode(FILE* input, int len, FILE* output) {
  std::string word;
  int num_upper = 0, num_lower = 0;
  for (int pos = 0; pos < len; ++pos) {
    unsigned char c = getc(input);
    bool advance = false;
    if (word.size() > longest_word_) {
      advance = true;
    } else if (c >= 'a' && c <= 'z') {
      if (num_upper > 1) {
        advance = true;
      } else {
        ++num_lower;
        word += c;
      }
    } else if (c >= 'A' && c <= 'Z') {
      if (num_lower > 0) {
        advance = true;
      } else {
        ++num_upper;
        word += (c - 'A' + 'a');
      }
    } else {
      advance = true;
    }
    if (pos == len - 1 && !advance) {
      EncodeWord(word, num_upper, false, output);
    }
    if (advance) {
      if (word.empty()) {
        EncodeByte(c, output);
      } else {
        bool next_lower = (c >= 'a' && c <= 'z');
        EncodeWord(word, num_upper, next_lower, output);
        num_lower = 0;
        num_upper = 0;
        word.clear();
        if (next_lower) {
          ++num_lower;
          word += c;
        } else if (c >= 'A' && c <= 'Z') {
          ++num_upper;
          word += (c - 'A' + 'a');
        } else {
          EncodeByte(c, output);
        }
        if (pos == len - 1 && !word.empty()) {
          EncodeWord(word, num_upper, false, output);
        }
      }
    }
  }
}

void Dictionary::EncodeWord(const std::string& word, int num_upper,
    bool next_lower, FILE* output) {
  if (num_upper > 1) putc(kUppercase, output);
  else if (num_upper == 1) putc(kCapitalized, output);
  unsigned int fast_code;
  if (word_map_.Find(word.data(), (unsigned int)word.size(), &fast_code)) {
    EncodeBytes(fast_code, output);
  } else if (!EncodeSubstring(word, output)) {
    for (unsigned int i = 0; i < word.size(); ++i) {
      putc((unsigned char)word[i], output);
    }
  }
  if (num_upper > 1 && next_lower) {
    putc(kEndUpper, output);
  }
}

// The partial-word rule (dictionary.h, kWrtSubMin): prefix only, the code
// always starts the word.
bool Dictionary::EncodeSubstring(const std::string& word, FILE* output) {
  if (word.size() <= kWrtSubMin) return false;
  unsigned int size = word.size() - 1;
  if (size > longest_word_) size = longest_word_;
  for (; size >= kWrtSubMin; --size) {
    unsigned int fast_code;
    if (word_map_.Find(word.data(), size, &fast_code)) {
      EncodeBytes(fast_code, output);
      for (unsigned int i = size; i < word.size(); ++i) {
        putc((unsigned char)word[i], output);
      }
      return true;
    }
  }
  return false;
}

unsigned char NextChar(FILE* input) {
  int c = getc(input);
  if (c>='{' && c<127) c+='P'-'{';
  else if (c>='P' && c<'T') c-='P'-'{';
  else if ( (c>=':' && c<='?') || (c>='J' && c<='O') ) c^=0x70;
  if (c=='X' || c=='`') c^='X'^'`';
  return c;
}

// Returns the next decoded byte.  A dictionary code leaves its word in
// word_map_ (wort_, rest_); its letters are handed out one per call, with the
// case flags as they stood when the code was read (nothing changes them
// before the word is used up).
unsigned char Dictionary::Decode(FILE* input) {
  for (;;) {
    if (rest_) {
      unsigned char x = (unsigned char)*wort_++;
      --rest_;
      if (decode_capital_) { x = (x - 'a') + 'A'; decode_capital_ = false; }
      if (decode_upper_) x = (x - 'a') + 'A';
      return x;
    }
    unsigned char c = NextChar(input);
    if (c == kEscape) {
      decode_upper_ = false;
      return NextChar(input);
    } else if (c == kUppercase) {
      decode_upper_ = true;
    } else if (c == kCapitalized) {
      decode_capital_ = true;
    } else if (c == kEndUpper) {
      decode_upper_ = false;
    } else if (c >= 0x80) {
      unsigned int bytes = c;
      if (c > 0xCF) {
        c = NextChar(input);
        bytes += c << 8;
        if (c > 0xCF) {
          c = NextChar(input);
          bytes += c << 16;
        }
      }
      // an unknown code leaves rest_ at 0 (empty word)
      word_map_.FindCode(bytes, &wort_, &rest_);
    } else {
      if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
        decode_upper_ = false;
      }
      if (decode_capital_ || decode_upper_) {
        c = (c - 'a') + 'A';
      }
      decode_capital_ = false;
      return c;
    }
  }
}

}
