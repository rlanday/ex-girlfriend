#ifndef NO_ARGS_H_
#define NO_ARGS_H_
// The compressor takes no arguments.  Everything about the benchmark is fixed,
// so comp9 started without arguments compresses ./enwik9 into ./archive9,
// exactly what `-e enwik9 archive9` does; the argument file of the entry is
// empty (0 B in S).
//
// archive9 is started without arguments as well.  The program tells the two
// apart by its own image, not by a file: the argument vector below lies in the
// encoder-only section of the linker script, which comp9 carries.  For
// archive9 the -e path cuts the packed image after the decompressor part and
// rewrites the stub header (self_extract.h), and the stub fills the
// encoder-only section with zeros when it unpacks.  So the same byte reads '-'
// in comp9 and 0 in archive9; the read is volatile, otherwise the compiler
// would replace it by the initialiser.  The decompressor executes only that
// read and never touches the vector.  A build check stops the build if the
// vector lies outside the encoder-only section.
//
// Every other start keeps its arguments (the self-calls of self_extract.h and
// `-e enwik9 archive9`).  A build without the linker script has no
// encoder-only section; the vector is then ordinary read-only data and such a
// binary started without arguments is a compressor.

namespace no_args {
// argv of the compression run.  argv[0] is the empty string: nothing in the
// program reads it (the own path comes from /proc/self/exe).
struct ArgVector {
  char* argv[5];
  char text[19];
};
__attribute__((section(".rodata.no_args"), used))
static const ArgVector kArgVector = {
    {(char*)kArgVector.text + 2, (char*)kArgVector.text, (char*)kArgVector.text + 3,
     (char*)kArgVector.text + 10, nullptr},
    "-e\0enwik9\0archive9"};

// true in comp9, false in archive9 (zeros where the encoder block was).
inline bool IsCompressor() {
  return *(const volatile char*)kArgVector.text == '-';
}
}  // namespace no_args

#endif
