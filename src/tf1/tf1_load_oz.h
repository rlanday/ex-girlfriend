#ifndef TF1_LOAD_OZ_H_
#define TF1_LOAD_OZ_H_

// Load and setup code of the transformer is compiled for size.

#if defined(__clang__)
#define LOAD_PATH_ATTR __attribute__((minsize))
#else
#define LOAD_PATH_ATTR
#endif

#endif
