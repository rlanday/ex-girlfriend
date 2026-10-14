#ifndef COLD_SIZE_H_
#define COLD_SIZE_H_

#if defined(__clang__)
#define COLD_SIZE_ATTR __attribute__((minsize))
#else
#define COLD_SIZE_ATTR
#endif

#endif
