#ifndef ATTRIBUTES_H_
#define ATTRIBUTES_H_

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE_ATTR __attribute__((noinline))
#else
#define NOINLINE_ATTR
#endif

#if defined(__clang__)
#define GLUE_ATTR __attribute__((minsize))
#elif defined(__GNUC__)
#define GLUE_ATTR __attribute__((optimize("Os")))
#else
#define GLUE_ATTR
#endif

#endif
