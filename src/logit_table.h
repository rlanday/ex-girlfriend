
#ifndef LOGIT_TABLE_H_
#define LOGIT_TABLE_H_

#include <stdint.h>
#include <string.h>
namespace logit_table {

enum { kEBase = 13184, kESize = 2945 };
inline unsigned EIndex(float x) {
  uint32_t u; memcpy(&u, &x, 4); u >>= 16;
  if (u <= (uint32_t)kEBase) return 0u;
  u -= (uint32_t)kEBase;
  if (u >= (uint32_t)kESize) return (unsigned)kESize - 1u;
  return (unsigned)u;
}
}

#endif
