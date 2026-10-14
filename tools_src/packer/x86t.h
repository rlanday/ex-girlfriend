/* x86t.h - exact x86-64 address transform.
 * Converts relative 32-bit fields to absolute virtual addresses:
 *   XT_RIP  disp32 of RIP-relative ModRM operands (mod=00, rm=101, no SIB)
 *   XT_REL  rel32 of E8/E9 and 0F 80..8F
 * Walks the code ranges with the length decoder of x86p.h.  The parse never
 * depends on disp/imm VALUES, so the same walk over the transformed bytes
 * finds exactly the same fields: the inverse is the same loop with -1.
 * Fields that would run past the range end are left alone. */
#ifndef X86T_H
#define X86T_H
#include "x86p.h"
#define XT_RIP 1
#define XT_REL 2
#define XT_JCC 4

static void x86t_range(unsigned char *b, long lo, long hi, long vadj, int what, int dir) {
  long i = lo;
  x86_init();
  while (i < hi) {
    int op, rip = 0, rel = 0; long f = -1, end;
    x86_step(b[i]); i++;
    if (xr_role == XR_DISP && xr_fsz == 4 && xr_idx == 0 && (xr_mrm & 0x1c7) == 0x105) rip = 1;
    if (xr_role == XR_IMM && xr_fsz == 4 && xr_idx == 0) {
      op = xr_op;
      if (op == 0xe8 || op == 0xe9) rel = 1;
      if (op >= 0x180 && op < 0x190) rel = 2;
    }
    if ((rip && (what & XT_RIP)) || (rel == 1 && (what & XT_REL)) || (rel == 2 && (what & XT_JCC))) {
      f = i;
      end = i + 4 + (rip ? xr_ilen : 0);
      if (end <= hi) {
        uint32_t v = (uint32_t)b[f] | (uint32_t)b[f + 1] << 8 | (uint32_t)b[f + 2] << 16 | (uint32_t)b[f + 3] << 24;
        uint32_t a = (uint32_t)(end + vadj);
        v = dir > 0 ? v + a : v - a;
        b[f] = (unsigned char)v; b[f + 1] = (unsigned char)(v >> 8);
        b[f + 2] = (unsigned char)(v >> 16); b[f + 3] = (unsigned char)(v >> 24);
      }
    }
  }
}
#endif
