/* x86p.h - linear x86-64 instruction length decoder used as a CONTEXT source
 * for the CM packer.  It only looks at bytes that are already decoded,
 * so encoder and decoder see the same state; no transform, no way back needed.
 * Mistakes (data in code, unknown opcodes) only cost compression, never
 * correctness: the parser resynchronises by itself.
 *
 * After x86_step(b) the fields describe the NEXT byte:
 *   xr_role  what the next byte is (XR_*)
 *   xr_idx   index inside a multi-byte field (disp/imm), 0-based
 *   xr_op    current opcode (0x000-0x0ff one byte, 0x100|op for 0F op,
 *            0x200|op for 0F38, 0x300|op for 0F3A); VEX maps the same way
 *   xr_mrm   ModRM byte of the current instruction (0 before it is seen)
 *   xr_pfx   prefix flags (bit0 66, bit1 F2, bit2 F3, bit3 REX.W, bit4 VEX, bit5 REX)
 */
#ifndef X86P_H
#define X86P_H
#include <stdint.h>

enum { XR_OP = 0, XR_OP2, XR_OP3, XR_VEX, XR_MRM, XR_SIB, XR_DISP, XR_IMM };

static int xr_role, xr_idx, xr_op, xr_mrm, xr_pfx;
static int xr_dlen, xr_ilen, xr_fsz;   /* pending disp/imm sizes, size of current field */
static int xr_vexn, xr_vmap;
static uint32_t xr_hist;
static int xr_lop, xr_lmrm, xr_sib, xr_ipos, xr_rex, xr_llmrm, xr_llop, xr_vexp;
static uint32_t xr_rh;                  /* register history: (R|reg)<<4 | (B|rm) per ModRM instruction */             /* previous instruction */               /* last opcodes (8 bit each, low = newest) */

/* one-byte opcode classes, 4 bits per opcode (generated from a table-building
 * routine that is not part of this package; layout): bit0..1 imm (0 none,
 * 1 ib, 2 iz, 3 special), bit2 ModRM, bit3 prefix (legacy/segment/REX) */
static const unsigned char xr_t1p[128] = {
  0x44,0x44,0x21,0x00,0x44,0x44,0x21,0x00,0x44,0x44,0x21,0x00,0x44,0x44,0x21,0x00,
  0x44,0x44,0x21,0x08,0x44,0x44,0x21,0x08,0x44,0x44,0x21,0x08,0x44,0x44,0x21,0x08,
  0x88,0x88,0x88,0x88,0x88,0x88,0x88,0x88,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
  0x00,0x44,0x88,0x88,0x62,0x51,0x00,0x00,0x11,0x11,0x11,0x11,0x11,0x11,0x11,0x11,
  0x65,0x55,0x44,0x44,0x44,0x44,0x44,0x44,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
  0x33,0x33,0x00,0x00,0x21,0x00,0x00,0x00,0x11,0x11,0x11,0x11,0x33,0x33,0x33,0x33,
  0x55,0x03,0x00,0x65,0x03,0x03,0x10,0x00,0x44,0x44,0x00,0x00,0x44,0x44,0x44,0x44,
  0x11,0x11,0x11,0x11,0x22,0x10,0x00,0x00,0x08,0x88,0x00,0x44,0x00,0x00,0x00,0x44,
};
#define XR_T1(b) ((xr_t1p[(b) >> 1] >> (((b) & 1) * 4)) & 15)

static void x86_init(void) {
  xr_role = XR_OP; xr_idx = 0; xr_op = 0; xr_mrm = 0; xr_pfx = 0;
  xr_dlen = xr_ilen = 0; xr_hist = 0; xr_lop = xr_lmrm = 0; xr_sib = 0; xr_ipos = 0; xr_rex = 0; xr_llmrm = xr_llop = 0; xr_vexp = 0; xr_rh = 0;
}

static int x86_iz(void) { return (xr_pfx & 1) ? 2 : 4; }

/* 0F-map (and VEX map 1): modrm? imm8? */
static int x86_0f_mrm(int o) {
  if (o == 0x05 || o == 0x06 || o == 0x07 || o == 0x08 || o == 0x09 || o == 0x0b || o == 0x0e) return 0;
  if (o >= 0x30 && o < 0x38) return 0;
  if (o == 0x77) return 0;
  if (o >= 0x80 && o < 0x90) return 0;
  if (o == 0xa0 || o == 0xa1 || o == 0xa2 || o == 0xa8 || o == 0xa9 || o == 0xaa) return 0;
  if (o >= 0xc8 && o < 0xd0) return 0;
  return 1;
}
static int x86_0f_imm(int o) {
  if (o >= 0x70 && o < 0x74) return 1;
  if (o == 0xa4 || o == 0xac || o == 0xba || o == 0xc2 || o == 0xc4 || o == 0xc5 || o == 0xc6) return 1;
  if (o >= 0x80 && o < 0x90) return 4;               /* jcc rel32 */
  return 0;
}

static void x86_end(void) {   /* instruction complete */
  if (xr_mrm) xr_rh = xr_rh << 8 | (uint32_t)(((xr_rex & 4) << 1 | (xr_mrm >> 3 & 7)) << 4 | ((xr_rex & 1) << 3 | (xr_mrm & 7)));
  xr_llop = xr_lop; xr_llmrm = xr_lmrm; xr_lop = xr_op; xr_lmrm = xr_mrm; xr_sib = 0; xr_ipos = -1; xr_rex = 0; xr_vexp = 0;
  xr_hist = (xr_hist << 8) | (uint32_t)(xr_op & 0xff) ^ (uint32_t)((xr_op >> 8) * 0x55);
  xr_role = XR_OP; xr_idx = 0; xr_pfx = 0; xr_mrm = 0; xr_op = 0;
}
static void x86_after_mrm_or_op(int has_mrm) {
  if (has_mrm) { xr_role = XR_MRM; xr_idx = 0; return; }
  if (xr_ilen) { xr_role = XR_IMM; xr_idx = 0; xr_fsz = xr_ilen; return; }
  x86_end();
}
static void x86_opdone(int op, int has_mrm, int imm) {
  xr_op = op; xr_ilen = imm; xr_dlen = 0;
  x86_after_mrm_or_op(has_mrm);
}

static void x86_step(int b) {
  xr_ipos++;
  switch (xr_role) {
  case XR_OP: {
    unsigned char t = (unsigned char)XR_T1(b);
    if (t & 8) {
      if (b == 0x66) xr_pfx |= 1; else if (b == 0xf2) xr_pfx |= 2; else if (b == 0xf3) xr_pfx |= 4;
      else if ((b & 0xf0) == 0x40) { xr_pfx |= 32; if (b & 8) xr_pfx |= 8; xr_rex = b; }
      xr_op = 0x400 | b;       /* marks "after prefix" */
      return;
    }
    if (b == 0x0f) { xr_role = XR_OP2; xr_op = 0x100; return; }
    if (b == 0xc4 || b == 0xc5) {
      xr_role = XR_VEX; xr_idx = 0; xr_vexn = (b == 0xc4) ? 2 : 1; xr_vmap = 1; xr_pfx |= 16; return;
    }
    {
      int imm = 0, mrm = (t >> 2) & 1;
      switch (t & 3) {
      case 1: imm = 1; break;
      case 2: imm = x86_iz(); break;
      case 3:
        if (b >= 0xb8) imm = (xr_pfx & 8) ? 8 : x86_iz();
        else if (b >= 0xa0 && b < 0xa4) imm = 8;
        else if (b == 0xc8) imm = 3;
        else imm = 2;
        break;
      }
      x86_opdone(b, mrm, imm);
    }
    return;
  }
  case XR_VEX:
    xr_vexp = xr_vexp << 8 | b;
    if (xr_vexn == 2 && xr_idx == 0) {
      xr_vmap = b & 31;
      xr_idx = 1; return;
    }
    if (xr_vexn == 2 && xr_idx == 1) { if (b & 0x80) xr_pfx |= 8; }
    /* last VEX payload byte: pp -> implied prefix */
    { int pp = b & 3; if (pp == 1) xr_pfx |= 1; else if (pp == 2) xr_pfx |= 4; else if (pp == 3) xr_pfx |= 2; }
    xr_role = (xr_vmap == 2 || xr_vmap == 3) ? XR_OP3 : XR_OP2;
    xr_op = xr_vmap == 2 ? 0x200 : xr_vmap == 3 ? 0x300 : 0x100;
    xr_idx = 0;
    return;
  case XR_OP2:
    if (!(xr_pfx & 16) && (b == 0x38 || b == 0x3a)) { xr_role = XR_OP3; xr_op = b == 0x38 ? 0x200 : 0x300; return; }
    x86_opdone(0x100 | b, x86_0f_mrm(b), x86_0f_imm(b));
    return;
  case XR_OP3:
    x86_opdone(xr_op | b, 1, (xr_op & 0x300) == 0x300 ? 1 : 0);
    return;
  case XR_MRM: {
    int mod = b >> 6, rm = b & 7, reg = (b >> 3) & 7;
    xr_mrm = b | 0x100;
    if ((xr_op == 0xf6 || xr_op == 0xf7) && reg < 2) xr_ilen = xr_op == 0xf6 ? 1 : x86_iz();
    xr_dlen = 0;
    if (mod == 1) xr_dlen = 1; else if (mod == 2) xr_dlen = 4; else if (mod == 0 && rm == 5) xr_dlen = 4;
    if (mod != 3 && rm == 4) { xr_role = XR_SIB; return; }
    if (mod == 3) xr_dlen = 0;
    goto fields;
  }
  case XR_SIB:
    xr_sib = b | 0x100;
    if ((b & 7) == 5 && (xr_mrm >> 6 & 3) == 0) xr_dlen = 4;
    goto fields;
  case XR_DISP:
    if (++xr_idx < xr_fsz) return;
    xr_dlen = 0;
    goto fields;
  case XR_IMM:
    if (++xr_idx < xr_fsz) return;
    x86_end();
    return;
  }
  return;
fields:
  if (xr_dlen) { xr_role = XR_DISP; xr_idx = 0; xr_fsz = xr_dlen; return; }
  if (xr_ilen) { xr_role = XR_IMM; xr_idx = 0; xr_fsz = xr_ilen; xr_ilen = 0; return; }
  x86_end();
}
#endif
