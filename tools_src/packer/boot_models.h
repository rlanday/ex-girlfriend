/* boot_models.h - model list of the bootstrap coder, found for the
 * stage-2 image by a greedy search that is not part of this package; it is
 * searched again whenever stage 2 changes (at most 9 models, stub_boot.c).  One
 * source of truth (the two lists); the offline tool reads boot_mask[]/boot_wt[],
 * the stub carries the same bytes in its model table (stub_boot.c).
 *   mask bit j (0..7) selects history byte -(j+1); wt = left shift of the
 *   model's counts in the mix. */
#ifndef BOOT_MODELS_H
#define BOOT_MODELS_H
#define BOOT_NM  9
#define BOOT_TB  22
#define BOOT_SHIFT 42          /* = 64 - BOOT_TB */
#define BOOT_TSZ 8388608       /* = 2 << BOOT_TB  (hash table bytes) */
#define BOOT_EPS 3
#define BOOT_CAP 255
#define BOOT_MASK_LIST 0x01,0x02,0x03,0x04,0x00,0x11,0x06,0x05,0x09
#define BOOT_WT_LIST      2,   1,   3,   0,   0,   3,   2,   3,   2
static const unsigned char boot_mask[BOOT_NM] = { BOOT_MASK_LIST };
static const unsigned char boot_wt[BOOT_NM]   = { BOOT_WT_LIST };
#endif
