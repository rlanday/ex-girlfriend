/* s1.lds - stage 1 of the packer stub: one PT_LOAD that maps the ELF and
 * program header too (no PT_GNU_STACK, no padding); the 24-byte header right
 * behind them (file offset 120), then the model table (.cutmodels), then the
 * code, then the stage-2 stream. */
PHDRS { load PT_LOAD FILEHDR PHDRS FLAGS(7); }
SECTIONS {
  . = 0x400000 + SIZEOF_HEADERS;
  .cuthdr : { KEEP(*(.cuthdr)) } :load
  .cutmodels : { KEEP(*(.cutmodels)) } :load
  .text : { *(.text .text.*) } :load
  s1_textend = .;
  .data : { KEEP(*(.data .data.*)) } :load
  .rodata : { *(.rodata .rodata.*) } :load
  .bss : { *(.bss .bss.*) *(COMMON) } :load
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
