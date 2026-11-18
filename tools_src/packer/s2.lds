SECTIONS {
  . = 0x10000000;
  s2_base = .;
  .text : { *(.text.s2entry) *(.text .text.*) }
  s2_textend = .;
  .rodata : { *(.rodata .rodata.*) }
  .data : { *(.data .data.*) }
  s2_filend = .;
  .bss : { *(.bss .bss.*) *(COMMON) }
  s2_end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
