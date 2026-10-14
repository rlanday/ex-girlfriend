SECTIONS {
  . = 0x10000000;
  .text : { *(.text.s2entry) *(.text .text.*) }
  .rodata : { *(.rodata .rodata.*) }
  .data : { *(.data .data.*) }
  s2_filend = .;
  .bss : { *(.bss .bss.*) *(COMMON) }
  s2_end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
