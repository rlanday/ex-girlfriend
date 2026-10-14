# Formats of the trace, the channel and the weight file

All values are little-endian.  "Model bit" means a coded bit on which the
compressor runs the BITLSTM.  There is none on bits fixed by the
alphabet or the grammar automaton, on bits the adaptive-depth stage skips, and
inside the metadata block.  The first record therefore belongs to the first bit
after the metadata block, at 0.93 % of the coder stream.

## Trace (`trace.bin`)

What the model receives from the coder, one record per model bit.

Header, 28 bytes in the trace of the shipped weights:

| offset | type | field |
|---:|---|---|
| 0 | u32 | magic `0x4c544942` (`BITL`) |
| 4 | u32 | version: 3 = a selection of stage-1 columns (mask below), 2 = all columns |
| 8 | u32 | n1, width of the stage-1 row: 28 |
| 12 | u32 | record length, 8 + 4 x recorded columns: 60 |
| 16 | u32 | number of recording windows: 0 (everything is recorded) |
| 20 | 16 x windows | per window u64 start, u64 end, in bytes of the coder stream |
| 20 + 16 x windows | u64 | version 3 only: column mask, bit j set = stage-1 column j recorded; `0x07e0007f` = 132,120,703, columns 0..6 and 21..26 |

Record, 60 bytes:

| offset | type | field |
|---:|---|---|
| 0 | u16 | base_p, the coder's discretized probability that the bit is 1, in 1/65536 |
| 2 | u8 | bit 0: the byte mixer forces the bit (override, no loss); bit 1: the coded bit |
| 3 | u8 | 0 |
| 4 | f32 | m1raw, the raw output of the final mixer |
| 8 | 13 x f32 | the recorded stage-1 columns, in column order |

The model drops stage-1 columns 7..20 and 27, so the 13 recorded columns are
exactly the ones it reads.  The replay rebuilds the row at its full width of 28
with zeros in the other columns.

## Channel (`channel.bin`)

The per-byte inputs of the model.  Header, 28 bytes, seven u32:

| field | value |
|---|---|
| magic | `0x315a4142` (`BAZ1`) |
| n_tf, transformer scalars per record | 4 |
| n_hid, values of the transformer-state block | 32 |
| the transformer-state block is a projection | 1 |
| n_lhid, values of the byte-LSTM-state block | 32 |
| the byte-LSTM-state block is a projection | 1 |
| a gate-zone value follows | 1 |

Record, 69 x f32 = 276 bytes, written at every model bit whose index (the count
of model bits before it) is a multiple of 8, so record k belongs to trace
records 8k .. 8k+7:

| values | content |
|---:|---|
| 4 | transformer scalars of the byte: gate bit (1 = the coded distribution is the transformer's own), entropy of its 208-way output / ln 208, largest probability, gap between the two largest (not read by this model) |
| 32 | projection of the transformer's 192-value state vector |
| 32 | projection of the byte LSTM's 81-value state vector (its 80 hidden values and a constant 1) |
| 1 | gate zone of the byte (3 = metadata block) |

The two projections are fixed.  Row r, column c is +1/sqrt(n) or -1/sqrt(n) (n =
192 or 81), the sign taken from consecutive draws of a splitmix64 generator,
row by row, seeded with `0x4849445052` for the transformer and `0x4c484450` for
the byte LSTM.  They do not depend on the model's seed or cell count, and the
trace build sums them in the same order as the compressor's model does.  In the
compressor the model computes the same projection itself, once every eight
model bits, from the vector registered at that bit.

A trace prefix of B model bits (B a multiple of 8) is the first 28 + 60 x B
bytes of the trace and the first 28 + 276 x B/8 bytes of the channel.

## Weight file (NBLB)

Header, 24 bytes: `NBLB`, u32 version (1 = f32 values, 2 = bf16 values), i32
cells 16, i32 nin 89 (inputs without the previous bit), i32 inw 96 (input row
padded to 8), i32 n1 28.  Then follow, in this order, in_proj weights 16 x 96,
in_proj bias 16, LSTM input weights 64 x 16, LSTM recurrent weights 64 x 16,
gate bias 64 (gate order input, forget, candidate, output), output weights 16
and output bias 1, together 3,681 values, of which the 96 padding columns of in_proj are zero (3,585
parameters).  Version 1 is 14,748 bytes, version 2 7,386 bytes.

Input row of the model (90 columns, padded with zeros to 96): the 13 stage-1
columns, m1raw, t/T, (t/T)^2 and sign(t) with t = logit(base_p) and T = 12.203,
a one-hot of the model-bit index modulo 8, the coded value of the previous model
bit, the 32 transformer-state values, the 32 byte-LSTM-state values.
