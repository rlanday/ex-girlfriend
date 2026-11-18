# `bitl/`: the BITLSTM weights of this entry

BITLSTM corrects the final probability of every bit in logit space (top-level
`README.md`). The program carries three frozen weight sets and switches between
them at fixed places of the stream; it has no backward pass. This directory
holds the tools and the record of their training.

| file | what it is |
|---|---|
| `record/meta.nblb`, `text1.nblb`, `text2.nblb` | the three weight sets (NBLB version 2, bfloat16, in the 32-cell layout of the program, 22,426 B each; the metadata set uses 16 cells, the two text sets 32) |
| `record/SHA256SUMS` | their checksums |
| `record/training.txt` | zones, commands and results of the training |
| `record/replay_text1.txt`, `replay_text2.txt` | replay of the candidate weight sets of the two text zones (the shipped ones are `q1k4_v2` and `q2k3_v2`) |
| `trace/features_trace.txt` | build line of the development tree that wrote the trace of the shipped weights |
| `tools/` | store builders, GPU trainers, fine-tuning, rounding, widening, evaluation, the header writer |

## 1. The trace

The trace build `cmix_trace` (`../tf/1_dump/build_dump_tools.sh <entry
directory> <tools directory> trace`: the source of the entry with
`../tf/1_dump/trainmodes.patch`, `TRAIN_MODE` 5) runs `-e enwik9` with the
transformer weights of this entry and records the inputs of BITLSTM for every
bit it sees:

```
mkdir trace && cd trace
cp <dictionary> .dict; cp <order v4> .new_article_order    # as in ../tf/1_dump/README.md, section 1
cp <entry>/weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 .
CMIX_BITL_TRACE=$PWD/spur.bin CMIX_BITL_TRACE_AUX=$PWD/kanal.bin CMIX_BITL_TRACE_AUX_PROJ=32 \
  <tools>/cmix_trace -e <enwik9> out
```

A record holds the 39 columns of the stage-1 mixer row (22 inputs of the first
mixer stage; 13 columns from the transformer and the bit position, namely the
fast-weight logit, the position of the bit in its byte, the state of the
transformer's prediction for the current byte in six classes and its
confidence in four classes, and the next bit of its most probable byte that is
still possible; 4 columns of SEMCACHE), the probability in front of BITLSTM,
the raw output of the final mixer, the override flag and the coded bit. The
channel file holds, once per eight such bits, four values and a 32-value
sketch of the transformer state and the gate zone (`CMIX_BITL_TRACE_AUX_PROJ`
projects the sketch to 32 values). The trace of the shipped weights has
1,694,239,343 records (277,855,252,272 B). It was written by a build of the
development tree with `trace/features_trace.txt`, whose BITLSTM is off in the
metadata blocks, and an addition to that build that records the bits there all
the same, with the same transformer weights in container format NP; the
shipped container (format NP2) stores some groups of neurons in another order,
which changes the transformer output only at the rounding level. With BITLSTM
running in the metadata blocks (`BITL_META_OFF=0`, as in `cmix_trace`) the
records are the same (37,421,056 records compared, about 3 % of the stream).
With the NP2 container, `cmix_trace` and the build of the development tree
with `BITL_META_OFF=0` write the same trace (`../tf/1_dump/README.md`,
section 6).

BITLSTM never feeds the predictor, so one trace holds the inputs for any of
its weights.

## 2. Zones

Record index = head bit, from 0. The compressor switches the weight sets at
boundaries of 1,024-bit windows and starts each set with zero state and the
previous bit 0.5; the trainer does the same.

| zone | records | content |
|---|---|---|
| metablock | [0, 30,990,992) | the metadata block and the census metadata block |
| text 1 | [30,991,360, 1,048,449,024) | the text up to the calendar block (page 143,227 of the article order) |
| text 2 | [1,048,449,024, 1,694,239,343) | the rest of the text |

The trace carries no byte position, so the first record of page 143,227 was
bounded from marks taken during the trace run (`tools/tb_zone1.py`); the
switch of the compressor counts the page itself.

## 3. Training

All weight sets are trained in-sample on their zone, on a GPU, in PyTorch
(`tools/bg_kern.py` is the replica of the C++ model and its Adam step,
`cm_kern.py` and `tb_kern.py` add the input row of this entry and the packed
stores). One step is one Adam update over 256 windows of 1,024 bits; the
learning rate falls along a cosine per step.

| set | cells | epochs | learning rate | fine-tuning on rounded weights | mantissa bits |
|---|---:|---:|---|---|---:|
| metablock | 16, widened to 32 | 1,024 | 1e-2 to 1e-4 | 128 epochs at 1e-4 to 1e-6, best epoch 104 | 3 |
| text 1 | 32 | 256 | 3e-3 to 3e-5 | 16 epochs at 1e-4 to 1e-6 | 4 |
| text 2 | 32 | 256 | 3e-3 to 3e-5 | 16 epochs at 1e-4 to 1e-6 | 3 |

In the metadata zone, the input columns that are constant there and the
sketch of the transformer (which is gated off in that zone) are fixed to zero.
The fine-tuning trains on the weights rounded to bfloat16 with k mantissa bits
(straight-through) and keeps the epoch end with the lowest cost; `bz_nblb.py
v2` writes the rounded file. The metadata set, trained with 16 cells, is
widened to the 32 cells of the program with zero weights (`cm_weiten.py`); the
added cells stay zero and the set computes as with 16 cells.

Commands and seeds: `record/training.txt`.

## 4. Into the program

```
python3 tools/weights_header.py record/meta.nblb record/text1.nblb record/text2.nblb > bit_head_weights.h
```

writes `src/bit_head_weights.h` of the entry byte for byte (sha256
`fa71c04d…`). The header holds only the live values (cells and input columns
that carry a weight) as a compact table; the tool decodes it again with the
loader loop of `Head::LoadBlob` and stops on any difference.

## 5. Effect in the zones

The replay of the frozen weights on the trace gives about -18.0 KB in the
metadata zone, -323.9 KB in text 1 and -212.0 KB in text 2, in bytes against
the same zone without BITLSTM and with the weights not counted. The program carries the three sets as one
compact table of 49,158 B (24,579 live values).
