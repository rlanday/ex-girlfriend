# fx2-cmix-transformer

## TL;DR

This is a submission for the Hutter Prize. Our contribution is replacing the LSTM by a transformer in Kaido Orav and Byron Knoll's fx2-cmix. More details about improvements upon the previous submission in `writeup.md`.

## Compiling

1. Run `./setup.sh` once to install requirements.
2. If you need to comply with the Hutter Prize's 10GB RAM limit, replace `bool mmap_to_disk = false;` by `bool mmap_to_disk = false;` in file `src/models/ppmd.cpp`. This may crash on some machines and will write a lot to the disk, which may reduce SSD's lifespans. We recommend leaving it at `false`.
3. Run `./build_and_construct_comp.sh` to compile. It produces file `./run/cmix`, which is the enwik9-specialized compressor. Beware that it is different from `./cmix` or `./run/cmix_orig`.

## Compressing

Create an empty directory and place `enwik9` and the `./run/cmix` (not `./cmix` or `./run/cmix_orig`) compressor that compling produced in it. Cd into this directory and run:

```bash
./cmix -e enwik9 enwik9.comp
```

This produces a self-extracting archive `archive9`. Ignore all the other files that it produces, including `enwik9.comp`.

## Decompressing

Create an empty directory and place the `archive9` self-extracting archive that compression produced in it and run:

```bash
./archive9
```

This produces file `enwik9_uncompressed` identical to `enwik9`. Ignore all the other files that it produces.

## Training the Transformer

First, save the PPMD's outputs that the transformer needs by running the following. You need around 300GB of disk space.

```bash
./make_training_data.sh
```

Then [install uv](https://docs.astral.sh/uv/getting-started/installation/) and train the transformer by running the following replacing `$N_GPUS` by the number of GPUs on your machine. If the number of GPUs you have is not a power of two, you may need to round it down to a power of two. We tested on a machine with eight RTX 5090 GPUs.
```bash
uv run torchrun --nproc_per_node $N_GPUS -m training_recipes.train_6m
uv run torchrun --nproc_per_node $N_GPUS -m training_recipes.quantize
uv run torchrun --nproc_per_node $N_GPUS -m training_recipes.cooldown
```

If you have less than 8 GPUs, you have two options:
- Divide the batch size and and multiply the number of gradient accumulation steps by `N_GPUS / 8` in the three scripts. This will reproduce the behavior exactly.
- Don't change batch size and gradient accumulation steps. This may run faster but may also change training dynamics as it slightly increases the number of articles per optimizer step. This may also not be an option if you don't have enough GPU memory.

## Development Environment

Setting `bool mmap_to_disk = true;` in file `src/models/ppmd.cpp` makes the compressor and decompressor crash on some machines. We tested it on a google cloud machine with os image `ubuntu-2004-focal-v20240731`, on which it worked.
