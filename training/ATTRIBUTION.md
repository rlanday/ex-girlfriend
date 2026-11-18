# Attribution and licences

This `training/` directory is distributed under the same terms as the rest of
the package, **GPL-3** only, without the option of a later version (the licence section of `README.md`
of the entry gives the reasoning). The full licence text is in `LICENSE` at the
top level of the package. Exceptions are the third-party files in
`tf/tree_upstream/src/ds/`, the altered `avx_mathfun.h` routines in
`tf/tree_upstream/cpp_infer/` (`../third_party/README.md`) and the Wikipedia
excerpts in `tf/tree_upstream/prof_input/` (section 5).

The training-side Python in this directory imports and subclasses upstream
code directly, so `training/` is itself a **derivative work** of
fx2-cmix-transformer and, for the transformer of this version, of
cmix-neif-beehive. Where a file in this directory is a fork of a file of
another project, a provenance header or docstring says so.

The descent of the compressor itself, and the authors of its components, are
listed in `NOTICE.md` of the entry and in `training/ATTRIBUTION.md` of
cmix-neif-1.0 (`https://github.com/marwoelk/cmix-neif-1.0`).

---

## 1. The lineage of the training code

| component | author | licence | origin | where in this directory |
|---|---|---|---|---|
| **fx2-cmix-transformer**, the transformer, its trainer, data loader, optimizer, quantization, weight export, container formats v1 and v2 | **Vladimir Ivanov** (GitHub `astOwOlfo`) | GPL-3 | the 2026-07-24 release; public as `github.com/astOwOlfo/fx2-cmix-transformer-v1` (GPL-3.0); details in `training/ATTRIBUTION.md` of cmix-neif-1.0 | `tf/tree_upstream/`, verbatim, 189 files, `LICENSE` and every copyright header untouched |
| **cmix-neif-1.0**, the overlay of the trainer (training form, dump reader and loaders, geometry registry, runner), the coder-side dump, the package build and checks, the export tools and container formats v3 and v4 | Marcus Wölk | GPL-3 | `https://github.com/marwoelk/cmix-neif-1.0`, commit `1c9ea5ed` | `tf/tree_overlay/`, `tf/1_dump/`, `tf/2_package/`, and the base of `tf/3_train/export/export4/` |
| **cmix-neif-beehive**, the architecture of the transformer of this version, its training scripts and recipe, the changes to the export tools for the new tensors and the row encoding `ENC_INT5M`, and the trained weights the training of this version started from | **Vladimir Malinovskii** | GPL-3 | `https://github.com/galqiwi/cmix-neif-beehive`, commit `92c07c0b` (22 October 2026), a derivative of cmix-neif-1.0 | `tf/3_train/tree/` (13 files), `tf/3_train/export/export_v3_ar.py`, `tf/3_train/export/export4/` (`export.sh`, `wpack2_bf16.py`, `weights_compress_v3.py`, `prune_dead_neurons.py`, `convert_container_v4.py`); each file names its source file and its sha256 |

## 2. What is new in this directory, and its sources

| component | author | licence | where | origin and changes |
|---|---|---|---|---|
| the training scripts of the transformer | Vladimir Malinovskii; changes Marcus Wölk | GPL-3 | `tf/3_train/tree/` | from cmix-neif-beehive (section 1). The changes of this version are marked `# 2.0`. Every rank loads the optimizer state, and the ranks compare their parameters at fixed updates; a failure of the data loader is raised (`safe_prefetch.py`, new), and the steps of an epoch are counted against the plan; the sublayers are checkpointed in the quantization-aware and fp32 stages; `os._exit(0)` ends the process |
| the export of the new tensors and the row encoding `ENC_INT5M` | Vladimir Malinovskii; base Marcus Wölk | GPL-3 | `tf/3_train/export/` | from cmix-neif-beehive, which extended the export tools of cmix-neif-1.0; `export.sh` passes the container on when no neuron is dead (marked `# 2.0`) |
| container format v5 | Marcus Wölk | GPL-3 | `tf/3_train/export/export4/convert_container_v5.py`, `weights_compress_v3.py` | new in cmix-neif after 1.0; the range coder and the bit trees are Ivanov's (`pysrc/weights_compress.py`), whose range coder follows the form of the LZMA range coder of Igor Pavlov (Public Domain) |
| container formats NP and NP2 | Marcus Wölk | GPL-3 | `tf/3_train/export/np2/tf1_np_codec.cpp`, `np2_container.py` | new; the encoder of `tf1_np_codec.cpp` is a C++ transcription of the encoder of Ivanov's `pysrc/weights_compress.py`, `np2_container.py` is a Python transcription of `tf1_np_codec.cpp` on top of `weights_compress_v3.py`. They use two methods from the literature, the factorial number system of a permutation (Lehmer code; C.-A. Laisant, 1888) and the bits-back argument (G. E. Hinton and D. van Camp, 1993) |
| the run modes on the source of the entry | Marcus Wölk | GPL-3 | `tf/1_dump/trainmodes.patch`, `build_dump_tools.sh` | generated from the development tree of the compressor (the code of cmix-neif and the code it inherits from cmix, fx-cmix and cmix-lex, with their authors as in `../NOTICE.md`) by the generator of the source package |
| conversion of a container back into a training state, lifting of zero rows, the chain driver | Marcus Wölk | GPL-3 | `tf/3_train/container_to_tch.py`, `mlp_lift.py`, `reproduce.sh` | new; `container_to_tch.py` is the inverse of the export of cmix-neif-beehive |
| the prior check | Marcus Wölk | GPL-3 | `tf/2_package/prior0_check.py` | new |
| the GPU trainer of BITLSTM and its tools | Marcus Wölk; the model form after David Freelan (section 3) | GPL-3 | `bitl/tools/` | new: a PyTorch replica of the BITLSTM of the compressor and of its Adam step, stores of the trace, trainers, fine-tuning on rounded weights, rounding, widening, evaluation, the header writer |

### 2.1 Methods the training of this version uses

Cited as the sources of ideas; no code of these publications is copied.

* **Attention residuals** (each sublayer reads a learned softmax mixture of
  earlier outputs in place of the residual sum) after Kimi Team, *Attention
  Residuals*, arXiv 2603.15031 (2026), as implemented by Vladimir Malinovskii in
  cmix-neif-beehive; the fused kernel `fla.ops.attnres` of
  `flash-linear-attention` (MIT).
* **One key and value head, value embeddings, a prior head on the logits, MLP
  width**: options of the model of fx2-cmix-transformer
  (`pysrc/model.py`: `n_key_value_heads`, `value_embedding_pattern`,
  `prior_logit_mixing`, `d_mlp`), whose `writeup.md` names Modded-NanoGPT
  (Keller Jordan and others) as the inspiration of the implementation. One
  key and value head for all query heads is the multi-query attention of Noam
  Shazeer (2019). Value embeddings were introduced by Braden Koszarsky in the
  NanoGPT speedrun of Modded-NanoGPT (record of 4 December 2024), inspired by the
  value residual learning of Zhanchao Zhou et al. (2024).
* **Allocation of the bit width per weight row by Fisher information and a
  term for the coded size in quantization-aware training**: the implementation
  and the levels of 2, 3 and 4 bits of cmix-neif-beehive.
* The Muon optimizer (Keller Jordan and co-authors, 2024) for the weight
  matrices and AdamW for the other parameters, as in cmix-neif-1.0.

## 3. Harvested from a sibling GPL-3 fork

| component | author | licence | origin | where | kind |
|---|---|---|---|---|---|
| **BITLSTM, the LSTM correction model behind the mixer** (name and design from his `bitlstm32`): the model form (in-projection, SiLU, LSTM cell, linear output correcting in logit space) and the approach of training it offline on a trace of a compression run and shipping frozen weights | **David Freelan** | GPL-3 | `https://huggingface.co/dfreelan/cmix-obias`, revision `51488a0c1228dbeab7c1be837fc90ceaed351728` | the model of `bitl/tools/bg_kern.py` (a replica of the C++ model of the compressor) | **idea and form**; the trainers, the stores, the input row of this entry and the recipe are new |

## 4. Runtime dependencies of the trainer

These are **installed, not redistributed**; naming them is sufficient.

| library | version | used for | licence |
|---|---|---|---|
| `torch` | 2.12.1 (CUDA 13.0 build) | model, autograd, distributed training | BSD-3-Clause |
| `numpy` | 2.4.6 | container I/O, stores | BSD-3-Clause |
| `flash-linear-attention` (`fla`) | 0.5.1 | KDA kernels, the fused attention-residual kernel, fused gated RMS norm, causal convolution | MIT |
| `triton` | 3.7.1 | kernels of `fla` | MIT |
| `flash-attn` | 2.8.3 | optional attention path (guarded import) | BSD-3-Clause |
| `pytorch-optimizer`, `jaxtyping`, `tqdm`, `python-dotenv`, `wandb`, `einops`, `safetensors`, `transformers`, `huggingface-hub` | as pinned by the upstream lock file | as in cmix-neif-1.0 | as stated in their package metadata |

**No pre-trained third-party model and no third-party tokenizer is used in
the transformer chain.** The vocabulary is byte-based (208 values derived from
the preprocessing of cmix-neif); nothing is loaded from a model hub. The start
weights of section 5 are the weights of cmix-neif-beehive.

## 5. Data artefacts

| artefact | author | licence | note |
|---|---|---|---|
| the start weights, `weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4` of cmix-neif-beehive (not shipped here, sha256 `141ee13c…`) | Vladimir Malinovskii | GPL-3 | trained from a random initialisation on the training package of cmix-neif-1.0, which was built from `enwik9` alone |
| the transformer weights of this entry (`weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5`) | Marcus Wölk, trained from the start weights above | GPL-3 | produced by the pipeline documented here from `enwik9` |
| the BITLSTM weights of this entry (`bitl/record/`) | Marcus Wölk | GPL-3 | produced by the pipeline documented here from a trace of a compression run of `enwik9` |
| `tf/tree_upstream/prof_input/`, the PGO training text of upstream | **Wikipedia contributors** | **GFDL / CC-BY-SA, not GPL** | an excerpt of `enwik9`, inherited from upstream. The modification notice of cmix-neif does not extend to it |

## 6. Obligations under GPL-3, and how this package meets them

1. **One copyleft chain.** fx2-cmix-transformer, cmix-neif-1.0 and
   cmix-neif-beehive are GPL-3, and so is every file of this directory outside
   the exceptions named at the top.
2. **All upstream copyright and licence notices are left unchanged.** The
   upstream training tree is redistributed in `tf/tree_upstream/`, verbatim,
   with its own `LICENSE`; not one of its 189 files was edited. The files of
   cmix-neif sit beside it in `tf/tree_overlay/`, never merged into it.
3. **Modification notice** (GPL-3 section 5a): the files taken from
   cmix-neif-beehive name their source file and mark the changes of this
   version with `# 2.0`; `README.md` section 10 lists the changes against the
   training of cmix-neif-1.0.
4. **No copyright claim is made over inherited code.** The claim of
   Marcus Wölk (Marcus Woelk) covers the 2026 modifications only.

## 7. Open points

1. **The files of fx2-cmix-transformer name no author.** The attribution to
   Vladimir Ivanov rests on the Hutter Prize page, the LTCB entry and the GitHub
   account `astOwOlfo` that publishes the source (`training/ATTRIBUTION.md` of
   cmix-neif-1.0, section 7).
2. **The new files of cmix-neif-beehive carry no copyright header of their
   own.** Its repository, release tag and submission name Vladimir Malinovskii
   as the author; the provenance headers here name him.
3. **The Python packages the training uses are not shipped;** their licences
   are as stated in their package metadata.
