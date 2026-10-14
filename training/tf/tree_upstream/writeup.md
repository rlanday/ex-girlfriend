# Hutter Prize Submission

## TL;DR

This is a description of a submission for the Hutter Prize with a compression ratio of 9.95x. This is an 8.43% improvement over the the previous record cmix-lex whose compression ratio is 9.12x.

## Summary

The Hutter Prize is a competition for losslessly compressing the first 10^9 bytes of a snapshot of English Wikipedia (called enwik9). A previous record - fx2-cmix by Kaido Orav and Byron Knoll - is a mix of statistical models and hand-written heuristics. One of the models is an LSTM with 0.4 million parameters trained online during compression and decompression. It is responsible for about half of the wallclock runtime. We replace the LSTM with a transformer with 6 million parameters incorporatig modern improvements to the trasformer architecture. We train it on 8 RTX 5090 GPUs for about 30 hours. We append the weights to the compressor and decompressor binaries, so our solution runs on a CPU. We quantize the weights, thus, they only add 2.9MB to each binary (or 5.9MB to both). We quantize the activations in addition to the weights and use efficient CPU kernels for low-precision matrix multiplications, thus, we do not break the Hutter Prize's runtime limit of 50 hours on a single CPU core. We do not do any modifications to fx2-cmix other than replacing the LSTM with a transformer, fixing a bug, and minor changes that are only relevant for debugging.

Note that while previous records are general purpose compressors tuned for enwik9, ours is inherently specialized to enwik9 and will compress anything else badly unless one retrains the transformer.

## Context: fx2-cmix

Currently, the latest accepted record is fx2-cmix by Kaido Orav and Byron Knoll (another record by Kaido Orav, Byron Knoll, and Ibrahim Marcouch is currently pending approval). It compresses enwik9 the following way. First, it preprocesses it, doing things like reordering articles and replacing English words by their indices in a dictionary. Then, it uses a mix of statistical models and hand-written heuristics to predict, having seen a prefix of preprocessed enwik9, a probability distribution on the next bit. By Shannon's source coding theorem, writing a predictor like this with a low cross entropy loss is equivalent to writing a lossless compressor with a high compression ratio. Arithmetic coding is an algorithm that allows to turn one into the other, which is what fx2-cmix does.

One of the statistical models is an LSTM. It is trained online during compression and decompression. An advantage of training online is that otherwise, one would have to append the weights to the compressor and decompressor binaries, whose sizes the prize's rules penalize (the prize's target is miminizing the size of the compressor plus the size of the decompressor plus the size of the archive). A disadvantage of training online is that time spent training counts towards the compressor's and decompressor's runtimes (the prize improses a time a limit of 50 hours on a single CPU core and does not allow GPUs). Thus, compared to training offline, training online reduces the size of the biggest model usable within the time limits and makes it impossible to heavily overtrain the model.

The LSTM runs in float32 precision and the code it uses to do matrix multiplications is not heavily optimized.

## Transformer (Our Contribution)

### Loss

enwik9 consists of multiple wikipedia articles. We split the articles and treat them separately. We train an autoregressive transformer with a next token prediction cross entropy loss. That is, we train a transformer to predict, given a prefix of an article, a probability distribution on the next byte, and the loss is negative the log of the probability it assigns to the correct byte. This is the same loss as the LSTM except that the LSTM processes the whole enwik9 as one sequence without splitting articles.

fx2-cmix also has a PPMD model that predicts a probability distribution on the next byte by looking at all the previous occurreces in enwik9 of the few bytes that precede it and looking at what followed them. It feeds the PPMD's predicted probability distribution to the LSTM. We feed it to the tranformer in the same way.

We don't tokenize - we do 1 byte = 1 token.

### Architecture

We use our own transformer implementation inspired by [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt/). Here are all the major differences from the vanilla transformer architecture as found in e.g. the [LLaMa 1 paper](https://arxiv.org/abs/2302.13971) and the reasons we added them:
- 75% of layers use [Kimi Linear](https://arxiv.org/abs/2510.26692) instead of attention
    - Reasons:
        - It makes inference faster.
        - It reduces loss. We conjectured that it would because the Kimi Linear paper claimed that it did on much bigger models. We confirmed this by running small scale experiments. Furthermore, we tested 20%, 33%, 50%, 80%, and 100% instead of 75% and all led to a higher loss than with 75%.
- Sliding attention window with a length of 1024 tokens
    - Reasons:
        - It makes inference faster.
        - We tested 512 tokens instead of 1024, but it increased loss. 2048 tokens would have made inference too slow.
- [Muon optimizer](https://kellerjordan.github.io/posts/muon/) instead of AdamW
    - Reason: This is what Modded-NanoGPT uses and it worked better than AdamW in our preliminary experiments.
- [RoPE](https://arxiv.org/abs/2104.09864), [query-key normalization](https://arxiv.org/abs/2010.04245)
    - Reason: this is standard in modern transformers and is what Modded-NanoGPT uses.
- ReLU squared activation function in MLPs, embedding normalization, skip connections, token embedding connections, logit softcapping, Kaiming uniform weight initialization
    - Reason: this is what Modded-NanoGPT uses, but we didn't check if it actually helps and think these architectural changes are not important

### Training

We train in bfloat16 without quantization for 320 epochs. Then, we train in bfloat16 with quantization aware training for 128 epochs. Then, we train in float32 with quantization aware training for 16 epochs with a small learning rate. The final quantized model has a loss of 0.911 nats per byte, compared to the LSTM's 1.23 nats per byte.

The reason we use bfloat16 is that it speeds up training by about a factor of 2 and is standard practice in modern machine learning. The reason we do a small number of epochs in float32 at the end is that when implementing transformer inference in C++, we wanted the reference model to be in float32 because bfloat16's lower precision would have made it harder to tell whether a small difference in outputs between the two implementations is because of a bug or because of a difference in rounding.

### Quantization Aware Training

We use a modified version of [ParetoQ](https://arxiv.org/abs/2502.02631) to train with 4-bit quantized weights and 8-bit quantized activations. The quantized model's loss is 2.6% higher than the bfloat16 model's. We tried higher and lower quantizations but higher didn't decrease loss enough to justify the increased size of the weights and lower decreased the loss too much.

The things we do differently from ParetoQ are:
- We quantize both weights and activations while ParetoQ only quantizes weights.
    - Reason: This way, we can do most matrix multiplications in int8 during inference. If we didn't quantize activations, we would have had to do them in float32, which is much slower.
- When quantizing linear layers' weights, we use one bfloat16 scale per input channel, while ParetoQ uses one per output channel.
    - Reason: If we used one per output channel, we could not do linear layer matrix mulitplications in int8 during inference and would have had to do them in float32. For ParetoQ, it doesn't matter, since they don't quantize activations and thus can't do any matrix multiplications in int8 anyway.
- We quantize embeddings and unembeddings, which ParetoQ does in some experiments and doesn't in others.
    - Reason: Our experiments showed that while not quantizing embeddings and unembeddings reduces loss somewhat, it doesn't reduce it enough to offset by how much it increases the size of the weights. We find this surprising because:
        - Literature on LLM quantization, including ParetoQ, often doesn't quantize embeddings and unembeddings because this degrades loss too much.
        - Our model processes bytes and not tokens, so it has a small vocabulary size and therefore embedding and unembedding matrices are a saller fraction of the weights than in most LLMs and quantizing them does not decrease the size of the weights that much. Thus, one would expect that quantizing embeddings and unembeddings is a bad compromise between the loss and size of the weights in our case.
- We use the range -7, ..., 7 for weight quantization instead of the standard -8, ..., 7 that ParetoQ uses.
    - Reason:
        - The values representable by int4 are the integers -8, ..., 7. Usually, int4 quantization uses the full int4 range because it stands nothing to gain from excluding -8. Indeed, while excludig -8 would in theory reduce the number of bits per weight from 4 to `log2(15)=3.81`, encoding weights in a way that they actually take 3.81 bits of memory per weight would make doing computations with them prohibitively slow. Since the point of LLM quantization is reducing the amount of GPU RAM that weights take, not the amount of disk space they take, LLM quantization cannot encode encode weights in ways that make computations with them slow. However, we only care about the amount of disk space weights take and can decompress them once at startup and keep them in a format that takes more space during inference. Thus, we can (and do) store weights in a format where one weight actually takes 3.81 bits.
        - Literature on LLM quantization sometimes suggests that suggests that symmetric ranges are better than asymetric ranges. Thus, we choose -7, ..., 7 over -8, ..., 7 because it is symmetric. However, we haven't actually experimentally compared the two or looked into literature that rigorously compares symmetric and asymmetric ranges.

### Inference

We implement efficient C++ kernels using AVX2. Note that fx2-cmix also uses AVX2, so it doesn't decrease portability. However, AVX2 is more crucial to our submission than to fx2-cix.

AVX2 is an extension of the x86 instruction set. It was introduced in 2013 and most x86 CPUs released after ~2015-2017 support it. Both test machines in the [official Hutter Prize rules](http://prize.hutter1.net/hrules.htm) support it. It adds SIMD instructions, that is, instructions that do one operation many times in a single CPU cycle. The instructions it adds that matter most for us are `VPMADDUBSW`, which does 32 int8 multiplications in a single CPU cycle, `VPADDW`, which does 16 int16 additions, and `VPADDD`, which does 8 int32 additions.

## Other Modifications to fx2-cmix

### Bug Fix

We fixed a bug that was already present in fx2-cmix. It made the `RunContextMap` model non deterministic, thus either corrupting the decompressed file or crashing the decompressor. The bug didn't get triggered in fx2-cmix but did get triggered in our submission, presumably because of luck.

**Bug description:** In file `src/models/fxcmv1.cpp`, function `alloc1` allocates memory with an aligned start address, that is, a start address divisible by 16. It does this by allocating the requested number of bytes and rounding the pointer up to the next multiple of 16. The caller then assumes that it can use the requested number of bytes after the pointer, however, if rounding happened, the last few bytes are outside of the allocation, thus using them is undefined behavior.

**Fix:** We fixed the bug by allocating more bytes, so that the number of allocated bytes after the rounded pointer is equal to the number of requested bytes. Here is the diff of the fix:

```c++
--- a/src/models/fxcmv1.cpp
+++ b/src/models/fxcmv1.cpp  
 // for aligned data
+// `data` is `ptr` rounded up to `align`, so it can start up to align-1 bytes
+// into the block; the block therefore has to hold that much more than the c
+// elements the caller indexes, or the last elements fall outside it.
+// RunContextMap asks for exactly the m bytes it indexes, so without this
+// padding it reads and writes up to align-1 bytes past its 16MB allocation
+// whenever calloc does not happen to return an `align`-aligned pointer. What
+// those bytes alias depends on the process's heap layout, which differs
+// between the compressor and the decompressor (and even between two
+// decompressor runs whose temporary filename lands on the other side of
+// std::string's small-string threshold), so the two sides eventually disagree
+// on a prediction and the arithmetic decoder desynchronises.
 template <class T> void alloc1(T*&data, int c,T*&ptr,const int align=16) {
-  ptr=(T*)calloc(c, sizeof(T));
+  ptr=(T*)calloc(c + (align + sizeof(T) - 1) / sizeof(T), sizeof(T));
   if (!ptr) exit(1);
   data=(T*)(((uintptr_t)ptr+(align-1)) & ~(uintptr_t)(align-1));
 }
```