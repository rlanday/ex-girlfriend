#ifndef TF1_CONFIG_H_
#define TF1_CONFIG_H_

// V rounded up to a multiple of 16 (buffer width of prior/logits/probs) and
// to a multiple of 32 (quantized prior width, matmul group stride).

#if 16 && (defined(FX2_XCHECK_NAIVE_KDA) || defined(FX2_XCHECK_NAIVE_ATTN))
#error "the naive cross-check runs the row-by-row form only (no AttnRes block)."
#endif

#endif
