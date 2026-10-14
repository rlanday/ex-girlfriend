#ifndef CONTEXT_MANAGER_H
#define CONTEXT_MANAGER_H

#include "states/nonstationary.h"
#include "states/run-map.h"
#include "contexts/context.h"
#include "contexts/bit-context.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/combined-context.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/indirect-hash.h"
#include "contexts/sparse.h"
#include "page_contexts.h"
#include "lstm_sharpen.h"
#include "mixer_selectors.h"
#include "dense_index.h"
#include "transformer.h"
#include "transformer_gate_ctx.h"

#include <cmath>
#include "ds/SmallVector.h"

#include <cstdint>
#include <vector>

struct ContextManager {
  ContextManager();

  template<typename ... Args>
  const BracketContext& AddBracketContext(Args&& ...args) {
    BracketContext tmp(std::forward<Args>(args)...);
    for (const auto& old : bracket_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    bracket_contexts_.emplace_back(std::forward<Args>(args)...);
    return bracket_contexts_[bracket_contexts_.size() - 1];
  }

  template<typename ... Args>
  const ContextHash& AddContextHashContext(Args&& ...args) {
    ContextHash tmp(std::forward<Args>(args)...);
    for (const auto& old : context_hash_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    context_hash_contexts_.emplace_back(std::forward<Args>(args)...);
    return context_hash_contexts_[context_hash_contexts_.size() - 1];
  }

  template<typename ... Args>
  const Sparse& AddSparseContext(Args&& ...args) {
    Sparse tmp(std::forward<Args>(args)...);
    for (const auto& old : sparse_contexts_) {
      if (old.IsEqual(&tmp)) return old;
    }
    sparse_contexts_.emplace_back(std::forward<Args>(args)...);
    return sparse_contexts_[sparse_contexts_.size() - 1];
  }

  void UpdateContexts(int bit);
  void UpdatePage();
  void UpdateLx1();
  void UpdateTf1Gate();  // gate zone of the transformer as a mixer context
  void UpdateReload();
  void UpdateHistory();
  void UpdateWords();
  void UpdateRecentBytes();
  void UpdateWRTContext();
  void UpdateLineState(unsigned char c);

  unsigned int bit_context_ = 1, wrt_state_ = 0, bpos=0;
  unsigned long long long_bit_context_ = 1, zero_context_ = 0, history_pos_ = 0,
      line_break_ = 0, longest_match_ = 0, auxiliary_context_ = 0,
      wrt_context_ = 0,
      b2stream=0,b2streamcxt=0, o2bState=0, n2bState=0, stream2bR=0,
      b3stream=0,b3streamcxt=0,o3bState=0, n3bState=0, stream3bR=0,
      b4stream=0,
      mx5=0,mx6=0 ,mx7=0,mx8=0,
      mx9=0,mx9cxt=0,
      mx10=0,mx10cxt=0,
      mx11=0,mx11cxt=0,
      mx12=0,mx12cxt=0,
      mx13=0,mx13cxt=0,
      mx14=0,mx14cxt=0,
      mx15=0,mx15cxt=0,
      mx16=0,mx16cxt=0,
      mx17=0,mx17cxt=0,
      mx18=0,mx18cxt=0,
      mxx=0,
      words=0,wordscxt=0,
      ind1=0,context1_ind=0,
      ind2=0,context1_ind2=0,
      ind3=0,context1_ind3=0,
      ind4=0,context1_ind4=0,
      ind5=0,context1_ind5=0,
      mx19cxt=0;
  unsigned long long bd1_d0_ = 0, bd1_d1_ = 0, bd1_d2_ = 0;
  unsigned long long page_sur = 0, page_math = 0, page_rec = 0, page_link = 0;
  unsigned long long lx1_top1 = 0, lx1_conf = 0, lx1_agree = 0, lx1_ctl = 0;
  unsigned long long tf1_gate_ctx1 = 0;
  unsigned int line_class_ = 0, line_prefix_hash_ = 0;
  unsigned long long reload_line = 0, reload_lpr = 0, reload_aux = 0, reload_ppm = 0,
      reload_mtc = 0, reload_brk = 0, reload_ord = 0;
  std::vector<unsigned char> history_, shared_map_;
  std::vector<unsigned long long> words_, recent_bytes_;
  llvm::SmallVector<ContextHash, 12> context_hash_contexts_;
  llvm::SmallVector<Sparse, 18> sparse_contexts_;
  llvm::SmallVector<BracketContext, 1> bracket_contexts_;
  // Narrow element types for the hash vectors, after an idea by L. Lakhani
  // (fx3-cmix-llakhani, Run #5, GPL-3.0).  Lossless.
  std::vector<std::uint8_t>  hashes_ind1,hashes_ind4;
  std::vector<std::uint32_t> hashes_ind2,hashes_ind3,hashes_ind5;
  RunMap run_map_;
  Nonstationary nonstationary_;
};

#endif
