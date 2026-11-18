# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/speedpatch.py, sha256 0fe5cc500bc721fa), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""speedpatch: bit-identical throughput patch for the bf16 yolo training (v3, Oct 14 2026).

1. The data loader keeps a pinned host copy of each micro-batch's cu_seqlens (side table keyed by the device pointer).
2. fla chunk_kda / causal_conv1d receive it as cu_seqlens_cpu, so prepare_chunk_indices runs on the host and the
   per-layer device->host syncs disappear; chunk_kda also gets disable_recompute=True (keeps forward intermediates
   for backward instead of recomputing them). Both are bitwise identical in outputs and gradients (bench_kda.py).
Enable with `import speedpatch; speedpatch.install()` BEFORE the model and data loader are built.
"""
import collections
import numpy as np
import torch
from torch import zeros, tensor

_CPU = collections.OrderedDict()   # device data_ptr -> pinned int64 host copy
_MAX = 256


def _remember(dev_t, host_t):
    p = dev_t.data_ptr()
    _CPU.pop(p, None)   # re-insert at the end: a reused (hot) address must not keep its old LRU position (v3 Oct 16)
    _CPU[p] = host_t
    while len(_CPU) > _MAX:
        _CPU.popitem(last=False)


def _lookup(ptr):
    h = _CPU.get(ptr)
    if h is not None:
        try:
            _CPU.move_to_end(ptr)   # an entry in use is the newest, never the next to evict (v3 Oct 16)
        except KeyError:        # a prefetching loader thread evicted it in between; h is still valid
            pass
    return h


def install(disable_recompute=True, verbose=True):
    import pysrc.data as pdata
    import pysrc.model as pmodel

    def _make_micro_batch(self, articles):
        # verbatim copy of pysrc.data.CollatedArticlesDataLoader._make_micro_batch plus the host-copy side table
        n_positions = self.micro_batch_tokens
        pin = self.device.type == "cuda"
        input_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        output_tokens = zeros((1, n_positions), dtype=torch.int64, pin_memory=pin)
        loss_mask = zeros((1, n_positions), dtype=torch.bool, pin_memory=pin)
        priors = zeros((1, n_positions, self._vocabulary_size), dtype=torch.float16, pin_memory=pin)
        boundaries = [0]
        position = 0
        for article in articles:
            start = int(self._article_starts[article])
            end = int(self._article_starts[article + 1])
            length = end - start - 1
            assert length > 0
            assert position + length <= n_positions
            input_tokens[0, position: position + length] = torch.from_numpy(self._tokens[start: end - 1].astype(np.int64))
            output_tokens[0, position: position + length] = torch.from_numpy(self._tokens[start + 1: end].astype(np.int64))
            loss_mask[0, position: position + length] = True
            priors[0, position: position + length] = torch.from_numpy(np.array(self._priors[start: end - 1]))
            position += length
            boundaries.append(position)
        if position < n_positions:
            boundaries.append(n_positions)
        cumulative_sequence_lengths = tensor([boundaries], dtype=torch.int32)
        dev_cu = cumulative_sequence_lengths.to(self.device)
        host = torch.tensor(boundaries, dtype=torch.int64)
        if pin:
            host = host.pin_memory()
        _remember(dev_cu, host)
        return pdata.MicroBatch(
            input_tokens=input_tokens.to(self.device, non_blocking=pin),
            output_tokens=output_tokens.to(self.device, non_blocking=pin),
            loss_mask=loss_mask.to(self.device, non_blocking=pin),
            priors=priors.to(self.device, non_blocking=pin),
            cumulative_sequence_lengths=dev_cu,
        )

    cls = pdata.CollatedArticlesDataLoader
    cls._make_micro_batch = _make_micro_batch

    orig_kda = pmodel.chunk_kda
    orig_conv = pmodel.causal_conv1d

    import os
    check = {"n": 0, "miss": 0, "on": os.environ.get("SPEEDPATCH_CHECK") == "1"}

    @torch._dynamo.disable
    def chunk_kda(*args, cu_seqlens=None, **kw):
        if cu_seqlens is not None:
            kw["cu_seqlens_cpu"] = _lookup(cu_seqlens.data_ptr())
            if check["on"]:
                check["n"] += 1
                if kw["cu_seqlens_cpu"] is None:
                    check["miss"] += 1
                elif check["n"] <= 50:
                    assert torch.equal(kw["cu_seqlens_cpu"], cu_seqlens.long().cpu()), "speedpatch: host cu_seqlens mismatch"
                if check["n"] in (12, 120, 1200):
                    print("speedpatch check: %d kda calls, %d without host cu_seqlens" % (check["n"], check["miss"]), flush=True)
        if disable_recompute and torch.is_grad_enabled():
            kw["disable_recompute"] = True
        return orig_kda(*args, cu_seqlens=cu_seqlens, **kw)

    @torch._dynamo.disable
    def causal_conv1d(*args, cu_seqlens=None, **kw):
        if cu_seqlens is not None:
            kw["cu_seqlens_cpu"] = _lookup(cu_seqlens.data_ptr())
        return orig_conv(*args, cu_seqlens=cu_seqlens, **kw)

    pmodel.chunk_kda = chunk_kda
    pmodel.causal_conv1d = causal_conv1d
    if verbose:
        print("speedpatch: host cu_seqlens for chunk_kda/causal_conv1d, disable_recompute=%s" % disable_recompute, flush=True)


def install_pkd(verbose=True):
    """Also feed host cu_seqlens from the pkd loader (zctx.zpdump.PaketDumpDataLoader), whose _make_micro_batch
    overrides the base one (v6 review, msg 678). Its only int32 tensor() call builds cu_seqlens from the host list
    'boundaries'; a thread-local hook keeps that host tensor and registers it under the device pointer."""
    import threading
    import zctx.zpdump as zp
    tl = threading.local()
    orig_tensor = zp.tensor

    def rec_tensor(data, *a, **kw):
        t = orig_tensor(data, *a, **kw)
        if kw.get("dtype") == torch.int32:
            tl.last = t
        return t

    zp.tensor = rec_tensor
    orig_mmb = zp.PaketDumpDataLoader._make_micro_batch

    def _make_micro_batch(self, articles):
        tl.last = None
        mb = orig_mmb(self, articles)
        host = getattr(tl, "last", None)
        if host is not None:
            h = host[0].to(torch.int64)
            _remember(mb.cumulative_sequence_lengths, h.pin_memory() if self.device.type == "cuda" else h)
        return mb

    zp.PaketDumpDataLoader._make_micro_batch = _make_micro_batch
    if verbose:
        print("speedpatch: pkd loader registers host cu_seqlens", flush=True)
