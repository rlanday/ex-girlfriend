# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/fastmuon.py, sha256 f3b8ef441c2a88b3), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""v10 batched Muon: same algorithm as torch.optim.Muon (torch 2.12.1 _single_tensor_muon), but all matrices of one
shape go through momentum, Newton-Schulz and the update as ONE stacked batch (bmm/baddbmm, foreach), instead of
~20 small kernels per matrix. torch Muon is launch-bound here (~150 matrices of 64..1024 x 192), the GPU idles.

State layout is unchanged: state[p]["momentum_buffer"] is a per-parameter tensor (a view into a per-group stack),
so opt.pt saved/loaded by torch.optim.Muon and by this class are interchangeable (load_state_dict re-stacks).

    import fastmuon; fastmuon.install()      # before make_optimizers(): pysrc.optimizer uses FastMuon
"""
from __future__ import annotations

import torch
from torch.optim import Muon
from torch.optim._muon import _adjust_lr


class FastMuon(Muon):
    def _groups(self, group):
        """[(params, transposed, stacked_buf)] by shape; rebuilds the stacks when the state was replaced"""
        params = [p for p in group["params"] if p.grad is not None]
        key = tuple(id(p) for p in params)
        cache = group.get("_v10_cache")
        if cache is not None and cache[0] == key and all(
                self.state[p]["momentum_buffer"].data_ptr() == ptr for p, ptr in cache[2]):
            return cache[1]
        by_shape: dict = {}
        for p in params:
            by_shape.setdefault(tuple(p.shape), []).append(p)
        out, ptrs = [], []
        for shape, ps in by_shape.items():
            stack = torch.zeros((len(ps),) + shape, dtype=ps[0].dtype, device=ps[0].device)
            for i, p in enumerate(ps):
                st = self.state[p]
                if "momentum_buffer" in st:
                    stack[i].copy_(st["momentum_buffer"])
                st["momentum_buffer"] = stack[i]
                ptrs.append((p, stack[i].data_ptr()))
            out.append((ps, shape[0] > shape[1], stack))
        group["_v10_cache"] = (key, out, ptrs)
        return out

    def state_dict(self):
        sd = super().state_dict()
        for g in sd["param_groups"]:
            g.pop("_v10_cache", None)
        return sd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            a, b, c = group["ns_coefficients"]
            eps = group["eps"]
            for ps, transposed, buf in self._groups(group):
                grads = torch.stack([p.grad for p in ps])
                buf.lerp_(grads, 1 - mom)
                upd = grads.lerp(buf, mom) if group["nesterov"] else buf
                x = upd.bfloat16()
                if transposed:
                    x = x.transpose(1, 2)
                # per-matrix Frobenius norm, as ortho_grad.norm() on the bf16 matrix
                nrm = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True).clamp(min=eps)
                x = x / nrm
                for _ in range(group["ns_steps"]):
                    gm = torch.bmm(x, x.transpose(1, 2))
                    gu = torch.baddbmm(gm, gm, gm, beta=b, alpha=c)
                    x = torch.baddbmm(x, gu, x, beta=a)
                if transposed:
                    x = x.transpose(1, 2)
                adj = _adjust_lr(lr, group["adjust_lr_fn"], ps[0].shape)
                torch._foreach_mul_(ps, 1 - lr * wd)
                torch._foreach_add_(ps, list(x.unbind(0)), alpha=-adj)
        return loss


def install(verbose=True):
    import pysrc.optimizer as po
    po.Muon = FastMuon
    if verbose:
        print("fastmuon: pysrc.optimizer.Muon -> FastMuon (batched by shape)", flush=True)
