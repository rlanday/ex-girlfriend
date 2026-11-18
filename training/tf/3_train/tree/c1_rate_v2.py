# cmix-neif-2.0: adapted from cmix-neif-beehive (Vladimir Malinovskii, 2026,
# github.com/galqiwi/cmix-neif-beehive, commit 92c07c0b, training/beehive/tree/c1_rate_v2.py, sha256 e5149103fd026981), GPL-3.
# Changes of cmix-neif-2.0 are marked '# 2.0'.
"""c1_rate_v2 = c1_rate_v1 + one numerical guard (log2 of an underflowed pool probability); see value_and_gradients."""
# SPDX-License-Identifier: GPL-3.0-only
"""C1 eager, first-order weight-entropy gradient hook.

Training experiment code. Rate is a fixed-topology symbol-entropy proxy, not
C4/container bytes. Explicit topology must include exported row/column masks.
No model forward, optimizer step, collective, or scale-gradient modification.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import torch

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class WeightSpec:
    name: str
    weight: torch.Tensor
    scale: torch.Tensor
    qmax: torch.Tensor
    pool_ids: torch.Tensor  # -1 omitted,0 shared INT4,1..5 id7 actual classes
    include_columns: torch.Tensor | None = None


class PooledSoftRate:
    def __init__(self, specs, *, tau=.25, chunk_elements=65536):
        self.specs = tuple(specs)
        if not self.specs or not math.isfinite(tau) or tau <= 0 or chunk_elements <= 0:
            raise ValueError('Nonempty specs, finite positive tau/chunk size required')
        self.tau, self.chunk_elements = float(tau), int(chunk_elements)
        device, dtype = self.specs[0].weight.device, self.specs[0].weight.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Initial helper requires FP32/FP64 latent weights')
        self.device, self.dtype = device, dtype
        qflat, pflat, self.layout = [], [], []
        offset = 0
        for spec in self.specs:
            w = spec.weight
            if w.ndim != 2 or w.device != device or w.dtype != dtype:
                raise ValueError('One device/dtype and2D matrices required')
            rows, cols = w.shape
            if spec.scale.numel() != rows:
                raise ValueError('One effective scale per matrix row required')
            q = spec.qmax.detach().cpu().reshape(-1)
            pools = spec.pool_ids.detach().cpu().to(torch.int64).reshape(-1)
            if q.numel() != rows or pools.numel() != rows:
                raise ValueError('qmax/pool label count must match rows')
            if not bool(((q == 1) | (q == 3) | (q == 7) | (q == 15) | (q == 31)).all()):
                raise ValueError('Unsupported mixed quantizer grid')
            if not bool(((pools >= -1) & (pools < 6)).all()):
                raise ValueError('Invalid coder-pool identifier')
            if bool(((pools == 0) & (q > 7)).any()):
                raise ValueError('INT4 topology with wider configured grid needs explicit treatment')
            columns = torch.ones(cols, dtype=torch.bool) if spec.include_columns is None else spec.include_columns.detach().cpu().bool().reshape(-1)
            if columns.numel() != cols:
                raise ValueError('Column mask must match matrix width')
            labels = pools[:, None].expand(rows, cols).clone()
            labels[:, ~columns] = -1
            qflat.append(q.to(torch.int8).repeat_interleave(cols))
            pflat.append(labels.reshape(-1).to(torch.int8))
            self.layout.append((offset, w.numel(), tuple(w.shape), bool((labels >= 0).any())))
            offset += w.numel()
        cpu_pools = torch.cat(pflat)
        self.pool_sizes = torch.tensor([(cpu_pools == p).sum().item() for p in range(6)], dtype=dtype, device=device)
        self.qmax = torch.cat(qflat).to(device)
        self.pool_ids = cpu_pools.to(device)
        self.centers = torch.arange(-31, 32, dtype=dtype, device=device)
        self.included_weights = int((cpu_pools >= 0).sum())
        self.total_weights = offset

    @torch.no_grad()
    def _flat(self):
        weights, scales = [], []
        for spec, (_, _, shape, _) in zip(self.specs, self.layout):
            if tuple(spec.weight.shape) != shape:
                raise ValueError('Model shape changed; rebuild topology/helper')
            weights.append(spec.weight.detach().reshape(-1))
            scales.append(spec.scale.detach().reshape(-1).to(torch.bfloat16).to(self.dtype).repeat_interleave(shape[1]))
        w, s = torch.cat(weights), torch.cat(scales)
        active = self.pool_ids >= 0
        w = torch.where(active, w, 0.)
        s = torch.where(active, s, 1.)
        valid = torch.isfinite(w).all() & torch.isfinite(s).all() & (s != 0).all()
        if not bool(valid):
            raise ValueError('Included weights/effective BF16 scales must be finite and scales nonzero')
        return w, s

    def _assignments(self, w, s, begin, end):
        labels = self.pool_ids[begin:end].long()
        active = labels >= 0
        x = w[begin:end] / s[begin:end]
        # Drop the common -x^2 term before softmax; its derivative cancels.
        logits = (2*x[:, None]*self.centers - self.centers.square()) / self.tau
        allowed = self.centers.abs()[None, :] <= self.qmax[begin:end, None]
        a = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
        return torch.where(active[:, None], a, 0.), labels

    @torch.no_grad()
    def value_and_gradients(self):
        w, scales = self._flat()
        counts = torch.zeros((6, 63), dtype=self.dtype, device=self.device)
        for begin in range(0, w.numel(), self.chunk_elements):
            end = min(begin+self.chunk_elements, w.numel())
            a, labels = self._assignments(w, scales, begin, end)
            # Fixed reductions, avoiding nondeterministic atomic index_add.
            for pool in range(6):
                counts[pool].add_((a*(labels == pool)[:, None]).sum(0))
        probabilities = counts / self.pool_sizes.clamp_min(1)[:, None]
        # v6 (2026-10-17, long-QAT C1-enable test): guard on probabilities, not counts. A positive count below
        # pool_size * 2^-149 (e.g. one denormal 1.4e-45 assignment mass in a 192-row pool) underflows to probability 0,
        # log2 -> -inf and rate -> inf. Identical to v1 whenever the probability is representable; the dropped terms are
        # count * log2(p) with count < 1e-40, i.e. zero contribution at fp32 precision.
        logp = torch.where(probabilities > 0, probabilities.log2(), 0.)
        rate = -(counts*logp).sum()
        gradient = torch.empty_like(w)
        for begin in range(0, w.numel(), self.chunk_elements):
            end = min(begin+self.chunk_elements, w.numel())
            a, labels = self._assignments(w, scales, begin, end)
            expectation = (a*self.centers).sum(-1, keepdim=True)
            lp = logp[labels.clamp_min(0)]
            gradient[begin:end] = (-2/self.tau)*(a*(self.centers-expectation)*lp).sum(-1)/scales[begin:end]
        if not bool(torch.isfinite(rate) & torch.isfinite(gradient).all()):
            raise FloatingPointError('Nonfinite C1 value/gradient; do not step optimizer')
        gradients = tuple(gradient[o:o+n].view(shape) for o, n, shape, _ in self.layout)
        return rate, gradients


def gradient_coefficient(lam, lambda_units, grad_normalizer=None):
    lam = float(lam)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be finite and nonnegative')
    if lambda_units == 'gradient_coefficient':
        return lam
    if lambda_units != 'mean_nats_per_weight_bit':
        raise ValueError('Declare lambda_units explicitly')
    if grad_normalizer is None or not math.isfinite(float(grad_normalizer)) or float(grad_normalizer) < 0:
        raise ValueError('Mean-nats price requires a common nonnegative grad_normalizer')
    return lam*float(grad_normalizer)


@torch.no_grad()
def snapshot_export_topology(raw_model, *, force_id7, zero_rows_log2, prune_mlp, clamp_sparse):
    """Snapshot the retained id7/structural-pruning rules, on CPU.

MLP deadness formula follows cmix-neif's GPL3 prune_dead_neurons.py.
This does not execute the exporter. A real export comparison remains required.
"""
    if clamp_sparse:
        raise ValueError('EXPORT_CLAMP_SPARSE is unsupported; supply a separately verified topology')
    if not math.isfinite(float(zero_rows_log2)):
        raise ValueError('A finite zero-row threshold is required')
    modules = dict(raw_model.named_modules())
    parameters = dict(raw_model.named_parameters())
    records = {}
    for name, module in modules.items():
        if not name.endswith('.quantize_weight') or not hasattr(module, 'quant') or module.batched:
            continue
        owner = name.removesuffix('.quantize_weight')
        w = parameters[owner+'.weight'].detach().float().cpu()
        scale = module.scale.detach().reshape(-1).bfloat16().float().cpu()
        qmax = getattr(module, 'qrow', torch.full((w.shape[0],), module.quant.qmax)).detach().float().cpu()
        if w.ndim != 2 or scale.numel() != w.shape[0] or qmax.numel() != w.shape[0]:
            raise ValueError('Unsupported export snapshot layout')
        if not bool(torch.isfinite(w).all() & torch.isfinite(scale).all()):
            raise ValueError('Nonfinite export snapshot')
        collapsed = (scale.abs() < 2.**float(zero_rows_log2)) if '.mlp.' not in name else torch.zeros_like(scale,dtype=torch.bool)
        if bool(((scale == 0) & ~collapsed).any()):
            raise ValueError('Unlifted zero MLP scale; initialize/lift before C1')
        safe = torch.where(scale != 0, scale, 1.)
        ints = (w/safe[:,None]).round()
        ints = torch.maximum(torch.minimum(ints,qmax[:,None]),-qmax[:,None]).long()
        ints[collapsed] = 0
        records[name] = {'ints':ints,'scale':scale,'rows':~collapsed,
                         'columns':torch.ones(w.shape[1],dtype=torch.bool)}
    if prune_mlp:
        for name, up in records.items():
            match = re.fullmatch(r'(.+)\.mlp\.up\.quantize_weight', name)
            if not match:
                continue
            prefix = match.group(1)+'.mlp.'
            down = records[prefix+'down.quantize_weight']
            sa = modules[prefix+'up.quantize_activation'].scale.detach().bfloat16().float().cpu().reshape(-1)
            sn = modules[prefix+'down.quantize_activation'].scale.detach().bfloat16().float().cpu().reshape(-1)
            if sa.numel()!=1 or sn.numel()!=1 or down['ints'].shape[1]!=up['ints'].shape[0]:
                raise ValueError('Unsupported MLP activation-scale/pruning layout')
            if not bool(torch.isfinite(sa).all() & torch.isfinite(sn).all()):
                raise ValueError('Nonfinite MLP activation scale')
            # Source folds in FP32, then multiplies by integer sums in FP64.
            q = up['ints']
            fold = (sa[0]*up['scale']).float()
            pos = q.clamp_min(0).sum(1); neg = (-q).clamp_min(0).sum(1)
            max_dot = 127*pos+128*neg; min_dot = -(128*pos+127*neg)
            max_y = torch.where(fold>=0,fold.double()*max_dot.double(),fold.double()*min_dot.double())
            keep = max_y >= math.sqrt(.5*sn[0].item()) if sn[0]>0 else torch.ones_like(fold,dtype=torch.bool)
            up['rows'] &= keep
            down['columns'] &= keep
    topology = {}
    for name, record in records.items():
        matrix = record['ints'][:,record['columns']]
        rows = record['rows']
        maxima = matrix.abs().amax(1) if matrix.shape[1] else torch.zeros(matrix.shape[0],dtype=torch.long)
        active = rows & (maxima>0)
        labels = torch.full_like(maxima,-1)
        use_id7 = bool(force_id7) or bool((maxima[rows]>7).any())
        if use_id7:
            # c0..c4 -> pool IDs1..5; INT4 is the separate pool0.
            labels[active] = 1+(maxima[active]>1).long()+(maxima[active]>3).long()+(maxima[active]>7).long()+(maxima[active]>15).long()
        else:
            labels[active] = 0
        topology[name] = {'pool_ids':labels.tolist(),'include_rows':rows.tolist(),
                          'include_columns':record['columns'].tolist()}
    return {'tensors':topology,'rules':{'force_id7':bool(force_id7),
            'zero_rows_log2':float(zero_rows_log2),'prune_mlp':bool(prune_mlp),
            'clamp_sparse':False},
            'scope':'Current raw checkpoint snapshot using retained export rules; actual C4 identity not verified.'}


def setup(raw_model, cfg):
    rc = cfg['rate_term']
    if int(rc.get('every', 1)) != 1:
        raise ValueError('Initial C1 prototype supports every=1 only')
    units = rc.get('lambda_units')
    if units not in ('mean_nats_per_weight_bit', 'gradient_coefficient'):
        raise ValueError('rate_term.lambda_units is required')
    topology = rc.get('topology')
    if rc.get('topology_mode') == 'snapshot_export_rules':
        rules = rc['export_rules']
        if set(rules) != {'force_id7','zero_rows_log2','prune_mlp','clamp_sparse'}:
            raise ValueError('Declare all retained export rules explicitly')
        topology = snapshot_export_topology(raw_model,**rules)
    elif topology is None:
        topology = json.loads(Path(rc['topology_json']).read_text())
    if 'tensors' in topology:
        topology = topology['tensors']
    topology = json.loads(json.dumps(topology))
    parameters, modules = dict(raw_model.named_parameters()), dict(raw_model.named_modules())
    specs = []
    for quantizer_name, rule in topology.items():
        quantizer = modules[quantizer_name]
        quant = quantizer.quant
        if quantizer.batched or quant.scale_dtype != torch.bfloat16 or quant.arithmetic_dtype != torch.float32 or quant.epsilon is not None:
            raise ValueError('Unsupported quantizer arithmetic/topology')
        weight_name = rule.get('weight_name')
        if weight_name is None:
            if not quantizer_name.endswith('.quantize_weight'):
                raise ValueError('Supply explicit weight_name for this quantizer')
            weight_name = quantizer_name.removesuffix('.quantize_weight')+'.weight'
        weight = parameters[weight_name]
        if quant.block_size != weight.shape[-1]:
            raise ValueError('Expected one weight block per row')
        qmax = getattr(quantizer, 'qrow', None)
        if qmax is None:
            qmax = torch.full((weight.shape[0],), quant.qmax)
        q_snapshot = qmax.detach().cpu().reshape(-1).tolist()
        if 'qmax_snapshot' in rule and rule['qmax_snapshot'] != q_snapshot:
            raise ValueError('Quantizer grid changed since the saved C1 topology')
        rule['qmax_snapshot'] = q_snapshot
        pools = torch.tensor(rule['pool_ids'], dtype=torch.int64)
        if 'include_rows' in rule:
            pools[~torch.tensor(rule['include_rows'], dtype=torch.bool)] = -1
        columns = torch.tensor(rule['include_columns'], dtype=torch.bool) if 'include_columns' in rule else None
        specs.append(WeightSpec(weight_name, weight, quantizer.scale, qmax, pools, columns))
    expected = {name for name, module in modules.items() if name.endswith('.quantize_weight') and hasattr(module, 'quant') and not module.batched}
    if set(topology) != expected:
        raise ValueError('Topology must cover every weight quantizer, including excluded rows')
    helper = PooledSoftRate(specs, tau=rc.get('tau', .25), chunk_elements=rc.get('chunk_elements', 65536))
    raw_model._v5_c1_rate = (helper, dict(rc))
    raw_model._v5_c1_topology = topology
    return {'tensors': len(specs), 'included_weights': helper.included_weights,
            'total_weights': helper.total_weights, 'lambda_units': units}


def state_dict(raw_model):
    helper, rc = raw_model._v5_c1_rate
    return {'format':1,'implementation_sha256':IMPLEMENTATION_SHA256,
            'topology':json.loads(json.dumps(raw_model._v5_c1_topology)),
            'tau':helper.tau,'chunk_elements':helper.chunk_elements,
            'lambda_units':rc['lambda_units'],
            'export_rules':json.loads(json.dumps(rc.get('export_rules')))}


def load_state_dict(raw_model, state, *, cfg=None):
    rc = cfg['rate_term'] if cfg is not None else raw_model._v5_c1_rate[1]
    if state['format'] != 1:
        raise ValueError('Unknown C1 state format')
    if state['implementation_sha256'] != IMPLEMENTATION_SHA256:
        raise ValueError('C1 implementation changed on resume; an explicit migration is required')
    for key, default in [('tau',.25),('chunk_elements',65536),('lambda_units',None)]:
        if rc.get(key,default) != state[key]:
            raise ValueError('C1 settings changed on resume: '+key)
    if rc.get('export_rules') != state['export_rules']:
        raise ValueError('C1 export rules changed on resume')
    config = dict(rc)
    config.pop('topology_mode',None)
    config.pop('topology_json',None)
    config['topology'] = state['topology']
    return setup(raw_model,{'rate_term':config})


@torch.no_grad()
def add_grad(raw_model, lam, step, *, grad_normalizer=None):
    helper, rc = raw_model._v5_c1_rate
    if float(lam) == 0:
        return {'applied': 0, 'gradient_coefficient': 0.}
    normalizer = grad_normalizer if grad_normalizer is not None else rc.get('grad_normalizer')
    coefficient = gradient_coefficient(lam, rc['lambda_units'], normalizer)
    if coefficient == 0:
        return {'applied': 0, 'gradient_coefficient': 0.}
    rate, gradients = helper.value_and_gradients()
    for spec, gradient, (_, _, _, participates) in zip(helper.specs, gradients, helper.layout):
        if not spec.weight.requires_grad or not participates:
            continue
        if spec.weight.grad is None:
            spec.weight.grad = gradient.mul(coefficient)
        else:
            spec.weight.grad.add_(gradient, alpha=coefficient)
    return {'applied': 1, 'rate_bits': rate.item(),
            'weight_symbol_entropy_proxy_B': rate.item()/8,
            'gradient_coefficient': coefficient,
            'included_weights': helper.included_weights}
