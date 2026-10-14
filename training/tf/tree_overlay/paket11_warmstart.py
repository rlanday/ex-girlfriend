#!/usr/bin/env python3
r"""Warm start and resume for stage 2

Two different things, kept apart on purpose:

**Warm start**: a new training (new package, new document set, new LR
schedule) starts not from random weights but from the weights of stage 1
(`poc4m-pk.tch`, bf16 epoch 14, plus `.opt`).  The data order is a different
one (different `n_articles`), and the epoch index starts at 0.

**Resume**: an interrupted training of the same configuration continues.  Then
the epoch index matters: `pysrc/data.py:140` creates all epoch permutations
when the loader is built, with `np.random.default_rng(seed=0)`; the permutation
of epoch k is the (k+1)-th draw from a fixed seed.  A restart at epoch E
therefore gets the same order as an uninterrupted run exactly when it skips
the first E epochs instead of drawing anew.  No torch RNG is needed for this:
`dropout = 0.0`, and there is no stochastic operator in the forward path.

Where a warm start silently goes wrong, and why this is a module of its own:

    `embedding.last_table.weight` is one matrix of `X * v_last` x `d_slot`,
    and slot k reads row `k*v_last + prev`.  With `X0ROW=1`, `v_last` grows
    from V+NSENT (216) to V+NSENT+1 (217).  Naively appending rows would shift
    all of slot 1 by one row: from slot 1 on the model would read the wrong
    row throughout, and nothing would fire.  Correct is a block-wise move per
    slot; the new row is initialised fresh.

Use as a tool (probe without GPU and without model):

    python3 warmstart.py probe <old.tch> [<old.tch.opt>] [--x0row]

(German names: Steckplatz = slot, umlagern = move, alt/neu = old/new,
frisch = fresh, verworfen = dropped.)
"""

from __future__ import annotations

import os
import sys

import torch

LAST = "embedding.last_table.weight"


def umlagern_last_table(alt: torch.Tensor, x: int, v_alt: int, v_neu: int,
                        streuung: float = 0.02,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """(X*v_alt, d) -> (X*v_neu, d), block-wise per slot

    The old rows of each slot move to their new address; the additional rows
    (v_neu - v_alt per slot) are drawn fresh.
    """
    assert alt.shape[0] == x * v_alt, (
        "last_table has %d rows, expected %d = %d slots x %d"
        % (alt.shape[0], x * v_alt, x, v_alt))
    assert v_neu >= v_alt
    d = alt.shape[1]
    neu = torch.empty((x * v_neu, d), dtype=alt.dtype)
    neu.normal_(mean=0.0, std=streuung, generator=generator)
    for k in range(x):
        neu[k * v_neu: k * v_neu + v_alt] = alt[k * v_alt: (k + 1) * v_alt]
    return neu


def gewichte_laden(model, pfad: str, *, x: int = 2, v_plus: int = 216,
                   x0row: bool = False, device="cpu", laut=True) -> dict:
    """stage-1 weights into a (possibly grown) model

    Returns a report with `uebernommen` (taken over), `umgelagert` (moved),
    `frisch` (fresh) and `verworfen` (dropped).
    """
    alt = torch.load(pfad, map_location=device)
    ziel = model.state_dict()
    v_neu = v_plus + (1 if x0row else 0)
    bericht = {"uebernommen": [], "umgelagert": [], "frisch": [], "verworfen": []}
    neu = {}
    for k, z in ziel.items():
        if k not in alt:
            bericht["frisch"].append(k)
            continue
        a = alt[k]
        if a.shape == z.shape:
            neu[k] = a
            bericht["uebernommen"].append(k)
        elif k == LAST and a.shape[0] == x * v_plus and z.shape[0] == x * v_neu:
            neu[k] = umlagern_last_table(a, x, v_plus, v_neu).to(z.dtype)
            bericht["umgelagert"].append(k)
        else:
            bericht["verworfen"].append("%s %s -> %s"
                                        % (k, tuple(a.shape), tuple(z.shape)))
    fehlt = [k for k in alt if k not in ziel]
    model.load_state_dict(neu, strict=False)
    if laut:
        print("warm start: %d taken over, %d moved (%s), %d fresh, "
              "%d dropped, %d unused in the checkpoint"
              % (len(bericht["uebernommen"]), len(bericht["umgelagert"]),
                 ",".join(bericht["umgelagert"]) or "-", len(bericht["frisch"]),
                 len(bericht["verworfen"]), len(fehlt)), flush=True)
        for z in bericht["verworfen"]:
            print("warm start: DROPPED %s" % z, flush=True)
        for z in bericht["frisch"]:
            print("warm start: freshly initialised %s" % z, flush=True)
    bericht["unbenutzt"] = fehlt
    return bericht


def optzustand_laden(optimizers, pfad: str, *, laut=True) -> dict:
    """Muon moments and AdamW states, with a shape check

    A parameter whose shape has changed (X0ROW: `last_table`) gets no old
    state, because its moment was computed on the old row layout.  It starts
    at 0, like a new parameter.
    """
    z = torch.load(pfad, map_location="cpu")
    bericht = {"epoch": float(z.get("epoch", 0.0)), "geladen": {}, "verworfen": {}}
    for name, opt in (("second_order", optimizers.second_order),
                      ("adamw", optimizers.adamw)):
        gespeichert = z.get(name)
        if gespeichert is None:
            continue
        jetzt = opt.state_dict()
        formen = {}
        i = 0
        for gruppe in opt.param_groups:
            for p in gruppe["params"]:
                formen[i] = tuple(p.shape)
                i += 1
        behalten, weg = {}, []
        for schluessel, zustand in gespeichert.get("state", {}).items():
            idx = int(schluessel)
            passt = True
            for feld, wert in zustand.items():
                if hasattr(wert, "shape") and wert.dim() > 0:
                    if tuple(wert.shape) != formen.get(idx):
                        passt = False
            if passt:
                behalten[schluessel] = zustand
            else:
                weg.append(idx)
        gespeichert = dict(gespeichert)
        gespeichert["state"] = behalten
        # the param_groups of the running optimizer win (new learning rate,
        # new schedule); only the state is taken over
        gespeichert["param_groups"] = jetzt["param_groups"]
        opt.load_state_dict(gespeichert)
        bericht["geladen"][name] = len(behalten)
        bericht["verworfen"][name] = weg
    if laut:
        print("warm start: optimizer from epoch %.1f; %s loaded, "
              "%s dropped because of a shape change"
              % (bericht["epoch"], bericht["geladen"], bericht["verworfen"]),
              flush=True)
    return bericht


def epochen_ueberspringen(data, e: int):
    """iterator that skips the first `e` epochs of the loader (resume)

    The loader draws its permutations in advance and independently of each
    other, so skipping is identical to running through, only without the
    compute time.
    """
    for batch in data:
        if batch.epoch < e:
            continue
        yield batch


# --------------------------------------------------------------------------
# probe without model and without GPU
# --------------------------------------------------------------------------

def probe(tch: str, opt: str | None, x0row: bool) -> int:
    sd = torch.load(tch, map_location="cpu")
    print("Checkpoint %s: %d tensors" % (os.path.basename(tch), len(sd)))
    a = sd[LAST]
    x, v_plus = 2, a.shape[0] // 2
    v_neu = v_plus + (1 if x0row else 0)
    print("%s %s -> %s (X=%d, v_last %d -> %d)"
          % (LAST, tuple(a.shape), (x * v_neu, a.shape[1]), x, v_plus, v_neu))
    g = torch.Generator().manual_seed(0)
    n = umlagern_last_table(a, x, v_plus, v_neu, generator=g)
    ok = True
    for k in range(x):
        gl = torch.equal(n[k * v_neu: k * v_neu + v_plus],
                         a[k * v_plus: (k + 1) * v_plus])
        print("  slot %d: old rows at the correct address: %s" % (k, gl))
        ok = ok and gl
    if x0row:
        for k in range(x):
            print("  slot %d: new edge-mark row %d, norm %.4f"
                  % (k, k * v_neu + v_plus,
                     float(n[k * v_neu + v_plus].norm())))
    # the naive form for comparison; it is exactly the silent error
    naiv = torch.cat([a, torch.zeros((x * (v_neu - v_plus), a.shape[1]))], 0)
    if x0row:
        gleich = torch.equal(naiv[v_neu: v_neu + v_plus],
                             a[v_plus: 2 * v_plus])
        print("  cross-check: the NAIVE version (appending) places slot 1 "
              "correctly: %s  <- must be False" % gleich)
        ok = ok and not gleich
    if opt:
        o = torch.load(opt, map_location="cpu")
        print("Optimizer state: epoch %.1f, second_order %d, adamw %d entries"
              % (float(o.get("epoch", -1)), len(o["second_order"]["state"]),
                 len(o["adamw"]["state"])))
        for name in ("second_order", "adamw"):
            treffer = [i for i, s in o[name]["state"].items()
                       for f, w in s.items()
                       if hasattr(w, "shape") and tuple(w.shape) == tuple(a.shape)]
            print("  %s: %d state fields with the last_table shape %s -> are "
                  "dropped with X0ROW" % (name, len(treffer), tuple(a.shape)))
    print("PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or args[0] != "probe":
        print(__doc__)
        sys.exit(2)
    sys.exit(probe(args[1], args[2] if len(args) > 2 else None,
                   "--x0row" in sys.argv))
