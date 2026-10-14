"""one driver for all three PoC training phases

Vladimir Ivanov's three recipes (training_recipes/train_6m.py, quantize.py,
cooldown.py of fx2-cmix-transformer, GPL-3) are the same 180
lines with four numbers changed.  This module is that file once; the phase
differences live in tft/config.py.

What it adds on top of his `train()`:

  * per-epoch checkpoints (his loop only saves at the very end, so a run that
    dies at epoch 40 of 48 would otherwise leave nothing),
  * a JSONL loss/throughput log per phase (his LoggerConfig already supports a
    file sink; the recipes only ever used wandb),
  * gradient accumulation derived from the GPU count so the optimizer batch
    stays at his 2**20 tokens on 1, 2, 4 or 8 GPUs.

Nothing in pysrc/ is modified: the checkpointing is a Logger subclass swapped
into pysrc.train before train() is called.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from math import cos, pi
from typing import Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import pysrc.train as pysrc_train
from pysrc.data import CollatedArticlesDataLoader
from pysrc.logger import Logger, LoggerConfig
from pysrc.model import Transformer
from pysrc.optimizer import AdamWConfig, MuonConfig, OptimizerConfig
from pysrc.train import TrainingConfig, train

from tft import config as tftcfg
from tft.config import Phase

# set by run_phase before train() is called; read by CheckpointingLogger
_CHECKPOINT: dict = {"path": None, "every_epochs": 1, "rank": 0, "raw_model": None}


class CheckpointingLogger(Logger):
    """his Logger plus a state_dict dump whenever the epoch counter rolls over"""

    def train_step(self, epoch, loss, n_tokens, learning_rates, last_step) -> None:
        previous_epoch = int(self._last_epoch)
        super().train_step(
            epoch=epoch,
            loss=loss,
            n_tokens=n_tokens,
            learning_rates=learning_rates,
            last_step=last_step,
        )
        if _CHECKPOINT["path"] is None or _CHECKPOINT["rank"] != 0:
            return
        rolled_over = int(epoch) != previous_epoch
        if not (rolled_over or last_step):
            return
        if int(epoch) % _CHECKPOINT["every_epochs"] != 0 and not last_step:
            return
        model = _CHECKPOINT["raw_model"]
        tmp = _CHECKPOINT["path"] + ".tmp"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, _CHECKPOINT["path"])
        # KINJ_OPTSTATE: save the Adam/Muon moments next to it, so that a
        # continued training does not start from zero.  Errors here must not
        # kill the run.
        opts = _CHECKPOINT.get("optimizers")
        if opts is not None:
            try:
                otmp = _CHECKPOINT["path"] + ".opt.tmp"
                torch.save(
                    {
                        "epoch": float(epoch),
                        "second_order": opts.second_order.state_dict(),
                        "adamw": opts.adamw.state_dict(),
                    },
                    otmp,
                )
                os.replace(otmp, _CHECKPOINT["path"] + ".opt")
            except Exception as fehler:  # noqa: BLE001
                print("context package: optimizer state not saved: %r" % (fehler,),
                      flush=True)


def cosine_multiplier_schedule(
    epochs: int,
    warmup_epochs: int,
    warmup_learning_rate_multiplier: float,
    final_learning_rate_multiplier: float,
) -> Callable[[float], float]:
    """his schedule, verbatim (cosine warmup down from a high multiplier, then
    a cosine decay to the final multiplier)"""

    def schedule(fraction_done: float) -> float:
        fraction_warmup = warmup_epochs / epochs
        if fraction_warmup > 0 and fraction_done < fraction_warmup:
            s = fraction_done / fraction_warmup
            t = (cos(s * pi) + 1) / 2
            return t * warmup_learning_rate_multiplier + (1 - t)
        denominator = 1 - fraction_warmup
        s = (fraction_done - fraction_warmup) / denominator if denominator > 0 else 1.0
        t = (cos(s * pi) + 1) / 2
        return t + (1 - t) * final_learning_rate_multiplier

    return schedule


def _plan_schedule(phase: str, schedule) -> Callable[[float], float]:
    """the learning-rate plan, also for a resumed run

    Without `TFT_SCHED_TOTAL_EPOCHS` this is his plan over the epochs of this
    run, the same call as in his recipes.

    With `TFT_SCHED_TOTAL_EPOCHS=<T>` and `TFT_SCHED_DONE_EPOCHS=<D>` the plan
    of the whole block is computed (T epochs, warm-up as for the whole block)
    and this run follows only its part of it: fraction 0..1 of this run maps
    to (D + fraction*(T-D))/T.  A run restarted after a lost host thereby
    continues the cosine where the lost run stood, instead of starting again
    at the warm-up multiplier.

    (German names: bruch = fraction, gesamt = total, grund = base plan,
    rest = remaining epochs, fortgesetzt = continued.)
    """
    total = int(os.environ.get("TFT_SCHED_TOTAL_EPOCHS", "0"))
    done = int(os.environ.get("TFT_SCHED_DONE_EPOCHS", "0"))
    if total <= 0:
        return cosine_multiplier_schedule(
            epochs=schedule.epochs,
            warmup_epochs=schedule.warmup_epochs,
            warmup_learning_rate_multiplier=schedule.warmup_learning_rate_multiplier,
            final_learning_rate_multiplier=schedule.final_learning_rate_multiplier,
        )
    assert 0 <= done < total, (
        "TFT_SCHED_DONE_EPOCHS %d must lie in [0, %d)" % (done, total))
    assert schedule.epochs <= total - done, (
        "this run wants %d epochs, the plan has only %d left"
        % (schedule.epochs, total - done))
    bruch = float(os.environ.get("TFT_WARMUP_FRACTION",
                                 tftcfg.WARMUP_FRACTION[phase]))
    warmup_gesamt = min(total, round(bruch * total))
    grund = cosine_multiplier_schedule(
        epochs=total,
        warmup_epochs=warmup_gesamt,
        warmup_learning_rate_multiplier=schedule.warmup_learning_rate_multiplier,
        final_learning_rate_multiplier=schedule.final_learning_rate_multiplier,
    )
    rest = total - done
    print("continued schedule: block of %d epochs (warm-up %d), here epochs "
          "%d..%d" % (total, warmup_gesamt, done, done + rest), flush=True)

    def fortgesetzt(fraction_done: float) -> float:
        return grund((done + fraction_done * rest) / total)

    return fortgesetzt


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    checkpoint: str
    epochs: int
    seconds: float
    world_size: int


def _log_dir() -> str:
    d = os.environ.get("TFT_LOG_DIR", "logs")
    os.makedirs(d, exist_ok=True)
    return d


def _unique_log_file(phase: str, geometry_name: str) -> str:
    tag = os.environ.get("TFT_RUN_TAG", time.strftime("%Y%m%d-%H%M%S"))
    return _log_dir() + "/" + geometry_name + "-" + phase + "-" + tag + ".jsonl"


def make_data_loader(
    epochs: int, device: torch.device, rank: int, world_size: int
) -> CollatedArticlesDataLoader:
    paths = tftcfg.data_paths()
    arguments = dict(
        tokens_filename=paths["tokens_filename"],
        priors_filename=paths["priors_filename"],
        article_boundaries_filename=paths["article_boundaries_filename"],
        # Depending on the data centre, RunPod mounts /workspace as a network
        # file system (MooseFS over FUSE).  Sequential reads are fine (about
        # 1.1 GB/s), but random reads of the 83 GB prior file starve the GPU
        # (measured: 25.1 instead of 5.6 min per epoch).  With TFT_MMAP=0 the
        # loader reads the file once, sequentially, into main memory; the data
        # are bit-identical, only the access path changes.
        mmap=os.environ.get("TFT_MMAP", "1") != "0",
        epochs=epochs,
        micro_batch_tokens=tftcfg.MICRO_BATCH_TOKENS,
        gradient_accumulation_steps=tftcfg.gradient_accumulation_steps(world_size),
        max_article_tokens=tftcfg.MAX_ARTICLE_TOKENS,
        max_merge_tokens=None,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    # Dictionary twin ("dictionary as article 0"): reduced
    # training weight for article 0.  Unset -> his loader, unchanged.
    if os.environ.get("D0_DICT_TOKENS"):
        from d0.dloader import Dict0DataLoader

        return Dict0DataLoader(**arguments)

    # context package (arm pk): the combined arm has a loader of its own;
    # the arms in zloader.py stay untouched.
    arm = os.environ.get("ZCTX_ARM", "none")
    # arm `pkd` reads its rows from the coder's dump instead of rebuilding them
    # from pk tables (the coder is the specification).  There,
    # `article_boundaries_filename` holds the document boundaries of the dump.
    if arm == "pkd":
        from zctx.zpdump import PaketDumpDataLoader

        dump_dir = os.environ["PK_DUMP_DIR"]
        arguments["article_boundaries_filename"] = os.path.join(
            dump_dir, "article-boundaries.int32")
        # The coder has already made the 2^17 piece cut; the loader must not
        # cut a second time.  It counts in stream positions, and a document
        # can span more than 2^17 of them when the cut falls on a gated
        # position (measured: 131,086 instead of 131,072).  A second cut would
        # create documents that do not exist in the dump.
        arguments["max_article_tokens"] = tftcfg.MICRO_BATCH_TOKENS
        return PaketDumpDataLoader(dump_dir=dump_dir, **arguments)

    if arm == "pk":
        from zctx.zploader import PaketDataLoader

        return PaketDataLoader(
            pk_dir=os.environ["PK_DIR"], **arguments
        )

    # ZCTX ablation twins.  Unset -> his loader, unchanged.
    if arm != "none":
        from zctx.zloader import ZDataLoader

        return ZDataLoader(
            arm=arm, z3_dir=os.environ.get("ZCTX_Z3_DIR") or None, **arguments
        )
    return CollatedArticlesDataLoader(**arguments)


def run_phase(phase: Phase, geometry_name: str | None = None) -> PhaseResult:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main_process = rank == 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # ZCTX ablation twins: with the same seed on both sides the two runs start
    # from the SAME weights (the Z1 projection is zero initialised), so the
    # measured difference cannot come from a different initialisation.  Unset
    # -> his recipe, which does not seed at all.
    zctx_seed = os.environ.get("ZCTX_SEED")
    if zctx_seed:
        torch.manual_seed(int(zctx_seed) + rank)
        if main_process:
            print("zctx: torch.manual_seed(%s + rank)" % zctx_seed, flush=True)

    g = tftcfg.geometry(geometry_name)
    schedule = tftcfg.poc_schedule(phase)
    cfg = tftcfg.make_config(
        phase,
        geometry_name=g.name,
        variable_sequence_length=phase != "fp32",
    )

    raw_model = Transformer(cfg).to(device)

    # ZCTX arm Z1: swap the token embedding for the one that also reads the
    # article ordinal and the stream position out of the packed input_tokens.
    # Must happen BEFORE load_state_dict so a Z1 checkpoint finds its slot.
    #
    # Build guard: every arm whose loader packs inputs that the plain model
    # cannot read (the unchanged token embedding would get the packed int64
    # values and fail in the first step with "index out of bounds:
    # 0 <= tmp4 < 208") is listed in this table; if its model change is
    # missing, the run stops before the first step.
    arm_env = os.environ.get("ZCTX_ARM", "none")
    if arm_env == "pkd":
        arm_env = "pk"          # NEIF-PAKET11: the same model change as `pk`
    needs_model = {"z1": "z1", "z1z3": "z1", "z4": "z4", "z4c": "z4", "g1": "gate",
                   "m1": "match", "pk": "paket", "pkd": "paket"}
    installed: set = set()

    if "z1" in arm_env:
        from zctx.zmodel import install_z1

        added = install_z1(raw_model, cfg)
        installed.add("z1")
        if main_process:
            print("zctx: Z1 installed, +%d projection parameters" % added, flush=True)

    # ZCTX arm Z4 "lastbytes": the same swap, but the embedding reads the X
    # preceding token identities out of the packed input.  Same placement rule.
    if arm_env in ("z4", "z4c"):
        from zctx.zmodel import install_z4

        z4_x = int(os.environ.get("ZCTX_LAST_X", "4"))
        z4_d = int(os.environ.get("ZCTX_LAST_DIM", "16"))
        added = install_z4(raw_model, cfg, x=z4_x, d_slot=z4_d)
        installed.add("z4")
        if main_process:
            print(
                "zctx: Z4 installed, X=%d d_slot=%d, +%d parameters"
                % (z4_x, z4_d, added),
                flush=True,
            )

    # Digit gating arm g1: sentinel rows above the vocabulary.
    if arm_env == "g1":
        from zctx.zgate import N_LENGTH_CLASSES
        from zctx.zmodel import install_gate

        added = install_gate(raw_model, cfg, N_LENGTH_CLASSES)
        installed.add("gate")
        if main_process:
            print("zgate: %d sentinel rows installed, +%d parameters"
                  % (N_LENGTH_CLASSES, added), flush=True)

    # Arm m1: the compressor's match model as a prior channel.  The loader
    # packs match byte and length class into `input_tokens`; without this
    # change the plain model reads them as a token index and fails at the
    # vocabulary bound (see the build guard above).
    if arm_env == "m1":
        from zctx.zmatch import install_match

        m1_d = int(os.environ.get("ZCTX_MATCH_DIM", "16"))
        added = install_match(raw_model, cfg, d_slot=m1_d)
        installed.add("match")
        if main_process:
            print("zmatch: M1 installed, d_slot=%d, +%d parameters"
                  % (m1_d, added), flush=True)

    # context package (arm pk): one embedding for sentinel rows, predecessor
    # channel and match channel at once; the three single changes exclude
    # each other, because each replaces `raw_model.embedding` as a whole.
    if arm_env == "pk":
        from zctx import zpaket
        from zctx.zpmodel import install_paket

        pk_x = int(os.environ.get("PK_LAST_X", str(zpaket.DEFAULT_X)))
        use_last = os.environ.get("PK_LAST", "1") != "0"
        use_match = os.environ.get("PK_MATCH", "1") != "0"
        added = install_paket(
            raw_model, cfg,
            x=pk_x if use_last else 0,
            d_slot=int(os.environ.get("PK_SLOT_DIM", "16")),
            use_last=use_last, use_match=use_match,
        )
        installed.add("paket")
        if main_process:
            print(
                "context package installed: X=%d last=%s match=%s sentinel=%d, +%d parameters"
                % (pk_x, use_last, use_match, zpaket.N_SENTINEL, added),
                flush=True,
            )

    need = needs_model.get(arm_env)
    assert need is None or need in installed, (
        "arm '%s' delivers inputs that the unchanged model cannot read, "
        "but its model change '%s' has not run; the run would die in the "
        "first step (device-side assert)." % (arm_env, need)
    )

    # ZCTX arm Z4 "lastbytes": the same swap, but the embedding reads the X
    # preceding token identities out of the packed input.  Same placement rule.
    if os.environ.get("ZCTX_ARM", "none") in ("z4", "z4c"):
        from zctx.zmodel import install_z4

        z4_x = int(os.environ.get("ZCTX_LAST_X", "4"))
        z4_d = int(os.environ.get("ZCTX_LAST_DIM", "16"))
        added = install_z4(raw_model, cfg, x=z4_x, d_slot=z4_d)
        if main_process:
            print(
                "zctx: Z4 installed, X=%d d_slot=%d, +%d parameters"
                % (z4_x, z4_d, added),
                flush=True,
            )

    # Warm start: a new training (new package, new set of documents) begins
    # from the weights of stage 1 instead of random numbers.  The epoch index
    # starts at 0; the weights are taken over shape-aware (X0ROW grows
    # embedding.last_table, and naively appending rows would shift slot 1;
    # see paket11_warmstart.py).
    load_from = {"bf16": None, "qat": tftcfg.model_path("bf16", g.name),
                 "fp32": tftcfg.model_path("qat", g.name)}[phase]
    warm = os.environ.get("TFT_WARMSTART")
    if warm and phase == "bf16":
        from zctx import zpaket
        import paket11_warmstart as p11w

        p11w.gewichte_laden(
            raw_model, warm,
            x=int(os.environ.get("PK_LAST_X", str(zpaket.DEFAULT_X))),
            v_plus=cfg.vocabulary_size + zpaket.N_SENTINEL,
            x0row=zpaket.X0ROW, device=device, laut=main_process)
        load_from = None
    if load_from is not None:
        # strict=False for the bf16 -> qat step: the quantized config adds the
        # Quantize.scale parameters that the bf16 checkpoint does not have
        raw_model.load_state_dict(
            torch.load(load_from, map_location=device), strict=phase == "fp32"
        )
        if main_process:
            print("loaded " + load_from, flush=True)

    model: DDP = DDP(raw_model)
    if phase == "bf16":
        # his comment in quantize.py/cooldown.py: do NOT compile the quantized
        # phases, compilation removes the bf16 round trip on the scales
        model = torch.compile(model)  # type: ignore[assignment]

    data = make_data_loader(schedule.epochs, device, rank, world_size)

    save_to = tftcfg.model_path(phase, g.name)
    os.makedirs("models", exist_ok=True)
    _CHECKPOINT.update(
        path=save_to,
        every_epochs=int(os.environ.get("TFT_CHECKPOINT_EVERY", "1")),
        rank=rank,
        raw_model=raw_model,
    )
    pysrc_train.Logger = CheckpointingLogger  # type: ignore[misc]

    logger_config = LoggerConfig(
        log_to_stdout=True,
        file=_unique_log_file(phase, g.name) if main_process else None,
        wandb_project=os.environ.get("TFT_WANDB_PROJECT") or None,
        wandb_run_name=g.name + "-" + phase,
    )

    training_config = TrainingConfig(
        initialize_scales_steps=schedule.initialize_scales_steps,
        initialize_weight_scales_first=True,
        optimizer_config=OptimizerConfig(
            learning_rate_multiplier_schedule=_plan_schedule(phase, schedule),
            second_order_config=MuonConfig(
                base_learning_rate=schedule.learning_rate,
                weight_decay=0.1,
                momentum=0.95,
                adjust_lr_fn="match_rms_adamw",
            ),
            adamw_config=AdamWConfig(
                base_learning_rate=schedule.learning_rate,
                weight_decay=0.01,
                betas=(0.95, 0.99),
            ),
        ),
        window_size_schedule=None,
        logger_config=logger_config,
    )

    if main_process:
        print(
            json.dumps(
                {
                    "phase": phase,
                    "geometry": g.name,
                    "vocabulary_size": cfg.vocabulary_size,
                    "n_layers": cfg.n_layers,
                    "d_model": cfg.d_model,
                    "epochs": schedule.epochs,
                    "warmup_epochs": schedule.warmup_epochs,
                    "learning_rate": schedule.learning_rate,
                    "world_size": world_size,
                    "micro_batch_tokens": tftcfg.MICRO_BATCH_TOKENS,
                    "gradient_accumulation_steps": tftcfg.gradient_accumulation_steps(
                        world_size
                    ),
                    "n_tokens": data.n_tokens,
                    "n_articles": data.n_articles,
                    "data_vocabulary_size": data.vocabulary_size,
                    "checkpoint": save_to,
                    "log": logger_config.file,
                }
            ),
            flush=True,
        )
        assert data.vocabulary_size == cfg.vocabulary_size, (
            "the priors file implies vocabulary "
            + str(data.vocabulary_size)
            + " but the model is built for "
            + str(cfg.vocabulary_size)
            + "; the prior width must equal the vocabulary"
        )

    if os.environ.get("KINJ_OPTSTATE", "0") != "0" and main_process:
        import pysrc.train as _pt

        _mk = _pt.make_optimizers

        def _pk_make_optimizers(_model, _cfg):
            opts = _mk(_model, _cfg)
            _CHECKPOINT["optimizers"] = opts
            print("context package: optimizer state is saved as well", flush=True)
            # take over the Muon/AdamW moments of stage 1.  A parameter whose
            # shape has changed (X0ROW) gets no old state; its moment would
            # refer to the old row layout.
            wopt = os.environ.get("TFT_WARMSTART_OPT")
            if wopt:
                import paket11_warmstart as p11w

                p11w.optzustand_laden(opts, wopt, laut=True)
            return opts

        _pt.make_optimizers = _pk_make_optimizers

    started = time.time()
    train(raw_model=raw_model, model=model, data=data, cfg=training_config)
    seconds = time.time() - started

    if main_process:
        torch.save(raw_model.state_dict(), save_to)
        print(
            "phase "
            + phase
            + " done in "
            + format(seconds / 3600, ".2f")
            + " h -> "
            + save_to,
            flush=True,
        )

    dist.destroy_process_group()
    return PhaseResult(
        phase=phase,
        checkpoint=save_to,
        epochs=schedule.epochs,
        seconds=seconds,
        world_size=world_size,
    )
