"""measure the three phases' throughput on THIS pod and size the night

Set-up of model, optimizer and data follows Vladimir Ivanov's
training_recipes/quantize.py and pysrc/train.py (fx2-cmix-transformer, GPL-3).

Nobody published tokens/s for this model, and the one number that can be
derived from the writeup (8x5090, 26 h, 464 epochs of ~601.5 M tokens =
~373 K tokens/s per GPU for the 6m) is an average over three phases with very
different costs.  So: measure, then plan.

The script runs a handful of real optimizer steps of each phase on the real
data package, reports tokens/s, and prints the TFT_EPOCHS_* environment that
fills a given wall clock budget while keeping HIS ratio between the phases
(320 : 128 : 16 epochs, the values his shipped recipes actually carry).

    uv run torchrun --nproc_per_node 1 -m tft.calibrate
    TFT_BUDGET_HOURS=10 TFT_CALIBRATION_STEPS=20 uv run torchrun ... -m tft.calibrate

Takes about ten minutes.  Run it once per geometry and data window; the
answer is stable.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pysrc.model import Transformer
from pysrc.optimizer import make_optimizers, OptimizerConfig, AdamWConfig, MuonConfig

from tft import config as tftcfg
from tft.runner import make_data_loader

# his shipped epoch counts: train_6m 256+64, quantize 128 (2026-08-21 tree,
# commit f1c95f3 "what we actually used"), cooldown 16
PHASE_EPOCH_RATIO = {"bf16": 320, "qat": 128, "fp32": 16}


def _optimizer_config(learning_rate: float) -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate_multiplier_schedule=lambda fraction_done: 1.0,
        second_order_config=MuonConfig(
            base_learning_rate=learning_rate,
            weight_decay=0.1,
            momentum=0.95,
            adjust_lr_fn="match_rms_adamw",
        ),
        adamw_config=AdamWConfig(
            base_learning_rate=learning_rate,
            weight_decay=0.01,
            betas=(0.95, 0.99),
        ),
    )


def measure_phase(phase: str, steps: int, warmup_steps: int, rank: int, world_size: int) -> dict:
    device = torch.device("cuda", rank)
    cfg = tftcfg.make_config(phase, variable_sequence_length=phase != "fp32")
    raw_model = Transformer(cfg).to(device)
    model: DDP = DDP(raw_model)
    if phase == "bf16":
        model = torch.compile(model)  # type: ignore[assignment]

    data = make_data_loader(1, device, rank, world_size)
    optimizers = make_optimizers(raw_model, _optimizer_config(1e-5))
    optimizers.update_learning_rate_schedule(fraction_done=0.0)

    tokens = 0
    started = None
    losses = []
    for index, batch in enumerate(data):
        if index == warmup_steps:
            torch.cuda.synchronize()
            started = time.perf_counter()
            tokens = 0
        for micro_batch in batch.micro_batches:
            loss = model(
                input_tokens=micro_batch.input_tokens,
                priors=micro_batch.priors,
                loss_mask=micro_batch.loss_mask,
                output_tokens=micro_batch.output_tokens,
                cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
            )
            if index >= warmup_steps:
                tokens += int(micro_batch.loss_mask.sum().item())
                losses.append(
                    (loss.detach() / micro_batch.loss_mask.long().sum()).item()
                )
            (loss / data.gradient_accumulation_steps).backward()
        optimizers.step()
        optimizers.zero_grad()
        if index + 1 >= warmup_steps + steps:
            break
    torch.cuda.synchronize()
    seconds = time.perf_counter() - (started or time.perf_counter())

    # scale to the whole job: every rank sees tokens/world_size of the stream
    tokens_total = tokens * world_size
    result = {
        "phase": phase,
        "steps": steps,
        "seconds": round(seconds, 3),
        "tokens_measured": tokens_total,
        "tokens_per_second": round(tokens_total / seconds, 1) if seconds > 0 else None,
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "peak_memory_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
    }
    del model, raw_model, optimizers, data
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return result


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    steps = int(os.environ.get("TFT_CALIBRATION_STEPS", "12"))
    warmup_steps = int(os.environ.get("TFT_CALIBRATION_WARMUP", "3"))
    budget_hours = float(os.environ.get("TFT_BUDGET_HOURS", "12"))

    data = make_data_loader(1, torch.device("cuda", rank), rank, world_size)
    tokens_per_epoch = data.n_tokens
    n_articles = data.n_articles
    vocabulary_size = data.vocabulary_size
    del data

    results = {}
    for phase in ("bf16", "qat", "fp32"):
        results[phase] = measure_phase(phase, steps, warmup_steps, rank, world_size)
        if rank == 0:
            print(json.dumps(results[phase]), flush=True)

    if rank != 0:
        dist.destroy_process_group()
        return

    # keep his epoch ratio, scale it to the budget
    seconds_per_epoch = {
        phase: tokens_per_epoch / results[phase]["tokens_per_second"]
        for phase in results
    }
    unit_seconds = sum(
        PHASE_EPOCH_RATIO[phase] * seconds_per_epoch[phase] for phase in results
    )
    scale = budget_hours * 3600 / unit_seconds
    epochs = {
        phase: max(1, round(scale * PHASE_EPOCH_RATIO[phase])) for phase in results
    }
    planned_hours = sum(epochs[p] * seconds_per_epoch[p] for p in epochs) / 3600

    summary = {
        "geometry": tftcfg.geometry().name,
        "world_size": world_size,
        "tokens_per_epoch": tokens_per_epoch,
        "n_articles": n_articles,
        "vocabulary_size": vocabulary_size,
        "seconds_per_epoch": {k: round(v, 1) for k, v in seconds_per_epoch.items()},
        "budget_hours": budget_hours,
        "epochs": epochs,
        "planned_hours": round(planned_hours, 2),
        "tokens_processed_total": sum(epochs[p] * tokens_per_epoch for p in epochs),
        "his_bf16_tokens_for_reference": "his short 4m schedule was 24 epochs of "
        "601.5M tokens = 14.4G tokens in the bf16 phase",
    }
    print(json.dumps(summary, indent=2), flush=True)
    print()
    print("# paste into the run:")
    print(
        "export TFT_EPOCHS_BF16="
        + str(epochs["bf16"])
        + " TFT_EPOCHS_QAT="
        + str(epochs["qat"])
        + " TFT_EPOCHS_FP32="
        + str(epochs["fp32"])
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
