import json
from os.path import isfile
from pathlib import Path
from time import perf_counter
from dataclasses import dataclass, field
from types import ModuleType

import torch
import torch.distributed as dist

from pysrc.model import (
    Transformer,
    TransformerParameterCounts,
    transformer_parameter_counts,
)
from pysrc.optimizer import LearningRates


@dataclass(slots=True)
class Timer:
    _last_step_timestamp_seconds: float = field(default_factory=lambda: perf_counter())

    def step(self) -> float:
        second_to_last_step_timestamp = self._last_step_timestamp_seconds
        self._last_step_timestamp_seconds = perf_counter()
        return self._last_step_timestamp_seconds - second_to_last_step_timestamp


@dataclass(frozen=True, slots=True)
class LoggerConfig:
    log_to_stdout: bool = False
    file: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    running_average_momentum: float = 0.99


@dataclass(slots=True)
class Logger:
    model: Transformer
    cfg: LoggerConfig
    rank: int
    world_size: int
    _timer: Timer = field(default_factory=lambda: Timer())
    _wandb: ModuleType = field(init=False)
    _wandb_initialized: bool = False
    _parameter_counts: TransformerParameterCounts = field(init=False)
    _n_active_parameters: int = field(init=False)
    _device: torch.device = field(init=False)
    _loss_running_average: float | None = None
    _epoch_total_loss: float = 0.0
    _epoch_n_steps: int = 0
    _last_epoch: float = 0.0

    def __post_init__(self) -> None:
        assert self.world_size >= 1
        assert 0 <= self.rank < self.world_size

        # TODO: properly log the config and number of parameters to wandb, files, and stdout
        self._parameter_counts = transformer_parameter_counts(self.model)
        self._n_active_parameters = self._parameter_counts.active.n_parameters
        self._device = next(self.model.parameters()).device

        if not self._is_main:
            # everything below (printing, wandb, file logging) happens on rank
            # zero only; the other ranks still call the aggregating methods so
            # that their contribution reaches the collective, but never emit.
            return

        self._print_parameter_counts()

        if self.cfg.wandb_project is not None:
            import wandb  # type: ignore

            self._wandb = wandb

            self._wandb_initialized = True
            self._wandb.init(
                project=self.cfg.wandb_project, name=self.cfg.wandb_run_name
            )
            self._wandb.define_metric("epoch")
            self._wandb.define_metric("*", step_metric="epoch")

        if self.cfg.file is not None:
            assert not isfile(self.cfg.file), (
                "trying to log to a file that already exists"
            )
            Path(self.cfg.file).parent.mkdir(exist_ok=True, parents=True)

    @property
    def _is_main(self) -> bool:
        return self.rank == 0

    def _all_reduce(self, value: float, average: bool) -> float:
        """sum `value` across all ranks (dividing by world_size if `average`).

        every rank must call this with the same `average` so the underlying
        collective stays in lockstep."""
        if self.world_size == 1:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=self._device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= self.world_size
        return tensor.item()

    def _log(self, stats: dict[str, int | float]) -> None:
        if not self._is_main:
            return

        if self.cfg.log_to_stdout:
            print(
                " ".join(f"{key}: {value}" for key, value in stats.items()), flush=True
            )

        if self.cfg.file is not None:
            with open(self.cfg.file, "a") as f:
                f.write(json.dumps(stats) + "\n")

        if self.cfg.wandb_project:
            self._wandb.log(stats)

    def _update_loss_running_average(self, loss: float) -> float:
        if self._loss_running_average is None:
            self._loss_running_average = loss
            return loss

        m = self.cfg.running_average_momentum
        self._loss_running_average = m * self._loss_running_average + (1 - m) * loss
        return self._loss_running_average

    def _update_epoch_loss(
        self, epoch: float, loss: float, last_step: bool
    ) -> float | None:
        self._epoch_total_loss += loss
        self._epoch_n_steps += 1
        new_epoch: bool = int(self._last_epoch) != int(epoch) or last_step
        self._last_epoch = epoch
        if not new_epoch:
            return
        epoch_loss: float = self._epoch_total_loss / self._epoch_n_steps
        self._epoch_total_loss = 0.0
        self._epoch_n_steps = 0
        return epoch_loss

    def train_step(
        self,
        epoch: float,
        loss: float,
        n_tokens: int,
        learning_rates: LearningRates,
        last_step: bool,
    ) -> None:
        step_duration_seconds: float = self._timer.step()

        # aggregate across ranks: mean loss, total tokens (throughput sums).
        loss = self._all_reduce(loss, average=True)
        n_tokens = int(self._all_reduce(n_tokens, average=False))

        mfu = mfu_teraflops(
            n_active_parameters=self._n_active_parameters,
            n_tokens=n_tokens,
            training=True,
            duration_seconds=step_duration_seconds,
        )

        loss_running_average: float = self._update_loss_running_average(loss)

        log = {
            "epoch": epoch,
            "train/loss": loss,
            "train/loss_running_average": loss_running_average,
            "perf/train_mfu_teraflops": mfu,
            "optim/lr_multiplier": learning_rates.multiplier,
            "optim/second_order_lr": learning_rates.second_order_lr,
            "optim/adamw_lr": learning_rates.adamw_lr,
        }

        epoch_loss: float | None = self._update_epoch_loss(
            epoch=epoch, loss=loss, last_step=last_step
        )
        if epoch_loss is not None:
            log["train/epoch_loss"] = epoch_loss

        self._log(log)

    def test_step(self, epoch: float, loss: float, n_tokens: int) -> None:
        step_duration_seconds: float = self._timer.step()

        # aggregate across ranks: mean loss, total tokens (throughput sums).
        loss = self._all_reduce(loss, average=True)
        n_tokens = int(self._all_reduce(n_tokens, average=False))

        mfu = mfu_teraflops(
            n_active_parameters=self._n_active_parameters,
            n_tokens=n_tokens,
            training=False,
            duration_seconds=step_duration_seconds,
        )
        self._log({"epoch": epoch, "test/loss": loss, "perf/test_mfu_teraflops": mfu})

    def _print_parameter_counts(self) -> None:
        for param_group_name, counts in [
            ("total", self._parameter_counts.total),
            (
                "non embedding unembedding",
                self._parameter_counts.non_embedding_unembedding,
            ),
            ("active", self._parameter_counts.active),
        ]:
            for size_name, size in [
                ("parameters", counts.n_parameters),
                ("bytes", counts.bytes),
            ]:
                print(param_group_name, size_name, ":", size / 1e6, "million")
        print()


def mfu_teraflops(
    n_active_parameters: int, n_tokens: int, training: bool, duration_seconds: float
) -> float:
    flops_per_active_parameter_per_token: int = 6 if training else 2
    flops_per_token: int = flops_per_active_parameter_per_token * n_active_parameters
    flops: int = flops_per_token * n_tokens
    flops_per_second: float = flops / duration_seconds
    tera: float = 1e12
    return flops_per_second / tera
