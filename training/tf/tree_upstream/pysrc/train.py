from torch import Tensor, inference_mode
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import trange, tqdm
from dataclasses import dataclass
from jaxtyping import Float, Int
from typing import Callable

from pysrc.model import Transformer
from pysrc.quantization import Quantize
from pysrc.data import CollatedArticlesDataLoader
from pysrc.model import InitializeScales
from pysrc.optimizer import make_optimizers, OptimizerConfig
from pysrc.logger import Logger, LoggerConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    initialize_scales_steps: int | None
    initialize_weight_scales_first: bool
    optimizer_config: OptimizerConfig
    window_size_schedule: Callable[[float], int | Int[Tensor, ""]] | None
    logger_config: LoggerConfig


def train(
    raw_model: Transformer,
    model: DDP | Transformer,
    data: CollatedArticlesDataLoader,
    cfg: TrainingConfig,
) -> None:
    model.train()
    optimizers = make_optimizers(raw_model, cfg.optimizer_config)

    initialize_scales(raw_model=raw_model, model=model, data=data, cfg=cfg)

    logger = Logger(
        model=raw_model,
        cfg=cfg.logger_config,
        rank=data.rank,
        world_size=data.world_size,
    )

    total_batches: int = (
        len(data)  # type: ignore[arg-type]
        if hasattr(data, "__len__")
        else data.approximate_length()  # type: ignore[union-attr]
    )
    for batch in tqdm(data, desc="training", total=total_batches):
        window_size: int | Int[Tensor, ""] | None = (
            cfg.window_size_schedule(batch.fraction_done)
            if cfg.window_size_schedule is not None
            else None
        )

        learning_rates = optimizers.update_learning_rate_schedule(
            fraction_done=batch.fraction_done
        )

        total_step_loss: Float[Tensor, ""] = 0.0  # type: ignore
        total_step_unmasked_tokens: Int[Tensor, ""] = 0  # type: ignore
        total_step_tokens: int = 0
        for micro_batch in batch.micro_batches:
            loss = model(
                input_tokens=micro_batch.input_tokens,
                priors=micro_batch.priors,
                loss_mask=micro_batch.loss_mask,
                output_tokens=micro_batch.output_tokens,
                cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
                override_window_size=window_size,
            )
            total_step_loss += loss.detach()
            total_step_unmasked_tokens += micro_batch.loss_mask.long().sum()
            loss = loss / data.gradient_accumulation_steps
            total_step_tokens += micro_batch.input_tokens.numel()
            loss.backward()

        optimizers.step()
        optimizers.zero_grad()

        logger.train_step(
            epoch=batch.epoch,
            loss=(total_step_loss / total_step_unmasked_tokens).item(),
            n_tokens=total_step_tokens,
            learning_rates=learning_rates,
            last_step=batch.is_last,
        )


@inference_mode()
def initialize_scales(
    raw_model: Transformer,
    model: DDP | Transformer,
    data: CollatedArticlesDataLoader,
    cfg: TrainingConfig,
) -> None:
    if cfg.initialize_scales_steps is None:
        return

    weight_steps: int = 1
    activation_steps: int = cfg.initialize_scales_steps

    initialize_weight_scales = InitializeScales(
        weight_steps=weight_steps, activation_steps=None
    )
    initialize_activations_scales = InitializeScales(
        weight_steps=None,
        activation_steps=activation_steps * data.gradient_accumulation_steps,
    )

    data_iterator = iter(data)

    for step in trange(
        weight_steps + activation_steps, desc="initializing quantization scales"
    ):
        batch = next(data_iterator)

        initialize_weight_scales_this_step: bool = (
            step < weight_steps
            if cfg.initialize_weight_scales_first
            else step >= activation_steps
        )
        initialize_scales = (
            initialize_weight_scales
            if initialize_weight_scales_this_step
            else initialize_activations_scales
        )

        for micro_batch in (
            [batch.micro_batches[0]]
            if initialize_weight_scales_this_step
            else batch.micro_batches
        ):
            model(
                input_tokens=micro_batch.input_tokens,
                priors=micro_batch.priors,
                loss_mask=micro_batch.loss_mask,
                output_tokens=micro_batch.output_tokens,
                cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
                override_window_size=cfg.window_size_schedule(0)
                if cfg.window_size_schedule is not None
                else None,
                initialize_scales=initialize_scales,
            )

    if dist.is_initialized():
        for module in raw_model.modules():
            if isinstance(module, Quantize):
                if module.quant.scale_init == "scaled_mean":
                    dist.all_reduce(module.scale.data, op=dist.ReduceOp.SUM)
                    module.scale.data /= dist.get_world_size()
                elif module.quant.scale_init == "scaled_max":
                    dist.all_reduce(module.scale.data, op=dist.ReduceOp.MAX)
                else:
                    assert False, "unreachable"


# ruff: noqa: F722
