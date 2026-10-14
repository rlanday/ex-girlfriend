from pytorch_optimizer.optimizer.soap import SOAP
from torch.optim import AdamW, Muon
from torch.nn import Parameter
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from pysrc.model import Transformer


@dataclass(frozen=True, slots=True)
class MuonConfig:
    base_learning_rate: float
    weight_decay: float
    momentum: float
    adjust_lr_fn: Literal["original", "match_rms_adamw"]


@dataclass(frozen=True, slots=True)
class SOAPConfig:
    base_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    max_precondition_dim: int


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    base_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate_multiplier_schedule: Callable[[float], float]
    second_order_config: MuonConfig | SOAPConfig
    adamw_config: AdamWConfig


@dataclass(frozen=True, slots=True)
class LearningRates:
    multiplier: float
    second_order_lr: float
    adamw_lr: float


@dataclass(frozen=True, slots=True)
class Optimizers:
    second_order: Muon | SOAP
    adamw: AdamW
    cfg: OptimizerConfig

    def step(self) -> None:
        self.second_order.step()
        self.adamw.step()

    def zero_grad(self) -> None:
        self.second_order.zero_grad()
        self.adamw.zero_grad()

    def _learning_rate_schedule(self, fraction_done: float) -> LearningRates:
        multiplier: float = self.cfg.learning_rate_multiplier_schedule(fraction_done)
        return LearningRates(
            multiplier=multiplier,
            second_order_lr=multiplier
            * self.cfg.second_order_config.base_learning_rate,
            adamw_lr=multiplier * self.cfg.adamw_config.base_learning_rate,
        )

    def _set_learning_rates(self, learning_rates: LearningRates) -> None:
        for group in self.second_order.param_groups:
            group["lr"] = learning_rates.second_order_lr
        for group in self.adamw.param_groups:
            group["lr"] = learning_rates.adamw_lr

    def update_learning_rate_schedule(self, fraction_done) -> LearningRates:
        learning_rates: LearningRates = self._learning_rate_schedule(
            fraction_done=fraction_done
        )
        self._set_learning_rates(learning_rates)
        return learning_rates


def make_optimizers(model: Transformer, cfg: OptimizerConfig) -> Optimizers:
    adamw_parameters: list[Parameter] = list(model.embedding.parameters())
    if model.cfg.prior_embedding:
        adamw_parameters += model.prior_embedding.parameters()
    if model.cfg.prior_embedding_connections:
        adamw_parameters += model.initial_prior_embedding_coefficient.parameters()
    if model.cfg.prior_logit_mixing:
        adamw_parameters += model.logit_mixing_head.parameters()
    if model.cfg.value_embedding_pattern is not None:
        adamw_parameters += model.value_embeddings.parameters()
    if model.cfg.prior_value_embedding:
        adamw_parameters += model.prior_value_embeddings.parameters()
    adamw_parameters += model.unembedding.parameters()
    if model.cfg.skip_connections:
        adamw_parameters += model.skip_connection_weights.parameters()

    non_embedding_parameters: list[Parameter] = list(model.blocks.parameters())
    assert all(p.ndim <= 2 for p in non_embedding_parameters)
    matrix_parameters: list[Parameter] = [
        p for p in non_embedding_parameters if p.ndim == 2
    ]
    scalar_parameters: list[Parameter] = [
        p for p in non_embedding_parameters if p.ndim != 2
    ]

    muon_parameters = matrix_parameters
    adamw_parameters += scalar_parameters

    assert len(muon_parameters) + len(adamw_parameters) == len(list(model.parameters()))
    muon_parameter_ids: set[int] = {id(p) for p in muon_parameters}
    adamw_parameter_ids: set[int] = {id(p) for p in adamw_parameters}
    assert muon_parameter_ids.isdisjoint(adamw_parameter_ids)

    return Optimizers(
        second_order=make_second_order_optimizer(
            muon_parameters, cfg.second_order_config
        ),
        adamw=make_adamw(adamw_parameters, cfg.adamw_config),
        cfg=cfg,
    )


def make_second_order_optimizer(
    parameters: Iterable[Parameter], cfg: MuonConfig | SOAPConfig
) -> Muon | SOAP:
    if isinstance(cfg, MuonConfig):
        return Muon(
            parameters,
            lr=cfg.base_learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            adjust_lr_fn=cfg.adjust_lr_fn,
        )

    if isinstance(cfg, SOAPConfig):
        return SOAP(
            parameters,
            lr=cfg.base_learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            max_precondition_dim=cfg.max_precondition_dim,
        )

    assert False, f"invalid optimizer config type '{type(cfg)}'"


def make_adamw(parameters: Iterable[Parameter], cfg: AdamWConfig) -> AdamW:
    return AdamW(
        parameters,
        lr=cfg.base_learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        fused=True,
    )
