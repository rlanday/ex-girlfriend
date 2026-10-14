from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
from os import makedirs
import os
from math import cos, pi
from dotenv import load_dotenv

from pysrc.model import Transformer, TransformerConfig
from pysrc.train import train, TrainingConfig
from pysrc.optimizer import AdamWConfig, MuonConfig, OptimizerConfig
from pysrc.data import CollatedArticlesDataLoader
from pysrc.save_outputs import save_transformer_outputs
from pysrc.logger import LoggerConfig


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main_process: bool = rank == 0
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    micro_batch_tokens: int = 2**20 // world_size
    max_article_tokens: int | None = 2**17

    raw_model: Transformer = Transformer(
        TransformerConfig(
            vocabulary_size=205,
            prior_embedding=True,
            prior_logit_mixing=False,
            max_variable_sequence_length=max_article_tokens
            if max_article_tokens is not None
            else micro_batch_tokens,
            n_layers=12,
            attention_weight_sharing_pattern=list(range(12)),
            mlp_weight_sharing_pattern=list(range(12)),
            d_model=192,
            kimi_linear=[True, True, True, False] * 3,
            window_size=1024,
            window_size_multipliers=None,
            n_query_heads=3,
            n_key_value_heads=3,
            d_head=64,
            d_mla=None,
            rope_base=10_000,
            half_truncate_rope=False,
            query_key_norm=True,
            query_key_norm_gain=False,
            kimi_linear_d_head=64,
            kimi_linear_n_heads=3,
            kimi_linear_convolution_size=4,
            value_embedding_pattern=None,
            prior_value_embedding=False,
            attention_scale=None,
            d_mlp=768,
            activation_function="relu_squared",
            embedding_norm=True,
            prior_embedding_norm=True,
            skip_connections=True,
            token_embedding_connections=True,
            prior_embedding_connections=False,
            dropout=0.0,
            logit_softcap=15.0,
            prior_logprob_cap=-12.0,
            prior_embedding_on_logprobs=False,
            gradient_checkpointing=False,
            embedding_init="normal",
            prior_embedding_init="normal",
            up_init="kaiming_uniform",
            down_init="kaiming_uniform",
            unembedding_init="kaiming_uniform",
            convolution_init="kaiming_uniform",
            logit_mixing_head_init="zero",
            activation_dtype=torch.bfloat16,
            attention_dtype=torch.bfloat16,
            logit_dtype=torch.float32,
            weight_dtype=torch.float32,
            mlp_quantization=None,
            attention_full_rank_quantization=None,
            attention_low_rank_quantization=None,
            qkv_quantization=None,
            kimi_linear_full_rank_quantization=None,
            kimi_linear_low_rank_quantization=None,
            kimi_linear_beta_projection_quantization=None,
            kimi_linear_convolution_quantization=None,
            embedding_quantization=None,
            prior_embedding_quantization=None,
            unembedding_quantization=None,
            logit_mixing_head_quantization=None,
            adamw_quantization_scales=True,
            attention_implementation="flash_attention_2",
        )
    )

    raw_model = raw_model.to(device)
    # raw_model.load_state_dict(torch.load("models/6m-warmed-up.tch"))
    model: DDP = DDP(raw_model)
    model = torch.compile(model)  # type: ignore

    learning_rate: float = 5e-4
    warmup_epochs: int = 64
    epochs: int = 256 + 64
    warmup_learning_rate_multiplier: int = 8
    final_learning_rate_multiplier: float = 0.025

    data = CollatedArticlesDataLoader(
        tokens_filename="data/ppmd-bytes.uint8",
        priors_filename="data/ppmd-probs.float16",
        article_boundaries_filename="data/article-boundaries.int32",
        mmap=True,
        epochs=epochs,
        micro_batch_tokens=micro_batch_tokens,
        gradient_accumulation_steps=1,
        max_article_tokens=max_article_tokens,
        max_merge_tokens=None,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    # if main_process:
    #     model.load_state_dict(torch.load("models/6m.tch"))
    #     save_transformer_outputs(model, data, "data/transformer-probs-6m.float16")
    #     return

    def learning_rate_multiplier_schedule(fraction_done: float) -> float:
        fraction_warmup = warmup_epochs / epochs
        if fraction_done < fraction_warmup:
            s = fraction_done / fraction_warmup
            t = (cos(s * pi) + 1) / 2
            return t * warmup_learning_rate_multiplier + (1 - t)

        s = (fraction_done - fraction_warmup) / (1 - fraction_warmup)
        t = (cos(s * pi) + 1) / 2
        return t + (1 - t) * final_learning_rate_multiplier

    training_config = TrainingConfig(
        initialize_scales_steps=None,
        initialize_weight_scales_first=True,
        optimizer_config=OptimizerConfig(
            learning_rate_multiplier_schedule=learning_rate_multiplier_schedule,
            second_order_config=MuonConfig(
                base_learning_rate=learning_rate,
                weight_decay=0.1,
                momentum=0.95,
                adjust_lr_fn="match_rms_adamw",
            ),
            # second_order_config=SOAPConfig(
            #     base_learning_rate=3e-3,
            #     weight_decay=1e-2,
            #     betas=(0.95, 0.95),
            #     max_precondition_dim=10_000,
            # ),
            adamw_config=AdamWConfig(
                base_learning_rate=learning_rate,
                weight_decay=0.01,
                betas=(0.95, 0.99),
            ),
        ),
        window_size_schedule=None,
        logger_config=LoggerConfig(
            wandb_project="fx2-cmix-transformer", wandb_run_name="6m"
        ),
    )

    train(raw_model=raw_model, model=model, data=data, cfg=training_config)

    if main_process:
        makedirs("models", exist_ok=True)
        torch.save(raw_model.state_dict(), "models/6m.tch")

    dist.destroy_process_group()


if __name__ == "__main__":
    load_dotenv()
    main()
