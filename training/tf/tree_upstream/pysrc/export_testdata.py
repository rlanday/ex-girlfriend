"""export the C++ inference test dataset to cpp_infer/data/ per
cpp_infer/SPEC.md section 5

run from the repository root:  .venv/bin/python -m pysrc.export_testdata

reconstructs the 1024-article subset exactly as
pysrc.save_outputs.save_transformer_outputs_for_article_subset did when
training_recipes/save_outputs.py produced data/*-1024-articles.* (and verifies
the reconstruction byte-exactly against those files before writing anything),
then writes:

- test_tokens.u8 / test_bounds.i32: copies of the subset tokens/boundaries
- test_priors.f16: the ppmd prior row of every subset token (row r pairs with
  input token r), gathered from the 240 GB priors mmap
- test_rope_offsets.i32: per subset article, its start position within its
  packed reference micro batch (the rope offset of its first token)
- test_ref_probs.f16: copy of data/transformer-probs-1024-articles.bfloat16
  (float16 data despite the name)
- ref_loss.txt: "fp32 <loss_sum> <count> <mean>" from rerunning the model on
  GPU over the exact reference micro batches, and "f16 <loss_sum> <count>
  <mean>" recomputed from the saved float16 probability file
- dumps/article{K}/: fp32 .npy dumps of per-component intermediates for three
  diagnostic articles (+ meta.json), for the C++ component tests
"""

import json
import os
import shutil

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from pysrc.data import CollatedArticlesDataLoader
from pysrc.export_weights import load_reference_model
from pysrc.model import Transformer
from pysrc.save_outputs import _iter_micro_batch_probabilities

OUT_DIR = "cpp_infer/data"
N_ARTICLES = 1024
SUBSET_SEED = 42
DUMP_POSITIONS = 4096
PRIOR_SPARSITY_SAMPLE_ROWS = 100_000
PRIOR_SPARSITY_SEED = 123

REFERENCE_TOKENS = "data/tokens-1024-articles.uint8"
REFERENCE_BOUNDS = "data/article-boundaries-1024-articles.int32"
REFERENCE_PROBS = "data/transformer-probs-1024-articles.bfloat16"


def make_data_loader(device: torch.device) -> CollatedArticlesDataLoader:
    """exactly the loader of training_recipes/save_outputs.py"""
    max_article_tokens: int = 2**17
    return CollatedArticlesDataLoader(
        tokens_filename="data/ppmd-bytes.uint8",
        priors_filename="data/ppmd-probs.float16",
        article_boundaries_filename="data/article-boundaries.int32",
        mmap=True,
        epochs=1,
        micro_batch_tokens=max_article_tokens,
        gradient_accumulation_steps=1,
        max_article_tokens=max_article_tokens,
        max_merge_tokens=None,
        device=device,
        rank=0,
        world_size=1,
    )


def bf16_value(scale: Tensor) -> float:
    """the fp32 value of the bfloat16-rounded scale, as used in arithmetic"""
    return float(scale.detach().to(torch.bfloat16).to(torch.float32).item())


class Capture:
    """forward hooks that slice the captured tensor to the dumped article's
    positions on the position axis (dim 1) and move it to cpu fp32 immediately"""

    def __init__(self, offset: int, n_positions: int) -> None:
        self.offset = offset
        self.n_positions = n_positions
        self.arrays: dict[str, np.ndarray] = {}
        self.handles: list = []

    def store(self, name: str, tensor: Tensor) -> None:
        assert name not in self.arrays, name
        sliced = tensor[:, self.offset : self.offset + self.n_positions]
        assert sliced.size(1) == self.n_positions, name
        self.arrays[name] = sliced.detach().to(torch.float32).cpu().numpy()[0]

    def output_hook(self, name: str):
        def hook(module, args, output) -> None:
            self.store(name, output)

        return hook

    def input_hook(self, name: str, arg_index: int = 0):
        def hook(module, args) -> None:
            self.store(name, args[arg_index])

        return hook

    def add(self, handle) -> None:
        self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def register_dump_hooks(model: Transformer, capture: Capture) -> None:
    kimi = model.cfg.kimi_linear
    assert kimi is not None

    # the input to blocks.0 is x0 = normed token embedding + normed prior
    # embedding (the pre hook fires before the block applies the residual
    # stream / token embedding coefficients)
    capture.add(
        model.blocks[0].register_forward_pre_hook(capture.input_hook("00_x0"))
    )

    for i, block in enumerate(model.blocks):
        p = f"{i:02d}"
        # input to blocks.i, after the skip connection add of destination layers
        # (the skip add happens in Transformer.compute_last_activation before
        # the block is called)
        capture.add(
            block.register_forward_pre_hook(capture.input_hook(f"{p}_block_input"))
        )
        capture.add(
            block.register_forward_hook(capture.output_hook(f"{p}_block_output"))
        )
        attention = block.attention
        capture.add(
            attention.register_forward_hook(capture.output_hook(f"{p}_attn_out"))
        )
        capture.add(
            block.mlp.register_forward_hook(capture.output_hook(f"{p}_mlp_out"))
        )

        if kimi[i]:
            capture.add(
                attention.query_convolution.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_conv_q")
                )
            )
            capture.add(
                attention.key_convolution.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_conv_k")
                )
            )
            capture.add(
                attention.value_convolution.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_conv_v")
                )
            )
            capture.add(
                attention.forget_gate_projection.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_g_raw")
                )
            )
            capture.add(
                attention.beta_projection.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_beta_raw")  # pre-sigmoid
                )
            )
            capture.add(
                attention.output_gate_projection.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_out_gate")
                )
            )
            # raw chunk_kda output and the gate, as passed to the fused norm
            capture.add(
                attention.output_fused_norm_gate.register_forward_pre_hook(
                    capture.input_hook(f"{p}_kimi_kda_out", 0)
                )
            )
            capture.add(
                attention.output_fused_norm_gate.register_forward_pre_hook(
                    capture.input_hook(f"{p}_kimi_gate_in", 1)
                )
            )
            capture.add(
                attention.output_fused_norm_gate.register_forward_hook(
                    capture.output_hook(f"{p}_kimi_gated_norm_out")
                )
            )
        else:
            # fake-quantized fp32 q/k/v after rope (per-head static scales)
            capture.add(
                attention.quantize_queries.register_forward_hook(
                    capture.output_hook(f"{p}_attn_q_quant")
                )
            )
            capture.add(
                attention.quantize_keys.register_forward_hook(
                    capture.output_hook(f"{p}_attn_k_quant")
                )
            )
            capture.add(
                attention.quantize_values.register_forward_hook(
                    capture.output_hook(f"{p}_attn_v_quant")
                )
            )
            # concatenated attention output before the output projection
            capture.add(
                attention.output_projection.register_forward_pre_hook(
                    capture.input_hook(f"{p}_attn_pre_oproj")
                )
            )

    # final rms_norm output = the input of the unembedding
    capture.add(
        model.unembedding.register_forward_pre_hook(capture.input_hook("12_final_norm"))
    )


class MicroBatchStats:
    """diagnostic statistics over the valid (non padding) positions of one
    micro batch: per-layer fraction of zero ints in the quantization of the mlp
    relu^2 activations, and per-component max-abs of the residual stream"""

    def __init__(self, valid_length: int) -> None:
        self.valid_length = valid_length
        self.mlp_zero_fraction: dict[int, float] = {}
        self.residual_max_abs: dict[int, dict[str, float]] = {}
        self.handles: list = []

    def _max_abs_pre(self, layer: int, component: str):
        def hook(module, args) -> None:
            value = float(
                args[0][:, : self.valid_length].detach().abs().max().item()
            )
            self.residual_max_abs.setdefault(layer, {})[component] = value

        return hook

    def _max_abs_out(self, layer: int, component: str):
        def hook(module, args, output) -> None:
            value = float(
                output[:, : self.valid_length].detach().abs().max().item()
            )
            self.residual_max_abs.setdefault(layer, {})[component] = value

        return hook

    def _mlp_zero_hook(self, layer: int, scale: float):
        def hook(module, args) -> None:
            h = args[0][:, : self.valid_length].detach()
            ints = torch.round(h / scale)  # clamp cannot change the zero count
            self.mlp_zero_fraction[layer] = float(
                (ints == 0).to(torch.float64).mean().item()
            )

        return hook

    def register(self, model: Transformer) -> None:
        for i, block in enumerate(model.blocks):
            self.handles.append(
                block.register_forward_pre_hook(self._max_abs_pre(i, "block_input"))
            )
            self.handles.append(
                block.attention.register_forward_hook(
                    self._max_abs_out(i, "attention_out")
                )
            )
            self.handles.append(
                block.mlp.register_forward_hook(self._max_abs_out(i, "mlp_out"))
            )
            self.handles.append(
                block.register_forward_hook(self._max_abs_out(i, "block_output"))
            )
            assert block.mlp.down.quantize_activation is not None
            self.handles.append(
                block.mlp.down.register_forward_pre_hook(
                    self._mlp_zero_hook(
                        i, bf16_value(block.mlp.down.quantize_activation.scale)
                    )
                )
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def dump_article(
    model: Transformer,
    data: CollatedArticlesDataLoader,
    subset_index: int,
    article_id: int,
    length: int,
    offset: int,
    micro_batch_index: int,
    micro_batch_articles: list[int],
    stats: MicroBatchStats | None,
) -> tuple[int, int]:
    """run the article's reference micro batch with capture hooks and write
    dumps/article{subset_index}/; returns (n dumped positions, total bytes)"""
    n_positions = length - 1
    n_dump = min(n_positions, DUMP_POSITIONS)

    capture = Capture(offset=offset, n_positions=n_dump)
    register_dump_hooks(model, capture)
    if stats is not None:
        stats.register(model)

    micro_batch = data._make_micro_batch(micro_batch_articles)
    with torch.inference_mode():
        logits = model.compute_logits(
            input_tokens=micro_batch.input_tokens,
            priors=micro_batch.priors,
            cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
        )
    capture.remove()
    if stats is not None:
        stats.remove()

    capture.store("12_logits", logits)
    probabilities = torch.softmax(
        logits[:, offset : offset + n_dump].to(torch.float32), dim=-1
    )
    capture.arrays["12_probabilities"] = probabilities.detach().cpu().numpy()[0]

    # int8 ints of the fake-quantized q/k/v of the vanilla attention layers
    # (recovered exactly: the captured values are q * s with s the bf16-rounded
    # per-head scale, and fp32 division by s returns exactly q)
    kimi = model.cfg.kimi_linear
    assert kimi is not None
    for i, block in enumerate(model.blocks):
        if kimi[i]:
            continue
        for which, quantize in (
            ("q", block.attention.quantize_queries),
            ("k", block.attention.quantize_keys),
            ("v", block.attention.quantize_values),
        ):
            scale = (
                quantize.scale.detach()
                .to(torch.bfloat16)
                .to(torch.float32)
                .cpu()
                .numpy()
            )  # (3,) per head
            fake_quantized = capture.arrays[f"{i:02d}_attn_{which}_quant"]
            ints = np.rint(fake_quantized / scale[None, :, None])
            assert -128 <= ints.min() and ints.max() <= 127  # buckets=256 range
            assert np.array_equal(
                ints.astype(np.float32) * scale[None, :, None], fake_quantized
            )
            capture.arrays[f"{i:02d}_attn_{which}_int8"] = ints.astype(np.int8)

    article_dir = os.path.join(OUT_DIR, "dumps", f"article{subset_index}")
    if os.path.exists(article_dir):
        shutil.rmtree(article_dir)
    os.makedirs(article_dir)

    total_bytes = 0
    for name in sorted(capture.arrays):
        path = os.path.join(article_dir, f"{name}.npy")
        np.save(path, capture.arrays[name])
        total_bytes += os.path.getsize(path)

    meta = {
        "subset_index": subset_index,
        "orig_article_id": article_id,
        "length_tokens": length,
        "n_positions": n_positions,
        "rope_offset": offset,
        "micro_batch_index": micro_batch_index,
        "dumped_positions": n_dump,
        "micro_batch_articles": micro_batch_articles,
        "components": {
            name: list(array.shape) + [str(array.dtype)]
            for name, array in sorted(capture.arrays.items())
        },
    }
    with open(os.path.join(article_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    return n_dump, total_bytes


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)

    data = make_data_loader(device)
    assert data.vocabulary_size == 205

    article_starts = data.article_starts
    selected = np.random.default_rng(seed=SUBSET_SEED).permutation(data.n_articles)[
        :N_ARTICLES
    ]
    lengths = (article_starts[selected + 1] - article_starts[selected]).astype(
        np.int64
    )
    new_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    new_n_tokens = int(new_starts[-1])

    # ---- verify the reconstruction against the reference subset files ----
    tokens = np.concatenate(
        [
            data.tokens[int(article_starts[a]) : int(article_starts[a + 1])]
            for a in selected
        ]
    )
    assert len(tokens) == new_n_tokens
    reference_tokens = np.fromfile(REFERENCE_TOKENS, dtype=np.uint8)
    reference_bounds = np.fromfile(REFERENCE_BOUNDS, dtype=np.int32)
    assert np.array_equal(tokens, reference_tokens), (
        "STOP: reconstructed subset tokens do not match "
        f"{REFERENCE_TOKENS} byte-exactly"
    )
    assert np.array_equal(new_starts.astype(np.int32), reference_bounds), (
        "STOP: reconstructed article boundaries do not match "
        f"{REFERENCE_BOUNDS}"
    )
    print(
        f"verified: subset reconstruction matches {REFERENCE_TOKENS} and "
        f"{REFERENCE_BOUNDS} byte-exactly "
        f"({N_ARTICLES} articles, {new_n_tokens} tokens)"
    )

    n_length_one = int((lengths == 1).sum())

    # ---- copies ----
    shutil.copyfile(REFERENCE_TOKENS, os.path.join(OUT_DIR, "test_tokens.u8"))
    shutil.copyfile(REFERENCE_BOUNDS, os.path.join(OUT_DIR, "test_bounds.i32"))
    print("copying reference probabilities (float16 data)")
    shutil.copyfile(REFERENCE_PROBS, os.path.join(OUT_DIR, "test_ref_probs.f16"))

    # ---- test_priors.f16: one 205-float16 row per subset token ----
    priors_path = os.path.join(OUT_DIR, "test_priors.f16")
    out_priors = np.memmap(
        priors_path,
        dtype=np.float16,
        mode="w+",
        shape=(new_n_tokens, data.vocabulary_size),
    )
    for i, article in enumerate(tqdm(selected, desc="gathering priors")):
        start = int(article_starts[article])
        end = int(article_starts[article + 1])
        out_priors[int(new_starts[i]) : int(new_starts[i + 1])] = data.priors[
            start:end
        ]
    out_priors.flush()

    # ---- test_rope_offsets.i32: article start position in its micro batch ----
    micro_batches: list[list[int]] = list(
        data._iter_micro_batch_articles(selected.tolist(), drop_last=False)
    )
    assert len(micro_batches) == data.n_micro_batches(selected.tolist())
    offset_of_article: dict[int, int] = {}
    micro_batch_of_article: dict[int, int] = {}
    for micro_batch_index, articles in enumerate(micro_batches):
        offset = 0
        for article in articles:
            offset_of_article[article] = offset
            micro_batch_of_article[article] = micro_batch_index
            offset += int(article_starts[article + 1] - article_starts[article] - 1)
        assert offset <= data.micro_batch_tokens

    rope_offsets = np.zeros(N_ARTICLES, dtype=np.int32)
    n_never_packed = 0
    for i, article in enumerate(selected.tolist()):
        if article in offset_of_article:
            rope_offsets[i] = offset_of_article[article]
        else:
            # length-1 article: contributes no positions, appears in no micro
            # batch, is never predicted; offset 0 is written and never used
            assert int(lengths[i]) == 1
            n_never_packed += 1
    rope_offsets.tofile(os.path.join(OUT_DIR, "test_rope_offsets.i32"))
    print(
        f"{len(micro_batches)} reference micro batches; "
        f"{n_length_one} length-1 articles ({n_never_packed} never packed)"
    )

    # ---- model: exact fp32 reference loss + probability re-computation ----
    model = load_reference_model(device)

    reference_probs = np.memmap(
        os.path.join(OUT_DIR, "test_ref_probs.f16"),
        dtype=np.float16,
        mode="r",
        shape=(new_n_tokens, data.vocabulary_size),
    )
    new_start_of_article = {
        int(article): int(new_start)
        for article, new_start in zip(selected, new_starts[:-1])
    }

    loss_sum = 0.0
    predicted_count = 0
    compare_max_abs_diff = 0.0
    compare_n_different = 0
    compare_n_values = 0
    yielded_micro_batches: list[list[int]] = []
    for articles, probabilities, micro_batch_loss, micro_batch_count in (
        _iter_micro_batch_probabilities(
            model=model,
            data=data,
            articles=selected.tolist(),
            description="reference loss",
        )
    ):
        yielded_micro_batches.append(list(articles))
        loss_sum += micro_batch_loss
        predicted_count += micro_batch_count

        position = 0
        for article in articles:
            length = int(article_starts[article + 1] - article_starts[article] - 1)
            new_start = new_start_of_article[article]
            fresh = probabilities[position : position + length]
            saved = np.asarray(reference_probs[new_start : new_start + length])
            if not np.array_equal(fresh.view(np.uint16), saved.view(np.uint16)):
                difference = np.abs(
                    fresh.astype(np.float32) - saved.astype(np.float32)
                )
                compare_max_abs_diff = max(
                    compare_max_abs_diff, float(difference.max())
                )
                compare_n_different += int((difference > 0).sum())
            compare_n_values += fresh.size
            position += length

    assert yielded_micro_batches == micro_batches, (
        "micro batch packing of the model run does not match the rope offsets walk"
    )
    assert predicted_count == new_n_tokens - N_ARTICLES
    mean_loss = loss_sum / predicted_count
    print(
        f"fp32 reference loss: sum {loss_sum!r} over {predicted_count} predicted "
        f"tokens, mean {mean_loss!r} nats/token"
    )
    print(
        f"probability re-computation vs {REFERENCE_PROBS}: compared "
        f"{compare_n_values} float16 values, {compare_n_different} differ, "
        f"max abs diff {compare_max_abs_diff!r}"
    )

    # ---- loss recomputed from the saved float16 probability file ----
    f16_loss_sum = 0.0
    f16_count = 0
    for i in range(N_ARTICLES):
        start = int(new_starts[i])
        end = int(new_starts[i + 1])
        if end - start < 2:
            continue
        rows = np.asarray(reference_probs[start : end - 1]).astype(np.float64)
        targets = reference_tokens[start + 1 : end]
        probabilities_of_targets = rows[np.arange(end - 1 - start), targets]
        f16_loss_sum += float(
            -np.log(np.maximum(probabilities_of_targets, 1e-12)).sum()
        )
        f16_count += end - 1 - start
    assert f16_count == predicted_count
    f16_mean = f16_loss_sum / f16_count
    print(
        f"f16-file loss: sum {f16_loss_sum!r} over {f16_count} predicted tokens, "
        f"mean {f16_mean!r} nats/token"
    )

    with open(os.path.join(OUT_DIR, "ref_loss.txt"), "w") as f:
        f.write(f"fp32 {loss_sum!r} {predicted_count} {mean_loss!r}\n")
        f.write(f"f16 {f16_loss_sum!r} {f16_count} {f16_mean!r}\n")

    # ---- prior-quantization sparsity over random subset rows ----
    assert model.prior_embedding.quantize_activation is not None
    prior_scale = bf16_value(model.prior_embedding.quantize_activation.scale)
    sample_rows = np.sort(
        np.random.default_rng(seed=PRIOR_SPARSITY_SEED).choice(
            new_n_tokens, size=PRIOR_SPARSITY_SAMPLE_ROWS, replace=False
        )
    )
    sampled = torch.from_numpy(np.asarray(out_priors[sample_rows])).to(device)
    sampled_ints = torch.clamp(
        torch.round(sampled.to(torch.float32) / prior_scale), -128, 127
    )
    nonzeros = (sampled_ints != 0).sum(-1).cpu().numpy()
    prior_sparsity = {
        "scale_bf16": prior_scale,
        "sample_rows": int(PRIOR_SPARSITY_SAMPLE_ROWS),
        "nonzero_mean": float(nonzeros.mean()),
        "nonzero_median": float(np.median(nonzeros)),
        "nonzero_p99": float(np.percentile(nonzeros, 99)),
        "nonzero_max": int(nonzeros.max()),
    }
    print(
        f"prior quantization sparsity (scale {prior_scale}): nonzero ints per "
        f"row over {PRIOR_SPARSITY_SAMPLE_ROWS} random subset rows: "
        f"mean {prior_sparsity['nonzero_mean']:.2f}, "
        f"median {prior_sparsity['nonzero_median']:.0f}, "
        f"p99 {prior_sparsity['nonzero_p99']:.0f}, "
        f"max {prior_sparsity['nonzero_max']}"
    )

    # ---- diagnostic article dumps ----
    def first_index(condition: np.ndarray) -> int | None:
        indices = np.nonzero(condition)[0]
        return int(indices[0]) if len(indices) else None

    dump_indices: list[int] = []
    assert int(lengths[0]) >= 2, "subset article 0 has length 1, cannot dump it"
    dump_indices.append(0)
    medium = first_index((lengths >= 2000) & (lengths <= 8000))
    assert medium is not None
    long = first_index(lengths >= 16384)
    if long is None:
        long = int(np.argmax(lengths))
    for index in (medium, long):
        if index not in dump_indices:
            dump_indices.append(index)

    stats = None
    dump_inventory = []
    for order, subset_index in enumerate(dump_indices):
        article_id = int(selected[subset_index])
        length = int(lengths[subset_index])
        offset = offset_of_article[article_id]
        micro_batch_index = micro_batch_of_article[article_id]
        micro_batch_articles = micro_batches[micro_batch_index]

        micro_batch_stats = None
        if order == 0:
            valid_length = sum(
                int(article_starts[a + 1] - article_starts[a] - 1)
                for a in micro_batch_articles
            )
            micro_batch_stats = MicroBatchStats(valid_length)

        n_dump, total_bytes = dump_article(
            model=model,
            data=data,
            subset_index=subset_index,
            article_id=article_id,
            length=length,
            offset=offset,
            micro_batch_index=micro_batch_index,
            micro_batch_articles=micro_batch_articles,
            stats=micro_batch_stats,
        )
        if micro_batch_stats is not None:
            stats = micro_batch_stats
        dump_inventory.append(
            {
                "subset_index": subset_index,
                "orig_article_id": article_id,
                "length_tokens": length,
                "rope_offset": offset,
                "micro_batch_index": micro_batch_index,
                "dumped_positions": n_dump,
                "bytes": total_bytes,
            }
        )
        print(
            f"dumped article{subset_index}: orig id {article_id}, "
            f"{length} tokens, rope offset {offset}, micro batch "
            f"{micro_batch_index}, {n_dump} positions, {total_bytes} bytes"
        )

    assert stats is not None
    print(
        "mlp relu^2 int quantization zero fraction by layer "
        f"(micro batch {dump_inventory[0]['micro_batch_index']}, "
        f"{stats.valid_length} valid positions):"
    )
    for layer in range(model.cfg.n_layers):
        print(f"  layer {layer:2d}: {stats.mlp_zero_fraction[layer]:.4f}")
    print("residual stream fp32 max-abs by layer (same micro batch):")
    print("  layer  block_input  attention_out  mlp_out  block_output")
    for layer in range(model.cfg.n_layers):
        r = stats.residual_max_abs[layer]
        print(
            f"  {layer:5d}  {r['block_input']:11.3f}  {r['attention_out']:13.3f}"
            f"  {r['mlp_out']:7.3f}  {r['block_output']:12.3f}"
        )

    with open(os.path.join(OUT_DIR, "dumps", "stats.json"), "w") as f:
        json.dump(
            {
                "n_micro_batches": len(micro_batches),
                "n_length_one_articles": n_length_one,
                "fp32_loss": {
                    "sum": loss_sum,
                    "count": predicted_count,
                    "mean": mean_loss,
                },
                "f16_file_loss": {
                    "sum": f16_loss_sum,
                    "count": f16_count,
                    "mean": f16_mean,
                },
                "probability_recomputation": {
                    "compared_values": compare_n_values,
                    "different_values": compare_n_different,
                    "max_abs_diff": compare_max_abs_diff,
                },
                "prior_quantization_sparsity": prior_sparsity,
                "mlp_relu2_int_zero_fraction_by_layer": stats.mlp_zero_fraction,
                "residual_stream_max_abs_by_layer": stats.residual_max_abs,
                "dumps": dump_inventory,
            },
            f,
            indent=1,
        )

    print("export complete")


if __name__ == "__main__":
    main()
