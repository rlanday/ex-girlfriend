import numpy as np
import torch
from math import log
from torch import Tensor
from torch.nn.functional import cross_entropy, softmax
from tqdm import tqdm
from jaxtyping import Float
from typing import Iterator, Sequence

from pysrc.data import CollatedArticlesDataLoader
from pysrc.model import Transformer


@torch.inference_mode()
def _iter_micro_batch_probabilities(
    model: Transformer,
    data: CollatedArticlesDataLoader,
    articles: Sequence[int],
    description: str,
) -> Iterator[tuple[list[int], np.ndarray, float, int]]:
    """run the model over the given articles, in the given order and batched the
    same way as during training, and yield for every micro batch its articles,
    its next token distributions (one row per packed position, so article a
    contributes its length - 1 rows starting where the previous article of the
    micro batch ended), the summed cross entropy over the predicted positions
    and their count"""
    for micro_batch_articles, micro_batch in tqdm(
        data.iter_micro_batches(articles),
        total=data.n_micro_batches(articles),
        desc=description,
    ):
        logits: Float[Tensor, "1 position vocabulary_size"] = model.compute_logits(
            input_tokens=micro_batch.input_tokens,
            priors=micro_batch.priors,
            cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
        )

        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            micro_batch.output_tokens.flatten(),
            reduction="none",
        )
        loss_sum: float = (
            torch.where(micro_batch.loss_mask.flatten(), loss.to(torch.float32), 0.0)
            .sum()
            .item()
        )

        probabilities: np.ndarray = (
            softmax(logits.to(torch.float32), dim=-1)[0].to(torch.float16).cpu().numpy()
        )

        yield (
            micro_batch_articles,
            probabilities,
            loss_sum,
            int(micro_batch.loss_mask.sum().item()),
        )


@torch.inference_mode()
def save_transformer_outputs(
    model: Transformer,
    data: CollatedArticlesDataLoader,
    output_filename: str,
) -> None:
    """evaluate the transformer on all articles of the dataloader (in file
    order, batched the same way as during training) and write its next token
    distributions to output_filename in the exact format of the dataloader's
    priors file: n_tokens rows of vocabulary_size float16 probabilities, row i
    being the distribution over token i + 1 given the tokens up to i; rows at
    which the transformer makes no prediction keep the ppmd's prior — the last
    row of each article (its target is the first token of the following
    article, which starts a fresh context) and the rows of length-1 articles

    also prints the transformer's loss in nats per token, averaged uniformly
    over all tokens it predicts, and the loss averaged uniformly over all
    predicted tokens (all tokens but the first, with the ppmd's loss at the
    tokens where the ppmd's distribution is saved)"""
    was_training: bool = model.training
    model.eval()

    article_starts: np.ndarray = data.article_starts
    outputs: np.ndarray = np.memmap(
        output_filename,
        dtype=np.float16,
        mode="w+",
        shape=(data.n_tokens, data.vocabulary_size),
    )

    total_loss: float = 0.0
    total_predicted_tokens: int = 0

    for (
        articles,
        probabilities,
        loss_sum,
        predicted_tokens,
    ) in _iter_micro_batch_probabilities(
        model=model,
        data=data,
        articles=range(data.n_articles),
        description="saving transformer outputs",
    ):
        total_loss += loss_sum
        total_predicted_tokens += predicted_tokens

        position: int = 0
        for article in articles:
            start: int = int(article_starts[article])
            end: int = int(article_starts[article + 1])
            length: int = end - start - 1
            outputs[start : end - 1] = probabilities[position : position + length]
            position += length

    # the last row of each article is never predicted by the transformer, and a
    # length-1 article consists only of its last row; each such row (except the
    # file's last, whose target is past the end of the stream) predicts the
    # first token of the following article using the ppmd's prior
    total_prior_loss: float = 0.0
    total_prior_predicted_tokens: int = 0
    for article in range(data.n_articles):
        end = int(article_starts[article + 1])
        outputs[end - 1] = data.priors[end - 1]
        if end < data.n_tokens:
            probability = float(data.priors[end - 1, data.tokens[end]])
            total_prior_loss += -log(max(probability, 1e-12))
            total_prior_predicted_tokens += 1

    outputs.flush()

    assert total_predicted_tokens + total_prior_predicted_tokens == data.n_tokens - 1, (
        "every token but the first must be predicted exactly once"
    )

    print(
        f"transformer loss: {total_loss / total_predicted_tokens:.4f} nats per token "
        f"(averaged over {total_predicted_tokens} predicted tokens)"
    )
    print(
        f"overall loss with the ppmd's predictions at the remaining tokens: "
        f"{(total_loss + total_prior_loss) / (data.n_tokens - 1):.4f} nats per token "
        f"(averaged over {data.n_tokens - 1} predicted tokens, "
        f"{total_prior_predicted_tokens} of them predicted by the ppmd)"
    )

    if was_training:
        model.train()


@torch.inference_mode()
def save_transformer_outputs_for_article_subset(
    model: Transformer,
    data: CollatedArticlesDataLoader,
    output_filename: str,
    n_articles: int,
    tokens_filename: str,
    article_boundaries_filename: str,
    seed: int = 42,
) -> None:
    """pick n_articles of the dataloader's articles uniformly at random and in a
    random order (both fixed by seed, without replacement), evaluate the
    transformer on them and write a complete, self contained dataset of the
    three files the dataloader reads, holding only those articles in that order:

    - tokens_filename: the selected articles' tokens concatenated, uint8, the
      format of the dataloader's tokens file
    - article_boundaries_filename: n_articles + 1 int32 token indices, 0 first
      and the new token count last, the format of the dataloader's article
      boundaries file
    - output_filename: one row of vocabulary_size float16 probabilities per new
      token, row i the distribution over token i + 1, the format of the
      dataloader's priors file — exactly as in save_transformer_outputs, the
      last row of every article keeps the ppmd's prior (the transformer makes no
      prediction there), and a length-1 article consists only of such a row

    note that the articles are the dataloader's, i.e. the boundaries file's
    articles after merging and splitting, so the written files reproduce the
    contexts the transformer was actually run on; the ppmd priors kept at the
    articles' last rows were computed in the original context and therefore
    predict the token that followed there, not the one that now follows

    also prints the same two losses as save_transformer_outputs, measured on the
    written files"""
    assert 1 <= n_articles <= data.n_articles, (
        f"cannot select {n_articles} of {data.n_articles} articles"
    )

    was_training: bool = model.training
    model.eval()

    article_starts: np.ndarray = data.article_starts

    # permutation then truncation gives a uniformly random subset in a uniformly
    # random order in one step
    selected: np.ndarray = np.random.default_rng(seed=seed).permutation(
        data.n_articles
    )[:n_articles]

    # the selected articles keep their lengths and are laid out back to back in
    # the selected order, so new_starts is to the new files what article_starts
    # is to the dataloader's: new_starts[i] is where selected[i] begins and
    # new_starts[i + 1] is one past its last token
    lengths: np.ndarray = (
        article_starts[selected + 1] - article_starts[selected]
    ).astype(np.int64)
    new_starts: np.ndarray = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(lengths))
    )
    new_n_tokens: int = int(new_starts[-1])
    assert new_n_tokens >= 2, "the written dataset must have at least two tokens"
    assert new_n_tokens <= np.iinfo(np.int32).max, (
        "the article boundaries file cannot hold token indices beyond int32"
    )

    tokens: np.ndarray = np.concatenate(
        [
            data.tokens[int(article_starts[article]) : int(article_starts[article + 1])]
            for article in selected
        ]
    )
    assert len(tokens) == new_n_tokens
    tokens.tofile(tokens_filename)
    new_starts.astype(np.int32).tofile(article_boundaries_filename)

    outputs: np.ndarray = np.memmap(
        output_filename,
        dtype=np.float16,
        mode="w+",
        shape=(new_n_tokens, data.vocabulary_size),
    )

    new_start_of_article: dict[int, int] = {
        int(article): int(new_start)
        for article, new_start in zip(selected, new_starts[:-1])
    }

    total_loss: float = 0.0
    total_predicted_tokens: int = 0

    for (
        articles,
        probabilities,
        loss_sum,
        predicted_tokens,
    ) in _iter_micro_batch_probabilities(
        model=model,
        data=data,
        articles=selected.tolist(),
        description=f"saving transformer outputs for {n_articles} articles",
    ):
        total_loss += loss_sum
        total_predicted_tokens += predicted_tokens

        position: int = 0
        for article in articles:
            start: int = int(article_starts[article])
            end: int = int(article_starts[article + 1])
            length: int = end - start - 1
            new_start: int = new_start_of_article[article]
            outputs[new_start : new_start + length] = probabilities[
                position : position + length
            ]
            position += length

    # as in save_transformer_outputs, the last row of each article gets the
    # ppmd's prior; here its target is the first token of the next *selected*
    # article, and only the last selected article's last row predicts nothing
    total_prior_loss: float = 0.0
    total_prior_predicted_tokens: int = 0
    for index, article in enumerate(selected):
        end = int(article_starts[article + 1])
        last_row: int = int(new_starts[index + 1]) - 1
        outputs[last_row] = data.priors[end - 1]
        if last_row + 1 < new_n_tokens:
            probability = float(data.priors[end - 1, tokens[last_row + 1]])
            total_prior_loss += -log(max(probability, 1e-12))
            total_prior_predicted_tokens += 1

    outputs.flush()

    assert total_predicted_tokens + total_prior_predicted_tokens == new_n_tokens - 1, (
        "every token but the first must be predicted exactly once"
    )

    print(
        f"wrote {n_articles} of {data.n_articles} articles ({new_n_tokens} of "
        f"{data.n_tokens} tokens) to {tokens_filename}, "
        f"{article_boundaries_filename} and {output_filename}"
    )
    if total_predicted_tokens > 0:
        print(
            f"transformer loss: {total_loss / total_predicted_tokens:.4f} nats per "
            f"token (averaged over {total_predicted_tokens} predicted tokens)"
        )
    print(
        f"overall loss with the ppmd's predictions at the remaining tokens: "
        f"{(total_loss + total_prior_loss) / (new_n_tokens - 1):.4f} nats per token "
        f"(averaged over {new_n_tokens - 1} predicted tokens, "
        f"{total_prior_predicted_tokens} of them predicted by the ppmd)"
    )

    if was_training:
        model.train()


# ruff: noqa: F722
