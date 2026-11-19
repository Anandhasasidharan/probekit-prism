"""Bounded sampling of feature activations from aligned Parquet row groups."""

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from pydantic import Field
from transformers import PreTrainedTokenizerBase

from probekit.config import Settings
from probekit.sae.architectures import FIRING_THRESHOLD, TopKSAE
from probekit.sae.trainer import is_heldout


class AutoInterpConfig(Settings):
    model: str = "claude-haiku-4-5-20251001"
    prompt_version: str = "v1"
    n_features: int = Field(default=100, gt=0)
    top_examples: int = Field(default=50, gt=0)
    reservoir_size: int = Field(default=2048, ge=4)
    score_examples_per_class: int = Field(default=20, gt=0)
    holdout_fraction: float = Field(default=0.25, gt=0, lt=1)
    context_tokens: int = Field(default=20, gt=0)


@dataclass(frozen=True)
class TokenExample:
    sequence_id: str
    token_pos: int
    token_ids: list[int]
    activation: float

    def context(self, tokenizer: PreTrainedTokenizerBase, window: int) -> str:
        pos = self.token_pos
        start = max(0, pos - window)
        left = tokenizer.decode(self.token_ids[start:pos])
        target = tokenizer.decode(self.token_ids[pos : pos + 1])
        right = tokenizer.decode(self.token_ids[pos + 1 : pos + 1 + window])
        return f"{left} [TARGET {target!r}] {right}"


class SampleHeap:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.entries: list[tuple[float, int, TokenExample]] = []
        self.counter = 0

    def add(
        self,
        priorities: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        group: "ActivationGroup",
    ) -> None:
        if len(indices) > self.limit:
            keep = np.argpartition(priorities, -self.limit)[-self.limit :]
            priorities, indices = priorities[keep], indices[keep]
        for priority, index in zip(priorities, indices, strict=True):
            if len(self.entries) == self.limit and priority <= self.entries[0][0]:
                continue
            index = int(index)
            sequence_id = group.sequence_ids[index]
            example = TokenExample(
                sequence_id,
                int(group.positions[index]),
                group.tokens[sequence_id],
                float(values[index]),
            )
            entry = (float(priority), self.counter, example)
            self.counter += 1
            if len(self.entries) < self.limit:
                heapq.heappush(self.entries, entry)
            else:
                heapq.heapreplace(self.entries, entry)

    def examples(self) -> list[TokenExample]:
        return [entry[2] for entry in sorted(self.entries, reverse=True)]


@dataclass
class FeatureSamples:
    top: list[TokenExample] = field(default_factory=list)
    quantile: list[TokenExample] = field(default_factory=list)
    positive: list[TokenExample] = field(default_factory=list)
    negative: list[TokenExample] = field(default_factory=list)


@dataclass
class ActivationGroup:
    sequence_ids: list[str]
    positions: np.ndarray
    tokens: dict[str, list[int]]
    latents: np.ndarray
    heldout: np.ndarray


def activation_groups(
    activation_path: Path,
    sequence_path: Path,
    sae: TopKSAE,
    seed: int,
    holdout: float,
    feature_ids: list[int] | None = None,
) -> Iterator[ActivationGroup]:
    acts, sidecar = pq.ParquetFile(activation_path), pq.ParquetFile(sequence_path)
    if acts.num_row_groups != sidecar.num_row_groups:
        raise ValueError("activation Parquet and sequence sidecar have different row groups")
    device = sae.encoder.weight.device
    for number in range(acts.num_row_groups):
        table = acts.read_row_group(number, columns=["sequence_id", "token_pos", "activation"])
        tokens = {
            row["sequence_id"]: row["token_ids"]
            for row in sidecar.read_row_group(
                number, columns=["sequence_id", "token_ids"]
            ).to_pylist()
        }
        ids = table["sequence_id"].to_pylist()
        if set(ids) != tokens.keys():
            raise ValueError("activation Parquet and sequence sidecar have different sequence IDs")
        raw = table["activation"].combine_chunks().flatten().to_numpy()
        with torch.inference_mode():
            inputs = torch.tensor(raw.reshape(-1, sae.d_model), device=device)
            latents = sae.encode(inputs)
            if feature_ids is not None:
                latents = latents[:, feature_ids]
            features = latents.cpu().numpy()
        mask = np.asarray([is_heldout(seq, seed, holdout) for seq in ids])
        yield ActivationGroup(ids, table["token_pos"].to_numpy(), tokens, features, mask)


def collect_examples(
    activation_path: Path,
    sequence_path: Path,
    sae: TopKSAE,
    seed: int,
    cfg: AutoInterpConfig,
) -> dict[int, FeatureSamples]:
    rng = np.random.default_rng(seed)
    live = np.zeros(sae.dict_size, dtype=bool)
    for group in activation_groups(activation_path, sequence_path, sae, seed, cfg.holdout_fraction):
        live |= (group.latents[~group.heldout] > FIRING_THRESHOLD).any(axis=0)
    live_ids = np.flatnonzero(live)
    if not len(live_ids):
        raise ValueError("no live features found in the explanation split")
    sampled_ids = sorted(
        rng.choice(live_ids, min(cfg.n_features, len(live_ids)), replace=False).tolist()
    )
    heaps = {
        feature_id: (
            SampleHeap(cfg.top_examples),
            SampleHeap(cfg.reservoir_size),
            SampleHeap(cfg.score_examples_per_class),
            SampleHeap(cfg.score_examples_per_class),
        )
        for feature_id in sampled_ids
    }

    for group in activation_groups(
        activation_path, sequence_path, sae, seed, cfg.holdout_fraction, sampled_ids
    ):
        for column, feature_id in enumerate(sampled_ids):
            values = group.latents[:, column]
            fired = values > FIRING_THRESHOLD
            train_pos = np.flatnonzero(fired & ~group.heldout)
            test_pos = np.flatnonzero(fired & group.heldout)
            test_neg = np.flatnonzero(~fired & group.heldout)
            top, reservoir, positive, negative = heaps[feature_id]
            top.add(values[train_pos], train_pos, values, group)
            for heap, indices in (
                (reservoir, train_pos),
                (positive, test_pos),
                (negative, test_neg),
            ):
                heap.add(rng.random(len(indices)), indices, values, group)

    return {feature_id: summarize(*sample_heaps) for feature_id, sample_heaps in heaps.items()}


def summarize(
    top: SampleHeap,
    reservoir: SampleHeap,
    positive: SampleHeap,
    negative: SampleHeap,
) -> FeatureSamples:
    ordered = sorted(reservoir.examples(), key=lambda item: item.activation)
    indices = sorted({round((len(ordered) - 1) * q) for q in (0.25, 0.5, 0.75)})
    pos, neg = positive.examples(), negative.examples()
    n = min(len(pos), len(neg))
    quantile_examples = list(map(ordered.__getitem__, indices))
    return FeatureSamples(top.examples(), quantile_examples, pos[:n], neg[:n])
