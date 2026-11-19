"""Labeled sequence splits and matched per-sequence feature extraction."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

import numpy as np
import torch
from pydantic import Field, model_validator
from torch import Tensor

from probekit.config import ModelConfig, Settings
from probekit.harvest.harvester import ActivationHarvester, SequenceInput, batches
from probekit.sae.trainer import SAECheckpoint

Space = Literal["raw", "sae"]
Pooling = Literal["mean", "max", "mean_max"]
Split = Literal["train", "val", "test", "ood"]
SPLITS = ("train", "val", "test", "ood")


class ProbeConfig(Settings):
    dataset: Path
    pooling: Pooling = "mean_max"
    c_values: list[float] = Field(min_length=1)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    sample_sizes: list[int] = Field(default_factory=lambda: [50, 200, 1000])
    sample_seeds: list[int] = Field(default_factory=lambda: [101, 102, 103, 104, 105], min_length=1)
    solver_max_iter: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def check_grid(self) -> Self:
        if any(not np.isfinite(c) or c <= 0 for c in self.c_values):
            raise ValueError("C must be positive and finite")
        if any(size < 2 for size in self.sample_sizes) or len(set(self.sample_sizes)) != len(
            self.sample_sizes
        ):
            raise ValueError("sample sizes must be unique and at least 2")
        if len(set(self.sample_seeds)) != len(self.sample_seeds) or any(
            not 0 <= seed < 2**32 for seed in self.sample_seeds
        ):
            raise ValueError("sample seeds must be unique unsigned 32-bit integers")
        return self


class LabeledSequence(Settings):
    sequence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal[0, 1]
    split: Split
    template_id: str = Field(min_length=1)
    kind: str


def read_dataset(path: Path) -> list[LabeledSequence]:
    with path.open(encoding="utf-8") as handle:
        rows = [LabeledSequence.model_validate_json(line) for line in handle if line.strip()]
    validate_splits(rows)
    return rows


def validate_splits(rows: list[LabeledSequence]) -> None:
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    templates: dict[str, set[str]] = {split: set() for split in SPLITS}
    labels: dict[str, set[int]] = {split: set() for split in SPLITS}
    for row in rows:
        text = " ".join(row.text.casefold().split())
        if row.sequence_id in seen_ids or text in seen_text:
            raise ValueError("duplicate sequence ID or normalized text in probe dataset")
        seen_ids.add(row.sequence_id)
        seen_text.add(text)
        templates[row.split].add(row.template_id)
        labels[row.split].add(row.label)
    if any(targets != {0, 1} for targets in labels.values()):
        raise ValueError("each split must contain both classes")
    if templates["ood"] & (templates["train"] | templates["val"] | templates["test"]):
        raise ValueError("OOD templates overlap the in-distribution dataset")


def pool_tokens(tokens: Tensor, mode: Pooling) -> Tensor:
    if mode == "mean":
        return tokens.mean(dim=0)
    if mode == "max":
        return tokens.amax(dim=0)
    return torch.cat([tokens.mean(dim=0), tokens.amax(dim=0)])


@dataclass
class ProbeDataset:
    rows: list[LabeledSequence]
    raw: np.ndarray
    sae: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([row.label for row in self.rows], dtype=np.int64)

    def indices(self, split: Split) -> np.ndarray:
        return np.flatnonzero([row.split == split for row in self.rows])

    def features(self, space: Space) -> np.ndarray:
        return self.raw if space == "raw" else self.sae


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_dataset(
    model: ModelConfig,
    checkpoint_path: Path,
    settings: ProbeConfig,
    cache_directory: Path,
) -> tuple[ProbeDataset, dict]:
    rows = read_dataset(settings.dataset)
    provenance = {
        "dataset_sha256": file_sha256(settings.dataset),
        "sae_id": file_sha256(checkpoint_path),
        "model": model.model_dump(mode="json", exclude={"local_dir"}),
        "pooling": settings.pooling,
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = cache_directory / f"features-{key}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as arrays:
            return ProbeDataset(rows, arrays["raw"], arrays["sae"]), provenance

    harvester = ActivationHarvester.from_config(model)
    checkpoint = SAECheckpoint.load(checkpoint_path, harvester.device)
    if (checkpoint.model_version, checkpoint.layer, checkpoint.sae.d_model) != (
        harvester.model_version,
        model.layer,
        harvester.width,
    ):
        raise ValueError("SAE checkpoint and GPT-2 extraction config do not match")
    result = featurize(rows, harvester, checkpoint, settings.pooling)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache.parent) as temporary:
        staged = Path(temporary) / cache.name
        np.savez(staged, raw=result.raw, sae=result.sae)
        staged.replace(cache)
    return result, provenance


@torch.inference_mode()
def featurize(
    rows: list[LabeledSequence],
    harvester: ActivationHarvester,
    checkpoint: SAECheckpoint,
    pooling: Pooling,
) -> ProbeDataset:
    raw_vectors, sae_vectors = [], []
    for subset in batches(rows, harvester.config.batch_size):
        sequences = harvester.tokenize([SequenceInput(row.sequence_id, row.text) for row in subset])
        if any(len(seq.token_ids) < 2 for seq in sequences):
            raise ValueError("probe examples must have at least two usable input tokens")
        residuals = harvester.capture(sequences)
        lengths = [len(x) for x in residuals]
        encoded = checkpoint.sae.encode(torch.cat(residuals).to(harvester.device)).cpu()
        raw_vectors.extend(pool_tokens(x, pooling) for x in residuals)
        sae_vectors.extend(pool_tokens(x, pooling) for x in encoded.split(lengths))
    return ProbeDataset(rows, torch.stack(raw_vectors).numpy(), torch.stack(sae_vectors).numpy())
