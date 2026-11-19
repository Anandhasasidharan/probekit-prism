"""Batch GPT-2 forwards and stream residual activations to Parquet."""

import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerBase

from probekit.config import CorpusConfig, ModelConfig


@dataclass(frozen=True)
class SequenceInput:
    sequence_id: str
    text: str


@dataclass(frozen=True)
class TokenizedSequence:
    source: SequenceInput
    token_ids: list[int]


@dataclass(frozen=True)
class HarvestSummary:
    sequences: int
    input_tokens: int
    activation_rows: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


BatchItem = TypeVar("BatchItem")


def batches(items: Iterable[BatchItem], size: int) -> Iterator[list[BatchItem]]:
    items = iter(items)
    while batch := list(islice(items, size)):
        yield batch


def resolve_device(config: ModelConfig) -> torch.device:
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(config.cpu_threads)
    return torch.device(device)


class ActivationHarvester:
    """Capture the output of one zero-based GPT-2 decoder block."""

    def __init__(
        self,
        config: ModelConfig,
        model: GPT2LMHeadModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        if config.layer >= len(model.transformer.h):
            raise ValueError(f"Model has no block {config.layer}")
        self.config = config
        self.device = resolve_device(config)
        self.model = model.to(self.device).eval()
        self.block = self.model.transformer.h[config.layer]
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.width = int(self.model.config.n_embd)
        self.model_version = f"{config.name}@{config.revision}"
        self.lock = Lock()

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ActivationHarvester":
        location = str(config.local_dir or config.name)
        model = GPT2LMHeadModel.from_pretrained(
            location, revision=config.revision, use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(location, revision=config.revision)
        return cls(config, model, tokenizer)

    def tokenize(self, batch: Sequence[SequenceInput]) -> list[TokenizedSequence]:
        token_ids = self.tokenizer(
            [item.text for item in batch],
            truncation=True,
            max_length=min(self.config.max_length, self.model.config.n_positions),
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        return [TokenizedSequence(item, ids) for item, ids in zip(batch, token_ids, strict=True)]

    def padded(self, sequences: Sequence[TokenizedSequence]) -> dict[str, Tensor]:
        batch = self.tokenizer.pad(
            {"input_ids": [seq.token_ids for seq in sequences]},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in batch.items()}

    @torch.inference_mode()
    def capture(self, sequences: Sequence[TokenizedSequence]) -> list[Tensor]:
        """Return one CPU float32 matrix per sequence, excluding its first position."""
        hidden: Tensor | None = None

        def hook(_module: nn.Module, _inputs: tuple, output: Tensor | tuple) -> None:
            nonlocal hidden
            hidden = output[0] if isinstance(output, tuple) else output

        inputs = self.padded(sequences)
        with self.lock:
            handle = self.block.register_forward_hook(hook)
            try:
                self.model.transformer(**inputs, use_cache=False)
            finally:
                handle.remove()
        if hidden is None:
            raise RuntimeError("GPT-2 hook did not capture a residual stream")
        hidden = hidden.to(device="cpu", dtype=torch.float32)
        return [hidden[i, 1 : len(seq.token_ids)] for i, seq in enumerate(sequences)]

    def _token_batches(
        self, sequences: Iterable[SequenceInput], max_tokens: int | None
    ) -> Iterator[list[TokenizedSequence]]:
        remaining = max_tokens
        for batch in batches(sequences, self.config.batch_size):
            kept = []
            for seq in self.tokenize(batch):
                ids = seq.token_ids if remaining is None else seq.token_ids[:remaining]
                if len(ids) >= 2:
                    kept.append(TokenizedSequence(seq.source, ids))
                    if remaining is not None:
                        remaining -= len(ids)
            if kept:
                yield kept
            if remaining is not None and remaining < 2:
                break

    def harvest(
        self,
        sequences: Iterable[SequenceInput],
        activation_path: Path,
        sequence_path: Path,
        max_tokens: int | None = None,
    ) -> HarvestSummary:
        """Stream zstd Parquet, publishing outputs only after successful forwards."""
        if activation_path.resolve() == sequence_path.resolve():
            raise ValueError("activation and sequence paths must differ")
        activation_schema = pa.schema(
            [
                ("sequence_id", pa.string()),
                ("token_pos", pa.int32()),
                ("layer", pa.int16()),
                ("model_version", pa.string()),
                ("activation", pa.list_(pa.float32(), self.width)),
            ]
        )
        sidecar_schema = pa.schema(
            [
                ("sequence_id", pa.string()),
                ("text", pa.string()),
                ("token_ids", pa.list_(pa.int32())),
            ]
        )
        activation_path.parent.mkdir(parents=True, exist_ok=True)
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
        sequences_seen: set[str] = set()
        input_tokens = 0
        with (
            TemporaryDirectory(dir=activation_path.parent) as activation_tmp,
            TemporaryDirectory(dir=sequence_path.parent) as sequence_tmp,
        ):
            activation_file = Path(activation_tmp) / "activations.parquet"
            sequence_file = Path(sequence_tmp) / "sequences.parquet"
            with (
                pq.ParquetWriter(activation_file, activation_schema, compression="zstd") as acts,
                pq.ParquetWriter(sequence_file, sidecar_schema, compression="zstd") as sidecar,
            ):
                for batch in self._token_batches(sequences, max_tokens):
                    for seq in batch:
                        if seq.source.sequence_id in sequences_seen:
                            raise ValueError(f"duplicate sequence id: {seq.source.sequence_id}")
                        sequences_seen.add(seq.source.sequence_id)
                        input_tokens += len(seq.token_ids)
                    acts.write_table(self._activation_table(batch, activation_schema))
                    sidecar.write_table(
                        pa.Table.from_pylist(
                            [
                                {
                                    "sequence_id": seq.source.sequence_id,
                                    "text": seq.source.text,
                                    "token_ids": seq.token_ids,
                                }
                                for seq in batch
                            ],
                            schema=sidecar_schema,
                        )
                    )
            if not sequences_seen:
                raise ValueError("Corpus contains no sequences with at least two usable tokens")
            sequence_file.replace(sequence_path)
            activation_file.replace(activation_path)
        return HarvestSummary(len(sequences_seen), input_tokens, input_tokens - len(sequences_seen))

    def _activation_table(self, batch: list[TokenizedSequence], schema: pa.Schema) -> pa.Table:
        sequence_ids, positions = [], []
        for seq in batch:
            positions.extend(range(1, len(seq.token_ids)))
            sequence_ids.extend([seq.source.sequence_id] * (len(seq.token_ids) - 1))
        count = len(positions)
        vectors = torch.cat(self.capture(batch)).numpy()
        flattened = pa.array(vectors.reshape(-1), type=pa.float32())
        return pa.Table.from_arrays(
            [
                pa.array(sequence_ids, type=pa.string()),
                pa.array(positions, type=pa.int32()),
                pa.array([self.config.layer] * count, type=pa.int16()),
                pa.array([self.model_version] * count, type=pa.string()),
                pa.FixedSizeListArray.from_arrays(flattened, self.width),
            ],
            schema=schema,
        )


def corpus_sequences(config: CorpusConfig) -> Iterator[SequenceInput]:
    """Stream either a local JSONL file or a Hugging Face dataset."""
    if config.jsonl is not None:
        with config.jsonl.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    row = json.loads(line)
                    yield SequenceInput(str(row.get("sequence_id", index)), row[config.text_field])
        return

    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset, config.subset, split=config.split, revision=config.revision, streaming=True
    )
    for index, row in enumerate(dataset):
        yield SequenceInput(str(index), row[config.text_field])
