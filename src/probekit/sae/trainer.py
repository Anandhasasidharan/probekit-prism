"""Train and checkpoint a TopK SAE from streamed Arrow batches."""

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

import pyarrow.parquet as pq
import torch
from pydantic import Field, model_validator
from torch import Tensor
from torch.nn import functional as F

from probekit.config import Settings
from probekit.harvest.harvester import seed_everything
from probekit.sae.architectures import FIRING_THRESHOLD, TopKSAE


class SAEConfig(Settings):
    dict_size: int = Field(default=16384, gt=0)
    k: int = Field(default=32, gt=0)
    target_mean_norm: float = Field(default=16.0, gt=0, allow_inf_nan=False)
    learning_rate: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=256, gt=0)
    epochs: int = Field(default=1, gt=0)
    checkpoint_every: int = Field(default=2000, ge=0)
    holdout_fraction: float = Field(default=0.15, gt=0, lt=1)
    eval_sequences: int = Field(default=64, gt=0)
    dead_after_steps: int = Field(default=128, gt=0)
    aux_weight: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    baseline_release: str = "gpt2-small-res-jb"
    baseline_local_dir: Path | None = None
    baseline_id: str = "blocks.9.hook_resid_pre"

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        if self.k > self.dict_size:
            raise ValueError("k cannot exceed dict_size")
        return self


@lru_cache(maxsize=65536)
def is_heldout(sequence_id: str, seed: int, fraction: float) -> bool:
    digest = hashlib.blake2b(f"{seed}:{sequence_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") < fraction * (1 << 64)


def validate_activations(path: Path, model_version: str, layer: int) -> None:
    file = pq.ParquetFile(path)
    for batch in file.iter_batches(columns=["layer", "model_version"]):
        if set(batch.column(0).unique().to_pylist()) != {layer} or set(
            batch.column(1).unique().to_pylist()
        ) != {model_version}:
            raise ValueError("activation model version or layer differs from configuration")


def activation_batches(
    path: Path, batch_size: int, seed: int, fraction: float, *, heldout: bool
) -> Iterator[Tensor]:
    file = pq.ParquetFile(path)
    width = file.schema_arrow.field("activation").type.list_size
    for batch in file.iter_batches(batch_size=batch_size, columns=["sequence_id", "activation"]):
        mask = [
            is_heldout(seq_id, seed, fraction) == heldout for seq_id in batch.column(0).to_pylist()
        ]
        if any(mask):
            matrix = batch.column(1).flatten().to_numpy().reshape(-1, width)
            yield torch.from_numpy(matrix[mask])


@dataclass
class SAECheckpoint:
    sae: TopKSAE
    seed: int
    step: int
    model_version: str
    layer: int
    holdout_fraction: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "format": 1,
            "d_model": self.sae.d_model,
            "dict_size": self.sae.dict_size,
            "k": self.sae.k,
            "state": self.sae.state_dict(),
            "seed": self.seed,
            "step": self.step,
            "model_version": self.model_version,
            "layer": self.layer,
            "holdout_fraction": self.holdout_fraction,
        }
        with TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / path.name
            torch.save(values, temporary)
            temporary.replace(path)

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> "SAECheckpoint":
        values = torch.load(path, map_location="cpu", weights_only=True)
        if values["format"] != 1:
            raise ValueError("unsupported SAE checkpoint format")
        # Layer initialization should not perturb the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            sae = TopKSAE(values["d_model"], values["dict_size"], values["k"])
        sae.load_state_dict(values["state"])
        return cls(
            sae.to(device).eval(),
            values["seed"],
            values["step"],
            values["model_version"],
            values["layer"],
            values["holdout_fraction"],
        )


def fit_scale(batches: Iterator[Tensor], target: float) -> tuple[float, int]:
    norms = 0.0
    count = 0
    for batch in batches:
        norms += batch.norm(dim=1).sum(dtype=torch.float64).item()
        count += len(batch)
    if not count or norms <= 0 or not math.isfinite(norms):
        raise ValueError("training split must contain finite, nonzero activations")
    return target * count / norms, count


def revive_loss(sae: TopKSAE, pre: Tensor, residual: Tensor, dead: Tensor) -> Tensor:
    count = min(sae.k, int(dead.sum().item()))
    if not count:
        return pre.new_zeros(())
    values, indices = pre.masked_fill(~dead, -torch.inf).topk(count, sorted=False)
    projected = (sae.decoder.weight.T[indices] * values.relu().unsqueeze(-1)).sum(dim=1)
    return F.mse_loss(projected, residual.detach() * sae.scale) / sae.scale.square()


def train_sae(
    path: Path,
    checkpoint_path: Path,
    config: SAEConfig,
    seed: int,
    device: torch.device,
    model_version: str,
    layer: int,
) -> dict:
    seed_everything(seed)
    validate_activations(path, model_version, layer)
    width = pq.ParquetFile(path).schema_arrow.field("activation").type.list_size

    def training_batches() -> Iterator[Tensor]:
        return activation_batches(
            path, config.batch_size, seed, config.holdout_fraction, heldout=False
        )

    scale, count = fit_scale(training_batches(), config.target_mean_norm)
    sae = TopKSAE(width, config.dict_size, config.k, scale).to(device)
    checkpoint = SAECheckpoint(sae, seed, 0, model_version, layer, config.holdout_fraction)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    last_seen = torch.zeros(config.dict_size, dtype=torch.long, device=device)
    report = {"training_tokens": count, "scale": scale, "epochs": []}
    for epoch in range(config.epochs):
        squared_error = 0.0
        for raw in training_batches():
            raw = raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            pre = sae.preactivations(raw)
            latents = sae.select(pre)
            recon = sae.decode(latents)
            mse = F.mse_loss(recon, raw)
            checkpoint.step += 1
            last_seen.masked_fill_((latents.detach() > FIRING_THRESHOLD).any(0), checkpoint.step)
            dead = checkpoint.step - last_seen >= config.dead_after_steps
            loss = mse
            if config.aux_weight:
                loss = loss + config.aux_weight * revive_loss(sae, pre, raw - recon, dead)
            loss.backward()
            optimizer.step()
            sae.renorm_decoder()
            squared_error += mse.item() * len(raw)
            if config.checkpoint_every and checkpoint.step % config.checkpoint_every == 0:
                name = f"{checkpoint_path.stem}.step-{checkpoint.step}{checkpoint_path.suffix}"
                checkpoint.save(checkpoint_path.with_name(name))
        report["epochs"].append(
            {
                "epoch": epoch + 1,
                "online_mse": squared_error / count,
                "ever_dead_rate": last_seen.eq(0).float().mean().item(),
            }
        )
    checkpoint.save(checkpoint_path)
    return report | {"steps": checkpoint.step, "seed": seed, "config": config.model_dump()}


def save_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
