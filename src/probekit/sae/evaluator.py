"""Held-out reconstruction and GPT-2 loss recovered evaluation."""

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F

from probekit.harvest.harvester import (
    ActivationHarvester,
    SequenceInput,
    TokenizedSequence,
    batches,
)
from probekit.sae.architectures import FIRING_THRESHOLD
from probekit.sae.trainer import (
    SAECheckpoint,
    SAEConfig,
    activation_batches,
    is_heldout,
    validate_activations,
)

SAEForward = Callable[[Tensor], tuple[Tensor, Tensor]]
JBLOOM_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
JBLOOM_REVISION = "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9"


@torch.inference_mode()
def reconstruction_metrics(
    sae: SAEForward,
    matrices: Iterable[Tensor],
    device: torch.device,
) -> dict:
    count = nonzero = 0
    squares = errors = 0.0
    total = alive = None
    for raw in matrices:
        raw = raw.to(device)
        recon, latent = sae(raw)
        count += len(raw)
        raw64 = raw.double()
        summed = raw64.sum(dim=0)
        total = summed if total is None else total + summed
        squares += raw64.square().sum().item()
        errors += (raw64 - recon.double()).square().sum().item()
        nonzero += latent.count_nonzero().item()
        hits = (latent.abs() > FIRING_THRESHOLD).any(dim=0)
        alive = hits if alive is None else alive | hits
    if total is None or alive is None:
        raise ValueError("held-out split is empty")
    variation = squares - total.square().sum().item() / count
    return {
        "tokens": count,
        "l0": nonzero / count,
        "reconstruction_mse": errors / (count * len(total)),
        "fraction_variance_explained": 1 - errors / variation if variation > 0 else None,
        "dead_feature_rate": 1 - alive.sum().item() / len(alive),
    }


def heldout_sequences(
    path: Path,
    seed: int,
    fraction: float,
    limit: int,
) -> Iterator[TokenizedSequence]:
    count = 0
    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            if is_heldout(row["sequence_id"], seed, fraction):
                yield TokenizedSequence(
                    SequenceInput(row["sequence_id"], row["text"]), row["token_ids"]
                )
                count += 1
                if count == limit:
                    return


@torch.inference_mode()
def loss_recovery(
    harvester: ActivationHarvester,
    sae: SAEForward,
    sequences: Iterable[TokenizedSequence],
) -> dict:
    totals = {"clean": 0.0, "spliced": 0.0, "zero": 0.0}
    targets_seen = sequence_count = 0
    for sequences_batch in batches(sequences, harvester.config.batch_size):
        inputs = harvester.padded(sequences_batch)
        mask = inputs["attention_mask"].bool()
        target_mask = mask[:, 1:]
        splice_mask = mask.clone()
        splice_mask[:, 0] = False
        targets = inputs["input_ids"][:, 1:][target_mask]
        targets_seen += targets.numel()
        sequence_count += len(sequences_batch)

        def intervene(_module: nn.Module, _inputs: tuple, output: Tensor | tuple) -> Tensor | tuple:
            hidden = output[0] if isinstance(output, tuple) else output
            updated = hidden.clone()
            updated[splice_mask] = sae(hidden[splice_mask])[0] if mode == "spliced" else 0
            return (updated, *output[1:]) if isinstance(output, tuple) else updated

        with harvester.lock:
            for mode in totals:
                handle = (
                    harvester.block.register_forward_hook(intervene) if mode != "clean" else None
                )
                try:
                    logits = harvester.model(**inputs, use_cache=False).logits[:, :-1]
                    totals[mode] += F.cross_entropy(
                        logits[target_mask], targets, reduction="sum"
                    ).item()
                finally:
                    if handle is not None:
                        handle.remove()
    if not targets_seen:
        raise ValueError("no held-out targets for LM loss")
    losses = {name: value / targets_seen for name, value in totals.items()}
    denominator = losses["zero"] - losses["clean"]
    recovered = 1 - (losses["spliced"] - losses["clean"]) / denominator if denominator > 0 else None
    return {
        "lm_loss": losses,
        "loss_recovered": recovered,
        "lm_targets": targets_seen,
        "lm_sequences": sequence_count,
    }


def lens_baseline(config: SAEConfig, harvester: ActivationHarvester) -> SAEForward:
    expected = f"blocks.{harvester.config.layer + 1}.hook_resid_pre"
    if config.baseline_release == "gpt2-small-res-jb" and config.baseline_id != expected:
        raise ValueError(f"Use {expected} to match the captured block output")
    if config.baseline_release == "gpt2-small-res-jb":
        return jbloom_baseline(config, harvester)
    try:
        from sae_lens import SAE
    except ImportError as error:
        raise RuntimeError(
            "Install the optional sae-lens package to evaluate the baseline"
        ) from error
    loaded = SAE.from_pretrained(
        release=config.baseline_release, sae_id=config.baseline_id, device=str(harvester.device)
    )
    baseline = loaded[0] if isinstance(loaded, tuple) else loaded
    baseline.eval()

    def forward(x: Tensor) -> tuple[Tensor, Tensor]:
        latents = baseline.encode(x)
        return baseline.decode(latents), latents

    return forward


def jbloom_baseline(config: SAEConfig, harvester: ActivationHarvester) -> SAEForward:
    def asset(name: str) -> Path:
        if config.baseline_local_dir is not None:
            return config.baseline_local_dir / name
        return Path(
            hf_hub_download(
                JBLOOM_REPO,
                f"{config.baseline_id}/{name}",
                revision=JBLOOM_REVISION,
            )
        )

    values = json.loads(asset("cfg.json").read_text(encoding="utf-8"))
    if (
        values["hook_point"] != config.baseline_id
        or values["d_in"] != harvester.width
        or values.get("normalize_activations", "none") != "none"
    ):
        raise ValueError("unsupported JBloom SAE hook, input width, or normalization")
    weights = load_file(asset("sae_weights.safetensors"), device=str(harvester.device))

    def forward(x: Tensor) -> tuple[Tensor, Tensor]:
        latents = ((x - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"]).relu()
        return latents @ weights["W_dec"] + weights["b_dec"], latents

    return forward


def evaluate_sae(
    checkpoint: SAECheckpoint,
    harvester: ActivationHarvester,
    activation_path: Path,
    sequence_path: Path,
    config: SAEConfig,
    *,
    baseline: SAEForward | None = None,
) -> dict:
    if (
        checkpoint.model_version != harvester.model_version
        or checkpoint.layer != harvester.config.layer
    ):
        raise ValueError("SAE checkpoint and GPT-2 extraction config do not match")

    validate_activations(activation_path, checkpoint.model_version, checkpoint.layer)

    def evaluate(sae: SAEForward) -> dict:
        activations = activation_batches(
            activation_path,
            config.batch_size,
            checkpoint.seed,
            checkpoint.holdout_fraction,
            heldout=True,
        )
        sequences = heldout_sequences(
            sequence_path, checkpoint.seed, checkpoint.holdout_fraction, config.eval_sequences
        )
        return reconstruction_metrics(sae, activations, harvester.device) | loss_recovery(
            harvester, sae, sequences
        )

    report = {
        "model_version": checkpoint.model_version,
        "layer": checkpoint.layer,
        "device": str(harvester.device),
        "torch_version": str(torch.__version__),
        "seed": checkpoint.seed,
        "holdout_fraction": checkpoint.holdout_fraction,
        "steps": checkpoint.step,
        "dict_size": checkpoint.sae.dict_size,
        "k": checkpoint.sae.k,
        "topk": evaluate(checkpoint.sae),
    }
    if baseline is not None:
        report["sae_lens"] = evaluate(baseline) | {
            "release": config.baseline_release,
            "id": config.baseline_id,
            "weights_revision": JBLOOM_REVISION
            if config.baseline_release == "gpt2-small-res-jb"
            else None,
        }
    return report
