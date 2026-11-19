"""Command-line interface for the ProbeKit pipeline."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from probekit.config import DEFAULT_CONFIG, DEFAULT_SAE_CONFIG, load_config, load_yaml

app = typer.Typer(no_args_is_help=True, help="Train and deploy GPT-2 probes.")


@app.callback()
def main() -> None:
    """Run an individual pipeline stage."""


@app.command()
def harvest(config: Path = typer.Option(DEFAULT_CONFIG, "--config")) -> None:
    """Harvest GPT-2 activations and token provenance into zstd Parquet."""
    from probekit.harvest.harvester import ActivationHarvester, corpus_sequences, seed_everything

    cfg = load_config(config)
    seed_everything(cfg.seed)
    harvester = ActivationHarvester.from_config(cfg.model)
    summary = harvester.harvest(
        corpus_sequences(cfg.corpus),
        cfg.paths.activations,
        cfg.paths.sequences,
        max_tokens=cfg.corpus.max_tokens,
    )
    typer.echo(json.dumps(asdict(summary)))


@app.command()
def train_sae(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    sae_config: Path = typer.Option(DEFAULT_SAE_CONFIG, "--sae-config"),
) -> None:
    """Train and checkpoint a TopK SAE on corpus activations."""
    import torch

    from probekit.sae.trainer import SAEConfig, save_json
    from probekit.sae.trainer import train_sae as fit

    cfg = load_config(config)
    settings = load_yaml(sae_config, SAEConfig)
    device = cfg.model.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(cfg.model.cpu_threads)
    version = f"{cfg.model.name}@{cfg.model.revision}"
    report = fit(
        cfg.paths.activations,
        cfg.paths.sae_checkpoint,
        settings,
        cfg.seed,
        torch.device(device),
        version,
        cfg.model.layer,
    )
    path = cfg.paths.runs / "sae-training.json"
    save_json(path, report)
    typer.echo(str(path))


@app.command()
def eval_sae(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    sae_config: Path = typer.Option(DEFAULT_SAE_CONFIG, "--sae-config"),
    sae_lens: bool = typer.Option(False, "--sae-lens"),
) -> None:
    """Measure held-out reconstruction and GPT-2 loss recovered."""
    from probekit.harvest import ActivationHarvester
    from probekit.sae.evaluator import evaluate_sae, lens_baseline
    from probekit.sae.trainer import SAECheckpoint, SAEConfig, save_json

    cfg = load_config(config)
    settings = load_yaml(sae_config, SAEConfig)
    harvester = ActivationHarvester.from_config(cfg.model)
    checkpoint = SAECheckpoint.load(cfg.paths.sae_checkpoint, harvester.device)
    baseline = lens_baseline(settings, harvester) if sae_lens else None
    report = evaluate_sae(
        checkpoint,
        harvester,
        cfg.paths.activations,
        cfg.paths.sequences,
        settings,
        baseline=baseline,
    )
    path = cfg.paths.runs / "sae-metrics.json"
    save_json(path, report)
    typer.echo(str(path))
