"""Command-line interface for the ProbeKit pipeline."""

import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path

import typer

from probekit.config import (
    DEFAULT_AUTOINTERP_CONFIG,
    DEFAULT_CONFIG,
    DEFAULT_PROBES_CONFIG,
    DEFAULT_SAE_CONFIG,
    load_config,
    load_yaml,
)


class Concept(StrEnum):
    pii = "pii"


class ProbeSpace(StrEnum):
    raw = "raw"
    sae = "sae"


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
    from probekit.harvest.harvester import resolve_device
    from probekit.sae.trainer import SAEConfig, save_json
    from probekit.sae.trainer import train_sae as fit

    cfg = load_config(config)
    settings = load_yaml(sae_config, SAEConfig)
    device = resolve_device(cfg.model)
    version = f"{cfg.model.name}@{cfg.model.revision}"
    report = fit(
        cfg.paths.activations,
        cfg.paths.sae_checkpoint,
        settings,
        cfg.seed,
        device,
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


@app.command()
def autointerp(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    autointerp_config: Path = typer.Option(DEFAULT_AUTOINTERP_CONFIG, "--autointerp-config"),
    n_features: int | None = typer.Option(None, "--n-features", min=1),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Label live features and score held-out detections."""
    import os

    from transformers import AutoTokenizer

    from probekit.autointerp.cache import checkpoint_id
    from probekit.autointerp.collector import AutoInterpConfig
    from probekit.autointerp.explainer import HaikuExplainer, OfflineExplainer
    from probekit.autointerp.pipeline import autointerpret
    from probekit.harvest.harvester import resolve_device
    from probekit.sae.trainer import SAECheckpoint

    cfg = load_config(config)
    settings = load_yaml(autointerp_config, AutoInterpConfig)
    if n_features is not None:
        settings = settings.model_copy(update={"n_features": n_features})
    if not offline and not os.environ.get("ANTHROPIC_API_KEY"):
        raise typer.BadParameter("Set ANTHROPIC_API_KEY or use --offline for a smoke run.")
    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.model.local_dir or cfg.model.name), revision=cfg.model.revision
    )
    checkpoint = SAECheckpoint.load(cfg.paths.sae_checkpoint, resolve_device(cfg.model))
    explainer = OfflineExplainer() if offline else HaikuExplainer(settings.model)
    rows = autointerpret(
        cfg.paths,
        checkpoint,
        checkpoint_id(cfg.paths.sae_checkpoint),
        tokenizer,
        explainer,
        settings,
    )
    typer.echo(f"{len(rows)} labels ({explainer.model}) written to {cfg.paths.feature_db}")


@app.command()
def train_probe(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    probes_config: Path = typer.Option(DEFAULT_PROBES_CONFIG, "--probes-config"),
    concept: Concept = typer.Option(Concept.pii, "--concept"),
    space: ProbeSpace = typer.Option(ProbeSpace.raw, "--space"),
) -> None:
    """Fit one probe and choose C using the validation split."""
    from probekit.probes.comparator import score_splits
    from probekit.probes.comparator import train_probe as fit
    from probekit.probes.datasets import ProbeConfig, build_dataset
    from probekit.sae.trainer import save_json

    cfg = load_config(config)
    settings = load_yaml(probes_config, ProbeConfig)
    data, provenance = build_dataset(
        cfg.model, cfg.paths.sae_checkpoint, settings, cfg.paths.probe_dir
    )
    probe = fit(data, data.indices("train"), settings, space.value, cfg.seed)
    path = cfg.paths.probe_dir / f"{concept.value}-{space.value}.npz"
    probe.save(path)
    report = score_splits(probe, data, space.value, settings)
    save_json(
        cfg.paths.runs / f"probe-{concept.value}-{space.value}.json",
        report | {"provenance": provenance},
    )
    typer.echo(str(path))


@app.command()
def compare(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    probes_config: Path = typer.Option(DEFAULT_PROBES_CONFIG, "--probes-config"),
    concept: Concept = typer.Option(Concept.pii, "--concept"),
) -> None:
    """Compare raw and SAE probes on paired ID and OOD splits."""
    from probekit.probes.comparator import compare as benchmark
    from probekit.probes.datasets import ProbeConfig, build_dataset

    cfg = load_config(config)
    settings = load_yaml(probes_config, ProbeConfig)
    data, provenance = build_dataset(
        cfg.model, cfg.paths.sae_checkpoint, settings, cfg.paths.probe_dir
    )
    benchmark(
        data, settings, cfg.seed, provenance, cfg.paths.probe_dir, cfg.paths.runs, concept.value
    )
    typer.echo(str(cfg.paths.runs / "probe-comparison.md"))
