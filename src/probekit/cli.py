"""Command-line interface for the ProbeKit pipeline."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from probekit.config import DEFAULT_CONFIG, load_config

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
