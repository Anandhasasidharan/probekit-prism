"""Collect feature contexts, label them, and produce cached detection reports."""

import csv
from dataclasses import asdict
from functools import partial
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from probekit.autointerp.cache import LabelCache
from probekit.autointerp.collector import AutoInterpConfig, TokenExample, collect_examples
from probekit.autointerp.explainer import Explainer
from probekit.autointerp.scorer import score_detection
from probekit.config import PathsConfig
from probekit.sae.trainer import SAECheckpoint


def autointerpret(
    paths: PathsConfig,
    checkpoint: SAECheckpoint,
    sae_id: str,
    tokenizer: PreTrainedTokenizerBase,
    explainer: Explainer,
    config: AutoInterpConfig,
) -> list[dict]:
    samples = collect_examples(
        paths.activations, paths.sequences, checkpoint.sae, checkpoint.seed, config
    )
    cache = LabelCache(paths.feature_db, sae_id, explainer.model, config.prompt_version)

    def context(example: TokenExample) -> str:
        return example.context(tokenizer, config.context_tokens)

    rows = []
    for feature_id, sample in samples.items():
        examples = [(context(item), item.activation) for item in sample.top + sample.quantile]
        description = cache.label(feature_id, partial(explainer.explain, feature_id, examples))
        score = cache.score(
            feature_id,
            partial(
                score_detection,
                explainer,
                positives=[context(item) for item in sample.positive],
                negatives=[context(item) for item in sample.negative],
                seed=checkpoint.seed + feature_id,
            ),
        )
        rows.append(
            {
                "sae_id": sae_id,
                "feature_id": feature_id,
                "model": explainer.model,
                "prompt_version": config.prompt_version,
                "description": description,
                **asdict(score),
            }
        )
    write_reports(paths.runs, rows)
    return rows


def write_reports(directory: Path, rows: list[dict]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "autointerp.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = [row["auroc"] for row in rows if row["auroc"] is not None]
    figure = Figure(figsize=(6, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.hist(scores, bins=10, range=(0, 1))
    axes.set(xlabel="Held-out detection AUROC", ylabel="Features", title="AutoInterp scores")
    figure.savefig(directory / "autointerp-scores.png")
