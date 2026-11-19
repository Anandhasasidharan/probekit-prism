"""Paired raw/SAE evaluation on in-distribution and OOD probe splits."""

from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from probekit.probes.datasets import ProbeConfig, ProbeDataset, Space, Split
from probekit.probes.raw_trainer import LinearProbe, train_raw_probe
from probekit.probes.sae_trainer import train_sae_probe
from probekit.sae.trainer import save_json

SPACES: tuple[Space, ...] = ("raw", "sae")
TEST_SPLITS: tuple[Split, ...] = ("test", "ood")
METRICS = ("f1", "precision", "recall", "auroc")


def train_probe(
    data: ProbeDataset,
    train: np.ndarray,
    config: ProbeConfig,
    space: Space,
    seed: int,
) -> LinearProbe:
    val = data.indices("val")
    if space == "sae":
        return train_sae_probe(data, train, val, config, seed)
    targets = data.labels
    return train_raw_probe(
        data.raw[train], targets[train], data.raw[val], targets[val], config, seed
    )


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def score_splits(
    probe: LinearProbe,
    data: ProbeDataset,
    space: Space,
    config: ProbeConfig,
) -> dict:
    features, targets = data.features(space), data.labels
    result = {}
    with threadpool_limits(limits=2):
        for split in TEST_SPLITS:
            selection = data.indices(split)
            result[split] = metrics(
                targets[selection], probe.predict(features[selection]), config.threshold
            )
    return {
        "selected_c": probe.c,
        "validation_auroc": probe.validation_auroc,
        "nonzero_weights": int(np.count_nonzero(probe.weights)),
        "total_weights": int(len(probe.weights)),
        "splits": result,
    }


def stratified_subset(
    train: np.ndarray,
    labels: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    if size == len(train):
        return train
    chosen, _ = train_test_split(train, train_size=size, random_state=seed, stratify=labels[train])
    return chosen


def compare(
    data: ProbeDataset,
    config: ProbeConfig,
    seed: int,
    provenance: dict,
    probe_directory: Path,
    report_directory: Path,
    concept: str,
) -> dict:
    train = data.indices("train")
    if any(size > len(train) for size in config.sample_sizes):
        raise ValueError("sample sizes exceed the number of training sequences")
    report: dict = {
        "concept": concept,
        "seed": seed,
        "provenance": provenance,
        "probe_config": config.model_dump(mode="json", exclude={"dataset"}),
        "sklearn_version": sklearn.__version__,
        "split_sizes": {
            split: len(data.indices(split)) for split in ("train", "val", "test", "ood")
        },
        "full_training": {},
        "sample_efficiency": {},
    }
    for space in SPACES:
        probe = train_probe(data, train, config, space, seed)
        probe.save(probe_directory / f"{concept}-{space}.npz")
        report["full_training"][space] = score_splits(probe, data, space, config)

    for size in config.sample_sizes:
        records = {space: [] for space in SPACES}
        for subset_seed in config.sample_seeds:
            subset = stratified_subset(train, data.labels, size, subset_seed)
            digest = subset_digest(data, subset)
            for space in SPACES:
                fit = train_probe(data, subset, config, space, subset_seed)
                scores = score_splits(fit, data, space, config)
                records[space].append({"seed": subset_seed, "subset_sha256": digest, **scores})
        report["sample_efficiency"][str(size)] = {
            space: {"runs": values, "summary": summarize(values)}
            for space, values in records.items()
        }
    report["sae_minus_raw"] = {
        split: {
            key: report["full_training"]["sae"]["splits"][split][key]
            - report["full_training"]["raw"]["splits"][split][key]
            for key in METRICS
        }
        for split in TEST_SPLITS
    }
    save_json(report_directory / "probe-comparison.json", report)
    (report_directory / "probe-comparison.md").write_text(markdown(report), encoding="utf-8")
    if config.sample_sizes:
        plot_curve(report, report_directory / "probe-sample-efficiency.png")
    return report


def subset_digest(data: ProbeDataset, subset: np.ndarray) -> str:
    import hashlib

    sequence_ids = sorted(data.rows[index].sequence_id for index in subset)
    return hashlib.sha256("\n".join(sequence_ids).encode()).hexdigest()


def summarize(runs: list[dict]) -> dict:
    result = {}
    for split in TEST_SPLITS:
        result[split] = {}
        for key in METRICS:
            values = [run["splits"][split][key] for run in runs]
            result[split][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return result


def markdown(report: dict) -> str:
    lines = [
        "# Raw vs SAE probes",
        "",
        "| Split | Space | F1 | Precision | Recall | AUROC | Nonzero weights |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in TEST_SPLITS:
        for space in SPACES:
            entry = report["full_training"][space]
            values = " | ".join(f"{entry['splits'][split][key]:.4f}" for key in METRICS)
            lines.append(f"| {split} | {space} | {values} | {entry['nonzero_weights']} |")
    ood = report["sae_minus_raw"]["ood"]
    if ood["f1"] < 0 and ood["auroc"] < 0:
        lines += ["", "The raw probe achieved higher OOD F1 and AUROC."]
    elif ood["f1"] > 0 and ood["auroc"] > 0:
        lines += ["", "The SAE probe achieved higher OOD F1 and AUROC."]
    else:
        lines += ["", "The OOD metrics show a mixed result or a tie."]
    lines += [""]
    for split, deltas in report["sae_minus_raw"].items():
        lines.append(
            f"- {split.upper()} SAE minus raw. F1 {deltas['f1']:+.4f}, "
            f"AUROC {deltas['auroc']:+.4f}."
        )
    lines += [
        "",
        "## Sample efficiency",
        "",
        "Values are mean ± population standard deviation over the configured paired subsets.",
        "",
        "| Labels | Split | Space | F1 | Precision | Recall | AUROC |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for size, scores in report["sample_efficiency"].items():
        for split in TEST_SPLITS:
            for space in SPACES:
                stats = scores[space]["summary"][split]
                values = " | ".join(
                    f"{stats[key]['mean']:.4f} ± {stats[key]['std']:.4f}" for key in METRICS
                )
                lines.append(f"| {size} | {split} | {space} | {values} |")
    return "\n".join(lines) + "\n"


def plot_curve(report: dict, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(figure)
    data = report["sample_efficiency"]
    sizes = sorted(map(int, data))
    axes = figure.subplots(1, 2)
    for axis, split in zip(axes, TEST_SPLITS, strict=True):
        for space in SPACES:
            values = [data[str(size)][space]["summary"][split]["auroc"] for size in sizes]
            axis.errorbar(
                sizes,
                [item["mean"] for item in values],
                yerr=[item["std"] for item in values],
                marker="o",
                label=space.upper(),
            )
        axis.set(
            title=split.upper(), xlabel="Labeled training examples", ylabel="AUROC", xticks=sizes
        )
        axis.legend()
    figure.savefig(path)
