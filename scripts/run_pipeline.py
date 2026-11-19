"""Run the CPU demo in one process and smoke-test the exported FastAPI app."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import yaml
from fastapi.testclient import TestClient
from make_synthetic_pii import make_rows
from pydantic import Field

from probekit.cli import Concept, autointerp, compare, eval_sae, export, harvest, train_sae
from probekit.config import Settings, load_config, load_yaml

CORPUS_LINES = (
    "A public report described seasonal trends in the local weather.",
    "The archives contain reports about the city's parks and public transport.",
    "Researchers collected samples and documented changes in the river level.",
    "Visitors wrote notes on the exhibits in the afternoon.",
    "Volunteers cataloged the items from the latest library donation.",
    "A review of the timetable noted changes to the morning services.",
)


class DemoConfig(Settings):
    base_config: Path
    probes_config: Path
    autointerp_config: Path
    serve_config: Path
    output: Path
    token_budget: int = Field(ge=2)
    corpus_sequences: int = Field(gt=0)
    probe_sizes: dict[str, int]
    sample_sizes: list[int]
    sample_seeds: list[int]
    sae: dict
    autointerp_features: int = Field(gt=0)


def write_yaml(path: Path, values: dict) -> Path:
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_demo(
    settings: DemoConfig,
    root: Path,
    model_dir: Path | None,
    layer: int | None,
) -> tuple[Path, Path, Path, Path, Path]:
    from probekit.autointerp.collector import AutoInterpConfig
    from probekit.probes.datasets import ProbeConfig
    from probekit.serve.app import ServeConfig

    root.mkdir(parents=True, exist_ok=True)
    base = load_config(settings.base_config)
    config = base.model_dump(mode="json")
    model = config["model"]
    model["device"] = "cpu"
    if model_dir is not None:
        model["local_dir"] = str(model_dir.resolve())
    if layer is not None:
        model["layer"] = layer
    write_jsonl(
        root / "corpus.jsonl",
        [
            {"text": f"{CORPUS_LINES[i % len(CORPUS_LINES)]} Entry {i} was retained."}
            for i in range(settings.corpus_sequences)
        ],
    )
    config["corpus"] = {"jsonl": "corpus.jsonl", "max_tokens": settings.token_budget}
    config["paths"] = {
        "activations": "corpus.parquet",
        "sequences": "sequences.parquet",
        "sae_checkpoint": "topk.pt",
        "feature_db": "features.sqlite",
        "probe_dir": "probes",
        "runs": "runs",
    }
    probes = load_yaml(settings.probes_config, ProbeConfig).model_dump(mode="json")
    probes.update(
        dataset="pii_detection.jsonl",
        sample_sizes=settings.sample_sizes,
        sample_seeds=settings.sample_seeds,
    )
    write_jsonl(root / "pii_detection.jsonl", make_rows(base.seed, settings.probe_sizes))
    labels = load_yaml(settings.autointerp_config, AutoInterpConfig).model_dump(mode="json")
    labels.update(
        n_features=settings.autointerp_features, top_examples=8, score_examples_per_class=4
    )
    serving = load_yaml(settings.serve_config, ServeConfig).model_dump(mode="json")
    serving.update(profile_warmups=1, profile_repeats=4, offline_labels=True, probe_space="sae")
    return (
        write_yaml(root / "pipeline.yaml", config),
        write_yaml(root / "sae.yaml", settings.sae),
        write_yaml(root / "probes.yaml", probes),
        write_yaml(root / "autointerp.yaml", labels),
        write_yaml(root / "serve.yaml", serving),
    )


def smoke_api(base: Path, labels: Path, serving: Path) -> dict:
    from probekit.autointerp.collector import AutoInterpConfig
    from probekit.serve.app import ServeConfig, create_app

    app = create_app(
        load_config(base), load_yaml(serving, ServeConfig), load_yaml(labels, AutoInterpConfig)
    )
    text = "Please confirm the entry in the public record."
    with TestClient(app) as client:
        health = client.get("/health").raise_for_status().json()
        probe = client.post("/probe", json={"text": text, "concept": "pii"})
        features = client.post("/features/activate", json={"text": text})
        return {
            "health": health,
            "probe": probe.raise_for_status().json(),
            "features_returned": len(features.raise_for_status().json()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    settings = load_yaml(args.demo_config, DemoConfig)
    output = (args.output or settings.output).resolve()
    started = perf_counter()
    base, sae, probes, labels, serving = prepare_demo(settings, output, args.model_dir, args.layer)
    harvest(base)
    train_sae(base, sae)
    eval_sae(base, sae, False)
    autointerp(base, labels, None, True)
    compare(base, probes, Concept.pii)
    export(base, probes, serving, labels, True, Concept.pii)
    report = smoke_api(base, labels, serving)
    report["elapsed_seconds"] = perf_counter() - started
    report["autointerp_provider"] = "offline-placeholder"
    summary = output / "runs" / "demo-smoke.json"
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Demo completed in {report['elapsed_seconds']:.2f} s. Reports are in {summary.parent}")


if __name__ == "__main__":
    main()
