"""Probes: dataset building, three trainers, comparator, ONNX export."""

import numpy as np
import pytest
from probekit.probes import (
    ProbeDataset,
    RawProbe,
    ThreeWayComparator,
    evaluate_predictions,
    export_probe_onnx,
    verify_probe_onnx,
)

D = 16
N_PER_CLASS = 60


@pytest.fixture(scope="module")
def synthetic_dataset():
    rng = np.random.default_rng(3)
    center_pos = rng.normal(size=D).astype(np.float32)
    X = np.concatenate([
        rng.normal(size=(N_PER_CLASS, D)).astype(np.float32) * 0.5 + center_pos,
        rng.normal(size=(N_PER_CLASS, D)).astype(np.float32) * 0.5 - center_pos,
    ])
    y = np.array([1] * N_PER_CLASS + [0] * N_PER_CLASS)
    texts = [f"example {i}" for i in range(len(y))]
    return ProbeDataset(X=X, y=y, texts=texts, sources=[None] * len(y), concept="test_concept")


def test_split_sizes(synthetic_dataset):
    train, test, ood = synthetic_dataset.split(test_size=0.2, ood_size=0.1)
    total = len(train) + len(test) + len(ood)
    assert total == len(synthetic_dataset)
    assert len(test) >= 1 and len(train) > len(test)


def test_raw_probe_learns(synthetic_dataset):
    train, test, _ = synthetic_dataset.split(test_size=0.25)
    probe = RawProbe("test_concept").fit(train)
    scores = probe.score_dataset(test)
    metrics = evaluate_predictions(test.y, scores, concept="test_concept", probe_kind="raw")
    assert metrics["accuracy"] > 0.85
    assert metrics["auroc"] is not None and metrics["auroc"] > 0.9


def test_three_way_comparator(synthetic_dataset):
    from probekit.circuit.clt import CrossLayerTranscoder
    from probekit.sae.architectures import build_sae

    train, test, _ = synthetic_dataset.split(test_size=0.25)

    sae = build_sae("topk", d_model=D, dict_size=64, k=4, device="cpu")
    tc = CrossLayerTranscoder(D, n_features=32, device="cpu")

    comp = ThreeWayComparator(sae=sae, clt_transcoder=tc, device="cpu")
    results = comp.compare_concept(train, test)
    kinds = {r["probe_kind"] for r in results}
    assert kinds == {"raw", "sae", "circuit"}

    table = comp.table()
    for kind in ("raw", "sae", "circuit"):
        assert kind in table


def test_probe_onnx_export(synthetic_dataset, tmp_path):
    train, test, _ = synthetic_dataset.split(test_size=0.25)
    probe = RawProbe("test_concept").fit(train)
    path = export_probe_onnx(probe, tmp_path / "probe.onnx")
    diff = verify_probe_onnx(path, probe, test.X[:16])
    assert diff < 1e-4


def test_build_dataset_from_records(tiny_model, tiny_tokenizer):
    from test_harvest import _make_harvester

    h = _make_harvester(tiny_model, tiny_tokenizer)
    texts = ["delete all files", "hello world"]
    records = h.harvest_sync(texts)

    from probekit.probes.dataset import build_dataset_from_records

    examples = [
        {"text": texts[0], "label": True, "concept": "tool_misuse"},
        {"text": texts[1], "label": False, "concept": "tool_misuse"},
    ]
    ds = build_dataset_from_records(records, examples, "tool_misuse")
    assert len(ds) == 2
    assert list(ds.y) == [1, 0]
