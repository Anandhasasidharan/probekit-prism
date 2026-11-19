import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from probekit.probes.comparator import compare
from probekit.probes.datasets import (
    SPLITS,
    LabeledSequence,
    ProbeConfig,
    ProbeDataset,
    read_dataset,
    validate_splits,
)


def test_committed_splits_are_disjoint() -> None:
    dataset = read_dataset(Path("data/examples/pii_detection.jsonl"))
    ids = {split: {r.sequence_id for r in dataset if r.split == split} for split in SPLITS}
    for i, split in enumerate(SPLITS):
        assert all(ids[split].isdisjoint(ids[other]) for other in SPLITS[i + 1 :])
    strata = Counter((row.split, row.kind, row.label) for row in dataset)
    assert all(
        len({count for key, count in strata.items() if key[0] == split}) == 1 for split in SPLITS
    )
    with pytest.raises(ValueError, match="duplicate sequence ID or normalized text"):
        validate_splits(dataset + [dataset[0].model_copy(update={"split": "val"})])
    with pytest.raises(ValueError, match="duplicate sequence ID or normalized text"):
        validate_splits(dataset + [dataset[0].model_copy(update={"sequence_id": "new"})])


def test_matched_probes_reach_high_auroc(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    rows = [
        LabeledSequence(
            sequence_id=f"{split}-{i}",
            text=f"test record {split}-{i}",
            label=i % 2,
            split=split,
            template_id="heldout" if split == "ood" else "id",
            kind="separable",
        )
        for split in SPLITS
        for i in range(30)
    ]
    labels = np.asarray([row.label for row in rows])
    raw = rng.normal(size=(len(rows), 8))
    raw[:, 2] = (labels * 2 - 1) * 8
    sae = np.maximum(0, np.column_stack([raw[:, 2], -raw[:, 2], raw[:, :2]]))
    result = compare(
        ProbeDataset(rows, raw, sae),
        ProbeConfig(
            dataset=tmp_path / "unused", c_values=[0.1, 1.0], sample_sizes=[20], sample_seeds=[4, 5]
        ),
        12,
        {"fixture": True},
        tmp_path / "weights",
        tmp_path / "reports",
        "pii",
    )
    for space in ("raw", "sae"):
        for split in ("test", "ood"):
            assert result["full_training"][space]["splits"][split]["auroc"] > 0.98
    per_space = result["sample_efficiency"]["20"]
    assert [r["subset_sha256"] for r in per_space["raw"]["runs"]] == [
        r["subset_sha256"] for r in per_space["sae"]["runs"]
    ]
    report = tmp_path / "reports"
    assert json.loads((report / "probe-comparison.json").read_text()) == result
    assert (report / "probe-comparison.md").is_file()
    assert (report / "probe-sample-efficiency.png").stat().st_size > 0
