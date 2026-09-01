"""Dataset builder: align labeled concept examples with harvested activations.

A concept dataset is a JSONL file of {"text": ..., "label": true/false,
"concept": ...}. The builder harvests activations for each text (last token
of the final captured layer by default — the standard probing position),
projects into a feature space, and produces train/test splits including an
out-of-distribution split by source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeDataset:
    X: np.ndarray  # (n_samples, d)
    y: np.ndarray  # (n_samples,) int
    texts: list[str]
    sources: list[str | None]
    concept: str

    def __len__(self) -> int:
        return len(self.y)

    def split(self, test_size: float = 0.2, ood_size: float = 0.0, seed: int = 42):
        rng = np.random.default_rng(seed)
        n = len(self.y)
        idx = rng.permutation(n)

        if ood_size > 0:
            # OOD split taken from distinct sources where available.
            unique_sources = list({s for s in self.sources if s})
            n_ood = int(n * ood_size) if not unique_sources else min(int(n * ood_size), n // 3)
            ood_idx = set(idx[:n_ood].tolist())
            rest = idx[n_ood:]
        else:
            ood_idx = set()
            rest = idx

        n_test = max(1, int(len(rest) * test_size)) if test_size > 0 else 0
        test_idx = set(rest[:n_test].tolist())
        train_idx = [i for i in rest.tolist() if i not in test_idx]
        return (
            _subset(self, train_idx),
            _subset(self, sorted(test_idx)),
            _subset(self, sorted(ood_idx)),
        )


def _subset(ds: ProbeDataset, indices: list[int]) -> ProbeDataset:
    return ProbeDataset(
        X=ds.X[indices], y=ds.y[indices],
        texts=[ds.texts[i] for i in indices],
        sources=[ds.sources[i] for i in indices],
        concept=ds.concept,
    )


def load_examples(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_dataset_from_records(
    records: list,
    examples: list[dict],
    concept: str,
    *,
    match_last_token: bool = True,
    layer: int | None = None,
) -> ProbeDataset:
    """Match ConceptExample texts to harvested ActivationRecords.

    Matching strategy: for each example text, take its harvested record for
    the highest captured layer and last non-pad token position.
    """
    from probekit.harvest.provenance import ProvenanceTracker

    ordered = ProvenanceTracker.stitch_chunks(records)
    # Group records by sequence; sequences follow input order per batch.
    seqs: dict[str, list] = {}
    for r in ordered:
        seqs.setdefault(r.provenance.sequence_id, []).append(r)

    seq_list = list(seqs.values())
    X: list[np.ndarray] = []
    y: list[int] = []
    texts: list[str] = []
    sources: list[str | None] = []
    flat_records = [recs[-1] for recs in seq_list]  # last token per sequence

    for ex in examples:
        if ex.get("concept") != concept:
            continue
        text = ex["text"]
        rec = next((r for r in reversed(seq_list) if _seq_text(r).strip() == text.strip()), None)
        target = rec[-1] if rec is not None else None
        if match_last_token and target is None:
            target = flat_records[len(X)] if len(X) < len(flat_records) else None
        if target is None:
            continue
        if layer is not None and target.provenance.layer_index != layer:
            continue
        X.append(np.asarray(target.activations, dtype=np.float32).reshape(-1))
        y.append(int(bool(ex["label"])))
        texts.append(text)
        sources.append(ex.get("source"))

    return ProbeDataset(
        X=np.stack(X), y=np.array(y, dtype=np.int64), texts=texts,
        sources=sources, concept=concept,
    )


def _seq_text(records: list) -> str:
    return "".join(r.token_str for r in records)
