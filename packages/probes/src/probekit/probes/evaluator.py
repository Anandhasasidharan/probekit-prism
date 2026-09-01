"""Probe evaluation metrics: accuracy, precision, recall, F1, AUROC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
                         concept: str = "", probe_kind: str = "") -> dict:
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "concept": concept or "unknown",
        "probe_kind": probe_kind or "unknown",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["auroc"] = None
    metrics["n_test"] = int(len(y_true))
    return metrics


def format_comparison_table(results: list[dict]) -> str:
    """Render a three-way comparison as an aligned markdown-ish table."""
    header = (
        f"{'kind':<10} {'concept':<24} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7}"
        f" {'auroc':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        auroc = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "-"
        lines.append(
            f"{r['probe_kind']:<10} {r['concept']:<24} "
            f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} "
            f"{r['f1']:>7.4f} {auroc:>7}"
        )
    return "\n".join(lines)
