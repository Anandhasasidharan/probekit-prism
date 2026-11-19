"""Score blind detections on balanced, independently held-out contexts."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from probekit.autointerp.explainer import Explainer


@dataclass(frozen=True)
class DetectionScore:
    accuracy: float | None
    auroc: float | None
    examples: int


def score_detection(
    explainer: Explainer,
    label: str,
    positives: list[str],
    negatives: list[str],
    seed: int,
) -> DetectionScore:
    if not positives or not negatives:
        return DetectionScore(None, None, 0)
    contexts = positives + negatives
    targets = np.asarray([1] * len(positives) + [0] * len(negatives))
    order = np.random.default_rng(seed).permutation(len(contexts))
    predictions = np.asarray(explainer.predict(label, [contexts[i] for i in order]), dtype=float)
    if (
        predictions.shape != (len(contexts),)
        or not np.isfinite(predictions).all()
        or (predictions < 0).any()
        or (predictions > 1).any()
    ):
        raise ValueError("expected one finite probability in [0, 1] per scoring example")
    return DetectionScore(
        float(accuracy_score(targets[order], predictions >= 0.5)),
        float(roc_auc_score(targets[order], predictions)),
        len(contexts),
    )
