"""Validation-only hyperparameter selection and portable logistic coefficients."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from probekit.probes.datasets import ProbeConfig


@dataclass(frozen=True)
class LinearProbe:
    weights: np.ndarray
    intercept: float
    c: float
    validation_auroc: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return expit(matrix @ self.weights + self.intercept)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary:
            staged = Path(temporary) / path.name
            np.savez(
                staged,
                weights=self.weights,
                intercept=self.intercept,
                c=self.c,
                validation_auroc=self.validation_auroc,
            )
            staged.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        with np.load(path, allow_pickle=False) as stored:
            return cls(
                stored["weights"],
                float(stored["intercept"]),
                float(stored["c"]),
                float(stored["validation_auroc"]),
            )


def train_raw_probe(
    train: np.ndarray,
    y_train: np.ndarray,
    val: np.ndarray,
    y_val: np.ndarray,
    config: ProbeConfig,
    seed: int,
) -> LinearProbe:
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    best = None
    with threadpool_limits(limits=2):
        for value in sorted(config.c_values):
            learner = LogisticRegression(
                C=value, solver="liblinear", random_state=seed, max_iter=config.solver_max_iter
            ).fit(train_scaled, y_train)
            score = float(roc_auc_score(y_val, learner.predict_proba(val_scaled)[:, 1]))
            if best is None or score > best.validation_auroc:
                weights = learner.coef_[0] / scaler.scale_
                intercept = float(learner.intercept_[0] - scaler.mean_ @ weights)
                best = LinearProbe(weights, intercept, value, score)
    if best is None:
        raise ValueError("no probe C candidates")
    return best
