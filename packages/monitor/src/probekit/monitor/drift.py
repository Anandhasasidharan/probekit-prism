"""Drift detection: feature frequency, probe accuracy, activation distribution.

- FrequencyDriftDetector: z-score of live feature frequency vs. a stored
  baseline mean/std — catches `feature_disappeared` and
  `feature_changed_meaning` (a once-rare feature firing constantly).
- ProbeDriftDetector: probe accuracy on periodic labeled eval batches vs. a
  baseline; fires `probe_degraded` when the drop exceeds tolerance.
- KLDriftDetector: KL divergence between stored and live activation
  distributions (histogram-based).
"""

from __future__ import annotations

import numpy as np
from probekit.core.types import DriftAlert


class FrequencyDriftDetector:
    def __init__(self, baseline_mean: float, baseline_std: float,
                 threshold_std: float = 3.0):
        self.baseline_mean = baseline_mean
        self.baseline_std = max(baseline_std, 1e-9)
        self.threshold_std = threshold_std

    def check(self, live_frequency: float, feature_id: int = -1) -> DriftAlert | None:
        z = (live_frequency - self.baseline_mean) / self.baseline_std
        if abs(z) < self.threshold_std:
            return None
        if abs(live_frequency) < 1e-9:
            kind, severity = "feature_disappeared", "warning"
        elif z > 0:
            kind, severity = "feature_changed_meaning", "critical"
        else:
            kind, severity = "frequency_drop", "info"
        return DriftAlert(
            kind=kind,
            severity=severity,
            message=(
                f"Feature {feature_id}: frequency {self.baseline_mean:.5f} -> "
                f"{live_frequency:.5f} ({z:+.1f} sigma)"
            ),
            details={"z_score": float(z), "baseline": self.baseline_mean,
                     "live": live_frequency},
        )


class ProbeDriftDetector:
    def __init__(self, baseline_accuracy: float, tolerance: float = 0.05):
        self.baseline_accuracy = baseline_accuracy
        self.tolerance = tolerance

    def check(self, live_accuracy: float) -> DriftAlert | None:
        drop = self.baseline_accuracy - live_accuracy
        if drop <= self.tolerance:
            return None
        return DriftAlert(
            kind="probe_degraded",
            severity="critical" if drop > 2 * self.tolerance else "warning",
            message=(
                f"Probe accuracy {self.baseline_accuracy:.3f} -> "
                f"{live_accuracy:.3f} (drop {drop:.3f} > tol {self.tolerance})"
            ),
            details={"baseline": self.baseline_accuracy, "live": live_accuracy,
                     "drop": float(drop)},
        )


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


class KLDriftDetector:
    def __init__(self, reference: np.ndarray, bins: int = 64):
        self.bins = bins
        lo, hi = float(reference.min()), float(reference.max())
        edges = np.linspace(lo, hi, bins + 1)
        hist, _ = np.histogram(reference, bins=edges)
        self.edges = edges
        self.ref_hist = hist.astype(np.float64)

    def check(self, live: np.ndarray, threshold_kl: float = 0.25) -> DriftAlert | None:
        hist, _ = np.histogram(live, bins=self.edges)
        kl = _kl(hist.astype(np.float64), self.ref_hist)
        if kl < threshold_kl:
            return None
        return DriftAlert(
            kind="activation_distribution_drift",
            severity="warning",
            message=f"Activation distribution drift: KL={kl:.4f} > {threshold_kl}",
            details={"kl_divergence": kl, "threshold": threshold_kl},
        )
