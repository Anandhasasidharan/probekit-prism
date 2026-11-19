"""Monitor: drift detectors, alerts."""

import numpy as np
from probekit.monitor import (
    AlertManager,
    FrequencyDriftDetector,
    KLDriftDetector,
    ProbeDriftDetector,
)


def test_frequency_no_alert_in_band():
    det = FrequencyDriftDetector(baseline_mean=0.01, baseline_std=0.001, threshold_std=3.0)
    assert det.check(0.0105, feature_id=1) is None


def test_frequency_alert_on_spike():
    det = FrequencyDriftDetector(baseline_mean=0.007, baseline_std=0.0005, threshold_std=3.0)
    alert = det.check(0.31, feature_id=2419)
    assert alert is not None
    assert alert.kind == "feature_changed_meaning"
    assert alert.severity == "critical"
    assert "2419" in alert.message


def test_frequency_alert_on_disappearance():
    det = FrequencyDriftDetector(baseline_mean=0.05, baseline_std=0.001, threshold_std=3.0)
    alert = det.check(0.0, feature_id=5)
    assert alert.kind == "feature_disappeared"


def test_probe_drift():
    det = ProbeDriftDetector(baseline_accuracy=0.94, tolerance=0.05)
    assert det.check(0.92) is None
    alert = det.check(0.80)
    assert alert.kind == "probe_degraded" and alert.severity == "critical"
    assert det.check(0.87).severity == "warning"


def test_kl_drift():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=5000)
    det = KLDriftDetector(reference, bins=32)

    same = det.check(rng.normal(size=2000), threshold_kl=0.25)
    assert same is None

    shifted = det.check(rng.normal(loc=2.0, size=2000), threshold_kl=0.25)
    assert shifted is not None
    assert shifted.details["kl_divergence"] > 0.25


def test_alert_manager_severity_gate_and_handlers():
    fired = []
    mgr = AlertManager(min_severity="warning")
    mgr.add_handler(lambda a: fired.append(a))

    mgr.fire(type("A", (), {"kind": "k", "severity": "info", "message": "m",
                            "details": {}, "created_at": ""})())
    mgr.fire(type("A", (), {"kind": "k", "severity": "critical", "message": "m",
                            "details": {}, "created_at": ""})())

    # info suppressed below min severity; critical delivered to handler.
    handler_calls = len(fired)
    assert handler_calls == 1
    assert mgr.history and len(mgr.history) == 2  # everything is logged
