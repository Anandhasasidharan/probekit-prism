"""Prometheus metric definitions (lazy import so tests run without it)."""

from __future__ import annotations


def build_metrics():
    from prometheus_client import Counter, Gauge, Histogram

    return {
        "probe_score": Gauge("probekit_probe_score", "Live probe score",
                             ["concept", "kind"]),
        "feature_frequency": Gauge("probekit_feature_frequency", "Feature firing frequency",
                                   ["feature_id", "space"]),
        "drift_alerts_total": Counter("probekit_drift_alerts_total", "Drift alerts fired",
                                      ["kind", "severity"]),
        "steering_interventions_total": Counter(
            "probekit_steering_interventions_total", "Steering interventions applied",
            ["strategy"]),
        "request_latency": Histogram("probekit_request_latency_seconds",
                                     "API request latency", ["route"]),
    }
