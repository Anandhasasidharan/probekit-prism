"""API: endpoints in registry-only mode (no model needed) + core config."""

import pytest
from fastapi.testclient import TestClient
from probekit.api.main import create_app
from probekit.core.config import GlobalConfig
from probekit.core.types import FeatureRecord


@pytest.fixture()
def client(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
registry:
  database_url: sqlite:///{tmp_path}/reg.db
"""
    )
    app = create_app(config_path)
    state = app.state.probekit

    # Seed some features so search/verify/drift have data.
    for fid, freq in [(1, 0.001), (2, 0.002)]:
        state.registry.add_feature(
            FeatureRecord(id=fid, layer=5, name=f"f{fid}",
                          description="file deletion feature")
        )
        state.drift_history.append((fid, "sae", [
            {"recorded_at": "t0", "frequency": freq, "probe_accuracy": None},
            {"recorded_at": "t1", "frequency": freq * 40, "probe_accuracy": None},
        ]))
    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["checks"]["registry"] is True
    assert ready["ready"] is False  # model/probes not loaded in this mode


def test_feature_search_and_verify(client):
    hits = client.get("/features/search", params={"q": "deletion"}).json()
    assert len(hits) == 2
    assert all("deletion" in h["description"] for h in hits)

    assert client.get("/features/search", params={"q": "zzz"}).json() == []

    r = client.post("/features/1/verify")
    assert r.status_code == 200 and r.json()["verified"] is True
    r = client.post("/features/999/verify")
    assert r.status_code == 404


def test_probe_score_404_without_model(client):
    r = client.post("/probe/score", json={"text": "hi", "concept": "pii_detection"})
    assert r.status_code == 404


def test_steering_unavailable_returns_503(client):
    r = client.post("/steer/check_and_steer",
                    json={"prompt": "delete files", "strategy": "suppress_risky"})
    assert r.status_code in (503, 500)


def test_monitor_drift_endpoint(client):
    body = client.get("/monitor/drift").json()
    assert body["checked_features"] == 2
    kinds = [a["kind"] for a in body["alerts"]]
    assert "feature_changed_meaning" in kinds


# ------------------------------------------------------------------ core config


def test_global_config_defaults_and_yaml(tmp_path):
    cfg = GlobalConfig()
    assert cfg.harvest.model_name == "gpt2"
    assert cfg.sae.architecture == "topk"

    p = tmp_path / "c.yaml"
    p.write_text("""
sae:
  architecture: gated
  k: 16
auto_interp:
  provider: anthropic
""")
    loaded = GlobalConfig.from_yaml(p)
    assert loaded.sae.architecture == "gated"
    assert loaded.sae.k == 16
    assert loaded.auto_interp.provider == "anthropic"

    missing = GlobalConfig.from_yaml(tmp_path / "nope.yaml")
    assert missing.sae.k == 32  # defaults when file absent


def test_device_selection():
    import torch
    from probekit.core.device import get_device

    dev = get_device(None)
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"
    assert get_device("cpu").type == "cpu"
