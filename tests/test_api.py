from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probekit.autointerp.cache import LabelCache
from probekit.autointerp.collector import AutoInterpConfig
from probekit.config import load_config
from probekit.harvest.harvester import ActivationHarvester
from probekit.onnx_utils import cpu_session
from probekit.probes.datasets import file_sha256
from probekit.probes.exporter import export_probe, predict_onnx
from probekit.probes.raw_trainer import LinearProbe
from probekit.sae.architectures import TopKSAE
from probekit.sae.exporter import encode_onnx, export_encoder
from probekit.serve.app import ServeConfig, create_app
from probekit.serve.bundle import Bundle


def test_onnx_matches_pytorch_and_sklearn(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(50, 8)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = LogisticRegression(C=0.5).fit(transformed, y)
    coefficients = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - scaler.mean_ @ coefficients)
    probe = LinearProbe(coefficients, intercept, 0.5, 1.0)
    probe_path = tmp_path / "probe.onnx"
    export_probe(probe, probe_path)
    probabilities = predict_onnx(cpu_session(probe_path), x)
    assert np.allclose(probabilities, model.predict_proba(transformed)[:, 1], atol=1e-4, rtol=0)

    torch.manual_seed(9)
    sae = TopKSAE(8, 20, 3, scale=0.25)
    with torch.no_grad():
        sae.decoder.bias.copy_(torch.arange(8) / 30)
        sae.encoder.bias.fill_(0.1)
        expected = sae.encode(torch.from_numpy(x)).numpy()
    encoder = tmp_path / "encoder.onnx"
    export_encoder(sae, encoder)
    observed = encode_onnx(cpu_session(encoder), x, sae.k)
    assert np.allclose(observed, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize("space", ["raw", "sae"])
def test_endpoints_use_exported_models(
    tiny_harvester: ActivationHarvester, tmp_path: Path, space: str
) -> None:
    snapshot = tmp_path / "tiny"
    tiny_harvester.model.save_pretrained(snapshot)
    tiny_harvester.tokenizer.save_pretrained(snapshot)
    config = load_config()
    model = tiny_harvester.config.model_copy(update={"local_dir": snapshot})
    paths = config.paths.model_copy(
        update={
            "probe_dir": tmp_path / "exports",
            "feature_db": tmp_path / "features.sqlite",
        }
    )
    config = config.model_copy(update={"model": model, "paths": paths})
    sae = TopKSAE(tiny_harvester.width, 24, 4)
    onnx_names = ["sae-encoder.onnx", "pii-raw.onnx", "pii-sae.onnx"]
    export_encoder(sae, paths.probe_dir / onnx_names[0])
    for name, width in zip(onnx_names[1:], (tiny_harvester.width, sae.dict_size), strict=True):
        probe = LinearProbe(np.zeros(width), 0.25, 0.1, 1.0)
        export_probe(probe, paths.probe_dir / name)
    hashes = {name: file_sha256(paths.probe_dir / name) for name in onnx_names}
    bundle = Bundle(
        model_version=tiny_harvester.model_version,
        layer=model.layer,
        max_length=model.max_length,
        pooling="mean",
        threshold=0.5,
        sae_id="test-sae",
        k=sae.k,
        concept="pii",
        sha256=hashes,
    )
    (paths.probe_dir / "bundle.json").write_text(bundle.model_dump_json(), encoding="utf-8")
    labels = AutoInterpConfig(model="test-explainer")
    cache = LabelCache(paths.feature_db, bundle.sae_id, labels.model, labels.prompt_version)
    for i in range(sae.dict_size):
        cache.label(i, lambda: "Fixture label.")
    settings = ServeConfig(probe_space=space, feature_limit=3, max_text_chars=32)

    with TestClient(create_app(config, settings, labels)) as client:
        assert client.get("/health").json() == {"status": "ok", "device": "cpu"}
        response = client.post("/probe", json={"text": "a b c d", "concept": "pii"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["score"] == pytest.approx(0.5621765)
        assert payload["label"] == 1
        assert payload["latency_ms"] > 0
        assert client.post("/probe", json={"text": "a b", "concept": "bad"}).status_code == 404
        assert client.post("/features/activate", json={"text": "a"}).status_code == 422
        assert client.post("/features/activate", json={"text": "x" * 33}).status_code == 422
        response = client.post("/features/activate", json={"text": "a b c d"})
        assert response.status_code == 200
        features = response.json()
        assert 0 < len(features) <= 3
        assert all(feature["label"] == "Fixture label." for feature in features)
        values = [feature["activation"] for feature in features]
        assert values == sorted(values, reverse=True)
