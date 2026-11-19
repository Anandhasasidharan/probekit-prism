"""SAE: architectures train, reconstruct, export to ONNX."""

import numpy as np
import pytest
import torch
from probekit.sae import (
    SAETrainer,
    build_sae,
    encode_matrix,
    evaluate_sae,
    export_sae_onnx,
    verify_sae_onnx,
)

D_MODEL = 32
N = 512


@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng(0)
    # Low-rank data so an SAE can actually learn to reconstruct it.
    basis = rng.normal(size=(4, D_MODEL)).astype(np.float32)
    coef = (rng.random((N, 4)) > 0.7) * rng.exponential(1.0, size=(N, 4))
    return (coef @ basis + 0.01 * rng.normal(size=(N, D_MODEL))).astype(np.float32)


@pytest.mark.parametrize("arch", ["topk", "gated", "jumprelu", "standard"])
def test_all_architectures_train(synthetic_data, arch):
    torch.manual_seed(0)
    sae = build_sae(arch, d_model=D_MODEL, dict_size=128, k=8, device="cpu")
    trainer = SAETrainer(sae, batch_size=64, device="cpu")
    trainer.train(synthetic_data, epochs=3, log_every=10)
    assert trainer.history, "no training history recorded"

    metrics = evaluate_sae(sae, synthetic_data, device="cpu")
    assert metrics["n_tokens"] == N
    assert 0.0 <= metrics["dead_feature_rate"] <= 1.0
    # Reconstruction must beat the zero baseline after training.
    assert metrics["explained_variance"] > -1.0


def test_topk_sparsity_enforced(synthetic_data):
    torch.manual_seed(1)
    sae = build_sae("topk", d_model=D_MODEL, dict_size=256, k=4, device="cpu")
    x = torch.from_numpy(synthetic_data[:16])
    acts = sae.encode(x)
    active_per_token = (acts != 0).sum(dim=-1)
    assert int(active_per_token.max()) <= 4


def test_onnx_export_and_verify(tmp_path, synthetic_data):
    sae = build_sae("topk", d_model=D_MODEL, dict_size=64, k=4, device="cpu")
    path = export_sae_onnx(sae, tmp_path / "sae.onnx")
    diff = verify_sae_onnx(path, sae, synthetic_data[:8])
    assert diff < 1e-4


def test_encode_matrix(synthetic_data):
    sae = build_sae("topk", d_model=D_MODEL, dict_size=64, k=4, device="cpu")
    latents = encode_matrix(sae, synthetic_data)
    assert latents.shape == (N, 64)


def test_save_load_roundtrip(tmp_path, synthetic_data):
    sae = build_sae("topk", d_model=D_MODEL, dict_size=64, k=4, device="cpu")
    trainer = SAETrainer(sae, device="cpu")
    trainer.train(synthetic_data, epochs=1)
    trainer.save(tmp_path / "sae.pt")

    fresh = build_sae("topk", d_model=D_MODEL, dict_size=64, k=4, device="cpu")
    trainer2 = SAETrainer(fresh, device="cpu")
    trainer2.load(tmp_path / "sae.pt")
    x = torch.from_numpy(synthetic_data[:4])
    assert torch.allclose(sae.encode(x), fresh.encode(x), atol=1e-6)
