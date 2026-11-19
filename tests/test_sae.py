from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from probekit.harvest.harvester import ActivationHarvester, SequenceInput
from probekit.sae.architectures import TopKSAE
from probekit.sae.evaluator import loss_recovery, reconstruction_metrics
from probekit.sae.trainer import SAECheckpoint, SAEConfig, activation_batches, train_sae


def test_sparsity_and_unit_decoder() -> None:
    model = TopKSAE(8, 24, 4)
    x = torch.cat([torch.zeros(1, 8), torch.randn(6, 8)])
    assert torch.all(model.encode(x).count_nonzero(dim=1) == 4)
    assert torch.allclose(model.decoder.weight.norm(dim=0), torch.ones(24), atol=1e-6)


def test_streaming_training_reduces_heldout_reconstruction(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    targets = torch.randn(256, 4) @ torch.randn(4, 8)
    activation_path, checkpoint_path = tmp_path / "acts.parquet", tmp_path / "sae.pt"
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(targets.flatten().numpy()), 8)
    pq.write_table(
        pa.table(
            {
                "sequence_id": [str(i // 4) for i in range(256)],
                "layer": [1] * 256,
                "model_version": ["gpt2@test"] * 256,
                "activation": vectors,
            }
        ),
        activation_path,
    )
    cfg = SAEConfig(
        dict_size=24,
        k=6,
        batch_size=32,
        epochs=15,
        learning_rate=0.007,
        holdout_fraction=0.2,
        checkpoint_every=50,
        dead_after_steps=8,
        aux_weight=0.1,
    )
    device = torch.device("cpu")
    report = train_sae(activation_path, checkpoint_path, cfg, 17, device, "gpt2@test", 1)
    checkpoint = SAECheckpoint.load(checkpoint_path)
    torch.manual_seed(17)
    initial = TopKSAE(8, cfg.dict_size, cfg.k, report["scale"])
    heldout = list(activation_batches(activation_path, 32, 17, 0.2, heldout=True))
    before = reconstruction_metrics(initial, heldout, device)
    after = reconstruction_metrics(checkpoint.sae, heldout, device)
    assert after["reconstruction_mse"] < 0.7 * before["reconstruction_mse"]
    assert torch.allclose(checkpoint.sae.decoder.weight.norm(dim=0), torch.ones(24), atol=1e-6)
    assert checkpoint.sae.scale.item() == pytest.approx(report["scale"])
    assert (tmp_path / "sae.step-50.pt").exists()


def test_loss_recovery_identity_and_zero(tiny_harvester: ActivationHarvester) -> None:
    sequences = tiny_harvester.tokenize(
        [SequenceInput("a", "a b c d a f g"), SequenceInput("b", "c d e b b")]
    )
    report = loss_recovery(tiny_harvester, lambda x: (x, x), sequences)
    assert report["lm_loss"]["clean"] == report["lm_loss"]["spliced"]
    zero = loss_recovery(tiny_harvester, lambda x: (torch.zeros_like(x), x), sequences)
    assert zero["lm_loss"]["spliced"] == zero["lm_loss"]["zero"]
    assert report["lm_targets"] == 10
