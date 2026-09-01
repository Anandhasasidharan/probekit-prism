"""Harvest: activation collection, provenance, parquet round-trip."""

import numpy as np
from probekit.harvest import (
    ActivationHarvester,
    ProvenanceTracker,
    read_activation_parquet,
    write_activation_parquet,
)


def test_harvest_sync_records(tiny_model, tiny_tokenizer, monkeypatch):
    h = ActivationHarvester.__new__(ActivationHarvester)
    # Manual wiring with the local tiny model (avoids HF download).
    from probekit.core.device import get_device
    from probekit.harvest.provenance import ProvenanceTracker

    h.model = tiny_model
    h.tokenizer = tiny_tokenizer
    h.device = get_device("cpu")
    h.capture_layers = [0, 3]
    h.batch_size = 4
    h.max_length = 64
    h._n_layers = tiny_model.config.n_layer
    from probekit.harvest.harvester import _resolve_layer_list

    h._resolved_layers = _resolve_layer_list(h.capture_layers, h._n_layers)
    h._tracker = ProvenanceTracker(model_version="tiny-gpt2-test")
    h._handles = []
    h._stored = {}

    prompts = ["the cat sat on the mat", "delete all files"]
    records = h.harvest_sync(prompts)

    n_layers = 2
    n_tokens_per_prompt = [
        len(tiny_tokenizer(p)["input_ids"]) for p in prompts
    ]
    expected = sum(n_tokens_per_prompt) * n_layers
    assert len(records) == expected
    for rec in records[:5]:
        assert rec.activations.shape == (tiny_model.config.n_embd,)
        assert rec.provenance.layer_index in (0, 3)
        assert rec.provenance.sequence_id.startswith("seq-")
        assert rec.token_id >= 0


def test_parquet_roundtrip(tmp_path, tiny_model, tiny_tokenizer, monkeypatch):
    from test_harvest import _make_harvester

    h = _make_harvester(tiny_model, tiny_tokenizer)
    records = h.harvest_sync(["hello world"])
    path = tmp_path / "acts.parquet"
    write_activation_parquet(records, path)
    loaded = read_activation_parquet(path)
    assert len(loaded) == len(records)
    for a, b in zip(records, loaded):
        assert a.token_str == b.token_str
        assert a.provenance.sequence_id == b.provenance.sequence_id
        assert np.allclose(a.activations, b.activations, atol=1e-5)


def test_stitch_chunks():
    tracker = ProvenanceTracker(model_version="m")
    sid = tracker.next_sequence()
    from probekit.core.types import ActivationRecord

    recs = [
        ActivationRecord(tracker.record(sequence_id=sid, token_position=1, layer_index=0,
                                        batch_index=0), np.zeros(3), 1, "b"),
        ActivationRecord(tracker.record(sequence_id=sid, token_position=0, layer_index=0,
                                        batch_index=0), np.zeros(3), 0, "a"),
    ]
    stitched = ProvenanceTracker.stitch_chunks(recs)
    assert [r.provenance.token_position for r in stitched] == [0, 1]


def _make_harvester(tiny_model, tiny_tokenizer):
    """Helper used by this module and the integration test."""
    from probekit.core.device import get_device
    from probekit.core.types import ActivationRecord  # noqa: F401
    from probekit.harvest.harvester import ActivationHarvester, _resolve_layer_list

    h = ActivationHarvester.__new__(ActivationHarvester)
    h.model = tiny_model
    h.tokenizer = tiny_tokenizer
    h.device = get_device("cpu")
    h.capture_layers = [0, 3]
    h.batch_size = 4
    h.max_length = 64
    h._n_layers = tiny_model.config.n_layer
    h._resolved_layers = _resolve_layer_list(h.capture_layers, h._n_layers)
    from probekit.harvest.provenance import ProvenanceTracker

    h._tracker = ProvenanceTracker(model_version="tiny-gpt2-test")
    h._handles = []
    h._stored = {}
    return h
