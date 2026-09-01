"""Registry: CRUD, search, verify, circuits, stats history."""

import pytest
from probekit.core.types import CircuitRecord, FeatureRecord
from probekit.registry import FeatureRegistry


@pytest.fixture()
def registry(tmp_path):
    return FeatureRegistry(f"sqlite:///{tmp_path}/registry.db")


def _feature(fid: int, **kw) -> FeatureRecord:
    defaults = dict(
        id=fid, layer=5, name=f"feature_{fid}",
        description=f"Activates on deletion commands {fid}",
        detection_score=0.9,
    )
    defaults.update(kw)
    return FeatureRecord(**defaults)


def test_feature_crud(registry):
    registry.add_feature(_feature(42))
    got = registry.get_feature(42)
    assert got is not None and got.id == 42 and got.layer == 5

    got.name = "renamed"
    registry.add_feature(got)  # upsert
    assert registry.get_feature(42).name == "renamed"

    assert registry.get_feature(999) is None


def test_search(registry):
    registry.add_feature(_feature(1, name="file_deletion",
                                  description="rm -rf and delete file patterns"))
    registry.add_feature(_feature(2, name="honesty", description="truthful statements"))
    registry.add_feature(_feature(3, name="deletion_v2", description="more deleting"))

    hits = registry.search_features("deletion")
    assert {h.id for h in hits} == {1, 3}
    hits = registry.search_features("truthful")
    assert [h.id for h in hits] == [2]
    assert len(registry.search_features("", limit=10)) == 3

    by_layer = registry.search_features("", layer=6)
    assert by_layer == []


def test_verify(registry):
    registry.add_feature(_feature(7))
    assert registry.get_feature(7).verified is False
    assert registry.verify_feature(7) is True
    assert registry.get_feature(7).verified is True
    assert registry.verify_feature(12345) is False


def test_circuits(registry):
    cid = registry.add_circuit(
        CircuitRecord(id=None, name="risky_delete_circuit",
                      description="deletion intent circuit",
                      feature_ids=[1, 2, 3], layers=[5, 8])
    )
    circuit = registry.get_circuit(cid)
    assert circuit.feature_ids == [1, 2, 3]
    assert circuit.layers == [5, 8]


def test_stats_history(registry):
    registry.record_stats(9, "sae", frequency=0.001)
    registry.record_stats(9, "sae", frequency=0.30)
    hist = registry.stats_history(9)
    assert len(hist) == 2
    assert hist[0]["frequency"] == pytest.approx(0.001)
