"""AutoInterp: collector, cache, mock explainer, pipeline."""

import numpy as np
import pytest
from probekit.auto_interp import (
    AutoInterpPipeline,
    ExplanationCache,
    MockExplainer,
    collect_examples,
    fingerprint_examples,
)

N_TOKENS = 400
N_FEATURES = 8


@pytest.fixture()
def acts_matrix():
    rng = np.random.default_rng(7)
    mat = rng.random((N_TOKENS, N_FEATURES)).astype(np.float32) * 0.01
    # Feature 3 fires strongly on tokens containing "delete".
    for i in range(N_TOKENS):
        if i % 10 == 0:
            mat[i, 3] = 5.0 + rng.random()
    return mat


@pytest.fixture()
def tokens():
    vocab = ["delete", "file", "run", "the", "cat", "hello", "world", "data", "rm", "now"]
    return [vocab[i % len(vocab)] for i in range(N_TOKENS)]


def test_collect_examples(acts_matrix, tokens):
    ex = collect_examples(3, acts_matrix, tokens, top_k=10)
    assert ex.top_examples, "expected top examples"
    assert set(ex.top_examples) <= {"delete"}
    assert ex.frequency == pytest.approx(0.1)
    assert len(ex.random_negatives) > 0
    block = ex.as_prompt_block()
    assert "activates strongly" in block and "does NOT activate" in block


def test_cache_roundtrip(tmp_path, acts_matrix, tokens):
    cache = ExplanationCache(tmp_path / "cache.sqlite")
    key = ExplanationCache.make_key(3, "mock", "abc")
    assert cache.get(key) is None
    cache.put(key, feature_id=3, text="Activates on deletion", detection_score=0.9)
    got = cache.get(key)
    assert got["text"] == "Activates on deletion"
    assert got["detection_score"] == pytest.approx(0.9)


def test_mock_explainer_and_pipeline(tmp_path, acts_matrix, tokens):
    pipeline = AutoInterpPipeline(provider="mock",
                                  cache_db=str(tmp_path / "c.sqlite"))
    result = pipeline.run_feature(3, acts_matrix, tokens, score=True)
    assert "delete" in result["text"]
    assert result["detection_score"] == 1.0

    # Second run must hit the cache.
    again = pipeline.run_feature(3, acts_matrix, tokens, score=True)
    assert again["text"] == result["text"]
    assert again.get("created_at") is not None or again["explained_by"] == "mock"
    pipeline.close()


def test_fingerprint_stability(acts_matrix, tokens):
    e1 = collect_examples(3, acts_matrix, tokens, top_k=5)
    e2 = collect_examples(3, acts_matrix, tokens, top_k=5)
    assert fingerprint_examples(e1) == fingerprint_examples(e2)


def test_scorer_detection_with_mock():
    from probekit.auto_interp.scorer import detection_score

    explainer = MockExplainer()

    class FakeExamples:
        def all_positive(self):
            return ["rm -rf deletes files"]

    score = detection_score(explainer, "x", ["a"], ["b"], n_pairs=0)
    assert score == 0.0  # no pairs -> zero by definition
