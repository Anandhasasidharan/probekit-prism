from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from probekit.autointerp.collector import AutoInterpConfig, collect_examples
from probekit.autointerp.explainer import HaikuExplainer
from probekit.autointerp.pipeline import autointerpret
from probekit.autointerp.scorer import score_detection
from probekit.config import load_config
from probekit.harvest.harvester import ActivationHarvester, SequenceInput
from probekit.sae.architectures import TopKSAE
from probekit.sae.trainer import SAECheckpoint


def test_cached_pipeline(tiny_harvester: ActivationHarvester, tmp_path: Path) -> None:
    paths = load_config().paths.model_copy(
        update={
            "activations": tmp_path / "activations.parquet",
            "sequences": tmp_path / "sequences.parquet",
            "feature_db": tmp_path / "features.sqlite",
            "runs": tmp_path,
        }
    )
    text = ["a b c d", "c d e f", "b a f d", "f g e a"]
    tiny_harvester.harvest(
        (SequenceInput(str(i), text[i % len(text)]) for i in range(60)),
        paths.activations,
        paths.sequences,
    )
    torch.manual_seed(12)
    model = TopKSAE(tiny_harvester.width, 24, 4)
    checkpoint = SAECheckpoint(model, 12, 3, "gpt2@tiny-test", 1, 0.15)
    settings = AutoInterpConfig(n_features=4, top_examples=3, score_examples_per_class=2)
    model_name = "mock-haiku"
    client = Mock()
    client.messages.create.side_effect = make_response
    explainer = HaikuExplainer(model_name, client)
    report = autointerpret(
        paths, checkpoint, "sae-id", tiny_harvester.tokenizer, explainer, settings
    )
    assert len(report) == settings.n_features
    assert all(row["description"] == "Fires on a local pattern." for row in report)
    assert any(row["examples"] == 4 for row in report)
    assert client.messages.create.called
    client.messages.create.reset_mock(side_effect=True)
    second = autointerpret(
        paths, checkpoint, "sae-id", tiny_harvester.tokenizer, explainer, settings
    )
    assert second == report
    client.messages.create.assert_not_called()
    assert (tmp_path / "autointerp.csv").exists()
    assert (tmp_path / "autointerp-scores.png").stat().st_size > 0

    samples = collect_examples(paths.activations, paths.sequences, model, 12, settings)
    for sampled in samples.values():
        assert sampled.quantile
        explanation_ids = {item.sequence_id for item in sampled.top + sampled.quantile}
        detection_ids = {item.sequence_id for item in sampled.positive + sampled.negative}
        assert explanation_ids.isdisjoint(detection_ids)


def make_response(**kwargs: object) -> Mock:
    import json

    payload = json.loads(kwargs["messages"][0]["content"])
    content = "Fires on a local pattern."
    if "contexts" in payload:
        content = json.dumps([0.5] * len(payload["contexts"]))
    return Mock(content=[Mock(type="text", text=content)])


def test_detection_metrics_and_invalid_response() -> None:
    explainer = Mock()
    explainer.predict.side_effect = lambda _, contexts: [float(x == "positive") for x in contexts]
    result = score_detection(explainer, "label", ["positive"] * 3, ["negative"] * 3, 42)
    assert result.accuracy == result.auroc == 1.0
    explainer.predict.side_effect = None
    explainer.predict.return_value = [-0.1, 0.9]
    with pytest.raises(ValueError, match="finite probability"):
        score_detection(explainer, "label", ["positive"], ["negative"], 42)
