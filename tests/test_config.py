from pathlib import Path

import pytest
from pydantic import ValidationError

from probekit.config import CorpusConfig, ModelConfig, load_config


def test_load_default_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    cfg = load_config(root / "configs/default.yaml")
    assert cfg.model.name == "gpt2"
    assert cfg.model.layer == 8
    assert cfg.paths.activations == root / "data/activations/corpus.parquet"


def test_invalid_config() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelConfig.model_validate({"revision": "test", "unknown": True})
    with pytest.raises(ValidationError, match="exactly one corpus source"):
        CorpusConfig(max_tokens=20)
