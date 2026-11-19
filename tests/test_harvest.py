import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from transformers import GPT2LMHeadModel

from probekit.config import load_config
from probekit.harvest.harvester import ActivationHarvester, SequenceInput


def test_saved_residuals_match_fresh_forwards(
    tiny_harvester: ActivationHarvester, tmp_path: Path
) -> None:
    corpus = [
        SequenceInput(str(i), " ".join(random.Random(i).choices("abcdefg", k=2 + i % 9)))
        for i in range(17)
    ]
    acts, sidecar = tmp_path / "acts.parquet", tmp_path / "sequences.parquet"
    summary = tiny_harvester.harvest(iter(corpus), acts, sidecar)
    assert summary.activation_rows == sum(len(item.text.split()) - 1 for item in corpus)

    recorded = pq.read_table(acts)
    saved_sequences = pq.read_table(sidecar).to_pylist()
    assert recorded.column_names == [
        "sequence_id",
        "token_pos",
        "layer",
        "model_version",
        "activation",
    ]
    assert str(recorded.schema.field("activation").type.value_type) == "float"
    assert pq.ParquetFile(acts).metadata.row_group(0).column(0).compression == "ZSTD"
    assert set(recorded["layer"].to_pylist()) == {1}
    assert set(recorded["model_version"].to_pylist()) == {"gpt2@tiny-test"}

    fresh = GPT2LMHeadModel(tiny_harvester.model.config).eval()
    fresh.load_state_dict(tiny_harvester.model.state_dict())
    check_parity(recorded.to_pylist(), saved_sequences, fresh, layer=1)


@pytest.mark.slow
def test_gpt2_small_layer_eight(tmp_path: Path) -> None:
    if not (local_dir := os.environ.get("PROBEKIT_GPT2_DIR")):
        pytest.skip("set PROBEKIT_GPT2_DIR to a local GPT-2 Small snapshot")
    config = load_config()
    model_config = config.model.model_copy(update={"local_dir": Path(local_dir), "device": "cpu"})
    texts = [
        f"A field report dated November {i} recorded the weather and local water levels."
        + (" The evening was cold." if i % 2 else "")
        for i in range(1, 18)
    ]
    acts, sidecar = tmp_path / "acts.parquet", tmp_path / "sequences.parquet"
    harvester = ActivationHarvester.from_config(model_config)
    harvester.harvest((SequenceInput(str(i), text) for i, text in enumerate(texts)), acts, sidecar)
    fresh = GPT2LMHeadModel.from_pretrained(local_dir, use_safetensors=True).eval()
    check_parity(
        pq.read_table(acts).to_pylist(), pq.read_table(sidecar).to_pylist(), fresh, layer=8
    )


def check_parity(
    rows: list[dict], sequences: list[dict], model: GPT2LMHeadModel, layer: int
) -> None:
    with torch.inference_mode():
        for seq in random.Random(2026).sample(sequences, 10):
            ids = seq["token_ids"]
            output = model.transformer(
                torch.tensor([ids]), output_hidden_states=True, use_cache=False
            )
            expected = output.hidden_states[layer + 1][0, 1:].numpy()
            matching = [row for row in rows if row["sequence_id"] == seq["sequence_id"]]
            assert [row["token_pos"] for row in matching] == list(range(1, len(ids)))
            assert np.allclose([row["activation"] for row in matching], expected, atol=1e-4)


def test_budget_and_duplicate_ids(tiny_harvester: ActivationHarvester, tmp_path: Path) -> None:
    acts, sidecar = tmp_path / "acts.parquet", tmp_path / "sequences.parquet"
    sequences = [
        SequenceInput("a", "a b c"),
        SequenceInput("b", "a b c d"),
        SequenceInput("c", "a b"),
    ]
    summary = tiny_harvester.harvest(sequences, acts, sidecar, max_tokens=5)
    assert summary.input_tokens == 5
    assert pq.read_table(sidecar)["token_ids"].to_pylist() == [[2, 3, 4], [2, 3]]
    with pytest.raises(ValueError, match="duplicate sequence id"):
        tiny_harvester.harvest([sequences[0]] * 2, acts, sidecar)
    assert pq.read_table(acts).num_rows == 3


def test_harvest_cli(tiny_harvester: ActivationHarvester, tmp_path: Path) -> None:
    import json

    import yaml
    from typer.testing import CliRunner

    from probekit.cli import app

    snapshot = tmp_path / "model"
    tiny_harvester.model.save_pretrained(snapshot)
    tiny_harvester.tokenizer.save_pretrained(snapshot)
    batch = [{"sequence_id": "example-1", "text": "a b c d"}]
    (tmp_path / "corpus.jsonl").write_text(json.dumps(batch[0]) + "\n", encoding="utf-8")
    values = load_config().model_dump(mode="json")
    values["model"] = tiny_harvester.config.model_dump(mode="json") | {"local_dir": "model"}
    values["corpus"] = {"jsonl": "corpus.jsonl", "max_tokens": 100}
    values["paths"]["activations"] = "activations.parquet"
    values["paths"]["sequences"] = "sequences.parquet"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    result = CliRunner().invoke(app, ["harvest", "--config", str(config_path)])

    assert result.exit_code == 0, result.exception
    assert json.loads(result.stdout) == {"sequences": 1, "input_tokens": 4, "activation_rows": 3}
    assert pq.read_table(tmp_path / "activations.parquet").num_rows == 3
    assert pq.read_table(tmp_path / "sequences.parquet")["sequence_id"].to_pylist() == ["example-1"]
