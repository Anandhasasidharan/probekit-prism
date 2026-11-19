import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from probekit.harvest.harvester import ActivationHarvester


@pytest.mark.slow
def test_shell_pipeline(tiny_harvester: ActivationHarvester, tmp_path: Path) -> None:
    model = tmp_path / "model"
    tiny_harvester.model.save_pretrained(model)
    tiny_harvester.tokenizer.save_pretrained(model)
    output = tmp_path / "out"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            "scripts/run_pipeline.sh",
            "--output",
            str(output),
            "--model-dir",
            str(model),
            "--layer",
            str(tiny_harvester.config.layer),
        ],
        cwd=root,
        env=os.environ | {"PROBEKIT_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report_dir = output / "runs"
    report = json.loads((report_dir / "demo-smoke.json").read_text())
    assert report["health"] == {"status": "ok", "device": "cpu"}
    assert report["probe"]["latency_ms"] > 0
    assert report["features_returned"] > 0
    assert report["elapsed_seconds"] < 300
    for name in (
        "sae-training.json",
        "sae-metrics.json",
        "autointerp.csv",
        "autointerp-scores.png",
        "probe-comparison.json",
        "onnx-parity.json",
        "latency.json",
    ):
        assert (report_dir / name).stat().st_size > 0
    assert (output / "probes/bundle.json").is_file()
    assert (output / "features.sqlite").is_file()
