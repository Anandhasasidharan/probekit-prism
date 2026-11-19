"""Write, verify and load the ONNX inference bundle."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from probekit.config import Config, Settings
from probekit.onnx_utils import cpu_session
from probekit.probes.comparator import SPACES
from probekit.probes.datasets import Pooling, ProbeConfig, ProbeDataset, file_sha256
from probekit.probes.exporter import export_probe, predict_onnx
from probekit.probes.raw_trainer import LinearProbe
from probekit.sae.exporter import encode_onnx, export_encoder
from probekit.sae.trainer import SAECheckpoint, activation_batches, save_json

ENCODER_NAME = "sae-encoder.onnx"
MANIFEST_NAME = "bundle.json"
PARITY_ATOL = 1e-4


class Bundle(Settings):
    model_version: str
    layer: int
    max_length: int
    pooling: Pooling
    threshold: float = Field(gt=0, lt=1)
    sae_id: str
    k: int = Field(gt=0)
    concept: Literal["pii"]
    sha256: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "Bundle":
        bundle = cls.model_validate_json((path / MANIFEST_NAME).read_text(encoding="utf-8"))
        names = {ENCODER_NAME, *(f"{bundle.concept}-{space}.onnx" for space in SPACES)}
        if set(bundle.sha256) != names:
            raise ValueError("invalid ONNX bundle file list")
        if any(file_sha256(path / name) != sha for name, sha in bundle.sha256.items()):
            raise ValueError("ONNX files differ from the exported bundle manifest")
        return bundle


def parity(expected: np.ndarray, observed: np.ndarray, name: str) -> float:
    if expected.shape != observed.shape or not np.allclose(
        expected, observed, rtol=0, atol=PARITY_ATOL
    ):
        raise ValueError(f"{name} ONNX parity failed at absolute tolerance {PARITY_ATOL}")
    return float(np.abs(expected - observed).max())


def validate_probe_report(
    cfg: Config,
    space: str,
    concept: str,
    provenance: dict,
) -> None:
    comparison = cfg.paths.runs / "probe-comparison.json"
    single = cfg.paths.runs / f"probe-{concept}-{space}.json"
    existing = [path for path in (comparison, single) if path.is_file()]
    if not any(
        json.loads(path.read_text(encoding="utf-8"))["provenance"] == provenance
        for path in existing
    ):
        raise ValueError(f"training report does not match {space} probe inputs")


def export_bundle(
    cfg: Config,
    config: ProbeConfig,
    data: ProbeDataset,
    provenance: dict,
    concept: Literal["pii"],
) -> dict:
    torch.set_num_threads(cfg.model.cpu_threads)
    sae = SAECheckpoint.load(cfg.paths.sae_checkpoint)
    version = f"{cfg.model.name}@{cfg.model.revision}"
    if (sae.model_version, sae.layer) != (version, cfg.model.layer):
        raise ValueError("SAE checkpoint and GPT-2 extraction config do not match")
    matrices = activation_batches(
        cfg.paths.activations,
        256,
        sae.seed,
        sae.holdout_fraction,
        heldout=True,
    )
    sample_rows = []
    count = 0
    for matrix in matrices:
        sample_rows.append(matrix)
        count += len(matrix)
        if count >= 256:
            break
    if not sample_rows:
        raise ValueError("held-out activations are required for encoder parity")
    sample = torch.cat(sample_rows)[:256]
    with torch.inference_mode():
        expected = sae.sae.encode(sample).numpy()
    directory = cfg.paths.probe_dir
    errors = {}
    with TemporaryDirectory(dir=directory) as staging:
        root = Path(staging)
        encoder = root / ENCODER_NAME
        export_encoder(sae.sae, encoder)
        actual = encode_onnx(cpu_session(encoder), sample.numpy(), sae.sae.k)
        errors["sae_encoder_with_numpy_topk"] = parity(expected, actual, "SAE encoder")
        files = [encoder]
        for space in SPACES:
            validate_probe_report(cfg, space, concept, provenance)
            probe = LinearProbe.load(directory / f"{concept}-{space}.npz")
            matrix = data.features(space)[data.indices("test")]
            target = root / f"{concept}-{space}.onnx"
            export_probe(probe, target)
            errors[f"{space}_probe"] = parity(
                probe.predict(matrix),
                predict_onnx(cpu_session(target), matrix),
                space,
            )
            files.append(target)
        manifest = Bundle(
            model_version=version,
            layer=cfg.model.layer,
            max_length=cfg.model.max_length,
            pooling=config.pooling,
            threshold=config.threshold,
            sae_id=provenance["sae_id"],
            k=sae.sae.k,
            concept=concept,
            sha256={file.name: file_sha256(file) for file in files},
        )
        for file in files:
            file.replace(directory / file.name)
        staged = root / MANIFEST_NAME
        staged.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        staged.replace(directory / MANIFEST_NAME)
    result = {
        "absolute_tolerance": PARITY_ATOL,
        "max_absolute_errors": errors,
        "probe_test_examples": len(data.indices("test")),
        "encoder_tokens": len(sample),
        "bundle": manifest.model_dump(),
    }
    save_json(cfg.paths.runs / "onnx-parity.json", result)
    return result
