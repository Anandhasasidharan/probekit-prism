"""Parquet (zstd) storage for activation records with full round-trip fidelity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from probekit.core.types import ActivationRecord, Provenance


def write_activation_parquet(
    records: list[ActivationRecord],
    path: str | Path,
    compression: str = "zstd",
) -> None:
    if not records:
        raise ValueError("No records to write")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "sequence_id": [r.provenance.sequence_id for r in records],
            "token_position": [r.provenance.token_position for r in records],
            "layer_index": [r.provenance.layer_index for r in records],
            "batch_index": [r.provenance.batch_index for r in records],
            "model_version": [r.provenance.model_version for r in records],
            "chunk_id": [r.provenance.chunk_id for r in records],
            "recorded_at": [r.provenance.recorded_at for r in records],
            "token_id": [r.token_id for r in records],
            "token_str": [r.token_str for r in records],
            "activation": [
                r.activations.astype(np.float32).tobytes() for r in records
            ],
            "activation_shape": [str(list(r.activations.shape)) for r in records],
        }
    )
    pq.write_table(table, str(path), compression=compression)


def read_activation_parquet(path: str | Path) -> list[ActivationRecord]:
    table = pq.read_table(str(path))
    records: list[ActivationRecord] = []
    for i in range(len(table)):
        prov = Provenance(
            sequence_id=table["sequence_id"][i].as_py(),
            token_position=table["token_position"][i].as_py(),
            layer_index=table["layer_index"][i].as_py(),
            batch_index=table["batch_index"][i].as_py(),
            model_version=table["model_version"][i].as_py(),
            chunk_id=table["chunk_id"][i].as_py(),
            recorded_at=table["recorded_at"][i].as_py(),
        )
        shape = eval(table["activation_shape"][i].as_py())  # noqa: S307 - own data
        acts = np.frombuffer(table["activation"][i].as_py(), dtype=np.float32)
        records.append(
            ActivationRecord(
                provenance=prov,
                activations=acts.reshape(shape),
                token_id=table["token_id"][i].as_py(),
                token_str=table["token_str"][i].as_py(),
            )
        )
    return records


def load_layer_matrix(path: str | Path, layer: int) -> tuple[np.ndarray, list[str]]:
    """Convenience: stacked activation matrix (n_tokens, d_model) + tokens."""
    records = read_activation_parquet(path)
    recs = [r for r in records if r.provenance.layer_index == layer]
    matrix = np.stack([r.activations.reshape(-1) for r in recs])
    tokens = [r.token_str for r in recs]
    return matrix, tokens
