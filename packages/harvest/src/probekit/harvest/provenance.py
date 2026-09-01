import uuid

from probekit.core.types import ActivationRecord, Provenance


class ProvenanceTracker:
    """Assigns unique sequence ids and builds provenance metadata for records.

    `stitch_chunks` reassembles records from chunked prefill passes into a
    single ordered stream using the stored token positions and layers.
    """

    def __init__(self, model_version: str):
        self._model_version = model_version
        self._sequence_counter = 0

    def next_sequence(self) -> str:
        self._sequence_counter += 1
        return f"seq-{self._sequence_counter:06d}-{uuid.uuid4().hex[:8]}"

    def record(
        self,
        *,
        sequence_id: str | None = None,
        token_position: int,
        layer_index: int,
        batch_index: int,
        chunk_id: str | None = None,
    ) -> Provenance:
        return Provenance(
            sequence_id=sequence_id or self.next_sequence(),
            token_position=token_position,
            layer_index=layer_index,
            batch_index=batch_index,
            model_version=self._model_version,
            chunk_id=chunk_id,
        )

    @staticmethod
    def stitch_chunks(records: list[ActivationRecord]) -> list[ActivationRecord]:
        if not records:
            return records
        return sorted(
            records,
            key=lambda r: (
                r.provenance.sequence_id,
                r.provenance.token_position,
                r.provenance.layer_index,
            ),
        )
