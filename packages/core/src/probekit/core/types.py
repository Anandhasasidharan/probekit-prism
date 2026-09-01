from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np


@dataclass
class Provenance:
    sequence_id: str
    token_position: int
    layer_index: int
    batch_index: int
    model_version: str
    chunk_id: str | None = None
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class ActivationRecord:
    provenance: Provenance
    activations: np.ndarray
    token_id: int
    token_str: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.activations.shape


@dataclass
class ConceptExample:
    text: str
    label: bool
    concept: str
    source: str | None = None


@dataclass
class FeatureActivation:
    feature_id: int
    layer: int
    activation_strength: float
    token: str
    token_position: int
    sequence_id: str


@dataclass
class Explanation:
    """AutoInterp output for a single feature."""

    feature_id: int
    layer: int
    text: str
    detection_score: float | None = None
    fuzzing_score: float | None = None
    explained_by: str = ""
    cost_usd: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class FeatureRecord:
    """Registry entry for a discovered (SAE or circuit) feature."""

    id: int
    layer: int
    name: str
    description: str
    space: str = "sae"  # "sae" | "circuit" | "raw"
    frequency: float | None = None
    detection_score: float | None = None
    fuzzing_score: float | None = None
    explained_by: str | None = None
    steerable: bool = True
    verified: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


# ---------------------------------------------------------------------------
# Circuit tracing types
# ---------------------------------------------------------------------------


@dataclass
class AttributionNode:
    feature_id: int
    layer: int
    activation: float
    attribution: float  # gradient-based influence on target logit


@dataclass
class AttributionEdge:
    src_feature: int
    src_layer: int
    dst_feature: int
    dst_layer: int
    weight: float


@dataclass
class AttributionGraph:
    input_token: str
    prompt: str
    target_logit: int
    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "prompt": self.prompt,
            "input_token": self.input_token,
            "target_logit": self.target_logit,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "top_nodes": [
                {
                    "feature": n.feature_id,
                    "layer": n.layer,
                    "attribution": round(n.attribution, 5),
                }
                for n in sorted(self.nodes, key=lambda n: -abs(n.attribution))[:10]
            ],
        }

    def to_json_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "input_token": self.input_token,
            "target_logit": self.target_logit,
            "nodes": [vars(n) for n in self.nodes],
            "edges": [vars(e) for e in self.edges],
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> AttributionGraph:
        return cls(
            prompt=d["prompt"],
            input_token=d["input_token"],
            target_logit=d["target_logit"],
            nodes=[AttributionNode(**n) for n in d["nodes"]],
            edges=[AttributionEdge(**e) for e in d["edges"]],
        )


@dataclass
class CircuitRecord:
    """A causally-validated set of features connected by attribution edges."""

    id: int | None
    name: str
    description: str
    feature_ids: list[int]
    layers: list[int]
    graphs: list[str] = field(default_factory=list)  # paths to serialized graphs
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


# ---------------------------------------------------------------------------
# Probe / monitoring result types
# ---------------------------------------------------------------------------


@dataclass
class ProbeMetrics:
    concept: str
    probe_kind: str  # "raw" | "sae" | "circuit"
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float | None = None
    n_train: int = 0
    n_test: int = 0

    def as_dict(self) -> dict:
        return vars(self)


@dataclass
class DriftAlert:
    kind: str  # feature_disappeared | feature_changed_meaning | probe_degraded | ...
    severity: str  # info | warning | critical
    message: str
    details: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class SteeringResult:
    action: str  # none | steered
    intervention: str | None
    before_score: float | None
    after_score: float | None
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return vars(self)
