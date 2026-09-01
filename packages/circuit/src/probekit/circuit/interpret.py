"""Automated attribution-graph interpretation (Hanna & Ameisen 2026 style).

A single graph for a short prompt can contain thousands of feature nodes and
take hours to read manually. `describe_graph` produces a deterministic
structured summary; `narrate_graph` optionally passes that summary to an LLM
explainer (probe-prompting) for a natural-language circuit description.
"""

from __future__ import annotations

from collections import defaultdict

from probekit.core.types import AttributionGraph


def describe_graph(graph: AttributionGraph, feature_names: dict[int, str] | None = None) -> str:
    """Deterministic structured summary of an attribution graph."""
    feature_names = feature_names or {}
    by_layer: dict[int, list] = defaultdict(list)
    for n in graph.nodes:
        by_layer[n.layer].append(n)
    for li in by_layer:
        by_layer[li].sort(key=lambda n: -abs(n.attribution))

    lines = [
        f"Attribution graph summary — prompt: {graph.prompt!r}",
        f"Target token: {graph.input_token!r} | nodes: {len(graph.nodes)} "
        f"| edges: {len(graph.edges)}",
        "",
    ]
    for layer in sorted(by_layer):
        lines.append(f"Layer {layer}:")
        for n in by_layer[layer][:8]:
            name = feature_names.get(n.feature_id, f"feature_{n.feature_id}")
            sign = "+" if n.attribution >= 0 else "-"
            lines.append(
                f"  [{sign}] {name} (id={n.feature_id}) "
                f"act={n.activation:.3f} attr={n.attribution:+.4f}"
            )

    if graph.edges:
        top_edges = sorted(graph.edges, key=lambda e: -e.weight)[:10]
        lines.append("")
        lines.append("Strongest edges:")
        for e in top_edges:
            src = feature_names.get(e.src_feature, f"f{e.src_feature}")
            dst = feature_names.get(e.dst_feature, f"f{e.dst_feature}")
            lines.append(f"  L{e.src_layer}:{src} -> L{e.dst_layer}:{dst} w={e.weight:.4f}")

    return "\n".join(lines)


def narrate_graph(
    graph: AttributionGraph,
    explainer,
    feature_names: dict[int, str] | None = None,
) -> str:
    """LLM narration of the structured summary (probe-prompting pattern)."""
    summary = describe_graph(graph, feature_names)
    question = (
        "You are reading an attribution graph from a language model. "
        "Describe the circuit: which features drive the output token, through "
        "which path, and what does this suggest the model is computing?\n\n"
        f"{summary}"
    )
    return explainer.answer(question)


def extract_circuit_features(graph: AttributionGraph, top_k: int = 16) -> list[tuple[int, int]]:
    """Top-(layer, feature_id) pairs by |attribution| — a causally-validated
    candidate circuit for steering."""
    ranked = sorted(graph.nodes, key=lambda n: -abs(n.attribution))
    return [(n.layer, n.feature_id) for n in ranked[:top_k]]
