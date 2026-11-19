"""Circuit tracing: CLT training, replacement model, attribution graphs."""

import pytest
import torch
from probekit.circuit import (
    CLTStack,
    CLTTrainer,
    ReplacementModel,
    build_attribution_graph,
    collect_mlp_pairs,
    describe_graph,
    extract_circuit_features,
)


@pytest.fixture(scope="module")
def trained_stack(tiny_model, tiny_tokenizer):
    texts = [
        "the cat sat on the mat",
        "delete all files",
        "hello world this is a test",
        "the dog ran in the park",
    ] * 3
    pairs = collect_mlp_pairs(tiny_model, tiny_tokenizer, texts,
                              layer_start=1, layer_end=3, batch_size=8)
    stack = CLTStack([1, 2], d_model=tiny_model.config.n_embd, n_features=32, device="cpu")
    trainer = CLTTrainer(stack, lr=5e-3, batch_size=128, device="cpu")
    trainer.train(pairs, epochs=150)
    return stack, pairs


def test_collect_pairs_shapes(tiny_model, tiny_tokenizer):
    pairs = collect_mlp_pairs(tiny_model, tiny_tokenizer,
                              ["hello world"], layer_start=0, layer_end=2)
    assert set(pairs.keys()) == {0, 1}
    for layer, (xs, ys) in pairs.items():
        n_tokens = len(tiny_tokenizer("hello world")["input_ids"])
        assert xs.shape[0] >= n_tokens
        assert xs.shape == ys.shape
        assert xs.shape[-1] == tiny_model.config.n_embd


def test_clt_learns_to_reproduce_mlp(trained_stack):
    stack, pairs = trained_stack
    for layer in (1, 2):
        xs, ys = pairs[layer]
        tc = stack.get(layer)
        with torch.no_grad():
            recon, _ = tc(torch.from_numpy(xs[:256]))
        mse = torch.nn.functional.mse_loss(recon, torch.from_numpy(ys[:256]))
        base = torch.nn.functional.mse_loss(torch.zeros_like(recon),
                                            torch.from_numpy(ys[:256]))
        assert float(mse) < float(base), f"CLT at layer {layer} worse than zero baseline"


def test_replacement_run_and_attribution(trained_stack, tiny_model, tiny_tokenizer):
    stack, _ = trained_stack
    tiny_model_cpu = tiny_model
    rm = ReplacementModel(tiny_model_cpu, tiny_tokenizer, stack)
    graph = build_attribution_graph(
        rm, "the cat sat on the mat",
        max_feature_nodes=50, node_threshold=0.5, edge_threshold=0.7,
    )
    assert len(graph.nodes) > 0
    assert all(abs(n.attribution) > 0 or n.activation > 0 for n in graph.nodes)

    summary = describe_graph(graph)
    assert "Attribution graph summary" in summary

    circuit_feats = extract_circuit_features(graph, top_k=5)
    assert 1 <= len(circuit_feats) <= 5
    for layer, feat in circuit_feats:
        assert layer in (1, 2)
        assert 0 <= feat < 32


def test_graph_json_roundtrip(trained_stack, tiny_model, tiny_tokenizer):
    from probekit.core.types import AttributionGraph

    stack, _ = trained_stack
    rm = ReplacementModel(tiny_model, tiny_tokenizer, stack)
    g = build_attribution_graph(rm, "delete all files", node_threshold=0.5,
                                edge_threshold=0.9, top_edges_per_node=4)
    d = g.to_json_dict()
    g2 = AttributionGraph.from_json_dict(d)
    assert g2.prompt == g.prompt
    assert len(g2.nodes) == len(g.nodes)
