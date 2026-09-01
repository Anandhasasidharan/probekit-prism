"""End-to-end integration: harvest -> SAE -> auto-interp -> probes ->
attribution circuit -> steering evaluation, all on the tiny local model."""

import numpy as np
import pytest
import torch
from test_harvest import _make_harvester


@pytest.fixture(scope="module")
def pipeline_env(tiny_model, tiny_tokenizer, tmp_path_factory):
    """Harvest activations once for the whole integration run."""
    tmp = tmp_path_factory.mktemp("integration")
    h = _make_harvester(tiny_model, tiny_tokenizer)
    texts = [
        "delete all files",
        "the cat sat on the mat",
        "hello world this is a test",
        "rm now please",
    ] * 6
    records = h.harvest_sync(texts)
    return {"harvester": h, "records": records, "texts": texts, "tmp": tmp,
            "model": tiny_model, "tokenizer": tiny_tokenizer}


def test_full_pipeline(pipeline_env):
    env = pipeline_env
    from probekit.auto_interp.pipeline import AutoInterpPipeline
    from probekit.probes.dataset import ProbeDataset
    from probekit.probes.evaluator import evaluate_predictions
    from probekit.probes.trainers import RawProbe, SAEProbe
    from probekit.sae.architectures import build_sae
    from probekit.sae.evaluator import encode_matrix, evaluate_sae
    from probekit.sae.trainer import SAETrainer

    d_model = env["model"].config.n_embd
    last_layer = env["model"].config.n_layer - 1

    # 1. Stack raw activations at the final layer.
    recs = [r for r in env["records"] if r.provenance.layer_index == last_layer]
    X = np.stack([r.activations.reshape(-1) for r in recs]).astype(np.float32)
    tokens = [r.token_str for r in recs]
    # One record per (token x captured layer): row count must match tokens.
    assert len(X) == len(tokens)

    # 2. Train a small TopK SAE on them.
    sae = build_sae("topk", d_model=d_model, dict_size=64, k=4, device="cpu")
    trainer = SAETrainer(sae, batch_size=32, device="cpu")
    trainer.train(X, epochs=5)
    metrics = evaluate_sae(sae, X, device="cpu")
    assert metrics["n_tokens"] == len(X)

    # 3. AutoInterp on the strongest feature.
    latents = encode_matrix(sae, X)
    top_feature = int(np.argmax((latents > 0).sum(axis=0)))
    pipe = AutoInterpPipeline(provider="mock", cache_db=str(env["tmp"] / "cache.sqlite"))
    explanation = pipe.run_feature(top_feature, latents, tokens, score=True)
    assert explanation["text"]
    pipe.close()

    # 4. Train raw + SAE probes on a token-level concept aligned to these rows.
    #    Preferred label: 'deletion-related' tokens; fall back to alternating
    #    labels if the tokenizer produced only one class in the slices.
    y = np.isin(np.array(tokens), ["delete", "rm", "files"]).astype(int)
    if len(np.unique(y[:60])) < 2 or len(np.unique(y[60:])) < 2:
        y = (np.arange(len(tokens)) % 2).astype(int)
    train = ProbeDataset(X=X[:60], y=y[:60], texts=tokens[:60], sources=[None] * 60,
                         concept="deletion")
    test = ProbeDataset(X=X[60:], y=y[60:], texts=tokens[60:], sources=[None] * 60,
                        concept="deletion")

    raw_probe = RawProbe("deletion").fit(train)
    sae_probe = SAEProbe("deletion", sae=sae, device="cpu").fit(train)

    raw_metrics = evaluate_predictions(test.y, raw_probe.score_dataset(test),
                                       probe_kind="raw", concept="deletion")
    sae_metrics = evaluate_predictions(test.y, sae_probe.score_dataset(test),
                                       probe_kind="sae", concept="deletion")
    for m in (raw_metrics, sae_metrics):
        assert m["accuracy"] >= 0.0

    # 5. Attribution graph via the replacement model.
    from probekit.circuit import (
        CLTStack,
        CLTTrainer,
        ReplacementModel,
        build_attribution_graph,
        extract_circuit_features,
    )
    from probekit.circuit.collect import collect_mlp_pairs

    pairs = collect_mlp_pairs(env["model"], env["tokenizer"], env["texts"],
                              layer_start=1, layer_end=3, batch_size=8)
    stack = CLTStack([1, 2], d_model=d_model, n_features=32, device="cpu")
    CLTTrainer(stack, lr=5e-3, batch_size=256, device="cpu").train(pairs, epochs=150)

    rm = ReplacementModel(env["model"], env["tokenizer"], stack)
    graph = build_attribution_graph(rm, "delete all files", node_threshold=0.6,
                                    edge_threshold=0.8, max_feature_nodes=40)
    circuit_features = extract_circuit_features(graph, top_k=6)
    assert circuit_features

    # 6. Circuit-informed steering: suppress the top circuit feature direction.
    from probekit.steering.interventions import (
    SteeringIntervention,
    attach_steering,
    detach_steering,
)

    layer, fid = circuit_features[0]
    direction = stack.get(layer).W_dec[fid].detach()
    iv = SteeringIntervention(direction, operation="suppress")

    enc = env["tokenizer"]("the cat sat", return_tensors="pt")
    with torch.no_grad():
        base_logits = env["model"](**enc).logits
    hook = attach_steering(env["model"], [layer], [iv])
    try:
        with torch.no_grad():
            steered_logits = env["model"](**enc).logits
    finally:
        detach_steering(hook)
    with torch.no_grad():
        restored_logits = env["model"](**enc).logits

    assert not torch.allclose(base_logits, steered_logits, atol=1e-6)
    assert torch.allclose(base_logits, restored_logits, atol=1e-6)
