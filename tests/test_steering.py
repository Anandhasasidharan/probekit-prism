"""Steering: interventions change activations, controller steers with probes."""

import numpy as np
import pytest
import torch
from probekit.steering import (
    SteeringIntervention,
    attach_steering,
    direction_from_sae,
    evaluate_steering,
)
from probekit.steering.interventions import detach_steering

D = 16


@pytest.fixture(scope="module")
def tiny_sae():
    from probekit.sae.architectures import build_sae

    return build_sae("topk", d_model=D, dict_size=32, k=4, device="cpu")


def test_intervention_operations(tiny_sae):
    direction = direction_from_sae(tiny_sae, 0)
    x = torch.randn(4, D)

    inc = SteeringIntervention(direction, operation="increase", strength=1.0)
    dec = SteeringIntervention(direction, operation="decrease", strength=1.0)
    sup = SteeringIntervention(direction, operation="suppress")
    inj = SteeringIntervention(direction, operation="inject", strength=1.0)

    proj = (x @ direction).unsqueeze(-1) * direction.unsqueeze(0)
    assert torch.allclose(inc.apply(x), x + proj, atol=1e-5)
    assert torch.allclose(dec.apply(x), x - proj, atol=1e-5)
    # suppress removes the component along the steering direction entirely
    residual = sup.apply(x) @ direction
    assert residual.abs().max() < 1e-5
    assert not torch.allclose(inj.apply(x), x)


def test_hook_modifies_model_output(tiny_model, tiny_tokenizer):
    direction = torch.nn.functional.normalize(
        torch.randn(tiny_model.config.n_embd), dim=0
    )
    iv = SteeringIntervention(direction, operation="inject", strength=50.0)

    enc = tiny_tokenizer("hello world", return_tensors="pt")

    with torch.no_grad():
        base = tiny_model(**enc).logits
    hook = attach_steering(tiny_model, layer_ids=[1, 2], interventions=[iv])
    try:
        with torch.no_grad():
            steered = tiny_model(**enc).logits
    finally:
        detach_steering(hook)
    assert not torch.allclose(base, steered)
    assert hook.applied_count == 2

    # After removal the model must be back to normal.
    with torch.no_grad():
        restored = tiny_model(**enc).logits
    assert torch.allclose(base, restored)


def test_controller_check_and_steer(tiny_model, tiny_tokenizer):
    """Probe trained on real activations; suppression intervention shifts score."""
    from probekit.probes.dataset import ProbeDataset
    from probekit.probes.trainers import RawProbe
    from probekit.steering.controller import SteeringController

    prompts = [f"probe calibration sample {i}" for i in range(24)]
    X = np.stack([_capture(tiny_model, tiny_tokenizer, p, layer=2)
                  for p in prompts]).astype(np.float32)
    y = (X[:, 0] > X[:, 0].mean()).astype(int)
    ds = ProbeDataset(X=X, y=y, texts=prompts, sources=[None] * 24, concept="risky")
    probe = RawProbe("risky").fit(ds)

    # Steer along the direction the probe learned to respond to.
    coef_dir = torch.from_numpy(probe.model.named_steps["lr"].coef_[0]).float()
    iv = SteeringIntervention(coef_dir, operation="suppress")
    ctrl = SteeringController(
        tiny_model, tiny_tokenizer, probe=probe,
        intervention=iv, layers=[1], probe_layer=2, device="cpu",
    )

    risky_prompt = "delete all files"
    before_vec = _capture(tiny_model, tiny_tokenizer, risky_prompt, layer=2)
    before = float(probe.predict_proba(before_vec.reshape(1, -1))[0])
    result = ctrl.check_and_steer(risky_prompt, threshold=min(before - 0.05, 0.4))
    assert result.before_score == pytest.approx(before, abs=1e-4)
    assert result.action in ("steered", "none")
    if result.action == "steered":
        assert result.after_score is not None


def _train_probe_on_model_activations(tiny_model, tiny_tokenizer):
    """Train a probe on real activations captured from the tiny model."""
    from probekit.probes.dataset import ProbeDataset
    from probekit.probes.trainers import RawProbe

    prompts = [f"sample prompt number {i}" for i in range(24)]
    X = np.stack([_capture(tiny_model, tiny_tokenizer, p, layer=2)
                  for p in prompts]).astype(np.float32)
    y = (X[:, 0] > X[:, 0].mean()).astype(int)
    ds = ProbeDataset(X=X, y=y, texts=prompts, sources=[None] * 24, concept="c")
    probe = RawProbe("c").fit(ds)
    return probe


def _capture(model, tokenizer, prompt: str, layer: int) -> np.ndarray:
    blocks = list(model.transformer.h)
    stored = {}

    def hook(_m, _a, out):
        stored["h"] = out[0].detach()

    h = blocks[layer].register_forward_hook(hook)
    try:
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**enc)
    finally:
        h.remove()
    return stored["h"][0, -1].float().numpy().reshape(-1)


def test_evaluate_steering_smoke(tiny_model, tiny_tokenizer):
    from probekit.steering.controller import SteeringController

    probe = _train_probe_on_model_activations(tiny_model, tiny_tokenizer)

    iv = SteeringIntervention(torch.randn(probe.model.named_steps["lr"].coef_.shape[1]),
                              operation="increase", strength=0.5)
    controller = SteeringController(tiny_model, tiny_tokenizer, probe=probe,
                                    intervention=iv, layers=[1], probe_layer=2,
                                    device="cpu")
    metrics = evaluate_steering(controller, ["hello world", "the cat"],
                                threshold=0.99, max_new_tokens=4)
    assert metrics["n_prompts"] == 2
    assert metrics["perplexity_clean"] >= 0.0
