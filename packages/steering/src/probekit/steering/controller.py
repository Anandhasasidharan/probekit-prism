"""Probe-driven auto-steering controller.

Flow (Goodfire AutoSteer pattern): score the prompt's activations with a
trained probe; if the risky-concept probability crosses the strategy threshold,
apply the configured steering intervention and re-score to confirm the shift.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from probekit.core.types import SteeringResult
from probekit.steering.interventions import (
    SteeringIntervention,
    attach_steering,
    detach_steering,
)


class SteeringController:
    def __init__(
        self,
        model,
        tokenizer,
        probe=None,
        *,
        intervention: SteeringIntervention | None = None,
        layers: list[int] | None = None,
        probe_layer: int | None = None,
        device: str | None = None,
        capture_position: str = "last",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe  # object with predict_proba(X)
        self.intervention = intervention
        self.layers = layers or []
        self.probe_layer = probe_layer
        self.device = device or next(model.parameters()).device
        self.capture_position = capture_position

    # ------------------------------------------------------------- activation

    def capture_activation(self, prompt: str, layer: int) -> np.ndarray:
        """Forward pass capturing residual stream at `layer`, last token."""
        blocks = _blocks(self.model)
        stored = {}

        def hook(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output
            stored["h"] = h.detach()

        handle = blocks[layer].register_forward_hook(hook)
        try:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**enc)
        finally:
            handle.remove()
        vec = stored["h"][0, -1] if self.capture_position == "last" else stored["h"][0, 0]
        return vec.float().cpu().numpy().reshape(-1)

    # ---------------------------------------------------------------- control

    def check_and_steer(self, prompt: str, threshold: float = 0.7) -> SteeringResult:
        """Score prompt with the probe; steer if above threshold; re-score."""
        if self.probe is None or self.probe_layer is None:
            raise ValueError("Controller needs a probe and probe_layer")

        before_vec = self.capture_activation(prompt, self.probe_layer)
        before_score = float(self.probe.predict_proba(before_vec.reshape(1, -1))[0])

        if before_score < threshold or self.intervention is None:
            reason = (
                "below_threshold" if before_score < threshold else "no_intervention"
            )
            return SteeringResult(
                "none", None, before_score, before_score, {"reason": reason}
            )

        hook = attach_steering(self.model, self.layers, [self.intervention])
        try:
            after_vec = self.capture_activation(prompt, self.probe_layer)
        finally:
            detach_steering(hook)
        after_score = float(self.probe.predict_proba(after_vec.reshape(1, -1))[0])

        return SteeringResult(
            action="steered",
            intervention=f"{self.intervention.operation}@layers{self.layers}",
            before_score=before_score,
            after_score=after_score,
            details={"delta": after_score - before_score},
        )

    def generate(self, prompt: str, max_new_tokens: int = 32, steer: bool = False) -> str:
        """Generate text, optionally with steering hooks attached."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        intervention = cast(SteeringIntervention, self.intervention)
        hook = (
            attach_steering(self.model, self.layers, [intervention])
            if steer and self.intervention is not None
            else None
        )
        try:
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            if hook is not None:
                detach_steering(hook)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


def _blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported architecture")
