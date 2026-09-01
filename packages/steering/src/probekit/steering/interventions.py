"""Steering interventions: activation modification at inference time.

Supported operations on a feature direction v (an SAE decoder row or a CLT
decoder row — causally validated when it comes from an attribution circuit):

  increase: x' = x + strength * ||v|| * v_hat   (amplify the feature)
  decrease: x' = x - strength * proj_v(x)       (subtract its component)
  suppress: x' = x - proj_v(x)                  (remove entirely)
  inject:   x' = x + alpha * v                  (add raw pattern)

Applied via forward hooks on decoder blocks (GPT-2 / Llama style).
"""

from __future__ import annotations

import torch


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp(min=1e-8)


class SteeringIntervention:
    def __init__(self, direction: torch.Tensor, operation: str = "suppress",
                 strength: float = 1.0, threshold: float | None = None):
        if operation not in ("increase", "decrease", "suppress", "inject"):
            raise ValueError(f"Unknown steering operation: {operation}")
        self.direction = normalize(direction.detach().float())
        self.operation = operation
        self.strength = strength
        self.threshold = threshold

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (..., d_model) residual stream."""
        d = self.direction.to(hidden.device, hidden.dtype)
        proj = (hidden @ d).unsqueeze(-1) * d.unsqueeze(0) if hidden.dim() > 1 else (hidden @ d) * d
        if self.operation == "increase":
            return hidden + self.strength * proj
        if self.operation == "decrease":
            return hidden - self.strength * proj
        if self.operation == "suppress":
            return hidden - proj
        # inject: add the raw (normalized) feature direction scaled by strength
        scale = float(hidden.norm(dim=-1, keepdim=True).mean().item()) if hidden.dim() > 1 else 1.0
        return hidden + self.strength * scale * d


class SteeringHook:
    """Attachable hook that applies one or more interventions to a block's output."""

    def __init__(self, interventions: list[SteeringIntervention]):
        self.interventions = interventions
        self.applied_count = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __call__(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for iv in self.interventions:
            hidden = iv.apply(hidden)
            self.applied_count += 1
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


def attach_steering(
    model,
    layer_ids: list[int],
    interventions: list[SteeringIntervention],
) -> SteeringHook:
    """Register a steering hook on the given decoder blocks.

    Returns the hook; call ``hook.detach()`` to remove it."""
    blocks = _get_blocks(model)
    hook = SteeringHook(interventions)
    hook._handles = [blocks[layer].register_forward_hook(hook) for layer in layer_ids]
    return hook


def detach_steering(hook: SteeringHook) -> None:
    for h in hook._handles:
        h.remove()
    hook._handles.clear()


def direction_from_sae(sae, feature_id: int) -> torch.Tensor:
    return sae.W_dec[feature_id].detach()


def direction_from_clt(transcoder, feature_id: int) -> torch.Tensor:
    return transcoder.W_dec[feature_id].detach()


def directions_from_circuit(clt_stack, layer_feature_pairs: list[tuple[int, int]]):
    """Group causally-validated features by layer -> {layer: [directions]}."""
    grouped: dict[int, list[torch.Tensor]] = {}
    for layer, feat in layer_feature_pairs:
        transcoder = clt_stack.get(layer)
        grouped.setdefault(layer, []).append(direction_from_clt(transcoder, feat))
    return grouped


def _get_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported architecture")
