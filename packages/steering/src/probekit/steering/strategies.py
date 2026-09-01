"""Pre-configured steering strategies, loaded from GlobalConfig.steering."""

from __future__ import annotations

from probekit.steering.interventions import (
    SteeringIntervention,
    direction_from_clt,
    direction_from_sae,
)


def build_interventions(
    strategy_name: str,
    config_strategies: dict,
    direction_provider,
    feature_ids: list[int] | None = None,
) -> list[SteeringIntervention]:
    """Build interventions for a named strategy.

    direction_provider(feature_id) -> torch.Tensor. If feature_ids is given,
    one intervention per feature; otherwise the provider is expected to return
    a single default direction for any id.
    """
    if strategy_name not in config_strategies:
        raise KeyError(f"Unknown steering strategy: {strategy_name}")
    cfg = config_strategies[strategy_name]
    ids = feature_ids or [0]

    interventions = []
    for fid in ids:
        direction = direction_provider(fid)
        interventions.append(
            SteeringIntervention(
                direction,
                operation=cfg.intervention,
                strength=cfg.strength,
                threshold=cfg.threshold,
            )
        )
    return interventions


STRATEGY_DIRECTIONS = {
    "suppress_risky": ("sae",),
    "increase_honesty": ("sae",),
    "reduce_deception": ("sae",),
}


def make_direction_lookup(kind: str, sae=None, clt_stack=None):
    """Return direction_provider(feature_id) or (layer, feature) aware lookup."""
    if kind == "sae":
        assert sae is not None
        return lambda fid: direction_from_sae(sae, fid)
    if kind == "circuit":
        assert clt_stack is not None
        def _lookup(layer_feature: tuple[int, int]):
            layer, fid = layer_feature
            return direction_from_clt(clt_stack.get(layer), fid)
        return _lookup
    raise ValueError(f"Unknown direction source: {kind}")
