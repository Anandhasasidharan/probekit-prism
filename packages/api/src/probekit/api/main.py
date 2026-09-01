"""ProbeKit unified FastAPI application.

The app is designed to work in three modes:
  1. Full mode: model + SAE + probes loaded (GPU/CPU inference endpoints).
  2. Registry-only mode: feature catalog + drift endpoints, no model needed.
  3. Demo/test mode: everything mocked or absent; health/readiness still work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from fastapi import FastAPI
from probekit.api.middleware import LoggingMiddleware, RateLimitMiddleware
from probekit.api.routes import build_routers
from probekit.core.config import GlobalConfig
from probekit.core.device import device_summary, get_device

if TYPE_CHECKING:
    from probekit.registry.feature_registry import FeatureRegistry

logger = logging.getLogger("probekit.api")


class AppState:
    """Shared state wired into route handlers."""

    def __init__(self, config: GlobalConfig):
        self.config = config
        self.device = get_device(None)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.sae = None
        self.probes: dict[tuple[str, str], tuple] = {}  # (concept, kind) -> (probe, layer)
        self.registry: FeatureRegistry | None = None
        self.monitored_features: list[int] = []
        self.drift_history: list[tuple[int, str, list[dict]]] = []
        self._controller = None

    # ------------------------------------------------------------ lazy loading

    def load_model(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self.config.harvest.model_name
        logger.info("Loading %s on %s", name, device_summary(self.device))
        tokenizer: Any = AutoTokenizer.from_pretrained(name)  # type: ignore[no-any-return]
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(name)  # type: ignore[no-any-return]
        model.to(device=str(self.device))
        self.model = model.eval()

    def capture_activation(self, text: str, layer: int) -> np.ndarray:
        """Residual stream at `layer`, last token — the probing position."""
        assert self.model is not None and self.tokenizer is not None
        blocks = (
            list(self.model.transformer.h)
            if hasattr(self.model, "transformer")
            else list(self.model.model.layers)
        )
        stored: dict[str, torch.Tensor] = {}

        def hook(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output
            stored["h"] = h.detach()

        handle = blocks[layer].register_forward_hook(hook)
        try:
            enc = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**enc)
        finally:
            handle.remove()
        return stored["h"][0, -1].float().cpu().numpy().reshape(-1)

    def build_controller(self):
        """Build a steering controller from the default strategy + SAE."""
        from probekit.steering.controller import SteeringController
        from probekit.steering.interventions import direction_from_sae
        from probekit.steering.strategies import build_interventions

        if self.model is None or self.sae is None:
            return None
        strategy = self.config.steering.default_strategy
        interventions = build_interventions(
            strategy,
            {k: v for k, v in self.config.steering.strategies.items()},
            lambda fid: direction_from_sae(self.sae, fid),
        )
        return SteeringController(
            self.model,
            self.tokenizer,
            probe=self._probe_for_strategy(),
            intervention=interventions[0],
            layers=[min(6, self.model.config.n_layer - 1)],
            probe_layer=min(9, self.model.config.n_layer - 1),
            device=str(self.device),
        )

    def _probe_for_strategy(self):
        concept = self.config.probes.concepts[0]
        entry = self.probes.get((concept, "sae")) or self.probes.get((concept, "raw"))
        return entry[0] if entry else None

    # ------------------------------------------------------------------ drift

    def run_drift_check(self):
        from probekit.monitor.drift import FrequencyDriftDetector

        alerts = []
        for feature_id, space, hist in self.drift_history:
            if len(hist) < 2:
                continue
            freqs = [h["frequency"] or 0.0 for h in hist]
            base_mean = float(np.mean(freqs[:-1]))
            base_std = float(np.std(freqs[:-1])) or abs(base_mean) * 0.1 or 1e-6
            detector = FrequencyDriftDetector(base_mean, base_std,
                                              self.config.monitor.drift_threshold_std)
            alert = detector.check(freqs[-1], feature_id=feature_id)
            if alert:
                alerts.append(alert)
        return alerts


def create_app(config_path: str | Path | None = None) -> FastAPI:
    config = GlobalConfig.from_yaml(config_path or "./config.yaml")
    state = AppState(config)
    try:
        from probekit.registry.feature_registry import FeatureRegistry

        state.registry = FeatureRegistry(config.registry.database_url)
    except Exception as exc:  # pragma: no cover - registry optional at boot
        logger.warning("Registry unavailable: %s", exc)

    app = FastAPI(title="ProbeKit v2", version="0.2.0",
                  description="Production interpretability infrastructure")
    app.state.probekit = state
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=config.api.rate_limit_per_minute)

    for router in build_routers(state):
        app.include_router(router)

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready():
        checks = {
            "registry": state.registry is not None,
            "model": state.model is not None,
            "probes": bool(state.probes),
        }
        return {"ready": all(checks.values()), "checks": checks}

    # Serve the unique PRISM UI at / — MOUNTED VERY LAST so every API route wins
    ui_dir = Path(__file__).resolve().parents[5] / "ui"
    if not ui_dir.exists():
        ui_dir = Path.cwd() / "ui"
    if ui_dir.exists() and (ui_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="prism-ui")

    return app
