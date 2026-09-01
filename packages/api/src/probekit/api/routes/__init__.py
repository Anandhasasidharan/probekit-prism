"""API routes: probes, features, steering, monitor."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------------- models

class ProbeScoreRequest(BaseModel):
    text: str = Field(min_length=1)
    concept: str = "pii_detection"
    probe_kind: str = "sae"


class ProbeScoreResponse(BaseModel):
    concept: str
    probe_kind: str
    score: float
    threshold: float
    fired: bool


class FeatureOut(BaseModel):
    id: int
    layer: int
    name: str
    description: str
    space: str = "sae"
    detection_score: float | None = None
    verified: bool = False


class SteerRequest(BaseModel):
    prompt: str = Field(min_length=1)
    strategy: str = "suppress_risky"
    threshold: float | None = None


class MonitorRunResponse(BaseModel):
    alerts: list[dict]
    checked_features: int


def build_routers(state) -> list[APIRouter]:
    """Build routers wired to the shared AppState created in main.create_app."""

    probes = APIRouter(prefix="/probe", tags=["probes"])
    features = APIRouter(prefix="/features", tags=["features"])
    steer = APIRouter(prefix="/steer", tags=["steering"])
    monitor = APIRouter(prefix="/monitor", tags=["monitor"])

    # ----------------------------------------------------------------- probes

    @probes.post("/score", response_model=ProbeScoreResponse)
    def score(req: ProbeScoreRequest) -> ProbeScoreResponse:
        entry = state.probes.get((req.concept, req.probe_kind))
        if entry is None:
            raise HTTPException(404, f"No {req.probe_kind} probe trained for '{req.concept}'")
        probe, layer = entry
        vec = state.capture_activation(req.text, layer).reshape(1, -1)
        score_val = float(probe.predict_proba(vec)[0])
        thr = 0.7  # per-concept thresholds live in config.yaml under steering
        return ProbeScoreResponse(
            concept=req.concept, probe_kind=req.probe_kind, score=score_val,
            threshold=thr, fired=score_val >= thr,
        )

    # --------------------------------------------------------------- features

    @features.get("/search", response_model=list[FeatureOut])
    def search(q: str = "", layer: int | None = None, limit: int = 20):
        return [
            FeatureOut(
                id=f.id, layer=f.layer, name=f.name, description=f.description,
                space=f.space, detection_score=f.detection_score, verified=f.verified,
            )
            for f in state.registry.search_features(q, layer=layer, limit=limit)
        ]

    @features.get("/drift")
    def drifted():
        out = []
        for feature_id, space, hist in state.drift_history:
            if len(hist) >= 2:
                first, last = hist[0], hist[-1]
                delta = (last["frequency"] or 0) - (first["frequency"] or 0)
                if abs(delta) > 1e-9:
                    out.append({
                        "feature_id": feature_id, "space": space,
                        "frequency_first": first["frequency"],
                        "frequency_last": last["frequency"], "delta": delta,
                    })
        return out

    @features.post("/{feature_id}/verify")
    def verify(feature_id: int, layer: int | None = None):
        ok = state.registry.verify_feature(feature_id, layer=layer)
        if not ok:
            raise HTTPException(404, f"Feature {feature_id} not found")
        return {"verified": True, "feature_id": feature_id}

    # ---------------------------------------------------------------- steering

    @steer.post("/check_and_steer")
    def check_and_steer(req: SteerRequest):
        controller = state.build_controller()
        if controller is None:
            raise HTTPException(503, "Steering controller not configured (no model loaded)")
        result = controller.check_and_steer(
            req.prompt,
            threshold=req.threshold if req.threshold is not None else 0.7,
        )
        return result.as_dict()

    # ---------------------------------------------------------------- monitor

    @monitor.get("/drift", response_model=MonitorRunResponse)
    def run_drift() -> MonitorRunResponse:
        alerts = state.run_drift_check()
        return MonitorRunResponse(
            alerts=[{"kind": a.kind, "severity": a.severity, "message": a.message}
                    for a in alerts],
            checked_features=len(state.drift_history),
        )

    return [probes, features, steer, monitor]
