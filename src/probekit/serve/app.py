"""FastAPI app for probing text and listing the strongest SAE features."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from probekit.autointerp.cache import LabelCache
from probekit.autointerp.collector import AutoInterpConfig
from probekit.autointerp.explainer import OfflineExplainer
from probekit.config import Config, Settings
from probekit.harvest.harvester import ActivationHarvester, SequenceInput
from probekit.onnx_utils import cpu_session
from probekit.probes.comparator import SPACES
from probekit.probes.datasets import Space, pool_tokens
from probekit.probes.exporter import predict_onnx
from probekit.sae.architectures import FIRING_THRESHOLD
from probekit.sae.exporter import encode_onnx
from probekit.serve.bundle import ENCODER_NAME, Bundle


class ServeConfig(Settings):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, le=65535)
    probe_space: Space = "raw"
    feature_limit: int = Field(default=10, gt=0)
    max_text_chars: int = Field(default=20000, gt=0)
    onnx_threads: int = Field(default=1, gt=0)
    profile_warmups: int = Field(default=3, ge=0)
    profile_repeats: int = Field(default=20, gt=0)
    offline_labels: bool = False


class TextInput(BaseModel):
    text: str = Field(min_length=1)


class ProbeInput(TextInput):
    concept: str


class ProbeOutput(BaseModel):
    score: float
    label: int
    latency_ms: float


class FiredFeature(BaseModel):
    feature_id: int
    activation: float
    label: str | None


@dataclass(frozen=True)
class Timings:
    gpt2_forward_ms: float
    onnx_probe_only_ms: float
    application_ms: float


class ProbeRuntime:
    def __init__(self, config: Config, settings: ServeConfig, labels: AutoInterpConfig) -> None:
        self.settings = settings
        directory = config.paths.probe_dir
        self.bundle = Bundle.load(directory)
        identity = (
            f"{config.model.name}@{config.model.revision}",
            config.model.layer,
            config.model.max_length,
        )
        if identity != (self.bundle.model_version, self.bundle.layer, self.bundle.max_length):
            raise ValueError("ONNX bundle and GPT-2 extraction config do not match")
        self.harvester = ActivationHarvester.from_config(config.model)
        self.encoder = cpu_session(directory / ENCODER_NAME, settings.onnx_threads)
        self.probes = {
            space: cpu_session(
                directory / f"{self.bundle.concept}-{space}.onnx", settings.onnx_threads
            )
            for space in SPACES
        }
        self.labels = {}
        if config.paths.feature_db.is_file():
            provider = OfflineExplainer.model if settings.offline_labels else labels.model
            self.labels = LabelCache(
                config.paths.feature_db, self.bundle.sae_id, provider, labels.prompt_version
            ).descriptions()

    def residuals(self, text: str, measure: bool = False) -> tuple[np.ndarray, float]:
        if not text.strip() or len(text) > self.settings.max_text_chars:
            raise ValueError("text is blank or exceeds the configured character limit")
        sequences = self.harvester.tokenize([SequenceInput("request", text)])
        if len(sequences[0].token_ids) < 2:
            raise ValueError("text must contain at least two input tokens")
        matrices, elapsed = self.harvester.capture_with_timing(sequences, measure=measure)
        return matrices[0].numpy(), elapsed

    def probe(self, text: str, space: Space, measure: bool = False) -> tuple[ProbeOutput, Timings]:
        started = perf_counter()
        residual, forward_ms = self.residuals(text, measure)
        matrix = encode_onnx(self.encoder, residual, self.bundle.k) if space == "sae" else residual
        pooled = pool_tokens(torch.from_numpy(matrix), self.bundle.pooling).numpy()[None, :]
        probe_started = perf_counter()
        score = float(predict_onnx(self.probes[space], pooled)[0])
        ended = perf_counter()
        latency = (ended - started) * 1000
        return (
            ProbeOutput(score=score, label=int(score >= self.bundle.threshold), latency_ms=latency),
            Timings(forward_ms, (ended - probe_started) * 1000, latency),
        )

    def features(self, text: str) -> list[FiredFeature]:
        residual, _ = self.residuals(text)
        activations = encode_onnx(self.encoder, residual, self.bundle.k).max(axis=0)
        order = np.argsort(-activations, kind="stable")
        return [
            FiredFeature(
                feature_id=int(i), activation=float(activations[i]), label=self.labels.get(int(i))
            )
            for i in order[: self.settings.feature_limit]
            if activations[i] > FIRING_THRESHOLD
        ]


def create_app(config: Config, settings: ServeConfig, labels: AutoInterpConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.runtime = ProbeRuntime(config, settings, labels)
        yield

    app = FastAPI(title="ProbeKit", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "device": str(request.app.state.runtime.harvester.device)}

    @app.post("/probe", response_model=ProbeOutput)
    def probe(body: ProbeInput, request: Request) -> ProbeOutput:
        runtime: ProbeRuntime = request.app.state.runtime
        if body.concept != runtime.bundle.concept:
            raise HTTPException(status_code=404, detail="Unknown probe concept")
        try:
            return runtime.probe(body.text, settings.probe_space)[0]
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/features/activate", response_model=list[FiredFeature])
    def features(body: TextInput, request: Request) -> list[FiredFeature]:
        runtime: ProbeRuntime = request.app.state.runtime
        try:
            return runtime.features(body.text)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def profile_runtime(runtime: ProbeRuntime, texts: list[str]) -> dict:
    if not texts:
        raise ValueError("provide texts for profiling")
    settings = runtime.settings
    profiles = {}
    for space in SPACES:
        for i in range(settings.profile_warmups):
            runtime.probe(texts[i % len(texts)], space)
        measurements = [
            asdict(runtime.probe(texts[i % len(texts)], space, measure=True)[1])
            for i in range(settings.profile_repeats)
        ]
        summary = {}
        for key in measurements[0]:
            p50, p95 = np.percentile([row[key] for row in measurements], [50, 95])
            summary[key] = {"p50": float(p50), "p95": float(p95)}
        profiles[space] = summary
    token_counts = [
        len(seq.token_ids)
        for seq in runtime.harvester.tokenize(
            [SequenceInput(str(i), texts[i % len(texts)]) for i in range(settings.profile_repeats)]
        )
    ]
    return {
        "device": str(runtime.harvester.device),
        "input_tokens_min": min(token_counts),
        "input_tokens_max": max(token_counts),
        "onnx_threads": settings.onnx_threads,
        "onnx_provider": "CPUExecutionProvider",
        "repeats_per_space": settings.profile_repeats,
        "warmups_per_space": settings.profile_warmups,
        "gpt2_cpu_threads": runtime.harvester.config.cpu_threads,
        "model_version": runtime.bundle.model_version,
        "latency": profiles,
    }
