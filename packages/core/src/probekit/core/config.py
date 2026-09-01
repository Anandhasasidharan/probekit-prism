from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class HarvestConfig(BaseModel):
    model_name: str = "gpt2"
    capture_layers: list[int | str] = [0, 5, 10, "last"]
    batch_size: int = 32
    output_format: Literal["parquet", "numpy"] = "parquet"
    compression: Literal["zstd", "snappy", "none"] = "zstd"


class SAEConfig(BaseModel):
    architecture: Literal["topk", "gated", "jumprelu", "standard"] = "topk"
    k: int = 32
    dict_size: int = 16384
    expansion_factor: int = 16
    lr: float = 3e-4
    sparsity_coef: float = 0.1
    model: str = "gpt2"
    layer: int = 5


class CircuitConfig(BaseModel):
    model_name: str = "gemma-2-2b"
    clt_size: Literal["426k", "2.5m"] = "426k"
    device: str = "cpu"
    max_feature_nodes: int = 7500
    node_threshold: float = 0.8
    edge_threshold: float = 0.95


class AutoInterpConfig(BaseModel):
    provider: Literal["openai", "anthropic", "mock"] = "openai"
    model: str = "gpt-4o-mini"
    top_k_examples: int = 50
    quantiles: list[float] = [0.25, 0.5, 0.75]
    cache_db: str = "./data/features/explanation_cache.sqlite"


class ProbeConfig(BaseModel):
    concepts: list[str] = ["tool_misuse", "overeager_execution", "pii_detection"]
    test_split: float = 0.2
    ood_split: float = 0.3


class SteeringStrategyConfig(BaseModel):
    intervention: Literal["increase", "decrease", "suppress", "inject"] = "suppress"
    threshold: float = 0.7
    strength: float = 1.0


class SteeringConfig(BaseModel):
    default_strategy: str = "suppress_risky"
    strategies: dict[str, SteeringStrategyConfig] = {
        "suppress_risky": SteeringStrategyConfig(),
        "increase_honesty": SteeringStrategyConfig(
            intervention="increase", threshold=0.3, strength=0.8
        ),
        "reduce_deception": SteeringStrategyConfig(
            intervention="suppress", threshold=0.6, strength=1.5
        ),
    }


class RegistryConfig(BaseModel):
    database_url: str = "sqlite:///./data/features/registry.db"


class MonitorConfig(BaseModel):
    drift_threshold_std: float = 3.0
    alert_channels: dict[str, str] = {"slack_webhook": ""}
    check_interval_seconds: int = 3600


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    rate_limit_per_minute: int = 60


class GlobalConfig(BaseModel):
    harvest: HarvestConfig = HarvestConfig()
    sae: SAEConfig = SAEConfig()
    circuit: CircuitConfig = CircuitConfig()
    auto_interp: AutoInterpConfig = AutoInterpConfig()
    probes: ProbeConfig = ProbeConfig()
    steering: SteeringConfig = SteeringConfig()
    registry: RegistryConfig = RegistryConfig()
    monitor: MonitorConfig = MonitorConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GlobalConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
