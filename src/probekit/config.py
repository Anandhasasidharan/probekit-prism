"""Validated YAML settings shared by ProbeKit stages."""

from pathlib import Path
from typing import Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def resolve_paths(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, Path) and info.context and not value.is_absolute():
            return (info.context["directory"] / value).resolve()
        return value


class ModelConfig(Settings):
    name: Literal["gpt2"] = "gpt2"
    revision: str
    local_dir: Path | None = None
    layer: int = Field(default=8, ge=0, lt=12)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_length: int = Field(default=128, ge=2, le=1024)
    batch_size: int = Field(default=8, gt=0)
    cpu_threads: int = Field(default=4, gt=0)


class CorpusConfig(Settings):
    dataset: str | None = None
    subset: str | None = None
    split: str = "train"
    text_field: str = "text"
    revision: str | None = None
    jsonl: Path | None = None
    max_tokens: int = Field(ge=2)

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if (self.dataset is None) == (self.jsonl is None):
            raise ValueError("provide exactly one corpus source (dataset or jsonl)")
        return self


class PathsConfig(Settings):
    activations: Path
    sequences: Path
    sae_checkpoint: Path
    feature_db: Path
    probe_dir: Path
    runs: Path


class Config(Settings):
    seed: int = Field(default=42, ge=0, lt=2**32)
    model: ModelConfig
    corpus: CorpusConfig
    paths: PathsConfig


ConfigType = TypeVar("ConfigType", bound=Settings)
DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_SAE_CONFIG = Path("configs/sae_topk.yaml")


def load_yaml(path: str | Path, schema: type[ConfigType]) -> ConfigType:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return schema.model_validate(values, context={"directory": path.parent})


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    return load_yaml(path, Config)
