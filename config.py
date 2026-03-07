from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import math
import yaml


PaddingMode = Literal["zeros", "circular"]
Architecture = Literal["resnet", "unet"]
OptimizerName = Literal["adam"]
LRScheduleName = Literal["exponential_from_half_life"]
NoiseScheduleName = Literal["cosine"]
SamplerName = Literal["ddim"]
PredictionType = Literal["eps"]


@dataclass
class ExperimentConfig:
    seed: int
    device: str
    deterministic: bool
    allow_tf32: bool


@dataclass
class OptimizerConfig:
    name: OptimizerName
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float


@dataclass
class LRScheduleConfig:
    name: LRScheduleName
    halve_every_epochs: float
    step_unit: Literal["optimizer_step"]


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    lr_schedule: LRScheduleConfig


@dataclass
class SamplerConfig:
    name: SamplerName
    steps: int
    eta: float


@dataclass
class DiffusionConfig:
    prediction_type: PredictionType
    horizon: int
    noise_schedule: NoiseScheduleName
    sampler: SamplerConfig


@dataclass
class DatasetConfig:
    name: Literal["mnist", "cifar10"]
    image_size: int
    channels: int
    num_classes: Optional[int]
    conditional: bool


@dataclass
class ModelConfig:
    architecture: Architecture
    in_channels: int
    out_channels: int
    embedding_dim: int
    normalization: Literal["none"]
    padding: PaddingMode

    # ResNet-only
    hidden_channels: Optional[int] = None
    kernel_size: Optional[int] = None
    num_mid_layers: Optional[int] = None
    embedding_injection: Optional[str] = None

    # UNet-only
    channel_mults: Optional[list[int]] = None
    residual: Optional[bool] = None
    downsample: Optional[str] = None
    upsample: Optional[str] = None
    skip_connection: Optional[str] = None


@dataclass
class LoggingConfig:
    save_every_epochs: int
    sample_every_epochs: int


@dataclass
class ArtifactsConfig:
    checkpoints_dir: str
    samples_dir: str
    seeds_dir: str


@dataclass
class FullConfig:
    experiment: ExperimentConfig
    training: TrainingConfig
    diffusion: DiffusionConfig
    dataset: DatasetConfig
    model: ModelConfig
    logging: LoggingConfig
    artifacts: ArtifactsConfig


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    raw = _load_yaml(path)

    if "inherits" in raw:
        base_path = path.parent / raw["inherits"]
        base = _load_yaml(base_path)
        raw = _deep_update(base, {k: v for k, v in raw.items() if k != "inherits"})

    cfg = FullConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        training=TrainingConfig(
            epochs=raw["training"]["epochs"],
            batch_size=raw["training"]["batch_size"],
            optimizer=OptimizerConfig(
                name=raw["training"]["optimizer"]["name"],
                lr=raw["training"]["optimizer"]["lr"],
                weight_decay=raw["training"]["optimizer"]["weight_decay"],
                betas=tuple(raw["training"]["optimizer"]["betas"]),
                eps=raw["training"]["optimizer"]["eps"],
            ),
            lr_schedule=LRScheduleConfig(**raw["training"]["lr_schedule"]),
        ),
        diffusion=DiffusionConfig(
            prediction_type=raw["diffusion"]["prediction_type"],
            horizon=raw["diffusion"]["horizon"],
            noise_schedule=raw["diffusion"]["noise_schedule"],
            sampler=SamplerConfig(**raw["diffusion"]["sampler"]),
        ),
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(
            architecture=raw["model"]["architecture"],
            in_channels=raw["model"]["in_channels"],
            out_channels=raw["model"]["out_channels"],
            embedding_dim=raw["model"].get("embedding_dim", raw["model"].get("hidden_channels", 256)),
            normalization=raw["model"].get("normalization", raw.get("model", {}).get("normalization", "none")),
            padding=raw["model"]["padding"],
            hidden_channels=raw["model"].get("hidden_channels"),
            kernel_size=raw["model"].get("kernel_size"),
            num_mid_layers=raw["model"].get("num_mid_layers"),
            embedding_injection=raw["model"].get("embedding_injection"),
            channel_mults=raw["model"].get("channel_mults"),
            residual=raw["model"].get("residual"),
            downsample=raw["model"].get("downsample"),
            upsample=raw["model"].get("upsample"),
            skip_connection=raw["model"].get("skip_connection"),
        ),
        logging=LoggingConfig(**raw["logging"]),
        artifacts=ArtifactsConfig(**raw["artifacts"]),
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: FullConfig) -> None:
    if cfg.dataset.conditional and cfg.dataset.num_classes is None:
        raise ValueError("Conditional dataset must specify num_classes.")
    if not cfg.dataset.conditional and cfg.dataset.num_classes is not None:
        raise ValueError("Unconditional dataset should have num_classes=None.")

    if cfg.dataset.channels != cfg.model.in_channels:
        raise ValueError("dataset.channels must match model.in_channels.")
    if cfg.dataset.channels != cfg.model.out_channels:
        raise ValueError("dataset.channels must match model.out_channels.")

    if cfg.model.architecture == "resnet":
        required = [cfg.model.hidden_channels, cfg.model.kernel_size, cfg.model.num_mid_layers]
        if any(v is None for v in required):
            raise ValueError("ResNet config missing required fields.")
    elif cfg.model.architecture == "unet":
        if cfg.model.channel_mults is None:
            raise ValueError("UNet config missing channel_mults.")
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture}")


def compute_per_step_gamma(
    *,
    dataset_size: int,
    batch_size: int,
    halve_every_epochs: float,
) -> float:
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_steps_to_half = halve_every_epochs * steps_per_epoch
    return 0.5 ** (1.0 / total_steps_to_half)