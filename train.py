from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

from config import FullConfig, compute_per_step_gamma, load_config


DATASET_SIZES = {
    "mnist": 60000,
    "cifar10": 50000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 config smoke test")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    return parser.parse_args()


def ensure_artifact_dirs(cfg: FullConfig) -> None:
    Path(cfg.artifacts.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.artifacts.samples_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.artifacts.seeds_dir).mkdir(parents=True, exist_ok=True)


def summarize_config(cfg: FullConfig) -> dict:
    return {
        "experiment": {
            "seed": cfg.experiment.seed,
            "device": cfg.experiment.device,
            "deterministic": cfg.experiment.deterministic,
            "allow_tf32": cfg.experiment.allow_tf32,
        },
        "dataset": {
            "name": cfg.dataset.name,
            "image_size": cfg.dataset.image_size,
            "channels": cfg.dataset.channels,
            "conditional": cfg.dataset.conditional,
            "num_classes": cfg.dataset.num_classes,
        },
        "model": {
            "architecture": cfg.model.architecture,
            "in_channels": cfg.model.in_channels,
            "out_channels": cfg.model.out_channels,
            "embedding_dim": cfg.model.embedding_dim,
            "padding": cfg.model.padding,
            "normalization": cfg.model.normalization,
            "hidden_channels": cfg.model.hidden_channels,
            "kernel_size": cfg.model.kernel_size,
            "num_mid_layers": cfg.model.num_mid_layers,
            "channel_mults": cfg.model.channel_mults,
            "downsample": cfg.model.downsample,
            "upsample": cfg.model.upsample,
            "skip_connection": cfg.model.skip_connection,
        },
        "training": {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "optimizer": {
                "name": cfg.training.optimizer.name,
                "lr": cfg.training.optimizer.lr,
                "weight_decay": cfg.training.optimizer.weight_decay,
                "betas": cfg.training.optimizer.betas,
                "eps": cfg.training.optimizer.eps,
            },
            "lr_schedule": {
                "name": cfg.training.lr_schedule.name,
                "halve_every_epochs": cfg.training.lr_schedule.halve_every_epochs,
                "step_unit": cfg.training.lr_schedule.step_unit,
            },
        },
        "diffusion": {
            "prediction_type": cfg.diffusion.prediction_type,
            "horizon": cfg.diffusion.horizon,
            "noise_schedule": cfg.diffusion.noise_schedule,
            "sampler": {
                "name": cfg.diffusion.sampler.name,
                "steps": cfg.diffusion.sampler.steps,
                "eta": cfg.diffusion.sampler.eta,
            },
        },
        "artifacts": {
            "checkpoints_dir": cfg.artifacts.checkpoints_dir,
            "samples_dir": cfg.artifacts.samples_dir,
            "seeds_dir": cfg.artifacts.seeds_dir,
        },
    }


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    if cfg.dataset.name not in DATASET_SIZES:
        raise ValueError(
            f"Unknown dataset '{cfg.dataset.name}'. "
            f"Expected one of: {sorted(DATASET_SIZES.keys())}"
        )

    dataset_size = DATASET_SIZES[cfg.dataset.name]
    steps_per_epoch = (dataset_size + cfg.training.batch_size - 1) // cfg.training.batch_size

    gamma = compute_per_step_gamma(
        dataset_size=dataset_size,
        batch_size=cfg.training.batch_size,
        halve_every_epochs=cfg.training.lr_schedule.halve_every_epochs,
    )

    print("=" * 80)
    print("CONFIG SMOKE TEST PASSED")
    print("=" * 80)
    print()
    print("Resolved configuration:")
    print(pformat(summarize_config(cfg), sort_dicts=False))
    print()
    print("Derived quantities:")
    print(f"  dataset_size            : {dataset_size}")
    print(f"  steps_per_epoch         : {steps_per_epoch}")
    print(f"  lr_half_life_epochs     : {cfg.training.lr_schedule.halve_every_epochs}")
    print(f"  derived_step_gamma      : {gamma:.12f}")
    print()
    print("Artifact directories:")
    print(f"  checkpoints             : {cfg.artifacts.checkpoints_dir}")
    print(f"  samples                 : {cfg.artifacts.samples_dir}")
    print(f"  seeds                   : {cfg.artifacts.seeds_dir}")
    print()
    print("No model, dataloader, diffusion process, or training loop was executed.")
    print("This was a Step 1 configuration-only smoke test.")
    print("=" * 80)


if __name__ == "__main__":
    main()