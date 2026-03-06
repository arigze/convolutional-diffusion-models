from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import FullConfig, compute_per_step_gamma, load_config
from models.ddim import DDIMDiffusion
from models.resnet import MinimalResNet
from models.unet import MinimalUNet


DATASET_SIZES = {
    "mnist": 60000,
    "cifar10": 50000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5 - MNIST ResNet training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory where torchvision datasets will be stored",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 for max debuggability/determinism.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a checkpoint to resume from",
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
        "logging": {
            "save_every_epochs": cfg.logging.save_every_epochs,
            "sample_every_epochs": cfg.logging.sample_every_epochs,
        },
        "artifacts": {
            "checkpoints_dir": cfg.artifacts.checkpoints_dir,
            "samples_dir": cfg.artifacts.samples_dir,
            "seeds_dir": cfg.artifacts.seeds_dir,
        },
    }


def load_checkpoint_for_resume(
    *,
    path: str | Path,
    device: torch.device,
    backbone: torch.nn.Module,
    diffusion: DDIMDiffusion,
    optimizer: Adam,
    scheduler: ExponentialLR,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device)

    required_keys = [
        "epoch",
        "global_step",
        "backbone_state_dict",
        "diffusion_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise KeyError(f"Resume checkpoint is missing required keys: {missing}")

    backbone.load_state_dict(ckpt["backbone_state_dict"])
    diffusion.load_state_dict(ckpt["diffusion_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    last_epoch = int(ckpt["epoch"])
    global_step = int(ckpt["global_step"])
    last_loss = float(ckpt.get("last_loss", float("nan")))

    return last_epoch, global_step, last_loss


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(cfg: FullConfig) -> torch.device:
    if cfg.experiment.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Config requests CUDA but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda")
    elif cfg.experiment.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device in config: {cfg.experiment.device!r}")

    if cfg.experiment.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = cfg.experiment.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.experiment.allow_tf32

    return device


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def build_mnist_dataloader(
    cfg: FullConfig,
    *,
    data_root: str,
    num_workers: int,
) -> DataLoader:
    if cfg.dataset.name != "mnist":
        raise ValueError(
            f"Step 5 trainer is intentionally restricted to MNIST first. "
            f"Got dataset={cfg.dataset.name!r}."
        )
    if cfg.dataset.conditional:
        raise ValueError("Step 5 trainer expects unconditional MNIST.")
    if cfg.model.architecture != "resnet":
        raise ValueError(
            f"Step 5 trainer is intentionally restricted to ResNet first. "
            f"Got architecture={cfg.model.architecture!r}."
        )
    if cfg.dataset.channels != 1:
        raise ValueError(f"MNIST should have 1 channel, got {cfg.dataset.channels}")
    if cfg.dataset.image_size != 28:
        raise ValueError(f"MNIST should have image_size=28, got {cfg.dataset.image_size}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0,1], shape [1,28,28]
        ]
    )

    dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(cfg.experiment.seed)

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(cfg.experiment.device == "cuda"),
        drop_last=False,
        generator=generator,
    )
    return loader


def build_backbone(cfg: FullConfig) -> torch.nn.Module:
    if cfg.model.architecture == "resnet":
        return MinimalResNet.from_config(cfg)
    if cfg.model.architecture == "unet":
        return MinimalUNet.from_config(cfg)
    raise ValueError(f"Unsupported architecture: {cfg.model.architecture!r}")


def build_diffusion(cfg: FullConfig, backbone: torch.nn.Module) -> DDIMDiffusion:
    return DDIMDiffusion.from_config(backbone=backbone, cfg=cfg)


def build_optimizer(cfg: FullConfig, model: torch.nn.Module) -> Adam:
    if cfg.training.optimizer.name != "adam":
        raise ValueError(
            f"Only Adam is supported right now, got {cfg.training.optimizer.name!r}"
        )

    return Adam(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        betas=cfg.training.optimizer.betas,
        eps=cfg.training.optimizer.eps,
        weight_decay=cfg.training.optimizer.weight_decay,
    )


def build_scheduler(cfg: FullConfig, optimizer: Adam) -> ExponentialLR:
    if cfg.dataset.name not in DATASET_SIZES:
        raise ValueError(
            f"Unknown dataset '{cfg.dataset.name}'. "
            f"Expected one of: {sorted(DATASET_SIZES.keys())}"
        )

    gamma = compute_per_step_gamma(
        dataset_size=DATASET_SIZES[cfg.dataset.name],
        batch_size=cfg.training.batch_size,
        halve_every_epochs=cfg.training.lr_schedule.halve_every_epochs,
    )
    return ExponentialLR(optimizer, gamma=gamma)


def sample_timesteps(
    batch_size: int,
    horizon: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    # Uniform over {0, ..., T-1}
    return torch.randint(
        low=0,
        high=horizon,
        size=(batch_size,),
        device=device,
        dtype=torch.long,
    )


def normalize_images_to_model_range(x: torch.Tensor) -> torch.Tensor:
    # torchvision ToTensor() gives [0,1]. Diffusion model typically trains in [-1,1].
    return x * 2.0 - 1.0


def denormalize_model_range_to_image(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _make_grayscale_grid(samples: torch.Tensor, nrow: int = 4) -> np.ndarray:
    """
    samples: [B,1,H,W] in [0,1]
    returns: uint8 image array [H_grid, W_grid]
    """
    if samples.ndim != 4 or samples.shape[1] != 1:
        raise ValueError(f"Expected samples with shape [B,1,H,W], got {tuple(samples.shape)}")

    samples = samples.detach().cpu()
    b, _, h, w = samples.shape
    ncol = int(np.ceil(b / nrow))

    grid = torch.zeros((ncol * h, nrow * w), dtype=torch.float32)

    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = samples[idx, 0]

    grid_np = (grid.numpy() * 255.0).round().astype(np.uint8)
    return grid_np


@torch.no_grad()
def save_sample_grid(
    diffusion: DDIMDiffusion,
    *,
    batch_size: int,
    sample_seed: int,
    out_path: str | Path,
    device: torch.device,
) -> None:
    samples = diffusion.sample(
        batch_size=batch_size,
        nsteps=diffusion.default_sampling_steps,
        seed=sample_seed,
        labels=None,
        device=device,
    )
    samples = denormalize_model_range_to_image(samples)
    grid = _make_grayscale_grid(samples, nrow=4)
    img = Image.fromarray(grid, mode="L")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_checkpoint(
    *,
    path: str | Path,
    epoch: int,
    global_step: int,
    cfg: FullConfig,
    backbone: torch.nn.Module,
    diffusion: DDIMDiffusion,
    optimizer: Adam,
    scheduler: ExponentialLR,
    last_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "cfg": asdict(cfg) if is_dataclass(cfg) else cfg,
            "backbone_state_dict": backbone.state_dict(),
            "diffusion_state_dict": diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "last_loss": last_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    set_global_seed(cfg.experiment.seed)
    device = configure_torch(cfg)

    dataloader = build_mnist_dataloader(
        cfg,
        data_root=args.data_root,
        num_workers=args.num_workers,
    )

    backbone = build_backbone(cfg).to(device)
    diffusion = build_diffusion(cfg, backbone).to(device)

    optimizer = build_optimizer(cfg, diffusion)
    scheduler = build_scheduler(cfg, optimizer)

    dataset_size = DATASET_SIZES[cfg.dataset.name]
    steps_per_epoch = (dataset_size + cfg.training.batch_size - 1) // cfg.training.batch_size
    derived_gamma = compute_per_step_gamma(
        dataset_size=dataset_size,
        batch_size=cfg.training.batch_size,
        halve_every_epochs=cfg.training.lr_schedule.halve_every_epochs,
    )

    print("=" * 80)
    print("STEP 5 TRAINING START")
    print("=" * 80)
    print()
    print("Resolved configuration:")
    print(pformat(summarize_config(cfg), sort_dicts=False))
    print()
    print("Derived quantities:")
    print(f"  dataset_size            : {dataset_size}")
    print(f"  steps_per_epoch         : {steps_per_epoch}")
    print(f"  lr_half_life_epochs     : {cfg.training.lr_schedule.halve_every_epochs}")
    print(f"  derived_step_gamma      : {derived_gamma:.12f}")
    print(f"  device                  : {device}")
    print()

    run_name = Path(args.config).stem

    save_json(
        Path(cfg.artifacts.checkpoints_dir) / f"{run_name}_resolved_config.json",
        summarize_config(cfg),
    )

    try:
        shutil.copy2(args.config, Path(cfg.artifacts.checkpoints_dir) / f"{run_name}_source_config.yaml")
    except Exception:
        pass

    sample_seed = cfg.experiment.seed
    save_json(
        Path(cfg.artifacts.seeds_dir) / f"{run_name}_sample_seed.json",
        {
            "run_name": run_name,
            "sample_seed": sample_seed,
            "conditional": False,
            "labels": None,
            "sampling_steps": cfg.diffusion.sampler.steps,
        },
    )

    diffusion.train()
    global_step = 0
    last_loss = float("nan")
    start_epoch = 1

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resumed_epoch, global_step, last_loss = load_checkpoint_for_resume(
            path=resume_path,
            device=device,
            backbone=backbone,
            diffusion=diffusion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = resumed_epoch + 1

        print()
        print("=" * 80)
        print("RESUMED TRAINING")
        print("=" * 80)
        print(f"Checkpoint path        : {resume_path}")
        print(f"Resumed from epoch     : {resumed_epoch}")
        print(f"Next epoch to run      : {start_epoch}")
        print(f"Resumed global_step    : {global_step}")
        print(f"Resumed last_loss      : {last_loss:.6f}")
        print(f"Resumed scheduler lr   : {scheduler.get_last_lr()[0]:.8e}")
        print()

    if start_epoch > cfg.training.epochs:
        print(
            f"Checkpoint already reached/passed configured training length "
            f"(start_epoch={start_epoch}, total_epochs={cfg.training.epochs})."
        )
        return

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_loss_sum = 0.0
        epoch_items = 0

        for batch_idx, (images, _) in enumerate(dataloader, start=1):
            images = images.to(device, non_blocking=True)
            x0 = normalize_images_to_model_range(images)

            bsz = x0.shape[0]
            t = sample_timesteps(
                batch_size=bsz,
                horizon=cfg.diffusion.horizon,
                device=device,
            )
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise=noise)

            optimizer.zero_grad(set_to_none=True)

            pred_noise = diffusion(x_t, t, labels=None)
            loss = F.mse_loss(pred_noise, noise)

            loss.backward()
            optimizer.step()
            scheduler.step()

            last_loss = float(loss.item())
            epoch_loss_sum += float(loss.item()) * bsz
            epoch_items += bsz
            global_step += 1

            if batch_idx % 100 == 0 or batch_idx == len(dataloader):
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch:03d}/{cfg.training.epochs:03d}] "
                    f"[batch {batch_idx:04d}/{len(dataloader):04d}] "
                    f"loss={loss.item():.6f} "
                    f"lr={current_lr:.8e}"
                )

        epoch_avg_loss = epoch_loss_sum / max(epoch_items, 1)
        print(
            f"Epoch {epoch:03d} complete | "
            f"avg_loss={epoch_avg_loss:.6f} | "
            f"last_loss={last_loss:.6f}"
        )

        if (epoch % cfg.logging.save_every_epochs == 0) or (epoch == cfg.training.epochs):
            ckpt_path = Path(cfg.artifacts.checkpoints_dir) / f"{run_name}_epoch_{epoch:03d}.pt"
            save_checkpoint(
                path=ckpt_path,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                backbone=backbone,
                diffusion=diffusion,
                optimizer=optimizer,
                scheduler=scheduler,
                last_loss=last_loss,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        if (epoch % cfg.logging.sample_every_epochs == 0) or (epoch == cfg.training.epochs):
            diffusion.eval()
            sample_path = Path(cfg.artifacts.samples_dir) / f"{run_name}_epoch_{epoch:03d}.png"
            save_sample_grid(
                diffusion,
                batch_size=16,
                sample_seed=sample_seed,
                out_path=sample_path,
                device=device,
            )
            print(f"Saved sample grid: {sample_path}")
            diffusion.train()

    final_ckpt = Path(cfg.artifacts.checkpoints_dir) / f"{run_name}_final.pt"
    save_checkpoint(
        path=final_ckpt,
        epoch=cfg.training.epochs,
        global_step=global_step,
        cfg=cfg,
        backbone=backbone,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        last_loss=last_loss,
    )
    print(f"Saved final checkpoint: {final_ckpt}")

    diffusion.eval()
    final_sample_path = Path(cfg.artifacts.samples_dir) / f"{run_name}_final.png"
    save_sample_grid(
        diffusion,
        batch_size=16,
        sample_seed=sample_seed,
        out_path=final_sample_path,
        device=device,
    )
    print(f"Saved final sample grid: {final_sample_path}")

    print()
    print("=" * 80)
    print("STEP 5 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()