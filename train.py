from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from config import FullConfig, compute_per_step_gamma, load_config
from models import MinimalUNet, MinimalResNet, DDIM
from utils.noise_schedules import cosine_noise_schedule


DATASET_SIZES = {
    "mnist": 60000,
    "cifar10": 50000,
    "fashion-mnist": 60000,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion model training (UNet / ResNet)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--data-root", type=str, default="./data", help="Directory for torchvision datasets")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    return parser.parse_args()


def make_run_id(config_path: str) -> str:
    stem = Path(config_path).stem
    short_id = uuid.uuid4().hex[:8]
    return f"{stem}_{short_id}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(cfg: FullConfig) -> torch.device:
    if cfg.experiment.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests CUDA but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif cfg.experiment.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device: {cfg.experiment.device!r}")

    if cfg.experiment.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = cfg.experiment.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.experiment.allow_tf32

    return device


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloader(cfg: FullConfig, *, data_root: str, num_workers: int) -> tuple[DataLoader, int]:
    """Returns (dataloader, factor) where factor accounts for dataset subsetting."""
    name = cfg.dataset.name
    image_size = cfg.dataset.image_size
    channels = cfg.dataset.channels

    norm = transforms.Normalize([0.5] * channels, [0.5] * channels)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), norm])

    if name == "mnist":
        dataset = tv_datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    elif name == "cifar10":
        dataset = tv_datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    elif name == "fashion-mnist":
        dataset = tv_datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name!r}")

    # Subset support: if maxsamps is set and smaller than the full dataset,
    # truncate and record the factor by which epochs/save_interval should scale.
    factor = 1
    maxsamps = cfg.dataset.get("maxsamps", None) if hasattr(cfg.dataset, "get") else getattr(cfg.dataset, "maxsamps", None)
    if maxsamps is not None and maxsamps < len(dataset):
        factor = len(dataset) // maxsamps
        dataset = torch.utils.data.Subset(dataset, list(range(maxsamps)))

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
    return loader, factor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: FullConfig) -> DDIM:
    """Build backbone + wrap in DDIM, matching train_script_original behaviour."""
    mult = getattr(cfg.model, "mult", 1)
    layers = getattr(cfg.model, "layers", 3)
    normal = None if getattr(cfg.model, "nonorm", True) else "GroupNorm"
    padding_mode = cfg.model.padding

    if cfg.model.architecture == "unet":
        backbone = MinimalUNet(
            channels=cfg.model.in_channels,
            fchannels=[mult * 32 * (2 ** i) for i in range(layers)],
            emb_dim=cfg.model.embedding_dim,
            padding_mode=padding_mode,
            conditional=cfg.dataset.conditional,
            num_classes=cfg.dataset.num_classes,
        )
    elif cfg.model.architecture == "resnet":
        backbone = MinimalResNet(
            channels=cfg.model.in_channels,
            emb_dim=128 * mult,
            mode=padding_mode,
            conditional=cfg.dataset.conditional,
            num_classes=cfg.dataset.num_classes,
            kernel_size=3,
            num_layers=layers,
            normalization=normal,
            lastksize=3,
        )
    else:
        raise ValueError(f"Unsupported architecture: {cfg.model.architecture!r}")

    model = DDIM(
        pretrained_backbone=backbone,
        default_imsize=cfg.dataset.image_size,
        in_channels=cfg.model.in_channels,
        noise_schedule=cosine_noise_schedule,
    )
    return model


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(cfg: FullConfig, model: torch.nn.Module) -> Adam:
    if cfg.training.optimizer.name != "adam":
        raise ValueError(f"Only Adam is supported, got {cfg.training.optimizer.name!r}")
    return Adam(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        betas=cfg.training.optimizer.betas,
        eps=cfg.training.optimizer.eps,
        weight_decay=cfg.training.optimizer.weight_decay,
    )


def build_scheduler(cfg: FullConfig, optimizer: Adam, factor: int = 1) -> ExponentialLR:
    if cfg.dataset.name not in DATASET_SIZES:
        raise ValueError(f"Unknown dataset {cfg.dataset.name!r}. Expected one of {sorted(DATASET_SIZES)}")
    gamma = compute_per_step_gamma(
        dataset_size=DATASET_SIZES[cfg.dataset.name],
        batch_size=cfg.training.batch_size,
        halve_every_epochs=cfg.training.lr_schedule.halve_every_epochs,
    )
    return ExponentialLR(optimizer, gamma=gamma)


# ---------------------------------------------------------------------------
# Checkpointing & logging
# ---------------------------------------------------------------------------

def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    global_step: int,
    cfg: FullConfig,
    model: torch.nn.Module,
    optimizer: Adam,
    scheduler: ExponentialLR,
    last_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "cfg": asdict(cfg) if is_dataclass(cfg) else cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "last_loss": last_loss,
        },
        path,
    )


def load_checkpoint(
    *,
    path: Path,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: Adam,
    scheduler: ExponentialLR,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device)
    missing = [k for k in ("epoch", "global_step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict") if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"]), int(ckpt["global_step"]), float(ckpt.get("last_loss", float("nan")))


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_samples(
    *,
    path: Path,
    model: DDIM,
    cfg: FullConfig,
    device: torch.device,
    n_samples: int = 16,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = None
    if cfg.dataset.conditional:
        label = torch.arange(n_samples, device=device) % cfg.dataset.num_classes
    with torch.no_grad():
        samples = model.sample(
            batch_size=n_samples,
            label=label,
            nsteps=cfg.diffusion.sampler.steps,
        )
    model.train()
    samples = (samples * 0.5 + 0.5).clamp(0, 1)
    nrow = int(n_samples ** 0.5)
    tv_utils.save_image(samples, path, nrow=nrow)
    print(f"Saved samples    : {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_id = make_run_id(args.config)

    checkpoints_dir = Path(cfg.artifacts.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.experiment.seed)
    device = configure_torch(cfg)

    print(f"Run ID    : {run_id}")
    print(f"Config    : {args.config}")
    print(f"Model     : {cfg.model.architecture} | Dataset: {cfg.dataset.name} | Device: {device}")
    print(f"Training  : {cfg.training.epochs} epochs, batch {cfg.training.batch_size}")

    dataloader, factor = build_dataloader(cfg, data_root=args.data_root, num_workers=args.num_workers)
    if factor > 1:
        print(f"Subset    : dataset reduced, epoch/save_interval factor = {factor}")

    # Scale epochs and save interval by factor (matches train_script_original subset behaviour)
    total_epochs = cfg.training.epochs * factor
    save_every = cfg.logging.save_every_epochs * factor
    sample_every = cfg.logging.sample_every_epochs * factor

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, factor=factor)

    start_epoch = 1
    global_step = 0
    last_loss = float("nan")
    loss_log: list[dict] = []

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resumed_epoch, global_step, last_loss = load_checkpoint(
            path=resume_path,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = resumed_epoch + 1
        print(f"Resumed   : epoch {resumed_epoch}, step {global_step}, loss {last_loss:.6f}")

    if start_epoch > total_epochs:
        print("Already completed all configured epochs.")
        return

    model.train()
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_loss_sum = 0.0
        epoch_items = 0

        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            bsz = images.shape[0]
            t = torch.randint(0, cfg.diffusion.horizon, (bsz,), device=device).float() / cfg.diffusion.horizon
            noise = torch.randn_like(images)
            beta = cosine_noise_schedule(t)
            x_t = (1.0 - beta).sqrt()[:, None, None, None] * images + beta.sqrt()[:, None, None, None] * noise

            optimizer.zero_grad(set_to_none=True)

            # Match train__original.py: pass label only when conditional
            if cfg.dataset.conditional:
                pred_noise = model(t, x_t, label=labels)
            else:
                pred_noise = model(t, x_t)

            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            scheduler.step()

            last_loss = float(loss.item())
            epoch_loss_sum += last_loss * bsz
            epoch_items += bsz
            global_step += 1

            if batch_idx % 100 == 0 or batch_idx == len(dataloader):
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch:03d}/{total_epochs:03d}]"
                    f"[batch {batch_idx:04d}/{len(dataloader):04d}]"
                    f" loss={last_loss:.6f} lr={lr:.3e}"
                )

        epoch_avg_loss = epoch_loss_sum / max(epoch_items, 1)
        loss_log.append({"epoch": epoch, "avg_loss": epoch_avg_loss, "last_loss": last_loss})
        print(f"Epoch {epoch:03d} complete | avg_loss={epoch_avg_loss:.6f}")

        if epoch % sample_every == 0 or epoch == total_epochs:
            sample_path = Path(cfg.artifacts.samples_dir) / f"{run_id}_epoch_{epoch:03d}.png"
            save_samples(path=sample_path, model=model, cfg=cfg, device=device)

        if epoch % save_every == 0 or epoch == total_epochs:
            ckpt_path = checkpoints_dir / f"{run_id}_epoch_{epoch:03d}.pt"
            save_checkpoint(
                path=ckpt_path,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                last_loss=last_loss,
            )
            print(f"Saved checkpoint : {ckpt_path}")

    loss_log_path = checkpoints_dir / f"{run_id}_loss_log.json"
    save_json(loss_log_path, loss_log)
    print(f"Saved loss log   : {loss_log_path}")

    final_ckpt = checkpoints_dir / f"{run_id}_final.pt"
    save_checkpoint(
        path=final_ckpt,
        epoch=total_epochs,
        global_step=global_step,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        last_loss=last_loss,
    )
    print(f"Saved final ckpt : {final_ckpt}")
    print("Training complete.")


if __name__ == "__main__":
    main()