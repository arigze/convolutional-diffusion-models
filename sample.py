from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image

from config import FullConfig, load_config
from models.ddim import DDIMDiffusion
from models.resnet import MinimalResNet
from models.unet import MinimalUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample deterministically from a saved diffusion checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config used to rebuild the model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output PNG grid",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Deterministic sampling seed",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help="Override DDIM sampling step count. Defaults to config value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override device. Defaults to config device.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="*",
        default=None,
        help="Optional class labels for conditional sampling. Omit for unconditional models.",
    )
    return parser.parse_args()


def configure_torch(device_name: str, deterministic: bool, allow_tf32: bool) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device: {device_name!r}")

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    return device


def build_backbone(cfg: FullConfig) -> torch.nn.Module:
    if cfg.model.architecture == "resnet":
        return MinimalResNet.from_config(cfg)
    if cfg.model.architecture == "unet":
        return MinimalUNet.from_config(cfg)
    raise ValueError(f"Unsupported architecture: {cfg.model.architecture!r}")


def build_diffusion(cfg: FullConfig, backbone: torch.nn.Module) -> DDIMDiffusion:
    return DDIMDiffusion.from_config(backbone=backbone, cfg=cfg)


def load_checkpoint(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)!r}")
    return ckpt


def maybe_build_labels(
    labels_arg: Optional[list[int]],
    *,
    batch_size: int,
    conditional: bool,
    num_classes: Optional[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not conditional:
        if labels_arg is not None:
            raise ValueError("This config is unconditional, so --labels must not be provided.")
        return None

    if labels_arg is None:
        raise ValueError("This config is conditional, so --labels is required.")

    if len(labels_arg) == 0:
        raise ValueError("If provided, --labels must contain at least one integer.")

    if len(labels_arg) == 1:
        labels_arg = labels_arg * batch_size
    elif len(labels_arg) != batch_size:
        raise ValueError(
            f"For conditional sampling, number of labels must be either 1 or batch_size. "
            f"Got len(labels)={len(labels_arg)}, batch_size={batch_size}."
        )

    if num_classes is not None:
        bad = [x for x in labels_arg if x < 0 or x >= num_classes]
        if bad:
            raise ValueError(
                f"Labels out of range for num_classes={num_classes}: {bad}"
            )

    return torch.tensor(labels_arg, dtype=torch.long, device=device)


def denormalize_model_range_to_image(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def make_grayscale_grid(samples: torch.Tensor, nrow: int = 4) -> Image.Image:
    if samples.ndim != 4 or samples.shape[1] != 1:
        raise ValueError(f"Expected grayscale samples [B,1,H,W], got {tuple(samples.shape)}")

    samples = samples.detach().cpu()
    b, _, h, w = samples.shape
    nrow = max(1, nrow)
    ncol = (b + nrow - 1) // nrow

    grid = torch.zeros((ncol * h, nrow * w), dtype=torch.float32)

    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = samples[idx, 0]

    arr = (grid.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="L")


def make_rgb_grid(samples: torch.Tensor, nrow: int = 4) -> Image.Image:
    if samples.ndim != 4 or samples.shape[1] != 3:
        raise ValueError(f"Expected RGB samples [B,3,H,W], got {tuple(samples.shape)}")

    samples = samples.detach().cpu()
    b, c, h, w = samples.shape
    nrow = max(1, nrow)
    ncol = (b + nrow - 1) // nrow

    grid = torch.zeros((ncol * h, nrow * w, c), dtype=torch.float32)

    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        tile = samples[idx].permute(1, 2, 0)
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = tile

    arr = (grid.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def save_image_grid(samples: torch.Tensor, out_path: str | Path, nrow: int = 4) -> None:
    samples = denormalize_model_range_to_image(samples)
    channels = samples.shape[1]

    if channels == 1:
        img = make_grayscale_grid(samples, nrow=nrow)
    elif channels == 3:
        img = make_rgb_grid(samples, nrow=nrow)
    else:
        raise ValueError(f"Unsupported channel count for image saving: {channels}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)

    device_name = args.device or cfg.experiment.device
    device = configure_torch(
        device_name=device_name,
        deterministic=cfg.experiment.deterministic,
        allow_tf32=cfg.experiment.allow_tf32,
    )

    backbone = build_backbone(cfg).to(device)
    diffusion = build_diffusion(cfg, backbone).to(device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)

    if "backbone_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'backbone_state_dict'")
    if "diffusion_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'diffusion_state_dict'")

    backbone.load_state_dict(ckpt["backbone_state_dict"])
    diffusion.load_state_dict(ckpt["diffusion_state_dict"])

    diffusion.eval()

    labels = maybe_build_labels(
        args.labels,
        batch_size=args.batch_size,
        conditional=cfg.dataset.conditional,
        num_classes=cfg.dataset.num_classes,
        device=device,
    )

    nsteps = args.nsteps if args.nsteps is not None else cfg.diffusion.sampler.steps

    with torch.no_grad():
        samples = diffusion.sample(
            batch_size=args.batch_size,
            nsteps=nsteps,
            seed=args.seed,
            labels=labels,
            device=device,
        )

    save_image_grid(samples, args.output, nrow=4)

    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "output": args.output,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "nsteps": nsteps,
        "device": str(device),
        "conditional": cfg.dataset.conditional,
        "labels": None if labels is None else labels.detach().cpu().tolist(),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_global_step": ckpt.get("global_step"),
    }

    meta_path = Path(args.output).with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Saved samples to : {args.output}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()