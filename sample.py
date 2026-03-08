"""
Compare samplers: UNet, ResNet (DDIM), IdealScoreMachine, LocalScoreMachine, EquivariantLocalScoreMachine.

Outputs per sampler:
  samples/comparisons/{seed}/{name}.npy  -- raw float tensor [C, H, W], use for R² scoring
  samples/comparisons/{seed}/{name}.png  -- uint8 image normalized to [0,1], use for visualization

Usage example:
  python sample.py --dataset mnist --unet-id b5750c10 --is --ls --seed 42
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import get_dataset
from models import DDIM, MinimalResNet, MinimalUNet
from score_machines import (
    EquivariantLocalScoreMachine,
    IdealScoreMachine,
    LocalScoreMachine,
)
from utils.noise_schedules import cosine_noise_schedule


DATASET_INFO = {
    "mnist":   {"image_size": 28, "channels": 1},
    "cifar10": {"image_size": 32, "channels": 3},
}

CHECKPOINTS_ROOT = Path("artifacts/checkpoints")


# ---------------------------------------------------------------------------
# Noise schedule for score machines (derived from DDIM's cosine alpha_bar)
# ---------------------------------------------------------------------------

def ddim_derived_beta_schedule(timesteps: int) -> torch.Tensor:
    """Per-step betas derived from DDIM's cosine alpha_bar curve.

    Evaluates DDIM's alpha_bar = cos²(t·π/2 / 1.008) at T+1 equally-spaced
    points, then derives per-step betas as beta[t] = 1 - alpha_bar[t] / alpha_bar[t-1].
    This makes the score machine operate in the same noise space as DDIM.
    """
    t = torch.linspace(0.0, 1.0, timesteps + 1)
    alpha_bar = torch.cos(t / 1.008 * math.pi / 2.0) ** 2
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_checkpoint(dataset: str, model_type: str, model_id: str) -> Path:
    path = CHECKPOINTS_ROOT / f"{dataset}_{model_type}_{model_id}" / "final.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Expected: artifacts/checkpoints/{dataset}_{model_type}_{model_id}/final.pt"
        )
    return path


def load_ddim(dataset: str, model_type: str, model_id: str, device: torch.device) -> DDIM:
    ckpt_path = _resolve_checkpoint(dataset, model_type, model_id)
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    arch = cfg["model"]["architecture"]
    channels = cfg["model"]["in_channels"]
    image_size = cfg["dataset"]["image_size"]
    padding_mode = cfg["model"]["padding"]
    emb_dim = cfg["model"].get("embedding_dim") or cfg["model"].get("hidden_channels", 256)
    conditional = cfg["dataset"]["conditional"]
    num_classes = cfg["dataset"]["num_classes"]

    if arch == "unet":
        backbone = MinimalUNet(
            channels=channels,
            fchannels=cfg["model"]["fchannels"],
            emb_dim=emb_dim,
            padding_mode=padding_mode,
            conditional=conditional,
            num_classes=num_classes,
        )
    elif arch == "resnet":
        backbone = MinimalResNet(
            default_imsize=image_size,
            k=3,
            n_mid_layers=cfg["model"]["num_mid_layers"],
            hidden_channels=cfg["model"]["hidden_channels"],
            padding_mode=padding_mode,
            channels=channels,
            conditional=conditional,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {arch!r}")

    model = DDIM(
        pretrained_backbone=backbone,
        default_imsize=image_size,
        in_channels=channels,
        noise_schedule=cosine_noise_schedule,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_sample(tensor: torch.Tensor, out_dir: Path, name: str) -> None:
    """
    Save a [C, H, W] float tensor as:
      - {name}.npy : raw floats (for R² computation)
      - {name}.png : uint8 normalized to display range (for visualization)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = tensor.detach().cpu().float().numpy()  # [C, H, W]
    np.save(out_dir / f"{name}.npy", arr)

    # Normalize from model range [-1, 1] -> [0, 255]
    vis = ((arr * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    if vis.shape[0] == 1:
        img = Image.fromarray(vis[0], mode="L")
    else:
        img = Image.fromarray(vis.transpose(1, 2, 0), mode="RGB")
    img.save(out_dir / f"{name}.png")

    print(f"  Saved: {out_dir / name}.npy + .png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from diffusion models and score machines, saving outputs for comparison."
    )
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    parser.add_argument("--unet-id",   default=None, help="Unique ID of the UNet checkpoint (e.g. b5750c10)")
    parser.add_argument("--resnet-id", default=None, help="Unique ID of the ResNet checkpoint")
    parser.add_argument("--is",  dest="ideal_score",             action="store_true", help="Sample IdealScoreMachine")
    parser.add_argument("--ls",  dest="local_score",             action="store_true", help="Sample LocalScoreMachine")
    parser.add_argument("--els", dest="equivariant_local_score", action="store_true", help="Sample EquivariantLocalScoreMachine")
    parser.add_argument("--seed",         type=int, required=True,  help="Seed for shared initial noise")
    parser.add_argument("--machine-steps", type=int, default=20,     help="Sampling steps shared across all models and score machines (default: 20)")
    parser.add_argument("--ddim-steps",   type=int, default=None,   help="DDIM sampling steps for neural models; overrides --machine-steps if set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}  |  Seed: {args.seed}")

    info = DATASET_INFO[args.dataset]
    channels   = info["channels"]
    image_size = info["image_size"]

    # Shared initial noise [C, H, W] — same starting point for every sampler
    torch.manual_seed(args.seed)
    x0 = torch.randn(channels, image_size, image_size)

    out_dir = Path("samples") / "comparisons" / str(args.seed)

    # ------------------------------------------------------------------
    # Neural DDIM models (UNet / ResNet)
    # ------------------------------------------------------------------
    for model_type, model_id in [("unet", args.unet_id), ("resnet", args.resnet_id)]:
        if model_id is None:
            continue
        print(f"\n[{model_type.upper()} {model_id}]")
        model = load_ddim(args.dataset, model_type, model_id, device)
        x_in = x0.unsqueeze(0).to(device)  # [1, C, H, W]
        ddim_steps = args.ddim_steps if args.ddim_steps is not None else args.machine_steps
        with torch.no_grad():
            sample = model.sample(batch_size=1, x=x_in, nsteps=ddim_steps, device=device)
        save_sample(sample.squeeze(0).cpu(), out_dir, f"{model_type}_{model_id}")

    # ------------------------------------------------------------------
    # Score machines (non-parametric, need the full dataset)
    # ------------------------------------------------------------------
    need_dataset = args.ideal_score or args.local_score or args.equivariant_local_score
    if need_dataset:
        print("\nLoading dataset for score machines...")
        dataset = get_dataset(args.dataset, (image_size, image_size))

        score_machines = []
        if args.ideal_score:
            score_machines.append(("ideal_score_machine", IdealScoreMachine))
        if args.local_score:
            score_machines.append(("local_score_machine", LocalScoreMachine))
        if args.equivariant_local_score:
            score_machines.append(("equivariant_local_score_machine", EquivariantLocalScoreMachine))

        for name, SMClass in score_machines:
            print(f"\n[{name}]")
            sm = SMClass(ddim_derived_beta_schedule, dataset, 256, args.machine_steps)
            with torch.no_grad():
                sample = sm.sample(x0.clone(), device=device)
            save_sample(sample.cpu(), out_dir, name)

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
