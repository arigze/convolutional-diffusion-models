"""
Pipeline: train three models, then generate samples from every discovered checkpoint.

Training order:
  1. MNIST  ResNet   (configs/mnist_resnet.yaml)
  2. CIFAR-10 UNet   (configs/cifar10_unet.yaml)
  3. CIFAR-10 ResNet (configs/cifar10_resnet.yaml)

Generation:
  - Discovers every final.pt under artifacts/checkpoints/ (skips old/)
  - Groups models by dataset; pairs unet + resnet IDs for combined sample.py calls
  - Runs with --is --ls --els, two batches: seeds 1-5 then 6-10

Usage:
  python temp.py
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path


CHECKPOINTS_ROOT = Path("artifacts/checkpoints")

TRAIN_CONFIGS = [
    # "configs/mnist_unet.yaml",
    "configs/mnist_resnet.yaml",
    "configs/cifar10_unet.yaml",
    "configs/cifar10_resnet.yaml",
]

SEED_BATCHES = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
]


# ──────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    """Print and execute a command, raising on non-zero exit."""
    print(f"\n{'=' * 70}")
    print("RUN:", " ".join(cmd))
    print("=" * 70, flush=True)
    subprocess.run(cmd, check=True)


def discover_checkpoints() -> list[dict]:
    """
    Scan CHECKPOINTS_ROOT for subdirs that contain final.pt (excluding old/).
    Returns a list of dicts with keys: dataset, arch, model_id.
    """
    models = []
    for subdir in sorted(CHECKPOINTS_ROOT.iterdir()):
        if subdir.name == "old" or not subdir.is_dir():
            continue
        if not (subdir / "final.pt").exists():
            continue
        name = subdir.name
        # Folder format: {dataset}_{arch}_{model_id}
        # arch is always 'unet' or 'resnet'
        for arch in ("unet", "resnet"):
            marker = f"_{arch}_"
            if marker in name:
                dataset, model_id = name.split(marker, 1)
                models.append({"dataset": dataset, "arch": arch, "model_id": model_id})
                print(f"  found: {dataset:10s} {arch:8s} {model_id}")
                break
        else:
            print(f"  [warn] unrecognised checkpoint folder: {name!r} — skipping")
    return models


def generate_for_dataset(dataset: str, unet_ids: list[str], resnet_ids: list[str]) -> None:
    """
    Pair up unet and resnet IDs (None-padded when counts differ) and run
    sample.py for each pair, for both seed batches.
    """
    for unet_id, resnet_id in zip_longest(unet_ids, resnet_ids):
        for seeds in SEED_BATCHES:
            cmd = [sys.executable, "sample.py", "--dataset", dataset]
            if unet_id is not None:
                cmd += ["--unet-id", unet_id]
            if resnet_id is not None:
                cmd += ["--resnet-id", resnet_id]
            cmd += ["--is", "--ls", "--els"]
            cmd += ["--seeds"] + [str(s) for s in seeds]
            cmd += ["--machine-steps", "20"]
            run(cmd)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Phase 1: train ────────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)
    for config in TRAIN_CONFIGS:
        run([sys.executable, "train.py", "--config", config, "--num-workers", 4])

    # ── Phase 2: discover ─────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("PHASE 2: DISCOVERING CHECKPOINTS")
    print("=" * 70)
    models = discover_checkpoints()
    print(f"\n{len(models)} checkpoint(s) found.")

    if not models:
        print("Nothing to generate — exiting.")
        return

    # Group by dataset
    by_dataset: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"unet": [], "resnet": []}
    )
    for m in models:
        by_dataset[m["dataset"]][m["arch"]].append(m["model_id"])

    # ── Phase 3: generate ─────────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("PHASE 3: GENERATING SAMPLES")
    print("=" * 70)
    for dataset, arches in sorted(by_dataset.items()):
        print(f"\n--- dataset: {dataset} ---")
        generate_for_dataset(dataset, arches["unet"], arches["resnet"])

    print("\nAll done.")


if __name__ == "__main__":
    main()
