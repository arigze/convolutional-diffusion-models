"""
Build comparison grid figures from samples/comparisons/.

For each dataset (mnist, cifar10):
  - Rows  : seed numbers, sorted numerically
  - Cols  : sample sources (score machines first, then neural models)
  - Output: figures/{dataset}_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

COMPARISONS_ROOT = Path("samples/comparisons")
FIGURES_DIR      = Path("figures")

DATASETS = ["mnist", "cifar10"]

# Fixed column priority (sources matched by prefix order)
COLUMN_PRIORITY = [
    "ideal_score_machine",
    "local_score_machine",
    "equivariant_local_score_machine",
    "unet_",
    "resnet_",
]


def _label(source: str) -> str:
    labels = {
        "ideal_score_machine":            "IS",
        "local_score_machine":            "LS",
        "equivariant_local_score_machine": "ELS",
    }
    if source in labels:
        return labels[source]
    if source.startswith("unet_"):
        return "UNet"
    if source.startswith("resnet_"):
        return "ResNet"
    return source


def _sort_key(source: str) -> tuple:
    for i, prefix in enumerate(COLUMN_PRIORITY):
        if source == prefix or source.startswith(prefix):
            return (i, source)
    return (len(COLUMN_PRIORITY), source)


def make_figure(dataset: str) -> None:
    # Collect seed dirs, sorted numerically
    seed_dirs = sorted(
        [d for d in COMPARISONS_ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not seed_dirs:
        print(f"  No seed dirs found in {COMPARISONS_ROOT}")
        return

    # Discover all sources for this dataset across all seeds
    prefix = f"{dataset}_"
    all_sources: set[str] = set()
    for seed_dir in seed_dirs:
        for png in seed_dir.glob(f"{prefix}*.png"):
            source = png.stem[len(prefix):]
            all_sources.add(source)

    if not all_sources:
        print(f"  No {dataset} samples found — skipping.")
        return

    sources = sorted(all_sources, key=_sort_key)
    n_rows = len(seed_dirs)
    n_cols = len(sources)

    cell_size = 2.0  # inches per cell
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size),
        squeeze=False,
    )

    for col, source in enumerate(sources):
        axes[0, col].set_title(_label(source), fontsize=30, pad=6)

    for row, seed_dir in enumerate(seed_dirs):
        for col, source in enumerate(sources):
            ax = axes[row, col]
            ax.axis("off")
            png_path = seed_dir / f"{prefix}{source}.png"
            if png_path.exists():
                img = mpimg.imread(str(png_path))
                ax.imshow(img, cmap="gray" if dataset == "mnist" else None)

    fig.tight_layout(pad=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{dataset}_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}  ({n_rows} rows × {n_cols} cols)")


def main() -> None:
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        make_figure(dataset)


if __name__ == "__main__":
    main()
