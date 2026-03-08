"""
Compute average R² scores between score machine outputs and neural model outputs.

For each dataset (mnist, cifar10), prints a table:
  rows    = neural models (UNet, ResNet)
  columns = score machines (IS, LS, ELS)
  cells   = mean R² across all seeds (and model IDs if multiple exist)

R² is computed per seed: treating the neural model output as the reference,
the score machine output as the prediction.
  R² = 1 - sum((ref - pred)²) / sum((ref - mean(ref))²)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

COMPARISONS_ROOT = Path("samples/comparisons")
DATASETS = ["mnist", "cifar10"]

SCORE_MACHINES = [
    ("IS",  "ideal_score_machine"),
    ("LS",  "local_score_machine"),
    ("ELS", "equivariant_local_score_machine"),
]

MODEL_ARCHS = ["unet", "resnet"]


def r2(ref: np.ndarray, pred: np.ndarray) -> float:
    return r2_score(ref.ravel().astype(np.float64), pred.ravel().astype(np.float64))


def compute_table(dataset: str) -> dict[str, dict[str, float]]:
    """
    Returns results[arch][sm_label] = mean R² across all seeds and model IDs.
    """
    seed_dirs = sorted(
        [d for d in COMPARISONS_ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    # Find all model IDs for each arch
    arch_ids: dict[str, list[str]] = {arch: set() for arch in MODEL_ARCHS}
    for seed_dir in seed_dirs:
        for arch in MODEL_ARCHS:
            for npy in seed_dir.glob(f"{dataset}_{arch}_*.npy"):
                model_id = npy.stem[len(f"{dataset}_{arch}_"):]
                arch_ids[arch].add(model_id)
    arch_ids = {arch: sorted(ids) for arch, ids in arch_ids.items()}

    results: dict[str, dict[str, float]] = {}

    for arch in MODEL_ARCHS:
        if not arch_ids[arch]:
            continue
        results[arch] = {}
        for sm_label, sm_source in SCORE_MACHINES:
            scores = []
            for seed_dir in seed_dirs:
                sm_path = seed_dir / f"{dataset}_{sm_source}.npy"
                if not sm_path.exists():
                    continue
                sm_arr = np.load(sm_path)
                for model_id in arch_ids[arch]:
                    model_path = seed_dir / f"{dataset}_{arch}_{model_id}.npy"
                    if not model_path.exists():
                        continue
                    model_arr = np.load(model_path)
                    scores.append(r2(model_arr, sm_arr))
            results[arch][sm_label] = float(np.nanmean(scores)) if scores else float("nan")

    return results


def print_table(dataset: str, results: dict[str, dict[str, float]]) -> None:
    sm_labels = [label for label, _ in SCORE_MACHINES]
    archs = [a for a in MODEL_ARCHS if a in results]

    col_w = 10
    header = f"{'':10s}" + "".join(f"{lbl:>{col_w}}" for lbl in sm_labels)
    sep    = "-" * len(header)

    print(f"\n{dataset.upper()} — Average R²")
    print(sep)
    print(header)
    print(sep)
    for arch in archs:
        row = f"{arch.capitalize():<10}"
        for lbl in sm_labels:
            val = results[arch].get(lbl, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
    print(sep)


def main() -> None:
    for dataset in DATASETS:
        results = compute_table(dataset)
        if results:
            print_table(dataset, results)
        else:
            print(f"\n{dataset.upper()}: no samples found.")


if __name__ == "__main__":
    main()
