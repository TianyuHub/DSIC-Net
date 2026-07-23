"""Deterministic synthetic data for software verification only."""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

DEMO_SUBJECTS = (
    ("demo_001", 29.0, 0, "train"),
    ("demo_002", 34.0, 1, "train"),
    ("demo_003", 39.0, 0, "train"),
    ("demo_004", 47.0, 1, "train"),
    ("demo_005", 55.0, 1, "val"),
    ("demo_006", 36.0, 0, "test"),
)


def _synthetic_brain(age: float, seed: int, shape: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = [np.linspace(-1.0, 1.0, length, dtype=np.float32) for length in shape]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    radius = (xx / 0.78) ** 2 + (yy / 0.88) ** 2 + (zz / 0.74) ** 2
    brain_mask = radius <= 1.0
    cortex = np.exp(-2.5 * radius)
    ventricles = np.exp(-35.0 * ((xx / 0.22) ** 2 + (yy / 0.14) ** 2 + (zz / 0.32) ** 2))
    age_scale = (age - 20.0) / 60.0
    volume = cortex - (0.20 + 0.15 * age_scale) * ventricles
    volume += rng.normal(0.0, 0.025, size=shape)
    volume *= brain_mask
    return volume.astype(np.float32)


def _synthetic_r2sn(
    age: float, seed: int, nodes: int, features: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    node_features = rng.normal(0.0, 1.0, size=(nodes, features)).astype(np.float32)
    node_features[:, 0] += (age - 40.0) / 20.0
    normalized = node_features / np.maximum(
        np.linalg.norm(node_features, axis=1, keepdims=True), 1e-6
    )
    similarity = normalized @ normalized.T
    adjacency = np.maximum(similarity, 0.0)
    adjacency[adjacency < 0.25] = 0.0
    adjacency = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adjacency, 0.0)
    return node_features, adjacency.astype(np.float32)


def generate_demo_dataset(
    output_dir: str | Path,
    *,
    seed: int = 7,
    shape: tuple[int, int, int] = (32, 32, 32),
    nodes: int = 12,
    features: int = 8,
    overwrite: bool = False,
) -> Path:
    """Create small valid NIfTI volumes, graph files, and a CSV manifest."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    graphs_dir = output_dir / "graphs"
    manifest_path = output_dir / "manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        return manifest_path

    rows: list[dict[str, str | float | int]] = []
    for index, (subject_id, age, sex, split) in enumerate(DEMO_SUBJECTS):
        image_name = f"{subject_id}.nii.gz"
        graph_name = f"{subject_id}.npz"
        volume = _synthetic_brain(age, seed + index, shape)
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), images_dir / image_name)
        node_features, adjacency = _synthetic_r2sn(age, seed + 100 + index, nodes, features)
        np.savez_compressed(
            graphs_dir / graph_name,
            node_features=node_features,
            adjacency=adjacency,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "image_path": f"images/{image_name}",
                "graph_path": f"graphs/{graph_name}",
                "age": age,
                "sex": sex,
                "split": split,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path
