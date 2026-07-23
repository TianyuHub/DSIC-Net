"""Manifest validation and NIfTI/R2SN loading for DSIC-Net."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REQUIRED_COLUMNS = {
    "subject_id",
    "image_path",
    "graph_path",
    "age",
    "sex",
    "split",
}
ALLOWED_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    image_path: Path
    graph_path: Path
    age: float
    sex: int
    split: str


def read_manifest(manifest_path: str | Path) -> list[SubjectRecord]:
    """Read a portable CSV manifest with paths relative to the manifest."""
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)

    records: list[SubjectRecord] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            record = SubjectRecord(
                subject_id=row["subject_id"].strip(),
                image_path=(manifest_path.parent / row["image_path"]).resolve(),
                graph_path=(manifest_path.parent / row["graph_path"]).resolve(),
                age=float(row["age"]),
                sex=int(row["sex"]),
                split=row["split"].strip().lower(),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid value on manifest line {line_number}") from error
        records.append(record)
    return records


def validate_manifest(
    manifest_path: str | Path,
    *,
    inspect_files: bool = True,
) -> dict[str, int]:
    """Validate labels, split leakage, NIfTI files, and graph arrays."""
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError("manifest contains no subjects")

    seen: set[str] = set()
    counts = {split: 0 for split in sorted(ALLOWED_SPLITS)}
    expected_graph_shape: tuple[int, int] | None = None
    for record in records:
        if not record.subject_id:
            raise ValueError("subject_id cannot be empty")
        if record.subject_id in seen:
            raise ValueError(f"duplicate subject_id across splits: {record.subject_id}")
        seen.add(record.subject_id)
        if record.split not in ALLOWED_SPLITS:
            raise ValueError(f"invalid split for {record.subject_id}: {record.split}")
        if not np.isfinite(record.age) or record.age <= 0:
            raise ValueError(f"invalid age for {record.subject_id}: {record.age}")
        if record.sex not in (0, 1):
            raise ValueError(f"sex must be 0 or 1 for {record.subject_id}")
        counts[record.split] += 1

        if not inspect_files:
            continue
        if not record.image_path.is_file():
            raise FileNotFoundError(record.image_path)
        if not record.graph_path.is_file():
            raise FileNotFoundError(record.graph_path)

        image = nib.load(record.image_path)
        if len(image.shape) != 3 or min(image.shape) < 8:
            raise ValueError(f"expected a 3D NIfTI volume for {record.subject_id}")
        volume = np.asarray(image.dataobj)
        if not np.isfinite(volume).all():
            raise ValueError(f"non-finite voxels for {record.subject_id}")

        with np.load(record.graph_path) as graph:
            if not {"node_features", "adjacency"}.issubset(graph.files):
                raise ValueError(f"graph file lacks required arrays: {record.graph_path}")
            node_features = np.asarray(graph["node_features"])
            adjacency = np.asarray(graph["adjacency"])
        if node_features.ndim != 2:
            raise ValueError(f"node_features must be 2D for {record.subject_id}")
        if adjacency.shape != (node_features.shape[0], node_features.shape[0]):
            raise ValueError(f"adjacency shape mismatch for {record.subject_id}")
        if not np.isfinite(node_features).all() or not np.isfinite(adjacency).all():
            raise ValueError(f"non-finite graph values for {record.subject_id}")
        if not np.allclose(adjacency, adjacency.T, atol=1e-5):
            raise ValueError(f"adjacency is not symmetric for {record.subject_id}")
        graph_shape = (node_features.shape[0], node_features.shape[1])
        if expected_graph_shape is None:
            expected_graph_shape = graph_shape
        elif graph_shape != expected_graph_shape:
            raise ValueError(
                f"inconsistent graph shape for {record.subject_id}: "
                f"{graph_shape} != {expected_graph_shape}"
            )

    return {"total": len(records), **counts}


def _zscore_nonzero(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mask = volume != 0
    values = volume[mask] if mask.any() else volume.reshape(-1)
    mean = float(values.mean())
    std = float(values.std())
    normalized = (volume - mean) / max(std, 1e-6)
    if mask.any():
        normalized[~mask] = 0.0
    return normalized


class DSICDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Dataset for preprocessed 3D images and precomputed subject R2SNs."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: tuple[int, int, int] = (32, 32, 32),
    ) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        self.records = [record for record in read_manifest(manifest_path) if record.split == split]
        if not self.records:
            raise ValueError(f"manifest has no samples in split '{split}'")
        self.image_size = tuple(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        volume = _zscore_nonzero(np.asarray(nib.load(record.image_path).dataobj))
        image = torch.from_numpy(volume.copy()).unsqueeze(0).unsqueeze(0)
        image = F.interpolate(
            image,
            size=self.image_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        with np.load(record.graph_path) as graph:
            node_features = torch.from_numpy(
                np.asarray(graph["node_features"], dtype=np.float32).copy()
            )
            adjacency = torch.from_numpy(np.asarray(graph["adjacency"], dtype=np.float32).copy())
        return {
            "subject_id": record.subject_id,
            "image": image,
            "node_features": node_features,
            "adjacency": adjacency,
            "age": torch.tensor([record.age], dtype=torch.float32),
            "sex": torch.tensor(record.sex, dtype=torch.long),
        }


def create_data_loader(
    manifest_path: str | Path,
    split: str,
    *,
    batch_size: int = 1,
    image_size: tuple[int, int, int] = (32, 32, 32),
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DSICDataset(manifest_path, split, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
    )


def infer_graph_dimensions(records: Iterable[SubjectRecord]) -> tuple[int, int]:
    """Return (number of nodes, node-feature dimension) from the first graph."""
    try:
        first = next(iter(records))
    except StopIteration as error:
        raise ValueError("no records supplied") from error
    with np.load(first.graph_path) as graph:
        shape = graph["node_features"].shape
    if len(shape) != 2:
        raise ValueError("node_features must be a 2D array")
    return int(shape[0]), int(shape[1])
