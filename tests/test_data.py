from __future__ import annotations

from dsic_net.data import DSICDataset, validate_manifest
from dsic_net.demo_data import generate_demo_dataset


def test_generated_manifest_and_dataset(tmp_path) -> None:
    manifest = generate_demo_dataset(tmp_path / "demo")
    counts = validate_manifest(manifest)
    assert counts == {"total": 6, "test": 1, "train": 4, "val": 1}
    sample = DSICDataset(manifest, "test")[0]
    assert sample["image"].shape == (1, 32, 32, 32)
    assert sample["node_features"].shape == (12, 8)
    assert sample["adjacency"].shape == (12, 12)
