#!/usr/bin/env python3
"""Run deterministic end-to-end DSIC-Net inference on the demo test subject."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dsic_net.data import (
    create_data_loader,
    infer_graph_dimensions,
    read_manifest,
    validate_manifest,
)
from dsic_net.demo_data import generate_demo_dataset
from dsic_net.models import DSICNet
from dsic_net.training import batch_to_device, set_reproducible_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/demo/manifest.csv")
    parser.add_argument("--output", default="outputs/demo_predictions.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--regenerate-data", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if args.regenerate_data or not manifest_path.exists():
        manifest_path = generate_demo_dataset(manifest_path.parent, overwrite=args.regenerate_data)
    counts = validate_manifest(manifest_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    set_reproducible_seed(7)
    if device.type == "cpu":
        torch.set_num_threads(1)

    _, node_feature_dim = infer_graph_dimensions(read_manifest(manifest_path))
    model = DSICNet(
        node_feature_dim=node_feature_dim,
        stage1_base_channels=4,
        stage1_attention_heads=(1, 2),
        stage2_image_base_channels=4,
        graph_hidden_dim=16,
        graph_embedding_dim=16,
        dropout=0.0,
        kappa=5.0,
    ).to(device)
    model.eval()

    rows: list[dict[str, str | float]] = []
    with torch.no_grad():
        for batch in create_data_loader(manifest_path, "test", batch_size=1):
            batch = batch_to_device(batch, device)
            prediction = model(batch["image"], batch["node_features"], batch["adjacency"])
            rows.append(
                {
                    "subject_id": batch["subject_id"][0],
                    "chronological_age": float(batch["age"].item()),
                    "stage1_age": float(prediction["stage1_age"].item()),
                    "image_residual": float(prediction["image_residual"].item()),
                    "base_age": float(prediction["base_age"].item()),
                    "graph_correction": float(prediction["graph_correction"].item()),
                    "final_age": float(prediction["final_age"].item()),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "ok",
        "note": "Randomly initialized weights; values are a software test, not scientific results.",
        "device": str(device),
        "manifest_counts": counts,
        "prediction": rows[0],
        "correction_bound_satisfied": abs(float(rows[0]["graph_correction"])) < 5.0,
        "output": str(output_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
