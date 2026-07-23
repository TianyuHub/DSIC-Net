#!/usr/bin/env python3
"""Run a trained DSIC-Net checkpoint on one manifest split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from dsic_net.data import create_data_loader, validate_manifest
from dsic_net.models import DSICNet
from dsic_net.training import batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    validate_manifest(args.manifest)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DSICNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in create_data_loader(args.manifest, args.split, batch_size=1, shuffle=False):
            batch = batch_to_device(batch, device)
            prediction = model(batch["image"], batch["node_features"], batch["adjacency"])
            rows.append(
                {
                    "subject_id": batch["subject_id"][0],
                    "chronological_age": float(batch["age"].item()),
                    "stage1_age": float(prediction["stage1_age"].item()),
                    "image_residual": float(prediction["image_residual"].item()),
                    "graph_correction": float(prediction["graph_correction"].item()),
                    "predicted_age": float(prediction["final_age"].item()),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
