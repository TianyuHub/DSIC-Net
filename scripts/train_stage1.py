#!/usr/bin/env python3
"""Train Stage 1 (Image Space Net) from a manifest."""

from __future__ import annotations

import argparse
import json

import torch

from dsic_net.checkpoints import save_checkpoint
from dsic_net.data import create_data_loader, validate_manifest
from dsic_net.models import ImageSpaceNet
from dsic_net.training import evaluate_stage1, set_reproducible_seed, train_stage1_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/demo/manifest.csv")
    parser.add_argument("--output", default="outputs/stage1.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    validate_manifest(args.manifest)
    set_reproducible_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(1)
    heads = (1, 2) if args.base_channels % 2 == 0 else (1, 1)
    config = {
        "in_channels": 1,
        "base_channels": args.base_channels,
        "attention_heads": heads,
        "dropout": 0.1,
    }
    model = ImageSpaceNet(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    train_loader = create_data_loader(args.manifest, "train", batch_size=args.batch_size)
    val_loader = create_data_loader(args.manifest, "val", batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_stage1_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate_stage1(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "train_mse": train_loss, "val_mae": val_mae}))
    save_checkpoint(
        args.output,
        model,
        epoch=args.epochs,
        config=config,
        metrics={"val_mae": val_mae},
    )


if __name__ == "__main__":
    main()
