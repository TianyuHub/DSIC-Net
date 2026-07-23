#!/usr/bin/env python3
"""Train Stage 2 while retaining a frozen Stage-1 anchor."""

from __future__ import annotations

import argparse
import json

import torch

from dsic_net.checkpoints import save_checkpoint
from dsic_net.data import (
    create_data_loader,
    infer_graph_dimensions,
    read_manifest,
    validate_manifest,
)
from dsic_net.models import DSICNet
from dsic_net.training import evaluate_stage2, set_reproducible_seed, train_stage2_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/demo/manifest.csv")
    parser.add_argument("--stage1-checkpoint", default="outputs/stage1.pt")
    parser.add_argument("--output", default="outputs/dsic_net.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kappa", type=float, default=5.0)
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
    stage1_checkpoint = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    stage1_config = stage1_checkpoint["config"]
    _, node_feature_dim = infer_graph_dimensions(read_manifest(args.manifest))
    config = {
        "node_feature_dim": node_feature_dim,
        "in_channels": stage1_config["in_channels"],
        "stage1_base_channels": stage1_config["base_channels"],
        "stage1_attention_heads": tuple(stage1_config["attention_heads"]),
        "stage2_image_base_channels": 4,
        "graph_hidden_dim": 16,
        "graph_embedding_dim": 16,
        "dropout": 0.1,
        "kappa": args.kappa,
    }
    model = DSICNet(**config).to(device)
    model.image_space_net.load_state_dict(stage1_checkpoint["model_state_dict"])
    for parameter in model.image_space_net.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.bounded_graph_guided_residual_correction_net.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4,
    )
    train_loader = create_data_loader(args.manifest, "train", batch_size=args.batch_size)
    val_loader = create_data_loader(args.manifest, "val", batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_stage2_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate_stage2(model, val_loader, device)
        print(json.dumps({"epoch": epoch, "train_smooth_l1": train_loss, "val_mae": val_mae}))
    save_checkpoint(
        args.output,
        model,
        epoch=args.epochs,
        config=config,
        metrics={"val_mae": val_mae},
    )


if __name__ == "__main__":
    main()
