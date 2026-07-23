"""Minimal reproducible training loops for the two DSIC-Net stages."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_stage1(model: nn.Module, loader: Iterable[dict], device: torch.device) -> float:
    model.eval()
    absolute_errors: list[torch.Tensor] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        absolute_errors.append((model(batch["image"]) - batch["age"]).abs().cpu())
    return float(torch.cat(absolute_errors).mean())


def train_stage1_epoch(
    model: nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["image"])
        loss = nn.functional.mse_loss(prediction, batch["age"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_stage2(model: nn.Module, loader: Iterable[dict], device: torch.device) -> float:
    model.eval()
    absolute_errors: list[torch.Tensor] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        prediction = model(
            batch["image"],
            batch["node_features"],
            batch["adjacency"],
            detach_stage1=True,
        )
        absolute_errors.append((prediction["final_age"] - batch["age"]).abs().cpu())
    return float(torch.cat(absolute_errors).mean())


def train_stage2_epoch(
    model: nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    model.image_space_net.eval()
    losses: list[float] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch["image"],
            batch["node_features"],
            batch["adjacency"],
            detach_stage1=True,
        )
        loss = nn.functional.smooth_l1_loss(prediction["final_age"], batch["age"])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))
