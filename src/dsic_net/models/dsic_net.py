"""End-to-end DSIC-Net composition."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from .graph_correction_net import BoundedGraphGuidedResidualCorrectionNet
from .image_space_net import ImageSpaceNet


class DSICPrediction(TypedDict):
    stage1_age: torch.Tensor
    image_residual: torch.Tensor
    base_age: torch.Tensor
    graph_response: torch.Tensor
    graph_correction: torch.Tensor
    final_age: torch.Tensor


class DSICNet(nn.Module):
    """Two-stage cascaded prediction-and-correction network (DSIC-Net)."""

    def __init__(
        self,
        node_feature_dim: int,
        in_channels: int = 1,
        stage1_base_channels: int = 16,
        stage1_attention_heads: tuple[int, int] = (2, 4),
        stage2_image_base_channels: int = 8,
        graph_hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        dropout: float = 0.1,
        kappa: float = 5.0,
    ) -> None:
        super().__init__()
        self.image_space_net = ImageSpaceNet(
            in_channels,
            stage1_base_channels,
            stage1_attention_heads,
            dropout,
        )
        self.bounded_graph_guided_residual_correction_net = BoundedGraphGuidedResidualCorrectionNet(
            node_feature_dim=node_feature_dim,
            in_channels=in_channels,
            image_base_channels=stage2_image_base_channels,
            graph_hidden_dim=graph_hidden_dim,
            graph_embedding_dim=graph_embedding_dim,
            graph_dropout=dropout,
            kappa=kappa,
        )

    def forward(
        self,
        volume: torch.Tensor,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        detach_stage1: bool = False,
    ) -> DSICPrediction:
        stage1_age = self.image_space_net(volume)
        stage1_anchor = stage1_age.detach() if detach_stage1 else stage1_age
        stage2 = self.bounded_graph_guided_residual_correction_net(
            volume, stage1_anchor, node_features, adjacency
        )
        return {"stage1_age": stage1_age, **stage2}
