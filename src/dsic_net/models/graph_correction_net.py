"""Stage 2: bounded graph-guided residual correction network."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from .image_space_net import ConvGNAct3D


class Stage2Prediction(TypedDict):
    image_residual: torch.Tensor
    base_age: torch.Tensor
    graph_response: torch.Tensor
    graph_correction: torch.Tensor
    final_age: torch.Tensor


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Add self-loops and compute D^-1/2 (A + I) D^-1/2."""
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape (batch, nodes, nodes)")
    nodes = adjacency.shape[-1]
    identity = torch.eye(nodes, dtype=adjacency.dtype, device=adjacency.device)
    with_self_loops = adjacency + identity.unsqueeze(0)
    degree = with_self_loops.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt_degree = degree.rsqrt()
    return inv_sqrt_degree.unsqueeze(-1) * with_self_loops * inv_sqrt_degree.unsqueeze(-2)


class GraphConvolution(nn.Module):
    """GCN layer implementing ReLU(A_bar X W + b)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.bmm(adjacency, node_features))


class R2SNGraphEncoder(nn.Module):
    """Two-layer GCN, graph-level max pool, LayerNorm, and scalar projection."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gcn_layer_1 = GraphConvolution(node_feature_dim, hidden_dim)
        self.gcn_layer_2 = GraphConvolution(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.scalar_projection = nn.Linear(embedding_dim, 1)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        normalized_features = F.layer_norm(node_features, (node_features.shape[-1],))
        normalized_adjacency = normalize_adjacency(adjacency)
        hidden = F.relu(self.gcn_layer_1(normalized_features, normalized_adjacency))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.gcn_layer_2(hidden, normalized_adjacency))
        graph_embedding = hidden.max(dim=1).values
        return self.scalar_projection(self.layer_norm(graph_embedding))


class ImagingBackbone(nn.Module):
    """Stage-2 imaging backbone Phi_M used for image-conditioned correction."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvGNAct3D(in_channels, base_channels, stride=2),
            ConvGNAct3D(base_channels, 2 * base_channels, stride=2),
            ConvGNAct3D(2 * base_channels, 4 * base_channels, stride=2),
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(start_dim=1),
        )
        self.output_dim = 4 * base_channels

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.features(volume)


class ImageConditionedBasePrediction(nn.Module):
    """Methods Eqs. (4)-(6): image residual anchored to Stage-1 age."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        self.imaging_backbone = ImagingBackbone(in_channels, base_channels)
        age_embedding_dim = max(4, base_channels)
        self.stage1_age_embedding = nn.Sequential(
            nn.Linear(1, age_embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.residual_regression = nn.Sequential(
            nn.Linear(self.imaging_backbone.output_dim + age_embedding_dim, 4 * base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(4 * base_channels, 1),
        )

    def forward(
        self, volume: torch.Tensor, stage1_age: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.imaging_backbone(volume)
        age_features = self.stage1_age_embedding(stage1_age)
        image_residual = self.residual_regression(torch.cat((image_features, age_features), dim=1))
        return image_residual, stage1_age + image_residual


class BoundedGraphGuidedResidualCorrectionNet(nn.Module):
    """Methods Sec. 2.3.2 and Eq. (12)."""

    def __init__(
        self,
        node_feature_dim: int,
        in_channels: int = 1,
        image_base_channels: int = 8,
        graph_hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        graph_dropout: float = 0.1,
        kappa: float = 5.0,
    ) -> None:
        super().__init__()
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        self.image_conditioned_base_prediction = ImageConditionedBasePrediction(
            in_channels, image_base_channels
        )
        self.r2sn_graph_encoder = R2SNGraphEncoder(
            node_feature_dim,
            graph_hidden_dim,
            graph_embedding_dim,
            graph_dropout,
        )
        self.kappa = float(kappa)

    def forward(
        self,
        volume: torch.Tensor,
        stage1_age: torch.Tensor,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Stage2Prediction:
        image_residual, base_age = self.image_conditioned_base_prediction(volume, stage1_age)
        graph_response = self.r2sn_graph_encoder(node_features, adjacency)
        # Avoid floating-point tanh saturation returning exactly +/-1 so that
        # the strict Methods inequality |delta_g| < kappa also holds numerically.
        unit_limit = 1.0 - torch.finfo(graph_response.dtype).eps
        bounded_response = torch.tanh(graph_response).clamp(min=-unit_limit, max=unit_limit)
        graph_correction = self.kappa * bounded_response
        final_age = base_age + graph_correction
        return {
            "image_residual": image_residual,
            "base_age": base_age,
            "graph_response": graph_response,
            "graph_correction": graph_correction,
            "final_age": final_age,
        }
