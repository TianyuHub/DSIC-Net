from __future__ import annotations

import torch

from dsic_net.models import DSICNet, ImageSpaceNet, LocalSpectralAxialFusionBlock
from dsic_net.models.graph_correction_net import normalize_adjacency


def test_lsafb_preserves_shape_and_supports_gradients() -> None:
    block = LocalSpectralAxialFusionBlock(channels=4, num_heads=1)
    features = torch.randn(2, 4, 5, 6, 7, requires_grad=True)
    output = block(features)
    assert output.shape == features.shape
    output.mean().backward()
    assert features.grad is not None


def test_image_space_net_returns_one_age_per_subject() -> None:
    model = ImageSpaceNet(base_channels=4, attention_heads=(1, 2), dropout=0.0)
    prediction = model(torch.randn(2, 1, 24, 24, 24))
    assert prediction.shape == (2, 1)


def test_dsic_equation_and_graph_bound() -> None:
    kappa = 3.0
    model = DSICNet(
        node_feature_dim=6,
        stage1_base_channels=4,
        stage1_attention_heads=(1, 2),
        stage2_image_base_channels=4,
        graph_hidden_dim=8,
        graph_embedding_dim=8,
        dropout=0.0,
        kappa=kappa,
    )
    with torch.no_grad():
        graph_projection = (
            model.bounded_graph_guided_residual_correction_net.r2sn_graph_encoder.scalar_projection
        )
        graph_projection.bias.fill_(1e6)
    prediction = model(
        torch.randn(2, 1, 24, 24, 24),
        torch.randn(2, 10, 6),
        torch.rand(2, 10, 10),
    )
    assert prediction["final_age"].shape == (2, 1)
    assert torch.allclose(
        prediction["base_age"],
        prediction["stage1_age"] + prediction["image_residual"],
    )
    assert torch.allclose(
        prediction["final_age"],
        prediction["base_age"] + prediction["graph_correction"],
    )
    assert torch.all(prediction["graph_correction"].abs() < kappa)


def test_normalized_adjacency_is_symmetric() -> None:
    adjacency = torch.tensor([[[0.0, 0.5], [0.5, 0.0]]])
    normalized = normalize_adjacency(adjacency)
    assert torch.allclose(normalized, normalized.transpose(-1, -2))
    assert torch.isfinite(normalized).all()
