"""Stage 1: Image Space Net and the Local-Spectral-Axial Fusion Block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _valid_group_count(channels: int, requested: int = 8) -> int:
    groups = min(channels, requested)
    while channels % groups != 0:
        groups -= 1
    return groups


class ConvGNAct3D(nn.Sequential):
    """3D convolution, Group Normalization, and GELU (ConvGNA in Methods)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class SqueezeExcitation3D(nn.Module):
    """SE3D channel recalibration used by the local branch."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.reduce = nn.Conv3d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        descriptor = F.adaptive_avg_pool3d(features, output_size=1)
        weights = torch.sigmoid(self.expand(F.gelu(self.reduce(descriptor))))
        return features * weights


class SpectralGating3D(nn.Module):
    """FFT -> learnable channel gate -> IFFT spectral-domain branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 2 * sigmoid(0) = 1, so the branch starts as an identity transform.
        self.logits = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_shape = features.shape[-3:]
        spectrum = torch.fft.rfftn(features, dim=(-3, -2, -1), norm="ortho")
        gated_spectrum = spectrum * (2.0 * torch.sigmoid(self.logits))
        return torch.fft.irfftn(
            gated_spectrum,
            s=spatial_shape,
            dim=(-3, -2, -1),
            norm="ortho",
        )


class AxialAttention3D(nn.Module):
    """Multi-head attention applied independently along D, H, and W."""

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.depth_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.height_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.width_attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )

    @staticmethod
    def _self_attention(module: nn.MultiheadAttention, sequence: torch.Tensor) -> torch.Tensor:
        attended, _ = module(sequence, sequence, sequence, need_weights=False)
        return attended

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = features.shape
        channels_last = features.permute(0, 2, 3, 4, 1).contiguous()

        depth_seq = channels_last.permute(0, 2, 3, 1, 4).reshape(
            batch * height * width, depth, channels
        )
        depth_out = self._self_attention(self.depth_attention, depth_seq)
        depth_out = depth_out.reshape(batch, height, width, depth, channels).permute(0, 3, 1, 2, 4)

        height_seq = channels_last.permute(0, 1, 3, 2, 4).reshape(
            batch * depth * width, height, channels
        )
        height_out = self._self_attention(self.height_attention, height_seq)
        height_out = height_out.reshape(batch, depth, width, height, channels).permute(
            0, 1, 3, 2, 4
        )

        width_seq = channels_last.reshape(batch * depth * height, width, channels)
        width_out = self._self_attention(self.width_attention, width_seq)
        width_out = width_out.reshape(batch, depth, height, width, channels)

        fused = (depth_out + height_out + width_out) / 3.0
        return fused.permute(0, 4, 1, 2, 3).contiguous()


class LocalSpectralAxialFusionBlock(nn.Module):
    """LSAFB from Methods Eq. (3): local, spectral, and axial residual fusion."""

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.local_branch = nn.Sequential(
            ConvGNAct3D(channels, channels),
            ConvGNAct3D(channels, channels),
            SqueezeExcitation3D(channels),
        )
        self.spectral_branch = SpectralGating3D(channels)
        self.axial_branch = AxialAttention3D(channels, num_heads, dropout)
        self.fuse = nn.Conv3d(3 * channels, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        local = self.local_branch(features)
        spectral = self.spectral_branch(features)
        axial = self.axial_branch(features)
        return features + self.fuse(torch.cat((local, spectral, axial), dim=1))


class ImageSpaceNet(nn.Module):
    """Stage-1 image-space brain-age estimator described in Methods Sec. 2.3.1."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        attention_heads: tuple[int, int] = (2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(attention_heads) != 2:
            raise ValueError("attention_heads must contain one value per LSAFB group")

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(base_channels), base_channels),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.lsafb_group_1 = nn.Sequential(
            *[
                LocalSpectralAxialFusionBlock(base_channels, attention_heads[0], dropout)
                for _ in range(2)
            ]
        )
        self.downsample = ConvGNAct3D(
            base_channels,
            2 * base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.lsafb_group_2 = nn.Sequential(
            *[
                LocalSpectralAxialFusionBlock(2 * base_channels, attention_heads[1], dropout)
                for _ in range(2)
            ]
        )
        self.global_average_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.regression_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(2 * base_channels, 4 * base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * base_channels, 1),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        features = self.stem(volume)
        features = self.lsafb_group_1(features)
        features = self.downsample(features)
        features = self.lsafb_group_2(features)
        return self.regression_head(self.global_average_pool(features))
