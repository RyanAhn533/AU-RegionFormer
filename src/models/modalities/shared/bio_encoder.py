"""
Bio Encoder v2 — Bidirectional Mamba for Physiological Signals
===============================================================
V4-full / V4-lite 공통 모듈.

Architecture:
  raw bio (B, C, T) → PatchEmbed → (B, L, d) → Bi-Mamba ×N → (B, L, d)

Upgrade from v1 (TCN):
  - TCN: causal (past-only), fixed receptive field
  - Bi-Mamba: bidirectional, selective, linear complexity, unlimited range
  - EDA response peak 2-5s after stimulus → backward scan essential

Ablation support:
  - BioEncoderMamba: 2026-grade (default)
  - BioEncoderTCN: baseline for ablation comparison

K-EmoCon segment: 7ch × 160 timesteps (32Hz × 5sec)
  ch0: bvp, ch1: eda, ch2: temp, ch3-5: acc_xyz, ch6: hr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.shared.mamba_ssm import BiMambaBlock


# ─────────────────────────────────────────────────────────────
#  Patch Embedding for 1D multi-channel signals
# ─────────────────────────────────────────────────────────────
class BioSignalPatchEmbed(nn.Module):
    """
    Multi-channel 1D signal → non-overlapping patch tokens.

    (B, C, T) → (B, L, d) where L = T // patch_size

    Each patch captures a short temporal window across all channels,
    then projected to d_model. This is analogous to ViT's patch embedding
    but for 1D time series.
    """
    def __init__(self, in_channels: int = 7, patch_size: int = 10,
                 d_model: int = 128):
        super().__init__()
        self.patch_size = patch_size
        # Conv1d with kernel=stride=patch_size → non-overlapping patches
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                              stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, C, T) → (B, L, d)  where L = T // patch_size
        """
        x = self.proj(x)            # (B, d, L)
        x = x.transpose(1, 2)       # (B, L, d)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────
#  BioEncoderMamba: 2026-grade (Bi-Mamba)
# ─────────────────────────────────────────────────────────────
class BioEncoderMamba(nn.Module):
    """
    Bidirectional Mamba encoder for raw physiological signals.

    Pipeline:
      1. Input BatchNorm (per-channel normalization)
      2. Patch embedding: (B, 7, 160) → (B, 16, d)
      3. Positional embedding (learnable)
      4. Bi-Mamba blocks × n_layers
      5. Output: (B, L, d) bio token sequence

    Args:
        in_channels: bio channels (7: bvp, eda, temp, acc_xyz, hr)
        d_model: embedding dim
        patch_size: temporal patch size (160 / patch_size = L tokens)
        n_layers: number of Bi-Mamba blocks
        d_state: SSM state dimension
        d_conv: SSM conv width
        expand: SSM expansion factor
        dropout: dropout rate
    """
    def __init__(
        self,
        in_channels: int = 7,
        d_model: int = 128,
        patch_size: int = 10,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)

        # Patch embedding: (B, 7, 160) → (B, 16, d)
        self.patch_embed = BioSignalPatchEmbed(in_channels, patch_size, d_model)
        num_tokens = 160 // patch_size  # default: 16

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)

        # Bi-Mamba stack
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, C, T) raw bio signals → (B, L, d) bio tokens
        """
        x = self.input_norm(x)                  # (B, C, T)
        x = self.patch_embed(x)                  # (B, L, d)
        x = x + self.pos_embed[:, :x.size(1)]   # positional

        for layer in self.layers:
            x = layer(x)                         # Bi-Mamba residual

        return self.final_norm(x)                # (B, L, d)


# ─────────────────────────────────────────────────────────────
#  BioEncoderTCN: baseline for ablation
# ─────────────────────────────────────────────────────────────
class _TCNBlock(nn.Module):
    """Dilated causal convolution block with residual."""
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.trim = pad

    def forward(self, x):
        res = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)[:, :, :x.size(2)]))
        out = self.drop(out)
        out = F.relu(self.bn2(self.conv2(out)[:, :, :x.size(2)]))
        out = self.drop(out)
        return F.relu(out + res)


class BioEncoderTCN(nn.Module):
    """TCN baseline bio encoder for ablation comparison."""
    def __init__(self, in_channels=7, d_model=128, num_tokens=16,
                 hidden_dims=(32, 64, 128), kernel_size=5, dropout=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)
        layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(hidden_dims):
            layers.append(_TCNBlock(ch_in, ch_out, kernel_size, 2**i, dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden_dims[-1], d_model, 1)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, x):
        """x: (B, C, T) → (B, L, d)"""
        x = self.input_norm(x)
        x = self.tcn(x)
        x = self.proj(x)
        x = self.pool(x)
        return x.transpose(1, 2)


# ─────────────────────────────────────────────────────────────
#  Factory function
# ─────────────────────────────────────────────────────────────
def build_bio_encoder(variant: str = "mamba", **kwargs) -> nn.Module:
    """
    Build bio encoder by variant name.
    Args:
        variant: "mamba" (default, 2026-grade) or "tcn" (ablation baseline)
    """
    if variant == "mamba":
        return BioEncoderMamba(**kwargs)
    elif variant == "tcn":
        return BioEncoderTCN(**kwargs)
    else:
        raise ValueError(f"Unknown bio encoder variant: {variant}")
