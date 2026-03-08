"""
Quality-Aware Gating & Cross-Modal Conditioning v2
====================================================
V4-full (3-modal), V4-lite (2-modal) 공통.

Upgrade:
  - AudioBioConditioner → now wraps SelectiveCrossSSM (default)
    or vanilla cross-attention (ablation mode)
  - QualityAwareGate: unchanged (already good design)
  - UncertaintyWeightedLoss: NEW — auto-balances multi-task loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.shared.mamba_ssm import SelectiveCrossSSM


# ─────────────────────────────────────────────────────────────
#  Conditioner: SelectiveCrossSSM (default) or vanilla (ablation)
# ─────────────────────────────────────────────────────────────
class VanillaCrossAttention(nn.Module):
    """Vanilla cross-attention conditioner (ablation baseline)."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, h_audio, h_bio, return_attn=False):
        q = self.norm_q(h_audio)
        kv = self.norm_kv(h_bio)
        out, w = self.mha(q, kv, kv, need_weights=return_attn, average_attn_weights=True)
        if not return_attn:
            w = None
        h = h_audio + out
        h = h + self.ffn(self.norm_ffn(h))
        return h, w


class AudioBioConditioner(nn.Module):
    """
    Cross-modal conditioner: audio queries bio.

    variant="selective_ssm" (default):
      MambaMER-inspired selective cross-state-space model.
      Audio context modulates bio's Δ → selective filtering.

    variant="cross_attention" (ablation):
      Vanilla multi-head cross-attention.
    """
    def __init__(self, d_model: int, n_heads: int = 4,
                 variant: str = "selective_ssm", dropout: float = 0.1):
        super().__init__()
        self.variant = variant
        if variant == "selective_ssm":
            self.module = SelectiveCrossSSM(
                d_model, d_state=16, d_conv=4, expand=2,
                n_heads=n_heads, dropout=dropout,
            )
        elif variant == "cross_attention":
            self.module = VanillaCrossAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown conditioner variant: {variant}")

    def forward(self, h_audio, h_bio, return_attn=False):
        return self.module(h_audio, h_bio, return_attn=return_attn)


# ─────────────────────────────────────────────────────────────
#  Quality Estimation
# ─────────────────────────────────────────────────────────────
class QualityEstimator(nn.Module):
    """q = σ(w · z + b) — neural proxy for sensor reliability."""
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, z):
        """z: (B, d) → q: (B, 1) ∈ (0, 1)"""
        return torch.sigmoid(self.head(z))


# ─────────────────────────────────────────────────────────────
#  Quality-Aware Gate
# ─────────────────────────────────────────────────────────────
class QualityAwareGate(nn.Module):
    """
    Quality-aware N-modal gating.

    g = MLP([z_1; ...; z_N; q_1; ...; q_N])
    α = softmax(g / τ)
    z_fused = Σ α_i · z_i

    Temperature τ annealing:
      τ=2.0 (early): soft mixing → stable training
      τ=0.5 (late): sharp gating → modality selection
    """
    def __init__(self, d_model: int, n_modalities: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_mod = n_modalities
        in_dim = d_model * n_modalities + n_modalities
        self.gate = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_modalities),
        )
        self.temperature = 1.0

    def set_temperature(self, tau: float):
        self.temperature = max(tau, 0.01)

    def forward(self, z_list, q_list):
        """
        z_list: [(B, d)] × N, q_list: [(B, 1)] × N
        Returns: z_fused (B, d), gate_weights (B, N)
        """
        cat = torch.cat(z_list + q_list, dim=-1)
        logits = self.gate(cat)
        weights = F.softmax(logits / self.temperature, dim=-1)
        z_stack = torch.stack(z_list, dim=1)
        z_fused = (weights.unsqueeze(-1) * z_stack).sum(dim=1)
        return z_fused, weights


# ─────────────────────────────────────────────────────────────
#  Uncertainty-Weighted Multi-Task Loss
# ─────────────────────────────────────────────────────────────
class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall et al. (CVPR 2018) "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics."

    L = Σ (1 / 2σ²_i) · L_i  +  Σ log(σ_i)

    Learns per-task σ_i (homoscedastic uncertainty) that automatically
    balances loss contributions. Key benefits:
      - V_bin의 σ가 자동으로 커짐 → imbalanced task 기여 감소
      - Manual λ_A, λ_V, β_A, β_V 전부 제거 → cleaner ablation
      - Regularizer log(σ) prevents σ → ∞

    Usage:
        uwl = UncertaintyWeightedLoss(n_tasks=4)
        total_loss = uwl([loss_A_reg, loss_V_reg, loss_A_bin, loss_V_bin])
    """
    def __init__(self, n_tasks: int = 4):
        super().__init__()
        # Initialize log(σ²) = 0 → σ = 1 → equal weights initially
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list):
        """
        losses: list of scalar loss tensors, len = n_tasks
        Returns: weighted total loss (scalar)
        """
        total = 0.0
        for i, loss in enumerate(losses):
            # precision = 1 / (2 * σ²) = 1 / (2 * exp(log_var))
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    def get_weights(self):
        """Current effective weights (for logging)."""
        with torch.no_grad():
            sigmas = torch.exp(self.log_vars * 0.5)
            weights = 1.0 / (2.0 * sigmas ** 2)
            return {
                f"task_{i}_sigma": sigmas[i].item()
                for i in range(len(self.log_vars))
            } | {
                f"task_{i}_weight": weights[i].item()
                for i in range(len(self.log_vars))
            }
