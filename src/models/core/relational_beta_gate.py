"""
Relational Beta Uncertainty Gate.

Three Beta heads available, combinable per-experiment:
  - IsolatedBetaHead:    r_iso   from AU_i alone        (current L2)
  - RelationalBetaHead:  r_rel   from [global, AU_i]    (NEW — AU↔Global agreement)
  - DualBetaGate:        combines r_iso + r_rel via learnable mix → softmax token gate

Design rationale: Tokens that come from independent AU patch encoders carry
genuine information; Beta gating must measure both per-token reliability AND
relationship to the global face. softmax over combined r forces token competition.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _beta_r(alpha, beta):
    return alpha / (alpha + beta)


class _BetaHead(nn.Module):
    """Predict Beta(α, β) from a token-level input.

    Last layer initialized to zero so initial output is α≈β≈ln 2 → r≈0.5 (uniform).
    """

    def __init__(self, in_dim: int, hidden: Optional[int] = None, eps: float = 1e-3):
        super().__init__()
        h = hidden if hidden is not None else max(64, in_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )
        self.eps = eps
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.net(x)
        alpha = F.softplus(ab[..., 0]) + self.eps
        beta = F.softplus(ab[..., 1]) + self.eps
        return alpha, beta


class IsolatedBetaHead(nn.Module):
    """r_iso: each AU token's reliability from its embedding alone."""

    def __init__(self, d_emb: int):
        super().__init__()
        self.head = _BetaHead(d_emb)

    def forward(self, au_tokens: torch.Tensor) -> torch.Tensor:
        # au_tokens: [B, K, D]
        a, b = self.head(au_tokens)
        return _beta_r(a, b)  # [B, K]


class RelationalBetaHead(nn.Module):
    """r_rel: each AU token's agreement with global token.

    Input = concat([global_feat broadcast to K, au_token]) → 2D-d.
    """

    def __init__(self, d_emb: int):
        super().__init__()
        self.head = _BetaHead(2 * d_emb)

    def forward(self, global_feat: torch.Tensor, au_tokens: torch.Tensor) -> torch.Tensor:
        # global_feat: [B, D], au_tokens: [B, K, D]
        K = au_tokens.shape[1]
        g = global_feat.unsqueeze(1).expand(-1, K, -1)
        rel_in = torch.cat([g, au_tokens], dim=-1)         # [B, K, 2D]
        a, b = self.head(rel_in)
        return _beta_r(a, b)                                # [B, K]


class DualBetaGate(nn.Module):
    """Combine r_iso + r_rel into a softmax token gate.

    Modes for `combine`:
      "iso_only"    : r = r_iso
      "rel_only"    : r = r_rel
      "product"     : r = r_iso * r_rel
      "weighted"    : r = w * r_iso + (1-w) * r_rel  (learnable scalar w)

    Final gating: gated_i = K * softmax(r/τ)_i * AU_i (token competition).
    """

    def __init__(self,
                 d_emb: int,
                 use_iso: bool = True,
                 use_rel: bool = True,
                 combine: str = "weighted",
                 tau: float = 1.0):
        super().__init__()
        assert use_iso or use_rel, "At least one of use_iso/use_rel must be True"
        if combine not in ("iso_only", "rel_only", "product", "weighted"):
            raise ValueError(f"unknown combine mode: {combine}")
        if combine == "iso_only" and not use_iso:
            raise ValueError("combine='iso_only' requires use_iso=True")
        if combine == "rel_only" and not use_rel:
            raise ValueError("combine='rel_only' requires use_rel=True")
        if combine in ("product", "weighted") and not (use_iso and use_rel):
            raise ValueError(f"combine='{combine}' requires both heads")

        self.use_iso = use_iso
        self.use_rel = use_rel
        self.combine = combine

        if use_iso:
            self.iso_head = IsolatedBetaHead(d_emb)
        if use_rel:
            self.rel_head = RelationalBetaHead(d_emb)

        if combine == "weighted":
            # learnable scalar in (0,1) via sigmoid
            self.mix_logit = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("tau", torch.tensor(float(tau)))

    def set_tau(self, tau: float):
        self.tau.fill_(float(tau))

    def forward(self,
                global_feat: torch.Tensor,
                au_tokens: torch.Tensor):
        """
        Args:
            global_feat: [B, D]
            au_tokens:   [B, K, D]
        Returns:
            gated_au_tokens: [B, K, D]
            r_combined:      [B, K] (final reliability scores)
            aux: dict with optional r_iso, r_rel for analysis
        """
        aux = {}
        r_iso = r_rel = None
        if self.use_iso:
            r_iso = self.iso_head(au_tokens)           # [B, K]
            aux["r_iso"] = r_iso.detach()
        if self.use_rel:
            r_rel = self.rel_head(global_feat, au_tokens)
            aux["r_rel"] = r_rel.detach()

        if self.combine == "iso_only":
            r = r_iso
        elif self.combine == "rel_only":
            r = r_rel
        elif self.combine == "product":
            r = r_iso * r_rel
        else:  # "weighted"
            w = torch.sigmoid(self.mix_logit)
            r = w * r_iso + (1.0 - w) * r_rel
            aux["mix_w"] = float(w.detach())

        K = au_tokens.shape[1]
        weights = F.softmax(r / self.tau.clamp(min=1e-3), dim=1)   # [B, K]
        gated = au_tokens * (K * weights).unsqueeze(-1)
        return gated, r, aux
