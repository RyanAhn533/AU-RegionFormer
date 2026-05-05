"""
Beta uncertainty gating for AU-RegionFormer.
Reference: SGMT-BU Beta gate (KL OFF mode validated KEMDy20 2.4% → 32.6% bio gate weight).

Three modes (paper ablation):
  L2: per-AU-token gate (this module's primary use)
  L3: stream-level gate (Global-token vs AU-stream)
  L1: per-AU-intensity gate (input level — applies before token formation)

Each level computes Beta(α, β) over its target tokens, derives reliability r = α/(α+β),
and uses r as a soft gating weight (or τ-tempered softmax).

Default: KL OFF, τ annealed 2 → 1 over training (set externally via set_tau()).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaUncertaintyHead(nn.Module):
    """Predict Beta(α, β) per token from token embedding."""

    def __init__(self, d_emb: int, hidden: int = None, eps: float = 1e-3):
        super().__init__()
        h = hidden if hidden is not None else max(64, d_emb // 4)
        self.net = nn.Sequential(
            nn.Linear(d_emb, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )
        self.eps = eps
        # init last layer near zero so α ≈ β ≈ softplus(0) = ln 2 ≈ 0.69 (uniform-ish)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: [..., d_emb]
        Returns:
            alpha, beta: [...]  (each strictly positive)
        """
        ab = self.net(tokens)                # [..., 2]
        alpha = F.softplus(ab[..., 0]) + self.eps
        beta  = F.softplus(ab[..., 1]) + self.eps
        return alpha, beta


def beta_reliability(alpha, beta):
    """Mean of Beta(α, β) = α/(α+β) — reliability score in (0, 1)."""
    return alpha / (alpha + beta)


def beta_variance(alpha, beta):
    """Variance of Beta(α, β) — uncertainty proxy."""
    s = alpha + beta
    return (alpha * beta) / (s * s * (s + 1.0))


class TokenBetaGate(nn.Module):
    """
    Apply Beta gating to a sequence of tokens [B, K, D].
    Output: gated tokens of same shape, plus per-token reliability scores.

    Modes:
      "scale"   : token_gated_i = r_i * token_i           (soft scaling)
      "softmax" : token_gated_i = K * w_i * token_i  with w = softmax(r/τ)
                  (preserves total magnitude; competition between tokens)
    """

    def __init__(self, d_emb: int, mode: str = "scale", tau: float = 1.0):
        super().__init__()
        assert mode in ("scale", "softmax")
        self.head = BetaUncertaintyHead(d_emb)
        self.mode = mode
        # Tau as buffer so it can be modified externally without re-creating module
        self.register_buffer("tau", torch.tensor(float(tau)))

    def set_tau(self, tau: float):
        self.tau.fill_(float(tau))

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: [B, K, D]
        Returns:
            gated_tokens: [B, K, D]
            reliability: [B, K]
            variance:    [B, K]
        """
        alpha, beta = self.head(tokens)              # [B, K] each
        r = beta_reliability(alpha, beta)            # [B, K]
        v = beta_variance(alpha, beta)               # [B, K]

        if self.mode == "scale":
            gated = tokens * r.unsqueeze(-1)
        else:  # softmax
            K = tokens.shape[1]
            w = F.softmax(r / self.tau.clamp(min=1e-3), dim=1)   # [B, K]
            gated = tokens * (K * w).unsqueeze(-1)

        return gated, r, v


class StreamBetaGate(nn.Module):
    """
    L3-style: gate two streams (e.g., Global token vs AU mean token).
    Returns blended single representation.
    """

    def __init__(self, d_emb: int, n_streams: int = 2, tau: float = 1.0):
        super().__init__()
        self.heads = nn.ModuleList([BetaUncertaintyHead(d_emb) for _ in range(n_streams)])
        self.n_streams = n_streams
        self.register_buffer("tau", torch.tensor(float(tau)))

    def set_tau(self, tau: float):
        self.tau.fill_(float(tau))

    def forward(self, *streams):
        """
        Args:
            *streams: each [B, D]
        Returns:
            blended: [B, D]
            reliability: [B, n_streams]
        """
        assert len(streams) == self.n_streams
        rs = []
        for h, s in zip(self.heads, streams):
            a, b = h(s)
            rs.append(beta_reliability(a, b))           # [B]
        r = torch.stack(rs, dim=1)                       # [B, n_streams]
        w = F.softmax(r / self.tau.clamp(min=1e-3), dim=1)  # [B, n_streams]

        stack = torch.stack(streams, dim=1)              # [B, n_streams, D]
        blended = (w.unsqueeze(-1) * stack).sum(dim=1)   # [B, D]
        return blended, r
