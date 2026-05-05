"""
Cross-Attention Fusion v2
==========================

v1 대비 변경:
  1. AU Self-Attention 추가: AU tokens끼리 relation modeling 먼저
     → sad/hurt/anxious 같은 미세 감정은 AU 조합으로 구분되므로
        AU 간 관계를 먼저 학습한 뒤 CLS가 query해야 더 rich한 정보
  2. Cross → Self → FFN 순서 유지하되, AU self-attn이 앞에 추가
  3. 기존 1-layer 호환 유지 (n_layers=1이면 v1과 동일 구조 + AU self-attn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath


class CrossAttentionFusion(nn.Module):
    """
    [CLS] + [Global] + [AU_1, ..., AU_K] 토큰 시퀀스를 융합.

    Flow per layer:
      1. AU Self-Attention: AU tokens끼리 relation modeling
      2. Cross-Attention: CLS+Global이 AU tokens를 query
      3. Gated Residual
      4. Full Self-Attention: 전체 tokens 상호작용
      5. FFN
    """

    def __init__(self, d_model: int = 384, n_heads: int = 8,
                 n_layers: int = 1, dropout: float = 0.1,
                 gate_init: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Stochastic depth: linearly increasing drop rate per layer
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]
        self.layers = nn.ModuleList([
            FusionLayer(d_model, n_heads, dropout, gate_init, drop_path=dpr[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, cls_token: torch.Tensor, global_feat: torch.Tensor,
                au_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token:   [B, d] learnable CLS token (expanded)
            global_feat: [B, d] global image feature
            au_tokens:   [B, K, d] AU region features

        Returns:
            tokens: [B, K+2, d] fused token sequence (CLS at index 0)
        """
        B, K, d = au_tokens.shape

        # Assemble token sequence: [CLS, Global, AU_1, ..., AU_K]
        tokens = torch.cat([
            cls_token.unsqueeze(1),    # [B, 1, d]
            global_feat.unsqueeze(1),  # [B, 1, d]
            au_tokens,                 # [B, K, d]
        ], dim=1)  # [B, K+2, d]

        for layer in self.layers:
            tokens = layer(tokens, K)

        return self.final_norm(tokens)


class FusionLayer(nn.Module):
    """Single fusion layer: AU-Self → Cross → Full-Self → FFN"""

    def __init__(self, d_model, n_heads, dropout, gate_init, drop_path=0.0):
        super().__init__()

        # DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # AU Self-Attention: AU tokens끼리 relation modeling
        self.au_self_norm = nn.LayerNorm(d_model)
        self.au_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-Attention: Q=[CLS,Global], KV=[AU tokens]
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Learnable gate (per-dim for fine control)
        self.gate = nn.Parameter(torch.full((d_model,), gate_init))

        # Self-Attention: all tokens
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, K: int) -> torch.Tensor:
        """
        tokens: [B, K+2, d] where first 2 are CLS and Global
        K: number of AU tokens
        """
        # ── AU Self-Attention: AU tokens끼리 relation modeling ──
        au_tokens = tokens[:, 2:, :]  # [B, K, d]
        au_normed = self.au_self_norm(au_tokens)
        au_out, _ = self.au_self_attn(au_normed, au_normed, au_normed)
        au_tokens = au_tokens + self.drop_path(au_out)
        tokens = torch.cat([tokens[:, :2, :], au_tokens], dim=1)

        # ── Cross-Attention: CLS+Global query AU tokens ──
        q_tokens = tokens[:, :2, :]   # [B, 2, d]
        kv_tokens = tokens[:, 2:, :]  # [B, K, d]

        q_normed = self.cross_norm_q(q_tokens)
        kv_normed = self.cross_norm_kv(kv_tokens)

        cross_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)

        # Gated residual: sigmoid gate controls information flow
        gate = torch.sigmoid(self.gate)  # [d]
        q_tokens = q_tokens + gate * self.drop_path(cross_out)

        # Reassemble
        tokens = torch.cat([q_tokens, kv_tokens], dim=1)

        # ── Self-Attention: all tokens interact ──
        normed = self.self_norm(tokens)
        self_out, _ = self.self_attn(normed, normed, normed)
        tokens = tokens + self.drop_path(self_out)

        # ── FFN ──
        tokens = tokens + self.drop_path(self.ffn(self.ffn_norm(tokens)))

        return tokens
