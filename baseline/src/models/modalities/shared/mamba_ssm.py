"""
Bidirectional Selective State Space Model (Bi-Mamba)
=====================================================
Pure PyTorch — no external dependency (no einops, no mamba-ssm).

References:
  - Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  - Zhu et al. (2024) "Vision Mamba" (bidirectional extension)
  - EEGMamba (ICLR 2025) — Bi-Mamba for physiological signals
  - MambaMER (MICCAI 2025) — Selective cross-SSM for multimodal emotion

Modules:
  SelectiveSSM      — unidirectional selective scan
  BiMambaBlock      — forward + backward SSM with residual
  SelectiveCrossSSM — audio-modulated bio filtering + gated fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  SelectiveSSM: core Mamba S6 block
# ─────────────────────────────────────────────────────────────
class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6).
    Input-dependent Δ, B, C enable selective information filtering.
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        dbl = self.x_proj(x_ssm)
        dt_x = dbl[:, :, :self.dt_rank]
        B_x = dbl[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_x = dbl[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log)

        y = self._scan(x_ssm, dt, A, B_x, C_x)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

    def _scan(self, x, dt, A, B_p, C_p):
        """Sequential selective scan. Fast enough for L≤256."""
        Bs, L, D = x.shape
        N = self.d_state
        h = torch.zeros(Bs, D, N, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A.unsqueeze(0))
            dB = dt_t * B_p[:, t, :].unsqueeze(1)
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            outs.append((h * C_p[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(outs, dim=1)
        return y + self.D.unsqueeze(0).unsqueeze(0) * x


# ─────────────────────────────────────────────────────────────
#  BiMambaBlock: forward + backward SSM
# ─────────────────────────────────────────────────────────────
class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba = forward SSM + backward SSM + linear fusion.
    EEGMamba (ICLR 2025): Bi-Mamba >> unidirectional for physio signals.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = SelectiveSSM(d_model, d_state, d_conv, expand, dropout=dropout)
        self.bwd = SelectiveSSM(d_model, d_state, d_conv, expand, dropout=dropout)
        self.fuse = nn.Linear(d_model * 2, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, L, d) → (B, L, d) with residual"""
        res = x
        xn = self.norm(x)
        y_f = self.fwd(xn)
        y_b = torch.flip(self.bwd(torch.flip(xn, [1])), [1])
        return res + self.drop(self.fuse(torch.cat([y_f, y_b], dim=-1)))


# ─────────────────────────────────────────────────────────────
#  SelectiveCrossSSM: audio-modulated bio filtering
# ─────────────────────────────────────────────────────────────
class SelectiveCrossSSM(nn.Module):
    """
    Selective Cross-State-Space Conditioning (MambaMER-inspired).

    Audio context modulates bio's Δ → bio에서 audio-relevant 정보만
    선택적으로 통과. Vanilla cross-attention 대비:
      - Attention: 모든 bio 토큰에 softmax weight (noise도 비례 유입)
      - This: Δ gate가 bio 토큰별 pass/block, audio가 Δ를 modulate

    Output: conditioned audio = audio + gated(filtered_bio, attn_bio)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # Audio context → modulates bio selectivity
        self.audio_ctx = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

        # Bio pathway
        self.bio_in = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)

        # Cross-modal Δ
        self.cross_dt = nn.Linear(self.d_inner + d_model, self.d_inner, bias=True)

        # SSM params
        self.bc_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.ssm_out = nn.Linear(self.d_inner, d_model, bias=False)

        # Residual cross-attention
        self.xattn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Gated fusion
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )

    def forward(self, h_audio, h_bio, return_attn=False):
        """
        h_audio: (B, L, d), h_bio: (B, L, d)
        Returns: h_cond (B, L, d), attn_w or None
        """
        B, L, d = h_audio.shape

        # 1) Audio context
        ctx = self.audio_ctx(h_audio)

        # 2) Bio → inner
        bxz = self.bio_in(h_bio)
        bx, bz = bxz.chunk(2, dim=-1)
        bx = self.conv1d(bx.transpose(1, 2))[:, :, :L].transpose(1, 2)
        bx = F.silu(bx)

        # 3) Cross-modal Δ
        dt = F.softplus(self.cross_dt(torch.cat([bx, ctx], dim=-1)))
        bc = self.bc_proj(bx)
        B_x, C_x = bc[:, :, :self.d_state], bc[:, :, self.d_state:]
        A = -torch.exp(self.A_log)

        # 4) Selective scan
        D_dim = self.d_inner
        N = self.d_state
        h_s = torch.zeros(B, D_dim, N, device=h_bio.device, dtype=h_bio.dtype)
        outs = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A.unsqueeze(0))
            dB = dt_t * B_x[:, t, :].unsqueeze(1)
            h_s = dA * h_s + dB * bx[:, t, :].unsqueeze(-1)
            outs.append((h_s * C_x[:, t, :].unsqueeze(1)).sum(-1))

        bio_f = torch.stack(outs, dim=1)
        bio_f = bio_f + self.D.unsqueeze(0).unsqueeze(0) * bx
        bio_f = bio_f * F.silu(bz)
        bio_f = self.ssm_out(bio_f)

        # 5) Residual cross-attention
        q = self.norm_q(h_audio)
        kv = self.norm_kv(bio_f)
        a_out, a_w = self.xattn(q, kv, kv, need_weights=return_attn, average_attn_weights=True)
        if not return_attn:
            a_w = None

        # 6) Gated fusion
        g = self.gate(torch.cat([bio_f, a_out], dim=-1))
        h_c = h_audio + g * a_out + (1.0 - g) * bio_f
        h_c = h_c + self.ffn(h_c)

        return h_c, a_w
