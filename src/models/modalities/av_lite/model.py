"""
V4-lite v2: Audio + Bio → Continuous & Binary A/V Prediction
==============================================================
2026-grade architecture with:
  - Bi-Mamba bio encoder (replaces TCN)
  - Selective Cross-SSM conditioning (replaces vanilla cross-attention)
  - Multi-head output (continuous regression + binary auxiliary)
  - Modality dropout for missing-sensor robustness
  - Uncertainty-weighted multi-task loss

Contribution statement:
  "V4-lite employs bidirectional state-space models for bio-signal encoding
   and selective cross-state-space conditioning for audio-bio alignment,
   achieving robust A/V estimation under partial sensor failure on edge devices."

Ablation support:
  - bio_variant: "mamba" (default) vs "tcn"
  - conditioner_variant: "selective_ssm" (default) vs "cross_attention"
  - modality_dropout: True/False
  - uncertainty_loss: True/False
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.shared.bio_encoder import build_bio_encoder
from modalities.shared.audio_encoder import AudioEncoderStudent, DistillationProjector
from modalities.shared.quality_gate import (
    AudioBioConditioner, QualityEstimator, QualityAwareGate,
    UncertaintyWeightedLoss,
)


class AVLiteModel(nn.Module):
    """
    V4-lite v2: 2-modal (audio + bio) → A/V continuous + binary.

    Args:
        d_model: embedding dimension
        num_tokens: temporal token count (L = 160/patch_size for Mamba, explicit for TCN)
        bio_channels: number of bio signal channels
        bio_variant: "mamba" or "tcn"
        conditioner_variant: "selective_ssm" or "cross_attention"
        n_heads: attention heads
        dropout: dropout rate
        modality_dropout: enable modality dropout during training
        p_drop_audio: prob of dropping audio (training only)
        p_drop_bio: prob of dropping bio (training only)
        use_speaker_embed: include speaker embedding
        d_teacher: teacher embedding dim (0 to disable distillation)
    """
    def __init__(
        self,
        d_model: int = 128,
        num_tokens: int = 16,
        bio_channels: int = 7,
        bio_variant: str = "mamba",
        bio_n_layers: int = 2,
        conditioner_variant: str = "selective_ssm",
        n_heads: int = 4,
        dropout: float = 0.2,
        modality_dropout: bool = True,
        p_drop_audio: float = 0.1,
        p_drop_bio: float = 0.3,
        use_speaker_embed: bool = True,
        d_teacher: int = 768,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.use_speaker = use_speaker_embed
        self.modality_dropout = modality_dropout
        self.p_drop_audio = p_drop_audio
        self.p_drop_bio = p_drop_bio

        # ── Encoders ──
        self.audio_enc = AudioEncoderStudent(
            d_model=d_model, num_tokens=num_tokens, dropout=dropout,
        )

        bio_kwargs = dict(in_channels=bio_channels, d_model=d_model, dropout=dropout)
        if bio_variant == "mamba":
            bio_kwargs.update(dict(
                patch_size=160 // num_tokens,  # 160/16=10
                n_layers=bio_n_layers,
                d_state=16, d_conv=4, expand=2,
            ))
        else:  # tcn
            bio_kwargs.update(dict(
                num_tokens=num_tokens,
                hidden_dims=(32, 64, 128),
                kernel_size=5,
            ))
        self.bio_enc = build_bio_encoder(bio_variant, **bio_kwargs)

        # ── Embeddings ──
        self.pos_embed = nn.Embedding(num_tokens, d_model)
        self.type_embed = nn.Embedding(2, d_model)  # 0=audio, 1=bio
        if use_speaker_embed:
            self.speaker_embed = nn.Embedding(2, d_model)

        # ── Cross-modal conditioning ──
        self.conditioner = AudioBioConditioner(
            d_model, n_heads=n_heads,
            variant=conditioner_variant, dropout=dropout,
        )

        # ── Quality estimation ──
        self.q_audio = QualityEstimator(d_model)
        self.q_bio = QualityEstimator(d_model)

        # ── Quality-aware gate ──
        self.gate = QualityAwareGate(d_model, n_modalities=2, dropout=dropout)

        # ── Multi-head output ──
        # Continuous regression heads (main task)
        self.head_A_reg = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Tanh(),  # output ∈ [-1, 1]
        )
        self.head_V_reg = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Tanh(),
        )
        # Binary auxiliary heads
        self.head_A_bin = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.head_V_bin = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # ── Uncertainty-weighted loss ──
        # Tasks: [A_reg, V_reg, A_bin, V_bin]
        self.uncertainty_loss = UncertaintyWeightedLoss(n_tasks=4)

        # ── Distillation projector ──
        self.d_teacher = d_teacher
        if d_teacher > 0:
            self.kd_proj = DistillationProjector(d_model, d_teacher)
        else:
            self.kd_proj = None

    def set_gate_temperature(self, tau: float):
        self.gate.set_temperature(tau)

    def _apply_modality_dropout(self, h_audio, h_bio):
        """
        Training-only: randomly zero out entire modality.
        Quality gate learns to handle missing modality gracefully.
        """
        if not self.training or not self.modality_dropout:
            return h_audio, h_bio

        if random.random() < self.p_drop_audio:
            h_audio = torch.zeros_like(h_audio)
        if random.random() < self.p_drop_bio:
            h_bio = torch.zeros_like(h_bio)

        return h_audio, h_bio

    def forward(
        self,
        logmel: torch.Tensor,
        bio_raw: torch.Tensor,
        speaker_ids: torch.Tensor = None,
        return_artifacts: bool = False,
    ):
        """
        Args:
            logmel:      (B, 1, 64, T_mel) log-mel spectrogram
            bio_raw:     (B, C, T_bio) raw bio signals (7ch, 160 steps)
            speaker_ids: (B,) speaker ID (0 or 1), optional
            return_artifacts: return intermediate values for analysis

        Returns dict:
            A_reg:   (B, 1) continuous arousal ∈ [-1, 1]
            V_reg:   (B, 1) continuous valence ∈ [-1, 1]
            A_logit: (B, 1) binary arousal logit
            V_logit: (B, 1) binary valence logit
            artifacts: dict (optional)
        """
        B = logmel.size(0)
        device = logmel.device
        t_idx = torch.arange(self.num_tokens, device=device).unsqueeze(0).expand(B, -1)

        # ── Encode ──
        h_audio = self.audio_enc(logmel)     # (B, L, d)
        h_bio = self.bio_enc(bio_raw)        # (B, L, d)

        # ── Modality dropout ──
        h_audio, h_bio = self._apply_modality_dropout(h_audio, h_bio)

        # ── Add positional + type embeddings ──
        pos = self.pos_embed(t_idx)
        h_audio = h_audio + pos + self.type_embed(torch.zeros_like(t_idx))
        h_bio = h_bio + pos + self.type_embed(torch.ones_like(t_idx))

        if self.use_speaker and speaker_ids is not None:
            spk = self.speaker_embed(speaker_ids).unsqueeze(1)
            h_audio = h_audio + spk
            h_bio = h_bio + spk

        # ── Cross-modal conditioning ──
        h_audio_cond, cross_attn = self.conditioner(
            h_audio, h_bio, return_attn=return_artifacts,
        )

        # ── Pool → summary ──
        z_audio = h_audio_cond.mean(dim=1)   # (B, d)
        z_bio = h_bio.mean(dim=1)            # (B, d)

        # ── Quality estimation ──
        q_audio = self.q_audio(z_audio)      # (B, 1)
        q_bio = self.q_bio(z_bio)            # (B, 1)

        # ── Quality-aware gate ──
        z_fused, gate_w = self.gate([z_audio, z_bio], [q_audio, q_bio])

        # ── Heads ──
        a_reg = self.head_A_reg(z_fused)     # (B, 1) ∈ [-1, 1]
        v_reg = self.head_V_reg(z_fused)     # (B, 1) ∈ [-1, 1]
        a_logit = self.head_A_bin(z_fused)   # (B, 1) raw logit
        v_logit = self.head_V_bin(z_fused)   # (B, 1) raw logit

        out = {
            "A_reg": a_reg, "V_reg": v_reg,
            "A_logit": a_logit, "V_logit": v_logit,
        }

        if return_artifacts:
            out["artifacts"] = {
                "z_audio": z_audio.detach(),
                "z_bio": z_bio.detach(),
                "z_fused": z_fused.detach(),
                "q_audio": q_audio.detach(),
                "q_bio": q_bio.detach(),
                "gate_weights": gate_w.detach(),
                "cross_attn": cross_attn,
            }
            if self.kd_proj is not None:
                out["artifacts"]["z_student_proj"] = self.kd_proj(z_audio).detach()

        return out

    def compute_loss(self, out, batch, mu_kd=0.0, z_teacher=None):
        """
        Compute uncertainty-weighted multi-task loss.

        Tasks: [A_reg, V_reg, A_bin, V_bin]
        σ_i learned automatically → no manual λ tuning.
        """
        a_reg = out["A_reg"].squeeze(-1)
        v_reg = out["V_reg"].squeeze(-1)
        a_logit = out["A_logit"].squeeze(-1)
        v_logit = out["V_logit"].squeeze(-1)

        # Continuous targets (normalized ∈ [-1, 1])
        a_true = batch["a_norm"]
        v_true = batch["v_norm"]

        # Binary targets
        a_bin = batch["label_A"]
        v_bin = batch["label_V"]

        # Per-task losses
        loss_A_reg = F.mse_loss(a_reg, a_true)
        loss_V_reg = F.mse_loss(v_reg, v_true)
        loss_A_bin = F.binary_cross_entropy_with_logits(a_logit, a_bin)
        loss_V_bin = F.binary_cross_entropy_with_logits(v_logit, v_bin)

        # Uncertainty-weighted combination
        total = self.uncertainty_loss([loss_A_reg, loss_V_reg, loss_A_bin, loss_V_bin])

        # Knowledge distillation
        if mu_kd > 0 and z_teacher is not None and self.kd_proj is not None:
            z_student = self.kd_proj(out["artifacts"]["z_audio"] if "artifacts" in out
                                     else self._get_z_audio(out))
            loss_kd = F.mse_loss(z_student, z_teacher)
            total = total + mu_kd * loss_kd

        return total, {
            "loss_A_reg": loss_A_reg.item(),
            "loss_V_reg": loss_V_reg.item(),
            "loss_A_bin": loss_A_bin.item(),
            "loss_V_bin": loss_V_bin.item(),
            "loss_total": total.item(),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "total_M": round(total / 1e6, 2),
                "trainable_M": round(trainable / 1e6, 2)}
