"""
Audio Encoder — MobileNetV3 Student + AST Teacher
===================================================
V4-lite용 경량 audio encoder.

Student: MobileNetV3-Small (~2.5M params)
  - logmel (1, 64, T) → 이미지처럼 처리
  - AdaptiveAvgPool → L tokens
  - Jetson에서 ~5ms

Teacher: AST (Audio Spectrogram Transformer, ~87M params)
  - 학습 시에만 사용 (frozen)
  - Feature-level distillation: ||P(z_student) - z_teacher||²
  - 추론 시 제거
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoderStudent(nn.Module):
    """
    MobileNetV3-Small based audio encoder.

    Input:  (B, 1, 64, T_mel) logmel spectrogram
    Output: (B, L, d) audio token sequence
    """
    def __init__(
        self,
        d_model: int = 128,
        num_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model

        import torchvision.models as models
        # MobileNetV3-Small: lightweight, good for spectrograms
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Modify first conv for 1-channel input (logmel)
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            backbone.features[0][0].weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.features = backbone.features
        # MobileNetV3-Small feature output: 576 channels
        self.feat_dim = 576

        # Pool frequency → 1, keep time → L tokens
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse freq
        self.time_pool = nn.AdaptiveAvgPool1d(num_tokens)  # downsample time

        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logmel: (B, 1, 64, T_mel)
        Returns:
            (B, L, d) audio token sequence
        """
        # Feature extraction
        x = self.features(logmel)        # (B, 576, H', W')

        # Pool frequency axis, keep time
        x = self.freq_pool(x)           # (B, 576, 1, W')
        x = x.squeeze(2)                # (B, 576, W')

        # Pool time to L tokens
        x = self.time_pool(x)           # (B, 576, L)
        x = x.transpose(1, 2)           # (B, L, 576)

        # Project to d_model
        x = self.proj(x)                # (B, L, d)
        return x

    def get_summary(self, logmel: torch.Tensor) -> torch.Tensor:
        """Global audio summary for distillation. Returns (B, d)."""
        tokens = self.forward(logmel)    # (B, L, d)
        return tokens.mean(dim=1)        # (B, d)


class AudioTeacherAST(nn.Module):
    """
    AST Teacher wrapper (frozen, training-only).
    Provides teacher embeddings for knowledge distillation.

    Usage:
        teacher = AudioTeacherAST(d_teacher=768)
        teacher.load_from_v4_checkpoint("v4_best.pth")
        z_teacher = teacher(logmel)  # (B, d_teacher)
    """
    def __init__(
        self,
        ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        d_teacher: int = 768,
    ):
        super().__init__()
        self.d_teacher = d_teacher

        try:
            from transformers import ASTModel
            self.ast = ASTModel.from_pretrained(ast_model_name)
            self.available = True
        except (ImportError, OSError):
            print("[WARN] AST model not available. Distillation disabled.")
            self.ast = None
            self.available = False

        # Freeze all parameters
        if self.ast is not None:
            for p in self.ast.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logmel: (B, 1, 64, T_mel)
        Returns:
            (B, d_teacher) teacher audio embedding
        """
        if not self.available:
            B = logmel.size(0)
            return torch.zeros(B, self.d_teacher, device=logmel.device)

        # AST expects (B, T, 128) input
        x = F.interpolate(logmel, size=(128, 1024), mode="bilinear", align_corners=False)
        x = x.squeeze(1).transpose(1, 2)  # (B, 1024, 128)

        out = self.ast(input_values=x)
        # Use CLS token or mean pool
        h = out.last_hidden_state[:, 0, :]  # CLS token: (B, d_teacher)
        return h

    def load_from_v4_checkpoint(self, ckpt_path: str):
        """Load AST weights from V4-full trained checkpoint."""
        import os
        if not os.path.exists(ckpt_path):
            print(f"[WARN] V4 checkpoint not found: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # Extract AST weights from V4 model
        ast_state = {}
        prefix = "aud_enc.ast."
        for k, v in state.items():
            if k.startswith(prefix):
                ast_state[k[len(prefix):]] = v

        if ast_state and self.ast is not None:
            self.ast.load_state_dict(ast_state, strict=False)
            print(f"[INFO] Loaded AST weights from V4 checkpoint ({len(ast_state)} params)")


class DistillationProjector(nn.Module):
    """
    Student → Teacher dimension projector for KD loss.
    P: R^d_student → R^d_teacher
    """
    def __init__(self, d_student: int, d_teacher: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_student, d_teacher),
            nn.GELU(),
            nn.Linear(d_teacher, d_teacher),
        )

    def forward(self, z_student: torch.Tensor) -> torch.Tensor:
        return self.proj(z_student)
