"""FER Classification Head v3 - CLS + Global + AU pooling for full token utilization."""

import torch
import torch.nn as nn


class FERHead(nn.Module):
    """
    v3: CLS + Global + AU mean/max pooling → 4d input.
    AU tokens를 버리던 v2 대비, fusion에서 학습된 AU 정보를 head까지 전달.
    use_au_pool=False이면 v2와 동일 (backward compatible).
    """

    def __init__(self, d_model: int = 384, num_classes: int = 7,
                 dropout: float = 0.2, use_au_pool: bool = False):
        super().__init__()
        self.use_au_pool = use_au_pool
        input_dim = d_model * 4 if use_au_pool else d_model * 2
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, K+2, d] fused tokens (CLS at 0, Global at 1, AU at 2:)
        Returns:
            logits: [B, num_classes]
        """
        cls_token = tokens[:, 0, :]     # [B, d]
        global_token = tokens[:, 1, :]  # [B, d]

        if self.use_au_pool:
            au_tokens = tokens[:, 2:, :]  # [B, K, d]
            au_mean = au_tokens.mean(dim=1)  # [B, d]
            au_max = au_tokens.max(dim=1).values  # [B, d]
            combined = torch.cat([cls_token, global_token, au_mean, au_max], dim=-1)
        else:
            combined = torch.cat([cls_token, global_token], dim=-1)

        return self.head(self.norm(combined))
