"""
Loss Functions
===============
Focal Loss: class imbalance를 자연스럽게 처리.
기존 CE + class_weight + label_smoothing 조합 대체.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss with optional label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    label_smoothing: soft target로 변환. 0.1이면 정답 0.9, 나머지 0.1/(C-1).
    MixUp 사용 시 targets_soft [B, C]를 직접 받을 수도 있음.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits
            targets: [B] class indices (hard) or [B, C] soft targets (from MixUp)
        """
        C = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Build soft targets
        if targets.dim() == 1:
            # Hard labels → soft with optional smoothing
            targets_soft = F.one_hot(targets, num_classes=C).float()
            if self.label_smoothing > 0:
                targets_soft = targets_soft * (1 - self.label_smoothing) + \
                               self.label_smoothing / C
        else:
            # Already soft (from MixUp)
            targets_soft = targets

        # Focal weight based on predicted probability of target distribution
        p_t = (probs * targets_soft).sum(dim=-1)  # [B]
        focal_weight = (1 - p_t) ** self.gamma     # [B]

        # Alpha weighting
        if self.alpha is not None:
            if targets.dim() == 1:
                alpha_t = self.alpha[targets]
            else:
                # Soft targets: weighted average of alpha by target distribution
                alpha_t = (targets_soft * self.alpha.unsqueeze(0)).sum(dim=-1)
            focal_weight = alpha_t * focal_weight

        # Cross-entropy with soft targets
        loss = -(targets_soft * log_probs).sum(dim=-1)  # [B]
        loss = focal_weight * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(csv_path: str, label2id: Dict[str, int]) -> torch.Tensor:
    """
    Inverse-frequency class weights, normalized to mean=1.
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    maxn = counts.max()
    weights = []
    for lb in sorted(label2id.keys(), key=lambda x: label2id[x]):
        n = counts.get(lb, 1)
        weights.append(maxn / n)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w * (len(w) / w.sum())  # normalize mean=1
    return w
