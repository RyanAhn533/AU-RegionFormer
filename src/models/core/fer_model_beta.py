"""
AU-RegionFormer + Beta uncertainty gating (subclass, non-destructive).

Adds optional gates at:
  L2 (token level)   — per-AU-token Beta gate (default ON when use_beta=True)
  L3 (stream level)  — Global-token vs AU-mean Beta gate (optional)

L1 (per-AU-intensity) is upstream of token formation and not handled here;
apply L1 to AU intensity vectors before they enter feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fer_model import AUFERModel
from .beta_gate import TokenBetaGate, StreamBetaGate


class AUFERModelBeta(AUFERModel):
    """
    Drop-in replacement for AUFERModel that adds optional Beta gating.

    Args (in addition to AUFERModel):
        use_l2: enable per-AU-token gate
        l2_mode: "scale" | "softmax"
        use_l3: enable Global-vs-AU stream gate
    """

    def __init__(self,
                 use_l2: bool = True,
                 l2_mode: str = "scale",
                 use_l3: bool = False,
                 tau_init: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_l2 = use_l2
        self.use_l3 = use_l3

        d_emb = self.proj[0].out_features if isinstance(self.proj, nn.Sequential) else None
        if d_emb is None:
            d_emb = kwargs.get("d_emb", 384)

        if self.use_l2:
            self.l2_gate = TokenBetaGate(d_emb=d_emb, mode=l2_mode, tau=tau_init)
        if self.use_l3:
            self.l3_gate = StreamBetaGate(d_emb=d_emb, n_streams=2, tau=tau_init)

        # Cache for last reliability (for logging / analysis)
        self._last_l2_r = None
        self._last_l3_r = None

    def set_tau(self, tau: float):
        if self.use_l2:
            self.l2_gate.set_tau(tau)
        if self.use_l3:
            self.l3_gate.set_tau(tau)

    def forward(self, images: torch.Tensor, au_coords: torch.Tensor,
                return_features: bool = False,
                mixup_lambda: float = None, mixup_index: torch.Tensor = None):
        B = images.shape[0]

        feat_map, global_feat = self.backbone(images)
        feat_map_proj = self.feat_proj(feat_map)
        global_feat_proj = self.proj(global_feat)

        if mixup_lambda is not None and mixup_index is not None:
            lam = mixup_lambda
            feat_map_proj = lam * feat_map_proj + (1 - lam) * feat_map_proj[mixup_index]
            global_feat_proj = lam * global_feat_proj + (1 - lam) * global_feat_proj[mixup_index]
            au_coords = lam * au_coords + (1 - lam) * au_coords[mixup_index]
            au_coords = au_coords.clamp(0, self.img_size - 1)

        au_tokens = self.au_roi(feat_map_proj, au_coords, self.img_size)

        # ── L2: per-AU-token Beta gate ──
        if self.use_l2:
            au_tokens, r_l2, _v_l2 = self.l2_gate(au_tokens)
            self._last_l2_r = r_l2.detach()

        # ── L3: Global vs AU-mean stream gate (optional) ──
        if self.use_l3:
            au_mean = au_tokens.mean(dim=1)                        # [B, D]
            blended, r_l3 = self.l3_gate(global_feat_proj, au_mean)
            self._last_l3_r = r_l3.detach()
            global_feat_proj = blended

        cls = self.cls_token.expand(B, -1)
        tokens = self.fusion(cls, global_feat_proj, au_tokens)
        logits = self.head(tokens)

        if return_features:
            return logits, global_feat_proj
        return logits

    def get_last_reliability(self):
        """Return cached reliability tensors from last forward (for analysis)."""
        return {
            "l2_per_token": self._last_l2_r,   # [B, K] or None
            "l3_per_stream": self._last_l3_r,  # [B, 2] or None
        }
