"""
AU-RegionFormer V2: Patch-based architecture with multi-axis Beta gating.

Restored from the original AU-RegionFormer design (independent per-AU patch encoding)
plus three relationship axes:
  - Stage A : AU↔AU self-attention (co-activation patterns)
  - Stage B : Dual Beta gate (isolated + relational)
  - Stage C : Cross-attention fusion (CLS attends to global + gated AU tokens)

Forward inputs:
  image       : [B, 3, H, W]   full face (224×224)
  au_patches  : [B, K, 3, P, P]  per-AU cropped patches (96×96 default)

K = number of AU regions (8 for current data).
"""
from typing import Optional

import torch
import torch.nn as nn
import timm

from .relational_beta_gate import DualBetaGate
from .fusion import CrossAttentionFusion
from .heads import FERHead, ExpressionMagnitudeScorer


def _build_timm_encoder(name: str, pretrained: bool, img_size: int = 224):
    """timm wrapper that auto-handles DINOv2 / ViT models needing img_size override."""
    kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="avg")
    if "dinov2" in name or "patch14" in name:
        kwargs["img_size"] = img_size
        kwargs["dynamic_img_size"] = True
        # DINOv2 pretrained ckpt has 'norm.*' but timm with global_pool='avg'
        # creates 'fc_norm.*' → key mismatch. Use 'token' pool (CLS token) instead.
        kwargs["global_pool"] = "token"
    return timm.create_model(name, **kwargs)


def _get_feat_dim(model: nn.Module, dummy_size: int = 224) -> int:
    model.eval()
    with torch.no_grad():
        f = model(torch.zeros(1, 3, dummy_size, dummy_size))
    return f.shape[-1]


class AURegionFormerV2(nn.Module):

    def __init__(self,
                 # encoders
                 global_encoder: str = "mobilenetv4_conv_medium",
                 patch_encoder: str = "mobilenetv4_conv_small",
                 pretrained: bool = True,
                 num_au: int = 10,
                 num_classes: int = 4,
                 d_emb: int = 384,
                 patch_size: int = 96,
                 img_size: int = 224,

                 # Stage A
                 use_stage_a: bool = True,
                 stage_a_layers: int = 1,
                 stage_a_heads: int = 8,

                 # Stage B
                 use_stage_b: bool = True,
                 use_iso: bool = True,
                 use_rel: bool = True,
                 combine: str = "weighted",
                 tau_init: float = 2.0,

                 # Stage C
                 n_fusion_layers: int = 1,
                 n_fusion_heads: int = 8,
                 dropout: float = 0.1,
                 head_dropout: float = 0.2,
                 gate_init: float = 0.0,
                 drop_path: float = 0.0,
                 use_au_pool: bool = False):
        super().__init__()
        self.num_au = num_au
        self.patch_size = patch_size
        self.img_size = img_size
        self.use_stage_a = use_stage_a
        self.use_stage_b = use_stage_b

        # ── Encoders ──
        self.global_enc = _build_timm_encoder(global_encoder, pretrained, img_size=img_size)
        self.patch_enc  = _build_timm_encoder(patch_encoder,  pretrained, img_size=patch_size)
        global_feat_dim = _get_feat_dim(self.global_enc, img_size)
        patch_feat_dim  = _get_feat_dim(self.patch_enc, patch_size)

        # Normalization stats from global encoder
        global_cfg = timm.data.resolve_model_data_config(self.global_enc)
        self.norm_mean = global_cfg.get("mean", (0.485, 0.456, 0.406))
        self.norm_std = global_cfg.get("std", (0.229, 0.224, 0.225))

        # Patch encoder norm stats (may differ; expose for dataset use)
        patch_cfg = timm.data.resolve_model_data_config(self.patch_enc)
        self.patch_norm_mean = patch_cfg.get("mean", (0.485, 0.456, 0.406))
        self.patch_norm_std = patch_cfg.get("std", (0.229, 0.224, 0.225))

        # Project to d_emb
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, d_emb),
            nn.LayerNorm(d_emb),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_feat_dim, d_emb),
            nn.LayerNorm(d_emb),
        )

        # ── Stage A : AU-AU self-attention (optional) ──
        if use_stage_a:
            au_self_layers = []
            for _ in range(stage_a_layers):
                au_self_layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=d_emb, nhead=stage_a_heads,
                        dim_feedforward=d_emb * 4, dropout=dropout,
                        activation="gelu", batch_first=True, norm_first=True,
                    )
                )
            self.stage_a = nn.Sequential(*au_self_layers)

        # ── Stage B : Dual Beta gate (optional) ──
        if use_stage_b:
            self.stage_b = DualBetaGate(
                d_emb=d_emb,
                use_iso=use_iso,
                use_rel=use_rel,
                combine=combine,
                tau=tau_init,
            )

        # ── CLS token + Stage C fusion ──
        self.cls_token = nn.Parameter(torch.zeros(1, d_emb))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # JEPA mask token (Stage 7 only — created lazily, ignored if jepa_weight=0 in trainer)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_emb))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.fusion = CrossAttentionFusion(
            d_model=d_emb, n_heads=n_fusion_heads,
            n_layers=n_fusion_layers, dropout=dropout,
            gate_init=gate_init, drop_path=drop_path,
        )

        # ── Head ──
        self.head = FERHead(d_model=d_emb, num_classes=num_classes,
                            dropout=head_dropout, use_au_pool=use_au_pool)
        self.expr_scorer = ExpressionMagnitudeScorer(d_model=d_emb)

        # Cache for analysis
        self._last_r = None
        self._last_aux = None

    # ── External tau control (cosine anneal hook) ──
    def set_tau(self, tau: float):
        if self.use_stage_b:
            self.stage_b.set_tau(tau)

    # ── freeze/unfreeze encoders together (for trainer phase 1) ──
    def freeze_backbone(self):
        for p in self.global_enc.parameters():
            p.requires_grad = False
        for p in self.patch_enc.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.global_enc.parameters():
            p.requires_grad = True
        for p in self.patch_enc.parameters():
            p.requires_grad = True

    def get_param_groups(self, base_lr: float, backbone_lr_scale: float = 0.1):
        backbone_ids = set()
        for p in self.global_enc.parameters():
            backbone_ids.add(id(p))
        for p in self.patch_enc.parameters():
            backbone_ids.add(id(p))
        groups = [
            {"params": [p for p in self.global_enc.parameters() if p.requires_grad]
                       + [p for p in self.patch_enc.parameters() if p.requires_grad],
             "lr": base_lr * backbone_lr_scale, "name": "backbone"},
            {"params": [p for p in self.parameters()
                        if p.requires_grad and id(p) not in backbone_ids],
             "lr": base_lr, "name": "head"},
        ]
        return [g for g in groups if len(g["params"]) > 0]

    # ── Forward ──
    def forward(self,
                image: torch.Tensor,
                au_patches: torch.Tensor,
                return_features: bool = False,
                mixup_lambda: Optional[float] = None,
                mixup_index: Optional[torch.Tensor] = None):
        """
        image      : [B, 3, H, W]
        au_patches : [B, K, 3, P, P]
        """
        B, K = au_patches.shape[:2]

        # Encode full face → global token
        gf = self.global_enc(image)                 # [B, Cg]
        global_feat = self.global_proj(gf)          # [B, D]

        # Encode AU patches (weight-shared single forward)
        patches = au_patches.view(B * K, *au_patches.shape[2:])  # [B*K, 3, P, P]
        pf = self.patch_enc(patches)                              # [B*K, Cp]
        au_tokens = self.patch_proj(pf).view(B, K, -1)            # [B, K, D]

        # Manifold MixUp (in feature space)
        if mixup_lambda is not None and mixup_index is not None:
            lam = mixup_lambda
            global_feat = lam * global_feat + (1 - lam) * global_feat[mixup_index]
            au_tokens = lam * au_tokens + (1 - lam) * au_tokens[mixup_index]

        # ── Stage A : AU-AU self-attention ──
        if self.use_stage_a:
            au_tokens = self.stage_a(au_tokens)     # [B, K, D]

        # ── Stage B : Dual Beta gate ──
        if self.use_stage_b:
            au_tokens, r, aux = self.stage_b(global_feat, au_tokens)
            self._last_r = r.detach()
            self._last_aux = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in aux.items()}

        # ── Stage C : Cross-attention fusion ──
        cls = self.cls_token.expand(B, -1)
        tokens = self.fusion(cls, global_feat, au_tokens)        # [B, K+2, D]
        logits = self.head(tokens)

        if return_features:
            return logits, global_feat
        return logits

    def get_last_reliability(self):
        return {"r_combined": self._last_r, "aux": self._last_aux}

    def jepa_loss(self, image: torch.Tensor, au_patches: torch.Tensor,
                  n_mask: int = 1) -> torch.Tensor:
        """Self-supervised JEPA-style auxiliary loss.

        Mask `n_mask` random AU tokens with a learnable mask token, run Stage A
        as predictor, compare predicted at mask positions to (stop-gradient) target.
        Returns scalar 1 - cos_sim averaged over masked positions.
        Requires use_stage_a=True; returns 0 if Stage A not present.
        """
        if not self.use_stage_a:
            return torch.zeros((), device=image.device)
        B, K = au_patches.shape[:2]
        # Encode AU patches (target view)
        flat = au_patches.view(B * K, *au_patches.shape[2:])
        pf = self.patch_enc(flat)
        target = self.patch_proj(pf).view(B, K, -1).detach()       # [B, K, D] no grad

        # Build masked input
        masked = target.clone().requires_grad_(False) * 0.0 + target  # detached copy w/ grad path via mask_token
        # But we need gradients to flow through mask_token + Stage A; replace selected positions with mask_token
        mask_tok = self.mask_token.expand(B, K, -1)
        idx = torch.stack([
            torch.randperm(K, device=image.device)[:n_mask] for _ in range(B)
        ], dim=0)                                                   # [B, n_mask]

        masked_input = target.clone()
        masked_input.requires_grad_(False)
        for b in range(B):
            for j in range(n_mask):
                masked_input[b, idx[b, j]] = mask_tok[b, idx[b, j]]

        # Predict via Stage A (re-runs the same self-attn used in forward)
        predicted = self.stage_a(masked_input)                      # [B, K, D]

        # Gather predictions at mask positions vs targets
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, predicted.shape[-1])
        pred_at = torch.gather(predicted, 1, gather_idx)             # [B, n_mask, D]
        targ_at = torch.gather(target,   1, gather_idx)              # [B, n_mask, D]

        cos = nn.functional.cosine_similarity(pred_at, targ_at, dim=-1)  # [B, n_mask]
        return (1.0 - cos).mean()
