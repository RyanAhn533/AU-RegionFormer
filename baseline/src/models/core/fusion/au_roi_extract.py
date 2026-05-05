"""
AU RoI Extraction from Feature Map
===================================

핵심 개선점:
  기존: K개 AU patch를 각각 224→128 crop → 각각 backbone forward (321회)
  신규: 전체 이미지 1회 backbone forward → feature map에서 AU 좌표로 sampling (1회)

AU 좌표는 CSV에서 (cx, cy) 픽셀 좌표로 제공됨.
이를 feature map 좌표로 변환 후 bilinear sampling 또는 RoI Align으로 추출.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AURoIExtractor(nn.Module):
    """
    Feature map에서 AU landmark 위치의 feature를 추출.

    Mode:
      - "bilinear": 좌표를 grid_sample로 bilinear interpolation (가장 빠름)
      - "roi_align": torchvision.ops.roi_align 사용 (spatial context 더 넓음)
    """

    def __init__(self, feat_dim: int, num_au: int = 6,
                 mode: str = "bilinear", roi_spatial: int = 1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_au = num_au
        self.mode = mode
        self.roi_spatial = roi_spatial

        # AU positional embedding: 각 AU region에 고유 ID 부여
        self.au_embed = nn.Embedding(num_au, feat_dim)

        # roi_spatial > 1이면 pooling 후 projection 필요
        if roi_spatial > 1:
            self.roi_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(feat_dim, feat_dim),
            )

    def forward(self, feat_map: torch.Tensor, au_coords: torch.Tensor,
                img_size: int = 224) -> torch.Tensor:
        """
        Args:
            feat_map:  [B, C, h, w] backbone feature map
            au_coords: [B, K, 2] AU center coordinates in PIXEL space (img_size 기준)
                       Each row is (cx, cy) for one AU region
            img_size:  original image size (assumes square, or pass the actual size)

        Returns:
            au_tokens: [B, K, C] AU feature tokens with positional embedding
        """
        B, C, h, w = feat_map.shape
        K = au_coords.shape[1]

        if self.mode == "bilinear":
            return self._bilinear_sample(feat_map, au_coords, img_size)
        else:
            return self._roi_align_sample(feat_map, au_coords, img_size)

    def _bilinear_sample(self, feat_map, au_coords, img_size):
        """
        grid_sample로 feature map에서 AU 위치의 feature 추출.
        roi_spatial > 1이면 AU 중심 주변 NxN 그리드를 샘플링하여 spatial context 확보.
        """
        B, C, h, w = feat_map.shape
        K = au_coords.shape[1]
        S = self.roi_spatial

        if S > 1:
            # AU 중심 주변 SxS 그리드 샘플링
            # feature map 기준 1 pixel = img_size/h pixel in image space
            pixel_stride = img_size / h  # ~32 pixels per feat cell
            offsets = torch.linspace(-(S // 2), S // 2, S, device=feat_map.device) * pixel_stride * 0.5

            # Build SxS grid offsets
            oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")  # [S, S]
            ox = ox.reshape(-1)  # [S*S]
            oy = oy.reshape(-1)  # [S*S]

            # Expand AU coords to SxS points per AU
            cx = au_coords[:, :, 0].unsqueeze(-1) + ox.unsqueeze(0).unsqueeze(0)  # [B, K, S*S]
            cy = au_coords[:, :, 1].unsqueeze(-1) + oy.unsqueeze(0).unsqueeze(0)  # [B, K, S*S]

            # Clamp to valid range
            cx = cx.clamp(0, img_size - 1)
            cy = cy.clamp(0, img_size - 1)

            # Normalize to [-1, 1]
            grid_x = (cx / img_size) * 2 - 1  # [B, K, S*S]
            grid_y = (cy / img_size) * 2 - 1

            # Reshape for grid_sample: [B, 1, K*S*S, 2]
            grid_x = grid_x.reshape(B, 1, K * S * S)
            grid_y = grid_y.reshape(B, 1, K * S * S)
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, 1, K*S*S, 2]

            sampled = F.grid_sample(
                feat_map, grid,
                mode="bilinear", padding_mode="border", align_corners=False
            )  # [B, C, 1, K*S*S]
            sampled = sampled.squeeze(2)  # [B, C, K*S*S]
            sampled = sampled.reshape(B, C, K, S * S)  # [B, C, K, S*S]
            au_tokens = sampled.mean(dim=-1).permute(0, 2, 1)  # [B, K, C] avg pool
        else:
            # Original single-point sampling
            grid_x = (au_coords[:, :, 0] / img_size) * 2 - 1  # [B, K]
            grid_y = (au_coords[:, :, 1] / img_size) * 2 - 1  # [B, K]

            grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, K, 2]
            grid = grid.unsqueeze(1)  # [B, 1, K, 2]

            sampled = F.grid_sample(
                feat_map, grid,
                mode="bilinear", padding_mode="border", align_corners=False
            )
            au_tokens = sampled.squeeze(2).permute(0, 2, 1)  # [B, K, C]

        # Add AU positional embedding
        au_ids = torch.arange(K, device=au_tokens.device).unsqueeze(0).expand(B, -1)
        au_tokens = au_tokens + self.au_embed(au_ids)

        return au_tokens

    def _roi_align_sample(self, feat_map, au_coords, img_size):
        """
        RoI Align으로 AU 주변 영역의 feature 추출.
        torchvision 필요.
        """
        from torchvision.ops import roi_align

        B, C, h, w = feat_map.shape
        K = au_coords.shape[1]
        spatial = self.roi_spatial

        # AU center → ROI box (center ± half_size in feature map coords)
        scale = h / img_size
        half = spatial / 2.0

        # Build ROI boxes: [N, 5] where each row is (batch_idx, x1, y1, x2, y2)
        boxes = []
        for b in range(B):
            for k in range(K):
                cx = au_coords[b, k, 0].item() * scale
                cy = au_coords[b, k, 1].item() * scale
                boxes.append([b, cx - half, cy - half, cx + half, cy + half])

        boxes = torch.tensor(boxes, device=feat_map.device, dtype=feat_map.dtype)

        # RoI Align: output [B*K, C, spatial, spatial]
        roi_feats = roi_align(feat_map, boxes, output_size=spatial,
                              spatial_scale=1.0, aligned=True)

        if spatial > 1:
            au_tokens = self.roi_proj(roi_feats)  # [B*K, C]
        else:
            au_tokens = roi_feats.flatten(2).squeeze(-1)  # [B*K, C]

        au_tokens = au_tokens.view(B, K, -1)  # [B, K, C]

        # Add AU positional embedding
        au_ids = torch.arange(K, device=au_tokens.device).unsqueeze(0).expand(B, -1)
        au_tokens = au_tokens + self.au_embed(au_ids)

        return au_tokens
