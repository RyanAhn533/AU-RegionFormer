"""
AUFERDataset V2 — returns AU patches alongside the global face image.

Reuses build_label_mapping / find_region_prefixes / coord parsing from v1,
adds AU patch cropping using bbox columns (_wx1/_wy1/_wx2/_wy2). If only
center coords (_cx/_cy) are present, falls back to a fixed-size square crop
around each center.

Output per sample:
  image        : [3, H, H]   normalized full face (global encoder norm)
  au_patches   : [K, 3, P, P] normalized AU patches (patch encoder norm)
  au_coords    : [K, 2]      kept for backward compat / robustness eval
  label        : long
  path         : str
"""
from typing import Dict, List, Tuple

import math
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .dataset import build_label_mapping, find_region_prefixes  # reuse


class AUFERDatasetV2(Dataset):
    def __init__(self,
                 csv_path: str,
                 label2id: Dict[str, int],
                 region_prefixes: List[str],
                 img_size: int = 224,
                 patch_size: int = 96,
                 patch_pad_ratio: float = 0.0,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 patch_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 patch_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 is_train: bool = True,
                 fallback_box_frac: float = 0.20,
                 patch_dropout_n: int = 0,
                 patch_scale: float = 1.0,
                 patch_scale_per_region: Dict[str, float] = None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.region_prefixes = region_prefixes
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_pad_ratio = patch_pad_ratio
        self.fallback_box_frac = fallback_box_frac
        self.patch_dropout_n = patch_dropout_n
        self.is_train = is_train
        # Per-region scale: dict prefix→scale; falls back to global patch_scale
        self.patch_scale = float(patch_scale)
        self.patch_scale_per_region = patch_scale_per_region or {}

        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self._pmean = torch.tensor(patch_mean, dtype=torch.float32).view(3, 1, 1)
        self._pstd = torch.tensor(patch_std, dtype=torch.float32).view(3, 1, 1)

        # ── Coordinate column detection ──
        sample = region_prefixes[0]
        self.has_bbox = f"{sample}_wx1" in self.df.columns
        self.has_center = f"{sample}_cx" in self.df.columns
        if not (self.has_bbox or self.has_center):
            raise KeyError(f"CSV missing both bbox and center cols for {sample}")

        if self.has_bbox:
            self.wx1_cols = [f"{p}_wx1" for p in region_prefixes]
            self.wy1_cols = [f"{p}_wy1" for p in region_prefixes]
            self.wx2_cols = [f"{p}_wx2" for p in region_prefixes]
            self.wy2_cols = [f"{p}_wy2" for p in region_prefixes]
        if self.has_center:
            self.cx_cols = [f"{p}_cx" for p in region_prefixes]
            self.cy_cols = [f"{p}_cy" for p in region_prefixes]

        self.swap_pairs = self._find_swap_pairs()

    # ── helpers ──
    def _find_swap_pairs(self) -> List[Tuple[int, int]]:
        pairs, used = [], set()
        names = self.region_prefixes
        for i, ni in enumerate(names):
            if i in used:
                continue
            if "left" in ni:
                partner = ni.replace("left", "right")
                for j, nj in enumerate(names):
                    if nj == partner and j not in used:
                        pairs.append((i, j))
                        used.add(i); used.add(j)
                        break
        return pairs

    def __len__(self):
        return len(self.df)

    def _parse_boxes(self, row, work_w, work_h):
        """Return cxs, cys, wx1, wy1, wx2, wy2 (in img_size pixel coords)."""
        sx = self.img_size / work_w
        sy = self.img_size / work_h

        if self.has_bbox:
            wx1 = np.array([float(row[c]) for c in self.wx1_cols], dtype=np.float32) * sx
            wy1 = np.array([float(row[c]) for c in self.wy1_cols], dtype=np.float32) * sy
            wx2 = np.array([float(row[c]) for c in self.wx2_cols], dtype=np.float32) * sx
            wy2 = np.array([float(row[c]) for c in self.wy2_cols], dtype=np.float32) * sy
            cxs = (wx1 + wx2) / 2.0
            cys = (wy1 + wy2) / 2.0
        else:
            cxs = np.array([float(row[c]) for c in self.cx_cols], dtype=np.float32) * sx
            cys = np.array([float(row[c]) for c in self.cy_cols], dtype=np.float32) * sy
            half = self.fallback_box_frac * self.img_size / 2.0
            wx1 = cxs - half; wy1 = cys - half
            wx2 = cxs + half; wy2 = cys + half

        # Apply patch_scale (1.0 = unchanged; 1.5 = 1.5× bbox around center)
        scales = np.array([
            self.patch_scale_per_region.get(p, self.patch_scale)
            for p in self.region_prefixes
        ], dtype=np.float32)
        if not np.all(scales == 1.0):
            half_w = (wx2 - wx1) * 0.5 * scales
            half_h = (wy2 - wy1) * 0.5 * scales
            wx1 = cxs - half_w; wx2 = cxs + half_w
            wy1 = cys - half_h; wy2 = cys + half_h

        # Clamp
        cxs = np.clip(cxs, 0, self.img_size - 1)
        cys = np.clip(cys, 0, self.img_size - 1)
        return cxs, cys, wx1, wy1, wx2, wy2

    def _crop_patches(self, img: Image.Image, wx1, wy1, wx2, wy2) -> torch.Tensor:
        """Crop K patches from PIL image, resize to (P, P), return [K, 3, P, P]."""
        K = len(wx1)
        P = self.patch_size
        out = torch.empty(K, 3, P, P, dtype=torch.float32)
        for i in range(K):
            x1 = max(0, int(round(wx1[i])))
            y1 = max(0, int(round(wy1[i])))
            x2 = min(self.img_size, int(round(wx2[i])))
            y2 = min(self.img_size, int(round(wy2[i])))
            if x2 - x1 < 4 or y2 - y1 < 4:
                # Degenerate → fallback to center-square
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                half = max(8, P // 4)
                x1 = max(0, cx - half); y1 = max(0, cy - half)
                x2 = min(self.img_size, cx + half); y2 = min(self.img_size, cy + half)
            patch = img.crop((x1, y1, x2, y2)).resize((P, P), Image.BILINEAR)
            t = TF.to_tensor(patch)
            t = (t - self._pmean) / self._pstd
            out[i] = t
        return out

    def _augment(self, img, cxs, cys, boxes):
        wx1, wy1, wx2, wy2 = boxes
        # Horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            new_wx1 = self.img_size - wx2
            new_wx2 = self.img_size - wx1
            wx1, wx2 = new_wx1, new_wx2
            cxs = self.img_size - cxs
            for i, j in self.swap_pairs:
                cxs[i], cxs[j] = cxs[j], cxs[i]
                cys[i], cys[j] = cys[j], cys[i]
                wx1[i], wx1[j] = wx1[j], wx1[i]
                wx2[i], wx2[j] = wx2[j], wx2[i]
                wy1[i], wy1[j] = wy1[j], wy1[i]
                wy2[i], wy2[j] = wy2[j], wy2[i]

        # Random rotation ±15°
        if random.random() < 0.5:
            angle = (random.random() - 0.5) * 30
            img = TF.rotate(img, angle)
            rad = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            cc = self.img_size / 2.0
            def rot(xs, ys):
                dx = xs - cc; dy = ys - cc
                rx = cos_a * dx - sin_a * dy + cc
                ry = sin_a * dx + cos_a * dy + cc
                return np.clip(rx, 0, self.img_size - 1), np.clip(ry, 0, self.img_size - 1)
            cxs, cys = rot(cxs, cys)
            x1n, y1n = rot(wx1, wy1)
            x2n, y2n = rot(wx2, wy2)
            wx1 = np.minimum(x1n, x2n); wx2 = np.maximum(x1n, x2n)
            wy1 = np.minimum(y1n, y2n); wy2 = np.maximum(y1n, y2n)

        # Color jitter
        img = TF.adjust_brightness(img, 1 + (random.random() - 0.5) * 0.4)
        img = TF.adjust_contrast(img, 1 + (random.random() - 0.5) * 0.4)
        img = TF.adjust_saturation(img, 1 + (random.random() - 0.5) * 0.3)
        img = TF.adjust_hue(img, (random.random() - 0.5) * 0.06)
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        return img, cxs, cys, (wx1, wy1, wx2, wy2)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["path"])
        label_name = str(row["label"])
        label_id = self.label2id[label_name]
        work_w = float(row["work_w"]); work_h = float(row["work_h"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            import logging
            logging.getLogger("dataset_v2").warning(f"Failed to load {img_path}: {e}")
            img = Image.fromarray(np.zeros((int(work_h), int(work_w), 3), dtype=np.uint8))

        cxs, cys, wx1, wy1, wx2, wy2 = self._parse_boxes(row, work_w, work_h)
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        if self.is_train:
            img, cxs, cys, (wx1, wy1, wx2, wy2) = self._augment(img, cxs, cys, (wx1, wy1, wx2, wy2))

        # AU patches (cropped from already-resized + augmented img)
        au_patches = self._crop_patches(img, wx1, wy1, wx2, wy2)  # [K, 3, P, P]

        # Optional: random AU patch dropout (Stage 7' control — JEPA effect isolation)
        if self.is_train and getattr(self, "patch_dropout_n", 0) > 0:
            n = int(self.patch_dropout_n)
            K = au_patches.shape[0]
            if n < K:
                drop_idx = np.random.choice(K, size=n, replace=False)
                au_patches[drop_idx] = 0.0

        # Full image normalized for global encoder
        img_t = TF.to_tensor(img)
        img_t = (img_t - self._mean) / self._std

        au_coords = torch.tensor(np.stack([cxs, cys], axis=-1), dtype=torch.float32)

        # Optional quality_score column (Phase 6 master CSV) — default 1.0 if absent
        if "quality_score" in row.index:
            try:
                qs = float(row["quality_score"])
            except Exception:
                qs = 1.0
        else:
            qs = 1.0

        # Optional mean_is_selected (Yonsei reject probability) — default 0.0 (assume agreed)
        if "mean_is_selected" in row.index:
            try:
                mr = float(row["mean_is_selected"])
            except Exception:
                mr = 0.0
        else:
            mr = 0.0

        return {
            "image": img_t,
            "au_patches": au_patches,
            "au_coords": au_coords,
            "label": torch.tensor(label_id, dtype=torch.long),
            "quality_score": torch.tensor(qs, dtype=torch.float32),
            "mean_is_selected": torch.tensor(mr, dtype=torch.float32),
            "path": img_path,
        }


def collate_fn_v2(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "au_patches": torch.stack([b["au_patches"] for b in batch]),
        "au_coords": torch.stack([b["au_coords"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "quality_score": torch.stack([b.get("quality_score", torch.tensor(1.0)) for b in batch]),
        "mean_is_selected": torch.stack([b.get("mean_is_selected", torch.tensor(0.0)) for b in batch]),
        "path": [b["path"] for b in batch],
    }
