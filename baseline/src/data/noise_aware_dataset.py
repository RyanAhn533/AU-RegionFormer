"""
Noise-Aware Dataset
====================
soft label + sample weight를 포함한 AU FER Dataset.
index_train_full_quality.csv를 읽어서 soft_* 컬럼과 final_quality를 반환.
"""

import random
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF


class NoiseAwareDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 label2id: Dict[str, int],
                 region_prefixes: List[str],
                 emotion_order: List[str],
                 img_size: int = 224,
                 mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 is_train: bool = True):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.label2id = label2id
        self.region_prefixes = region_prefixes
        self.emotion_order = emotion_order
        self.img_size = img_size
        self.is_train = is_train
        self.mean = mean
        self.std = std

        # Soft label columns
        self.soft_cols = [f"soft_{e}" for e in emotion_order]

        # AU coordinate columns
        self.cx_cols = [f"{p}_cx" for p in region_prefixes]
        self.cy_cols = [f"{p}_cy" for p in region_prefixes]

        # Swap pairs for horizontal flip
        self.swap_pairs = self._find_swap_pairs()

    def _find_swap_pairs(self):
        pairs = []
        names = self.region_prefixes
        used = set()
        for i, name_i in enumerate(names):
            if i in used:
                continue
            if "left" in name_i:
                partner = name_i.replace("left", "right")
                for j, name_j in enumerate(names):
                    if name_j == partner and j not in used:
                        pairs.append((i, j))
                        used.update([i, j])
                        break
        return pairs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = str(row["path"])
        label_name = str(row["label"])
        label_id = self.label2id[label_name]
        work_w = float(row["work_w"])
        work_h = float(row["work_h"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.fromarray(
                np.random.randint(0, 255, (int(work_h), int(work_w), 3), dtype=np.uint8)
            )

        # AU coordinates
        cxs = np.array([float(row[c]) for c in self.cx_cols], dtype=np.float32)
        cys = np.array([float(row[c]) for c in self.cy_cols], dtype=np.float32)
        cxs = cxs * (self.img_size / work_w)
        cys = cys * (self.img_size / work_h)
        cxs = np.clip(cxs, 0, self.img_size - 1)
        cys = np.clip(cys, 0, self.img_size - 1)

        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        if self.is_train:
            img, cxs, cys = self._augment(img, cxs, cys)

        img_tensor = TF.to_tensor(img)
        mean_t = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
        img_tensor = (img_tensor - mean_t) / std_t

        au_coords = torch.tensor(np.stack([cxs, cys], axis=-1), dtype=torch.float32)

        # Soft label
        soft_label = torch.tensor(
            [float(row[c]) for c in self.soft_cols], dtype=torch.float32
        )

        # Sample weight from final_quality
        sample_weight = torch.tensor(
            float(row.get("final_quality", 1.0)), dtype=torch.float32
        )

        return {
            "image": img_tensor,
            "au_coords": au_coords,
            "label": torch.tensor(label_id, dtype=torch.long),
            "soft_label": soft_label,
            "sample_weight": sample_weight,
            "path": img_path,
        }

    def _augment(self, img, cxs, cys):
        if random.random() < 0.5:
            img = TF.hflip(img)
            cxs = self.img_size - cxs
            for i, j in self.swap_pairs:
                cxs[i], cxs[j] = cxs[j], cxs[i]
                cys[i], cys[j] = cys[j], cys[i]

        if random.random() < 0.5:
            angle = (random.random() - 0.5) * 30
            img = TF.rotate(img, angle)
            rad = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            cx_c = self.img_size / 2.0
            cy_c = self.img_size / 2.0
            dx, dy = cxs - cx_c, cys - cy_c
            cxs = cos_a * dx - sin_a * dy + cx_c
            cys = sin_a * dx + cos_a * dy + cy_c
            cxs = np.clip(cxs, 0, self.img_size - 1)
            cys = np.clip(cys, 0, self.img_size - 1)

        img = TF.adjust_brightness(img, 1 + (random.random() - 0.5) * 0.4)
        img = TF.adjust_contrast(img, 1 + (random.random() - 0.5) * 0.4)
        img = TF.adjust_saturation(img, 1 + (random.random() - 0.5) * 0.3)
        img = TF.adjust_hue(img, (random.random() - 0.5) * 0.06)

        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        return img, cxs, cys
