# Phase 0.1c — Linear Probe Results (명확한 수치)

**Setup**: L2 norm → PCA(256) → Standardize → LogReg 3-fold CV, 30K stratified samples

| Config | Linear probe acc | Macro F1 | K-means ARI |
|--------|-----------------|----------|-------------|
| ConvNeXt-base-22k | POOLED-8region | **81.5%** ± 0.3 | 0.815 | 0.062 |
| MobileViT-v2-150 | POOLED-8region | **80.2%** ± 0.2 | 0.801 | 0.080 |
| ConvNeXt-base-22k | mouth | **75.2%** ± 0.2 | 0.749 | 0.082 |
| ConvNeXt-base-22k | nose | **75.0%** ± 0.3 | 0.748 | 0.070 |
| MobileViT-v2-150 | mouth | **71.9%** ± 0.3 | 0.716 | 0.184 |
| MobileViT-v2-150 | nose | **68.7%** ± 0.3 | 0.685 | 0.076 |
| ConvNeXt-base-22k | cheek_left | **61.9%** ± 0.6 | 0.615 | 0.023 |
| ConvNeXt-base-22k | eyes_right | **61.7%** ± 0.1 | 0.615 | 0.019 |
| ConvNeXt-base-22k | eyes_left | **61.5%** ± 0.3 | 0.613 | 0.018 |
| ConvNeXt-base-22k | cheek_right | **61.4%** ± 0.3 | 0.610 | 0.024 |
| MobileViT-v2-150 | cheek_right | **59.6%** ± 0.1 | 0.592 | 0.013 |
| MobileViT-v2-150 | eyes_right | **58.2%** ± 0.2 | 0.581 | 0.025 |
| MobileViT-v2-150 | eyes_left | **57.7%** ± 0.3 | 0.575 | 0.040 |
| MobileViT-v2-150 | cheek_left | **57.4%** ± 0.3 | 0.569 | 0.010 |
| ConvNeXt-base-22k | chin | **56.7%** ± 0.5 | 0.563 | 0.002 |
| MobileViT-v2-150 | chin | **54.4%** ± 0.7 | 0.541 | 0.005 |
| ConvNeXt-base-22k | forehead | **45.0%** ± 0.3 | 0.451 | 0.000 |
| MobileViT-v2-150 | forehead | **39.0%** ± 0.2 | 0.390 | 0.002 |
| [Baseline] Random uniform | **25.0%** ± 0.0 | 0.250 | 0.000 |

## 해석
- **Random baseline = 25%** (4-class uniform)
- Acc > 35% → embedding에 class 정보 있음
- Acc > 50% → 충분히 class-discriminative
- Acc ~ 25% → 진짜 class 분리 안 됨