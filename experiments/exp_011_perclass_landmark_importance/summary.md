# Exp 011 — Per-class F1 + Landmark importance

## 1. Triplet (Region+Landmark+Graph) per-class F1

- **Overall**: acc=87.36% ± 0.19  Macro F1=0.873

| Emotion | F1 |
|---------|-----|
| 기쁨 | **0.949** |
| 분노 | **0.840** |
| 슬픔 | **0.835** |
| 중립 | **0.869** |

## 2. Landmark single-feature importance (17 features, sorted by acc)
| Rank | Feature | Fisher | Single-probe acc | Macro F1 |
|-----|---------|--------|-----------------|----------|
| 1 | lip_corner_angle | 0.975 | 49.39% | 0.478 |
| 2 | mouth_width | 0.936 | 48.60% | 0.462 |
| 3 | mar | 0.411 | 42.76% | 0.367 |
| 4 | cheek_raise_avg | 0.348 | 39.45% | 0.348 |
| 5 | cheek_raise_right | 0.306 | 39.30% | 0.344 |
| 6 | ear_avg | 0.345 | 39.11% | 0.377 |
| 7 | cheek_raise_left | 0.302 | 38.88% | 0.342 |
| 8 | ear_right | 0.319 | 38.75% | 0.373 |
| 9 | ear_left | 0.321 | 38.63% | 0.371 |
| 10 | brow_height_left | 0.079 | 33.99% | 0.285 |
| 11 | brow_height_avg | 0.086 | 33.97% | 0.286 |
| 12 | brow_height_right | 0.076 | 33.46% | 0.281 |
| 13 | brow_furrow | 0.108 | 32.98% | 0.291 |
| 14 | forehead_height | 0.039 | 30.39% | 0.247 |
| 15 | face_aspect_ratio | 0.021 | 28.95% | 0.229 |
| 16 | nose_bridge | 0.006 | 27.19% | 0.201 |
| 17 | chin_length | 0.004 | 26.88% | 0.196 |