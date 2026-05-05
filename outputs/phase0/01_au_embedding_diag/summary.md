# Phase 0.1 — AU Embedding Diagnostic Summary
Generated: 2026-04-21T12:17:10.472991

## Judgment (pooled silhouette)
| Backbone | Silhouette (cos) | Inter-class cos sim | Judgment |
|---|---|---|---|
| mobilevitv2_150 | 0.0072 | 0.9726 | NO-GO — Backbone 교체 검토 |
| convnext_base.fb_in22k_ft_in1k | 0.0113 | 0.9676 | NO-GO — Backbone 교체 검토 |

## Per-region silhouette
| Region | mobilevitv2_150 | convnext_base.fb_in22k_ft_in1k |
|---|---|---|
| eyes_left | -0.0196 | -0.0197 |
| eyes_right | -0.0232 | -0.0210 |
| nose | -0.0059 | 0.0026 |
| mouth | 0.0094 | 0.0194 |
| forehead | -0.0085 | -0.0237 |
| chin | -0.0087 | -0.0147 |
| cheek_left | -0.0012 | 0.0037 |
| cheek_right | 0.0015 | 0.0037 |

## Inter-class cosine similarity (pooled)
목표 수치 확인: '0.97-0.99'가 실제로 맞는지

### mobilevitv2_150
- Mean off-diagonal cos sim: **0.9726**
- Min off-diagonal cos sim: **0.9517**
- Classes: ['기쁨', '분노', '슬픔', '중립']

### convnext_base.fb_in22k_ft_in1k
- Mean off-diagonal cos sim: **0.9676**
- Min off-diagonal cos sim: **0.9390**
- Classes: ['기쁨', '분노', '슬픔', '중립']

## Files
- `metrics.json` — all numeric results
- `umap_*.png` — 2D visualization per region/pooled
- `cossim_*.png` — inter-class cosine similarity heatmap
- `pca_variance.png` — explained variance curves