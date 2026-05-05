# Exp 004 — AU + Region Joint (clean subset)

n = 30000, fold=3, same seed, clean (is_selected=0) + valid AU subset

| Variant | Dim | Accuracy | Macro F1 |
|---------|-----|----------|----------|
| region_only | 8192 | **84.15%** ± 0.42 | 0.841 |
| au_only | 41 | **61.85%** ± 0.37 | 0.617 |
| region+au | 8233 | **84.21%** ± 0.42 | 0.841 |

## Δ Joint vs Region only: **+0.06%p**
→ marginal 증가 — AU 정보가 대부분 region에 흡수됨