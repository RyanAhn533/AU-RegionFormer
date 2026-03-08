# AU-RegionFormer Experiment Notes

## Training Protocol

### Two-Phase Learning Strategy

| Phase | Epochs | Backbone | Head LR | Backbone LR | Purpose |
|-------|--------|----------|---------|-------------|---------|
| 1 | 1-3 | Frozen | 3e-4 | 0 | Stabilize new layers |
| 2 | 4-200 | Unfrozen | 3e-4 | 3e-5 (10x lower) | Full finetuning |

### Loss Configuration

- **Focal Loss** (gamma=2.0): Handles class imbalance inherent in FER datasets
- **Class Weighting**: Inverse frequency, normalized to mean=1
- **No label smoothing**: Conflicts with focal loss, both try to reduce overconfidence

### Backbone Comparison

| Backbone | Params | Expected Acc | Speed | Memory |
|----------|--------|-------------|-------|--------|
| MobileViTv2_100 | 5M | ~78-82% | Fast | 2.1 GB |
| MobileViTv2_150 | 7M | ~80-84% | Medium | 3.0 GB |
| MobileViTv2_200 | 10M | ~82-87% | Slow | 4.5 GB |

### Fusion Layer Ablation

| n_layers | n_heads | Complexity | Notes |
|----------|---------|-----------|-------|
| 1 | 8 | Baseline | Sufficient for most FER datasets |
| 2 | 8 | +15% compute | Marginal improvement |
| 3 | 12 | +40% compute | Risk of overfitting on small datasets |

## Auto-Generated Artifacts

Every training run produces in `paper_artifacts/`:

1. **metrics_summary.json** — All numeric metrics
2. **classification_report.txt** — Per-class P/R/F1
3. **confusion_matrix.png** — Raw + normalized
4. **per_class_f1_bar.png** — F1 distribution
5. **roc_curves.png** — Per-class ROC with AUC
6. **pr_curves.png** — Precision-Recall curves
7. **training_curves.png** — 4-panel: loss, acc, F1, LR
8. **predictions.csv** — Full predictions with probabilities
9. **misclassified_samples.csv** — Failure analysis
10. **model_summary.txt** — Parameter count, FLOPs
11. **attention_gate_values.json** — Fusion gate statistics
12. **latency_benchmark.json** — CPU/GPU inference speed
13. **au_attention_heatmap.png** — AU importance per emotion

## Key Observations

### AU Region Importance per Emotion

- **Happy**: Cheek regions dominant (smile muscles)
- **Angry/Disgust**: Forehead + Nose regions
- **Fear/Surprise**: Eye regions dominant
- **Sad**: Global feature more important than individual AUs
- **Neutral**: Low magnitude across all regions (by design)
