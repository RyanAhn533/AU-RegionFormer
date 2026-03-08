# AU-RegionFormer Architecture Details

## Core Architecture

### AUFERModel: Single-Backbone AU-Guided Transformer

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AUFERModel                                  │
│                                                                     │
│  Input: RGB Image (B, 3, 224, 224) + AU Coords (B, 6, 2)           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  MobileViTv2 Backbone (1x forward pass)                     │    │
│  │  ImageNet pretrained, ~5M params                            │    │
│  │                                                             │    │
│  │  Output: Feature Map [B, 384, 7, 7]                         │    │
│  │          Global Pool  [B, 384]                              │    │
│  └──────────────┬──────────────────────────┬───────────────────┘    │
│                 │                          │                        │
│                 ▼                          │                        │
│  ┌──────────────────────────┐              │                        │
│  │  AU RoI Extraction        │              │                        │
│  │                           │              │                        │
│  │  6 regions:               │              │                        │
│  │  ┌─────┐ ┌─────┐ ┌─────┐│              │                        │
│  │  │Fore-│ │Eye_L│ │Eye_R││              │                        │
│  │  │head │ │     │ │     ││              │                        │
│  │  └─────┘ └─────┘ └─────┘│              │                        │
│  │  ┌─────┐ ┌─────┐ ┌─────┐│              │                        │
│  │  │Nose │ │Chk_L│ │Chk_R││              │                        │
│  │  └─────┘ └─────┘ └─────┘│              │                        │
│  │                           │              │                        │
│  │  Method: bilinear grid    │              │                        │
│  │  sample on feature map    │              │                        │
│  │  Output: [B, 6, 384]     │              │                        │
│  └──────────────┬───────────┘              │                        │
│                 │                          │                        │
│                 ▼                          ▼                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  Token Assembly                                          │      │
│  │                                                          │      │
│  │  [CLS] + [Global] + [AU_1, AU_2, AU_3, AU_4, AU_5, AU_6]│      │
│  │   ↑        ↑         ↑ (+ learnable AU position embed)  │      │
│  │  learnable backbone   from RoI extraction                │      │
│  │                                                          │      │
│  │  Final: [B, 8, 384] token sequence                       │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  CrossAttentionFusion (1 layer)                           │      │
│  │                                                          │      │
│  │  1. Cross-Attention (Pre-Norm)                           │      │
│  │     Q: [CLS, Global]  →  K,V: [AU_1...AU_6]              │      │
│  │     8 heads, d_k = 384/8 = 48                            │      │
│  │                                                          │      │
│  │  2. Gated Residual                                       │      │
│  │     gate = sigmoid(param) ∈ R^384                        │      │
│  │     out = gate * cross_out + (1-gate) * input            │      │
│  │                                                          │      │
│  │  3. Self-Attention (Pre-Norm)                            │      │
│  │     All 8 tokens interact                                │      │
│  │                                                          │      │
│  │  4. FFN: 384 → 1536 → 384 (GELU)                        │      │
│  └──────────────────────────┬───────────────────────────────┘      │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  FER Head                                                 │      │
│  │                                                          │      │
│  │  CLS token [B, 384]                                      │      │
│  │    → LayerNorm → Linear(384→384) → GELU → Dropout(0.2)  │      │
│  │    → Linear(384→7) → Logits [B, 7]                      │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  Output: 7-class logits (angry, disgust, fear, happy,              │
│          neutral, sad, surprise)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Why Single-Backbone?

### Computational Comparison

| Method | Backbone Forwards | AU Processing | Total |
|--------|-------------------|---------------|-------|
| Multi-Patch (traditional) | 6x (one per region) | 6x crop | 6x cost |
| AU-RegionFormer | **1x** | grid_sample (free) | **1x cost** |

### Speedup Factor: 321x

Historical benchmarking showed that multi-patch preprocessing + multiple backbone forwards resulted in 321x overhead compared to our single-forward approach.

## Expression Magnitude Scorer

Peak frame selection without full attention computation:

```
Training: EMA update neutral centroid (momentum=0.99)
          centroid = 0.99 * centroid + 0.01 * mean(neutral_features)

Inference: magnitude = ||global_feat - neutral_centroid||_2
           peak_frame = argmax(magnitudes over 1-sec window)
```

## Multimodal Integration Pipeline

```
                FER (7-class)
                    │
                    ├──── + Arousal/Valence ──► EmotionRefiner (10-class)
                    │     (from audio/bio)      "happy" + high_A → "excited"
                    │
FaceMesh EAR ──────├──── + Arousal ──────────► DrowsinessJudge (3-level)
(30 fps)           │     (optional)             PERCLOS threshold based
                    │
                    └──► MultimodalFuser ──────► Final Output
                         orchestrates all
```
