---
phase: 6
stage: 16
status: in_progress
launched: 2026-05-03 22:02 KST
expected_end: 2026-05-04 04:30 KST
purpose: Plan 1순위 — 5-variant ablation matrix (실제는 4 + Stage 13 + Stage 6 baseline = 6 cells)
plan_ref: /home/ajy/.claude/plans/woolly-sniffing-karp.md (action 4)
---

# Phase 6 Stage 16 — Beta Gate Ablation Matrix

## Hypothesis

3-axis Beta gate (Stage A self-attn + B-iso + B-rel)가 in-domain 4-class
Korean FER에서 효과 있는가? 변형 간 effect size > noise level (0.002) 인가?

## Variants

| Variant | Stage A | B-iso | B-rel | combine | F1 (TBD) |
|---|---|---|---|---|---|
| baseline (Stage 6 full) | ✓ | ✓ | ✓ | weighted | **0.9256** (이미) |
| -A (Stage 13) | ✗ | ✓ | ✓ | weighted | **0.9239** (이미) |
| 16a iso_only | ✓ | ✓ | ✗ | iso_only | **0.9252** (Δ-0.0004, noise) |
| 16b rel_only | ✓ | ✗ | ✓ | rel_only | **0.9267** (Δ+0.0011, noise. early-stop E25, best E17) |
| 16c a_only | ✓ | ✗ | ✗ | iso_only(degenerate) | **DROPPED** (model assertion: use_iso or use_rel must be True. Stage 13 no_stage_a로 ablation cell 대체) |
| 16d product | ✓ | ✓ | ✓ | product | TBD |

## Setup

- Base config: stage6_full.yaml
- 1 seed (seed=42)
- 30 epochs, batch 384, AdamW lr=8e-4 cosine warmup
- master_train.csv (203K Korean) → master_val.csv (50K)
- 4-class: angry/happy/neutral/sad

## Decision rules

- **Effect검출**: max-min Δ > 0.005 → 우리 method 효과 있음
- **All noise**: 모든 변형 0.923 ± 0.003 → 우리 method 효과 0, Q2 fallback

## Execution

```bash
bash experiments/phase6_yonsei_paired/scripts/run_final_queue_parallel.sh
```

Round 1: 16a + 16b 병렬 (~3h)  
Round 2: 16c + 16d 병렬 (~3h)  
Round 3: AffectNet finetune  
Round 4: xeval + aggregate

## Result (2026-05-04 06:33 KST partial)

### Stage 16 ablation (in-domain Korean F1)

| Cell | Stage A | B-iso | B-rel | combine | F1 | Δ vs S6 baseline |
|---|---|---|---|---|---|---|
| Stage 6 full (baseline) | ✓ | ✓ | ✓ | weighted | **0.9256** | — |
| Stage 13 no_stage_a | ✗ | ✓ | ✓ | weighted | 0.9239 | -0.0017 |
| Stage 16a iso_only | ✓ | ✓ | ✗ | iso_only | 0.9252 | -0.0004 |
| Stage 16b rel_only | ✓ | ✗ | ✓ | rel_only | 0.9267 | +0.0011 |
| Stage 16c a_only | DROPPED | | | | — | (degenerate) |
| Stage 16d product | ✓ | ✓ | ✓ | product | 0.9230 | -0.0026 (early-stop E22) |

→ **In-domain ablation 모두 ±0.003 noise level**. 우리 method 효과 in-domain noise.

### Cross-cultural (AffectNet zero-shot, F1)

| Stage 16 variant | AffectNet zero-shot F1 |
|---|---|
| 16a iso_only | 0.3528 |
| 16b rel_only | 0.3220 |
| 16d product | 0.3208 |

→ in-domain noise이지만 cross-cultural는 16a iso_only가 약간 (+1pp) 나음. 단 noise band 내.

### AffectNet finetune (Q1 decision point, 2026-05-04 06:33 완료)

- init_from Stage 6 Korean prior, 8 epochs, batch 64, lr 1e-4
- E001 F1=0.5805 → **E008 F1=0.8794**
- Zero-shot baseline 0.312 → finetune **+56pp**
- vs Q1 threshold 0.65: **PASS**

→ **Q1 evidence 강함**: Korean-trained 3-axis Beta gate FER이 AffectNet (4-class)에서
8-epoch finetune만으로 F1=0.88 → Korean prior가 culturally-invariant AU representation
제공한다는 강한 신호.

### Korean prior vs Scratch — 4 datasets (2026-05-04 09:19 완료)

| Dataset | Style | N | Korean prior F1 | Scratch F1 | Δ |
|---|---|---|---|---|---|
| AffectNet | naturalistic in-the-wild | 16,450 | **0.8794** | 0.8342 | **+4.5pp** ✅ |
| CK+ | naturalistic lab-controlled | 284 | 0.6101 | 0.3070 | **+30.3pp** 🔥 |
| SFEW Train | dramatic movie stills | 466 | 0.5876 | 0.7318 | **-14.4pp** ❌ |
| AFEW Train | dramatic movie videos | 1,124 | 0.6085 | 0.7814 | **-17.3pp** ❌ |

→ **Style-dependent transfer**:
  - Korean naturalistic prior boosts naturalistic Western FER (CK+/AffectNet)
  - Korean naturalistic prior causes **negative transfer** on theatrical FER (SFEW/AFEW)

→ **Q1 publishable claim (revised)**: "Cross-cultural FER transfer is style-dependent
   rather than universal: a Korean Yonsei-calibrated naturalistic prior boosts
   naturalistic Western FER (+4.5~+30pp on AffectNet/CK+) but causes negative transfer
   on theatrical FER (-14~-17pp on SFEW/AFEW). Style mismatch overrides cultural
   similarity in pretrained-FER transfer."

→ **Caveats** (rebuttal-proof 위해 추가 control 권장):
   - SFEW Train만 3-class CSV (neutral 빠짐) → head reinit. 다른 식 비교 모호.
   - 8-epoch 만으로 Korean prior가 dramatic style adapt 못 했을 가능성. Longer finetune 추가 필요.
   - Multi-seed variance bar (현재 single seed=42).
