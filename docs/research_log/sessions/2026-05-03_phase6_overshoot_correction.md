---
date: 2026-05-03
session: phase6_overshoot_correction
duration_h: ~10 (이전 세션 36h 누적 + 오늘 정리 4h)
participants: [JY, Claude Opus 4.7]
status: in_progress (큐 ETA 04:30 KST 다음날)
claude_directions:
  - id: D001
    content: "Stage 6 multi-seed × 4 + Stage 7 JEPA + Stage 7c patch_dropout + Stage 8 DINOv2 + Stage 11 humankl × 5 lambdas + Stage 13 + Stage 14 + Stage 15 × 3 backbones를 weekend queue로 launch (plan 1순위 ablation matrix는 후순위로)"
    rationale: "GPU 풀 사용 + 다양한 variant 한 번에 검증"
    risk: "plan에 명시된 5-variant ablation matrix가 늦어짐"
    accepted: true
    hindsight_score: null
    outcome: "in-domain F1 모두 ±0.002 noise level. 36시간 GPU 사용했으나 paper-worthy contribution 미검증. JY 평가: '존나병신처럼 이것저것'"
  - id: D002
    content: "7-class trained 모델로 ceiling 가설 검증하자고 제안"
    rationale: "4-class에서 saturation 현상이 architectural trick의 효과를 가리는 듯"
    risk: "plan 명시 '7-class 먼저 X, 4-class fast iter로 신호 검증부터' 위반"
    accepted: false
    hindsight_score: null
    outcome: "JY 거부 + 호통: '쓸때없는 짓이나 쳐하고있네'. 철회."
  - id: D003
    content: "Stage 14/15 (plan 외) 죽이고 Stage 16 ablation matrix (plan 핵심) + AffectNet finetune (Q1 결정점)만 남긴 슬림 큐로 재배치"
    rationale: "plan 1순위로 회귀 + 6시간 단축"
    risk: "Stage 14/15 결과를 영영 못 봄 (paper에 들어갈 가치 없으니 OK)"
    accepted: true
    hindsight_score: null
    outcome: "큐 launch 완료. Stage 16a + 16b 병렬 진행 중. ETA 6.5h."
  - id: D004
    content: "BF16 autocast + cudnn.benchmark + TF32 + parallel 2개 launch로 throughput 최적화"
    rationale: "JY가 GPU_효율화_정리.md 가리킴 → 적용 가능한 최적화"
    risk: "병렬 시 OOM 가능 — 실제로는 안전 (각 stage ~17GB, 합쳐도 34GB)"
    accepted: true
    hindsight_score: null
    outcome: "병렬 launch 성공. 단 학습 속도 단축 효과는 mobilenet 베이스라 미미. 병렬화로 sequential 10h → 6.5h."
  - id: D005
    content: "현재 큐 결과 후 Q1 (TAFFC/AAAI) 가능성 약함 → Q2 ESWA + 새 paper로 VLM hybrid pivot 권장"
    rationale: "vision-only FER은 2026 saturated, 우리 method 3-axis Beta gate / Yonsei distillation 효과 noise level"
    risk: "VLM pivot은 plan 명시 '안 함'에 가까움. 시간 risk."
    accepted: pending
    hindsight_score: null
    outcome: pending — JY가 결과 본 후 결정
decisions:
  accepted: [D001, D003, D004]
  rejected: [D002]
  pending: [D005]
---

# 2026-05-03 Phase 6 — Overshoot Correction Session

## 핵심

JY 명시 plan(`woolly-sniffing-karp.md`)의 1순위 = **5-variant ablation matrix
(baseline / +A / +B-iso / +B-rel / full) × 1 seed**.  
지난 36시간 동안 plan 외 stage들(7/8/11/14/15)을 무차별로 돌렸으나 in-domain
F1이 ±0.002 noise level이라 의미 못 짠 거 발견. JY 호통받고 정리 시작.

## 결과 요약

### In-domain F1 (Korean master_val, 4-class) — 모두 ceiling

| Stage | 설명 | F1 |
|---|---|---|
| Stage 6 full (baseline) | Stage A + Beta-iso + Beta-rel | **0.9256** |
| Stage 6 seed 123/777/999/2024 | multi-seed | 0.9228-0.9255 (mean ~0.9240, σ~0.001) |
| Stage 7 jepa | + JEPA aux | 0.9248 |
| Stage 7c patch_dropout | + patch dropout | 0.9255 |
| Stage 8 dinov2 | DINOv2 global encoder | 0.9203 |
| Stage 11 humankl_lam {01,03,05,07,10} | Yonsei rater KL distillation sweep | 0.9250-0.9258 (lam03 best) |
| Stage 13 no_stage_a | Stage A 제거 | 0.9239 |

→ **모든 변형 noise level. 우리 method가 in-domain 4-class에서 의미 있는 lift 못 만듦.**

### Cross-cultural zero-shot (Korean→Western, F1, chance=0.25)

| Ckpt | AffectNet | SFEW_Tr | AFEW_Tr | CK+ |
|---|---|---|---|---|
| Stage 6 full | 0.312 | 0.178 | 0.218 | 0.239 |
| Stage 7 jepa | 0.341 | 0.184 | 0.195 | 0.263 |
| **Stage 8 dinov2** | **0.377** | **0.258** | **0.225** | 0.256 |
| Stage 11 lam05 | 0.342 | 0.187 | 0.217 | 0.211 |

→ **DINOv2 pretrained backbone이 cross-cultural transfer 핵심**. 단 trivial finding
("pretrained better"는 모두가 안다). 우리 method 덕은 아님.

### Stage 9 alignment (Yonsei 298 raters consensus signal)

- 14 ckpts × C1-C8 분석 완료
- Per-class Spearman: n=4라 wildly varying (-1.0 ~ +1.0)
- Per-image Pearson: 0.05-0.11 (basically zero correlation)
- **alignment 자체가 약함** → Yonsei signal이 paper claim 키 못 됨

## 진행 중 큐 (final_queue_parallel.sh)

| 라운드 | 작업 | ETA |
|---|---|---|
| Round 1 | Stage 16a iso_only + Stage 16b rel_only (병렬) | ~3h |
| Round 2 | Stage 16c a_only + Stage 16d product (병렬) | ~3h |
| Round 3 | AffectNet finetune (init_from Stage 6) | ~30min |
| Round 4 | xeval Stage 16 ckpts on AffectNet + aggregate | ~10min |

총 ~6.5h, 종료 예상 04:30 KST.

## Stage 16 ablation matrix 의미

| Variant | Stage A | B-iso | B-rel | combine |
|---|---|---|---|---|
| Stage 13 (이미 완료) | ✗ | ✓ | ✓ | weighted |
| Stage 16a iso_only | ✓ | ✓ | ✗ | iso_only |
| Stage 16b rel_only | ✓ | ✗ | ✓ | rel_only |
| Stage 16c a_only | ✓ | ✗ | ✗ | iso_only(degenerate) |
| Stage 16d product | ✓ | ✓ | ✓ | product |
| Stage 6 full (baseline) | ✓ | ✓ | ✓ | weighted |

→ Stage A 효과(S13 vs S6), Beta gate 변형 효과(16a/b/c/d) 한 번에 검증.

## Q1 vs Q2 결정 기준 (04:30 KST 결과 본 후)

- **Q1 push**: AffectNet finetune F1 ≥ 0.65 + Stage 16 변형 간 Δ > 0.005 (effect size 검출)
- **Q2 stop**: AffectNet finetune F1 < 0.55 또는 Stage 16 다 noise

## 다음 paper 방향 (D005, pending)

- A. Q2 ESWA 마무리 (4개월) — 안전, 트렌드 부합 약함
- B. VLM hybrid (Qwen2.5-VL × AU) pivot (3-4개월) — 트렌드 정통, 새 paper
- C. Trimodal vision+bio+audio (6-9개월) — 졸업 6개월에 무리

## 교훈 (이번 세션)

- Plan에 명시된 1순위(ablation matrix)를 지키지 않고 overshoot한 36시간이 결정적 손실.
- "GPU 풀 사용 = 좋은 일" 자명하지 않음. **plan 우선순위와 align된 GPU 사용**이 진짜 가치.
- vision-only FER은 2026 saturated. 우리 method 효과 noise면 framing/method pivot 필요.
- "DINOv2 pretrained better"는 paper claim 아님 (trivial). 진짜 contribution은 우리 만의 method 효과.
