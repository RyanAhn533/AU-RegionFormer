# Experiment History

AU-RegionFormer에 도달하기까지 12회의 실험과 5회의 학습 발산(NaN)을 거쳤습니다.

---

## 전체 실험 요약

| # | Phase | Model | Backbone | Acc | F1 | 상태 | 핵심 교훈 |
|---|-------|-------|----------|-----|-----|------|----------|
| 1 | Baseline | CNN Baseline | CNN | 74.7% | 0.745 | ✅ | sad/hurt 등 미세 감정에서 한계 |
| 2 | ViT 도입 | ViT Pretrained | ViT-S | 43.1% | 0.399 | ❌ | domain gap, local bias 부재 |
| 3 | ViT 도입 | BlurDWViT | ViT-S | — | — | ✅ | local bias 보강으로 개선, 하지만 "어디를 볼지" 모름 |
| 4 | AU 검증 | Full-face xattn only | ViT-S | 46.9% | 0.447 | ❌ | AU 없는 attention은 무의미 |
| 5 | **AU 도입** | **AU-CNN-ViT** | **ViT-S** | **79.7%** | **0.797** | **✅** | **AU prior → +5%p 돌파** |
| 6 | 경량화 | MobileViT 단순 교체 | MobileViT | 66.5% | 0.654 | ❌ | backbone만 바꾸면 안 됨 |
| 7 | 경량화 | AU-GNN | MobileNetV3 | 77.6% | 0.775 | 💥 ep30 | GNN gradient 폭발 |
| 8 | 경량화 | CAG-MoE | MobileNetV3 | 73.8% | 0.732 | 💥 ep9 | MoE gating 불안정 |
| 9 | 경량화 | AU-ViT3 | MobileNetV3 | 62.9% | 0.622 | 💥 ep3 | 경량 ViT + 복잡 fusion = 즉시 발산 |
| 10 | 경량화 | AU-Attention v2 | MobileNetV3 | 75.1% | 0.752 | 💥 ep17 | Focal + Label Smoothing 충돌 |
| 11 | 경량화 | ResNet18+Linear | ResNet18 | 60.8% | 0.559 | ❌ | 단순 head로는 AU 효과 못 살림 |
| 12 | **최종** | **AU-RegionFormer** | **MobileViTv2** | **79.7%** | **0.795** | **✅** | **feature-level AU + 단순 fusion** |

💥 = NaN 발산으로 학습 실패

---

## 연구 흐름

```
DeepFace 분석 ─── 감정 = facial interaction 문제
      │
      ▼
CNN Baseline (#1) ─── 74.7%, sad/hurt 취약
      │
      ▼
ViT 시도 (#2,3) ─── global context는 가능, "어디를 볼지" 모름
      │
      ▼
FACS 발견 ─── Action Unit = 심리학적 region prior
      │
      ▼
FaceMesh AU 좌표 실험 ─── 468 landmark → 6개 최적 영역 탐색
      │
      ▼
AU-CNN-ViT (#5) ─── 79.7%, +5%p 돌파
      │
      ▼
경량화 시도 (#6~11) ─── 5번 NaN 발산
      │
      ▼
AU-RegionFormer (#12) ─── feature-level sampling, 단일 forward
```

---

## Phase별 설계 판단

### 왜 ViT를 도입했는가

CNN baseline에서 `sad`(F1=0.624), `hurt`(F1=0.585) 성능이 낮았습니다.
이 감정들은 단일 부위가 아니라 눈썹+입꼬리+볼의 **조합 변화**로 구분됩니다.
→ 부위 간 관계를 볼 수 있는 Transformer 구조가 필요하다고 판단했습니다.

### 왜 AU를 도입했는가

ViT는 관계를 볼 수 있지만, **어디를** 봐야 하는지 모릅니다.
FACS의 Action Unit이 이 문제의 답이었습니다.
- AU 없이 cross-attention만 쓴 실험(#4): **46.9%** — 완전 실패
- AU를 넣은 실험(#5): **79.7%** — 성공
→ **AU prior가 핵심**임을 확인.

### 왜 feature-level sampling인가

AU-CNN-ViT는 6개 패치를 각각 별도 CNN으로 인코딩 → 7회 forward.
이를 경량 backbone으로 바꾸자 5번 연속 NaN 발산.
→ 패치를 따로 인코딩하는 방식 자체를 버리고,
backbone feature map에서 AU 좌표를 직접 sampling하는 방식으로 전환.
**Forward 7회 → 1회, 추가 연산 0.3% 미만.**

---

## 클래스별 성능 변화

| Class | CNN Baseline | AU-RegionFormer | 변화 |
|-------|-------------|-----------------|------|
| angry | 0.796 | 0.870 | +0.074 |
| anxious | 0.624 | 0.696 | +0.072 |
| happy | 0.945 | 0.968 | +0.023 |
| hurt | 0.585 | 0.635 | +0.050 |
| neutral | 0.857 | 0.919 | +0.062 |
| sad | 0.624 | 0.637 | +0.013 |
| surprised | 0.786 | 0.842 | +0.056 |
| **Macro F1** | **0.745** | **0.795** | **+0.050** |

---

## 학습 설정 (최종 모델)

| 항목 | 설정 |
|------|------|
| Backbone | MobileViTv2_100 (5M, ImageNet pretrained) |
| 2-Stage Training | ep 1~3 freeze → ep 4~200 unfreeze (lr×0.1) |
| Loss | Focal Loss (γ=2.0), Label Smoothing 미사용 |
| Optimizer | AdamW (lr=3e-4, wd=0.05) |
| Scheduler | Cosine Annealing + 5% warmup |
| Gradient Clipping | max_norm=5.0 |
| AMP | FP16 mixed precision |
| 학습 시간 | RTX A6000, 약 35시간 (200 epoch) |
| NaN 발생 | 없음 |

---

## AU Region별 감정 기여도

| 감정 | 주요 AU 영역 |
|------|-------------|
| Happy | Cheek (smile muscles) |
| Angry | Forehead + Nose |
| Surprise/Fear | Eye regions |
| Sad | Global feature 의존도 높음 |
| Neutral | 전 영역 low magnitude |
