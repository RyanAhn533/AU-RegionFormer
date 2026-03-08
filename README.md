# AU-RegionFormer

**Action Unit 기반 경량 얼굴 감정 인식 모델**

FACS(Facial Action Coding System)의 Action Unit을 vision architecture에 통합하여,
단일 forward pass로 전체 얼굴 맥락과 AU 영역 특징을 동시에 처리합니다.

---

## 핵심 결과

| 항목 | 수치 |
|------|------|
| 7-class Accuracy | **79.7%** |
| Macro F1 | **0.795** |
| Backbone | MobileViTv2-100 (5M params) |
| AU 처리 추가 연산 | 전체의 0.3% 미만 |
| Baseline 대비 | **+5.0%p** (CNN 74.7% → 79.7%) |

12회 실험, 5회 학습 발산(NaN)을 거쳐 도달한 구조입니다. → [실험 히스토리](docs/experiments.md)

---

## 아키텍처

![AU-RegionFormer Architecture](Architecture.png)

```
Input (224x224)
  → MobileViTv2 (single forward)
  → Feature Map [B, 384, 7, 7]
  ├── Global Average Pooling → Global Token
  └── Bilinear Grid Sampling (6 AU 좌표) → AU Tokens
  → [CLS] + [Global] + [AU x6] → Cross-Attention Fusion → 7-class
```

**핵심**: AU 패치를 이미지에서 따로 잘라 6번 인코딩하는 대신,
backbone feature map에서 AU 좌표를 bilinear sampling으로 직접 추출합니다.

→ [아키텍처 상세](docs/architecture.md)

---

## 왜 AU인가

기존 FER 모델은 얼굴의 **어디를 봐야 하는지** 모릅니다.

FACS의 Action Unit은 각 감정이 **어떤 얼굴 근육의 조합**인지 정의합니다:

| 감정 | AU 조합 |
|------|--------|
| happy | AU6(볼 올림) + AU12(입꼬리 올림) |
| sad | AU1(눈썹 올림) + AU4(눈썹 내림) + AU15(입꼬리 내림) |
| angry | AU4(눈썹 내림) + AU5(눈 크게) + AU23(입술 조임) |

이 심리학적 prior를 MediaPipe FaceMesh 468 landmark 기반으로 좌표화하여,
모델 구조에 **explicit region prior**로 주입했습니다.

```
forehead → AU1,2    eyes → AU5,7    nose → AU9
cheeks   → AU6      mouth → AU12,15  chin → AU17
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 학습
python scripts/train.py \
    --config configs/mobilevit_fer.yaml \
    --epochs 200 --batch_size 64

# 추론
python -c "
from src.inference.fer_inferencer import FERInferencer
inferencer = FERInferencer('outputs/best.pth', device='cuda')
result = inferencer.predict(frame)
print(result['emotion'], result['confidence'])
"
```

---

## 실전 배포

Jetson Orin 기반 Driver Monitoring System의 감정 인식 모듈로 통합되었습니다.

```
RealSense D435 → YOLOv8 → FaceMesh → AU-RegionFormer → Multimodal Fusion → Vehicle Gateway
```

10-item 프로토콜 기준 **Macro 98.0%** 인식 정확도 달성.

---

## 문서

| 문서 | 내용 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 모듈별 구조 상세, 수식, multimodal 통합 |
| [docs/experiments.md](docs/experiments.md) | 12회 실험 요약, 연구 흐름, 설계 판단 근거 |

---

## Repo Structure

```
AU-RegionFormer/
├── configs/             # 학습 설정 (YAML)
├── scripts/             # 진입점 (train, pipeline)
├── src/
│   ├── preprocessing/   # 얼굴 검출 + AU 추출
│   ├── datasets/        # 데이터 로딩
│   ├── models/core/     # 모델 아키텍처
│   │   ├── fer_model.py #   AUFERModel (메인)
│   │   ├── backbones/   #   MobileViTv2
│   │   ├── fusion/      #   CrossAttention + AU RoI
│   │   └── heads/       #   FER head
│   ├── training/        # 학습 루프, loss, scheduler
│   ├── inference/       # 실시간 추론
│   └── integration/     # 멀티모달 통합
└── docs/                # 문서
```

---

## Technical Specs

| Component | Details |
|-----------|---------|
| Backbone | MobileViTv2_100 (5M, ImageNet pretrained) |
| AU Regions | 6 (Forehead, Eyes, Nose, Cheeks, Mouth, Chin) |
| Fusion | 1-layer cross-attention, 8 heads, gated residual |
| Loss | Focal (γ=2.0) + class weighting |
| Optimizer | AdamW (lr=3e-4, wd=0.05) |
| Training | 2-phase (freeze 3ep → unfreeze lr×0.1) |
| Precision | AMP FP16 |
