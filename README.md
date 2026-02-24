# AU-Patch Attention FER: Action Unit 기반 얼굴 표정 인식 파이프라인

> MediaPipe FaceMesh 랜드마크로 얼굴 부위별 패치(AU)를 추출하고,  
> Global + AU-Patch Fusion Attention 모델로 감정을 분류하는 End-to-End 파이프라인

---

## 프로젝트 개요

한국인 감정인식 복합 영상 데이터셋(AI Hub)을 대상으로, 얼굴 전체 이미지(Global)와 Action Unit 영역 패치를 함께 활용하여 표정을 분류합니다.

**핵심 아이디어**: 얼굴 전체만 보는 게 아니라, 이마·눈·코·볼·입·턱 등 표정에 핵심적인 6개 영역을 별도 패치로 잘라내서 Transformer Attention으로 융합합니다.

### 파이프라인 흐름

```
원본 이미지
  │
  ├─ [Step 0] YOLOv8 Face Crop (선택) ─→ 얼굴 영역만 잘라낸 이미지
  │
  ├─ [Step 1] AU Patch 좌표 추출 (MediaPipe FaceMesh)
  │     └─ 리사이즈 → 180° 자동 보정 → 랜드마크 좌표 → CSV 저장
  │
  ├─ [Step 2] CSV 검증
  │     └─ NaN/inf 체크, 경로 존재 여부 확인
  │
  └─ [Step 3] AU Fusion Attention 학습
        └─ Global Encoder + Patch Encoder → Fusion Block → FER Head
```

---

## 파일 구조 및 역할

| 파일 | 단계 | 설명 |
|------|------|------|
| `1_facecrop_yolov8.py` | Step 0 (선택) | YOLOv8으로 얼굴 검출 → 가장 큰 얼굴 crop 후 저장 |
| `250809_AU_crop_csv_copy_3.py` | Step 1 (핵심) | MediaPipe FaceMesh 기반 AU 패치 좌표 추출 → CSV 생성 |
| `250810_csv_check_nan.py` | Step 2 | 생성된 CSV의 NaN/inf 및 파일 경로 유효성 검증 |
| `250820_AU_attention1_250926best.py` | Step 3 (핵심) | AU-Patch + Global Fusion Attention 모델 학습 |

---

## Step 0: YOLOv8 Face Crop (선택적 전처리)

**파일**: `1_facecrop_yolov8.py`

배경이 복잡하거나 다수의 인물이 있는 원본 이미지에서 얼굴 영역만 먼저 잘라내는 전처리 단계입니다.

**동작 방식**:
- HuggingFace Hub에서 `arnabdhar/YOLOv8-Face-Detection` 모델 다운로드
- PIL로 이미지 로드 시도 → 실패 시 OpenCV fallback (corrupted 이미지 대응)
- 여러 얼굴이 검출되면 **가장 큰 bounding box**를 선택하여 crop
- 원본 디렉토리 구조를 그대로 유지하며 저장

```bash
# 사용법: 경로를 수정 후 실행
python 1_facecrop_yolov8.py
```

---

## Step 1: AU 패치 좌표 추출

**파일**: `250809_AU_crop_csv_copy_3.py`

이 단계가 파이프라인의 **핵심 전처리**입니다. 얼굴 이미지에서 MediaPipe FaceMesh 랜드마크를 검출하고, 6개 얼굴 영역(AU)의 패치 좌표를 CSV로 저장합니다.

### AU 영역 정의

| 영역 | MediaPipe Landmark Index | 설명 |
|------|--------------------------|------|
| forehead | 69, 299, 9 | 이마 (좌·우·중앙) |
| eyes | 159, 386 | 눈 (좌·우) |
| nose | 195 | 코 |
| cheeks | 186, 410 | 볼 (좌·우) |
| mouth | 13 | 입 |
| chin | 18 | 턱 |

→ 총 **11개 패치** (복수 랜드마크를 가진 영역은 각각 별도 패치로 확장)

### 주요 처리 로직

1. **리사이즈 통일**: 입력 이미지의 짧은 변을 `--work-short` (기본 800px)로 통일
2. **180° 자동 보정**: 0°와 180° 회전 중 눈 간 거리(eye distance)가 더 큰 쪽을 선택
3. **좌표 기반 CSV 생성**: 이미지 자체를 저장하지 않고, 좌표(center, window, clamped)만 CSV에 기록 → 디스크 절약
4. **멀티프로세싱**: `spawn` 방식으로 MediaPipe 인스턴스를 워커별로 독립 생성하여 병렬 처리

### CSV 스키마

```
path, label, rot_deg, work_w, work_h, patch_in, patch_out,
{au_name}_cx, {au_name}_cy,           # 랜드마크 중심점
{au_name}_wx1, _wy1, _wx2, _wy2,      # 패치 윈도 좌표 (경계 미보정)
{au_name}_cx1, _cy1, _cx2, _cy2       # 클램핑된 좌표 (이미지 경계 보정)
```

### 사용법

```bash
# 기본 실행 (미리보기 5장 + train/val CSV 생성)
python 250809_AU_crop_csv_copy_3.py \
    --train /path/to/train \
    --val /path/to/val \
    --outdir /path/to/output \
    --work-short 800 \
    --patch-in 256 \
    --patch-out 256

# CSV 기반으로 실제 패치 이미지도 저장하고 싶을 때
python 250809_AU_crop_csv_copy_3.py --export-from-csv
```

---

## Step 2: CSV 검증

**파일**: `250810_csv_check_nan.py`

Step 1에서 생성된 CSV의 품질을 빠르게 점검하는 유틸리티입니다.

**검증 항목**:
- 데이터 타입 및 기본 통계 (`df.info()`, `df.head()`)
- NaN 값 존재 여부 (컬럼별)
- Inf 값 존재 여부 (수치 컬럼)
- `path` 컬럼에 기록된 파일 경로의 실제 존재 여부

---

## Step 3: AU Fusion Attention 모델 학습

**파일**: `250820_AU_attention1_250926best.py`

### 모델 아키텍처: `AUFusionModel`

```
입력 이미지 ──→ GlobalEncoder (MobileNetV3 + Depthwise Conv Tail) ──→ g [B, d]
                                                                          │
AU 패치 x11 ──→ PatchEncoder (MobileNetV3 + Linear Proj) ──→ au [B, K, d] │
                           + AU Positional Embedding                       │
                                                                          │
                         ┌────────────────────────────────────────────────┘
                         ▼
              [CLS] + g + au_tokens ──→ FusionBlock
                                         ├─ Self-Attention (Transformer Encoder)
                                         └─ Cross-Attention (Q=[CLS,g], KV=AU)
                                              │
                                              ▼
                                          FERHead (MLP) ──→ logits [B, num_classes]
```

### 구성 모듈 상세

| 모듈 | 역할 | 구조 |
|------|------|------|
| `PatchEncoder` | AU 패치 → 임베딩 | MobileNetV3 backbone (576-d) → Linear → d |
| `GlobalEncoder` | 전체 얼굴 → 임베딩 | MobileNetV3 backbone → 1×1 Conv → DW Conv → 1×1 Conv → GAP → d |
| `FusionBlock` | Global + AU 융합 | Self-Attention + Cross-Attention (learnable gate γ) |
| `FERHead` | 분류 | LayerNorm → Linear → SiLU → Dropout → Linear |

### 학습 전략

- **Backbone Freeze**: 처음 5 에폭은 MobileNetV3 backbone을 동결하고 head만 학습, 이후 전체 fine-tuning
- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.05)
- **Scheduler**: Warmup (5%) + Cosine Decay
- **Loss**: Cross-Entropy + Label Smoothing (0.05) + 클래스별 가중치 (불균형 보정)
- **Mixed Precision**: `torch.cuda.amp` AMP 학습
- **Gradient Clipping**: max norm 5.0
- **Best Model 기준**: Validation Macro F1-Score

### 데이터 증강

| 대상 | Train | Validation |
|------|-------|------------|
| Global (224×224) | RandomHorizontalFlip, ColorJitter | Resize only |
| Patch (128×128) | RandomHorizontalFlip | Resize only |
| 공통 | ImageNet Normalize (timm 기반 mean/std) | 동일 |

### 출력물

```
output_dir/
├── best.pth              # Best macro-F1 체크포인트
├── last.pth              # 마지막 에폭 체크포인트
├── confusion_best.png    # Best 에폭의 Confusion Matrix
├── report_best.txt       # Best 에폭의 Classification Report
└── history.jsonl         # 에폭별 학습 로그 (train/val loss, f1, acc, lr)
```

### 사용법

```bash
# 설정 변수를 파일 상단에서 수정 후 실행
python 250820_AU_attention1_250926best.py
```

---

## 하이퍼파라미터 요약

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `EPOCHS` | 300 | 최대 학습 에폭 |
| `BATCH_SIZE` | 32 | 배치 크기 |
| `BASE_LR` | 2e-4 | 초기 학습률 |
| `WEIGHT_DECAY` | 0.05 | AdamW 가중치 감쇠 |
| `LABEL_SMOOTHING` | 0.05 | 라벨 스무딩 계수 |
| `D_EMB` | 512 | 임베딩 차원 |
| `N_HEAD` | 8 | Attention Head 수 |
| `N_LAYERS` | 2 | Transformer Encoder 레이어 수 |
| `GLOBAL_IMG_SIZE` | 224 | 전체 얼굴 이미지 입력 크기 |
| `PATCH_OUT_SIZE` | 128 | AU 패치 입력 크기 |
| `FREEZE_BACKBONE_EPOCHS` | 5 | Backbone 동결 에폭 수 |

---

## 의존성

```
torch >= 2.0
timm
torchvision
ultralytics          # Step 0 (YOLOv8)
mediapipe            # Step 1 (FaceMesh)
opencv-python
Pillow
numpy
pandas
scikit-learn
matplotlib
tqdm
huggingface_hub      # Step 0 (모델 다운로드)
```

---

## 데이터셋

- **출처**: [AI Hub - 한국인 감정인식을 위한 복합 영상](https://aihub.or.kr/)
- **구조**: `{split}/{emotion_label}/{image_files}`
- **감정 클래스**: CSV의 `label` 컬럼에서 자동 추출 (디렉토리명 기반)

---

## License

TBD

---

## Acknowledgments

- AI Hub 한국인 감정인식 복합 영상 데이터셋
- [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [YOLOv8 Face Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
