---
date_planned: 2026-04-21
date_executed:
phase: Phase 0.2
experiment_id: phase0_02
status: planned
claude_directions: [D002]
depends_on: [phase0_01]
---

## 1. 가설

**Target 현상**: AU embedding 감정 간 cosine similarity 0.97-0.99 (기존 보고)
- Phase 0.1에서 실제 값 재측정 후, 0.9 이상이면 본 실험 진행
- 0.9 미만이면 가설 자체 재검토

**Three hypotheses**:
- **H1 (Pooling)**: Global mean pooling이 class-specific feature를 희석
- **H2 (Backbone)**: MobileViT/ConvNeXt는 ImageNet pretrained, AU-specific 아님
- **H3 (Data)**: 배우 연기 데이터라 감정 표현이 인위적/균질

## 2. 방법

### H1 검증: Pooling 세밀화
- 각 8 region을 sub-patch 4개로 분할 (예: 16×16 → 4×4×4)
- Sub-patch별 cos sim 재측정
- Sub-patch를 attention pooling으로 결합 시 cos sim 변화
- **판정**: sub-patch cos sim이 global pooling보다 낮으면 H1 지지

### H2 검증: AU-specialized backbone 비교
- OpenGraphAU pretrained 모델로 embedding 재추출 (subset 10K)
- ME-GraphAU pretrained 모델 (if available)
- 세 backbone의 cos sim 및 silhouette 비교
- **판정**: AU backbone cos sim이 현저히 낮으면 H2 지지

### H3 검증: 데이터 subset 분리 측정
- 배우 subset (actor=True) vs 일반인 subset 각각 cos sim
- `is_selected=1` (연세대 reject) vs `=0` 분리
- 감정별 variance 비교
- **판정**: 배우/일반인 격차 크면 H3 지지

### 입력 데이터
- 기존 embedding: `data/label_quality/au_embeddings/`
- 배우/일반인 라벨: `meta_*.csv`의 path에서 추정 or 별도 csv
- OpenGraphAU checkpoint: 다운로드 필요

## 3. 실행

### 코드 위치
- `src/analysis/phase0/02a_pooling_refinement.py`
- `src/analysis/phase0/02b_backbone_comparison.py`
- `src/analysis/phase0/02c_data_subset.py`

### 리소스
- GPU: H2에서 embedding 재추출에 필요 (10GB)
- 예상 시간: H1 2h, H2 4h (embedding 재추출), H3 1h

## 4-7. (TBD)
