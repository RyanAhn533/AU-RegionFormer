---
date_planned: 2026-04-21
date_executed:
phase: Phase 0.3
experiment_id: phase0_03
status: planned
claude_directions: [D002]
---

## 1. 가설

**H0**: 17개 landmark feature 중 어느 하나도 감정 class-discriminative 하지 않다
**H1**: 최소 1-3개 feature가 유의한 Fisher ratio (> 0.3)를 가진다 (특히 기쁨 입꼬리)

선행 발견: "기쁨 입꼬리 d=0.79" — 이게 Fisher score로도 상위에 오는지 검증

## 2. 방법

### 입력
- `data/label_quality/face_features.csv` (237K × 17)
- 17 feature 이름 확인 필요

### 측정
- **Fisher discriminant ratio** per feature × emotion pair
  $F = \frac{\sigma_{between}^2}{\sigma_{within}^2}$
- **Univariate ANOVA F-statistic** per feature (4-class)
- **Per-class boxplot**: top 5 feature
- **Cohen's d**: pairwise emotion comparison

### 판정
| 조건 | 결론 |
|------|------|
| Top feature F > 10 | Landmark가 class-discriminative → Graph node feature 후보 |
| Max F < 3 | Landmark 단독으로는 약함 → CNN embedding 병용 |

## 3. 실행

### 코드 위치
`src/analysis/phase0/03_landmark_fisher.py`

### 예상 시간
30분 (표 계산 + 시각화)

## 4-7. (TBD)
