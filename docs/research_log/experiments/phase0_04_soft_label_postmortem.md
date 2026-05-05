---
date_planned: 2026-04-21
date_executed:
phase: Phase 0.4
experiment_id: phase0_04
status: planned
claude_directions: [D002]
---

## 1. 가설

**Target**: v5 soft label 실험 F1=0.538 실패 원인 규명

**3가지 가설**:
- **H1 (Loss 희석)**: KL divergence weight가 CE weight를 overpower
- **H2 (Target distribution 왜곡)**: Soft target이 entropy 너무 높아 gradient 모호
- **H3 (Hyperparameter)**: Temperature scaling / mixup weight 튜닝 부족

## 2. 방법

### 읽을 파일
- `src/training/noise_aware_trainer.py`
- `src/data/noise_aware_dataset.py`
- v5 training config + loss curve log

### 분석
- v5 loss curve (train/val CE, KL 별도)
- Target distribution entropy per sample
- Per-class validation F1 추이
- KL weight ablation (가능하면 재학습 없이 log 재분석)

### 판정
- **실패 원인 1문장**으로 요약
- Phase 3 consensus-aware graph 설계 시 이 실패를 어떻게 회피할지 constraint 도출

## 3. 실행

### 코드 위치
`src/analysis/phase0/04_soft_label_postmortem.py`

### 예상 시간
2일 (코드 읽기 + log 분석)

## 4-7. (TBD)
