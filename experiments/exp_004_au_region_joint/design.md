---
exp_id: exp_004_au_region_joint
iteration: 4
date: 2026-04-22
category: feature_engineering
strategy: d (조합 — top 2~5위 ensemble)
---

# Exp 004 — AU + Region Joint Linear Probe

## 연구 질문
AU-level (59.9%) + Region-level (81.5%)을 합치면:
1. Region 단독보다 상승? (capacity 상한?)
2. AU 정보가 Region과 **orthogonal**인가 (상승 여부로 판정)

## 이 실험이 논문에 주는 기여
- Region embedding이 이미 AU 정보를 내포하는가? → **method novelty** 판정
  - 만약 AU+Region이 Region 단독과 동일하면: Region이 AU를 다 흡수 → AU-level 분석은 **해석용**만
  - 만약 상승하면: AU가 추가 signal → **joint feature**가 future graph 노드로 의미

## 핵심 아이디어 (3줄)
1. ConvNeXt POOLED (8192d) + OpenGraphAU 41 AU intensity (41d) concat → 8233d
2. Linear probe 동일 protocol
3. Region 단독 81.5% vs Joint 비교

## 구체 수정 포인트
- 신규 스크립트 `01e_joint_probe.py`:
  - Region embedding load (Phase 0.1c와 동일)
  - AU parquet load (Phase 0.2 v2)
  - Image path 기준 join (resolved_path 또는 index 매칭)
  - Concat + L2 norm + PCA(256) + LogReg

## 예상 metric 변화
- Region POOLED 81.5% → **joint 82~83%** 예상
- AU 기여 추정: +0.5~1.5%p (marginal)
- 만약 +3%p 이상이면 대성공 (method novelty 확보)

## Paper section 매핑
- §4.2 Joint feature analysis
- §5.1 Discussion (Region이 AU를 얼마나 흡수하는가)

## 주의
- Join 시 이미지 매칭 정확성 중요 (sad subset 매핑 새로 됐으니 검증)
- AU parquet에 image index column 필요 (현재 resolved_path 있음)
- embedding npy는 meta CSV row-wise index — 별도 index 매칭 필요

## 예상 시간
- 20-30분
