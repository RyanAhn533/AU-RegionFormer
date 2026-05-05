---
exp_id: exp_002_yonsei_clean_subset
iteration: 2
date: 2026-04-22
category: data
strategy: c (failure case analysis → high-quality subset)
depends_on: exp_001 (sad-fixed parquet)
---

# Exp 002 — 연세대 Verified Clean Subset Linear Probe

## 연구 질문
298명 연세대 학부생이 "이 감정이 아니다" 판정한 샘플(`is_selected=1`)을 제거하면:
1. Linear probe accuracy가 상승하는가? (data noise → accuracy)
2. Jack 2012 재검증 nuance가 더 명확해지나? (clean subset에서 mouth-eye gap)
3. FACS canonical 재현이 강해지나? (Happy=AU6+12 in clean data)

## 원 자료
JY asset: 연세대 심리학과 298명 × ~24만장 검증. `is_selected=1` = 최소 1명 이상이 "이 감정 아니다" 부정 판정.

## 핵심 아이디어 (3줄)
1. `is_selected=0` (사회적 합의 passed) 샘플만 filter
2. Phase 0.1c + Phase 0.3 동일 protocol로 linear probe 재측정
3. `is_selected=1` (rejected)와 비교 → noise effect size

## 구체 수정 포인트
- Phase 0.1c 스크립트: `--filter is_selected==0` 추가
- Phase 0.3 스크립트: 동일
- leaderboard entry에 `subset: "yonsei_clean"` 필드

## 예상 metric 변화
- Region POOLED (현재 81.5%): +1~3%p 예상 (0.815 → 0.83~0.84)
  - 근거: 연세대 rejection 비율 ~8% (slight data cleaning)
- AU-level 59.9%: +2~5%p 예상
- **Primary 3 (Jack reversal, AU level)**: 현재 +1.2%p → +2~5%p 확장 예상
- **FACS Sad 재현**: 현재 43.6% → 50%+ (노이즈 제거 효과 큼)

## Paper section 매핑
- §4.5 Social consensus analysis (핵심)
- §4.1 Ablation ("with/without 연세대 filter")

## 실패 시 fallback
- Clean subset accuracy가 raw와 차이 없음 → "298명 판정이 label noise 아님 — 다른 축(annotator style 등)"으로 해석 (여전히 §4.5 기여)

## Primary 개선 판정
- accuracy +0.01 이상 AND p<0.05 → leaderboard best 갱신 후보
- accuracy 차이 작아도 per-class F1 floor(worst class) 상승 > 2%p면 "noise removal" 효과로 기록

## 예상 시간
- 15-20분 (extraction 불필요 — 기존 parquet filter + probe 재실행)
