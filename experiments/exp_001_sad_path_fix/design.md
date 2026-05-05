---
exp_id: exp_001_sad_path_fix
iteration: 1
date: 2026-04-22
category: data
strategy: c (failure case analysis → weakness targeting)
depends_on: phase0_03 (leaderboard exp phase0_03_v1)
---

# Exp 001 — Sad Class Path Resolver Fix

## 문제 (phase0_03_v1 failure case)

Phase 0.3 linear probe 결과에서 슬픔 class 58,106 중 5,386 (9.3%)만 이미지 매칭.
원인: `sad/` 폴더만 파일명이 `sad_00001.jpg` 형식으로 re-numbered, 다른 emotion은 원본 해시명.

## 원 논문/기법 출처
N/A (data pipeline bug fix). Path resolution logic 추가.

## 핵심 아이디어 (3줄)
1. `sad_to_orig_mapping.csv` (58,460 rows) 이미 존재 — `orig_fname` ↔ `sad_fname` 매핑
2. OpenGraphAU extraction resolver에서 emotion=슬픔일 때 mapping으로 sad_fname 조회
3. `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/sad/sad_XXXXX.jpg` 경로 생성

## 구체 수정 포인트
- `src/analysis/phase0/02_opengraphau_extract.py`:
  - `find_image_path` 함수 수정: 슬픔 emotion 분기
  - mapping CSV 글로벌 dict 로드: `{orig_fname → sad_fname}`
  - 수정 후 slot-in 재추출 (슬픔만 subset run — 시간 절약)

## 예상 성능 변화와 근거
- Phase 0.3 per-class F1: 슬픔 F1 현재 0.15~0.25 추정 (data underrepresented) → 0.55+ 예상
- All-AU 41d accuracy: 59.9% → 60~65% (slight 상승 — 슬픔 class 신뢰도 복구)
- Ekman "Sad=AU1+4+15" 재현도 43.6% → 더 정확한 수치 가능 (신뢰도 상승)

## Primary metric 예측
- Phase 0.1c region POOLED는 **변화 없음** (이미 237K 전체로 돌림, 슬픔은 original path)
- Phase 0.3 AU-level는 upward 재평가 예상

## 실패 시 fallback
- mapping CSV의 일부 orig_fname이 encoding 문제로 meta CSV와 매칭 안 되면:
  - 1) hash prefix만 매칭 (앞 64자)
  - 2) unmatched는 제외 + per-class balance 재확인
  - 3) 3-class (기쁨/분노/중립) 실험으로 축소
