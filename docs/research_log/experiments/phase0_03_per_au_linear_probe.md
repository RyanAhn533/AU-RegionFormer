---
date_planned: 2026-04-21
date_executed:
phase: Phase 0.3
experiment_id: phase0_03
status: planned
claude_directions: [D008, D010]
depends_on: [phase0_02]
unblocks: [phase1_1, phase1_2]
---

# Phase 0.3 — Per-AU Linear Probe (AU 단위 감정 구분력)

---

## 0. 이전 단계와의 연결

Phase 0.1c: **region-level** 분석 → mouth 75%, eyes 61%, forehead 45% → AI 관점 mouth 우위 확인
Phase 0.2: OpenGraphAU로 **AU-level** intensity 237K 추출 (41 AU)

이 실험이 답하려는 것: **"어떤 AU가 한국인 감정 구분에 가장 중요한가? Ekman FACS canonical mapping이 실증 데이터에서 재현되는가? Jack 2012의 '한국인은 eyes 중심' 가설이 AU 단위로 검증되는가?"**

---

## 1. 목적

### Primary questions
1. **각 AU가 감정 구분에 얼마나 기여하나?** (per-AU 단독 accuracy)
2. **FACS canonical (Happy=AU6+12, Sad=AU1+4+15) 한국인 재현?**
3. **Eye-AU group vs Mouth-AU group — Jack 2012 직접 검증?**

### Paper section 매핑
- **§4.2 AU-level class-discriminative power** — per-AU ranking
- **§4.3 Jack 2012 재검증** — eye vs mouth AU group 비교 (논문 핵심 Figure)
- **§4.4 FACS canonical validation** — Ekman mapping 재현도

### 성공 조건
| 조건 | 다음 액션 |
|-----|---------|
| Mouth-AU group > Eye-AU group accuracy (>5%p 차이) | Jack 2012 정면 반박 확보 → Phase 1 GO |
| 차이 < 2%p | Jack 가설 부분 지지 — 더 세밀한 분석 |
| Eye-AU > Mouth-AU | Jack 가설 지지 — thesis 재구성 |

---

## 2. 방법

### Data
- 입력: `data/label_quality/au_features/opengraphau_41au_237k.parquet` (Phase 0.2 산출)
- 41 AU intensity (0-100 scale)
- 4-class emotion label (기쁨/분노/슬픔/중립)

### 실험 구성 (총 5가지)

| # | Feature | 목적 |
|---|--------|------|
| 3.1 | **Per-AU 단독** (41 AU × scalar) | 어느 AU가 가장 class-discriminative |
| 3.2 | **All-AU 41d** | AU-level 전체 accuracy (region 80%와 비교) |
| 3.3 | **Eye-AU group** (AU1,2,4,5,6,7 + L/R variants) | Jack 2012 예측 region |
| 3.4 | **Mouth-AU group** (AU10,12,14,15,17,20,22,23,24,25,26,27 + L/R) | AI 예측 region |
| 3.5 | **FACS canonical subset** (Happy=AU6+12, Sad=AU1+4+15, Anger=AU4+5+7+23) | Ekman 재현도 |

### Metric (Phase 0.1c와 동일 — 통일)
- **Primary: Linear probe accuracy** (3-fold CV, 30K stratified, L2 norm → Standardize → LogReg)
- Macro F1 per class
- **Per-class confusion matrix**: 어느 감정이 어느 AU로 구분되는가

### Baseline
- Random: 25.0%
- **Phase 0.1c Region baseline**: ConvNeXt POOLED 81.5%, MobileViT POOLED 80.2%
- AU-level이 region-level을 이기면 → AU 추출이 의미 있음

### Reproducibility
- Seed: 42
- CV: 3-fold
- Sample: 30K stratified

---

## 3. 결과 (실행 후 업데이트)

### Primary table — Per-AU 단독 ranking
(pending)

### All-AU vs Eye-group vs Mouth-group
(pending)

### FACS canonical 재현
(pending)

---

## 4. 해석 (실행 후)

### Finding 1 (예상): Mouth-AU group > Eye-AU group
- 예상: Mouth group 70-75%, Eye group 55-60%
- Jack 2012 직접 반박 → 논문 §4.3 Figure 2

### Finding 2 (예상): FACS canonical 부분 재현
- 예상: Happy (AU6+12)는 강하게 재현, Sad (AU1+4+15)는 약할 것
- 한국인 감정 표현의 "AU 문법"이 서양과 다름을 시사

### Finding 3 (예상): All-AU 41d vs Region POOLED
- 예상: 비슷한 accuracy (80% 내외)
- AU/region 층위가 다르지만 capacity는 비슷

---

## 5. 판정 (실행 후)

- [ ] GO Phase 1.1 (FACS canonical 재현도 상세 + §4.4 draft)
- [ ] GO Phase 1.2 (Jack 2012 재검증 Figure — eye vs mouth AU group § 4.3 draft)

---

## 6. Risks / Caveats

| Risk | Impact | Mitigation |
|-----|-------|-----------|
| OpenGraphAU가 서양 얼굴 편향 | AU 수치 편향 | KUFEC-II subset에서 ICC validation (별도) |
| 41 AU 일부는 sparse (AU32 rare) | linear probe 약화 | per-AU 분포 먼저 확인 |
| Subset group 구성 자의성 | 논문 방어력 | Ekman 원문 + Jack 2012 정의 인용 |

---

## 7. Paper section으로 이관 (pending)

---

## 8. Claude direction 평가 (실행 후)
