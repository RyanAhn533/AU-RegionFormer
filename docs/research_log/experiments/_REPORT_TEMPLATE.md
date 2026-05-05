# Phase X.Y — (실험 이름)

> **논리 전개의 한 단위.** 이 실험은 이전 실험에서 나온 질문에 답하고, 다음 실험의 필요성을 정당화한다.

---

## 0. 이전 단계와의 연결

이전 Phase에서 나온 질문: **"..."**
이 실험이 답하려는 것: **"..."**

(이 실험이 독립적으로 왜 필요한지 1-2줄로 정당화. 없으면 실험 취소.)

---

## 1. 목적

### Primary question
**"..."** (yes/no 또는 quantity로 답 가능해야 함)

### Paper section 매핑
- 이 실험 결과가 논문 §X.X를 채움
- Figure N이 됨 (있으면)

### 성공 조건 (numeric threshold)
| 조건 | 다음 액션 |
|-----|---------|
| 수치 > A | GO (다음 Phase) |
| 수치 ∈ [B, A] | MARGINAL (추가 검증) |
| 수치 < B | NO-GO (pivot or abort) |

---

## 2. 방법

### Data
- 규모:
- 출처:
- Preprocessing:

### Metric (명확히)
| 지표 | 계산 | 해석 가능한 범위 |
|-----|-----|--------------|
| Primary | ... | Random baseline 대비 의미 있는 차이 |
| Secondary | ... | |

### Baseline
- Random: X%
- 선행 기법: Y%
- Oracle (있으면): Z%

### Reproducibility
- Seed: 42
- CV folds: N
- Sample size: N

---

## 3. 결과 (명확한 수치)

### 핵심 표
| Config | Primary metric | Secondary | vs Random |
|--------|--------------|----------|----------|
| ... | **XX.X% ± Y** | ... | +Np (유의?) |

### 그림
- `path/to/figure1.png` — what it shows
- `path/to/figure2.png` — what it shows

---

## 4. 해석

### Finding 1: ...
(수치가 어떤 의미인가. 선행연구 대비 어떻게 놓이는가.)

### Finding 2: ...

### 예상과 맞았나
| 기대 | 실제 | 차이 |
|-----|-----|-----|
| ... | ... | ... |

### 선행연구와의 관계
- Jack 2012 / FACS / ME-GraphAU 등과의 일치/불일치

---

## 5. 판정

- [ ] **GO** — 다음 Phase 진입 (reason: ...)
- [ ] **MARGINAL** — 추가 검증 후 결정 (action: ...)
- [ ] **NO-GO** — pivot 또는 abort (new direction: ...)

### 다음 실험 justification
이 결과가 다음 실험(Phase X.Z)을 **왜** 필요하게 만드는가?

---

## 6. Risks / Caveats

| Risk | Impact | Mitigation |
|-----|-------|-----------|
| ... | ... | ... |

---

## 7. Paper section으로 이관

### §X.X에 들어갈 문장 (draft)

> 실험 결과 한 단락 (논문용).

### §X.X의 Figure/Table

- Table N: ...
- Figure N: ...

---

## 8. Claude direction 평가

- Direction ID (디렉션 받은 근원): Dxxx
- Logic chain ID: LC###
- hindsight_score (실행 후 1주 후 JY 평가): /10
- lesson: ...
