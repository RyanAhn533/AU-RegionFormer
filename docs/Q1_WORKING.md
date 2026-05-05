# Q1 논문 Working Workspace
## (pivot) Emotion Agent with Korean Cultural Priors

> **Status** (2026-04-22):
> - Q2 draft-ready (후배 1저자로 인계 확정, ESWA/PRL)
> - Q1 **agent angle로 pivot 중** (Moon 교수 관심 주제 + 졸업 thesis 중심)
>
> **Target** (revised):
> - 현실: ICMI / ACII / Sensors (IF 5-10 범위)
> - Stretch: IEEE TAFFC (IF 11)
> - Psychology co-author (연세대 컨소시엄) 확보 시 NHB/PNAS 확률 상승

---

## 🎯 Q1 새 direction (agent pivot, 2026-04-22)

### 한 줄 thesis (draft)
*"Culturally-aware multimodal emotion agent with bio-grounded labeling: 한국인 감정 맥락을 LLM reasoning loop에 주입한 agent."*

### 3 축

1. **Bio-grounded labeling** (JY 다른 서버 진행 중)
   - Bio signal → emotion label 자동 생성 agent
   - SGMT/S-PACE의 attention anchor를 label 생성으로 확장
   - KEMDy20 / K-EmoCon bio 데이터 활용

2. **Cultural priors** (우리 Q2 finding 흡수)
   - Jack 2012 재검증 결과 + 연세대 consensus = agent의 cultural context
   - "한국 사용자용 emotion agent"의 설득력 있는 근거

3. **Reasoning loop** (LLM 개입)
   - FER/SGMT perception → LLM agent reasoning → action/response
   - Jetson Orin DMS 같은 real deployment에 붙이면 killer demo

### Q2 자산 → Q1 흡수 map

| Q2 finding | Q1에서 쓰는 곳 |
|-----------|----------|
| 87.56% Triplet (iter 9) | §Perception module baseline |
| Yonsei consensus filter (iter 2) | §Reliability estimation |
| Jack 2012 multi-layer (iter 3, 11) | §Cultural prior 근거 |
| AU redundant finding (iter 4) | §Method choice 정당화 (AU 대신 Region 씀) |
| Demographic fairness (iter 5) | §Limitations |

---

## 다음 할 것 (직관 요약)

| 우선순위 | 내용 | 상태 |
|--------|-----|-----|
| 🥇 | Moon 교수에게 Q1 agent pivot + 연세대 공저 offer 상의 | 대기 (JY action) |
| 🥈 | Bio-grounded labeling 실험 (다른 서버) 현황 sync | JY 진행 중 |
| 🥉 | Q1 agent 논문 outline draft | 컨셉 확정 후 |
| - | 후배에게 Q2 material 인계 + writing guide | Moon 교수 확인 후 |

---

**구 thesis (`AI vs Psychology gap` 중심, FER-only)는 Q2로 내려감**. Q1은 agent 쪽으로 상위 기획.

---

## ⚡ 대원칙 (매 실험 전 확인)

**"모든 실험은 Q1 논문의 어떤 section을 채우는가?"**

독립 사이드 프로젝트 금지. 모든 실험은 다음 중 하나에 반드시 매핑:
- **§3 Methods** (방법 서술용)
- **§4 Results** (주요 finding)
- **§5 Discussion / Limitations** (해석 또는 reviewer 공격 방어)
- **Figure 1-N** (논문 핵심 시각화)

매 실험 시작 전 **"이 실험은 논문 §X의 무엇을 채우나?"**를 1줄로 답하지 못하면 실험 취소.

### Blind side 실험 = Reviewer 공격 선제 방어
| JY가 제안한 blind side | 방어하는 reviewer 공격 | 들어갈 section |
|-------------------|-------------------|----------|
| Jack 2012 재검증 | "왜 한국만? East Asian 일반화?" | §4.3 (main finding) |
| Py-Feat 한국인 validation | "서양 편향 도구 쓴 거 아니야?" | §3.2, §5.4 limitations |
| 영역 크기/위치 분석 | "mouth가 큰 region이라 유리한 거 아냐?" | §4.5, §5.1 H1 |
| is_selected subset | "사회적 합의가 AI 관점 지지한다는 증거?" | §4.5, §4.5 |
| FACS canonical 검증 | "Ekman과 unmatch면 이론 무시한 거 아냐?" | §4.4 |
| 4emo_before/after attention | "필터링으로 cherry-pick한 거 아냐?" | §4.1, §5.4 |

모든 blind side 보완 실험은 **§4 Results 또는 §5 Discussion**에 들어간다.

---

## 🚦 전체 진행도

```
Phase 0 (진단)           [█████░░░░░] 50%  — 0.1+0.1c 완료, 0.2 대기
Phase 1 (AU 지도)        [░░░░░░░░░░]  0%
Phase 2 (사회적 합의)    [░░░░░░░░░░]  0%
Phase 3 (Graph 학습)     [░░░░░░░░░░]  0%
Phase 4 (Cross-cultural) [░░░░░░░░░░]  0%
Paper Draft              [███░░░░░░░] 20%  — §4.1 linear probe 수치 확정
```

---

## 📄 논문 Outline (Live Drafting)

### Title (후보)
1. **"The AI-Psychology Gap in Korean Facial Emotion: A Large-scale AU Analysis"** ⭐
2. "When AI sees the mouth but psychology expects the eyes: Culturally-situated FER in 237K Koreans"
3. "Revisiting Jack (2012) with large-scale data: Does East Asian emotion really rely on eyes?"

### Abstract (skeleton — 완성은 Phase 2 이후)
```
[Background]      Jack et al. (2012) showed East Asian faces express emotion
                  through eyes while Western faces use mouth. Modern FER models,
                  however, predominantly attend to the mouth region.

[Gap]             Does this AI-psychology gap persist at large scale in Korean
                  data? If so, which view does social consensus (human agreement)
                  support?

[Method]          [PLACEHOLDER — Phase 3 이후 업데이트]
                  - 237K Korean emotion images with 3-annotator + 298-person verification
                  - Py-Feat + OpenGraphAU AU intensity extraction
                  - Consensus-aware AU graph learning

[Findings]        [PLACEHOLDER — Phase 1-3 이후]
                  - AI region ranking: mouth >> cheek > nose > forehead > eyes
                  - FACS canonical mapping partially held (AU6+12 for happy: YES/NO)
                  - Social consensus aligns with: [AI / Psych / neither]
                  - Consensus-aware learning improves hurt/anxious/sad F1 by X%p

[Implication]     [PLACEHOLDER — 완성 후]
```

### 1. Introduction
**상태**: 🟡 아이디어 정리됨
- [ ] Jack 2012 선행 문제 제기
- [ ] East Asian emotion 이론의 시대 (1970s-2020s)
- [ ] 딥러닝 FER 모델의 mouth 편향 (선행연구에서 반복 확인됨)
- [ ] **Gap**: 대규모 실증 데이터에서 이 괴리가 정말 존재하는가? 사회적 합의는 어느 쪽?
- [ ] Contribution 3가지 (descriptive/social/methodological)

### 2. Related Work
**상태**: 🟢 references 파일 축적 중

See: [`docs/research_log/references/au_region_importance_2026-04-21.md`](research_log/references/au_region_importance_2026-04-21.md)

- 2.1 Cross-cultural emotion (Jack 2012, Yuki 2007, Li 2025 review, Matsumoto display rules)
- 2.2 FACS and AU-based FER (Ekman 1978, ME-GraphAU 2022, ANFL 2023)
- 2.3 Korean emotion datasets (KUFEC-II 2017, AI Hub)
- 2.4 Consensus-aware / noisy label learning (gap!)
- 2.5 Graph neural networks for FER (ME-GraphAU, FG-Net 2024)

### 3. Methods
**상태**: 🔴 Phase 3 완료 후 작성

- 3.1 Data
  - 237K AI Hub Korean emotion dataset (7→4 emotion filtered)
  - 3-annotator + 298 Yonsei verification
  - Demographic metadata
- 3.2 AU extraction
  - Py-Feat 20 AU (primary) / OpenGraphAU 41 AU (comparison)
  - Korean validation (KUFEC-II)
- 3.3 Analysis
  - Per-AU linear probe accuracy (AU 하나씩 단독 분류 정확도)
  - Jack 2012 재검증: eyes-AU vs mouth-AU subset
- 3.4 Consensus-aware graph learning
  - [PLACEHOLDER — Phase 3 설계 후]

### 4. Results
**상태**: 🟢 Phase 0.1c (linear probe) 확정 수치

- **4.1 Region-level class-discriminative power** (Phase 0.1c)
  - 🧭 **용어 가이드**: 
    - **Linear probe accuracy** = embedding 위에 간단 분류기 얹어서 감정 맞추는 정확도(%). **높을수록 좋음**. 무작위 추측 = 25% (4-class)
    - **Macro F1** = 감정별 정확도 평균 (class 불균형 보정). 0~1, 1이 완벽
    - (버린 metric) silhouette = 클러스터 분리 점수, 0~1. 고차원에서 편향되어 사용 중단
  - Random baseline = 25% (4-class)
  - **ConvNeXt POOLED 8-region: 81.5% ± 0.3** (Macro F1 0.815) ← best
  - **MobileViT POOLED 8-region: 80.2% ± 0.2** (F1 0.801) — v1 학습모델(79.7%) 수준 재현
  - **Mouth (AI top)**: ConvNeXt 75.2%, MobileViT 71.9%
  - Nose: ConvNeXt 75.0%, MobileViT 68.7%
  - **Eyes (Jack 2012 predicted top)**: ConvNeXt ~61%, MobileViT ~58% — **mouth보다 낮음** ⚠️
  - **Forehead (worst)**: ConvNeXt 45%, MobileViT 39% — 약 20%p 낮음
  - → **논문 Figure 1**: region importance bar chart 확보
- ⚠️ **초기 실험(버린 것)**: silhouette score(클러스터 분리도, 0~1)로 0.007 나와서 "분리 안 됨" 오판. 원인: 고차원 + L2 정규화 누락으로 metric 자체가 부적절. Linear probe(분류 정확도)로 재측정 → 80%+. **Embedding은 원래 충분히 좋았음.**

- 4.2 AU-level analysis (Phase 0.3 pending — OpenGraphAU 41 AU 추출 + per-AU linear probe)
- 4.3 **Jack 2012 재검증** (Phase 1.2 pending — eyes-AU group vs mouth-AU group accuracy 비교)
- 4.4 FACS canonical validation (Phase 1.1 pending — Happy=AU6+12 한국인 재현?)
- 4.5 Social consensus analysis (Phase 2 pending)
- 4.6 Cross-cultural AU dialect (Phase 4 pending)

### 5. Discussion
**상태**: 🔴 Results 이후

- 5.1 AI-Psychology gap — why does it exist?
  - Hypothesis 1: mouth has higher physical variability (testable from our data)
  - Hypothesis 2: Training data distribution bias
  - Hypothesis 3: Cultural display rules vs expression rules
- 5.2 Implication for FER model design
- 5.3 Implication for cross-cultural psychology
- 5.4 Limitations
  - Py-Feat bias (mitigated by KUFEC-II validation)
  - 298명 annotators = college students (generalization?)
  - 7-emotion vs 4-emotion filtering decision

### 6. Conclusion
(Results 확정 후)

---

## 🎯 Key Experiments → Paper Section 매핑

각 실험이 완료되면 해당 section의 PLACEHOLDER가 채워진다.

| Experiment | → Paper Section | Status |
|-----------|----------------|--------|
| Phase 0.1 region diag | §4.1 (완료) | ✅ |
| Phase 0.2 Py-Feat AU | §3.2, §4.2 | 🏃 진행 중 |
| Phase 0.3 per-AU analysis | §4.2 | 대기 |
| Phase 1.1 FACS canonical | §4.4 | 대기 |
| Phase 1.2 Jack 2012 재검증 | **§4.3 (핵심)** | 대기 |
| Phase 1.3 region size/position | §4.5, §5.1 H1 | 대기 |
| Phase 1.4 is_selected subset | §4.5 | 대기 |
| Phase 2.1-2.3 social | §4.5, §5.3 | 대기 |
| Phase 3 consensus-aware | §3.4, §4.7 | 대기 |
| Phase 4 cross-cultural | §4.6 | 대기 |

---

## ❓ Open Questions (답이 필요한 것)

| # | Question | 누가/어떻게 답 | 영향 |
|---|---------|-------------|------|
| Q1 | 298명 연세대 실험 IRB/pre-reg 상태? | 담당 교수 문의 | NHB 가능성 |
| Q2 | Psychology co-author 영입 가능? | JY 결정 + 섭외 | NHB 가능성 |
| Q3 | Py-Feat 한국인 정확도 > 80%? | KUFEC-II validation (Phase 0.2b) | 전체 결과 신뢰도 |
| Q4 | "한국인" = East Asian 일반화 타당? | Phase 4 KUFEC-II + Jpn/Chi 비교 시 | Jack 2012 재검증 해석 |
| Q5 | Consensus-aware가 ME-GraphAU 대비 SOTA? | Phase 3 실험 | Method novelty |

---

## 🔜 Next Actions (바로 다음)

### 이번 주
1. Py-Feat env dependency 안정화 + dry-run 100 이미지
2. 성공 시 237K full run (6-8h bg)
3. 실패 시 LibreFace 또는 OpenGraphAU 대체

### 다음 주
4. Phase 0.3 per-AU linear probe (AU 하나씩 accuracy)
5. Phase 1.1 FACS canonical 검증
6. Phase 1.2 Jack 2012 재검증 (⭐ paper §4.3)

### 3주 내
7. Phase 1.3, 1.4 완료 → 논문 Figure 1-5 draft
8. Phase 2 설계 시작

---

## 💼 Q2 분리 후보 (중간 단계에서 뽑을 논문)

Phase 중 하나를 **독립 논문**으로 분리. Q1과 겹치지 않게 feature/method 다른 층위로 설정.

| 후보 | Phase subset | Q2 thesis | Target (Q2급) | 난이도 | Q1과 겹침 |
|------|-------------|---------|-------|------|---------|
| **A. AU Detector Benchmark** | 0.2+0.2b | "Py-Feat vs OpenGraphAU vs LibreFace on Korean faces" | ESWA / IEEE Access | 중 | 없음 ✓ |
| **B. Landmark × Geometric FER** | 0.4 + 1.3 subset | "Geometric landmark features for Korean FER: classical ML vs DL" | ESWA / PR Letters | 낮 (ready) | 없음 ✓ |
| **C. Region size/position bias** | 1.3 | "Spatial bias in facial AU attribution — does region size drive accuracy?" | PR Letters | 중 | 부분 ⚠️ |
| **D. Label quality via crowd** | 1.4 + cleanlab | "Multi-source noise detection: 3-annotator + crowd (298명) + cleanlab" | IEEE Access / Behavior Research Methods | 중 | 부분 ⚠️ |
| **E. Demographic × AU** | 2.3 | "Age/Gender effects on AU expression in Korean emotion" | Frontiers in Psychology | 낮 | 약함 ✓ |

**추천**: A (AU detector benchmark) 또는 B (Landmark). 둘 다 Q1과 완전 분리됨.

**결정 시점**: Phase 0.2 종료 후 (AU 추출 결과 보고 A/B 사이 결정)

---

## 🧠 Claude 논리 강화 시스템 (RLHF 데이터 축적)

**목적**: JY-Claude 상호작용의 논리 흐름을 +/− 평가로 기록 → 향후 Claude 모델 강화학습 데이터.

### 두 가지 기록 병행

| 파일 | 단위 | 평가 | 용도 |
|-----|-----|-----|-----|
| `directions.jsonl` | 단일 제안 | hindsight_score (1-10) | Direction 품질 |
| `logic_chains.jsonl` | reasoning sequence | step별 +/-/~ | **논리 흐름 강화** |

### Logic Chain 기록 포맷

```json
{
  "chain_id": "LC###",
  "trigger": "JY 요청 원문",
  "steps": [
    {"n": 1, "reasoning": "...", "action": "...", "outcome": "...", "sign": "+"},
    {"n": 2, "reasoning": "...", "action": "...", "outcome": "...", "sign": "-"}
  ],
  "final": "success | partial | failure",
  "jy_feedback": "원문",
  "lesson": "다음엔 이렇게",
  "good_pattern": "어떤 논리가 좋았나",
  "bad_pattern": "어떤 논리가 틀렸나",
  "dpo_extractable": true
}
```

### + / − 평가 기준

| Sign | 의미 |
|-----|-----|
| **+** | Step이 결과 개선에 기여 (JY 수용, 객관 근거 확보, 구조화 성공) |
| **−** | Step이 시간낭비/오류/혼선 초래 (JY 거부, 용어 오용, 방향 오인) |
| **~** | 중립 (더 관찰 필요) |

### 오늘 세션까지 축적된 chain (6개)

- **LC001**: ChatGPT 6단계 로드맵 리뷰 (partial, 용어 혼용 실수)
- **LC002**: 선행연구 서치 후 thesis 업그레이드 (success, 재설계 실패 1회)
- **LC003**: Q1 확률 재평가 (full success, 모든 step +)
- **LC004**: AU/region 정정 (success, correction 후 회복 모범)
- **LC005**: 문서 파편화 지적 → working doc 전환 (partial, 정적 HTML 실수)
- **LC006**: Py-Feat 설치 (partial, deps 순차 설치 실수)

### RLHF 적용 가능 pattern (이번 세션에서 도출)

**Good (chosen)**:
- 3인 페르소나 토론 + 구체 수치 + 폐기/수정/수용 구분 (LC001)
- WebSearch 선행연구 먼저 + 실험 결과를 선행연구와 대조 (LC002)
- Venue별 % + 조건별 상승폭 + 현실/dream 구분 (LC003)
- 즉시 admit + 차이 표 + 로드맵 diff (LC004)

**Bad (rejected)**:
- 도메인 용어의 세밀한 구분 생략 (LC001 step 5)
- 사용자 의도 미확인 상태에서 정적 HTML 생성 (LC005 step 3)
- deps 하나씩 순차 설치 (LC006 step 5)

---

## 🧭 Decisions Log (Q1 디렉팅 결정 기록)

**이 섹션은 논문 쓸 때 "왜 이렇게 했나" 답변 근거.**

| 날짜 | 결정 | 대안 | 이유 |
|-----|-----|-----|-----|
| 2026-04-21 | Thesis = AI-psych 괴리 | "한국형 관계 구조" | Jack 2012 후속으로 counterintuitive finding |
| 2026-04-21 | Target = TAFFC 먼저 | NHB 직행 | psych co-author 없음, 리스크 분산 |
| 2026-04-21 | 8-node AU graph 폐기 | 유지 | reviewer 공격 포인트, Set Transformer과 구분 X |
| 2026-04-21 | Sample-level 237K graph | AU graph 단독 | 노드 수 확보 + kNN propagation 의미 생김 |
| 2026-04-21 | Py-Feat primary | OpenFace / LibreFace | pip 설치, 표준, 빠름 |
| 2026-04-21 | 석사 Q2 선정 deferred | 즉시 분리 | GNN 실험 부산물로 결정 |

---

## 📊 Risk Registry

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | Py-Feat 한국인 편향 | High | High | KUFEC-II validation (Phase 0.2b), OpenGraphAU 병행 |
| R2 | 298명 IRB 없음 → NHB 불가 | Medium | High (NHB) | 담당 교수 확인, TAFFC로 fallback |
| R3 | AU-level도 accuracy 낮으면? (random 25% 근처) | Medium | Critical | AU-RegionFormer v3 finetuned backbone embedding 시도 |
| R4 | ME-GraphAU 대비 method novelty 불충분 | Medium | Medium | Consensus-aware 조합 강조 |
| R5 | AffectNet이 "Western" 아닌 혼합 | High | Medium | KUFEC-II + CK+ + JAFFE 별도 curation |

---

## 💬 Claude 사용법

**매 세션 시작 시**:
1. 이 파일을 **first read**
2. 🚦 진행도와 🔜 Next Actions 확인
3. 관련 Phase의 experiment 파일로 이동

**매 실험 완료 시**:
1. Results (§4) 해당 section 업데이트
2. Decisions log에 추가
3. Open questions 갱신
4. 진행도 바 조정

**매 세션 종료 시**:
1. "Last update" 날짜 갱신
2. `docs/research_log/sessions/` 별도 세션 파일 저장
3. `directions.jsonl` 에 Claude direction 기록
