# AU-RegionFormer — 전체 전략 (Single Source of Truth)

> **이 문서 = 모든 전략의 마스터 파일.** 매번 여기서 시작. 다른 문서(세션 로그, experiment 파일)는 detail.

**Last updated**: 2026-04-22
**Status**: Q2 draft-ready (후배 1저자로 인계) · Q1 agent pivot 준비 중

---

## ✅ 2026-04-22 — 단계 정리

### Q2 (Korean FER multi-view fusion) — **Draft-ready, 후배에게 인계**

| 항목 | 값 |
|-----|-----|
| Best metric | **87.56%** linear probe (Region + Landmark + kNN graph, clean subset) |
| 비교 기준 | v1 학습모델 79.7% **+7.86%p** |
| Per-class F1 | 전 class > 0.83 (기쁨 0.949 / 분노 0.840 / 슬픔 0.835 / 중립 0.869) |
| Ablation stack | Region → +kNN → +Landmark → +Yonsei clean 다단 누적 (+5.97%p mixed) |
| Multi-layer Jack counter-evidence | Region +13.6%p / AU +1.0%p / Landmark +10.3%p — 3 layers 일관 |
| Figure/Table 준비 | ✅ (exp_003 per-region, exp_005 fairness, exp_011 landmark ranking, exp_010 ablation) |

**저자 구조 (계획)**: 석사 후배 1저자 · JY 2저자 · 연세대 컨소시엄 측 공저 · Moon 교수 교신
**Target venue**: ESWA (IF 7.5) 또는 PRL (IF 5.1) — 후배 1저자 실적 중심 선택
**남은 작업**: writing, 연세대 공저 포함 협의 (Moon 교수 경유)
**기여한 10 iterations (iter 1-11) 자산 그대로 인계 가능**: `state/leaderboard.jsonl`, `experiments/exp_NNN/`, `docs/research_log/references/`

### Q1 (JY 1저자, agent angle) — **Pivot 진행 중**

Q2에서 확정된 자산을 Q1 논문의 preliminary/related work로 **흡수**.
새 Q1 thesis 축:
1. **Bio-grounded emotion labeling agent** (JY 다른 서버에서 준비 중)
2. **Cultural-aware emotion agent** (Jack 2012 × 한국인 데이터)
3. **SGMT/S-PACE backbone 확장** (multimodal + LLM reasoning loop)

**장점**:
- Moon 교수 관심 주제(agent)와 직접 연결 → 지원 확보
- Jetson Orin DMS 배포 경험 killer demo
- 연세대 공저 확보 시 psychology angle 자동 강화 (NHB/PNAS 자격 ↑)

---

---

## 🎯 1. Thesis (한 줄)

> **"한국인 감정 표현에서 AI 모델이 포착한 핵심 region은 심리학 이론(Jack 2012)이 예측한 region과 괴리가 있다. 이 괴리는 AU 수준에서 어떻게 나타나며, 298명의 사회적 합의는 어느 쪽에 가까운가?"**

### Counterintuitive finding (논문의 핵심 한 줄)
- **Jack 2012 PNAS**: East Asian은 **eyes 중심**으로 감정 표현
- **JY Phase 0.1c 결과** (linear probe): AI 모델은 **mouth가 top (75%)**, eyes는 상대적으로 낮음 (57-61%)
- Forehead는 가장 약함 (39-45%) — 상단 face 전체가 AI에겐 약한 신호
- → **정량화된 괴리**. 이게 논문 Figure 1.

---

## 🎯 2. Target Venue

| 우선순위 | Venue | IF | 확률 | 역할 |
|--------|-------|-----|------|-----|
| **1** | **IEEE TAFFC** | 11 | **50-60%** ⭐ | **현실 메인 타겟** |
| 2 | Information Fusion | 18 | 25-30% | method 강조 시 |
| 3 | PNAS Social Sciences | 9.4 | 20-25% | psychology angle 강조 |
| 4 | Nature Human Behaviour | 21 | 15-20% | dream (6주 더 투자 시 40-45%) |
| 5 | Nature Communications | 16 | 10-15% | interdisciplinary |
| safety | Scientific Reports | 3.8 | 60-70% | 안전망 (Q1) |

### NHB/PNAS로 올라가려면 (아직 미충족)
1. 심리학 co-author 영입 (연세대 담당 교수)
2. 298명 실험 IRB/pre-registration 문서화
3. Py-Feat 한국인 AU detection validation (KUFEC-II 활용)
4. Korean vs Western 엄밀 curation

---

## 🎯 3. Novelty 3축

1. **Descriptive** (Phase 1): 대규모 한국인 데이터에서 AI vs Psychology region 괴리 정량화
2. **Social** (Phase 2): 사회적 합의(298명)가 AI 관점 vs 심리학 관점 중 어느 쪽 지지?
3. **Methodological** (Phase 3): Consensus-aware AU graph learning

---

## 🎯 4. 로드맵 (Phase 0-4)

### 전체 흐름

```
Phase 0 (진단) ─┬─→ Phase 1 (한국형 AU 지도) ─→ Phase 2 (사회적 합의) ─┐
              │                                                      ├─→ 논문 submit
              └─→ Phase 3 (Graph learning) ─────────────────────────→┤
                            ↑                                        │
              Phase 4 (Cross-cultural) ←───────────────────────────────
```

### Phase별 상세

| Phase | 기간 | 목적 | 핵심 실험 | 상태 |
|-------|------|------|---------|-----|
| **0. 진단 + AU 기초** | 3-4주 | Graph 할 자격 + AU 추출 | 0.1 region✅, 0.2 Py-Feat AU, 0.3 per-AU analysis, 0.4 landmark Fisher, 0.5 v5 postmortem | **0.1 완료** |
| **1. 한국형 AU 지도** (descriptive) | 4주 | Figure 1~5 생성 | FACS canonical 검증, **Jack 2012 재검증**, 영역 크기/위치, is_selected subset | 대기 |
| **2. 사회적 합의 구조** (social) | 4주 | AI vs Psych 괴리를 298명이 어느쪽 지지 | annotator disagreement × AU, 298명 decision tree, demographic × AU | 대기 |
| **3. Graph learning** (methodological) | 6주 | Method novelty | AU graph (41 node), Sample graph (237K), Hetero graph, **Consensus-aware** | 대기 |
| **4. Cross-cultural** (defensive) | 6주 | NHB/PNAS 방어 | AffectNet AU 추출, dialect matrix, Jack 재검증 on Western | 대기 |
| 5. 시뮬레이터 | — | **별도 논문** (KMER multimodal) | — | 제외 |

---

## 🎯 5. 용어 정의 (중요 — 혼용 금지)

| 용어 | 의미 | 층위 | 실험 예시 |
|-----|-----|-----|---------|
| **Region** | 얼굴 공간 영역 (eyes, mouth, nose...) | 공간 | Phase 0.1의 8 region embedding |
| **AU (FACS)** | 얼굴 근육 단위 움직임 (AU1~46) | 근육 | Phase 0.2의 Py-Feat AU intensity |

- "AU-RegionFormer" 프로젝트 이름 = 실제로는 **region-based** (AU 단위 아님)
- Jack 2012의 "eyes vs mouth" = region 층위
- Ekman FACS의 "AU6+12 = Happy" = AU 층위
- 두 층위는 **many-to-many** 관계

---

## 🎯 6. 선행연구 (핵심만)

| 논문 | 한줄 | 우리와 관계 |
|-----|-----|----------|
| Jack 2012 PNAS | East Asian = eyes, Western = mouth | **직접 재검증 대상** (Phase 1.2) |
| Ekman FACS | AU canonical mapping (Happy=6+12 등) | Phase 1.1 검증 기준 |
| KUFEC-II (Korea Univ 2017) | 한국인 FACS stimuli 이미 있음 | Phase 4에서 Western vs KUFEC-II 비교 |
| Li 2025 SPPS | 문화×FER systematic review | Phase 4 theoretical framework |
| ME-GraphAU (ECCV'22) | AU graph SOTA | Phase 3 graph baseline 비교 |
| Koreans vs Americans (Hogrefe 2022) | 한국인은 외적 감정에 덜 가중치 | Phase 2 social consensus 근거 |

**상세**: `docs/research_log/references/au_region_importance_2026-04-21.md`

---

## 🎯 7. 논문 Packaging

| # | 제목 | 타겟 | Phase 활용 | Author |
|---|------|------|---------|------|
| **1 (main)** | AI vs Psychology in Korean facial emotion: large-scale AU analysis | **TAFFC** / NHB (dream) | 1+2+4 | JY 1저자 |
| **2 (method)** | Consensus-aware AU graph learning for culturally-situated FER | TAFFC / Information Fusion | 3+4 | JY 1저자 |
| **3 (석사 Q2)** | Landmark × Region benchmark for Korean FER | ESWA / PR Letters | 0.4+1.3 | 석사 1저자 |
| **4 (별도)** | KMER multimodal driver emotion | ACM MM / ICMI | Phase 5 (별도) | 미정 |

---

## 🎯 8. Q1/Q2 전략

### JY (Q1)
- **Feature space**: CNN/AU embedding + graph
- **Contribution**: AI-psych 괴리 + consensus-aware method
- **Target**: TAFFC main (논문 1+2)

### 석사 (Q2)
- **Feature space**: Landmark 17 feature (geometric) OR AU detector benchmark
- **Contribution**: Korean FER benchmark OR detector 정확도 비교
- **Target**: ESWA / PR Letters
- **선정 시점**: Phase 0.2-0.3 결과 후 (GNN 실험 부산물 보고 결정)

---

## 🎯 9. 현재 상태 (스냅샷)

### ✅ 완료
- [x] Phase 0.1 Region embedding 진단 (silhouette) → metric 오선택 판정
- [x] **Phase 0.1c Region embedding linear probe** → POOLED 80-81%, mouth 72-75%, eyes 58-62%, forehead 39-45% (진짜 baseline)
- [x] 선행연구 조사 (Jack 2012, FACS, KUFEC-II, Li 2025)
- [x] Thesis 업그레이드 (2026-04-21)
- [x] 연구 일지 시스템 구축

### 📋 대기 (승인 필요)
- [ ] Phase 0.2 Py-Feat AU 추출 (환경 설치 + 237K 추출)
- [ ] OpenGraphAU 부가 실험 여부 결정 (D012)

### 🔜 이어서
- Phase 0.3 AU-level analysis → Phase 1.1/1.2 (Jack 재검증)

### ⚠️ 블라인드 스팟 (해결 필요)
1. Py-Feat 서양 얼굴 편향 → 한국인 validation 필요 (KUFEC-II 활용 검토)
2. 298명 실험 IRB/pre-reg 상태 불명 → NHB 도전 시 연세대 담당 교수 확인
3. AffectNet이 "Western"이 아닌 혼합 → Korean vs Western curation 필요

---

## 🎯 10. 핵심 수치 (외워야 할 것)

### Phase 0.1 결과 (linear probe 기준, 2026-04-21 재측정)

**측정**: Linear probe accuracy = embedding + 로지스틱 회귀로 감정 4-class 맞추는 정확도(%). 높을수록 좋음. 무작위 추측 = 25%.

```
기준선 (무작위 추측):       25.0%

ConvNeXt POOLED 8-region:  81.5% ± 0.3    ← 최고 (v1 학습모델 초과)
MobileViT POOLED:          80.2% ± 0.2    — v1 학습모델(79.7%) 수준 재현
Mouth  (AI 관점 top):      CNX 75.2% / MVT 71.9%
Nose:                      CNX 75.0% / MVT 68.7%
Eyes   (Jack 2012 예측):   CNX 61.5% / MVT 57.7%   ← mouth보다 13%p 낮음
Forehead (최약):           CNX 45.0% / MVT 39.0%

⚠️ 초기 실험(버린 것): silhouette score(클러스터 분리 점수)로 0.007 → "분리 안됨" 오판.
   원인: 고차원(6144-8192d) + L2 정규화 누락으로 metric 자체가 부적절.
   Linear probe가 표준. 앞으로 전 Phase 통일.
```

### 기존 자료
- AU cosine sim 0.97-0.99 (Phase 0.1 재확인)
- 모델 v1→v3: 80.7% 벽
- annotator 일치: 기쁨 96% > 중립 74% > 분노 51% > 슬픔 48%
- 혼동 비대칭: 슬픔→상처 11.9%, 역방향 0%
- 연세대 298명 inter-rater: 98.7%
- soft label v5: F1=0.538 실패

---

## 🎯 11. 데이터/코드 위치 (빠른 참조)

| 자원 | 위치 |
|-----|-----|
| 원본 이미지 (237K) | `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/` |
| Region embedding | `AU-RegionFormer/data/label_quality/au_embeddings/` |
| Landmark features | `AU-RegionFormer/data/label_quality/face_features.csv` |
| 모델 ckpt | `AU-RegionFormer/outputs/{v2,v3,4emo_*,v5_softlabel}/` |
| Phase 0.1 결과 | `AU-RegionFormer/outputs/phase0/01_au_embedding_diag/` |
| **전략 (이 파일)** | `AU-RegionFormer/docs/research_log/STRATEGY.md` |
| 세션 로그 | `AU-RegionFormer/docs/research_log/sessions/` |
| 실험 로그 | `AU-RegionFormer/docs/research_log/experiments/` |
| 선행연구 | `AU-RegionFormer/docs/research_log/references/` |
| Claude direction | `AU-RegionFormer/docs/research_log/claude_evaluation/directions.jsonl` |

---

## 🎯 12. 이 문서 유지 규칙

- **전략 변경 = 이 파일 수정 + git commit** (향후 연구 사이클에 한해)
- **Phase 완료마다 Section 9 (현재 상태) 업데이트**
- **Venue 확률 변경마다 Section 2 업데이트**
- **논문 packaging 변경마다 Section 7 업데이트**
- **Claude는 세션 시작 시 이 파일을 먼저 읽음** (다음 세션 시작할 때 우선 로드)
