---
date: 2026-04-21
topic: AU/Region importance in FER, cross-cultural facial expression
related_phase: Phase 0-4 전반
---

## Purpose
한국인 감정 인식에서 어느 얼굴 부위/AU가 중요한지 확인. 삽질 방지 + novelty gap 설계용.

---

## 1. Jack et al. 2012 PNAS — **Facial expressions of emotion are not culturally universal**
- **URL**: https://www.pnas.org/doi/10.1073/pnas.1200155109
- **Core claim**: East Asian observers는 **eyes** 중심으로 감정 강도 표현 (특히 happy, fear, disgust, anger). Western Caucasian은 **mouth**가 더 informative.
- **Method**: 30 Western + 30 East Asian, generative grammar + 얼굴 애니메이션으로 mental representation 재구성
- **Key evidence**: 동아시아 이모티콘 (^.^) 웃음 vs 서양 :) 웃음 — 동아시아는 눈으로, 서양은 입으로
- **Limitation**: N=30 each (small), lab-generated stimuli
- **JY 관련성**:
  - JY Phase 0.1 결과와 **정면 괴리** (AI 모델은 mouth 유리)
  - 이 괴리 자체가 **논문 thesis**가 될 수 있음
  - Phase 1.2에서 직접 재검증 실험 가능

## 2. Ekman & Friesen — **FACS** (Facial Action Coding System)
- **URL (reference)**: https://py-feat.org/pages/au_reference.html, https://en.wikipedia.org/wiki/Facial_Action_Coding_System
- **Canonical AU mapping to basic emotions**:
  - **Happiness**: AU6 (cheek raiser) + AU12 (lip corner puller)
  - **Sadness**: AU1 (inner brow raise) + AU4 (brow lowerer) + AU15 (lip corner depress)
  - **Anger**: AU4 + AU5 (upper lid raise) + AU7 (lid tighten) + AU23 (lip tighten)
  - **Disgust**: AU9 (nose wrinkle) + AU15 + AU16
  - **Fear**: AU1 + AU2 + AU4 + AU5 + AU7 + AU20 + AU26
  - **Surprise**: AU1 + AU2 + AU5 + AU26
- **JY 관련성**: Phase 1.1에서 한국인 Happy 데이터에서 AU6+12가 실제 top discriminative인지 검증

## 3. Lee et al. 2017 — **KUFEC-II** (Korea University Facial Expression Collection)
- **URL**: https://www.frontiersin.org/articles/10.3389/fpsyg.2017.00769/full
- **Core**: 한국인 얼굴 감정 stimuli dataset, Ekman FACS 기반
- **JY 관련성**:
  - 이미 한국인 AU 데이터가 있다 → JY 237K 데이터의 novelty는 **scale + 사회적 합의(298명) + in-the-wild**
  - KUFEC-II는 통제된 posed expression, JY는 AI Hub 다양한 소스
  - Phase 4 cross-cultural에서 KUFEC-II를 Western dataset과 같이 비교 포인트로 활용 가능

## 4. Li et al. 2025 SPPS — **Why Do Cultures Affect Facial Emotion Perception? Systematic Review**
- **URL**: https://journals.sagepub.com/doi/10.1177/00220221251334811
- **Core**: 문화가 FER에 영향 주는 메커니즘 systematic review (2025 최신)
- **Key findings**:
  - 104 cultural accents in facial expression production
  - Collectivist culture = inhibited expression (gaze aversion, head down)
  - Core AU는 consistent but culture-specific 변이 존재
- **JY 관련성**: Phase 4 cross-cultural의 이론 framework로 직접 인용

## 5. FER 딥러닝 (attention mechanism)
- Perception CNN (PCNN, 2025 arXiv): 5-parallel networks for eyes, cheeks, mouth → FER2013, CK+, RAF-DB, FERPlus SOTA
- Mini-Xception+CBAM: eyes/eyebrows 중심 dynamic attention
- STAR-Former (2026): spatio-temporal adaptive region-aware transformer
- **Core 공통 발견**: 딥러닝 FER 모델의 **top attention은 주로 mouth**
- **JY 관련성**: JY Phase 0.1의 mouth 중심 결과가 보편 현상임을 확인. Jack 2012 괴리는 JY 고유 발견이 아니라 field-wide pattern

## 6. Emotion Perception Rules — Koreans vs Americans (Hogrefe 2022)
- **URL**: https://econtent.hogrefe.com/doi/10.1027/1618-3169/a000550
- **Core**: 한국인은 외적 감정 표현에 덜 가중치 (Americans는 joyful expression에 더 가중치)
- **JY 관련성**: Phase 2 "사회적 합의 구조" 설계 시 이 논문 cite. 298명 연세대 실험이 이와 연결됨

---

## Novelty Gap Analysis

### 선행연구가 이미 답한 것 ❌ (우리가 안 해도 되는 것)
| 답한 질문 | 답 | 답한 논문 |
|----------|---|---------|
| AU가 감정 분류에 유의미한가 | YES | Ekman 1978, FACS |
| 동아시아인은 어느 부위로 감정 표현하나 | eyes 중심 | Jack 2012 |
| 한국인 FACS stimuli 필요한가 | 이미 있음 | KUFEC-II 2017 |
| 딥러닝 FER이 어느 region에 attention 주나 | mouth 중심 | PCNN, CBAM 등 |
| 문화가 FER에 영향 주나 | YES, 104 accents | Li 2025 review |

### 선행연구가 안 답한 것 ✅ (JY 기여)
| 우리가 답할 질문 | 가용 asset |
|--------------|----------|
| 대규모 실제 한국인 데이터에서 AI가 본 region 우선순위는? | 237K AU Hub |
| Jack 2012의 East Asian=eyes 가설이 AI 모델 관점과 실제로 괴리인가? | 237K + Py-Feat AU |
| 298명 사회적 합의가 AI 관점 또는 심리학 관점 중 어디에 가까운가? | 연세대 is_selected |
| AU 크기/위치 × 중요도의 정량 관계는? | novel (선행연구 없음) |
| 한국인 ↔ 서양인 AU dialect matrix | AffectNet + KUFEC-II 조합 가능 |

---

## 적용 (이 reference가 Phase 별로 어디 쓰이나)

| Phase | 쓸 레퍼런스 |
|-------|----------|
| Phase 0.2 (Py-Feat AU 추출) | FACS canonical (Ekman) — 어떤 AU 뽑을지 기준 |
| Phase 1.1 (FACS 검증) | Ekman + Py-Feat AU reference |
| Phase 1.2 (Jack 재검증) | **Jack 2012** 직접 재검증 |
| Phase 2 (사회적 합의) | Koreans vs Americans 2022, Li 2025 |
| Phase 3 (graph learning) | ME-GraphAU, ANFL, PCNN (메소드 비교) |
| Phase 4 (cross-cultural) | Jack 2012 + KUFEC-II + Li 2025 |

---

## Open questions (후속 조사 필요)

1. **ME-GraphAU (ECCV 2022)** 논문 상세 읽기 — 우리 AU graph의 직접 비교 대상
2. **OpenGraphAU** pretrained checkpoint 접근 방법
3. **AffectNet의 인종 breakdown** — 정말 "Western" 포함 비율이 얼마나?
4. **KUFEC-II 접근성** — 다운로드 가능한가?
5. **연세대 298명 실험의 IRB/pre-registration 상태** — 담당자 확인 필요
