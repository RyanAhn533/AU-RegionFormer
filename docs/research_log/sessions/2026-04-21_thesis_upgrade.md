---
date: 2026-04-21
session_id: home-ajy-session-part2
phase: Phase 0.1 완료 + Thesis 재설계
tags: [phase0_1_result, thesis_upgrade, jack_2012, au_vs_region, py_feat, prior_research]
claude_directions: [D008, D009, D010, D011, D012, D013]
decisions:
  accepted: [D008, D009, D010, D011, D013]
  deferred: [D012]
---

## 1. 맥락
- 직전 세션에서 Phase 0.1 (AU embedding 진단) 실행 승인
- 이 세션에서 실제 실행 + 결과 분석 + thesis 재설계
- JY가 추가로 "한국형 AU 중요 부위", "영역 크기/위치", "선행연구 조사" 요청

## 2. JY 요청들 (시간순)

1. "AU중요도도 중요도인데 어느 부위 즉 영역크기 영역 위치 등도 좀 더 파보자"
2. "실제 FER에서는 어디 영역이 최고 중요한데? 이런거 선행연구있나 확인해보고 해야하면하고 레퍼런스할꺼있으면 해서 좀 더 컴팩트하고 핵심적이게 삽질안하게 진행"
3. "메모리에도 저장해 뭔가 진행할때는 선행연구 조사해서 삽질이나 이런거 안하게 하기"
4. "region으로 실험한게 맞아? FACS에서 AU는 또 달라" (**중요한 지적**)
5. "이런것들로 하면 지금 전체 실험 로드맵이 조금 수정될수있을것같아 방향과 정리해봐"
6. "Q1까지 갈 수 있을 것같아? 충분해?"

## 3. Phase 0.1 실제 결과

### 수치 (Pooled)
| Backbone | Silhouette | Inter-class cos sim |
|----------|-----------|---------------------|
| MobileViT v2 | 0.0072 | 0.9726 |
| ConvNeXt-base | 0.0113 | 0.9676 |

### Per-region (핵심)
- **mouth**: MobileViT 0.0094, ConvNeXt 0.0194 (유일하게 양수 두 backbone)
- **eyes**: -0.02 (가장 나쁨, class 섞음)
- **forehead/chin/cheek**: 거의 0

### 판정
- 양쪽 backbone NO-GO (silhouette < 0.05)
- 단, **mouth region에 약한 signal 존재** → AU-level로 내려가면 살아날 가능성
- cos sim 0.97-0.99 기존 보고치 재확인됨

## 4. Claude 제안 direction

### D008: AU vs region 용어 정정 (JY 지적 수용)
- **내 실수**: "AU embedding" 용어를 region-level 실험에 오용
- **정정**: Phase 0.1은 face region (공간) 실험. FACS의 AU (근육)는 별도 층위
- **액션**: Phase 0.2에서 Py-Feat/OpenGraphAU로 진짜 AU 추출 필수

### D009: Thesis 업그레이드 — AI vs Psychology 괴리
- 선행연구 조사 결과 **Jack 2012 PNAS**: East Asian = eyes 중심
- JY Phase 0.1 결과: AI 모델은 mouth만 유용
- → **정면 괴리 = counterintuitive finding = NHB-급 질문 성립**
- 논문 thesis를 "한국형 관계 구조"에서 "AI-심리학 괴리 정량화"로 업그레이드

### D010: Phase 1 재설계 — 한국형 AU 지도
- 이전: 기초 graph statistics 5개
- 새: FACS canonical 검증 + Jack 2012 재검증 + 영역 크기/위치 + is_selected subset
- 이게 논문 Figure 1~5 됨

### D011: Q1 재평가
- **TAFFC**: 30-40% → **50-60%**
- **NHB/PNAS**: 5-10% → **15-20%** (충분조건 아직 아님)
- **전략**: TAFFC 먼저, NHB 후속 확장

### D012 (deferred): Py-Feat + OpenGraphAU 병행
- 둘 다 추출하면 detector 비교 실험도 부산물 (Q2 재료)
- JY 답 대기 중

### D013: 선행연구 우선 원칙 (memory 신설)
- `feedback_prior_research_first.md` 저장
- 실험/방법 제안 전 WebSearch 먼저

## 5. JY 판단
- D008, D009, D010, D011, D013: 수용 ("좋아보여")
- D012: "계속 진행해보자" — 이는 로드맵 진행 승인이며 Py-Feat vs OpenGraphAU 선택은 여전히 open

## 6. 액션 아이템

- [x] Phase 0.1 experiment 파일에 결과 반영
- [x] directions.jsonl에 D008~D013 추가
- [x] 선행연구 references 파일 저장
- [x] 이 세션 기록
- [ ] project memory 업데이트 (새 thesis + 새 로드맵)
- [ ] CLAUDE.md에 Phase 0-4 새 로드맵 반영
- [ ] Phase 0.2 Py-Feat AU 추출 계획 문서 작성
- [ ] JY 최종 승인 후 Py-Feat 설치 + 추출 시작

## 7. Hindsight 평가 (실행 후 1주 뒤)
(deferred)

## 8. 이 세션의 교훈

1. **선행연구 조사 없이 제안하면 용어 실수 남** — JY가 FACS AU와 region 차이 지적 → feedback_prior_research_first.md로 시스템화
2. **"실험 결과"가 선행연구와 직접 대화하면 thesis가 강해진다** — Phase 0.1 mouth 결과가 Jack 2012 재검증 기회로 확장됨 (예측 못한 ripple)
3. **JY는 "Q1 충분해?" 형태로 냉정한 확률 재확인을 반복함** — 각 설계 변경마다 % 업데이트 필요. 체크포인트 형태로 제공하기
4. **"GNN 실험 쭉" → "Q2 재료 부산물"** 전략은 매 Phase마다 부산물 가능성을 명시해야 작동
