---
date: 2026-04-21
session_id: home-ajy-session
phase: planning (Phase 0 직전)
tags: [gnn, roadmap, q1_strategy, q2_strategy, korean_emotion, social_consensus]
claude_directions:
  - id: D001
    content: "ChatGPT 6단계 로드맵의 AU 8-node graph 폐기, sample-level 237K graph로 전환"
    hindsight_score: null
    outcome: null
  - id: D002
    content: "Phase 0 진단 단계 추가 (AU embedding이 class-discriminative한지 먼저 확인)"
    hindsight_score: null
    outcome: null
  - id: D003
    content: "Cross-cultural (AffectNet/RAF-DB) 비교를 Phase 4에 필수로 배치"
    hindsight_score: null
    outcome: null
  - id: D004
    content: "Target venue를 NHB/PNAS가 아닌 TAFFC (IF 11)로 현실 설정. NHB는 2년+심리학 co-author 조건부"
    hindsight_score: null
    outcome: null
  - id: D005
    content: "Q1(JY) AU embedding graph + Q2(석사) landmark-based FER로 asset 공유하되 feature space 분리"
    hindsight_score: null
    outcome: null
  - id: D006
    content: "KMER 시뮬레이터는 석사 Q2로 주지 말 것 (데이터 정리 오버헤드 3개월)"
    hindsight_score: null
    outcome: null
  - id: D007
    content: "연세대 298명 실험 담당 심리학과 교수를 co-author로 영입할 것 (NHB/PNAS 도전 시 필수 레버리지)"
    hindsight_score: null
    outcome: null
decisions:
  accepted: [D001, D002, D003, D004, D006]
  rejected: []
  deferred: [D005, D007]
---

## 1. 맥락
- JY가 ChatGPT로부터 FER × GNN 6단계 로드맵 제안 받음
- AI top-tier 학계 연구자 관점에서 냉철한 비판 + 로드맵 재설계 요청
- 목표가 "80% 벽 돌파"에서 "한국형 감정 정의 + 사회실험 AI 접목 + Q1 논문"으로 재정의됨
- JY 자산: 237K AU embedding, 3-annotator 라벨, 연세대 298명 검증, demographic, KMER 시뮬레이터 98명
- JY 과거 이력: v1(F1=0.795) → v3(F1=0.807) 80% 벽, v5 soft label 실패(F1=0.538)

## 2. JY 요청 / 질문

### 2.1 초기 요청
ChatGPT 6단계 로드맵(AU correlation → GCN/GAT → uncertainty-aware → hetero graph → graph transformer → multimodal temporal graph)에 대한 학계 연구자 수준 리뷰 + 전체 실험 로드맵 설계

### 2.2 목표 재정의
> "80%를 넘기기 위해서 하는게 아니야. 가장 중요한건 한국형 감정이 뭔지, 그리고 사회실험을 인공지능에 접목시켜서 Q1이상의 논문을 쓰는게 목표"

### 2.3 venue 현실 확인
> "냉정하게 저정도 갈만해?"

### 2.4 Q1/Q2 분리 전략
> "다른 석사한테 Q2 하나 쓰라고 주게. Q2하나 Q1하나 뽑아보는거 어때"

### 2.5 최종 결정
> "우선 GNN 관련해서 쭉 실험해보자. 그러면 석사한테도 이걸로 Q2쓰세요 할만한게 나올듯"

## 3. Claude 제안 direction

### D001: AU 8-node graph 폐기, sample-level 237K graph로 전환
- **내용**: ChatGPT가 제안한 "8개 AU region을 노드로 하는 graph"는 학술적으로 약함 (Set Transformer와 구분 안 됨). 대신 237K 이미지를 노드로 하는 kNN graph로 가야 진짜 graph problem.
- **근거**: ME-GraphAU, ANFL 등 선행연구는 수십~수백 node 사용. 8 node는 reviewer 즉시 공격 포인트.
- **리스크**: 237K 노드는 full-batch 불가 → GraphSAINT/SAGE 샘플링 필요. 메모리/시간 오버헤드.

### D002: Phase 0 진단 단계 필수 삽입
- **내용**: AU embedding cos sim 0.97-0.99 문제를 방치한 채 GNN 올리면 무효. PCA/UMAP/silhouette로 "AU가 class-discriminative한가" 먼저 확인.
- **근거**: 기초 진단 없이 복잡한 모델 쌓으면 6개월 헛삽질 가능.
- **리스크**: 진단 결과가 NO-GO면 backbone 교체부터 해야 함.

### D003: Cross-cultural 필수 배치
- **내용**: "한국인" novelty만으로는 TAFFC/NHB 못 감. AffectNet/RAF-DB 비교로 "한국인 AU dialect" 수치화 필수.
- **근거**: Jack et al. 2012 PNAS, Matsumoto display rules 등 선행연구 존재. 단독 한국 연구는 dataset paper로 분류됨.
- **리스크**: AffectNet은 labeling protocol 다름 → direct comparison 어려움.

### D004: Venue 현실화 (TAFFC 메인, NHB는 드림)
- **내용**: NHB/PNAS는 현재 setup으로 5-10% 확률. TAFFC(IF 11)가 현실 30-40%. NHB 도전하려면 심리학 co-author + pre-registration + 2년 준비 필요.
- **근거**:
  - Psychology rigor 부족 (IRB, pre-registration 불명)
  - Jack 2012 PNAS가 East Asian vs Western 이미 선점
  - "한국인 AU dialect 있다" = expected finding, not counterintuitive
- **리스크**: TAFFC로도 30-40%. Ablation과 cross-cultural이 필수.

### D005: Q1/Q2 asset 분리 — JY는 CNN embedding, 석사는 landmark
- **내용**: 같은 237K 데이터에서 JY(AU embedding + graph + social consensus)와 석사(landmark 17feature + 전통 ML 벤치마크)로 feature space 완전 분리.
- **근거**: self-plagiarism 회피. 석사가 3-6개월 안에 완성 가능해야 함.
- **리스크**: 석사 background에 따라 적합성 달라짐.

### D006: KMER 시뮬레이터는 석사에게 주지 말 것
- **내용**: 486GB 데이터 정리 + 피험자-세션 매칭 + PPG preprocessing = 석사 3개월 날림. JY Phase 5 별도 논문으로 남김.
- **근거**: 석사 졸업 기한 압박. "연구 아닌 노가다" 최소화.

### D007: 연세대 심리학과 교수 co-author 영입
- **내용**: NHB/PNAS 도전 시 Psychology 도메인 전문가 필수. 298명 실험 담당자가 있다면 직접 접촉.
- **근거**: AI 랩 단독으로 psychology 저널 리젝률 매우 높음.
- **상태**: JY 미결정.

## 4. JY 판단
- D001, D002, D003, D004, D006: 수용
- D005: "GNN 실험 쭉 해보면 자연스럽게 Q2 재료 나올 것" — 명시적 분리 전에 실험 먼저
- D007: 이번 세션에서 결론 안 남 (deferred)

## 5. 액션 아이템
- [ ] Phase 0.1 AU embedding PCA/UMAP (3일)
- [ ] Phase 0.2 per-class cos sim 원인 규명 (4일)
- [ ] Phase 0.3 Landmark Fisher score (2일)
- [ ] Phase 0.4 Soft label v5 postmortem (2일)
- [ ] 각 실험 결과를 experiments/ 디렉토리에 기록
- [ ] Phase 0 완료 후 Phase 1 여부 결정

## 6. Hindsight 평가 (실행 후 1주 뒤)
(미실행)

## 7. 이 세션의 교훈
- JY는 "성능" 프레이밍을 싫어한다. "한국형 감정 정의 + 사회실험 AI 접목"이 진짜 목표.
- Venue 질문할 때 "저정도 갈만해?" = JY는 dream pitch가 아닌 현실 확률 원함. 솔직한 %로 답해야 함.
- Q1/Q2 동시 진행 전략은 한국 AI 랩의 일반적 패턴. asset 분리 원칙을 미리 짜두면 좋음.
