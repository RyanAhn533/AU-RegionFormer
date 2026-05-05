# AU-RegionFormer Research Log

## 목적
1. 모든 연구 세션/실험을 일지로 기록
2. Claude의 direction 품질을 hindsight로 평가
3. RLHF/DPO 학습용 preference 데이터 축적

## 디렉토리 구조
```
research_log/
├── sessions/                       # 논의/기획 세션
│   └── YYYY-MM-DD_주제.md
├── experiments/                    # 실행된 실험
│   └── phaseN_NN_실험명.md
└── claude_evaluation/
    ├── directions.jsonl            # Claude 제안 모음 (RLHF raw)
    └── templates/
        ├── session.md
        ├── experiment.md
        └── direction_eval.md
```

## 기록 원칙

### 기록 시점
- **세션 종료 시**: 논의 내용 `sessions/`에 저장
- **실험 시작 시**: 계획 `experiments/`에 저장 (status: planned)
- **실험 종료 시**: 결과 추가 + status 업데이트
- **1주 후**: Claude direction의 hindsight score 채우기

### 필수 메타데이터 (frontmatter)
```yaml
date: YYYY-MM-DD
session_id: claude-session-id (가능하면)
phase: Phase N.N
tags: [gnn, roadmap, q1_strategy]
claude_directions:
  - id: D001
    content: "Phase 0 진단부터"
    hindsight_score: null   # 실행 후 채움 (1-10)
    outcome: null            # 실행 후 요약
decisions:
  - accepted: [D001]
  - rejected: [D002]
  - modified: [D003]
```

## Claude Evaluation Framework

### Direction 평가 기준 (hindsight_score 1-10)
| Score | 의미 |
|-------|-----|
| 10 | 실행 후 결정적 breakthrough |
| 8-9 | 옳았고 효과 있음 |
| 6-7 | 옳았지만 부분적 |
| 4-5 | 효과 애매 |
| 2-3 | 틀렸지만 배움 있음 |
| 1 | 완전히 틀림, 시간 낭비 |

### 평가 축
- **Correctness**: 기술적으로 맞았나
- **Prioritization**: 우선순위 판단 좋았나
- **Foresight**: 미래 리스크 미리 봤나
- **Actionability**: 실행 가능했나

### DPO pair 추출
`directions.jsonl`에서 hindsight_score 차이가 큰 쌍을 `chosen/rejected` 형태로 추출.

예:
```json
{
  "context": "GNN 로드맵 설계",
  "chosen": "Phase 0 진단부터 (score 9)",
  "rejected": "바로 GAT 구현 (score 3)"
}
```

## 업데이트 규칙
- 세션 끝날 때마다 해당 날짜 파일 업데이트
- 실험 hindsight는 최소 1주 후 평가
- JY는 score 매기고 Claude는 작성만 (bias 방지)
