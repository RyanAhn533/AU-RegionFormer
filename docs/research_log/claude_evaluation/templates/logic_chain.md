---
chain_id: LC###
date: YYYY-MM-DD
session: session_file_ref
trigger_type: question|proposal|correction|feedback
final: success|partial|failure
---

## Trigger
(JY의 원 요청 또는 질문 원문)

## Logic Chain

### Step 1 — (단계 제목)
- **Reasoning**: Claude가 어떤 논리로 판단했나
- **Action**: 실제 행동 (제안, 코드 작성, 질문 등)
- **Outcome**: 즉시 관찰 결과
- **Sign**: + (good) / - (bad) / ~ (neutral)
- **JY Reaction**: 있으면 원문

### Step 2 — ...
(같은 구조)

## Final
- **Outcome**: success / partial / failure
- **JY Feedback** (원문):
- **Lesson** (Claude가 배울 것):

## Signals for RLHF
- **Good pattern** (chosen): Claude의 어떤 논리 단계가 좋았나
- **Bad pattern** (rejected): 어떤 논리 단계가 틀렸나
- **DPO pair extractable**: Yes/No
