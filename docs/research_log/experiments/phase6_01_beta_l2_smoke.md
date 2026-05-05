---
date_planned: 2026-04-28
date_executed: 2026-04-28
phase: Phase 6.1
experiment_id: phase6_01_beta_l2_smoke
status: completed
claude_directions: [D101]
---

## 1. 가설
AU-RegionFormer 토큰 레벨에 Beta uncertainty gate (L2)를 끼워 넣으면 학습이 정상 진행되며, 토큰별 reliability r=α/(α+β)가 AU 영역별로 의미 있게 분화한다.

## 2. 방법
### 입력 데이터
- train_csv: `/mnt/hdd/ajy_25/au_csv/index_train.csv` (413,122 rows, 7-class)
- val_csv: `/mnt/hdd/ajy_25/au_csv/index_val.csv` (51,804)
- AU regions: 8 (cheek_left/right, chin, eyes_left/right, forehead, mouth, nose)

### 실험 설계
- 독립변수: l2_mode ∈ {scale, softmax}
- 종속변수: val F1, per-AU mean reliability
- 통제: 같은 backbone (mobilevitv2_100), 2 epochs, batch=32, freeze 1ep + unfreeze 1ep, seed=42

### 판정 기준
- GO: 학습 정상 완료 + per-AU r 분산이 0이 아님
- NO-GO: NaN / OOM / 모든 r=1.0 (gate 무력화)

## 3. 실행
### 명령어
```bash
python scripts/train.py --config configs/mobilevit_fer_beta_smoke.yaml
python scripts/analyze_beta_reliability.py \
  --config configs/mobilevit_fer_beta_smoke.yaml \
  --ckpt /mnt/hdd/ajy_25/results/mobilevit_fer_beta_smoke/best.pth
```
### 리소스
- GPU: A6000 1장, ~3.2GB (smoke), ~37min/2ep

## 4. 결과
### scale 모드 (1차)
- E1 frozen: val F1 = 0.5203, acc = 0.5405
- E2 unfrozen: val F1 = 0.6611, acc = 0.6689
- per-AU mean r (val 100 batches): **모든 8 AU 영역 r = 1.000 (포화)**
- → gate 무력화 (사실상 identity 매핑)

### softmax 모드 (2차)
- E1 frozen: val F1 = 0.5173, acc = 0.5390
- E2 unfrozen: val F1 = 0.6583, acc = 0.6661 (scale와 사실상 동일, ±0.003 noise 범위)
- per-AU mean r (val 51K samples):
  - forehead 0.915, mouth 0.864, chin 0.792, eyes_R 0.498, nose 0.322, eyes_L 0.162, cheek_R 0.101, cheek_L 0.032
- **Per-class 의미 있는 분화 관찰됨** (2 epoch 만에):
  - happy: cheek_R=0.175, cheek_L=0.066 (다른 class 평균 대비 ~2배) — 광대 raise (미소)
  - sad: chin=0.829, mouth=0.887 (최고치) — 입꼬리 처짐
  - surprised: cheek_L=0.016 (최저) — 광대보다 눈썹/입 위주
  - neutral: eyes_L=0.091 (최저) — 표정 정보 없음

### 해석
- "scale" mode (`gated_i = r_i × token_i`)은 trivial optimum이 r=1 모두에 있음 (token magnitude 보존이 downstream attention에 유리). KL regularization 없으면 saturate.
- SGMT-BU 원래 검증된 setting은 stream-level (CBBF bio-gate)로 token-level과 다름. Token-level gating에서는 token 간 경쟁이 필요.
- **Fix**: `gated_i = K × softmax(r/τ)_i × token_i` — total magnitude 보존 + token 간 강제 경쟁.

## 5. 판정
- [ ] GO (scale mode)
- [x] NO-GO (scale mode) — r 포화로 gate 무력화
- [x] GO (softmax mode) — F1 동등 + r 의미 있는 분화 + per-class 해석 가능

## 6. Claude direction 평가
- D101: "L2 → L3 → L1 순서로 적용"
- 부분 검증: L2 자체는 학습 안정. 다만 "scale" 모드는 degenerate함을 발견 (사전 인지 못함).
- hindsight_score: TBD
- 틀렸던 점: scale vs softmax 선택의 영향 사전 분석 부족.

## 7. 다음 단계
1. softmax smoke 결과 확인 → r 분산 검증
2. 만약 softmax도 부족하면 KL(Beta(α,β) ‖ Beta(1,1)) 정규화 추가
3. softmax 검증되면 production 200ep 페어 (Beta L2 vs no-Beta baseline) 동시 실행
