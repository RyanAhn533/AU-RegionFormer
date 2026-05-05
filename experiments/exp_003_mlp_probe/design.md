---
exp_id: exp_003_mlp_probe
iteration: 3
date: 2026-04-22
category: probe_capacity
strategy: a (최근 논문 — Alain & Bengio 2016 variant)
depends_on: exp_002 (or independent)
---

# Exp 003 — MLP Probe (non-linear) vs Linear Probe

## 연구 질문
Region/AU embedding이 **비선형 패턴**을 가지고 있나? Linear probe가 underestimate인가?

## 원 논문 출처
- Alain & Bengio (2016) "Understanding intermediate layers using linear classifier probes" — linear probe
- Locatello et al. (2020) "A Sober Look at Probes" — MLP probe 주의점 (overfitting)

## 핵심 아이디어 (3줄)
1. 기존 linear probe (1-layer LogReg) 위에 2-layer MLP (256→64→4) 추가
2. 동일 protocol (L2 norm + PCA256 + 3-fold CV + 30K)
3. Linear vs MLP delta = non-linearity signal

## 구체 수정 포인트
- `src/analysis/phase0/01c_linear_probe.py` 사본 → `01d_mlp_probe.py`
- `LogisticRegression` → `MLPClassifier(hidden_layer_sizes=(256,64), max_iter=300)`
- Same seed, same split, same subsample

## 예상 metric 변화
- Region POOLED 81.5% → 82~84% (비선형으로 +1-3%p)
- AU-level 59.9% → 65~70% (AU 간 interaction 포착)
- 예상: Linear와 delta < 3%p면 embedding은 이미 linearly separable

## Paper section 매핑
- §4.1 Ablation ("linear vs MLP probe")
- §3.3 Method justification (왜 linear probe 썼는지 근거)

## 예상 시간
- 30-40분 (MLP 학습 오래 걸림)

## Fallback
- MLP가 과적합 → early stopping + validation split
- Delta 너무 크면 overfitting 의심 → dropout 추가 + 재측정
