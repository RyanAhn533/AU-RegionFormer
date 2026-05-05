---
exp_id: exp_003_jack_clean_per_region
iteration: 3
date: 2026-04-22
category: data_finding (paper direct §4.3)
strategy: d (combination: exp_002 clean + Phase 0.1c per-region)
---

# Exp 003 — Jack 2012 재검증 on Clean Subset (per-region)

## 연구 질문
Phase 0.1c: region raw에서 mouth 75.2% vs eyes 61.5% (gap +13.7%p, AI > Eye 우위)
exp_002: clean subset에서 POOLED +2.78%p 상승

이 실험: **Clean subset에서 per-region ranking은 어떻게 변하는가?**
- Mouth-Eye gap이 더 크게 나오면 → Jack 2012 반박 더 강력 (논문 §4.3 핵심)
- gap 줄어들면 → "noise가 mouth 과대평가 원인" 해석 가능

## 원 논문 연결
- Jack et al. 2012 PNAS: East Asian = eyes 중심
- Phase 0.1c 원본 vs exp_003 clean → 연세대 298명 사회적 합의로 filter 후에도 gap 유지?

## 구체 구현
- Exp 002와 동일 framework. 단 per-region 8개 + POOLED 독립 측정
- Raw vs Clean 각 8+1 = 9 variant per backbone (ConvNeXt)
- Paper table: Region × {Raw, Clean} acc matrix

## Primary metric
- **Mouth acc (clean) - Eyes acc (clean)** → Jack reversal effect size
- Phase 0.1c raw 기준: mouth 75.2% - eyes 61.5% = +13.7%p
- 예상 clean: mouth ~77-78%, eyes ~63-65%, gap ~13-14%p (robust 예상)

## Paper section 매핑 (직접)
- **§4.3 Figure**: Region importance bar chart (Raw vs Clean × 8 regions)
- **§4.5 Ablation**: social consensus filtering 효과 per-region

## 예상 시간: 10-15분
