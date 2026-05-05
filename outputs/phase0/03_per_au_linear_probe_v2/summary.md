# Phase 0.3 — Per-AU Linear Probe Results

**측정**: 각 AU intensity (OpenGraphAU) 위에 로지스틱 회귀로 감정 4-class 맞추는 정확도.
Random = 25%, Phase 0.1c region POOLED = 80.2-81.5%.

## 3.3 Jack 2012 재검증 (Eye-AU vs Mouth-AU)
| Group | # AU | Accuracy | Macro F1 |
|-------|------|----------|----------|
| All 41 AU | 41 | **60.4%** ± 0.4 | 0.603 |
| Mouth-AU group | 19 | **55.4%** ± 0.7 | 0.552 |
| Eye-AU group | 14 | **54.4%** ± 0.2 | 0.535 |

**Mouth - Eye = +1.0%p** → Mouth group 우위 (Jack 2012 **반박**)

## 3.4 Ekman canonical mapping 재현도
| Emotion + AU | Accuracy | 해석 |
|-----|----------|------|
| 기쁨 (Happy, AU6+12) | **44.7%** | 강하게 재현 |
| 슬픔 (Sad, AU1+4+15) | **43.9%** | 강하게 재현 |
| 분노 (Anger, AU4+5+7+23) | **49.2%** | 강하게 재현 |

## 3.1 Top 10 per-AU ranking (single feature)
| AU | Accuracy | 해석 |
|-----|----------|------|
| AU6 (Cheek raiser) | 40.7% | |
| AU12 (Lip corner puller) | 40.2% | |
| AU10 (Upper lip raiser) | 39.1% | |
| AU7 (Lid tightener) | 37.6% | |
| AU25 (Lips part) | 37.5% | |
| AU4 (Brow lowerer) | 35.0% | |
| AU15 (Lip corner depressor) | 33.8% | |
| AU26 (Jaw drop) | 33.1% | |
| AU16 ((sub-AU variant)) | 33.0% | |
| AU5 (Upper lid raiser) | 33.0% | |