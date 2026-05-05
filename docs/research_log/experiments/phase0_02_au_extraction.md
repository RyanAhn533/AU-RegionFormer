---
date_planned: 2026-04-21
date_executed:
phase: Phase 0.2
experiment_id: phase0_02
status: planned
claude_directions: [D008, D012]
depends_on: [phase0_01]
unblocks: [phase0_03, phase1_1, phase1_2]
---

## 1. 가설

**Main**: FACS의 AU 단위 feature는 region-level (Phase 0.1 NO-GO)과 달리 class-discriminative할 것

**Sub**:
- H1: 일부 AU(예: AU6, AU12, AU4, AU15)는 per-AU silhouette > 0.1
- H2: Py-Feat 기반 AU intensity가 한국인 얼굴에서도 ICC > 0.7 (validation 필요)
- H3: Ekman canonical (Happy=AU6+12, Sad=AU1+4+15 등)가 한국인 데이터에서 재현

## 2. 방법

### Detector 선택
| Option | AU 개수 | 장점 | 단점 |
|--------|--------|------|------|
| **Py-Feat** (추천 primary) | 20 AU (AU1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26,28,43) | pip 설치, 빠름, 표준, 20만+ 이미지 가능 | 서양 얼굴 편향 가능 |
| **OpenGraphAU** (추천 secondary) | 41 AU | SOTA, 선행연구 정면 비교 | GPU heavy, checkpoint 다운 필요 |
| OpenFace 2.0 | 17 AU | 오래된 표준 | 업데이트 중단 |
| JAA-Net | 12 AU | 경량 | AU 개수 부족 |

**결정**: Py-Feat 우선 237K 추출. OpenGraphAU는 subset 10K만 (비교용). **둘 다 한국인 validation 필요** — KUFEC-II 일부 사용 가능하면 함께 비교.

### 입력
- 이미지 원본: `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/` (37GB, 237K)
- Meta: `data/label_quality/au_embeddings/meta_mobilevitv2_150.csv` (path, emotion, is_selected)

### 출력
- `data/label_quality/au_features/pyfeat_20au_237k.parquet` (237K × 20 AU intensity + 20 AU presence)
- `data/label_quality/au_features/opengraphau_41au_10k.parquet` (subset 10K × 41 AU)
- `outputs/phase0/02_au_extraction/extraction_stats.json`
- `outputs/phase0/02_au_extraction/au_distribution_per_emotion.png`

### 세부 단계
1. **환경 설치**: `pip install py-feat` (기존 conda base에 추가)
2. **Dry-run**: 100 이미지로 속도/메모리 측정
3. **Full extraction**: 237K (batched, GPU 활용, 중간 save)
4. **Integrity check**: missing/NaN/outlier rate
5. **Per-emotion AU intensity 평균/표준편차** 계산
6. **OpenGraphAU subset**: 10K stratified sample만

### 리소스
- GPU: Py-Feat은 ~6GB VRAM 사용 예정 (JY가 12GB 남기라 한 한계 내)
- CPU: bg 프로세스로 여러 batch 병렬
- RAM: ~20GB
- Storage: ~200MB (parquet)
- 예상 시간: Py-Feat 237K 6-8h (batch_size=64 기준)

### 판정
| 조건 | 다음 액션 |
|------|---------|
| AU intensity mean/std가 emotion 간 유의미 차이 (ANOVA F > 5) | Phase 0.3 AU silhouette 진행 |
| Phase 0.1과 동일하게 no discrimination | **detector 교체** (OpenGraphAU로 전면 전환) |
| Extraction 실패 > 10% | 이미지 품질 필터링 후 재시도 |

## 3. 실행

### 코드 위치
- `src/analysis/phase0/02a_pyfeat_extraction.py` (작성 예정)
- `src/analysis/phase0/02b_opengraphau_subset.py` (작성 예정, 2순위)

### 실행 순서
```bash
# 1. Dry-run (100 images)
python src/analysis/phase0/02a_pyfeat_extraction.py --dry-run --n 100

# 2. Full run (237K, 4h+)
python src/analysis/phase0/02a_pyfeat_extraction.py \
  --image-root /home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea \
  --meta-csv data/label_quality/au_embeddings/meta_mobilevitv2_150.csv \
  --output data/label_quality/au_features/pyfeat_20au_237k.parquet \
  --batch-size 64 --num-workers 4 --gpu 0

# 3. Sanity check
python src/analysis/phase0/02_sanity.py --au-file pyfeat_20au_237k.parquet
```

## 4. 결과
(미실행)

## 5. 판정
- [ ] GO Phase 0.3 (AU-level analysis)
- [ ] NO-GO (detector 교체)
- [ ] Partial (Py-Feat 한국인 정확도 부족 → OpenGraphAU primary로)

## 6. Claude direction 평가
- Direction ID: D008 (AU vs region 정정), D012 (detector 선택)
- hindsight_score: (TBD)

## 7. 다음 단계
- **Phase 0.3**: 추출된 AU로 per-AU silhouette/Fisher/ANOVA
- **Phase 1.1**: Ekman canonical 검증 (Happy=AU6+12 한국인 재현?)
- **Phase 1.2**: Jack 2012 재검증 (eyes-AU vs mouth-AU 감정 구분력)
- **한국인 validation 별도**: KUFEC-II로 Py-Feat vs OpenGraphAU ICC 측정

## 주의사항

1. **한국인 validation 없으면 결과 해석 주의**: detector 정확도가 낮으면 전체 분석이 쓰레기 → Phase 0.3 초반에 validation 필수
2. **Py-Feat 설치 시 torch 버전 충돌 가능**: conda base 환경 보존 필요. 필요 시 별도 venv
3. **JY 서버 GPU 12GB 제약**: Py-Feat batch_size 적절히 (64 이하)
4. **중간 저장 필수**: 237K 6-8h 작업이라 crash 리스크. 1K마다 checkpoint parquet
