# AU-RegionFormer 프로젝트 전체 현황

## 1. 데이터 위치

### 이미지 데이터

| 데이터 | 위치 | 크기 | 내용 |
|--------|------|------|------|
| AU-RegionFormer 학습용 | `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea/` | 37GB | 7감정, 얼굴 crop 이미지 413K장 |
| AU-RegionFormer val | `/home/ajy/FER_03_aihub_au_vit/data2/data_processed_korea_validation/` | 4.7GB | val set 52K장 |
| 연세대 원본 (전신사진) | `/mnt/ssd2/한국인 감정인식을 위한 복합 영상/` | 1.4TB | 4감정 원천 이미지 + merged |
| 연세대 원본 (링크) | `/data/heartlab/shared_ssd2/한국인 감정인식을 위한 복합 영상/` | 동일 | /mnt/ssd2로 이동됨 |
| 시뮬레이터 초기 | `/data/heartlab/Simulator_data/` | 252GB | C001~C019, video+audio+ppg |
| 시뮬레이터 본 데이터 | `/mnt/ssd2/KMER_Sensing_Backup/` | 486GB | C001+C040~C116 (79명) |
| 시뮬레이터 복구분 | `/mnt/ssd2/KMER_Sensing_Backup_RECOVERED/` | 86GB | 일부 복구 데이터 |

### CSV / 분석 데이터

| 파일 | 위치 | 내용 |
|------|------|------|
| AU 학습 CSV (원본) | `/mnt/hdd/ajy_25/au_csv/index_train.csv` | 413K장 AU 좌표 + 라벨 |
| AU val CSV | `/mnt/hdd/ajy_25/au_csv/index_val.csv` | 52K장 |
| quality score | `/mnt/hdd/ajy_25/au_csv/index_train_full_quality.csv` | 3-stage noise 점수 |
| 연세대 설문 결과 | `shared_ssd2/한국인 감정인식.../python_code/final_result.csv` | 247K 응답 |
| 연세대 최종 유지목록 | `shared_ssd2/.../python_code/260128_...xlsx` | 235K장 |
| sad 매핑 | `AU-RegionFormer/data/label_quality/sad_to_orig_mapping.csv` | sad↔원본 58K |
| 얼굴 특징 (landmark) | `AU-RegionFormer/data/label_quality/face_features.csv` | 237K장 17개 특징 |
| model vs human | `AU-RegionFormer/data/label_quality/model_vs_human.csv` | 모델예측 vs 인간판정 |
| AU embedding | `AU-RegionFormer/data/label_quality/au_embeddings/` | MobileViT+ConvNeXt 8영역 |

### 디스크 현황

| 디스크 | 마운트 | 용량 | 사용 | 용도 |
|--------|--------|------|------|------|
| NVMe (/) | / | 926GB | 86% | OS, 코드, 학습 이미지 |
| SSD2 | /data | 3.6TB | 26% | 연세대 원본, 시뮬레이터 초기 |
| NVMe2 | /mnt/ssd2 | 3.6TB | 57% | KMER 센싱, 한국인 감정 원본 |
| HDD | /mnt/hdd | 7.3TB | 9% | CSV, 이전 실험, 캐시 |
| munchebu | /mnt/munchebu | 3.7TB | 33% | DECA, sc 데이터 |

---

## 2. 연세대 검증 데이터셋 — 뭔지

AI Hub "한국인 감정인식을 위한 복합 영상" 원본 데이터(24만장)에 대해, 연세대 심리학과 학생 298명이 "이 사진이 정말 이 감정이 맞는지" 검증한 실험 데이터.

- **방법**: 같은 감정 사진 10장 보여주고 "이 감정이 아닌 것" 고르기
- **대상**: 기쁨, 분노, 슬픔, 중립 (4감정만)
- **결과**: 기쁨/중립은 거의 안 걸러짐, 분노 7.6% 슬픔 8.1% 제거
- **핵심**: `is_selected=1` = "이 감정 아니다" (부정 선택)

---

## 3. 실험 및 결과

### 3.1 모델 학습 실험

| 실험 | F1 | 뭘 바꿨나 | 인사이트 |
|------|-----|----------|---------|
| v1 | 0.795 | MobileViTv2-100 baseline | - |
| v2 | **0.805** | backbone↑(150) + fusion 2층 | 구조 개선 효과 +1%p |
| v3 | **0.807** | + EMA + distill + regularization | 거의 변화 없음, 과적합 심화 |
| v5 (soft label) | 0.538 | soft label + sample weight | **실패** — KL loss가 학습 방향 희석 |
| 4emo_before | 진행중 | 4감정 필터링 전 | 비교 실험 |
| 4emo_after | 진행중 | 4감정 필터링 후 | AU heatmap 변화 확인용 |

**핵심 결론**: 80% 벽은 모델 구조 문제가 아니라 데이터/감정 구조 문제.

### 3.2 데이터 분석

| 분석 | 결과 | 의미 |
|------|------|------|
| annotator 3인 일치율 | 기쁨 96% > 중립 74% > 분노 51% > 슬픔 48% | 부정 감정 라벨 자체가 모호 |
| 연세대 검증 교차 | 5개 독립 신호 동일 순서, p<10⁻³⁷ | 경향성 통계적으로 확실 |
| 혼동 비대칭 | 슬픔→상처 11.9%, 역방향 0% | 일방향 혼동 = 문화적 절제 |
| 성별 | 남성 분노 +5.3%p | 여성 부정감정 더 절제 |
| 나이 | 20대 슬픔 67% vs 60대 86% | 젊은 세대 더 모호 |
| 전문인 vs 일반인 | 일반인이 배우보다 높음 (-5.7%p) | 연기 < 자연 표현 |
| landmark 분석 | 기쁨 입꼬리 d=0.79 | 감정별 물리적 차이 정량화 |
| kNN 불일치 | anxious 0.54, hurt 0.52 | feature space에서 분리 불가 |
| AU cosine sim | 감정 간 0.97-0.99 | 모든 감정 AU 패턴 거의 동일 |
| cleanlab | happy 0.91 > hurt 0.77 | 모델 기반으로도 동일 순서 |

### 3.3 추출 완료 데이터

| 데이터 | 규모 | 상태 |
|--------|------|------|
| 얼굴 landmark 특징 (17개) | 237K장 | ✅ 완료 |
| AU embedding (MobileViT) | 237K × 8영역 × 768d | ✅ 완료 |
| AU embedding (ConvNeXt) | 237K × 8영역 × 1024d | ✅ 완료 |
| sad 원본 매핑 | 58K장 매칭 (sim 0.91) | ✅ 완료 |
| gradient 분석 | 10K장 | ✅ 완료 |

---

## 4. 왜 이걸 했나 — 목표

### 궁극적 목표
**"한국인의 감정 표현이란 무엇인가"를 데이터로 정의**

### 접근 순서
```
1. 데이터 품질 분석 → "80% 벽의 원인은 라벨 모호성"
2. 한국인 감정 경향성 발견 → 혼동 비대칭, 성별/나이 차이
3. AU 수준 물리적 근거 → landmark + CNN embedding
4. 필터링 전후 모델 비교 → "진짜 한국인 AU 패턴" 도출 (진행중)
5. 다른 문화권 비교 → "한국인 AU dialect" 정립 (예정)
6. 모델에 반영 → 한국인 특화 AU 가중치 (예정)
```

### 논문 방향
- **한국인 감정 표현의 문화적 특성** (AU 분석 + 인구통계 + 교차검증)
- **"감정 인식 성능 한계가 데이터 구조에서 오는가?"** 분석 논문
- 향후: 멀티모달(시뮬레이터 데이터)과 결합

---

## 5. 시뮬레이터 데이터 (KMER Sensing)

차량 시뮬레이터에서 수집한 tri-modality 데이터.

### 구조
```
C0XX/
  20260XXX_HHMM/     (세션 = 타임스탬프)
    video_main.mp4   — 얼굴 카메라
    video_sub.mp4    — 서브 카메라  
    audio.wav        — 음성
    ppg.csv          — 광용적맥파
    gsr.csv          — 피부전도도 (일부)
    temp.csv         — 체온 (일부)
```

### 규모
- 초기 (Simulator_data): C001~C019, 252GB
- 본 데이터 (KMER_Sensing_Backup): C040~C116, 486GB
- 총 ~98명 참가자

### 할 일
- 피험자별 실험 로그(날짜/시간) ↔ 폴더 매칭
- Video에서 표정 추출 → AU-RegionFormer 적용
- Bio signal(PPG) + 표정 + 음성 → 멀티모달 감정 분석

---

## 6. 산출물

### 문서
- `docs/label_quality_analysis.md` — 전체 분석 상세 기록 (Layer 0~5)
- `docs/dataset_tendency_analysis.md` — 경향성 분석 markdown
- `docs/report/한국인_감정_데이터셋_선택경향성_분석보고서.docx` — 보고서 (그림 포함)

### 시각화
- `outputs/viz/report_figures/` — 22개 figure (전부 영어 라벨)

### 코드
- `src/label_quality/` — 데이터 분석 파이프라인 전체
- `src/training/noise_aware_trainer.py` — soft label trainer
- `src/data/noise_aware_dataset.py` — noise-aware dataset
- `configs/4emo_before.yaml`, `4emo_after.yaml` — 필터링 비교 실험

---

## 7. 다음 할 것

1. **4감정 필터링 전후 학습 완료** → AU heatmap 비교
2. **시뮬레이터 데이터 정리** — 피험자 매칭, 세션 구조화
3. **AffectNet 등 서양 데이터와 AU 패턴 비교** — "한국인 AU dialect"
4. **모델에 한국인 AU 특성 반영** → 성능 개선 시도
5. **논문 작성** — 한국인 감정 표현 문화적 특성
