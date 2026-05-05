#!/bin/bash
# Label Quality Pipeline
# Step 1: 데이터 준비 → Step 2: 모델 학습 → Step 3: 413K 데이터에 적용
#
# 사용법: bash scripts/run_label_quality.sh

set -e
cd /home/ajy/AU-RegionFormer/src

echo "============================================"
echo "Step 1: Preparing label quality data..."
echo "============================================"
python -u label_quality/prepare_data.py

echo ""
echo "============================================"
echo "Step 2: Training label quality model..."
echo "============================================"
python -u label_quality/train_quality_model.py

echo ""
echo "============================================"
echo "Step 3: Applying scores to AU-RegionFormer data..."
echo "============================================"
python -u label_quality/apply_scores.py

echo ""
echo "============================================"
echo "DONE. Check outputs:"
echo "  Model: outputs/label_quality/"
echo "  Scored data: /mnt/hdd/ajy_25/au_csv/index_train_scored.csv"
echo "============================================"
