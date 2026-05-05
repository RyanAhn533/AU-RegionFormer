#!/usr/bin/env bash
# 4-class fast iter: baseline → Beta L2(softmax) sequential
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=/mnt/hdd/ajy_25/results
mkdir -p "$LOG_DIR"

echo "[$(date)] === 4c Run 1: baseline (no Beta) ==="
python scripts/train.py --config configs/mobilevit_fer_4c_base.yaml \
    > "$LOG_DIR/4c_base.log" 2>&1
RC1=$?
echo "[$(date)] 4c base exit=$RC1"

echo "[$(date)] === 4c Run 2: Beta L2 (softmax) ==="
python scripts/train.py --config configs/mobilevit_fer_4c_beta_l2.yaml \
    > "$LOG_DIR/4c_beta_l2.log" 2>&1
RC2=$?
echo "[$(date)] 4c Beta exit=$RC2"

echo "[$(date)] === Reliability analysis on Beta best.pth ==="
python scripts/analyze_beta_reliability.py \
    --config configs/mobilevit_fer_4c_beta_l2.yaml \
    --ckpt /mnt/hdd/ajy_25/results/4c_beta_l2/best.pth \
    --split val --max_batches 1000 \
    > "$LOG_DIR/4c_beta_l2_reliability.log" 2>&1 || true

echo "[$(date)] === ALL DONE === base=$RC1 Beta=$RC2"
exit $((RC1 + RC2))
