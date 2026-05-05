#!/usr/bin/env bash
# Sequential paired runs: Beta-L2(softmax) → baseline (no Beta)
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=/mnt/hdd/ajy_25/results
mkdir -p "$LOG_DIR"

echo "[$(date)] === Run 1: v4 + Beta L2 (softmax) ==="
python scripts/train.py --config configs/mobilevit_fer_v4_beta_l2.yaml \
    > "$LOG_DIR/v4_beta_l2.log" 2>&1
RC1=$?
echo "[$(date)] Beta run exit=$RC1"

echo "[$(date)] === Run 2: v4 baseline (no Beta) ==="
python scripts/train.py --config configs/mobilevit_fer_v4_base.yaml \
    > "$LOG_DIR/v4_base.log" 2>&1
RC2=$?
echo "[$(date)] Baseline run exit=$RC2"

echo "[$(date)] === Reliability analysis on Beta best.pth ==="
python scripts/analyze_beta_reliability.py \
    --config configs/mobilevit_fer_v4_beta_l2.yaml \
    --ckpt /mnt/hdd/ajy_25/results/v4_beta_l2/best.pth \
    --split val --max_batches 1000 \
    > "$LOG_DIR/v4_beta_l2_reliability.log" 2>&1 || true

echo "[$(date)] === ALL DONE === Beta exit=$RC1, Baseline exit=$RC2"
exit $((RC1 + RC2))
