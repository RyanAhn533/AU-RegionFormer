#!/usr/bin/env bash
# Phase 6 — Stage 5 → 6 → 7 → 8 sequential
set -uo pipefail
cd /home/ajy/AU-RegionFormer
PHASE=experiments/phase6_yonsei_paired

for STAGE in stage5_beta_anchored stage6_full stage7c_patch_dropout stage7_jepa stage8_dinov2; do
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config $PHASE/configs/${STAGE}.yaml \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
  echo "[$(date)] $STAGE exit=$?"
done
echo "[$(date)] === STAGES 5-8 DONE ==="
