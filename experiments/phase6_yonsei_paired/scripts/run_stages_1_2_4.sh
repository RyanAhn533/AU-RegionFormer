#!/usr/bin/env bash
# Phase 6 — Sequential Stage 1 → 2 → 4 (config-only stages, no trainer mods needed)
set -uo pipefail
cd /home/ajy/AU-RegionFormer

PHASE=experiments/phase6_yonsei_paired

for STAGE in stage2_beta_iso stage4_beta_rel; do
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config $PHASE/configs/${STAGE}.yaml \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
  echo "[$(date)] $STAGE exit=$?"
done

echo "[$(date)] === ALL 3 STAGES DONE ==="
