#!/usr/bin/env bash
# V2 ablation: 5 variants sequential on 4-class data.
# NOTE: do NOT use `set -e` — we want the loop to keep going even if a
# single variant's post-training artifact step crashes.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=/mnt/hdd/ajy_25/results
mkdir -p "$LOG_DIR"

VARIANTS=(
  "v2_4c_baseline"
  "v2_4c_aOnly"
  "v2_4c_bIso"
  "v2_4c_bRel"
  "v2_4c_full"
)

for V in "${VARIANTS[@]}"; do
  echo "[$(date)] === Run: $V ==="
  python scripts/train.py --config "configs/${V}.yaml" \
      > "$LOG_DIR/${V}.log" 2>&1 || true
  RC=$?
  echo "[$(date)] $V exit=$RC"
done

echo "[$(date)] === ALL 5 VARIANTS DONE ==="
