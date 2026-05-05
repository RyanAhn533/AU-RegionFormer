#!/usr/bin/env bash
# Sequential queue: wait for 16a (already running), then 16b → 16c → 16d → AffectNet finetune
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_full/best.pth

# ─── Wait for 16a to finish ───
echo "[$(date)] waiting for stage16a_iso_only..."
while pgrep -f "stage16a_iso_only.yaml" > /dev/null; do
  sleep 60
done
echo "[$(date)] stage16a_iso_only DONE"

run_stage() {
  local STAGE=$1
  local CFG="$PHASE/configs/${STAGE}.yaml"
  local OUT_BEST="/mnt/hdd/ajy_25/results/phase6_${STAGE}/best.pth"
  if [ -f "$OUT_BEST" ]; then
    echo "[$(date)] SKIP $STAGE"
    return
  fi
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config "$CFG" \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
  echo "[$(date)] $STAGE finished"
}

# ─── Sequential: 16b → AffectNet finetune (Q1 decision FIRST) → 16c → 16d ───
run_stage stage16b_rel_only

# AffectNet finetune (Q1 decision point — promoted before 16c/d for early result)
CFG=$PHASE/configs/finetune/finetune_affectnet.yaml
if [ -f "$CFG" ] && [ -f "$INIT_CKPT" ]; then
  OUT="/mnt/hdd/ajy_25/results/phase6_finetune_affectnet/best.pth"
  if [ ! -f "$OUT" ]; then
    echo "[$(date)] === FINETUNE: affectnet (Q1 decision point) ==="
    python scripts/train.py --config "$CFG" --init_from "$INIT_CKPT" \
        > $PHASE/logs/finetune_affectnet.log 2>&1 || true
    echo "[$(date)] finetune_affectnet finished"
  fi
fi

run_stage stage16c_a_only
run_stage stage16d_product

# ─── xeval Stage 16 ckpts on AffectNet ───
for STAGE in stage16a_iso_only stage16b_rel_only stage16c_a_only stage16d_product; do
  CKPT="/mnt/hdd/ajy_25/results/phase6_${STAGE}/best.pth"
  [ -f "$CKPT" ] || continue
  CSV=$PHASE/csvs/crossdataset/affectnet_4c.csv
  OUT="$PHASE/results/xeval_affectnet_phase6_${STAGE}"
  [ -f "$OUT/crossdataset_summary.json" ] && continue
  mkdir -p "$OUT"
  python $PHASE/scripts/cross_dataset_zeroshot.py \
      --config $PHASE/configs/${STAGE}.yaml \
      --ckpt "$CKPT" --val_csv "$CSV" --out_dir "$OUT" \
      > $PHASE/logs/xeval_affectnet_phase6_${STAGE}.log 2>&1 || true
done

python $PHASE/scripts/aggregate_results.py >> $PHASE/logs/aggregate.log 2>&1
echo "[$(date)] === FINAL QUEUE DONE ==="
