#!/usr/bin/env bash
# Patch-scale ablation: 3 variants × (30 epoch train + cross-cultural xeval)
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

run_variant() {
  local NAME=$1
  local CFG=$PHASE/configs/${NAME}.yaml
  local CKPT=/mnt/hdd/ajy_25/results/phase6_${NAME}/best.pth
  local LOG=$PHASE/logs/${NAME}.log

  if [ -f "$CKPT" ]; then
    echo "[$(date)] SKIP train $NAME (best.pth exists)"
  else
    echo "[$(date)] === TRAIN: $NAME ==="
    python scripts/train.py --config "$CFG" > "$LOG" 2>&1 || true
    echo "[$(date)] $NAME train done"
  fi

  # Cross-cultural xeval × 6
  for ds in affectnet ckplus sfew_train sfew_val afew_train afew_val; do
    local CSV=$PHASE/csvs_v3/${ds}_4c.csv
    [ -f "$CSV" ] || continue
    local OUT=$PHASE/results/xeval_v3_${ds}_phase6_${NAME}
    [ -f "$OUT/crossdataset_summary.json" ] && continue
    [ -f "$CKPT" ] || continue
    mkdir -p "$OUT"
    python $PHASE/scripts/cross_dataset_zeroshot.py \
      --config "$CFG" --ckpt "$CKPT" --val_csv "$CSV" --out_dir "$OUT" \
      > $PHASE/logs/xeval_v3_${ds}_${NAME}.log 2>&1 || true
  done
  echo "[$(date)] $NAME xeval done"
}

run_variant stage6_v3_patch125
run_variant stage6_v3_patch150
run_variant stage6_v3_forehead_big

echo "[$(date)] === PATCH ABLATION CHAIN DONE ==="
