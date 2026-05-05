#!/usr/bin/env bash
# Cross-dataset eval ONLY on already-completed ckpts. Runs parallel with training.
set -uo pipefail
cd /home/ajy/AU-RegionFormer
PHASE=experiments/phase6_yonsei_paired

CSVS=(
  "affectnet:$PHASE/csvs/crossdataset/affectnet_4c.csv"
  "sfew_train:$PHASE/csvs/crossdataset/sfew_train_4c.csv"
  "sfew_val:$PHASE/csvs/crossdataset/sfew_val_4c.csv"
  "afew_train:$PHASE/csvs/crossdataset/afew_train_4c.csv"
  "afew_val:$PHASE/csvs/crossdataset/afew_val_4c.csv"
  "ckplus:$PHASE/csvs/crossdataset/ckplus_4c.csv"
)

# already-done ckpts (8개) - lam07/10/stage13/seed999/2024 추가됨
DONE_CKPTS=(
  "phase6_stage6_full:stage6_full"
  "phase6_stage6_seed123:stage6_seed123"
  "phase6_stage6_seed777:stage6_seed777"
  "phase6_stage6_seed999:stage6_seed999"
  "phase6_stage6_seed2024:stage6_seed2024"
  "phase6_stage7c_patch_dropout:stage7c_patch_dropout"
  "phase6_stage7_jepa:stage7_jepa"
  "phase6_stage8_dinov2:stage8_dinov2"
  "phase6_stage11_humankl_lam01:stage11_humankl_lam01"
  "phase6_stage11_humankl_lam03:stage11_humankl_lam03"
  "phase6_stage11_humankl_lam05:stage11_humankl_lam05"
)

for entry in "${DONE_CKPTS[@]}"; do
  NAME="${entry%%:*}"
  CFG="${entry#*:}"
  CKPT="/mnt/hdd/ajy_25/results/${NAME}/best.pth"
  [ -f "$CKPT" ] || { echo "SKIP $NAME"; continue; }
  for csv_entry in "${CSVS[@]}"; do
    TAG="${csv_entry%%:*}"
    CSV="${csv_entry#*:}"
    [ -f "$CSV" ] || continue
    OUT="$PHASE/results/xeval_${TAG}_${NAME}"
    [ -f "$OUT/crossdataset_summary.json" ] && continue   # skip if done
    mkdir -p "$OUT"
    echo "[$(date)] xeval $NAME on $TAG"
    python $PHASE/scripts/cross_dataset_zeroshot.py \
        --config $PHASE/configs/${CFG}.yaml \
        --ckpt "$CKPT" \
        --val_csv "$CSV" \
        --out_dir "$OUT" \
        > $PHASE/logs/xeval_${TAG}_${NAME}.log 2>&1 || true
  done
done

echo "[$(date)] xeval-now batch DONE"
python $PHASE/scripts/aggregate_results.py >> $PHASE/logs/aggregate.log 2>&1
