#!/usr/bin/env bash
# Retry cross-dataset finetune with --init_from (fixed optimizer/head mismatch bugs)
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_full/best.pth
[ -f "$INIT_CKPT" ] || { echo "FAIL: no Stage 6 ckpt"; exit 1; }

for DS in affectnet sfew_train afew_train ckplus; do
  CFG=$PHASE/configs/finetune/finetune_${DS}.yaml
  [ -f "$CFG" ] || { echo "SKIP $DS: no config"; continue; }
  OUT_BEST=/mnt/hdd/ajy_25/results/phase6_finetune_${DS}/best.pth
  [ -f "$OUT_BEST" ] && { echo "SKIP $DS: already done"; continue; }
  echo "[$(date)] === RETRY FINETUNE: $DS ==="
  python scripts/train.py --config "$CFG" --init_from "$INIT_CKPT" \
      > $PHASE/logs/finetune_${DS}.log 2>&1 || true
  echo "[$(date)] $DS exit=$?"
done

echo "[$(date)] === RETRY FINETUNE DONE ==="

# Cross-dataset eval each finetuned ckpt on its OWN target val (in-domain finetune metric)
for DS in affectnet sfew_train afew_train ckplus; do
  CKPT=/mnt/hdd/ajy_25/results/phase6_finetune_${DS}/best.pth
  CSV=$PHASE/csvs/crossdataset/${DS}_4c.csv
  [ -f "$CKPT" ] && [ -f "$CSV" ] || continue
  OUT=$PHASE/results/finetune_eval_${DS}
  mkdir -p "$OUT"
  echo "[$(date)] eval finetune_${DS} on ${DS}"
  python $PHASE/scripts/cross_dataset_zeroshot.py \
      --config $PHASE/configs/finetune/finetune_${DS}.yaml \
      --ckpt "$CKPT" --val_csv "$CSV" --out_dir "$OUT" \
      > $PHASE/logs/finetune_eval_${DS}.log 2>&1 || true
done

python $PHASE/scripts/aggregate_results.py >> $PHASE/logs/aggregate.log 2>&1
echo "[$(date)] === FINETUNE EVAL DONE ==="
