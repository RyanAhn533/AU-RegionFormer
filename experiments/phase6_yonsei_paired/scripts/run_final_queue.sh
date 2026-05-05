#!/usr/bin/env bash
# FINAL QUEUE — only what matters
# 1. Stage 16 a/b/c/d ablation matrix (4 variants × 1 seed × 30ep)
# 2. AffectNet finetune (init_from Stage 6) — Q1 decision point
# 3. Final aggregate
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_full/best.pth

# ─── 1. Stage 16 ablation matrix (sequential, full GPU each) ───
for STAGE in stage16a_iso_only stage16b_rel_only stage16c_a_only stage16d_product; do
  CFG="$PHASE/configs/${STAGE}.yaml"
  OUT="/mnt/hdd/ajy_25/results/phase6_${STAGE}/best.pth"
  if [ -f "$OUT" ]; then
    echo "[$(date)] SKIP $STAGE (already done)"
    continue
  fi
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config "$CFG" \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
  echo "[$(date)] $STAGE exit=$?"
done

# ─── 2. AffectNet finetune (Q1 decision point) ───
CFG=$PHASE/configs/finetune/finetune_affectnet.yaml
if [ -f "$CFG" ] && [ -f "$INIT_CKPT" ]; then
  OUT="/mnt/hdd/ajy_25/results/phase6_finetune_affectnet/best.pth"
  if [ ! -f "$OUT" ]; then
    echo "[$(date)] === FINETUNE: affectnet (init from Stage 6) ==="
    python scripts/train.py --config "$CFG" --init_from "$INIT_CKPT" \
        > $PHASE/logs/finetune_affectnet.log 2>&1 || true
    echo "[$(date)] affectnet exit=$?"
  else
    echo "[$(date)] SKIP affectnet finetune (done)"
  fi
fi

# ─── 3. xeval Stage 16 ckpts on Korean master_val (in-domain Beta gate ablation) + AffectNet ───
for STAGE in stage16a_iso_only stage16b_rel_only stage16c_a_only stage16d_product; do
  CKPT="/mnt/hdd/ajy_25/results/phase6_${STAGE}/best.pth"
  [ -f "$CKPT" ] || continue
  for csv_entry in "affectnet:$PHASE/csvs/crossdataset/affectnet_4c.csv"; do
    TAG="${csv_entry%%:*}"; CSV="${csv_entry#*:}"
    OUT="$PHASE/results/xeval_${TAG}_phase6_${STAGE}"
    [ -f "$OUT/crossdataset_summary.json" ] && continue
    mkdir -p "$OUT"
    python $PHASE/scripts/cross_dataset_zeroshot.py \
        --config $PHASE/configs/${STAGE}.yaml \
        --ckpt "$CKPT" --val_csv "$CSV" --out_dir "$OUT" \
        > $PHASE/logs/xeval_${TAG}_phase6_${STAGE}.log 2>&1 || true
  done
done

# ─── 4. Final aggregate ───
python $PHASE/scripts/aggregate_results.py >> $PHASE/logs/aggregate.log 2>&1
echo "[$(date)] === FINAL QUEUE DONE ==="
