#!/usr/bin/env bash
# Parallel final queue: 2 stages at a time (each ~17GB on bf16, fits 48GB)
set -uo pipefail
PHASE=/home/ajy/AU-RegionFormer/experiments/phase6_yonsei_paired
cd /home/ajy/AU-RegionFormer

INIT_CKPT=/mnt/hdd/ajy_25/results/phase6_stage6_full/best.pth

run_stage() {
  local STAGE=$1
  local CFG="$PHASE/configs/${STAGE}.yaml"
  local OUT_BEST="/mnt/hdd/ajy_25/results/phase6_${STAGE}/best.pth"
  if [ -f "$OUT_BEST" ]; then
    echo "[$(date)] SKIP $STAGE (already done)"
    return
  fi
  echo "[$(date)] === RUN: $STAGE ==="
  python scripts/train.py --config "$CFG" \
      > $PHASE/logs/${STAGE}.log 2>&1 || true
  echo "[$(date)] $STAGE finished"
}

# ─── Round 1: 16a + 16b in parallel ───
echo "[$(date)] === ROUND 1: 16a + 16b parallel ==="
run_stage stage16a_iso_only &
PID_A=$!
run_stage stage16b_rel_only &
PID_B=$!
wait $PID_A $PID_B
echo "[$(date)] === ROUND 1 DONE ==="

# ─── Round 2: 16c + 16d in parallel ───
echo "[$(date)] === ROUND 2: 16c + 16d parallel ==="
run_stage stage16c_a_only &
PID_C=$!
run_stage stage16d_product &
PID_D=$!
wait $PID_C $PID_D
echo "[$(date)] === ROUND 2 DONE ==="

# ─── Round 3: AffectNet finetune (alone, init_from Stage 6) ───
echo "[$(date)] === ROUND 3: AffectNet finetune ==="
CFG=$PHASE/configs/finetune/finetune_affectnet.yaml
if [ -f "$CFG" ] && [ -f "$INIT_CKPT" ]; then
  OUT="/mnt/hdd/ajy_25/results/phase6_finetune_affectnet/best.pth"
  if [ ! -f "$OUT" ]; then
    python scripts/train.py --config "$CFG" --init_from "$INIT_CKPT" \
        > $PHASE/logs/finetune_affectnet.log 2>&1 || true
  fi
fi
echo "[$(date)] === ROUND 3 DONE ==="

# ─── Round 4: xeval Stage 16 ckpts on AffectNet (Q1 cross-cultural ablation) ───
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
